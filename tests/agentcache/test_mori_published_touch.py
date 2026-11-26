# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Compare publication recency with full HiRadix matching on identical cache states."""

from itertools import product
from types import SimpleNamespace

import pytest

from agentinfer.agentcache.mori.hiradix import HiRadix
from agentinfer.scheduling.domain import ProgramRef
from tests.helpers.marks import test_marks

pytestmark = test_marks(hardware="cpu")


class Clock:
    def __init__(self):
        self.value = 0

    def __next__(self):
        tick = self.value
        self.value += 1
        return tick


class LookupLog(dict):
    def __init__(self, page, record):
        super().__init__(page.children)
        self.key = page.key
        self.record = record

    def get(self, key):
        self.record.append(("lookup", self.key, key))
        return super().get(key)

    def values(self):
        self.record.append(("suffix", self.key))
        return super().values()


def build_tree(tiers):
    tree = HiRadix(None, None, page_size=16)
    tree.clock = Clock()
    program = tree.placement.register("p")
    parents = (None, 1, 2, 3, 4, 5, 2, 7, None)
    for key, parent in enumerate(parents, 1):
        tree.publish(SimpleNamespace(block_id=key), key, parent, "p")
        tree.device.on_block_ready(key)
    # Vary the internal nodes only. The device leaf at 6 keeps all routes
    # present even if invalidation removes the final replica of an ancestor.
    for key, tier in zip((3, 4, 5), tiers, strict=True):
        if tier in ("host", "both"):
            tree.backup(key, SimpleNamespace(block_id=key))
            tree.host.on_block_ready(key)
        if tier in ("host", "hole", "loading"):
            tree.device.on_block_removed(key)
        if tier == "loading":
            tree._loading[tree.pages[key]] = SimpleNamespace(block_id=100 + key)
        tree.pages[key].load_epoch = 1
        tree.pages[key].write_epoch = 3
    # A real second GPU allocation of the same hash can be independently ready.
    tree.publish(SimpleNamespace(block_id=60), 6, 5, "p")
    tree.device.on_block_ready(60)
    tree.pages[2].stamped_for = program
    tree.pages[2].shared = True
    record = []
    for page in (tree.root, *tree.pages.values()):
        page.children = LookupLog(page, record)
    return tree, record


def cache_state(tree):
    nodes = {
        key: (
            page.accessed,
            page.boundary,
            page.shared,
            page.stamped_for,
            page.program,
            page.load_epoch,
            page.write_epoch,
            tree.device._priority(page),
            tree.host._priority(page),
        )
        for key, page in tree.pages.items()
    }
    plans = [
        tier.plan(demand, protected, include_uncached=False)
        for tier in (tree.device, tree.host)
        for demand in (1, 3, 10)
        for protected in ((), (6, 8))
    ]
    return nodes, plans, tree.clock.value


@pytest.mark.parametrize("for_lpm", (False, True))
def test_touch_matches_full_lookup_and_victim_order(for_lpm):
    for tiers in product(("device", "host", "both", "hole", "loading"), repeat=3):
        oracle, full_visits = build_tree(tiers)
        candidate, touch_visits = build_tree(tiers)
        for keys, depth in (
            ((1, 2, 3, 4, 5, 6), 0),  # empty
            ((9, 100), 2),  # miss after a hit
            ((1, 2, 3, 4, 5, 6), 3),  # split in a compressed edge
            ((1, 2, 7, 8), 4),  # sibling
            ((1, 2, 3, 4, 5, 6), 5),
            ((1, 2, 3, 4, 5, 6), 6),
            ((1, 2, 7, 100), 4),  # miss inside an edge
            ((90,), 1),  # immediate miss
            ((1, 2, 3, 4, 5, 6), 6),  # repeat after the previous split
        ):
            full_visits.clear()
            touch_visits.clear()
            device, _ = oracle.match(keys[:depth], include_loading=for_lpm, mark_load=for_lpm)
            assert candidate.touch_prefix(keys, depth, for_lpm=for_lpm) == len(device)
            assert touch_visits == full_visits, (tiers, keys, depth)
            assert cache_state(candidate) == cache_state(oracle), (tiers, keys, depth)


def test_lpm_skips_cache_disabled_waiter_and_counts_last_prompt_token():
    from agentinfer.agentcache.mori.scheduler import MoriScheduler

    tree, visits = build_tree(("device", "device", "device"))
    owner = MoriScheduler.__new__(MoriScheduler)
    owner.kv_cache_manager = SimpleNamespace(hiradix=tree)
    keys = (1, 2, 3, 4, 5, 6)
    full, short, skipped = (
        SimpleNamespace(block_hashes=keys, num_tokens=tokens, skip_reading_prefix_cache=skip)
        for tokens, skip in ((96, False), (95, False), (96, True))
    )
    visits.clear()
    assert owner._order_waiting([short, skipped, full]) == [full, short, skipped]
    assert [entry[2] for entry in visits if entry[0] == "lookup"] == list(keys[:5]) + list(keys)


def test_stamp_owner_identity_preserves_equal_distinct_and_retired_generations():
    tree = HiRadix(None, None, page_size=16)
    old = ProgramRef("p", 0)
    refs = {"p": old}
    tree.placement.current_ref = refs.get
    tree.placement.register(old)
    equal = ProgramRef("p", 0)
    assert equal == old and equal is not old
    tree.publish(SimpleNamespace(block_id=1), b"root", None, equal)
    tree.stamp_prefill(b"root", old)
    page = tree.pages[b"root"]
    assert page.program is equal and not page.shared

    tree.placement.release(old)
    refs["p"] = new = ProgramRef("p", 1)
    tree.placement.register(new)
    tree.stamp_prefill(b"root", old)
    assert page.program is equal and not page.shared
    tree.stamp_prefill(b"root", new)
    assert page.program is new and not page.shared
    refs["other"] = other = ProgramRef("other", 0)
    tree.stamp_prefill(b"root", other)
    assert page.shared
