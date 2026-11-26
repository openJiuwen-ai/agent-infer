# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

from __future__ import annotations

from collections.abc import Iterable

from vllm.sampling_params import SamplingParams
from vllm.v1.core.kv_cache_utils import BlockHash, make_block_hash_with_group_id
from vllm.v1.request import Request

from agentinfer.agentcache.mori.hiradix import HiRadix
from agentinfer.agentcache.mori.offload import MoriHiCacheController
from agentinfer.agentcache.mori.scheduler import MoriScheduler


def key(index: int) -> bytes:
    return index.to_bytes(32, "big")


def insert(tree: HiRadix, keys: Iterable[int], parent: int | None = None, program: str = "p"):
    pool = tree.device.pool
    ids = list(keys)
    blocks = pool.get_new_blocks(len(ids))
    tree.placement.register(program)
    depth = 0 if parent is None else next(iter(tree.pages[key(parent)].device.values())).block_hash_num_tokens
    for index, block in zip(ids, blocks, strict=True):
        depth += tree.page_size
        pool._insert_block_hash(make_block_hash_with_group_id(BlockHash(key(index)), 0), block, num_tokens=depth)
        tree.publish(block, key(index), key(parent) if parent is not None else None, program)
        parent = index
    tree.pages[key(ids[-1])].boundary = True
    tree.stamp(blocks[-1], program)
    pool.free_blocks(blocks)
    return blocks


def backup(tree: HiRadix, blocks):
    pool = tree.host.pool
    copies = pool.get_new_blocks(len(blocks))
    for source, target in zip(blocks, copies, strict=True):
        pool._insert_block_hash(source.block_hash, target, num_tokens=source.block_hash_num_tokens)
        tree.backup(source.block_id, target)
    pool.free_blocks(copies)
    return copies


def _request(name, hashes, length, max_tokens=1):
    request = Request(name, [0] * length, SamplingParams(max_tokens=max_tokens), None)
    request.block_hashes.extend(BlockHash(key(index)) for index in hashes)
    return request


def _test_controller(cache):
    offload = MoriHiCacheController.__new__(MoriHiCacheController)
    offload.bind_tree(cache)
    cache.cache_controller = offload
    offload.cpu_block_pool = cache.host.pool
    offload._gpu_block_pool = cache.device.pool
    offload._lazy_mode = False
    offload.hash_block_size = offload.block_size = 16
    offload.group_block_sizes = [16]
    offload.prefix_cacheable_group_ids = {0}
    offload._pending_cpu_hits = {}
    offload._reqs_to_load = {}
    offload._reqs_to_store = {}
    offload._load_event_to_reqs = {}
    offload._interval_load_blocks_completed = 0
    offload._eviction_store_gpu_ids = set()
    offload._prefetch_pins = {}
    offload._pending_finished_stores = []
    return offload


def make_scheduler():
    scheduler = MoriScheduler.__new__(MoriScheduler)
    scheduler.lora_config = None
    return scheduler
