"""Tests for the vLLM V1 scheduler plugin template.

The real plugin loads inside the pinned vLLM container; here we exercise the
bridge logic against a fake ``Scheduler`` base injected into sys.modules — no
GPU needed. Per the adversarial review, the FCFS waiting queue is a ``deque``
subclass (reorderable) while the priority queue is a heap that must be skipped.
"""
from __future__ import annotations

import ast
import sys
import time
import types
from collections import deque
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from targets.scheduling.plugin_template import (  # noqa: E402
    PLUGIN_INVOKED_MARKER,
    SCHEDULER_QUALNAME,
    render_and_write,
    render_plugin,
    reorder_waiting,
)


def test_reorder_waiting_basic():
    assert reorder_waiting(["a", "b", "c"], ["c", "a"]) == ["c", "a", "b"]


def test_reorder_waiting_ignores_unknown_priority_ids():
    assert reorder_waiting(["a", "b"], ["x", "b"]) == ["b", "a"]


def test_reorder_waiting_empty_priority_keeps_order():
    assert reorder_waiting(["a", "b", "c"], []) == ["a", "b", "c"]


def test_scheduler_qualname_is_dot_form():
    # vLLM resolves --scheduler-cls via rsplit('.', 1); colon form crashes.
    assert SCHEDULER_QUALNAME == "generated_scheduler.EvolvedScheduler"
    assert ":" not in SCHEDULER_QUALNAME


def test_render_produces_valid_v1_plugin():
    seed_src = (ROOT / "targets" / "scheduling" / "seed.py").read_text(encoding="utf-8")
    code = render_plugin(seed_src)
    ast.parse(code)  # must be syntactically valid Python
    assert "class EvolvedScheduler(AsyncScheduler):" in code
    assert "def schedule(self)" in code
    assert "generated_scheduler.EvolvedScheduler" in code   # dot form
    assert "generated_scheduler:EvolvedScheduler" not in code  # never the colon form
    assert PLUGIN_INVOKED_MARKER in code
    # the broken V0 surface must be gone
    assert "SequenceGroup" not in code
    assert "--scheduler-plugin" not in code


def test_render_strips_frontier_types_and_keeps_defer_decision_standalone():
    policy = """\
from integrations.frontier.ve_policy_api import ScheduleDecision
from targets.scheduling.skeleton import RequestInfo

def schedule_batch(*args, **kwargs) -> ScheduleDecision:
    return ScheduleDecision(prefill_batch=["a"], decode_batch=[], defer_ids=["b"])
"""
    code = render_plugin(policy)
    ast.parse(code)
    assert "from integrations.frontier.ve_policy_api import" not in code
    assert "from targets.scheduling.skeleton import" not in code
    assert "defer_ids: list = field(default_factory=list)" in code


def test_render_and_write_creates_generated_scheduler(tmp_path):
    seed_src = (ROOT / "targets" / "scheduling" / "seed.py").read_text(encoding="utf-8")
    path = render_and_write(tmp_path, seed_src)
    assert path.name == "generated_scheduler.py"   # exact name vLLM imports
    assert path.exists()
    assert "class EvolvedScheduler(AsyncScheduler):" in path.read_text(
        encoding="utf-8"
    )


def _install_fake_vllm(monkeypatch):
    """Inject minimal fake vLLM synchronous and asynchronous bases."""
    class _FakeScheduler:
        def schedule(self):
            return "SUPER_OUTPUT"

        def add_request(self, request):
            self.waiting.append(request)

        def _preempt_request(self, request, timestamp):
            request.num_preemptions += 1
            self.waiting.appendleft(request)

    class _FakeAsyncScheduler(_FakeScheduler):
        pass

    sched_mod = types.ModuleType("vllm.v1.core.sched.scheduler")
    sched_mod.Scheduler = _FakeScheduler
    async_sched_mod = types.ModuleType("vllm.v1.core.sched.async_scheduler")
    async_sched_mod.AsyncScheduler = _FakeAsyncScheduler
    for name in ["vllm", "vllm.v1", "vllm.v1.core", "vllm.v1.core.sched"]:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "vllm.v1.core.sched.scheduler", sched_mod)
    monkeypatch.setitem(
        sys.modules,
        "vllm.v1.core.sched.async_scheduler",
        async_sched_mod,
    )
    return _FakeAsyncScheduler


def _exec_plugin(code):
    ns: dict = {}
    exec(compile(code, "generated_scheduler.py", "exec"), ns)
    return ns["EvolvedScheduler"]


def _req(rid, arrival, prompt=100):
    return SimpleNamespace(
        request_id=rid, num_prompt_tokens=prompt, num_computed_tokens=0,
        num_tokens=prompt, num_output_tokens=0, max_tokens=64,
        num_preemptions=0, arrival_time=arrival,
    )


class _FakePriorityQueue:
    """Stand-in for PriorityRequestQueue: iterable, but not a deque and with no
    clear()/append() — reordering must be skipped, not crash."""

    def __init__(self, items):
        self._items = list(items)

    def __iter__(self):
        return iter(self._items)


class _CountingDeque(deque):
    def __init__(self, items):
        super().__init__(items)
        self.iterations = 0

    def __iter__(self):
        self.iterations += 1
        return super().__iter__()


def test_rendered_scheduler_reorders_fcfs_deque_then_delegates(monkeypatch):
    _install_fake_vllm(monkeypatch)
    seed_src = (ROOT / "targets" / "scheduling" / "seed.py").read_text(encoding="utf-8")
    EvolvedScheduler = _exec_plugin(render_plugin(seed_src))

    inst = object.__new__(EvolvedScheduler)
    inst.scheduler_config = SimpleNamespace(max_num_batched_tokens=8192, max_num_seqs=256)
    inst.waiting = deque([_req("r2", 5.0), _req("r1", 1.0), _req("r3", 3.0)])  # FCFS = deque
    inst.running = []

    out = EvolvedScheduler.schedule(inst)
    assert out == "SUPER_OUTPUT"  # delegated to super().schedule()
    # FCFS seed orders waiting by arrival time -> r1(1), r3(3), r2(5)
    assert [r.request_id for r in inst.waiting] == ["r1", "r3", "r2"]


def test_rendered_scheduler_skips_priority_queue_without_crashing(monkeypatch):
    _install_fake_vllm(monkeypatch)
    seed_src = (ROOT / "targets" / "scheduling" / "seed.py").read_text(encoding="utf-8")
    EvolvedScheduler = _exec_plugin(render_plugin(seed_src))

    inst = object.__new__(EvolvedScheduler)
    inst.scheduler_config = SimpleNamespace(max_num_batched_tokens=8192, max_num_seqs=256)
    pq = _FakePriorityQueue([_req("r2", 5.0), _req("r1", 1.0), _req("r3", 3.0)])
    inst.waiting = pq
    inst.running = []

    out = EvolvedScheduler.schedule(inst)
    assert out == "SUPER_OUTPUT"                       # still delegates, no crash
    assert inst.waiting is pq                           # untouched (not reconstructed)
    assert [r.request_id for r in inst.waiting] == ["r2", "r1", "r3"]  # order unchanged


def test_rendered_scheduler_falls_back_on_policy_crash(monkeypatch):
    _install_fake_vllm(monkeypatch)
    bad_policy = "def schedule_batch(*a, **k):\n    raise RuntimeError('boom')\n"
    EvolvedScheduler = _exec_plugin(render_plugin(bad_policy))

    inst = object.__new__(EvolvedScheduler)
    inst.scheduler_config = SimpleNamespace(max_num_batched_tokens=8192, max_num_seqs=256)
    inst.waiting = deque([_req("r2", 5.0), _req("r1", 1.0), _req("r3", 3.0)])
    inst.running = []

    out = EvolvedScheduler.schedule(inst)
    assert out == "SUPER_OUTPUT"                       # still delegates
    assert [r.request_id for r in inst.waiting] == ["r2", "r1", "r3"]  # unchanged


def test_explicit_defer_hides_one_event_then_starvation_bound_forces_admission(
    monkeypatch, capsys,
):
    _install_fake_vllm(monkeypatch)
    policy = """\
MAX_DEFER_EVENTS = 2
def schedule_batch(waiting, running, *args):
    return ScheduleDecision(
        prefill_batch=[r.request_id for r in waiting if r.request_id != "a"],
        decode_batch=[],
        defer_ids=["a"],
        mechanism_applicable=True,
    )
"""
    namespace = {}
    exec(compile(render_plugin(policy), "generated_scheduler.py", "exec"), namespace)
    scheduler = namespace["EvolvedScheduler"]
    scheduler._ve_last_policy_fingerprint = None
    scheduler._ve_defer_counts = {}
    inst = object.__new__(scheduler)
    inst.scheduler_config = SimpleNamespace(max_num_batched_tokens=8192, max_num_seqs=4)
    inst.waiting = deque([_req("a", 1.0), _req("b", 2.0)])
    inst.running = []

    assert scheduler.schedule(inst) == "SUPER_OUTPUT"
    assert [request.request_id for request in inst.waiting] == ["b", "a"]
    first_log = capsys.readouterr().out
    assert "waiting request deferred" in first_log

    assert scheduler.schedule(inst) == "SUPER_OUTPUT"
    assert [request.request_id for request in inst.waiting] == ["a", "b"]
    second_log = capsys.readouterr().out
    assert "starvation bound forced admission" in second_log


def test_bridge_passes_live_free_kv_blocks_to_policy(monkeypatch):
    _install_fake_vllm(monkeypatch)
    policy = """\
seen_free_blocks = None
def schedule_batch(waiting, running, token_budget, seq_budget, free_blocks, hit_rate):
    global seen_free_blocks
    seen_free_blocks = free_blocks
    return ScheduleDecision(prefill_batch=[], decode_batch=[])
"""
    namespace = {}
    exec(compile(render_plugin(policy), "generated_scheduler.py", "exec"), namespace)
    scheduler = namespace["EvolvedScheduler"]
    scheduler._ve_last_policy_fingerprint = None
    inst = object.__new__(scheduler)
    inst.scheduler_config = SimpleNamespace(max_num_batched_tokens=8192, max_num_seqs=4)
    inst.waiting = deque([_req("a", 1.0)])
    inst.running = []
    inst.kv_cache_manager = SimpleNamespace(
        block_pool=SimpleNamespace(get_num_free_blocks=lambda: 37)
    )

    scheduler.schedule(inst)
    assert namespace["seen_free_blocks"] == 37


def test_policy_runs_only_when_admission_can_change(monkeypatch):
    _install_fake_vllm(monkeypatch)
    policy = """\
calls = 0
def schedule_batch(waiting, running, *args):
    global calls
    calls += 1
    return ScheduleDecision(
        prefill_batch=[r.request_id for r in reversed(waiting)],
        decode_batch=[],
    )
"""
    namespace = {}
    exec(compile(render_plugin(policy), "generated_scheduler.py", "exec"), namespace)
    scheduler = namespace["EvolvedScheduler"]
    scheduler._ve_last_policy_fingerprint = None
    inst = object.__new__(scheduler)
    inst.scheduler_config = SimpleNamespace(max_num_batched_tokens=8192, max_num_seqs=1)
    inst.waiting = _CountingDeque([_req("a", 1.0), _req("b", 2.0)])
    inst.running = [_req("running", 0.0)]

    scheduler.schedule(inst)
    assert namespace["calls"] == 0
    assert inst.waiting.iterations == 0

    inst.running = []
    scheduler.schedule(inst)
    assert namespace["calls"] == 1
    assert [r.request_id for r in inst.waiting] == ["b", "a"]

    scheduler.schedule(inst)
    assert namespace["calls"] == 1
    inst.waiting.append(_req("c", 3.0))
    scheduler.schedule(inst)
    assert namespace["calls"] == 2


def test_bounded_waiting_window_observes_and_reorders_only_prefix(monkeypatch):
    _install_fake_vllm(monkeypatch)
    policy = """\
POLICY_WAITING_WINDOW = 2
seen_ids = []
def schedule_batch(waiting, running, *args):
    global seen_ids
    seen_ids = [request.request_id for request in waiting]
    return ScheduleDecision(
        prefill_batch=[request.request_id for request in reversed(waiting)],
        decode_batch=[],
        mechanism_applicable=True,
    )
"""
    namespace = {}
    exec(compile(render_plugin(policy), "generated_scheduler.py", "exec"), namespace)
    scheduler = namespace["EvolvedScheduler"]
    scheduler._ve_last_policy_fingerprint = None
    inst = object.__new__(scheduler)
    inst.scheduler_config = SimpleNamespace(
        max_num_batched_tokens=8192,
        max_num_seqs=4,
    )
    tail = [_req("c", 3.0), _req("d", 4.0)]
    inst.waiting = deque([_req("a", 1.0), _req("b", 2.0), *tail])
    inst.running = []

    assert scheduler.schedule(inst) == "SUPER_OUTPUT"
    assert namespace["seen_ids"] == ["a", "b"]
    assert [request.request_id for request in inst.waiting] == [
        "b",
        "a",
        "c",
        "d",
    ]
    assert list(inst.waiting)[2:] == tail


def test_zero_waiting_window_preserves_legacy_whole_queue_surface(monkeypatch):
    _install_fake_vllm(monkeypatch)
    policy = """\
seen_ids = []
def schedule_batch(waiting, running, *args):
    global seen_ids
    seen_ids = [request.request_id for request in waiting]
    return ScheduleDecision(prefill_batch=[], decode_batch=[])
"""
    namespace = {}
    exec(compile(render_plugin(policy), "generated_scheduler.py", "exec"), namespace)
    scheduler = namespace["EvolvedScheduler"]
    scheduler._ve_last_policy_fingerprint = None
    inst = object.__new__(scheduler)
    inst.scheduler_config = SimpleNamespace(
        max_num_batched_tokens=8192,
        max_num_seqs=4,
    )
    inst.waiting = deque(
        [_req("a", 1.0), _req("b", 2.0), _req("c", 3.0)]
    )
    inst.running = []

    assert scheduler.schedule(inst) == "SUPER_OUTPUT"
    assert namespace["seen_ids"] == ["a", "b", "c"]


def test_nonpreemptive_policy_can_opt_out_of_running_request_view(monkeypatch):
    _install_fake_vllm(monkeypatch)
    policy = """\
POLICY_WAITING_WINDOW = 2
POLICY_NEEDS_RUNNING_REQUESTS = False
seen_running_ids = None
def schedule_batch(waiting, running, *args):
    global seen_running_ids
    seen_running_ids = [request.request_id for request in running]
    return ScheduleDecision(
        prefill_batch=[waiting[0].request_id],
        decode_batch=[],
        preempt_ids=[],
        defer_ids=[],
    )
"""
    namespace = {}
    exec(compile(render_plugin(policy), "generated_scheduler.py", "exec"), namespace)
    scheduler = namespace["EvolvedScheduler"]
    scheduler._ve_last_policy_fingerprint = None
    inst = object.__new__(scheduler)
    inst.scheduler_config = SimpleNamespace(
        max_num_batched_tokens=8192,
        max_num_seqs=4,
    )
    inst.waiting = deque([_req("a", 1.0), _req("b", 2.0)])
    inst.running = [_req("running", 0.0)]

    assert scheduler.schedule(inst) == "SUPER_OUTPUT"
    assert namespace["seen_running_ids"] == []
    assert len(inst.running) == 1


def test_running_request_view_is_enabled_by_default(monkeypatch):
    _install_fake_vllm(monkeypatch)
    policy = """\
seen_running_ids = None
def schedule_batch(waiting, running, *args):
    global seen_running_ids
    seen_running_ids = [request.request_id for request in running]
    return ScheduleDecision(prefill_batch=[], decode_batch=[])
"""
    namespace = {}
    exec(compile(render_plugin(policy), "generated_scheduler.py", "exec"), namespace)
    scheduler = namespace["EvolvedScheduler"]
    scheduler._ve_last_policy_fingerprint = None
    inst = object.__new__(scheduler)
    inst.scheduler_config = SimpleNamespace(
        max_num_batched_tokens=8192,
        max_num_seqs=4,
    )
    inst.waiting = deque([_req("a", 1.0)])
    inst.running = [_req("running", 0.0)]

    assert scheduler.schedule(inst) == "SUPER_OUTPUT"
    assert namespace["seen_running_ids"] == ["running"]


def test_global_recoverable_index_exposes_only_head_and_bounded_heap_frontiers(
    monkeypatch,
):
    _install_fake_vllm(monkeypatch)
    policy = """\
POLICY_GLOBAL_RECOVERABLE_INDEX = True
POLICY_INDEX_WIDTH = 2
POLICY_TTFT_SLO_S = 120.0
POLICY_NEEDS_RUNNING_REQUESTS = False
seen_ids = []
def schedule_batch(waiting, running, *args):
    global seen_ids
    seen_ids = [request.request_id for request in waiting]
    return ScheduleDecision(
        prefill_batch=["short-residency"],
        decode_batch=[],
        preempt_ids=[],
        defer_ids=[],
        mechanism_applicable=True,
    )
"""
    namespace = {}
    exec(compile(render_plugin(policy), "generated_scheduler.py", "exec"), namespace)
    scheduler = namespace["EvolvedScheduler"]
    scheduler._ve_last_policy_fingerprint = None
    inst = object.__new__(scheduler)
    inst.scheduler_config = SimpleNamespace(
        max_num_batched_tokens=8192,
        max_num_seqs=4,
    )
    inst.waiting = _CountingDeque([])
    inst.running = []
    now = time.time()
    requests = [
        _req("head", now, prompt=100),
        _req("second-residency", now + 0.001, prompt=90),
        _req("short-prompt", now + 0.002, prompt=10),
        _req("short-residency", now + 0.003, prompt=100),
    ]
    requests[0].max_tokens = 1000
    requests[1].max_tokens = 900
    requests[2].max_tokens = 1000
    requests[3].max_tokens = 1
    for request in requests:
        scheduler.add_request(inst, request)

    assert scheduler.schedule(inst) == "SUPER_OUTPUT"
    assert namespace["seen_ids"] == [
        "head",
        "short-residency",
        "second-residency",
        "short-prompt",
    ]
    # The Python bridge neither copies nor iterates the complete deque.
    assert inst.waiting.iterations == 0
    assert [request.request_id for request in inst.waiting] == [
        "short-residency",
        "head",
        "second-residency",
        "short-prompt",
    ]


def test_global_recoverable_index_leaves_expired_requests_in_residual_fcfs_lane(
    monkeypatch,
):
    _install_fake_vllm(monkeypatch)
    policy = """\
POLICY_GLOBAL_RECOVERABLE_INDEX = True
POLICY_INDEX_WIDTH = 4
POLICY_TTFT_SLO_S = 120.0
POLICY_NEEDS_RUNNING_REQUESTS = False
seen_ids = []
def schedule_batch(waiting, running, *args):
    global seen_ids
    seen_ids = [request.request_id for request in waiting]
    return ScheduleDecision(
        prefill_batch=["live"],
        decode_batch=[],
        preempt_ids=[],
        defer_ids=[],
        mechanism_applicable=True,
    )
"""
    namespace = {}
    exec(compile(render_plugin(policy), "generated_scheduler.py", "exec"), namespace)
    scheduler = namespace["EvolvedScheduler"]
    scheduler._ve_last_policy_fingerprint = None
    inst = object.__new__(scheduler)
    inst.scheduler_config = SimpleNamespace(
        max_num_batched_tokens=8192,
        max_num_seqs=4,
    )
    inst.waiting = deque()
    inst.running = []
    now = time.time()
    head = _req("head", now - 121.0, prompt=1000)
    expired = _req("expired", now - 121.0, prompt=1)
    live = _req("live", now, prompt=2)
    for request in (head, expired, live):
        scheduler.add_request(inst, request)

    assert scheduler.schedule(inst) == "SUPER_OUTPUT"
    assert namespace["seen_ids"] == ["head", "live"]
    assert [request.request_id for request in inst.waiting] == [
        "live",
        "head",
        "expired",
    ]
    assert "expired" not in inst._ve_index_active
    assert "head" not in inst._ve_index_active


def test_global_index_removes_newly_scheduled_ids_from_live_index(monkeypatch):
    _install_fake_vllm(monkeypatch)
    policy = """\
POLICY_GLOBAL_RECOVERABLE_INDEX = True
POLICY_INDEX_WIDTH = 1
POLICY_NEEDS_RUNNING_REQUESTS = False
def schedule_batch(waiting, running, *args):
    return ScheduleDecision(
        prefill_batch=[waiting[0].request_id],
        decode_batch=[],
        preempt_ids=[],
        defer_ids=[],
        mechanism_applicable=False,
    )
"""
    namespace = {}
    exec(compile(render_plugin(policy), "generated_scheduler.py", "exec"), namespace)
    scheduler = namespace["EvolvedScheduler"]
    scheduler._ve_last_policy_fingerprint = None
    scheduler_output = SimpleNamespace(
        scheduled_new_reqs=[SimpleNamespace(req_id="head")]
    )
    monkeypatch.setattr(
        scheduler.__mro__[1],
        "schedule",
        lambda self: scheduler_output,
    )
    inst = object.__new__(scheduler)
    inst.scheduler_config = SimpleNamespace(
        max_num_batched_tokens=8192,
        max_num_seqs=4,
    )
    inst.waiting = deque()
    inst.running = []
    scheduler.add_request(inst, _req("head", time.time(), prompt=64))

    assert scheduler.schedule(inst) is scheduler_output
    assert "head" not in inst._ve_index_active


def test_global_recoverable_index_is_default_off_on_add_request(monkeypatch):
    _install_fake_vllm(monkeypatch)
    seed_src = (
        ROOT / "targets" / "scheduling" / "seed.py"
    ).read_text(encoding="utf-8")
    scheduler = _exec_plugin(render_plugin(seed_src))
    inst = object.__new__(scheduler)
    inst.waiting = deque()
    scheduler.add_request(inst, _req("head", time.time(), prompt=64))

    assert not hasattr(inst, "_ve_index_active")
    assert [request.request_id for request in inst.waiting] == ["head"]


def test_request_mapping_uses_requested_decode_budget_not_generated_count(monkeypatch):
    _install_fake_vllm(monkeypatch)
    seed_src = (ROOT / "targets" / "scheduling" / "seed.py").read_text(encoding="utf-8")
    namespace = {}
    exec(compile(render_plugin(seed_src), "generated_scheduler.py", "exec"), namespace)
    request = SimpleNamespace(
        request_id="r", num_prompt_tokens=128, num_computed_tokens=0,
        num_output_tokens=0, max_tokens=777, num_preemptions=3, arrival_time=1.0,
    )
    info = namespace["_request_to_info"](request, 2.0, True)
    assert info.num_output_tokens == 777
    assert info.num_preemptions == 3


def test_active_preemption_rotates_one_elephant_and_preserves_async_state(
    monkeypatch, capsys,
):
    _install_fake_vllm(monkeypatch)
    policy = (
        ROOT / "targets" / "scheduling" / "seeds" / "ftc.py"
    ).read_text(encoding="utf-8")
    namespace = {}
    exec(compile(render_plugin(policy), "generated_scheduler.py", "exec"), namespace)
    scheduler = namespace["EvolvedScheduler"]
    scheduler._ve_last_policy_fingerprint = None
    scheduler_output = SimpleNamespace(preempted_req_ids=None)
    monkeypatch.setattr(
        scheduler.__mro__[1],
        "schedule",
        lambda self: scheduler_output,
    )
    inst = object.__new__(scheduler)
    inst.scheduler_config = SimpleNamespace(
        max_num_batched_tokens=8192,
        max_num_seqs=4,
        async_scheduling=True,
    )
    inst.prev_step_scheduled_req_ids = {"run-0"}
    inst._ve_has_seen_elephant = True
    urgent = _req("urgent", time.time(), prompt=64)
    inst.waiting = deque([urgent])
    inst.running = [
        _req(f"run-{index}", time.time() - 1.0, prompt=64)
        for index in range(4)
    ]
    for request in inst.running:
        request.max_tokens = 1792
        request.output_token_ids = []
        request.num_output_placeholders = 1
        request.discard_latest_async_tokens = False

    # The waiter arrives before any elephant emits its first token. The raw
    # active-policy guard must stay on the stock fast path, then re-evaluate
    # when that lifecycle edge changes even though the queue itself is unchanged.
    assert scheduler.schedule(inst) is scheduler_output
    assert len(inst.running) == 4
    for request in inst.running:
        request.output_token_ids = [1]

    assert scheduler.schedule(inst) is scheduler_output
    assert len(inst.running) == 3
    assert inst.running[0].request_id == "run-1"
    assert [request.request_id for request in inst.waiting] == [
        "urgent",
        "run-0",
    ]
    victim = inst.waiting[-1]
    assert victim.num_preemptions == 1
    assert victim.num_output_placeholders == 0
    assert victim.discard_latest_async_tokens is True
    assert inst.prev_step_scheduled_req_ids == set()
    assert scheduler_output.preempted_req_ids == {"run-0"}
    assert "vllm-evolve: running request preempted" in capsys.readouterr().out


def test_active_preemption_classifies_elephant_on_cold_arrival_path(monkeypatch):
    _install_fake_vllm(monkeypatch)
    policy = (
        ROOT / "targets" / "scheduling" / "seeds" / "ftc.py"
    ).read_text(encoding="utf-8")
    namespace = {}
    exec(compile(render_plugin(policy), "generated_scheduler.py", "exec"), namespace)
    scheduler = namespace["EvolvedScheduler"]
    inst = object.__new__(scheduler)
    inst.waiting = deque()
    inst._ve_has_seen_elephant = False

    short = _req("short", time.time(), prompt=64)
    short.max_tokens = 256
    scheduler.add_request(inst, short)
    assert inst._ve_has_seen_elephant is False

    elephant = _req("elephant", time.time(), prompt=64)
    elephant.max_tokens = 1792
    scheduler.add_request(inst, elephant)
    assert inst._ve_has_seen_elephant is True


def test_prefix_hint_requires_a_shared_real_block_hash(monkeypatch):
    _install_fake_vllm(monkeypatch)
    seed_src = (ROOT / "targets" / "scheduling" / "seed.py").read_text(encoding="utf-8")
    namespace = {}
    exec(compile(render_plugin(seed_src), "generated_scheduler.py", "exec"), namespace)
    first = SimpleNamespace(block_hash=123)
    second = SimpleNamespace(block_hash=456)
    assert namespace["_prefix_key"](SimpleNamespace(block_hashes=[first])) == 123
    counts = {123: 2, 456: 1}
    assert counts[namespace["_prefix_key"](SimpleNamespace(block_hashes=[first]))] > 1
    assert counts[namespace["_prefix_key"](SimpleNamespace(block_hashes=[second]))] == 1


def test_prefix_key_falls_back_to_waiting_prompt_block(monkeypatch):
    _install_fake_vllm(monkeypatch)
    seed_src = (ROOT / "targets" / "scheduling" / "seed.py").read_text(encoding="utf-8")
    namespace = {}
    exec(compile(render_plugin(seed_src), "generated_scheduler.py", "exec"), namespace)
    shared = list(range(16))
    first = SimpleNamespace(block_hashes=[], prompt_token_ids=shared + [20])
    second = SimpleNamespace(block_hashes=None, prompt_token_ids=shared + [21])
    short = SimpleNamespace(block_hashes=[], prompt_token_ids=[1, 2, 3])
    assert namespace["_prefix_key"](first) == namespace["_prefix_key"](second)
    assert namespace["_prefix_key"](short) is None
