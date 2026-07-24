# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentCache project

from unittest.mock import MagicMock

import pytest
from vllm.v1.core.sched.request_queue import FCFSRequestQueue, RequestQueue

from agentinfer.agentcache.core.request_queue import AgentAwareQueue

pytestmark = pytest.mark.cpu_test


def _mock_request(request_id: str) -> MagicMock:
    r = MagicMock()
    r.request_id = request_id
    return r


class TestAgentAwareQueueABC:
    def test_implements_request_queue_abc(self):
        assert issubclass(AgentAwareQueue, RequestQueue)

    def test_can_instantiate(self):
        q = AgentAwareQueue()
        assert isinstance(q, RequestQueue)

    def test_wraps_fcfs_request_queue(self):
        q = AgentAwareQueue()
        assert isinstance(q._q, FCFSRequestQueue)


class TestAgentAwareQueueEmpty:
    def test_bool_false(self):
        assert not AgentAwareQueue()

    def test_len_zero(self):
        assert len(AgentAwareQueue()) == 0

    def test_pop_raises_on_empty(self):
        with pytest.raises(IndexError):
            AgentAwareQueue().pop_request()

    def test_peek_raises_on_empty(self):
        with pytest.raises(IndexError):
            AgentAwareQueue().peek_request()

    def test_iter_yields_nothing(self):
        assert list(AgentAwareQueue()) == []

    def test_reversed_yields_nothing(self):
        assert list(reversed(AgentAwareQueue())) == []


class TestAgentAwareQueueFCFS:
    """Verify FCFS ordering — identical to :class:`FCFSRequestQueue`."""

    def test_add_then_pop_fcfs_order(self):
        r1, r2, r3 = _mock_request("a"), _mock_request("b"), _mock_request("c")
        q = AgentAwareQueue()
        q.add_request(r1)
        q.add_request(r2)
        q.add_request(r3)
        assert q.pop_request().request_id == "a"
        assert q.pop_request().request_id == "b"
        assert q.pop_request().request_id == "c"

    def test_same_order_as_fcfsrequestqueue(self):
        r1, r2, r3 = _mock_request("a"), _mock_request("b"), _mock_request("c")
        aq = AgentAwareQueue()
        fq = FCFSRequestQueue()
        for r in (r1, r2, r3):
            aq.add_request(r)
            fq.add_request(r)
        while aq:
            assert aq.pop_request().request_id == fq.pop_request().request_id


class TestAgentAwareQueuePeek:
    def test_peek_does_not_remove(self):
        r = _mock_request("a")
        q = AgentAwareQueue()
        q.add_request(r)
        assert q.peek_request().request_id == "a"
        assert len(q) == 1

    def test_peek_empty_raises(self):
        with pytest.raises(IndexError):
            AgentAwareQueue().peek_request()


class TestAgentAwareQueuePrepend:
    def test_prepend_front(self):
        r1, r2 = _mock_request("a"), _mock_request("b")
        q = AgentAwareQueue()
        q.add_request(r1)
        q.prepend_request(r2)
        assert q.pop_request().request_id == "b"
        assert q.pop_request().request_id == "a"

    def test_prepend_requests_bulk(self):
        r1 = _mock_request("a")
        r2, r3 = _mock_request("b"), _mock_request("c")
        q = AgentAwareQueue()
        q.add_request(r1)
        q.prepend_requests([r2, r3])
        assert q.pop_request().request_id == "c"
        assert q.pop_request().request_id == "b"
        assert q.pop_request().request_id == "a"


class TestAgentAwareQueueRemove:
    def test_remove_existing(self):
        r1, r2 = _mock_request("a"), _mock_request("b")
        q = AgentAwareQueue()
        q.add_request(r1)
        q.add_request(r2)
        q.remove_request(r1)
        assert len(q) == 1
        assert q.pop_request().request_id == "b"

    def test_remove_nonexistent_raises_valueerror(self):
        q = AgentAwareQueue()
        q.add_request(_mock_request("a"))
        with pytest.raises(ValueError):
            q.remove_request(_mock_request("z"))
        assert len(q) == 1

    def test_remove_requests_bulk(self):
        r1, r2, r3 = _mock_request("a"), _mock_request("b"), _mock_request("c")
        q = AgentAwareQueue()
        for r in (r1, r2, r3):
            q.add_request(r)
        q.remove_requests([r1, r3])
        assert len(q) == 1
        assert q.pop_request().request_id == "b"


class TestAgentAwareQueueLenAndBool:
    def test_len_reflects_contents(self):
        q = AgentAwareQueue()
        q.add_request(_mock_request("a"))
        q.add_request(_mock_request("b"))
        assert len(q) == 2
        q.pop_request()
        assert len(q) == 1
        q.pop_request()
        assert len(q) == 0

    def test_bool_reflects_contents(self):
        q = AgentAwareQueue()
        assert not q
        q.add_request(_mock_request("a"))
        assert q

    def test_bool_after_pop_all(self):
        q = AgentAwareQueue()
        q.add_request(_mock_request("a"))
        q.pop_request()
        assert not q


class TestAgentAwareQueueIter:
    def test_iter_order_fcfs(self):
        r1, r2 = _mock_request("a"), _mock_request("b")
        q = AgentAwareQueue()
        q.add_request(r1)
        q.add_request(r2)
        assert [r.request_id for r in q] == ["a", "b"]

    def test_reversed_order(self):
        r1, r2 = _mock_request("a"), _mock_request("b")
        q = AgentAwareQueue()
        q.add_request(r1)
        q.add_request(r2)
        assert [r.request_id for r in reversed(q)] == ["b", "a"]


class TestAgentAwareQueueMultipleOperations:
    def test_interleaved_add_pop_prepend(self):
        q = AgentAwareQueue()
        q.add_request(_mock_request("a"))
        q.add_request(_mock_request("b"))
        assert q.pop_request().request_id == "a"
        q.prepend_request(_mock_request("c"))
        q.add_request(_mock_request("d"))
        assert q.pop_request().request_id == "c"
        assert q.pop_request().request_id == "b"
        assert q.pop_request().request_id == "d"
