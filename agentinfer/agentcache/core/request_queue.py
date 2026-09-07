# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentCache project

from vllm.v1.core.sched.request_queue import FCFSRequestQueue, RequestQueue


class AgentAwareQueue(RequestQueue):
    """Wraps :class:`FCFSRequestQueue` with identical FCFS behaviour.

    The wrapper exists as an extension point so subclasses or patches
    can inject agent-aware scheduling policies later without changing
    the scheduler itself.
    """

    def __init__(self) -> None:
        self._q = FCFSRequestQueue()

    # -- RequestQueue ABC ---------------------------------------------------

    def add_request(self, request) -> None:
        self._q.add_request(request)

    def pop_request(self):
        return self._q.pop_request()

    def peek_request(self):
        return self._q.peek_request()

    def prepend_request(self, request) -> None:
        self._q.prepend_request(request)

    def prepend_requests(self, requests: list) -> None:
        self._q.prepend_requests(requests)

    def remove_request(self, request) -> None:
        self._q.remove_request(request)

    def remove_requests(self, requests: list) -> None:
        self._q.remove_requests(requests)

    def __bool__(self) -> bool:
        return bool(self._q)

    def __len__(self) -> int:
        return len(self._q)

    def __iter__(self):
        return iter(self._q)

    def __reversed__(self):
        return reversed(self._q)
