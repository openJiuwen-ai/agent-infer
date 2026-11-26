# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Program identities and cache classes on the EngineCore thread."""

from collections.abc import Callable
from enum import Enum
from itertools import count

from agentinfer.scheduling.domain import ProgramRef


class CacheClass(str, Enum):
    INACTIVE = "inactive"
    IDLE = "idle"
    BUSY = "busy"
    SHARED = "shared"


PROGRAM_ARG = "mori_program_id"
PROGRAM_TYPES = {"interactive": CacheClass.BUSY, "async": CacheClass.IDLE, "inactive": CacheClass.INACTIVE}


def request_program(request) -> ProgramRef | str | None:
    """Return the admitted generation on engine requests, or the pre-admission identifier."""
    if hasattr(request, "_mori_program_ref"):
        return request._mori_program_ref
    params = request.sampling_params
    value = (params.extra_args or {}).get(PROGRAM_ARG) if params is not None else None
    return value if isinstance(value, str) and value else None


class MoriProgramRegistry:
    """Track cache classes on the EngineCore thread, using router decisions."""

    def __init__(self) -> None:
        self.current_ref: Callable[[str], ProgramRef | None] | None = None
        self._generation = count()
        self._programs: dict[str, ProgramRef] = {}
        self._classes: dict[ProgramRef, CacheClass] = {}

    @property
    def has_programs(self) -> bool:
        return bool(self._programs)

    def has_program(self, program: str | ProgramRef | None) -> bool:
        if isinstance(program, ProgramRef):
            return program in self._classes
        return program is not None and program in self._programs

    def type_for_id(self, program: str | ProgramRef | None) -> CacheClass:
        ref = self._programs.get(program) if isinstance(program, str) else program
        return self._classes.get(ref, CacheClass.INACTIVE)

    def register(self, program: str | ProgramRef) -> ProgramRef | None:
        if isinstance(program, ProgramRef):
            ref = program
            if self.current_ref is None or self.current_ref(ref.program_id) != ref:
                return None
            previous = self._programs.get(ref.program_id)
            if previous == ref:
                return previous
            if previous is not None:
                self._classes.pop(previous)
        else:
            if self.current_ref is not None:
                raise ValueError("Embedded HiRadix requires an admitted Program generation")
            if (ref := self._programs.get(program)) is not None:
                return ref
            ref = ProgramRef(program, next(self._generation))
        self._programs[ref.program_id] = ref
        self._classes[ref] = CacheClass.BUSY
        return ref

    def mark(self, program: str | ProgramRef, program_type: str) -> None:
        if (ref := self.register(program)) is not None:
            self._classes[ref] = PROGRAM_TYPES[program_type]

    def release(self, program: str | ProgramRef) -> None:
        pid = program if isinstance(program, str) else program.program_id
        ref = self._programs.get(pid)
        if ref is not None and (isinstance(program, str) or program == ref):
            del self._programs[pid]
            del self._classes[ref]

    def placements(self) -> list[dict]:
        return [
            {"program_id": ref.program_id, "generation": ref.generation, "class": kind.value}
            for ref, kind in sorted(self._classes.items())
        ]
