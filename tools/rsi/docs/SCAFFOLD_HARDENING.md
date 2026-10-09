# Scaffold Hardening (Round 2 / AC-11)

> **Status:** Phase 1 scaffolding only. No live multi-runtime parity is
> claimed yet. The hardened surface exists so Phase 2 (OpenCode / Cursor /
> generic-LLM-API end-to-end parity) can build on a stable contract.

## What Round 2 Adds

Before Round 2, `vllm_evolve.protocols.ScaffoldProtocol` published a loose
capability list:

```python
class ScaffoldProtocol(Protocol):
    def generate(self, prompt: str) -> str: ...
    capabilities: frozenset[str]
```

That contract was easy to satisfy and equally easy to misuse: any
controller that wanted to know whether an adapter could do `tool_use`
had to inspect the string set by itself and decide what to do on
absence. Round 2 promotes the contract to a structured surface so
adapters and callers can negotiate explicitly:

```python
from vllm_evolve.protocols import (
    CapabilityDeclaration,
    NegotiationResult,
    ScaffoldProtocol,
    negotiate_default,
)
```

### `CapabilityDeclaration`

A frozen dataclass that describes a single capability with five fields:

| field | meaning |
|---|---|
| `name` | Stable identifier (e.g. `"tool_use"`). Must match the dict key the adapter publishes it under. |
| `version` | Semver-style contract version (e.g. `"1.0"`). Phase 2 callers may negotiate on version too. |
| `required_by_agent` | If `True`, the absence of this capability flips `NegotiationResult.ok` to `False`. Use sparingly; most capabilities should be optional with a documented fallback. |
| `fallback_semantics` | Human/agent-readable description of what happens when this capability is absent (e.g. *"strategy must collapse multi-turn iterations into a single SEARCH/REPLACE diff response"*). |
| `output_format_contract` | Symbolic name of the expected `generate(prompt)` output shape (`"SEARCH_REPLACE_DIFF"`, `"FULL_CODE"`, `"JSON_DECISION"`, ...). Round 2 does not enforce this; Phase 2 validation can. |

### `NegotiationResult`

A frozen dataclass returned by `negotiate_capabilities(required)`:

| field | meaning |
|---|---|
| `satisfied` | `frozenset[str]` of capabilities the adapter provides AND the caller asked for. |
| `missing` | `frozenset[str]` of capabilities the caller asked for but the adapter cannot provide. |
| `fallbacks` | `Mapping[str, str]` from each missing capability name to its `fallback_semantics` (or a generic fallback string if the adapter never declared that capability). |
| `ok` | `True` iff every missing capability is either *not* declared at all, or is declared with `required_by_agent=False`. Missing-but-optional capabilities do not flip `ok` to `False` on their own. |

### `negotiate_default(declarations, required)`

Helper that any adapter can delegate to. It compares `required` against
`declarations.keys()`, computes the partition, populates `fallbacks` from
the declaration when one exists (or a generic string when it does not),
and returns the `NegotiationResult`.

## How To Register A New Capability

1. **Pick a stable name.** Use lowercase snake_case (e.g.
   `"json_decision_output"`). The name is the dict key the adapter
   publishes the declaration under and what callers pass into
   `required`.
2. **Write a `CapabilityDeclaration`.** All five fields are required.
   Keep `fallback_semantics` agent-readable -- if your fallback string
   is "panic", future Phase-2 plumbing will simply repeat that to the
   strategy as a hint, so make it actionable.
3. **Add the entry to every adapter that supports the capability.**
   Adapters that do not support the capability simply omit the entry
   from their `capability_declarations`; they do NOT need to mention
   the missing capability anywhere -- the negotiation helper handles
   the missing case by surfacing the generic fallback string. Add the
   adapter to `tests/test_scaffold_hardening.py` parameter sets if it
   is new.
4. **Bump the version when changing the contract.** If a capability's
   semantics change, bump `version` rather than mutating the existing
   declaration. Phase 2 callers may pin to a specific version.

## How Phase 2 Plugs In

Phase 2 adds **end-to-end** runtime support for OpenCode, Cursor, and
generic-LLM-API adapters -- not just the protocol surface they all
already nominally satisfy. The Round-2 hardening is the scaffolding
for that:

1. **OpenCode wiring.** Today, `make_opencode_provider` is the same
   `FileIPCProvider` factory as `make_claude_code_provider` minus a
   prompt header comment. Phase 2 will (a) verify that an OpenCode
   session actually consumes the SKILL header, (b) record per-runtime
   variations in `capability_declarations` (for instance, OpenCode
   may support `multi_turn` only via an explicit slash-command
   pattern), and (c) add `tests/test_opencode_replay.py` with a
   deterministic recorded-response fixture and a CI hook.
2. **Cursor wiring.** Cursor's IDE-side semantics differ from a bare
   file-IPC watcher; Phase 2 will add a `CursorProvider` that wraps
   the Cursor MCP transport, declares its own capabilities, and plugs
   into the same negotiation surface.
3. **Generic-LLM-API REST adapter.** Phase 2 introduces an
   `OpenAICompatibleProvider` that takes a base URL + auth and
   declares only the capabilities it can prove (typically
   `multi_turn=False`, `tool_use` only if the endpoint advertises
   function-calling). The same `negotiate_capabilities` flow tells
   the controller which strategies degrade or fall back when the
   endpoint does not provide a capability the strategy wants.

In every case, the controller talks only to the `ScaffoldProtocol`
methods -- `generate`, `capability_declarations`, and
`negotiate_capabilities`. Adapter-specific code stays in
`src/vllm_evolve/scaffold/`.

## Explicit Phase-1 Disclaimers

* The Round-2 surface is **structural** only. It does not prove that
  any adapter other than Claude Code via `FileIPCProvider` actually
  drives this loop to completion. Today only Claude Code does.
* `make_opencode_provider` and `make_claude_code_provider` both
  produce `FileIPCProvider` instances. They differ only in the prompt
  header comment. Round 2 does NOT claim that an OpenCode session has
  ever driven this loop to completion.
* `AgentProvider` (the `claude --print` subprocess wrapper) declares
  no agentic capabilities. It is single-shot and is kept for
  backwards compatibility with the legacy `vllm-evolve` CLI workflow.
* `MockProvider` is for tests. It declares no capabilities and ships
  with a deterministic cycle of pre-built SEARCH/REPLACE diffs that
  target `targets/scheduling/seed.py`.

## Pointers

* Code: `src/vllm_evolve/protocols.py`,
  `src/vllm_evolve/scaffold/{mock, cli_agent, file_ipc}.py`.
* Tests: `tests/test_scaffold_hardening.py`.
* Plan reference: AC-11 in `.humanize/plans/vllm-evolve-rebuild.md`.
* Decision references: DEC-2 (multi-runtime parity is Phase 2+),
  DEC-8 (parity means adapter-invocation, not
  metric-reproducibility, and even that lands in Phase 2).
