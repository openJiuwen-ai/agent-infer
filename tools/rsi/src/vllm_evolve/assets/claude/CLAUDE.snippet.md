<!-- vllm-evolve:start -->
## vllm-evolve harness

This project uses the **vllm-evolve** harness to optimize vLLM serving policies.

- Use the `vllm-policy-optimizer` agent (or the `ve-*` skills) to run a round.
- A round is phase-locked via the `ve` CLI:
  `ve context` → `ve design` → edit → `ve verify` → `ve bench` → `ve compare` →
  `ve keep` / `ve discard`.
- A PreToolUse hook enforces the phase machine and **blocks edits to ground-truth
  files** (seeds, skeletons, configs, schemas, the archive, `.claude/`).
- Evaluation is **real vLLM only** (Docker, GPU). Never claim an improvement
  without an `ve bench` + `ve compare` verdict; never fake metrics.
<!-- vllm-evolve:end -->
