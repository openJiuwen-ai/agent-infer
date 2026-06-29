---
name: ac-integrate
description: Use when plugging AgentCache into an existing vLLM (or compatible) serving deployment; detects engine version, wires the prefix-cache adapter, and validates cache hits.
---

# ac-integrate

Wire AgentCache into an existing LLM serving deployment (vLLM or compatible),
version-check the engine's prefix-cache API, and validate that caching actually
fires.

## WHEN TO INVOKE

- The user asks to "integrate AgentCache with vLLM", "plug AgentCache into my
  deployment", or "wire the adapter" for a serving engine.
- A new engine adapter is being added under `src/agentcache/adapters/`.

Do NOT invoke for: adding a cache backend (use `ac-bootstrap`), benchmarking
(use `ac-benchmark`), or editing integration docs only.

## STEPS

1. Detect the target engine and version:

   ```bash
   python -c "import vllm; print(vllm.__version__)" 2>/dev/null \
     || echo "vllm not importable; read the deployment's declared version"
   ```

   Record the version — the prefix-cache API differs across vLLM releases.
2. Locate the integration adapter under `src/agentcache/adapters/`. If the
   directory or adapter does not exist, create it:

   ```bash
   mkdir -p src/agentcache/adapters
   ```

3. Wire the adapter per the engine's prefix-cache API **for the detected
   version**. Do not assume a cache API shape without checking that version's
   docs.
4. Run a cache-hit validation: send the same prompt prefix twice and assert
   the second call is faster (or reports a cache hit). Example shape:

   ```bash
   python - <<'PY'
   import os
   from agentcache.adapters import vllm as acv

   url = os.getenv("AGENTCACHE_VLLM_URL", "")
   ac = acv.connect(url)
   t1 = ac.time_call("Once upon a time,")
   t2 = ac.time_call("Once upon a time,")
   assert t2 < t1, f"no cache speedup: {t2=}, {t1=}"
   print("cache hit validated")
   PY
   ```

5. Document the deployment specifics (engine, version, endpoint shape, any
   non-default config) in the integration notes so the next integration is
   reproducible.

## DON'T

- Don't assume a vLLM version's cache API without checking that version.
- Don't mutate the user's deployment config files without explicit
  confirmation.
- Don't declare integration done until the cache-hit validation passes.

## AFTER

Once the adapter is wired and cache hits are validated, invoke `ac-review` to
validate the diff against repo conventions before opening the PR.
