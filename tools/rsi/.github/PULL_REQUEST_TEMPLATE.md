## Summary

<!-- 1–3 bullet points describing what this PR does -->

## Type of change

- [ ] New scheduling seed policy
- [ ] New KV eviction seed policy
- [ ] Bug fix
- [ ] Simulator improvement
- [ ] Documentation
- [ ] Tests
- [ ] Other

## Testing

<!-- Paste the relevant output showing your change works -->

```bash
python -m pytest tests/ -q
# Expected: 21 passed

python benchmarks/eval_scheduling.py
# If adding a scheduling policy: show its row in the output table
```

## Checklist

- [ ] `pytest tests/ -q` passes (21 tests)
- [ ] My policy imports only allowed modules (`math`, `random`, `heapq`, `collections`, `itertools`, `functools`)
- [ ] I import fitness from `vllm_evolve.fitness`, not a local copy
- [ ] I did not modify `vllm/` (vendored upstream — read-only)
- [ ] I did not use `kv_cache` as a target name (use `kv_eviction`)
