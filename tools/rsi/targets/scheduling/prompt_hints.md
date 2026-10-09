# Scheduling Target — Domain Knowledge

## What you're optimizing

`schedule_batch()` decides which waiting requests to admit for prefill and which
running requests to continue decoding, subject to token budget and KV block constraints.
This is called every scheduling step (~10ms) in vLLM's continuous batching loop.

## Key fields

| Field | Type | Meaning |
|-------|------|---------|
| `num_prompt_tokens` | int | Total prompt length (input tokens) |
| `num_computed_tokens` | int | Tokens already computed (0 for new requests) |
| `num_output_tokens` | int | Tokens generated so far (0 for prefill) |
| `arrival_time_s` | float | When the request arrived (seconds since start) |
| `is_prefill` | bool | True if request is still in prefill phase |
| `kv_blocks_used` | int | KV cache blocks currently allocated to this request |
| `prefix_cached_tokens` | int | Tokens that hit the prefix cache (avoid recompute) |
| `num_preemptions` | int | Times this request was preempted (higher = starved) |
| `available_kv_blocks` | int | Free KV blocks in the cache (resource signal) |
| `prefix_cache_hit_rate` | float | Global cache hit rate (0-1) |

## vLLM's real behavior (seed should mimic this)

vLLM's actual scheduler uses **decode-first priority**:
1. Always continue running decode requests first (they're cheap per-token)
2. Then admit new prefill requests up to `max_num_batched_tokens` budget
3. Chunked prefill: large prompts can be split across multiple steps
4. Preemption: when KV is full, preempt the newest/lowest-priority request

## Promising mutation directions

| Symptom | Direction | Why it works |
|---------|-----------|-------------|
| High P99 TTFT | SRPT: prioritize requests with fewer remaining tokens | Short requests finish fast, free resources |
| Low throughput at high QPS | Tighter decode packing before admitting prefill | More tokens generated per step |
| KV pressure causing preemption cascades | KV-gated admission: reduce admission rate as KV fills | Prevents thrashing |
| All metrics flat vs FCFS | Check if `available_kv_blocks` is actually used | Many seeds ignore this signal |
| Decode starvation | Age penalty: boost priority of long-waiting requests | Prevents indefinite blocking |
| Bimodal workload issues | Classify requests as short/long, route differently | Prevents head-of-line blocking |

## Field usage frequency (which fields are underexploited)

| Field | Used by seed? | Used by evolved best? | Untapped potential |
|-------|:---:|:---:|---|
| `arrival_time_s` | ✓ (FCFS sort) | ✓ (age bonus) | Low — already exploited |
| `remaining_prompt_tokens` | ✓ (budget check) | ✓ (SRPT sort) | Low |
| `available_kv_blocks` | ✗ | ✓ (KV gating) | Medium — used but could be more sophisticated |
| `prefix_cached_tokens` | ✗ | Partially | **HIGH** — cache-affinity scheduling barely explored |
| `num_preemptions` | ✗ | ✓ (starvation bonus) | Medium |
| `kv_blocks_used` | ✗ | ✗ | **HIGH** — per-request KV cost never used for priority |
| `num_output_tokens` | ✗ | ✗ | **HIGH** — decode progress never used to prioritize near-completion |
| `prefix_cache_hit_rate` | ✗ | ✗ | **HIGH** — global cache state never drives admission decisions |

## Interaction effects (non-obvious)

1. **SRPT + high KV pressure = bad**: SRPT admits many short requests → many KV allocations → KV fills fast → preemption cascade. Fix: combine SRPT with KV gating.
2. **Aggressive admission + prefix caching = good**: Admitting more requests that share prefixes increases hit rate, which amortizes the KV cost.
3. **Decode-first + chunked prefill = complementary**: Decode is cheap per-step. Chunking long prefill across steps lets decode continue. Both reduce TTFT.

## Proactive preemption via `preempt_ids`

`ScheduleDecision` has a third field, `preempt_ids`, that most seeds leave empty.
Setting it tells the simulator to evict the listed sequences from KV cache and move
them back to the waiting queue.  This is the **only** way the evolved scheduler can
trigger preemption proactively — otherwise preemption only fires reactively when
KV is completely exhausted (free_blocks == 0), which causes latency spikes.

### When to preempt proactively

| Signal | Strategy | Benefit |
|--------|----------|---------|
| `available_kv_blocks` dropping below a threshold (e.g., < 10 % of capacity) | Preempt the decode sequence using the most `kv_blocks_used` | Frees KV before it hits zero; prevents reactive preemption storms |
| A decode request has very high `num_output_tokens` relative to others | Preempt it to let shorter, near-completion requests finish | Reduces tail latency; SRPT-style fairness |
| `num_preemptions > 0` on a waiting request while a low-priority decode runs | Preempt the low-priority decode to re-admit the starved request | Prevents indefinite starvation |
| `prefix_cache_hit_rate` is high but KV is tight | Preempt sequences with no prefix-cache benefit (`prefix_cached_tokens == 0`) | Preserves cached blocks for requests that can reuse them |

### Example usage

```python
preempt_ids = []
# Proactive preemption: if KV is getting tight and there are waiting requests
kv_pressure = available_kv_blocks / max(max_num_batched_tokens // 16, 1)
if kv_pressure < 0.15 and waiting_requests:
    # Preempt the decode request holding the most KV blocks
    decode_reqs = [r for r in running_requests if not r.is_prefill]
    if decode_reqs:
        victim = max(decode_reqs, key=lambda r: r.kv_blocks_used)
        preempt_ids.append(victim.request_id)

return ScheduleDecision(
    prefill_batch=prefill_batch,
    decode_batch=[r for r in decode_batch if r not in preempt_ids],
    preempt_ids=preempt_ids,
)
```

### Preemption pitfalls

- **Thrashing**: preempting a request and re-admitting it next step wastes all its
  decoded tokens (they must be re-prefilled).  Only preempt when the freed KV will
  be used by a higher-value admission.
- **Ping-pong**: avoid preempting a request that has `num_preemptions > 2` — it has
  already lost significant work.  Prefer victims with `num_preemptions == 0`.
- **Double-listing**: a request in `preempt_ids` must NOT also appear in
  `decode_batch`.  The simulator handles conflicts but the intent is clearer when
  they are mutually exclusive.

## Constraints

- `prefill_ids` must be a subset of `[r.request_id for r in waiting_requests]`
- `decode_ids` must be a subset of `[r.request_id for r in running_requests]`
- `preempt_ids` must be a subset of `[r.request_id for r in running_requests]` (only running requests can be preempted)
- Total tokens in batch must not exceed `max_num_batched_tokens`
- Must not exceed `max_num_seqs` sequences in one batch
- Empty decision is valid (do nothing this step) but hurts throughput
- No O(n^2) operations — `waiting_requests` can be thousands of items
