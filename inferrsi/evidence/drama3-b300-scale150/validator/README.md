Frozen for [acceptance issue #10](https://github.com/ThinkFlowLab/OmniRSI/issues/10).

Run `python validate.py <manifest.json>`. The script reads the sibling plan, recomputes metrics from raw request records, requires all quality repetitions, and checks the per-concurrency gates. RunSpec provenance and numerical validation artifacts still require review; the script cannot authenticate hardware or source identity. Missing evidence returns INCONCLUSIVE.

Freeze smoke: a complete real baseline C1 repetition passed internal consistency checks; missing quality, doubled summary throughput, and a missing sample were rejected. These are validator checks, not campaign acceptance.
