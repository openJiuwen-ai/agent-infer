---
name: ve-decide
description: Phases 6-7 (KEEP_OR_DISCARD → COMMIT_OR_ROLLBACK) of a vllm-evolve round — compare candidate vs baseline and keep or discard based on the verdict.
---

# ve-decide — compare and commit

1. `ve phase set KEEP_OR_DISCARD`, then
   `ve compare <baseline.json> <candidate.json>`. The verdict is one of
   **better** / **worse** / **inconclusive** / **high_variance_inconclusive**
   (a real difference must exceed seed-to-seed noise).
2. `ve phase set COMMIT_OR_ROLLBACK`, then:
   - **better** → `ve keep <policy> --eval-result <candidate.json>` (archives +
     stores + git-commits the kept policy).
   - **worse / inconclusive** → `ve discard <policy> --reason "<why>"`, then loop
     back to ve-context with a refined idea.

Never keep on anything but a real `better` verdict. `high_variance_inconclusive`
means the measurement isn't trustworthy — re-bench with more seeds, don't keep.
