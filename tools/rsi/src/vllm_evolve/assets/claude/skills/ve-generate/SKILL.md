---
name: ve-generate
description: Phase 3 (GENERATE) of a vllm-evolve round — implement the designed change in the policy file. This is the ONLY phase where policy files may be edited.
---

# ve-generate — write the change

Run `ve phase set GENERATE`, then edit the policy file (e.g.
`targets/scheduling/work.py`). Constraints:

- Edit **only inside** the `# EVOLVE-BLOCK-START` / `# EVOLVE-BLOCK-END` markers.
- Keep the function signature identical to the skeleton.
- Implement exactly the change from your design note — nothing extra.
- The phase hook blocks policy-file edits outside GENERATE and blocks edits to
  seeds, skeletons, configs, schemas, and the archive.

Next: the **ve-verify** skill.
