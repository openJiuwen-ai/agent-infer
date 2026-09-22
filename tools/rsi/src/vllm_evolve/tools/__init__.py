"""
Tools — atomic capabilities exposed to scaffolds via CLI / MCP / Python API.

Each tool is self-contained and can be called independently:
  simulate  — evaluate code or config (auto-verifies, auto-stores)
  verify    — safety/legality check
  store     — CRUD for policies, configs, evaluations
  context   — assemble evolution context from DB
  diff      — parse/apply SEARCH/REPLACE diffs
  configure — Day-0 config search space operations
"""
