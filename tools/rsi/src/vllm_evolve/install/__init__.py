"""Installer for the Claude Code assets (`ve init`).

Materializes the packaged agent / skills / hook / settings fragment into a
target project's ``.claude/`` directory, deep-merging ``settings.json`` rather
than clobbering it, and records a manifest so the install is idempotent and
cleanly reversible (``ve init --uninstall``).
"""
