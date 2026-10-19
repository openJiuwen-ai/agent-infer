"""``ve`` command-line interface (L3 surface).

``cli/main.py`` holds the argument parser + verb dispatch (renamed from the
former top-level ``ar_cli.py`` with the ``ar``→``ve`` rename). The verb→phase
table is owned by ``tools/phase_guard.py`` (M4), so this package carries no
import-time registration side effect.
"""
