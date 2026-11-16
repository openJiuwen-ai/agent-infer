#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Require the AgentInfer SPDX header on source files.

Python and shell files use the ``#`` header. Rust files use ``//``. A shebang
stays on line 1. An older AgentCache copyright line is rewritten in place.
Only ``.py``, ``.pyi``, ``.sh``, and ``.rs`` files are checked; other paths are ignored.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

LICENSE_TEXT = "SPDX-License-Identifier: Apache-2.0"
COPYRIGHT_TEXT = "SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project"
LEGACY_COPYRIGHT_TEXT = "SPDX-FileCopyrightText: Copyright contributors to the AgentCache project"


@dataclass(frozen=True)
class HeaderStyle:
    """Comment syntax and preamble handling for an SPDX header."""

    comment_prefix: str
    preserve_shebang: bool = False


class SPDXStatus(Enum):
    """SPDX header status enumeration."""

    EMPTY = "empty"
    COMPLETE = "complete"
    MISSING_LICENSE = "missing_license"
    MISSING_COPYRIGHT = "missing_copyright"
    MISSING_BOTH = "missing_both"


FILE_STYLES = {
    ".py": HeaderStyle("#", preserve_shebang=True),
    ".pyi": HeaderStyle("#"),
    ".sh": HeaderStyle("#", preserve_shebang=True),
    ".rs": HeaderStyle("//"),
}


def file_style(file_path: str) -> HeaderStyle:
    """Return the declared header style for a file."""
    suffix = Path(file_path).suffix
    try:
        return FILE_STYLES[suffix]
    except KeyError:
        raise ValueError(f"Unsupported file type: {file_path}") from None


def spdx_header(style: HeaderStyle) -> tuple[str, str]:
    """Return the SPDX header lines for a file style."""
    return (
        f"{style.comment_prefix} {LICENSE_TEXT}",
        f"{style.comment_prefix} {COPYRIGHT_TEXT}",
    )


def header_insertion_index(style: HeaderStyle, lines: list[str]) -> int:
    """Return the line index where a missing header should be inserted."""
    if style.preserve_shebang and lines and lines[0].startswith("#!"):
        return 1
    return 0


def check_spdx_header_status(file_path: str) -> SPDXStatus:
    """Check SPDX header status of the file."""
    style = file_style(file_path)
    license_line, copyright_line = spdx_header(style)
    lines = Path(file_path).read_text(encoding="UTF-8").splitlines()
    if not lines:
        return SPDXStatus.EMPTY

    has_license = any(line.strip() == license_line for line in lines)
    has_copyright = any(line.strip() == copyright_line for line in lines)
    if has_license and has_copyright:
        return SPDXStatus.COMPLETE
    if has_license:
        return SPDXStatus.MISSING_COPYRIGHT
    if has_copyright:
        return SPDXStatus.MISSING_LICENSE
    return SPDXStatus.MISSING_BOTH


def add_header(file_path: str, status: SPDXStatus) -> None:
    """Add or supplement the SPDX header based on status."""
    style = file_style(file_path)
    license_line, copyright_line = spdx_header(style)
    stale_copyright_line = f"{style.comment_prefix} {LEGACY_COPYRIGHT_TEXT}"
    path = Path(file_path)
    lines = path.read_text(encoding="UTF-8").splitlines(keepends=True)

    if status == SPDXStatus.MISSING_BOTH:
        insertion_index = header_insertion_index(style, lines)
        header = f"{license_line}\n{copyright_line}\n"
        remaining = lines[insertion_index:]
        if remaining and remaining[0].strip():
            header += "\n"
        lines = lines[:insertion_index] + [header] + remaining
    elif status == SPDXStatus.MISSING_COPYRIGHT:
        replaced = False
        for index, line in enumerate(lines):
            if line.strip() == stale_copyright_line:
                lines[index] = f"{copyright_line}\n"
                replaced = True
                break
        if not replaced:
            for index, line in enumerate(lines):
                if line.strip() == license_line:
                    lines.insert(index + 1, f"{copyright_line}\n")
                    break
    elif status == SPDXStatus.MISSING_LICENSE:
        for index, line in enumerate(lines):
            if line.strip() == copyright_line:
                lines.insert(index, f"{license_line}\n")
                break

    path.write_text("".join(lines), encoding="UTF-8")


def main(argv: list[str]) -> int:
    """Check each path and insert a missing AgentInfer SPDX header."""
    to_fix: list[tuple[str, SPDXStatus]] = []
    for file_path in argv:
        if Path(file_path).suffix not in FILE_STYLES:
            continue
        status = check_spdx_header_status(file_path)
        if status in {SPDXStatus.COMPLETE, SPDXStatus.EMPTY}:
            continue
        to_fix.append((file_path, status))

    if not to_fix:
        return 0

    print("The following files are missing the SPDX header:")
    for file_path, status in to_fix:
        print(f"  {file_path}")
        add_header(file_path, status)
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
