# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Forward a ZCode synchronous hook to its task-local policy owner."""

import sys
import urllib.request


def main() -> None:
    request = urllib.request.Request(
        sys.argv[1], data=sys.stdin.buffer.read(), headers={"Content-Type": "application/json"}
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=10) as response:
        sys.stdout.buffer.write(response.read())


if __name__ == "__main__":
    main()
