#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

rustup target add wasm32-wasip2 >/dev/null
cargo build --target wasm32-wasip2 --release

# wit-bindgen + wasm32-wasip2 already emits a component module.
SRC="target/wasm32-wasip2/release/agent_hint_affinity.wasm"
DEST="$ROOT/agent_hint_affinity.component.wasm"
cp -f "$SRC" "$DEST"
echo "Built $DEST"
sha256sum "$DEST" || shasum -a 256 "$DEST"
