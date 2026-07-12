#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${1:-"$SCRIPT_DIR/build"}"
LIBRARY_NAME="libjavamodels_apple_foundation.dylib"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "Apple Foundation Models bridge requires macOS." >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-26.0}"

xcrun --sdk macosx swiftc \
  -O \
  -emit-library \
  -parse-as-library \
  -module-name JavaModelsAppleFoundationBridge \
  -o "$OUT_DIR/$LIBRARY_NAME" \
  "$SCRIPT_DIR/AppleFoundationBridge.swift"

echo "$OUT_DIR/$LIBRARY_NAME"
