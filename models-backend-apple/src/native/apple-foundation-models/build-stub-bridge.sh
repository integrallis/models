#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${1:-"$SCRIPT_DIR/build"}"

if ! command -v cc >/dev/null 2>&1; then
  echo "cc is required to build the Apple Foundation Models native stub." >&2
  exit 2
fi

mkdir -p "$OUT_DIR"

case "$(uname -s)" in
  Darwin)
    LIBRARY_NAME="libjavamodels_apple_foundation_stub.dylib"
    cc -O2 -dynamiclib -fPIC \
      -o "$OUT_DIR/$LIBRARY_NAME" \
      "$SCRIPT_DIR/AppleFoundationStubBridge.c"
    ;;
  Linux)
    LIBRARY_NAME="libjavamodels_apple_foundation_stub.so"
    cc -O2 -shared -fPIC \
      -o "$OUT_DIR/$LIBRARY_NAME" \
      "$SCRIPT_DIR/AppleFoundationStubBridge.c"
    ;;
  *)
    echo "unsupported OS for native stub bridge: $(uname -s)" >&2
    exit 2
    ;;
esac

echo "$OUT_DIR/$LIBRARY_NAME"
