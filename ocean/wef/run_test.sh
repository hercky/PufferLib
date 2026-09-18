#!/usr/bin/env bash
# Build and run the WEF-Cleanup unit checks on the CPU (no trainer, no GPU).
#   ./ocean/wef/run_test.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
OUT=build/wef_test
mkdir -p build
CC="${CC:-gcc}"
"$CC" -O1 -g -DNDEBUG_RAYLIB -mavx2 -mfma \
    -I./raylib-5.5_linux_amd64/include -I./src -I./vendor -I./ocean/wef \
    ocean/wef/wef_test.c -o "$OUT" \
    raylib-5.5_linux_amd64/lib/libraylib.a -lGL -lm -lpthread -ldl -lrt -DPLATFORM_DESKTOP
"$OUT"
