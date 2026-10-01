#!/usr/bin/env bash
# Build and run the WEF-Cleanup unit checks on the CPU (no trainer, no GPU).
#   ./ocean/wef/run_test.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
# CFLAGS=-DMAX_AGENTS=8 builds the 8-fish harness as build/wef_test8 (the suffix is the
# MAX_AGENTS value); the default build stays build/wef_test.
CFLAGS="${CFLAGS:-}"
SUFFIX=$(printf '%s' "$CFLAGS" | sed -n 's/.*-DMAX_AGENTS=\([0-9][0-9]*\).*/\1/p')
OUT=build/wef_test$SUFFIX
mkdir -p build
CC="${CC:-gcc}"
# shellcheck disable=SC2086
"$CC" -O1 -g -DNDEBUG_RAYLIB -mavx2 -mfma $CFLAGS \
    -I./raylib-5.5_linux_amd64/include -I./src -I./vendor -I./ocean/wef \
    ocean/wef/wef_test.c -o "$OUT" \
    raylib-5.5_linux_amd64/lib/libraylib.a -lGL -lm -lpthread -ldl -lrt -DPLATFORM_DESKTOP
"$OUT"
