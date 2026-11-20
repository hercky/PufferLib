#!/usr/bin/env bash
# Build and run the WEF unit checks on the CPU (no trainer, no GPU): the Cleanup / Harvest /
# Commons suite and the Allelopathic Harvest checks of docs/wef-allelopathic-harvest-v0-design.md
# 7.1 (U1-U10, the episode Log and the V3 trace in wef_test.c; U11 and U12 below).
#   ./ocean/wef/run_test.sh                        # default 4-fish build:  build/wef_test
#   CFLAGS=-DMAX_AGENTS=8 ./ocean/wef/run_test.sh  # 8-fish build:          build/wef_test8
#   CFLAGS="-DMAX_AGENTS=8 -DWEF_INST" ./ocean/wef/run_test.sh   # institution build: build/wef_test8i
#   CFLAGS=-DWEF_INST ./ocean/wef/run_test.sh                    # 4-fish institution build: build/wef_test4i
#   CFLAGS="-DMAX_AGENTS=8 -DWEF_INST -DWEF_META" ./ocean/wef/run_test.sh   # options build: build/wef_test8m
# An options build (-DWEF_META, the WEF_META block at the top of wef.h) runs its own unit checks (U57-U69; the raw-action suites need
# 4 action floats per fish) and, instead of U13, U13m: with every fish scripted its episodes must be bit-identical to the
# same-MAX_AGENTS institution build's (calibration hash=1 over obs rows and rewards, plus every printed metric).
# An institution build also runs U13: at inst_obs 0 its bench projection onto the base slots and its
# rewards hash must equal the same-MAX_AGENTS plain build's (project=MAX_AGENTS), i.e. the extra slots
# are the only difference.
# Either run ends with U11 (the default bench hash, scripts/bench_env.sh variant o2) and U12
# (projection identity: the 8-fish build at num_agents=4 reproduces the 4-fish build's rewards
# hash and 110-slot obs projection over 20K bench steps), so both binaries are built.
#   env: WEF_BENCH_SH=path   bench_env.sh to use for U11 (default: ../scripts/bench_env.sh
#                            next to PufferLib; a local -O2 build of the harness otherwise)
#        WEF_TEST_SKIP_BENCH=1  run the unit suite only
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
CFLAGS="${CFLAGS:-}"
NFISH=$(printf '%s' "$CFLAGS" | sed -n 's/.*-DMAX_AGENTS=\([0-9][0-9]*\).*/\1/p')
INST=0
META=0
case "$CFLAGS" in *-DWEF_INST*) INST=1 ;; esac
case "$CFLAGS" in *-DWEF_META*) META=1 ;; esac
SUFFIX=$NFISH
if [ "$INST" = 1 ]; then SUFFIX="${NFISH:-4}i"; fi
if [ "$META" = 1 ]; then SUFFIX="${NFISH:-4}m"; fi
BASE_CFLAGS=$(printf '%s' "$CFLAGS" | sed 's/-DMAX_AGENTS=[0-9][0-9]*//; s/-DWEF_INST//; s/-DWEF_META//')
OUT=build/wef_test$SUFFIX
mkdir -p build
CC="${CC:-gcc}"

compile() {  # compile <out> <cflags...>
    local out=$1
    shift
    "$CC" "$@" -DNDEBUG_RAYLIB \
        -I./raylib-5.5_linux_amd64/include -I./src -I./vendor -I./ocean/wef \
        ocean/wef/wef_test.c -o "$out" \
        raylib-5.5_linux_amd64/lib/libraylib.a -lGL -lm -lpthread -ldl -lrt -DPLATFORM_DESKTOP
}

# shellcheck disable=SC2086
compile "$OUT" -O1 -g -mavx2 -mfma $CFLAGS
TRACE_DIR="$ROOT/build/wef_test_trace.$$"
mkdir -p "$TRACE_DIR"
trap 'rm -rf "$TRACE_DIR"' EXIT
WEF_TEST_TRACE_DIR="$TRACE_DIR" "$OUT"

if [ "${WEF_TEST_SKIP_BENCH:-0}" = 1 ]; then
    exit 0
fi
fail=0

# U11: the default 4-fish build's bench hash (variant o2 = the trainer's host compile, -O2).
REF=189bd345928df398
BENCH_SH=${WEF_BENCH_SH:-$ROOT/../scripts/bench_env.sh}
if [ -x "$BENCH_SH" ]; then
    how="bench_env.sh variant o2"
    hash=$(VARIANTS=o2 CFLAGS= bash "$BENCH_SH" steps=20000 | sed -n 's/^  hash \([0-9a-f]*\)$/\1/p')
else
    how="local -O2 build (bench_env.sh not found at $BENCH_SH)"
    # shellcheck disable=SC2086
    compile build/wef_bench_o2 -O2 $BASE_CFLAGS
    hash=$(build/wef_bench_o2 bench steps=20000 | sed -n 's/^  hash \([0-9a-f]*\)$/\1/p')
fi
if [ "$hash" = "$REF" ]; then
    echo "ok   U11 default bench hash $hash unchanged ($how)"
else
    echo "FAIL U11 default bench hash '$hash' != $REF ($how)"
    fail=1
fi

# U12: projection identity. The 8-fish build at num_agents=4 must reproduce the 4-fish build's
# rewards hash and the 110-slot projection of every obs row (bench project=4), same flags.
# shellcheck disable=SC2086
[ "$SUFFIX" = "" ] || compile build/wef_test -O1 -g -mavx2 -mfma $BASE_CFLAGS
# shellcheck disable=SC2086
[ "$SUFFIX" = 8 ] || compile build/wef_test8 -O1 -g -mavx2 -mfma $BASE_CFLAGS -DMAX_AGENTS=8
out4=$(build/wef_test bench steps=20000 project=4) || { echo "FAIL U12: 4-fish bench project=4 failed"; echo "$out4"; fail=1; }
out8=$(build/wef_test8 bench steps=20000 num_agents=4 project=4) || { echo "FAIL U12: 8-fish bench project=4 failed"; echo "$out8"; fail=1; }
line4=$(printf '%s\n' "$out4" | sed -n 's/^  proj_hash \([0-9a-f]*\) .* rew_hash \([0-9a-f]*\)$/\1 \2/p')
line8=$(printf '%s\n' "$out8" | sed -n 's/^  proj_hash \([0-9a-f]*\) .* rew_hash \([0-9a-f]*\)$/\1 \2/p')
if [ -n "$line4" ] && [ "$line4" = "$line8" ]; then
    echo "ok   U12 projection identity: 8-fish build at num_agents=4 reproduces the 4-fish proj_hash / rew_hash ($line4) over 20K steps"
else
    echo "FAIL U12 projection identity: 4-fish '$line4' vs 8-fish at num_agents=4 '$line8'"
    fail=1
fi
hash8=$(printf '%s\n' "$out8" | sed -n 's/^  hash \([0-9a-f]*\)$/\1/p')
echo "     (8-fish build, num_agents=4, full 162-slot hash $hash8; 4-fish harness hash $(printf '%s\n' "$out4" | sed -n 's/^  hash \([0-9a-f]*\)$/\1/p'))"
if [ "$META" = 1 ]; then
    # U13m: the options build with every fish scripted (roles != 0 never take options) is the institution build:
    # same obs rows and rewards bit for bit (hash) and the same printed metrics, on the single Commons with the rule
    # on and marks, at the centre broadcast beacon and at a 5 cm beacon placed with inst_pos_x_cm / inst_pos_y_cm.
    n=${NFISH:-4}
    plain_i=build/wef_test${n}i
    # shellcheck disable=SC2086
    compile "$plain_i" -O1 -g -mavx2 -mfma $BASE_CFLAGS -DMAX_AGENTS="$n" -DWEF_INST
    roles=13,14,15,2; [ "$n" = 8 ] && roles=13,14,15,2,13,14,15,2
    for beacon in "inst_read_cm=36" "inst_read_cm=5 inst_pos_x_cm=5 inst_pos_y_cm=5"; do
        # shellcheck disable=SC2086
        set -- preset=inst_commons_single num_agents="$n" roles="$roles" inst_mode=1 inst_mark_steps=64 mark_bitten_reward=-5 episodes=2 hash=1 $beacon
        outi=$("$plain_i" "$@" 2>&1 | grep -v 'meta\|strategy') || true
        outm=$("$OUT" "$@" 2>&1 | grep -v 'meta\|strategy') || true
        hi=$(printf '%s\n' "$outi" | sed -n 's/^  hash \([0-9a-f]*\) .*/\1/p')
        if [ -n "$hi" ] && [ "$outi" = "$outm" ]; then
            echo "ok   U13m options build, all fish scripted ($beacon): bit-identical to the institution build (hash $hi, every metric line equal, 2 episodes)"
        else
            echo "FAIL U13m options build vs institution build ($beacon):"
            diff <(printf '%s\n' "$outi") <(printf '%s\n' "$outm") | head -20
            fail=1
        fi
    done
elif [ "$INST" = 1 ]; then
    # U13: the institution build at inst_obs 0 is the plain build plus zero-filled slots.
    n=${NFISH:-4}
    plain=build/wef_test; [ "$n" = 4 ] || plain=build/wef_test$n
    outp=$("$plain" bench steps=5000 preset=allelo num_agents="$n" project="$n" 2>&1) || true
    outi=$("$OUT" bench steps=5000 preset=allelo num_agents="$n" project="$n" 2>&1) || true
    if [ "$n" = 4 ]; then  # the 4-fish builds cannot run the 8-fish AH preset: use the default bench
        outp=$("$plain" bench steps=5000 project=4 2>&1) || true
        outi=$("$OUT" bench steps=5000 project=4 2>&1) || true
    fi
    lp=$(printf '%s\n' "$outp" | sed -n 's/^  proj_hash \([0-9a-f]*\) .* rew_hash \([0-9a-f]*\)$/\1 \2/p')
    li=$(printf '%s\n' "$outi" | sed -n 's/^  proj_hash \([0-9a-f]*\) .* rew_hash \([0-9a-f]*\)$/\1 \2/p')
    if [ -n "$lp" ] && [ "$lp" = "$li" ]; then
        echo "ok   U13 institution build at inst_obs 0: base-slot projection / rewards hash equal the plain $n-fish build's ($lp, 5000 steps)"
    else
        echo "FAIL U13 institution build vs plain $n-fish build: '$li' vs '$lp'"
        fail=1
    fi
fi
if [ "$fail" = 0 ]; then
    if [ "$META" = 1 ]; then
        echo "run_test.sh PASSED (unit suite, U11, U12, U13m)"
    else
        echo "run_test.sh PASSED (unit suite, U11, U12${INST:+, U13})"
    fi
else
    echo "run_test.sh FAILED"
    exit 1
fi
