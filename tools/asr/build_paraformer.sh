#!/bin/sh
set -eu
# CC/AR may select a cross toolchain. Extra arguments are compiler flags.
# Usage: build_paraformer.sh OUTPUT [cli|static] [FLAGS...]
# OPENMP=1 enables OpenMP; static consumers link -lm and optionally -fopenmp.
: "${1:?usage: build_paraformer.sh OUTPUT [cli|static] [FLAGS...]}"
output=$1
shift
target=${1:-cli}
if [ "$#" -gt 0 ]; then shift; fi
case "$target" in cli|static) ;; *) echo "Invalid target: $target" >&2; exit 2 ;; esac
root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
cc=${CC:-cc}
set -- -O2 -std=c11 -Wall -Wextra "$@"
if [ "${OPENMP:-0}" = 1 ]; then set -- "$@" -fopenmp; fi
set -- "$@" -fno-fast-math -ffp-contract=off
machine=$("$cc" -dumpmachine)
echo "[paraformer] target=$target machine=$machine output=$output"
if [ "$target" = cli ]; then
    set -x
    "$cc" "$@" "$root/src/model/vllm_paraformer.c" -o "$output" -lm
else
    ar=${AR:-$("$cc" -print-prog-name=ar)}
    work=$(mktemp -d)
    trap 'rm -rf "$work"' EXIT
    trap 'exit 1' HUP INT TERM
    set -x
    "$cc" "$@" -DVLLM_PARAFORMER_NO_MAIN -c "$root/src/model/vllm_paraformer.c" -o "$work/paraformer.o"
    "$ar" rcs "$work/libparaformer.a" "$work/paraformer.o"
    mv -f "$work/libparaformer.a" "$output"
fi
