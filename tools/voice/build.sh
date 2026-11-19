#!/bin/sh
set -eu
if [ "$#" -ne 3 ]; then
    echo "usage: sh build.sh C_API_INCLUDE_DIR SHERPA_LIB_DIR OUTPUT_SO" >&2
    exit 2
fi
src=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
"${CC:-gcc}" -std=c11 -O2 -Wall -Wextra -Werror -pedantic -fPIC -shared \
    "$src/kws_bridge.c" -I"$1" -L"$2" -lsherpa-onnx-c-api \
    -Wl,-rpath,"$2" -o "$3"
