#!/bin/bash
# Run pt_to_vqf.py on the K1 board to convert yolov8n.pt -> yolov8n_q8_320.vqf
# Usage (host-side): bash _k1_run.sh _k1_yolo_convert.sh
# Or direct SSH:     bash _k1_yolo_convert.sh
set -eu

DIR=/root/kv/kestrel_vllm_riscv/tools/yolo
PT=$DIR/yolov8n.pt
OUT=$DIR/yolov8n_q8_320.vqf
SCRIPT=$DIR/pt_to_vqf.py

cd "$DIR"

# --- sanity checks ---
if [ ! -f "$PT" ]; then
    echo "ERROR: $PT not found. Run _k1_yolo_put_pt.sh first." >&2
    exit 1
fi
if [ ! -f "$SCRIPT" ]; then
    echo "ERROR: $SCRIPT not found. Run _k1_upload.sh first." >&2
    exit 1
fi

PT_BYTES=$(stat -c %s "$PT")
echo "CONVERT_START pt=$PT ($PT_BYTES bytes)"

# --- convert ---
python3 "$SCRIPT" \
    --pt "$PT" \
    --out "$OUT" \
    --q8 \
    --imgsz 320x320

# --- report ---
if [ ! -f "$OUT" ]; then
    echo "ERROR: conversion produced no output file." >&2
    exit 1
fi
OUT_BYTES=$(stat -c %s "$OUT")
# FNV-1a 64-bit via python3 (matches write_vqf's fsum)
FNV=$(python3 - "$OUT" <<'PYEOF'
import sys, struct
def fnv1a64(data):
    h = 0xcbf29ce484222325
    for b in data:
        h = ((h ^ b) * 0x100000001b3) & 0xffffffffffffffff
    return h
out = sys.argv[1]
with open(out, "rb") as f:
    data = f.read()
print(f"{fnv1a64(data):016x}")
PYEOF
)
echo "CONVERT_OK out=$OUT bytes=$OUT_BYTES fnv=$FNV"
