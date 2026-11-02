#!/bin/bash
set -u
dpkg -L llama.cpp-tools-spacemit | grep -E 'bin/' | sed 's#.*/##' | tr '\n' ' '; echo
which llama-quantize llama-imatrix 2>/dev/null || echo "(no llama-quantize)"
echo "=== ort providers ==="
python3 - <<'PY'
try:
    import onnxruntime as ort
    print("ort", ort.__version__, ort.get_available_providers())
except Exception as e:
    print("ort import fail:", e)
PY
echo Q3_CHK_END
