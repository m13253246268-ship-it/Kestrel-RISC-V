#!/usr/bin/env python3
"""Compare two VQF v4 files: header + tensor table."""
import struct, sys

HDR_BYTES = 448
VQF_ALIGN = 64
VQF_HEADER_BYTES = 4096

def parse_vqf(path):
    with open(path, "rb") as f:
        data = f.read()
    magic, version, flags, n_tensors = struct.unpack_from("<IIII", data, 0)
    arch_raw = data[16:196]
    data_off, file_len = struct.unpack_from("<QQ", data, 200)
    # parse arch (VQFArchDet, 92 bytes)
    arch_fields = struct.unpack_from("<12I", arch_raw, 0)
    task, family, input_w, input_h, nc = arch_fields[0:5]
    strides = arch_fields[5:8]
    head, reg_max, n_ops, n_anchors = arch_fields[8:12]
    conf, iou = struct.unpack_from("<ff", arch_raw, 48)
    graph_bytes = struct.unpack_from("<I", arch_raw, 56)[0]
    # parse tensor directory
    dir_off = HDR_BYTES  # 448, aligned to 64
    tensors = []
    for i in range(n_tensors):
        base = dir_off + i * 64
        name_raw = data[base:base+32]
        name = name_raw.split(b'\x00')[0].decode('utf-8', errors='replace')
        qt, rows, cols, layout = struct.unpack_from("<IIII", data, base+32)
        off, nbytes = struct.unpack_from("<QQ", data, base+40)
        tensors.append((name, qt, rows, cols, layout, off, nbytes))
    return {
        'magic': magic, 'version': version, 'flags': flags, 'n_tensors': n_tensors,
        'data_off': data_off, 'file_len': file_len,
        'task': task, 'family': family, 'input_w': input_w, 'input_h': input_h,
        'nc': nc, 'strides': strides, 'head': head, 'reg_max': reg_max,
        'n_ops': n_ops, 'n_anchors': n_anchors, 'conf': conf, 'iou': iou,
        'graph_bytes': graph_bytes,
        'tensors': tensors,
        'raw': data,
    }

def main():
    ref_path = sys.argv[1] if len(sys.argv) > 1 else "yolov8n_q8_320.vqf"
    test_path = sys.argv[2] if len(sys.argv) > 2 else "test_out.vqf"
    ref = parse_vqf(ref_path)
    test = parse_vqf(test_path)

    print(f"=== Header Comparison ===")
    print(f"  magic:      ref=0x{ref['magic']:08X} test=0x{test['magic']:08X} {'OK' if ref['magic']==test['magic'] else 'DIFF'}")
    print(f"  version:    ref={ref['version']} test={test['version']} {'OK' if ref['version']==test['version'] else 'DIFF'}")
    print(f"  flags:      ref=0x{ref['flags']:08X} test=0x{test['flags']:08X} {'OK' if ref['flags']==test['flags'] else 'DIFF'}")
    print(f"  n_tensors:  ref={ref['n_tensors']} test={test['n_tensors']} {'OK' if ref['n_tensors']==test['n_tensors'] else 'DIFF'}")
    print(f"  data_off:   ref={ref['data_off']} test={test['data_off']} {'OK' if ref['data_off']==test['data_off'] else 'DIFF'}")
    print(f"  file_len:   ref={ref['file_len']} test={test['file_len']} {'OK' if ref['file_len']==test['file_len'] else 'DIFF'}")
    print(f"  task:       ref={ref['task']} test={test['task']}")
    print(f"  family:     ref={ref['family']} test={test['family']} {'OK' if ref['family']==test['family'] else 'DIFF'}")
    print(f"  input_w:    ref={ref['input_w']} test={test['input_w']}")
    print(f"  input_h:    ref={ref['input_h']} test={test['input_h']}")
    print(f"  nc:         ref={ref['nc']} test={test['nc']}")
    print(f"  strides:    ref={ref['strides']} test={test['strides']}")
    print(f"  head:       ref={ref['head']} test={test['head']}")
    print(f"  reg_max:    ref={ref['reg_max']} test={test['reg_max']}")
    print(f"  n_ops:      ref={ref['n_ops']} test={test['n_ops']} {'OK' if ref['n_ops']==test['n_ops'] else 'DIFF'}")
    print(f"  n_anchors:  ref={ref['n_anchors']} test={test['n_anchors']}")
    print(f"  conf:       ref={ref['conf']} test={test['conf']}")
    print(f"  iou:        ref={ref['iou']} test={test['iou']}")
    print(f"  graph_bytes:ref={ref['graph_bytes']} test={test['graph_bytes']} {'OK' if ref['graph_bytes']==test['graph_bytes'] else 'DIFF'}")

    print(f"\n=== File sizes ===")
    print(f"  ref: {len(ref['raw'])} bytes")
    print(f"  test: {len(test['raw'])} bytes")
    print(f"  diff: {len(test['raw']) - len(ref['raw'])} bytes")

    print(f"\n=== Tensor Table Comparison ===")
    rt = ref['tensors']
    tt = test['tensors']
    maxn = max(len(rt), len(tt))
    total_ref_bytes = 0
    total_test_bytes = 0
    diffs = []
    for i in range(maxn):
        if i < len(rt):
            total_ref_bytes += rt[i][6]
        if i < len(tt):
            total_test_bytes += tt[i][6]
        if i >= len(rt):
            print(f"  [{i}] EXTRA in test: name={tt[i][0]} qt={tt[i][1]} rows={tt[i][2]} cols={tt[i][3]} layout={tt[i][4]} bytes={tt[i][6]}")
            diffs.append(i)
            continue
        if i >= len(tt):
            print(f"  [{i}] MISSING in test: name={rt[i][0]} qt={rt[i][1]} rows={rt[i][2]} cols={rt[i][3]} layout={rt[i][4]} bytes={rt[i][6]}")
            diffs.append(i)
            continue
        r = rt[i]
        t = tt[i]
        field_diffs = []
        if r[0] != t[0]: field_diffs.append(f"name: ref={r[0]} test={t[0]}")
        if r[1] != t[1]: field_diffs.append(f"qt: ref={r[1]} test={t[1]}")
        if r[2] != t[2]: field_diffs.append(f"rows: ref={r[2]} test={t[2]}")
        if r[3] != t[3]: field_diffs.append(f"cols: ref={r[3]} test={t[3]}")
        if r[4] != t[4]: field_diffs.append(f"layout: ref={r[4]} test={t[4]}")
        if r[5] != t[5]: field_diffs.append(f"offset: ref={r[5]} test={t[5]}")
        if r[6] != t[6]: field_diffs.append(f"bytes: ref={r[6]} test={t[6]}")
        if field_diffs:
            print(f"  [{i}] {' '.join(field_diffs)}")
            diffs.append(i)

    print(f"\n=== Summary ===")
    print(f"  ref tensors:  {len(rt)}")
    print(f"  test tensors: {len(tt)}")
    print(f"  ref total tensor bytes:  {total_ref_bytes}")
    print(f"  test total tensor bytes: {total_test_bytes}")
    print(f"  byte diff: {total_test_bytes - total_ref_bytes}")
    print(f"  diffs: {len(diffs)} tensors")

    # Show first 20 tensor names from each
    print(f"\n=== First 20 ref tensors ===")
    for i, t in enumerate(rt[:20]):
        print(f"  [{i}] name={t[0]:30s} qt={t[1]} rows={t[2]:6d} cols={t[3]:6d} layout={t[4]} off={t[5]:8d} bytes={t[6]:8d}")
    print(f"\n=== First 20 test tensors ===")
    for i, t in enumerate(tt[:20]):
        print(f"  [{i}] name={t[0]:30s} qt={t[1]} rows={t[2]:6d} cols={t[3]:6d} layout={t[4]} off={t[5]:8d} bytes={t[6]:8d}")

    # Show last 20
    print(f"\n=== Last 20 ref tensors ===")
    for i in range(max(0, len(rt)-20), len(rt)):
        t = rt[i]
        print(f"  [{i}] name={t[0]:30s} qt={t[1]} rows={t[2]:6d} cols={t[3]:6d} layout={t[4]} off={t[5]:8d} bytes={t[6]:8d}")
    print(f"\n=== Last 20 test tensors ===")
    for i in range(max(0, len(tt)-20), len(tt)):
        t = tt[i]
        print(f"  [{i}] name={t[0]:30s} qt={t[1]} rows={t[2]:6d} cols={t[3]:6d} layout={t[4]} off={t[5]:8d} bytes={t[6]:8d}")

if __name__ == "__main__":
    main()
