# -*- coding: utf-8 -*-
"""Build engine vocab.bin from a HuggingFace ByteLevel BPE tokenizer.

Usage: python tools/build/build_vocab_bin.py tokenizer.json [old-vocab.bin] out.bin
The optional legacy input is accepted for command-line compatibility; added
vocabulary is read from tokenizer.json, without model-specific ID thresholds.
"""
import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gen_merges_bin import to_engine


def main():
    if len(sys.argv) not in (3, 4):
        raise SystemExit(__doc__)
    data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    decoder = data.get("decoder") or {}
    if data["model"]["type"] != "BPE" or decoder.get("type") != "ByteLevel":
        raise SystemExit("Only ByteLevel BPE vocabularies are supported")
    added = {int(a["id"]): a["content"].encode("utf-8")
             for a in data.get("added_tokens", [])}
    entries = {int(tid): added.get(int(tid), None) or to_engine(text)
               for text, tid in data["model"]["vocab"].items()}
    entries.update(added)
    if sorted(entries) != list(range(len(entries))):
        raise SystemExit("Vocabulary IDs must be contiguous")
    with open(sys.argv[-1], "wb") as out:
        out.write(struct.pack("<I", len(entries)))
        for tid, text in sorted(entries.items()):
            out.write(struct.pack("<IH", tid, len(text)))
            out.write(text)
    print(f"wrote {sys.argv[-1]}: {len(entries)} tokens")


if __name__ == "__main__":
    main()
