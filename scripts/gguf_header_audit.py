#!/usr/bin/env python3
"""GGUF header audit — metadata KV + per-tensor ggml-type dump (stdlib-only).

Why this exists (Issue 730 T0 + riir-ai Issue 879 T2): both need GGUF
INTROSPECTION, not inference — 730 counts full-attention vs DeltaNet layers
to recompute the 256K prefill KV wall; 879 audits quant classes of the GDN
gate parameters (A_log/dt_bias/conv1d/norms) and gate projections.

Usage:
  python scripts/gguf_header_audit.py <model.gguf> [--tensors] [--filter SUB]

  --tensors   also dump the tensor-info table (name, dims, ggml type)
  --filter    substring filter on tensor names (with --tensors)

No data section is read — header + metadata + tensor infos only.
"""

import argparse
import struct
import sys
from pathlib import Path

# Windows consoles default to a legacy codec (cp874/cp1252 here); the house
# pattern (katgpt-rs numbering-sweep lesson) degrades non-encodable glyphs
# instead of dying mid-report.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="backslashreplace")
    sys.stderr.reconfigure(errors="backslashreplace")

GGUF_MAGIC = 0x46554747  # "GGUF" little-endian

# GGUF metadata value types
T_U8, T_I8, T_U16, T_I16, T_U32, T_I32, T_F32 = range(7)
T_BOOL, T_STRING, T_ARRAY, T_U64, T_I64, T_F64 = range(7, 13)

# ggml tensor types (llama.cpp ggml.h enum; BF16=30 / IQ1_M=29 per the
# katgpt-rs Issue-717 loader-hardening record)
GGML_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
    8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K",
    13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS",
    18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S",
    23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64",
    29: "IQ1_M", 30: "BF16",
}

SCALAR_FMT = {T_U8: "<B", T_I8: "<b", T_U16: "<H", T_I16: "<h",
              T_U32: "<I", T_I32: "<i", T_F32: "<f", T_BOOL: "<B",
              T_U64: "<Q", T_I64: "<q", T_F64: "<d"}


class Reader:
    def __init__(self, path: Path):
        self.f = open(path, "rb")

    def unpack(self, fmt: str):
        size = struct.calcsize(fmt)
        return struct.unpack(fmt, self.f.read(size))[0]

    def string(self) -> str:
        n = self.unpack("<Q")
        return self.f.read(n).decode("utf-8", errors="replace")

    def value(self, vtype: int):
        if vtype == T_STRING:
            return self.string()
        if vtype == T_ARRAY:
            etype = self.unpack("<I")
            count = self.unpack("<Q")
            return [self.value(etype) for _ in range(count)]
        return self.unpack(SCALAR_FMT[vtype])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf", type=Path)
    ap.add_argument("--tensors", action="store_true")
    ap.add_argument("--filter", default=None)
    args = ap.parse_args()

    r = Reader(args.gguf)
    magic = r.unpack("<I")
    if magic != GGUF_MAGIC:
        print(f"not a GGUF file: magic=0x{magic:08x}", file=sys.stderr)
        return 1
    version = r.unpack("<I")
    tensor_count = r.unpack("<Q")
    kv_count = r.unpack("<Q")
    print(f"file: {args.gguf}")
    print(f"gguf version: {version}  tensors: {tensor_count}  kv pairs: {kv_count}")

    for _ in range(kv_count):
        key = r.string()
        vtype = r.unpack("<I")
        val = r.value(vtype)
        if isinstance(val, list) and len(val) > 24:
            shown = ", ".join(repr(v) for v in val[:8])
            print(f"  {key} [{vtype}] len={len(val)} = [{shown}, ...]")
        else:
            print(f"  {key} [{vtype}] = {val!r}")

    if not args.tensors:
        return 0

    print(f"--- tensor infos ({tensor_count}) ---")
    type_counts: dict[str, int] = {}
    for _ in range(tensor_count):
        name = r.string()
        n_dims = r.unpack("<I")
        shape = [r.unpack("<Q") for _ in range(n_dims)]
        ggml_type = r.unpack("<I")
        r.unpack("<Q")  # data offset (relative to alignment) — not needed
        if args.filter and args.filter not in name:
            continue
        tname = GGML_TYPES.get(ggml_type, f"TYPE_{ggml_type}")
        type_counts[tname] = type_counts.get(tname, 0) + 1
        dims = "x".join(str(s) for s in shape)
        print(f"  {name:48s} {tname:8s} [{dims}]")
    if not args.filter:
        print("--- tensor type census ---")
        for tname, n in sorted(type_counts.items(), key=lambda kv: -kv[1]):
            print(f"  {tname:10s} {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
