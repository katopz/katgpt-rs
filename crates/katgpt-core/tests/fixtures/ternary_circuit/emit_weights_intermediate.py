#!/usr/bin/env python3
"""Issue 1225 T1: emit the lossless weights.npz intermediate for the Rust fixture builder.

The npz members are DEFLATED-compressed npy v1 arrays; the Rust side carries no
zip/inflate dependency, so this stdlib-only script decompresses once and writes
the raw members verbatim:

  intermediate/trits.bin  -- the raw packed-trit bytes (5 trits/byte, base-3 digits)
  intermediate/vecs.f32   -- the raw little-endian f32 vector buffer (per-channel scales)
  intermediate/meta.txt   -- the meta string verbatim (UTF-32-LE decoded)

Shape/layout law: every matrix's shape AND byte length are taken from meta.txt at
assembly time -- nothing here interprets, transposes, or re-orders anything. The
emitter is convention-free by design; the axis conventions belong to the forward
port (Issue 1225 Arm A).

Lossless proof obligation (checked before writing): the meta grammar's trit-part
packed lengths must sum EXACTLY to the trits member byte length, and the vec-part
counts must sum EXACTLY to the vecs member element count. A mismatch refuses the
emission (the intermediate would silently mis-slice at assembly).

Regenerate: python3 emit_weights_intermediate.py <weights.npz> <out-dir>
Provenance: brianbell-x/codehygiene @ 423c7e1f02ef409bbaea209268bfcde36e0f1cb0
(MIT declared at pyproject.toml:10 only; attribution recorded in Issue 1225).
"""
import ast
import json
import math
import os
import struct
import sys
import zipfile


def npy_header(raw):
    if raw[:6] != b"\x93NUMPY":
        raise SystemExit("not an npy member")
    major = raw[6]
    if major == 1:
        hlen = struct.unpack("<H", raw[8:10])[0]
        off = 10
    elif major == 2:
        hlen = struct.unpack("<I", raw[10:14])[0]
        off = 14
    else:
        raise SystemExit("npy major version %d unsupported" % major)
    header = ast.literal_eval(raw[off : off + hlen].decode("latin1").strip())
    return header, off + hlen


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: emit_weights_intermediate.py <weights.npz> <out-dir>")
    npz_path, out_dir = sys.argv[1], sys.argv[2]

    z = zipfile.ZipFile(npz_path)
    members = {}
    for name in ("trits.npy", "vecs.npy", "meta.npy"):
        if name not in z.namelist():
            raise SystemExit("missing npz member: %s" % name)
        raw = z.read(name)
        header, off = npy_header(raw)
        members[name] = (header, raw[off:])

    trits_header, trits = members["trits.npy"]
    vecs_header, vecs = members["vecs.npy"]
    meta_header, meta_bytes = members["meta.npy"]

    if trits_header["descr"] != "|u1" or trits_header["fortran_order"]:
        raise SystemExit("trits.npy: expected C-order u1")
    if vecs_header["descr"] != "<f4" or vecs_header["fortran_order"]:
        raise SystemExit("vecs.npy: expected C-order <f4")
    if not meta_header["descr"].startswith("<U"):
        raise SystemExit("meta.npy: expected a <U unicode scalar, got %r" % meta_header["descr"])
    if tuple(meta_header["shape"]) != ():
        raise SystemExit("meta.npy: expected a scalar (0-d) array")
    if len(meta_bytes) % 4 != 0:
        raise SystemExit("meta.npy: UCS4 payload not 4-aligned")
    meta = meta_bytes.decode("utf-32-le")

    trit_part, vec_part = meta.split("|")

    packed_total = 0
    for entry in trit_part.split(" ; "):
        name, dims, packed_len_str = entry.split(":")
        shape = tuple(int(d) for d in dims.split(","))
        packed_len = int(packed_len_str)
        elems = math.prod(shape)
        # 5 trits/byte, base-3 digits; the tail byte is zero-padded.
        need = (elems + 4) // 5
        if packed_len != need:
            raise SystemExit(
                "meta %s: packed_len %d != ceil(elems/5) %d" % (name, packed_len, need)
            )
        packed_total += packed_len
    if packed_total != len(trits):
        raise SystemExit(
            "trit-part packed lengths sum to %d, trits member has %d bytes"
            % (packed_total, len(trits))
        )

    vec_total = 0
    for entry in vec_part.split(" ; "):
        name, dims, count_str = entry.split(":")
        shape = tuple(int(d) for d in dims.split(","))
        count = int(count_str)
        if math.prod(shape) != count:
            raise SystemExit("meta %s: count %d != prod(shape)" % (name, count))
        vec_total += count
    if vec_total * 4 != len(vecs):
        raise SystemExit(
            "vec-part counts sum to %d elements, vecs member has %d bytes (%d f32)"
            % (vec_total, len(vecs), len(vecs) // 4)
        )

    os.makedirs(os.path.join(out_dir, "intermediate"), exist_ok=True)
    with open(os.path.join(out_dir, "intermediate", "trits.bin"), "wb") as f:
        f.write(trits)
    with open(os.path.join(out_dir, "intermediate", "vecs.f32"), "wb") as f:
        f.write(vecs)
    with open(os.path.join(out_dir, "intermediate", "meta.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write(meta)
        f.write("\n")

    # The verbatim source artifact rides beside the intermediate so the Rust
    # assembly can pin its BLAKE3 without re-fetching the clone.
    with open(npz_path, "rb") as f:
        npz_bytes = f.read()
    with open(os.path.join(out_dir, "weights.npz"), "wb") as f:
        f.write(npz_bytes)

    census = {
        "source_pin": "423c7e1f02ef409bbaea209268bfcde36e0f1cb0",
        "trit_bytes": len(trits),
        "trits_total": packed_total * 5,
        "vec_f32": vec_total,
        "trit_layers": len(trit_part.split(" ; ")),
        "vec_entries": len(vec_part.split(" ; ")),
    }
    with open(
        os.path.join(out_dir, "intermediate", "census.json"), "w", encoding="utf-8", newline="\n"
    ) as f:
        json.dump(census, f, indent=1, sort_keys=True)
        f.write("\n")
    print(
        "emitted: %s (%d trit bytes, %d f32 vecs, %d trit layers)"
        % (out_dir, len(trits), vec_total, len(trit_part.split(" ; ")))
    )


if __name__ == "__main__":
    main()
