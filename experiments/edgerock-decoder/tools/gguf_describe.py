#!/usr/bin/env python3
"""Describe a GGUF from a range-fetched prefix: metadata AND the tensor list.

The tensor names and shapes are the architecture's ground truth. The model card and the paper
describe intent; only the tensors say what a decoder has to implement. Reading them first is the
rule that caught five blockers on an earlier model before any fleet spend, and it is the reason
this file exists before any Java does.
"""
import struct, sys, json

U8, I8, U16, I16, U32, I32, F32, BOOL, STR, ARR, U64, I64, F64 = range(13)
FIXED = {U8: ("<B", 1), I8: ("<b", 1), U16: ("<H", 2), I16: ("<h", 2), U32: ("<I", 4),
         I32: ("<i", 4), F32: ("<f", 4), BOOL: ("<B", 1), U64: ("<Q", 8), I64: ("<q", 8),
         F64: ("<d", 8)}
# GGML type ids we care about naming; anything else prints as its number.
GGML = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
        10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 30: "BF16"}


def read_str(d, o):
    (n,) = struct.unpack_from("<Q", d, o)
    o += 8
    return d[o:o + n].decode("utf-8", "replace"), o + n


def read_value(d, o, t):
    if t in FIXED:
        fmt, size = FIXED[t]
        (v,) = struct.unpack_from(fmt, d, o)
        return v, o + size
    if t == STR:
        return read_str(d, o)
    if t == ARR:
        (et,) = struct.unpack_from("<I", d, o); o += 4
        (n,) = struct.unpack_from("<Q", d, o); o += 8
        out = []
        for _ in range(n):
            v, o = read_value(d, o, et)
            out.append(v)
        return out, o
    raise ValueError(f"unknown GGUF value type {t} at offset {o}")


def describe(path):
    d = open(path, "rb").read()
    assert d[:4] == b"GGUF", "not a GGUF file"
    o = 4
    (version,) = struct.unpack_from("<I", d, o); o += 4
    (n_tensors,) = struct.unpack_from("<Q", d, o); o += 8
    (n_kv,) = struct.unpack_from("<Q", d, o); o += 8
    kv = {}
    for _ in range(n_kv):
        key, o = read_str(d, o)
        (t,) = struct.unpack_from("<I", d, o); o += 4
        try:
            kv[key], o = read_value(d, o, t)
        except (struct.error, IndexError):
            raise SystemExit(f"prefix too short inside key {key!r}; fetch more bytes")
    tensors = []
    for _ in range(n_tensors):
        try:
            name, o = read_str(d, o)
            (ndim,) = struct.unpack_from("<I", d, o); o += 4
            dims = list(struct.unpack_from("<" + "Q" * ndim, d, o)); o += 8 * ndim
            (tt,) = struct.unpack_from("<I", d, o); o += 4
            (off,) = struct.unpack_from("<Q", d, o); o += 8
        except (struct.error, IndexError):
            raise SystemExit(f"prefix too short after {len(tensors)} of {n_tensors} tensors; "
                             "fetch more bytes")
        tensors.append({"name": name, "dims": dims, "type": GGML.get(tt, str(tt)), "offset": off})
    return version, n_tensors, kv, tensors


if __name__ == "__main__":
    version, n_tensors, kv, tensors = describe(sys.argv[1])
    arch = kv.get("general.architecture")
    print(f"gguf v{version}  arch={arch}  tensors={n_tensors}  kv={len(kv)}")
    if "--json" in sys.argv:
        json.dump({"arch": arch, "kv": {k: v for k, v in kv.items()
                                        if not isinstance(v, list) or len(v) <= 64},
                   "tensors": tensors}, sys.stdout, indent=1)
