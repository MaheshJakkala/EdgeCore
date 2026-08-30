"""EdgeCore - model export and .ecm (EdgeCore Model) format.

The .ecm file layout:
  ECM1
  key=value ...  (meta section, until END_META)
  <name> <dtype> <dims...> ...  (tensor table, until END_TENSORS)
  <binary blobs, one per tensor, in table order>

dtypes: fp32 (4B), fp16 (2B), int8 (1B), text (bytes; dims = [nbytes])
"""
from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

MAGIC = "ECM1\n"


def _f32_to_fp16(a: np.ndarray) -> np.ndarray:
    return a.astype(np.float16)


def quantize_int8_symmetric(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-row symmetric int8 quantization. a: [rows, cols] fp32.
    Returns (qint8 [rows, cols], scales fp32 [rows])."""
    a = a.astype(np.float64)
    maxabs = np.max(np.abs(a), axis=1)
    scales = maxabs / 127.0
    scales = np.where(scales == 0, 1.0, scales)
    q = np.rint(a / scales[:, None]).astype(np.int32)
    q = np.clip(q, -128, 127).astype(np.int8)
    return q, scales.astype(np.float32)


class EcmWriter:
    def __init__(self) -> None:
        self.meta: dict[str, Any] = {}
        self.tensors: list[tuple[str, str, np.ndarray]] = []  # (name, dtype, arr)

    def set_meta(self, **kw: Any) -> None:
        self.meta.update(kw)

    def add(self, name: str, dtype: str, arr: np.ndarray) -> None:
        self.tensors.append((name, dtype, np.ascontiguousarray(arr)))

    def write(self, fh: BinaryIO) -> None:
        fh.write(MAGIC.encode())
        for k, v in self.meta.items():
            fh.write(f"{k}={v}\n".encode())
        fh.write(b"END_META\n")
        for name, dtype, arr in self.tensors:
            if dtype == "text":
                fh.write(f"{name} {dtype} {arr.nbytes}\n".encode())
            else:
                dims = " ".join(str(d) for d in arr.shape)
                fh.write(f"{name} {dtype} {dims}\n".encode())
        fh.write(b"END_TENSORS\n")
        for _, dtype, arr in self.tensors:
            if dtype == "text":
                fh.write(arr.tobytes())
            elif dtype == "fp16":
                fh.write(_f32_to_fp16(arr).tobytes())
            elif dtype == "fp32":
                fh.write(arr.astype(np.float32).tobytes())
            elif dtype == "int8":
                fh.write(arr.astype(np.int8).tobytes())
            else:
                raise ValueError(f"unknown dtype {dtype}")


def _load_safetensors(path: Path) -> dict[str, np.ndarray]:
    raw = path.read_bytes()
    if len(raw) < 8:
        raise ValueError(
            f"{path}: file is only {len(raw)} bytes - corrupt or empty download; "
            f"re-download with `python3 scripts/fetch_model.py --force`")
    (hlen,) = struct.unpack("<Q", raw[:8])
    if 8 + hlen > len(raw):
        raise ValueError(
            f"{path}: header claims {hlen} bytes but file is {len(raw)} bytes - "
            f"truncated download; re-download with "
            f"`python3 scripts/fetch_model.py --force`")
    header = json.loads(raw[8 : 8 + hlen])
    header.pop("__metadata__", None)
    data_base = 8 + hlen  # safetensors data_offsets are relative to the data section
    tensors: dict[str, np.ndarray] = {}
    dtmap = {"F32": np.float32, "F16": np.float16, "I64": np.int64}
    for name, info in header.items():
        begin, end = info["data_offsets"]
        buf = raw[data_base + begin : data_base + end]
        need = np.prod(info["shape"], dtype=int) * np.dtype(dtmap[info["dtype"]]).itemsize
        if len(buf) < need:
            raise ValueError(
                f"{path}: tensor '{name}' needs {need} bytes but only {len(buf)} "
                f"remain - truncated download; re-download with "
                f"`python3 scripts/fetch_model.py --force`")
        arr = np.frombuffer(buf, dtype=dtmap[info["dtype"]]).reshape(info["shape"])
        tensors[name] = arr.astype(np.float32) if info["dtype"] == "F32" else arr
    return tensors


def export_from_safetensors(src: Path, out: Path) -> dict[str, Any]:
    """Convert an HF GPT-2 directory (safetensors + config + vocab) to .ecm."""
    cfg = json.loads((src / "config.json").read_text())
    n_layer, n_head, n_embd = cfg["n_layer"], cfg["n_head"], cfg["n_embd"]
    n_ctx = cfg.get("n_positions", cfg.get("n_ctx", 1024))
    vocab = cfg["vocab_size"]
    model_name = cfg.get("_name_or_path", "gpt2")

    tensors = _load_safetensors(src / "model.safetensors")
    wr = EcmWriter()
    wr.set_meta(
        model_name=model_name, n_layer=n_layer, n_head=n_head, n_embd=n_embd,
        n_ctx=n_ctx, vocab=vocab, eps=1e-5,
    )

    # embeddings in fp16
    wr.add("wte", "fp16", tensors["wte.weight"])
    wr.add("wpe", "fp16", tensors["wpe.weight"])

    # final layernorm
    wr.add("ln_f.g", "fp32", tensors["ln_f.weight"])
    wr.add("ln_f.b", "fp32", tensors["ln_f.bias"])

    n_params = 0
    for i in range(n_layer):
        p = f"h.{i}"
        wr.add(f"blocks.{i}.ln_1.g", "fp32", tensors[f"{p}.ln_1.weight"])
        wr.add(f"blocks.{i}.ln_1.b", "fp32", tensors[f"{p}.ln_1.bias"])
        wr.add(f"blocks.{i}.ln_2.g", "fp32", tensors[f"{p}.ln_2.weight"])
        wr.add(f"blocks.{i}.ln_2.b", "fp32", tensors[f"{p}.ln_2.bias"])

        # HF stores [in, out]; we store [out, in]. c_attn columns = [Q; K; V]
        attn_w = tensors[f"{p}.attn.c_attn.weight"].T.copy()
        attn_b = tensors[f"{p}.attn.c_attn.bias"]
        proj_w = tensors[f"{p}.attn.c_proj.weight"].T.copy()
        proj_b = tensors[f"{p}.attn.c_proj.bias"]
        fc_w = tensors[f"{p}.mlp.c_fc.weight"].T.copy()
        fc_b = tensors[f"{p}.mlp.c_fc.bias"]
        fc2_w = tensors[f"{p}.mlp.c_proj.weight"].T.copy()
        fc2_b = tensors[f"{p}.mlp.c_proj.bias"]

        for suffix, w in (("attn", attn_w), ("proj", proj_w),
                         ("fc", fc_w), ("fc2", fc2_w)):
            q, s = quantize_int8_symmetric(w)
            wr.add(f"blocks.{i}.{suffix}.w", "int8", q)
            wr.add(f"blocks.{i}.{suffix}.s", "fp32", s)
        wr.add(f"blocks.{i}.attn.b", "fp32", attn_b)
        wr.add(f"blocks.{i}.proj.b", "fp32", proj_b)
        wr.add(f"blocks.{i}.fc.b", "fp32", fc_b)
        wr.add(f"blocks.{i}.fc2.b", "fp32", fc2_b)

        n_params += attn_w.size + proj_w.size + fc_w.size + fc2_w.size

    n_params += tensors["wte.weight"].size + tensors["wpe.weight"].size
    for nm in ("ln_1.weight", "ln_1.bias", "ln_2.weight", "ln_2.bias"):
        n_params += tensors[f"h.0.{nm}"].size * n_layer
    n_params += tensors["ln_f.weight"].size + tensors["ln_f.bias"].size
    wr.set_meta(n_params=n_params)

    # tokenizer blobs
    vocab_json = json.loads((src / "vocab.json").read_text())
    merges_txt = (src / "merges.txt").read_text().splitlines()
    vocab_blob = bytearray(struct.pack("<I", len(vocab_json)))
    for tok, tid in vocab_json.items():
        b = tok.encode("utf-8")
        vocab_blob += struct.pack("<II", tid, len(b)) + b
    merges_blob = bytearray(struct.pack("<I", len(merges_txt)))
    for line in merges_txt:
        if not line.strip():
            continue
        left, right = line.split(" ", 1)
        la, lb = left.encode("utf-8"), right.encode("utf-8")
        merges_blob += struct.pack("<I", len(la)) + la + struct.pack("<I", len(lb)) + lb
    wr.add("tokenizer.vocab", "text", np.frombuffer(bytes(vocab_blob), dtype=np.uint8))
    wr.add("tokenizer.merges", "text", np.frombuffer(bytes(merges_blob), dtype=np.uint8))

    with open(out, "wb") as fh:
        wr.write(fh)
    return {
        "model": model_name,
        "n_layer": n_layer,
        "n_head": n_head,
        "n_embd": n_embd,
        "n_ctx": n_ctx,
        "vocab": vocab,
        "n_params": n_params,
        "out": str(out),
        "size_mb": out.stat().st_size / 1e6,
    }


class EcmReader:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.data = path.read_bytes()
        self.meta: dict[str, Any] = {}
        self.tensor_meta: dict[str, dict[str, Any]] = {}
        self._parse()

    def _parse(self) -> None:
        d = self.data
        assert d.startswith(MAGIC.encode()), "bad magic"
        p = len(MAGIC)
        while True:
            nl = d.index(b"\n", p)
            line = d[p:nl].decode()
            p = nl + 1
            if line == "END_META":
                break
            k, _, v = line.partition("=")
            self.meta[k] = v

        names: list[str] = []
        while True:
            nl = d.index(b"\n", p)
            line = d[p:nl].decode()
            p = nl + 1
            if line == "END_TENSORS":
                break
            parts = line.split()
            name, dtype = parts[0], parts[1]
            if dtype == "text":
                nbytes = int(parts[2])
                nelems = nbytes
            else:
                dims = [int(x) for x in parts[2:]]
                nbytes = np.prod(dims, dtype=int) * {
                    "fp32": 4, "fp16": 2, "int8": 1}[dtype]
                nelems = np.prod(dims, dtype=int)
            self.tensor_meta[name] = {
                "dtype": dtype, "offset": 0, "bytes": nbytes,
                "dims": None if dtype == "text" else [int(x) for x in parts[2:]],
            }
            names.append(name)

        # assign blob offsets sequentially from the data region start
        off = p
        for name in names:
            self.tensor_meta[name]["offset"] = off
            off += self.tensor_meta[name]["bytes"]

    def tensor(self, name: str) -> np.ndarray:
        m = self.tensor_meta[name]
        buf = self.data[m["offset"] : m["offset"] + m["bytes"]]
        if m["dtype"] == "text":
            return np.frombuffer(buf, dtype=np.uint8)
        dt = {"fp32": np.float32, "fp16": np.float16, "int8": np.int8}[m["dtype"]]
        return np.frombuffer(buf, dtype=dt).reshape(m["dims"])
