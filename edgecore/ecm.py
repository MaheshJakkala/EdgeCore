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


def f32_to_bf16(a: np.ndarray) -> np.ndarray:
    """Convert fp32 to bfloat16 (round-to-nearest-even), returned as uint16.

    numpy has no native bfloat16, so the .ecm format stores bf16 weights as
    raw uint16 bit patterns. The C runtime dequantizes them to fp32 with a
    plain 16-bit left shift (bit-exact), so this rounding is the only place
    precision is lost on the export path.
    """
    x = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)
    nan_mask = (x & np.uint32(0x7FFFFFFF)) > np.uint32(0x7F800000)
    lsb = (x >> np.uint32(16)) & np.uint32(1)
    r = (x.astype(np.uint64) + np.uint64(0x7FFF) + lsb.astype(np.uint64)) >> np.uint64(16)
    r = r.astype(np.uint32)
    r = np.where(nan_mask, np.uint32(0x7FC0), r)
    return r.astype(np.uint16)


def bf16_to_f32(a: np.ndarray) -> np.ndarray:
    """Dequantize bfloat16 (uint16 bit patterns) to fp32, bit-exact."""
    u = np.asarray(a, dtype=np.uint16).astype(np.uint32)
    return (u << np.uint32(16)).view(np.float32)


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
            elif dtype == "bf16":
                if arr.dtype == np.uint16:
                    fh.write(arr.astype(np.uint16).tobytes())
                else:
                    fh.write(f32_to_bf16(arr).tobytes())
            elif dtype == "fp32":
                fh.write(arr.astype(np.float32).tobytes())
            elif dtype == "int8":
                fh.write(arr.astype(np.int8).tobytes())
            else:
                raise ValueError(f"unknown dtype {dtype}")


def detect_arch(cfg: dict[str, Any]) -> str:
    """Return 'qwen2' or 'gpt2' from an HF model config."""
    mt = str(cfg.get("model_type", "")).lower()
    archs = [str(a).lower() for a in cfg.get("architectures", [])]
    if mt in ("qwen2", "qwen2_moe", "olmo", "olmo3"):
        return "qwen2"
    if any("qwen2" in a for a in archs) or any("olmo" in a for a in archs):
        return "qwen2"
    return "gpt2"


def arch_from_meta(meta: dict[str, Any]) -> str:
    """Return 'qwen2' or 'gpt2' from an .ecm meta section."""
    arch = str(meta.get("arch", "")).lower()
    if arch in ("qwen2", "olmo3"):
        return "qwen2"
    return "gpt2"


def first_linear_key(meta: dict[str, Any]) -> str:
    """Tensor name of the first linear layer's weights (arch-aware)."""
    return "blocks.0.q.w" if arch_from_meta(meta) == "qwen2" else "blocks.0.attn.w"


def build_tokenizer_blobs_v2(src: Path) -> tuple[np.ndarray, np.ndarray]:
    """Serialize vocab.json + added special tokens + merges.txt.

    Qwen2 keeps its special tokens (``<|endoftext|>``, ``<|im_start|>`` ...)
    OUT of vocab.json, in ``tokenizer_config.json`` under
    ``added_tokens_decoder``. The C runtime looks these up by content, so
    they must be merged into the vocab blob for ``<|endoftext|>`` resolution
    and single-token special-token encoding to work.
    """
    import struct

    vocab_json = json.loads((src / "vocab.json").read_text())
    merges_txt = (src / "merges.txt").read_text().splitlines()

    tc_path = src / "tokenizer_config.json"
    added: dict[str, int] = {}
    if tc_path.exists():
        try:
            tc = json.loads(tc_path.read_text())
            dec = tc.get("added_tokens_decoder") or {}
            for tid, spec in dec.items():
                content = spec.get("content") if isinstance(spec, dict) else None
                if isinstance(content, str) and content not in vocab_json:
                    added[content] = int(tid)
        except Exception:
            added = {}

    merged = {**vocab_json, **added}
    vocab_blob = bytearray(struct.pack("<I", len(merged)))
    for tok, tid in merged.items():
        b = tok.encode("utf-8")
        vocab_blob += struct.pack("<II", tid, len(b)) + b

    merges_blob = bytearray(struct.pack("<I", len(merges_txt)))
    for line in merges_txt:
        if not line.strip():
            continue
        left, right = line.split(" ", 1)
        la, lb = left.encode("utf-8"), right.encode("utf-8")
        merges_blob += struct.pack("<I", len(la)) + la + struct.pack("<I", len(lb)) + lb

    return (
        np.frombuffer(bytes(vocab_blob), dtype=np.uint8),
        np.frombuffer(bytes(merges_blob), dtype=np.uint8),
    )


def _load_safetensors(path: Path,
                      keep_bf16: bool = False) -> dict[str, np.ndarray]:
    """Load a safetensors file into numpy arrays.

    With ``keep_bf16=False`` (default) BF16 tensors are upconverted to fp32
    (bit-exact). With ``keep_bf16=True`` they are returned as raw uint16 so
    an export can store them natively as ``bf16`` without any rounding.
    """
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
    dtmap = {"F32": np.float32, "F16": np.float16, "BF16": np.uint16, "I64": np.int64}
    for name, info in header.items():
        begin, end = info["data_offsets"]
        buf = raw[data_base + begin : data_base + end]
        dt = info["dtype"]
        need = np.prod(info["shape"], dtype=int) * np.dtype(dtmap[dt]).itemsize
        if len(buf) < need:
            raise ValueError(
                f"{path}: tensor '{name}' needs {need} bytes but only {len(buf)} "
                f"remain - truncated download; re-download with "
                f"`python3 scripts/fetch_model.py --force`")
        arr = np.frombuffer(buf, dtype=dtmap[dt]).reshape(info["shape"])
        if dt == "BF16" and not keep_bf16:
            # numpy has no native bfloat16; upconvert to fp32 (bit-exact).
            arr = (arr.astype(np.uint32) << 16).view(np.float32)
        elif dt == "F16":
            arr = arr.astype(np.float32)
        elif dt == "F32":
            arr = arr.astype(np.float32)
        tensors[name] = arr
    return tensors


def _qwen2_meta(cfg: dict[str, Any]) -> dict[str, Any]:
    """Derive the .ecm meta section for a Qwen2 / OLMo config."""
    rope_scaling = cfg.get("rope_scaling") or {}
    rope_type = "yarn" if str(rope_scaling.get("type", "")).lower() == "yarn" else "default"
    meta: dict[str, Any] = {
        "model_name": cfg.get("_name_or_path", "qwen2"),
        "n_layer": cfg["num_hidden_layers"],
        "n_head": cfg["num_attention_heads"],
        "n_kv_head": cfg.get("num_key_value_heads", cfg["num_attention_heads"]),
        "n_embd": cfg["hidden_size"],
        "intermediate": cfg.get("intermediate_size", 4 * cfg["hidden_size"]),
        "n_ctx": cfg.get("max_position_embeddings", 32768),
        "vocab": cfg.get("vocab_size", 151936),
        "eps": cfg.get("rms_norm_eps", 1e-6),
        "rope_theta": cfg.get("rope_theta", 10000.0),
        "rope_type": rope_type,
        "yarn_factor": float(rope_scaling.get("factor", 1.0) or 1.0),
        "yarn_orig_max": float(rope_scaling.get("original_max_position_embeddings",
                                                cfg.get("max_position_embeddings", 32768)) or 0.0),
        "yarn_attention_factor": float(rope_scaling.get("attention_factor", 1.0) or 1.0),
        "sliding_window": int(cfg.get("sliding_window", 0) or 0),
        "tie_word_embeddings": int(bool(cfg.get("tie_word_embeddings", False))),
        "eos_token_id": int(cfg.get("eos_token_id", -1)),
        "arch": "qwen2",
    }
    return meta


def export_qwen2_from_safetensors(src: Path, out: Path,
                                  precision: str = "fp16") -> dict[str, Any]:
    """Convert an HF Qwen2 / OLMo directory (safetensors + config) to .ecm.

    Layout matches the C runtime's ARCH_QWEN2 expectations:

      meta   arch=qwen2, n_kv_head, intermediate, rope_theta, rope_type,
             yarn_*, sliding_window, tie_word_embeddings, eos_token_id, ...
      wte    [vocab][n_embd] fp16|bf16
      ln_f.g [n_embd] fp32                          (final RMSNorm, no bias)
      lm_head.w [vocab][n_embd] fp16|bf16            (only if untied + present)
      blocks.{i}.ln_1.g / ln_2.g                     (RMSNorm weights, no bias)
      blocks.{i}.{q,k,v,o,gate,up,down}.w  [out][in] (fp16, bf16 or int8)
      blocks.{i}.{q,k,v,o,gate,up,down}.s  fp32 per-output scales (int8 only)

    ``precision``: ``fp16`` (fp16 linears), ``bf16`` (native bfloat16 linears
    and embeddings; bit-exact when the checkpoint already stores BF16), or
    ``int8`` (per-output-channel int8 linears + fp32 scales).
    """
    if precision not in ("fp16", "bf16", "int8"):
        raise ValueError(f"unsupported precision {precision!r}")
    cfg = json.loads((src / "config.json").read_text())
    n_layer = cfg["num_hidden_layers"]
    n_head = cfg["num_attention_heads"]
    n_kv_head = int(cfg.get("num_key_value_heads", n_head))
    hidden = cfg["hidden_size"]
    head_dim = int(cfg.get("head_dim") or (hidden // n_head))
    n_kv_embd = n_kv_head * head_dim
    vocab = cfg["vocab_size"]
    tied = bool(cfg.get("tie_word_embeddings", False))

    tensors = _load_safetensors(src / "model.safetensors",
                                keep_bf16=(precision == "bf16"))
    wr = EcmWriter()
    wr.set_meta(**_qwen2_meta(cfg))

    n_params = 0

    def add_embed(name: str, arr: np.ndarray) -> None:
        """Add embeddings / lm_head as fp16 (or native bf16) arrays.

        The C runtime and the tied lm-head path always read ``wte`` and
        ``lm_head.w`` as fp16/bf16 words, so these must NEVER be int8 even for
        an ``int8`` artifact (only the projection linears are quantized).
        """
        nonlocal n_params
        n_params += arr.size
        if precision == "bf16":
            if arr.dtype != np.uint16:
                arr = arr.astype(np.float32)
            wr.add(name, "bf16", arr)
        else:
            wr.add(name, "fp16", arr.astype(np.float32))

    def add_linear(stem: str, arr: np.ndarray) -> None:
        """Add a [out][in] projection; int8 emits fp32 per-output scales that
        the C runtime / reference look up as ``<stem>.s``."""
        nonlocal n_params
        n_params += arr.size
        if precision == "bf16":
            if arr.dtype != np.uint16:
                arr = arr.astype(np.float32)
            wr.add(stem + ".w", "bf16", arr)
        elif precision == "int8":
            q, s = quantize_int8_symmetric(arr.astype(np.float32))
            wr.add(stem + ".w", "int8", q)
            wr.add(stem + ".s", "fp32", s)
        else:
            wr.add(stem + ".w", "fp16", arr.astype(np.float32))

    add_embed("wte", tensors["model.embed_tokens.weight"])
    wr.add("ln_f.g", "fp32", tensors["model.norm.weight"])

    if not tied and "lm_head.weight" in tensors:
        add_embed("lm_head.w", tensors["lm_head.weight"])

    # per-layer qwen2 projections, all [out][in] in .ecm (HF stores [out, in]
    # unlike GPT-2's Conv1D which is [in, out]), so do NOT transpose.
    layers: list[tuple[str, str]] = [
        ("q", "model.layers.{i}.self_attn.q_proj.weight"),
        ("k", "model.layers.{i}.self_attn.k_proj.weight"),
        ("v", "model.layers.{i}.self_attn.v_proj.weight"),
        ("o", "model.layers.{i}.self_attn.o_proj.weight"),
        ("gate", "model.layers.{i}.mlp.gate_proj.weight"),
        ("up", "model.layers.{i}.mlp.up_proj.weight"),
        ("down", "model.layers.{i}.mlp.down_proj.weight"),
    ]

    for i in range(n_layer):
        wr.add(f"blocks.{i}.ln_1.g", "fp32",
               tensors[f"model.layers.{i}.input_layernorm.weight"])
        wr.add(f"blocks.{i}.ln_2.g", "fp32",
               tensors[f"model.layers.{i}.post_attention_layernorm.weight"])
        for suffix, hf_key in layers:
            add_linear(f"blocks.{i}.{suffix}", tensors[hf_key.format(i=i)])

    wr.set_meta(n_params=n_params)

    vocab_blob, merges_blob = build_tokenizer_blobs_v2(src)
    wr.add("tokenizer.vocab", "text", vocab_blob)
    wr.add("tokenizer.merges", "text", merges_blob)

    with open(out, "wb") as fh:
        wr.write(fh)
    return {
        "model": cfg.get("_name_or_path", "qwen2"),
        "arch": "qwen2",
        "precision": precision,
        "n_layer": n_layer,
        "n_head": n_head,
        "n_kv_head": n_kv_head,
        "n_kv_embd": n_kv_embd,
        "n_embd": hidden,
        "intermediate": cfg["intermediate_size"],
        "n_ctx": cfg.get("max_position_embeddings", 32768),
        "vocab": vocab,
        "n_params": n_params,
        "out": str(out),
        "size_mb": out.stat().st_size / 1e6,
    }


def export_gpt2_from_safetensors(src: Path, out: Path) -> dict[str, Any]:
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


def export_from_safetensors(src: Path, out: Path,
                            precision: str = "int8") -> dict[str, Any]:
    """Convert an HF model directory to .ecm, dispatching on the architecture.

    ``precision`` applies to Qwen2 linear weights (``fp16`` or ``int8``);
    the legacy GPT-2 path always emits int8 per-channel linears.
    """
    cfg = json.loads((src / "config.json").read_text())
    if detect_arch(cfg) == "qwen2":
        return export_qwen2_from_safetensors(src, out, precision=precision)
    return export_gpt2_from_safetensors(src, out)


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
                    "fp32": 4, "fp16": 2, "bf16": 2, "int8": 1}[dtype]
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
        dt = {"fp32": np.float32, "fp16": np.float16,
              "bf16": np.uint16, "int8": np.int8}[m["dtype"]]
        return np.frombuffer(buf, dtype=dt).reshape(m["dims"])
