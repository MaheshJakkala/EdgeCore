"""EdgeCore - numpy fp32 reference forward for GPT-2.

Step 6: provides a deterministic, dependency-light ground-truth
implementation used by the verification layer. It loads weights either
from an ``.ecm`` artifact (replicating the exact fp16/int8 representation
of the C runtime) or directly from an HF safetensors directory (pure fp32
ground truth), then runs a full forward pass over a token sequence in
numpy.

The forward replicates the C runtime arithmetic so that verification is
meaningful:

* embeddings (wte/wpe) are fp16 in ``.ecm`` and are dequantized the same
  way the C runtime does;
* fp16 linear layers dequantize each weight from fp16 to fp32;
* int8 linear layers run the exact per-row activation quantization and
  per-output-channel weight scales of ``kernels.c`` (int32 dot products);
* KV is optionally rounded through fp16 to mirror the runtime's fp16 KV
  cache (``simulate_fp16_kv``).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .ecm import EcmReader
from .ecm import _load_safetensors

__all__ = ["GPT2Reference", "load_ecm_reference", "load_safetensors_reference",
           "gelu_new", "layernorm", "softmax_2d"]


def gelu_new(x: np.ndarray) -> np.ndarray:
    """GPT-2 / GPT-J style GELU: 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))."""
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


def layernorm(x: np.ndarray, g: np.ndarray, b: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Layer norm over the last axis with fp64 accumulation (matches runtime)."""
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return ((x - mean) / np.sqrt(var + eps)) * g + b


def softmax_2d(x: np.ndarray) -> np.ndarray:
    """Row-wise softmax in fp64 then downcast to fp32."""
    x = x.astype(np.float64)
    m = x.max(axis=-1, keepdims=True)
    e = np.exp(x - m)
    return (e / e.sum(axis=-1, keepdims=True)).astype(np.float32)


def _quantize_rows_symmetric(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-row symmetric int8 quantization exactly as the runtime linear() does."""
    a = a.astype(np.float64)
    maxabs = np.max(np.abs(a), axis=1)
    scales = np.where(maxabs > 0.0, maxabs / 127.0, 1.0)
    q = np.rint(a / scales[:, None])
    q = np.clip(q, -128, 127).astype(np.int8)
    return q, scales.astype(np.float32)


class _Linear:
    """A single linear layer in one of three precisions."""

    __slots__ = ("mode", "w", "w_q", "w_s", "b", "fp16_kv")

    def __init__(self, mode: str, w: np.ndarray, w_q: np.ndarray | None,
                 w_s: np.ndarray | None, b: np.ndarray | None) -> None:
        self.mode = mode              # "fp32" | "fp16" | "int8"
        self.w = w                    # dequantized fp32 [out, in] (reference value)
        self.w_q = w_q                # int8 [out, in] (exact runtime value)
        self.w_s = w_s                # fp32 [out] per-output-channel scale
        self.b = b

    @classmethod
    def from_fp32(cls, w: np.ndarray, b: np.ndarray | None) -> "_Linear":
        return cls("fp32", np.ascontiguousarray(w, dtype=np.float32), None, None,
                   None if b is None else np.ascontiguousarray(b, dtype=np.float32))

    @classmethod
    def from_fp16(cls, w: np.ndarray, b: np.ndarray | None) -> "_Linear":
        w16 = np.ascontiguousarray(w, dtype=np.float16)
        return cls("fp16", w16.astype(np.float32), None, None,
                   None if b is None else np.ascontiguousarray(b, dtype=np.float32))

    @classmethod
    def from_int8(cls, q: np.ndarray, s: np.ndarray, b: np.ndarray | None) -> "_Linear":
        q = np.ascontiguousarray(q, dtype=np.int8)
        s = np.ascontiguousarray(s, dtype=np.float32)
        w = q.astype(np.float32) * s[:, None]
        return cls("int8", w, q, s,
                   None if b is None else np.ascontiguousarray(b, dtype=np.float32))

    def apply(self, x: np.ndarray) -> np.ndarray:
        """x: [M, K] fp32 -> [M, N] fp32. Replicates the runtime exactly."""
        if self.mode == "fp16":
            out = x.astype(np.float32) @ self.w.astype(np.float32).T
        elif self.mode == "int8":
            qa, a_scale = _quantize_rows_symmetric(x)
            dot = qa.astype(np.int32) @ self.w_q.astype(np.int32).T
            out = dot.astype(np.float32) * (a_scale[:, None] * self.w_s[None, :])
        else:
            out = x.astype(np.float32) @ self.w.T
        if self.b is not None:
            out = out + self.b[None, :]
        return out.astype(np.float32)


class GPT2Reference:
    """Numpy GPT-2 forward. Build via load_ecm_reference / load_safetensors_reference."""

    def __init__(self, meta: dict[str, Any], wte: np.ndarray, wpe: np.ndarray,
                 lnf_g: np.ndarray, lnf_b: np.ndarray, layers: list[dict[str, Any]],
                 eps: float = 1e-5, simulate_fp16_kv: bool = True) -> None:
        self.meta = meta
        self.n_layer = int(meta["n_layer"])
        self.n_head = int(meta["n_head"])
        self.n_embd = int(meta["n_embd"])
        self.n_ctx = int(meta["n_ctx"])
        self.vocab = int(meta["vocab"])
        self.head_dim = self.n_embd // self.n_head
        self.eps = eps
        self.simulate_fp16_kv = simulate_fp16_kv
        # embeddings: wte/wpe may be fp16-stored; keep both raw + dequantized
        self.wte_raw = wte
        self.wpe_raw = wpe
        self.wte = _dequant_embed(wte)
        self.wpe = _dequant_embed(wpe)
        self.lnf_g = lnf_g.astype(np.float32)
        self.lnf_b = lnf_b.astype(np.float32)
        self.layers = layers
        self._weight_precision = _detect_precision(wte, wpe)

    # ------------------------------------------------------------------
    def forward(self, tokens: np.ndarray, positions: np.ndarray | None = None) -> np.ndarray:
        """Full forward pass; returns logits [n, vocab] fp32 (causal)."""
        tokens = np.asarray(tokens, dtype=np.int64)
        n = tokens.shape[0]
        if positions is None:
            positions = np.arange(n, dtype=np.int64)
        E, H, V, L = self.n_embd, self.n_head, self.vocab, self.n_layer
        hd = self.head_dim
        inv_hd = 1.0 / math.sqrt(float(hd))

        # embeddings
        x = self.wte[tokens] + self.wpe[positions]          # [n, E] fp32

        for l in range(L):
            ln = self.layers[l]
            h1 = layernorm(x, ln["ln1_g"], ln["ln1_b"], self.eps)      # [n, E]
            qkv = ln["attn"].apply(h1)                                 # [n, 3E]
            q, k, v = qkv[:, :E], qkv[:, E:2 * E], qkv[:, 2 * E:]

            if self.simulate_fp16_kv:
                k = k.astype(np.float16).astype(np.float32)
                v = v.astype(np.float16).astype(np.float32)

            # per-head attention with causal mask
            q = q.reshape(n, H, hd)
            k = k.reshape(n, H, hd)
            v = v.reshape(n, H, hd)
            scores = np.einsum("nhd,mhd->nhm", q, k) * inv_hd            # [n, H, n]
            mask = np.triu(np.ones((n, n), dtype=bool), 1)
            scores = scores.astype(np.float64)
            scores = np.where(mask[:, None, :], -np.inf, scores)
            probs = softmax_2d(scores)                                   # [n, H, n]
            attn_out = np.einsum("nhm,mhd->nhd", probs, v).reshape(n, E) # [n, E]

            h = ln["proj"].apply(attn_out)
            x = x + h

            h2 = layernorm(x, ln["ln2_g"], ln["ln2_b"], self.eps)
            f = ln["fc"].apply(h2)
            f = gelu_new(f)
            h = ln["fc2"].apply(f)
            x = x + h

        h = layernorm(x, self.lnf_g, self.lnf_b, self.eps)
        logits = h.astype(np.float32) @ self.wte.T
        return logits.astype(np.float32)

    # ------------------------------------------------------------------
    def log_probs(self, tokens: np.ndarray) -> np.ndarray:
        """log-probabilities [n, vocab] via fp64 softmax."""
        logits = self.forward(tokens)
        logits = logits.astype(np.float64)
        m = logits.max(axis=-1, keepdims=True)
        e = np.exp(logits - m)
        lse = np.log(e.sum(axis=-1, keepdims=True))
        return logits - m - lse

    def perplexity(self, tokens: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        """Corpus perplexity (excl. first token) + per-token NLL + evaluated ids."""
        logp = self.log_probs(tokens)
        if logp.shape[0] < 2:
            return math.inf, logp, np.array([], dtype=np.int64)
        nll = -logp[:-1, tokens[1:]]
        ppl = math.exp(float(nll.mean()))
        return ppl, nll, tokens[1:]

    def top1(self, tokens: np.ndarray) -> np.ndarray:
        """Predicted next token id for each position (greedy)."""
        return np.argmax(self.forward(tokens), axis=-1).astype(np.int64)

    @property
    def weight_precision(self) -> str:
        return self._weight_precision


# ----------------------------------------------------------------------
def _dequant_embed(a: np.ndarray) -> np.ndarray:
    if a.dtype == np.float16:
        return a.astype(np.float32)
    return np.asarray(a, dtype=np.float32)


def _detect_precision(wte: np.ndarray, wpe: np.ndarray) -> str:
    if wte.dtype == np.float16 or wpe.dtype == np.float16:
        return "fp16-embeddings"
    return "fp32-embeddings"


def load_ecm_reference(path: str | Path, simulate_fp16_kv: bool = True) -> GPT2Reference:
    """Build a reference from an .ecm artifact, replicating its exact precision."""
    path = Path(path)
    r = EcmReader(path)
    meta = {k: _coerce(v) for k, v in r.meta.items()}

    wte = r.tensor("wte")
    wpe = r.tensor("wpe")
    lnf_g = r.tensor("ln_f.g")
    lnf_b = r.tensor("ln_f.b")

    layers: list[dict[str, Any]] = []
    for i in range(int(meta["n_layer"])):
        p = f"blocks.{i}"
        lw: dict[str, Any] = {
            "ln1_g": r.tensor(f"{p}.ln_1.g"),
            "ln1_b": r.tensor(f"{p}.ln_1.b"),
            "ln2_g": r.tensor(f"{p}.ln_2.g"),
            "ln2_b": r.tensor(f"{p}.ln_2.b"),
        }
        for suffix in ("attn", "proj", "fc", "fc2"):
            w = r.tensor(f"{p}.{suffix}.w")
            b = r.tensor(f"{p}.{suffix}.b")
            if w.dtype == np.int8:
                s = r.tensor(f"{p}.{suffix}.s")
                lw[suffix] = _Linear.from_int8(w, s, b)
            elif w.dtype == np.float16:
                lw[suffix] = _Linear.from_fp16(w, b)
            else:
                lw[suffix] = _Linear.from_fp32(w, b)
        layers.append(lw)

    return GPT2Reference(meta, wte, wpe, lnf_g, lnf_b, layers,
                         eps=float(meta.get("eps", 1e-5)),
                         simulate_fp16_kv=simulate_fp16_kv)


def load_safetensors_reference(hf_dir: str | Path, simulate_fp16_kv: bool = False) -> GPT2Reference:
    """Build a pure-fp32 reference directly from an HF safetensors directory."""
    hf_dir = Path(hf_dir)
    cfg = json.loads((hf_dir / "config.json").read_text())
    meta = {
        "model_name": cfg.get("_name_or_path", "gpt2"),
        "n_layer": cfg["n_layer"],
        "n_head": cfg["n_head"],
        "n_embd": cfg["n_embd"],
        "n_ctx": cfg.get("n_positions", cfg.get("n_ctx", 1024)),
        "vocab": cfg["vocab_size"],
        "eps": cfg.get("layer_norm_epsilon", 1e-5),
    }
    t = _load_safetensors(hf_dir / "model.safetensors")

    wte = t["wte.weight"].astype(np.float32)
    wpe = t["wpe.weight"].astype(np.float32)
    lnf_g = t["ln_f.weight"].astype(np.float32)
    lnf_b = t["ln_f.bias"].astype(np.float32)

    layers: list[dict[str, Any]] = []
    for i in range(int(meta["n_layer"])):
        p = f"h.{i}"
        # HF stores [in, out]; transpose to [out, in]
        lw: dict[str, Any] = {
            "ln1_g": t[f"{p}.ln_1.weight"].astype(np.float32),
            "ln1_b": t[f"{p}.ln_1.bias"].astype(np.float32),
            "ln2_g": t[f"{p}.ln_2.weight"].astype(np.float32),
            "ln2_b": t[f"{p}.ln_2.bias"].astype(np.float32),
        }
        linear_map = {
            "attn": f"{p}.attn.c_attn",
            "proj": f"{p}.attn.c_proj",
            "fc": f"{p}.mlp.c_fc",
            "fc2": f"{p}.mlp.c_proj",
        }
        for suffix, lname in linear_map.items():
            w = t[f"{lname}.weight"].T.astype(np.float32)
            b = t[f"{lname}.bias"].astype(np.float32)
            lw[suffix] = _Linear.from_fp32(w, b)
        layers.append(lw)

    return GPT2Reference(meta, wte, wpe, lnf_g, lnf_b, layers,
                         eps=float(meta["eps"]),
                         simulate_fp16_kv=simulate_fp16_kv)


def _coerce(v: str):
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        return v
