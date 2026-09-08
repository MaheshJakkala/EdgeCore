"""EdgeCore - numpy fp32 reference forward for GPT-2.

Step 6: provides a deterministic, dependency-light ground-truth
implementation used by the verification layer. It loads weights either
from an ``.ecm`` artifact (replicating the exact fp16/int8 representation
of the C runtime) or directly from an HF safetensors directory (pure fp32
ground truth), then runs a full forward pass over a token sequence in
numpy.

The forward replicates the C runtime arithmetic so that verification is
meaningful:

* embeddings (wte/wpe) are fp16/bf16 in ``.ecm`` and are dequantized the same
  way the C runtime does;
* fp16 linear layers dequantize each weight from fp16 to fp32;
* bf16 linear layers dequantize each weight from bf16 (uint16) to fp32;
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
from .ecm import bf16_to_f32

__all__ = ["GPT2Reference", "Qwen2Reference", "load_ecm_reference",
           "load_safetensors_reference", "gelu_new", "layernorm", "softmax_2d",
           "rmsnorm", "qwen_inv_freq", "qwen_rope"]


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


def rmsnorm(x: np.ndarray, g: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """RMSNorm over the last axis with fp64 accumulation (matches runtime)."""
    ms = (x.astype(np.float64) ** 2).mean(axis=-1, keepdims=True)
    rstd = 1.0 / np.sqrt(ms + eps)
    return (x * rstd) * g


def qwen_inv_freq(half: int, rope_theta: float, rope_type: str = "default",
                  yarn_factor: float = 1.0, yarn_orig_max: float = 0.0,
                  yarn_attention_factor: float = 1.0) -> np.ndarray:
    """Inverse frequencies exactly as gpt2.c's qwen_inv_freq."""
    dim = 2 * half
    inv_base = 1.0 / rope_theta
    inv_freq = np.asarray([inv_base ** (2 * i / dim) for i in range(half)],
                          dtype=np.float64)
    if rope_type != "yarn":
        return inv_freq

    factor = float(yarn_factor)
    max_pos = float(yarn_orig_max) if yarn_orig_max > 0 else float(dim)
    beta_fast, beta_slow = 32.0, 1.0
    pi2 = 6.28318530717958647692
    logbase = math.log(rope_theta)

    def corr_dim(rot: float) -> float:
        return dim * math.log(max_pos / (rot * pi2)) / (2.0 * logbase)

    lo = math.floor(corr_dim(beta_fast))
    hi = math.ceil(corr_dim(beta_slow))
    lo = max(0.0, lo)
    hi = min(float(dim - 1), hi)
    span = max(1.0, hi - lo)

    ramp = (np.arange(half, dtype=np.float64) - lo) / span
    ramp = np.clip(ramp, 0.0, 1.0)
    extrap = inv_freq
    interp = inv_freq / factor
    return interp * ramp + extrap * (1.0 - ramp)


def qwen_rope(q: np.ndarray, k: np.ndarray, positions: np.ndarray,
              inv_freq: np.ndarray, n_head: int, n_kv_head: int,
              head_dim: int, attn_scale: float = 1.0) -> None:
    """Apply RoPE in place to q [M, n_head*hd] and k [M, n_kv_head*hd].

    Matches gpt2.c qwen_rope_cos_sin + qwen_rope_rotate: the same cos/sin
    pair per position is used for every head.
    """
    half = head_dim // 2
    cosv = np.zeros(half, dtype=np.float64)
    sinv = np.zeros(half, dtype=np.float64)
    for m in range(positions.shape[0]):
        f = float(positions[m]) * inv_freq
        cosv[:] = np.cos(f) * attn_scale
        sinv[:] = np.sin(f) * attn_scale
        qh = q[m].reshape(n_head, head_dim)
        qa = qh[:, :half].copy()
        qb = qh[:, half:].copy()
        qh[:, :half] = qa * cosv[None, :] - qb * sinv[None, :]
        qh[:, half:] = qa * sinv[None, :] + qb * cosv[None, :]
        kh = k[m].reshape(n_kv_head, head_dim)
        ka = kh[:, :half].copy()
        kb = kh[:, half:].copy()
        kh[:, :half] = ka * cosv[None, :] - kb * sinv[None, :]
        kh[:, half:] = ka * sinv[None, :] + kb * cosv[None, :]


def _quantize_rows_symmetric(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-row symmetric int8 quantization exactly as the runtime linear() does.

    The C runtime performs every step in fp32 (``fabsf`` max, ``maxa / 127.0f``,
    ``lrintf(row[k]/s)``), so this replication must too; doing the max / scale /
    division in fp64 shifts a handful of elements onto different rounding
    boundaries and the resulting off-by-one int8 values drift through attention.
    """
    a = a.astype(np.float32)
    f127 = np.float32(127.0)
    one = np.float32(1.0)
    maxabs = np.max(np.abs(a), axis=1)
    scales = np.where(maxabs > 0.0, maxabs / f127, one).astype(np.float32)
    q = np.rint(a / scales[:, None])
    q = np.clip(q, -128, 127).astype(np.int8)
    return q, scales


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
    def from_bf16(cls, w: np.ndarray, b: np.ndarray | None) -> "_Linear":
        w16 = np.ascontiguousarray(w, dtype=np.uint16)
        return cls("bf16", bf16_to_f32(w16), None, None,
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
            # same association as kernels.c: (float)dot * a_scale * b_scale
            out = (dot.astype(np.float32) * a_scale[:, None]) * self.w_s[None, :]
        elif self.mode == "bf16":
            # runtime dequantizes bf16 -> fp32 (exact) then does fp32 GEMM
            out = x.astype(np.float32) @ self.w.astype(np.float32).T
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


class Qwen2Reference:
    """Numpy Qwen2 / OLMo decoder forward (RoPE + GQA + SwiGLU + RMSNorm).

    Replicates gpt2.c's qwen2_forward so that verification is meaningful.
    """

    def __init__(self, meta: dict[str, Any], wte: np.ndarray,
                 lnf_g: np.ndarray, layers: list[dict[str, Any]],
                 lm_head: np.ndarray | None = None, eps: float = 1e-6,
                 simulate_fp16_kv: bool = True) -> None:
        self.meta = meta
        self.n_layer = int(meta["n_layer"])
        self.n_head = int(meta["n_head"])
        self.n_embd = int(meta["n_embd"])
        self.n_ctx = int(meta["n_ctx"])
        self.vocab = int(meta["vocab"])
        self.n_kv_head = int(meta.get("n_kv_head", self.n_head))
        self.intermediate = int(meta.get("intermediate", 4 * self.n_embd))
        self.head_dim = int(meta.get("head_dim") or (self.n_embd // self.n_head))
        self.n_kv_embd = self.n_kv_head * self.head_dim
        self.eps = float(meta.get("eps", 1e-6))
        self.simulate_fp16_kv = simulate_fp16_kv
        self.rope_theta = float(meta.get("rope_theta", 10000.0))
        self.rope_type = str(meta.get("rope_type", "default"))
        self.yarn_factor = float(meta.get("yarn_factor", 1.0))
        self.yarn_orig_max = float(meta.get("yarn_orig_max", 0.0))
        self.yarn_attention_factor = float(meta.get("yarn_attention_factor", 1.0))
        self.sliding_window = int(meta.get("sliding_window", 0))
        self.wte_raw = wte
        self.wte = _dequant_embed(wte)
        self.lnf_g = lnf_g.astype(np.float32)
        self.lm_head = None if lm_head is None else _dequant_embed(lm_head)
        self.layers = layers
        half = self.head_dim // 2
        self._inv_freq = qwen_inv_freq(
            half, self.rope_theta, self.rope_type,
            self.yarn_factor, self.yarn_orig_max, self.yarn_attention_factor)
        self._weight_precision = _detect_precision(wte, wte)

    def forward(self, tokens: np.ndarray, positions: np.ndarray | None = None) -> np.ndarray:
        tokens = np.asarray(tokens, dtype=np.int64)
        n = tokens.shape[0]
        if positions is None:
            positions = np.arange(n, dtype=np.int64)
        E, H, V, L = self.n_embd, self.n_head, self.vocab, self.n_layer
        hd = self.head_dim
        half = hd // 2
        kv_embd = self.n_kv_embd
        inv_hd = 1.0 / math.sqrt(float(hd))

        x = self.wte[tokens].astype(np.float32)          # [n, E]

        for l in range(L):
            ln = self.layers[l]
            h1 = rmsnorm(x, ln["ln1_g"], self.eps).astype(np.float32)   # [n, E]

            q = ln["q"].apply(h1)                                        # [n, E]
            k = ln["k"].apply(h1)                                        # [n, kv_embd]
            v = ln["v"].apply(h1)                                        # [n, kv_embd]

            qwen_rope(q, k, positions, self._inv_freq, H, self.n_kv_head,
                      hd, self.yarn_attention_factor)

            if self.simulate_fp16_kv:
                k = k.astype(np.float16).astype(np.float32)
                v = v.astype(np.float16).astype(np.float32)

            # grouped-query attention with causal + sliding-window mask
            attn_out = np.zeros((n, E), dtype=np.float64)
            for h in range(H):
                kvh = h * self.n_kv_head // H
                qh = q[:, h * hd:(h + 1) * hd].astype(np.float64)        # [n, hd]
                kh = k[:, kvh * hd:(kvh + 1) * hd].astype(np.float64)    # [n, hd]
                vh = v[:, kvh * hd:(kvh + 1) * hd].astype(np.float64)
                for m in range(n):
                    lo = max(0, int(positions[m]) - self.sliding_window + 1) \
                        if self.sliding_window > 0 else 0
                    seg = slice(lo, int(positions[m]) + 1)
                    scores = qh[m] @ kh[seg].T * inv_hd                   # [seg]
                    probs = softmax_2d(scores[None, :])[0]               # fp64
                    attn_out[m, h * hd:(h + 1) * hd] = probs @ vh[seg]
            attn_out = attn_out.astype(np.float32)

            h = ln["o"].apply(attn_out)
            x = x + h

            h2 = rmsnorm(x, ln["ln2_g"], self.eps).astype(np.float32)
            gate = ln["gate"].apply(h2)                                  # [n, inter]
            up = ln["up"].apply(h2)
            f = (gate / (1.0 + np.exp(-gate))) * up
            h = ln["down"].apply(f)
            x = x + h

        h = rmsnorm(x, self.lnf_g, self.eps).astype(np.float32)
        lm = self.lm_head if self.lm_head is not None else self.wte
        logits = h.astype(np.float32) @ lm.T
        return logits.astype(np.float32)

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
    if a.dtype == np.uint16:
        return bf16_to_f32(a)
    return np.asarray(a, dtype=np.float32)


def _detect_precision(wte: np.ndarray, wpe: np.ndarray) -> str:
    if wte.dtype == np.float16 or wpe.dtype == np.float16:
        return "fp16-embeddings"
    if wte.dtype == np.uint16 or wpe.dtype == np.uint16:
        return "bf16-embeddings"
    return "fp32-embeddings"


def load_ecm_reference(path: str | Path, simulate_fp16_kv: bool = True):
    """Build a reference from an .ecm artifact, replicating its exact precision."""
    path = Path(path)
    r = EcmReader(path)
    meta = {k: _coerce(v) for k, v in r.meta.items()}
    if meta.get("arch") == "qwen2":
        return _load_qwen2_ecm(r, meta, simulate_fp16_kv)

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
            elif w.dtype == np.uint16:
                lw[suffix] = _Linear.from_bf16(w, b)
            else:
                lw[suffix] = _Linear.from_fp32(w, b)
        layers.append(lw)

    return GPT2Reference(meta, wte, wpe, lnf_g, lnf_b, layers,
                         eps=float(meta.get("eps", 1e-5)),
                         simulate_fp16_kv=simulate_fp16_kv)


def _load_qwen2_ecm(r: EcmReader, meta: dict[str, Any],
                    simulate_fp16_kv: bool) -> Qwen2Reference:
    wte = r.tensor("wte")
    lnf_g = r.tensor("ln_f.g")
    lm_head = r.tensor("lm_head.w") if "lm_head.w" in r.tensor_meta else None

    layers: list[dict[str, Any]] = []
    for i in range(int(meta["n_layer"])):
        p = f"blocks.{i}"
        lw: dict[str, Any] = {
            "ln1_g": r.tensor(f"{p}.ln_1.g"),
            "ln2_g": r.tensor(f"{p}.ln_2.g"),
        }
        for suffix in ("q", "k", "v", "o", "gate", "up", "down"):
            w = r.tensor(f"{p}.{suffix}.w")
            if w.dtype == np.int8:
                s = r.tensor(f"{p}.{suffix}.s")
                lw[suffix] = _Linear.from_int8(w, s, None)
            elif w.dtype == np.float16:
                lw[suffix] = _Linear.from_fp16(w, None)
            elif w.dtype == np.uint16:
                lw[suffix] = _Linear.from_bf16(w, None)
            else:
                lw[suffix] = _Linear.from_fp32(w, None)
        layers.append(lw)

    return Qwen2Reference(meta, wte, lnf_g, layers, lm_head=lm_head,
                          eps=float(meta.get("eps", 1e-6)),
                          simulate_fp16_kv=simulate_fp16_kv)


def load_qwen2_safetensors_reference(hf_dir: str | Path,
                                     simulate_fp16_kv: bool = False) -> Qwen2Reference:
    """Pure-fp32 Qwen2 reference from an HF safetensors directory."""
    from .ecm import _qwen2_meta

    hf_dir = Path(hf_dir)
    cfg = json.loads((hf_dir / "config.json").read_text())
    meta = _qwen2_meta(cfg)
    tied = bool(cfg.get("tie_word_embeddings", False))
    t = _load_safetensors(hf_dir / "model.safetensors")

    wte = t["model.embed_tokens.weight"].astype(np.float32)
    lnf_g = t["model.norm.weight"].astype(np.float32)
    lm_head = t["lm_head.weight"].astype(np.float32) if not tied else None

    layers: list[dict[str, Any]] = []
    for i in range(int(meta["n_layer"])):
        p = f"model.layers.{i}"
        lw: dict[str, Any] = {
            "ln1_g": t[f"{p}.input_layernorm.weight"].astype(np.float32),
            "ln2_g": t[f"{p}.post_attention_layernorm.weight"].astype(np.float32),
        }
        linear_map = {
            "q": f"{p}.self_attn.q_proj.weight",
            "k": f"{p}.self_attn.k_proj.weight",
            "v": f"{p}.self_attn.v_proj.weight",
            "o": f"{p}.self_attn.o_proj.weight",
            "gate": f"{p}.mlp.gate_proj.weight",
            "up": f"{p}.mlp.up_proj.weight",
            "down": f"{p}.mlp.down_proj.weight",
        }
        for suffix, lname in linear_map.items():
            # Qwen2 / OLMo linears are stored [out, in] in safetensors.
            w = t[lname].astype(np.float32)
            lw[suffix] = _Linear.from_fp32(w, None)
        layers.append(lw)

    return Qwen2Reference(meta, wte, lnf_g, layers, lm_head=lm_head,
                          eps=float(meta.get("eps", 1e-6)),
                          simulate_fp16_kv=simulate_fp16_kv)


def load_safetensors_reference(hf_dir: str | Path, simulate_fp16_kv: bool = False):
    """Build a pure-fp32 reference directly from an HF safetensors directory."""
    from .ecm import detect_arch

    hf_dir = Path(hf_dir)
    cfg = json.loads((hf_dir / "config.json").read_text())
    if detect_arch(cfg) == "qwen2":
        return load_qwen2_safetensors_reference(hf_dir, simulate_fp16_kv=simulate_fp16_kv)

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
