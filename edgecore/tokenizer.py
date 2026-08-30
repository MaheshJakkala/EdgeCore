"""EdgeCore - pure-Python GPT-2 byte-level BPE tokenizer.

Used by the Python tooling (reference forward, verification, benchmark
prompts). Mirrors the GPT-2 tokenizer: byte-level vocabulary, official
merges.txt, and the standard GPT-2 pre-tokenization regex. When the
``regex`` package is unavailable, falls back to a byte-classification
scanner that approximates the same splits (same approximation the C
runtime's tokenizer.c uses).
"""
from __future__ import annotations

import json
from pathlib import Path

try:
    import regex as _regex
    _RE = _regex.compile(
        r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    HAVE_REGEX = True
except Exception:  # pragma: no cover - fallback path
    import re as _re
    _HAVE_REGEX = False
    _RE = None
    HAVE_REGEX = False


def _byte_encoder_map() -> dict[int, str]:
    """GPT-2 bytes_to_unicode: raw byte -> single unicode char used in vocab."""
    bs = list(range(ord("!"), ord("~") + 1)) + \
        list(range(ord("\u00a1"), ord("\u00ac") + 1)) + \
        list(range(ord("\u00ae"), ord("\u00ff") + 1))
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


class ByteLevelBPE:
    """GPT-2 byte-level BPE using vocab.json + merges.txt."""

    def __init__(self, vocab_json: dict[str, int] | None = None,
                 merges: list[str] | None = None,
                 hf_dir: Path | None = None) -> None:
        if hf_dir is not None:
            with open(hf_dir / "vocab.json", encoding="utf-8") as f:
                vocab_json = json.load(f)
            with open(hf_dir / "merges.txt", encoding="utf-8") as f:
                merges = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#version")]

        self.encoder: dict[str, int] = dict(vocab_json or {})
        self.decoder: dict[int, str] = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks: dict[tuple[str, str], int] = {}
        for i, line in enumerate(merges or []):
            parts = line.split()
            if len(parts) == 2:
                self.bpe_ranks[(parts[0], parts[1])] = i
        self.cache: dict[str, str] = {}
        _be = _byte_encoder_map()
        self._byte_encoder = _be
        self._byte_decoder: dict[int, int] = {ord(c): b for b, c in _be.items()}

    # ------------------------------------------------------------------
    def _bpe(self, token: str) -> list[str]:
        """Byte-pair-encode a single byte-encoded token into vocab pieces."""
        if token in self.cache:
            return self.cache[token]
        word: list[str] = list(token)
        if len(word) == 1:
            self.cache[token] = word
            return word
        pairs = {(word[i], word[i + 1]) for i in range(len(word) - 1)}
        while pairs:
            pair = min(pairs, key=lambda p: self.bpe_ranks.get(p, float("inf")))
            if pair not in self.bpe_ranks:
                break
            a, b = pair
            new_word: list[str] = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                    new_word.append(a + b)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word
            if len(word) == 1:
                break
            pairs = {(word[i], word[i + 1]) for i in range(len(word) - 1)}
        self.cache[token] = word
        return word

    def _pretokenize(self, text: str) -> list[str]:
        if HAVE_REGEX and _RE is not None:
            return _RE.findall(text)
        return _pretokenize_fallback(text)

    def encode(self, text: str) -> list[int]:
        if text == "<|endoftext|>":
            eot = self.encoder.get("<|endoftext|>")
            return [eot] if eot is not None else []
        ids: list[int] = []
        for token in self._pretokenize(text):
            encoded = "".join(self._byte_encoder[b] for b in token.encode("utf-8"))
            for p in self._bpe(encoded):
                pid = self.encoder.get(p)
                if pid is None:
                    pid = _byte_fallback_id(p, self.encoder)
                if pid is not None:
                    ids.append(pid)
        return ids

    def decode(self, ids: list[int]) -> str:
        text = "".join(self.decoder.get(int(i), "") for i in ids)
        raw = bytearray()
        for ch in text:
            raw.append(self._byte_decoder.get(ord(ch), 0))
        return bytes(raw).decode("utf-8", errors="replace")


def _byte_fallback_id(token: str, encoder: dict[str, int]) -> int | None:
    """Fallback for unknown unicode pieces: map raw bytes to vocab entries."""
    for ch in token:
        b = ch.encode("utf-8")
        if b in encoder:
            return encoder[b]
    return None


def _pretokenize_fallback(text: str) -> list[str]:
    """Byte-classification split used by the C tokenizer when regex is absent."""
    parts: list[str] = []
    i = 0
    n = len(text)
    contractions = ("'s", "'t", "'re", "'ve", "'m", "'ll", "'d")
    while i < n:
        matched = False
        for c in contractions:
            if text.startswith(c, i):
                parts.append(c)
                i += len(c)
                matched = True
                break
        if matched:
            continue

        def is_letter(ch: str) -> bool:
            b = ord(ch)
            return (ord("A") <= b <= ord("Z")) or (ord("a") <= b <= ord("z")) or b >= 0x80

        def is_digit(ch: str) -> bool:
            return ch.isdigit()

        def is_space(ch: str) -> bool:
            return ch in " \t\n\r\v\f"

        sp = 1 if text[i] == " " else 0
        nxt = text[i + 1] if i + 1 < n else ""
        if sp:
            if nxt and is_letter(nxt):
                t = 2
            elif nxt and is_digit(nxt):
                t = 3
            elif nxt and not is_space(nxt):
                t = 4
            else:
                t = -1
        else:
            if is_letter(text[i]):
                t = 2
            elif is_digit(text[i]):
                t = 3
            elif not is_space(text[i]):
                t = 4
            else:
                t = -1
        j = i + sp
        if t == 2:
            while j < n and is_letter(text[j]):
                j += 1
        elif t == 3:
            while j < n and is_digit(text[j]):
                j += 1
        elif t == 4:
            while j < n and not is_space(text[j]) and not is_letter(text[j]) and not is_digit(text[j]):
                j += 1
        else:
            j = i
            while j < n and is_space(text[j]):
                j += 1
        parts.append(text[i:j])
        i = j
    return parts
