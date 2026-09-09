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
    """GPT-2 byte-level BPE using vocab.json + merges.txt.

    ``variant=1`` selects the Qwen2 pre-tokenizer (mirrors tokenizer.c's
    ``pretokenize_qwen2``); ``variant=0`` is the standard GPT-2 pattern.
    """

    def __init__(self, vocab_json: dict[str, int] | None = None,
                 merges: list[str] | None = None,
                 hf_dir: Path | None = None,
                 variant: int = 0) -> None:
        self.special_tokens: dict[str, int] = {}
        if hf_dir is not None:
            with open(hf_dir / "vocab.json", encoding="utf-8") as f:
                vocab_json = json.load(f)
            with open(hf_dir / "merges.txt", encoding="utf-8") as f:
                merges = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#version")]
            # Load special tokens from tokenizer_config.json
            tc_path = hf_dir / "tokenizer_config.json"
            if tc_path.exists():
                with open(tc_path, encoding="utf-8") as f:
                    tc = json.load(f)
                dec = tc.get("added_tokens_decoder") or {}
                for tid, spec in dec.items():
                    content = spec.get("content") if isinstance(spec, dict) else None
                    if isinstance(content, str):
                        self.special_tokens[content] = int(tid)
                        if content not in (vocab_json or {}):
                            if vocab_json is None:
                                vocab_json = {}
                            vocab_json[content] = int(tid)

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
        self.variant = int(variant)

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
        if self.variant == 1:
            return _pretokenize_qwen2_fallback(text)
        if HAVE_REGEX and _RE is not None:
            return _RE.findall(text)
        return _pretokenize_fallback(text)

    def encode(self, text: str) -> list[int]:
        # Check for exact special token matches first (before pretokenization)
        # This handles special tokens like "im_start", "im_end" that contain
        # characters that would be split by pretokenization
        for special_tok, special_id in self.special_tokens.items():
            if text == special_tok:
                return [special_id]
        
        # For the general case where special tokens might appear within text,
        # find all special token positions and encode segments between them.
        if self.special_tokens:
            # Sort special tokens by length (longest first) for longest-match
            sorted_special = sorted(self.special_tokens.items(), key=lambda x: -len(x[0]))
            ids: list[int] = []
            last_encoded_pos = 0  # Track position up to which we've encoded
            pos = 0
            while pos < len(text):
                matched = False
                for special_tok, special_id in sorted_special:
                    if text.startswith(special_tok, pos):
                        # Encode text between last_encoded_pos and pos (exclusive)
                        if pos > last_encoded_pos:
                            segment_ids = self._encode_plain(text[last_encoded_pos:pos])
                            ids.extend(segment_ids)
                        # Add the special token
                        ids.append(special_id)
                        # Update positions
                        pos += len(special_tok)
                        last_encoded_pos = pos
                        matched = True
                        break
                if not matched:
                    # No special token at this position, find next special token
                    next_special_pos = len(text)
                    for special_tok, _ in sorted_special:
                        found = text.find(special_tok, pos)
                        if found != -1 and found < next_special_pos:
                            next_special_pos = found
                    # Encode from pos to next_special_pos
                    if next_special_pos > pos:
                        segment_ids = self._encode_plain(text[pos:next_special_pos])
                        ids.extend(segment_ids)
                    pos = next_special_pos
                    last_encoded_pos = pos
            # Handle any remaining text after last special token
            if last_encoded_pos < len(text):
                segment_ids = self._encode_plain(text[last_encoded_pos:])
                ids.extend(segment_ids)
            return ids
        
        # Fallback to plain encoding if no special tokens
        return self._encode_plain(text)

    def _encode_plain(self, text: str) -> list[int]:
            """Encode plain text without special token handling."""
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


def _is_qw_letter(ch: str) -> bool:
        """Approximates \p{L} the way tokenizer.c does (ASCII letters + high bytes)."""
        o = ord(ch)
        return (ord("A") <= o <= ord("Z")) or (ord("a") <= o <= ord("z")) or o >= 0x80


def _is_qw_digit(ch: str) -> bool:
        """Approximates \p{N} as ASCII digits (matches tokenizer.c byte_is_digit)."""
        return "0" <= ch <= "9"


def _is_qw_space(ch: str) -> bool:
        return ch in " \t\n\r\v\f"


def _is_qw_newline(ch: str) -> bool:
        return ch in "\n\r"


def _pretokenize_qwen2_fallback(text: str) -> list[str]:
        """Byte-scanned mirror of tokenizer.c's ``pretokenize_qwen2``.

        Approximates the HF Qwen2 regex:
          '(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}|
          ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
        using the same byte-classification the C runtime uses, so the two
        tokenizers agree on the probe corpus.
        """
        parts: list[str] = []
        i = 0
        n = len(text)
        while i < n:
            mlen = 0

            # alt1: '(?i:[sdmt]|ll|ve|re)
            if text[i] == "'" and i + 1 < n:
                c1 = text[i + 1].lower()
                if c1 in "sdmt":
                    mlen = 2
                elif i + 2 < n and text[i + 2].lower() in "lver":
                    if (c1 == "l" and text[i + 2].lower() == "l") or \
                       (c1 == "v" and text[i + 2].lower() == "e") or \
                       (c1 == "r" and text[i + 2].lower() == "e"):
                        mlen = 3

            # alt2: [^\r\n\p{L}\p{N}]?\p{L}+
            if not mlen:
                j = i
                if j < n and not _is_qw_newline(text[j]) and not _is_qw_letter(text[j]) \
                        and not _is_qw_digit(text[j]):
                    if j + 1 < n and _is_qw_letter(text[j + 1]):
                        j += 1
                        while j < n and _is_qw_letter(text[j]):
                            j += 1
                        mlen = j - i
                if not mlen and _is_qw_letter(text[i]):
                    j = i
                    while j < n and _is_qw_letter(text[j]):
                        j += 1
                    mlen = j - i

            # alt3: \p{N}{1,3}
            if not mlen:
                j = i
                cnt = 0
                while j < n and cnt < 3 and _is_qw_digit(text[j]):
                    j += 1
                    cnt += 1
                if cnt:
                    mlen = j - i

            # alt4:  ?[^\s\p{L}\p{N}]+[\r\n]*
            if not mlen:
                j = i
                if j < n and text[j] == " ":
                    j += 1
                start = j
                while j < n and not _is_qw_space(text[j]) and not _is_qw_letter(text[j]) \
                        and not _is_qw_digit(text[j]):
                    j += 1
                if j > start:
                    while j < n and _is_qw_newline(text[j]):
                        j += 1
                    mlen = j - i

            # alt5: \s*[\r\n]+
            if not mlen:
                j = i
                while j < n and _is_qw_space(text[j]):
                    j += 1
                k = j
                while k > i and not _is_qw_newline(text[k - 1]):
                    k -= 1
                if k > i:
                    mlen = k - i

            # alt6: \s+(?!\S)
            if not mlen:
                j = i
                while j < n and _is_qw_space(text[j]):
                    j += 1
                if j > i:
                    if j >= n or _is_qw_space(text[j]):
                        mlen = j - i
                    elif j > i + 1:
                        mlen = j - i - 1

            # alt7: \s+
            if not mlen:
                j = i
                while j < n and _is_qw_space(text[j]):
                    j += 1
                if j > i:
                    mlen = j - i

            if not mlen:
                mlen = 1
            parts.append(text[i:i + mlen])
            i += mlen
        return parts


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
