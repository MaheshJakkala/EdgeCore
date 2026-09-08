#!/usr/bin/env python3
"""Download Qwen tokenizer files (not stored in GGUF) into a local HF dir.

GGUF checkpoints carry weights but no raw vocab/merges. The .ecm tokenizer
blobs and the C-tokenizer cross-check need the text tokenizer files, so pull
them (plus config.json, which stays as the .ecm metadata aid) from the HF
repo that matches the model:

    python3 scripts/fetch_tokenizer.py --name Qwen/Qwen2.5-0.5B \
        --out models/downloads/qwen2.5-0.5b
"""
import argparse
import os
import sys
import urllib.request

BASE = "https://huggingface.co"

FILES = ["config.json", "vocab.json", "merges.txt", "tokenizer.json",
         "tokenizer_config.json", "special_tokens_map.json",
         "generation_config.json"]

# vocab/merges/tokenizer_config feed the .ecm tokenizer blobs and the C
# tokenizer cross-check; the rest are optional metadata (some repos 404).
OPTIONAL = {"special_tokens_map.json", "generation_config.json"}


def download(url: str, dest: str, force: bool = False) -> bool:
    if os.path.exists(dest) and not force:
        print(f"  [skip] {dest}")
        return True
    print(f"  [get]  {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "EdgeCore/0.1"})
    try:
        with urllib.request.urlopen(req) as r, open(dest, "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
    except urllib.error.HTTPError as e:
        try:
            os.unlink(dest)
        except OSError:
            pass
        print(f"  [err] {e.code} for {url}")
        return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch Qwen tokenizer/config files")
    ap.add_argument("--name", default="Qwen/Qwen2.5-0.5B", help="HF repo id")
    ap.add_argument("--out", default="models/downloads/qwen2.5-0.5b",
                    help="output dir")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    missing = []
    for fn in FILES:
        url = f"{BASE}/{args.name}/resolve/main/{fn}"
        if not download(url, os.path.join(args.out, fn), force=args.force):
            if fn in OPTIONAL:
                print(f"  [warn] optional {fn} not present; continuing")
            else:
                missing.append(fn)
    if missing:
        print(f"\nERROR: missing {missing} from {args.name}", file=sys.stderr)
        sys.exit(1)
    print(f"\nDone. Tokenizer/config for {args.name} in {args.out}")


if __name__ == "__main__":
    main()
