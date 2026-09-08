#!/usr/bin/env python3
"""Download an HF model (safetensors + tokenizer) into a local dir.

Supports both GPT-2 style and Qwen2 style checkpoints (any repo id):
    python3 scripts/fetch_model.py --name Qwen/Qwen2.5-0.5B-Instruct --out models/downloads/qwen2
    python3 scripts/fetch_model.py --name gpt2 --out models/downloads/gpt2
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request

BASE = "https://huggingface.co"

REQUIRED_FILES = ["config.json", "model.safetensors"]
OPTIONAL_FILES = ["vocab.json", "merges.txt", "tokenizer_config.json",
                  "tokenizer.json", "special_tokens_map.json",
                  "generation_config.json"]


def download(url: str, dest: str, force: bool = False) -> bool:
    if os.path.exists(dest) and not force:
        print(f"  [skip] {dest}")
        return True
    print(f"  [get]  {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "EdgeCore/0.1"})
    try:
        with urllib.request.urlopen(req) as r, open(dest, "wb") as f:
            total = int(r.headers.get("Content-Length", 0))
            done = 0
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if total:
                    sys.stdout.write(f"\r  {done/1e6:.0f}/{total/1e6:.0f} MB")
                    sys.stdout.flush()
            if total:
                print()
    except urllib.error.HTTPError as e:
        try:
            os.unlink(dest)
        except OSError:
            pass
        print(f"  [404] {e.code} for {url}")
        return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch a GPT-2 or Qwen2 model from HF")
    ap.add_argument("--name", default="gpt2",
                    help="HF repo id, e.g. Qwen/Qwen2.5-0.5B-Instruct (default gpt2)")
    ap.add_argument("--out", default="models/downloads/gpt2", help="output dir")
    ap.add_argument("--force", action="store_true",
                    help="re-download files that already exist")
    args = ap.parse_args()

    model_id = args.name if "/" in args.name else f"openai-community/{args.name}"
    os.makedirs(args.out, exist_ok=True)

    missing_required = []
    for fn in REQUIRED_FILES:
        url = f"{BASE}/{model_id}/resolve/main/{fn}"
        if not download(url, os.path.join(args.out, fn), force=args.force):
            missing_required.append(fn)

    if missing_required:
        print(f"\nERROR: missing required files {missing_required} for {model_id}",
              file=sys.stderr)
        sys.exit(1)

    for fn in OPTIONAL_FILES:
        download(f"{BASE}/{model_id}/resolve/main/{fn}",
                 os.path.join(args.out, fn), force=args.force)

    cfg = json.load(open(os.path.join(args.out, "config.json")))
    model_name = cfg.get("_name_or_path", model_id)
    print(f"\nDone. Model in {args.out} ({model_name})")


if __name__ == "__main__":
    main()
