#!/usr/bin/env python3
"""Download a GPT-2 style HF model (safetensors + tokenizer) into a local dir."""
import argparse
import json
import os
import sys
import urllib.request

BASE = "https://huggingface.co/openai-community/gpt2/resolve/main"
FILES = ["config.json", "model.safetensors", "vocab.json", "merges.txt",
         "tokenizer_config.json"]


def download(url: str, dest: str, force: bool = False) -> None:
    if os.path.exists(dest) and not force:
        print(f"  [skip] {dest}")
        return
    print(f"  [get]  {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "EdgeCore/0.1"})
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch a GPT-2 model from HF")
    ap.add_argument("--name", default="gpt2", help="HF model id (default gpt2)")
    ap.add_argument("--out", default="models/downloads/gpt2", help="output dir")
    ap.add_argument("--force", action="store_true",
                    help="re-download files that already exist")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cfg = urllib.request.urlopen(
        urllib.request.Request(f"{BASE.replace('gpt2', args.name)}/config.json",
                               headers={"User-Agent": "EdgeCore/0.1"})).read()
    model_name = json.loads(cfg).get("_name_or_path", args.name)
    base = BASE.replace("gpt2", model_name)
    for fn in FILES:
        download(f"{base}/{fn}", os.path.join(args.out, fn), force=args.force)
    print(f"\nDone. Model in {args.out}")


if __name__ == "__main__":
    main()
