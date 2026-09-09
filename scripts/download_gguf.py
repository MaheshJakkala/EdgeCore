#!/usr/bin/env python3
"""Download Qwen2.5-0.5B-Instruct GGUF model from Hugging Face."""
import urllib.request
import os
import sys

MODEL_URL = "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-fp16.gguf"
OUTPUT_DIR = "/home/user/Documents/edgeCore19/workspace/models/baselines/qwen2.5-0.5b-instruct"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "qwen2.5-0.5b-instruct-fp16.gguf")

def download_with_progress(url, dest):
    """Download file with progress bar."""
    print(f"Downloading from {url}")
    print(f"Saving to {dest}")
    
    req = urllib.request.Request(url, headers={"User-Agent": "EdgeCore/0.1"})
    try:
        with urllib.request.urlopen(req) as response:
            total = int(response.headers.get("Content-Length", 0))
            done = 0
            with open(dest, "wb") as f:
                while True:
                    chunk = response.read(1 << 20)  # 1MB chunks
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    if total:
                        pct = done / total * 100
                        mb_done = done / 1e6
                        mb_total = total / 1e6
                        sys.stdout.write(f"\r  {mb_done:.1f}/{mb_total:.1f} MB ({pct:.1f}%)")
                        sys.stdout.flush()
            if total:
                print()
            print("Download complete!")
            return True
    except urllib.error.HTTPError as e:
        print(f"\nHTTP Error: {e.code} - {e.reason}")
        return False
    except Exception as e:
        print(f"\nError: {e}")
        return False

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if os.path.exists(OUTPUT_FILE):
        size_mb = os.path.getsize(OUTPUT_FILE) / 1e6
        expected_mb = 1266.4
        if size_mb >= expected_mb * 0.99:
            print(f"File already complete: {OUTPUT_FILE} ({size_mb:.1f} MB)")
            return
        else:
            print(f"File incomplete: {OUTPUT_FILE} ({size_mb:.1f} MB), continuing download...")
    
    success = download_with_progress(MODEL_URL, OUTPUT_FILE)
    if success:
        size_mb = os.path.getsize(OUTPUT_FILE) / 1e6
        print(f"Model saved: {OUTPUT_FILE} ({size_mb:.1f} MB)")
    else:
        print("Download failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()