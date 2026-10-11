import sys
import urllib.request
from pathlib import Path

import pandas as pd

BASE = "https://storage.googleapis.com/dx-scin-public-data/"
OUT = Path("data/raw/scin")  # keeps the original path layout: data/raw/scin/dataset/images/...
DOWNLOAD = "--download" in sys.argv

img = pd.read_csv("manifests/scin_candidate_images.csv")
print("images in manifest:", len(img), "| mode:", "DOWNLOAD" if DOWNLOAD else "DRY-RUN")

total, missing, failed = 0, [], []
for i, p in enumerate(img.image_path, 1):
    url = BASE + p
    dest = OUT / p
    try:
        if DOWNLOAD:
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                urllib.request.urlretrieve(url, dest)
            total += dest.stat().st_size
        else:
            req = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(req) as r:
                total += int(r.headers.get("Content-Length", 0))
    except Exception as e:
        (missing if "404" in str(e) else failed).append((p, str(e)))
    if i % 50 == 0:
        print(f"  {i}/{len(img)}")

print(f"\nTotal size: {total/1e6:.1f} MB")
print("missing (404):", len(missing), "| other failures:", len(failed))
Path("outputs/reports").mkdir(parents=True, exist_ok=True)
with open("outputs/reports/download_problems.txt", "w") as f:
    for p, e in missing + failed:
        f.write(f"{p}\t{e}\n")