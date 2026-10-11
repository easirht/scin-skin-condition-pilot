import json
import urllib.request
from pathlib import Path

BUCKET = "dx-scin-public-data"
LIST_URL = f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o?prefix=dataset/&delimiter=/"
OUT = Path("data/raw/scin")
OUT.mkdir(parents=True, exist_ok=True)

# 1) List objects under dataset/ via the public JSON API (no login needed)
with urllib.request.urlopen(LIST_URL) as r:
    listing = json.load(r)

items = listing.get("items", [])
print("Objects found under dataset/:")
for it in items:
    print(" ", it["name"], it.get("size"), "bytes")

# 2) Download only the CSV files
for it in items:
    name = it["name"]
    if name.endswith(".csv"):
        url = f"https://storage.googleapis.com/{BUCKET}/{name}"
        dest = OUT / Path(name).name
        urllib.request.urlretrieve(url, dest)
        print("Saved:", dest, dest.stat().st_size, "bytes")