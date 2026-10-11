import shutil
import time
import urllib.request
from pathlib import Path

import pandas as pd
from PIL import Image

BASE = "https://storage.googleapis.com/dx-scin-public-data/"
OUT = Path("data/raw/scin")
paths = pd.read_csv("manifests/v2_candidate_images.csv").image_path.drop_duplicates().tolist()

for leftover in OUT.rglob("*.part"):
    leftover.unlink()


def valid(f):
    try:
        with Image.open(f) as im:
            im.verify()
        return True
    except Exception:
        return False


todo, bad = [], 0
for p in paths:
    f = OUT / p
    if f.exists():
        if valid(f):
            continue
        f.unlink()  # truncated or corrupt from an interrupted run
        bad += 1
    todo.append(p)
print(f"unique files: {len(paths)} | valid already: {len(paths) - len(todo)} | "
      f"removed corrupt: {bad} | to download: {len(todo)}", flush=True)


def fetch(p):
    dest = OUT / p
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    err = ""
    for _ in range(3):
        try:
            with urllib.request.urlopen(BASE + p, timeout=30) as r, open(tmp, "wb") as fh:
                shutil.copyfileobj(r, fh)
            tmp.replace(dest)  # appears under the real name only when complete
            return ""
        except Exception as e:
            err = str(e)
            time.sleep(2)
    if tmp.exists():
        tmp.unlink()
    return err


failed = []
for i, p in enumerate(todo, 1):
    e = fetch(p)
    if e:
        failed.append((p, e))
    if i % 100 == 0:
        print(f"  {i}/{len(todo)}", flush=True)

ok = sum(valid(OUT / p) for p in paths if (OUT / p).exists())
print(f"\nvalid images now: {ok}/{len(paths)} | failed: {len(failed)}")
Path("outputs/reports").mkdir(parents=True, exist_ok=True)
Path("outputs/reports/v2_download_problems.txt").write_text("\n".join(f"{p}\t{e}" for p, e in failed))