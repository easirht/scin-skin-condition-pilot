import re
import shutil
from pathlib import Path

import pandas as pd

RAW = Path("data/raw/scin")
OUT = Path("data/processed/v2_by_class")  # derived copy; raw files are never touched
img = pd.read_csv("manifests/v2_candidate_images.csv", dtype={"case_id": str})


def safe(s):
    return re.sub(r'[<>:"/\\|?*]', "_", s).strip()


made, missing = 0, []
for r in img.itertuples():
    src = RAW / r.image_path
    if not src.exists():
        missing.append(r.image_path)
        continue
    d = OUT / safe(r.label)
    d.mkdir(parents=True, exist_ok=True)
    dst = d / f"c{r.case_id}_img{r.image_idx}.png"  # case id + slot, so the file maps back to the manifest
    if not dst.exists():
        shutil.copy2(src, dst)
        made += 1

print("rows:", len(img), "| copied:", made, "| missing source files:", len(missing))
for lab, n in img.groupby("label").size().items():
    d = OUT / safe(lab)
    got = len(list(d.glob("*.png"))) if d.exists() else 0
    print(f"  {lab:30s} manifest {n:5d} | folder {got:5d} | {'OK' if got == n else 'MISMATCH'}")