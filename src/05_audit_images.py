import hashlib
from pathlib import Path

import imagehash
import pandas as pd
from PIL import Image

ROOT = Path("data/raw/scin")
img = pd.read_csv("manifests/scin_candidate_images.csv")

rows = []
for r in img.itertuples():
    p = ROOT / r.image_path
    rec = {"image_path": r.image_path, "exists": p.exists()}
    if p.exists():
        rec["bytes"] = p.stat().st_size
        rec["sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
        try:
            with Image.open(p) as im:
                im.load()
                rec["width"], rec["height"] = im.size
                rec["mode"] = im.mode
                rec["phash"] = str(imagehash.phash(im.convert("RGB")))
                rec["ok"] = True
        except Exception as e:
            rec["ok"] = False
            rec["error"] = str(e)
    rows.append(rec)

aud = pd.DataFrame(rows)
# rows are in the same order as img: join by position, NOT by image_path
# (merging on image_path multiplies rows when one file appears in several slots/cases)
full = pd.concat([img.reset_index(drop=True),
                  aud.drop(columns=["image_path"]).reset_index(drop=True)], axis=1)
assert len(full) == len(img), (len(full), len(img))
full.to_csv("manifests/scin_candidate_images_audit.csv", index=False)

print("images:", len(full))
print("exist:", int(full.exists.sum()), "| opened OK:", int(full.ok.fillna(False).sum()))
print("\nwidth/height summary:")
print(full[["width", "height"]].describe().round(0).to_string())
print("\nmodes:", full["mode"].value_counts().to_dict())
print("shot types:", full["shot_type"].value_counts(dropna=False).to_dict())

# exact duplicates
dup = full[full.duplicated("sha256", keep=False)]
print("\nexact-duplicate images (same bytes):", len(dup),
      "| groups:", dup.sha256.nunique())
if len(dup):
    print("duplicates crossing different cases:",
          int((dup.groupby("sha256").case_id.nunique() > 1).sum()), "groups")

# near duplicates (perceptual hash Hamming distance <= 6)
ph = full.dropna(subset=["phash"]).reset_index(drop=True)
hashes = [imagehash.hex_to_hash(h) for h in ph.phash]
pairs = []
for i in range(len(hashes)):
    for j in range(i + 1, len(hashes)):
        d = hashes[i] - hashes[j]
        if d <= 6 and ph.sha256[i] != ph.sha256[j]:
            pairs.append((ph.case_id[i], ph.case_id[j], ph.label[i], ph.label[j], int(d)))
print("\nnear-duplicate pairs (phash dist<=6, not byte-identical):", len(pairs))
cross = [p for p in pairs if p[0] != p[1]]
print("  of which across DIFFERENT cases:", len(cross))
for p in cross[:15]:
    print("  ", p)