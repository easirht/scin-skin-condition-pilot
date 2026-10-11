import hashlib
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path("data/raw/scin")
HAMMING = 6
img = pd.read_csv("manifests/v2_candidate_images.csv", dtype={"case_id": str})
paths = img.image_path.drop_duplicates().tolist()

rows = []
for i, p in enumerate(paths, 1):
    f = ROOT / p
    rec = {"image_path": p, "exists": f.exists()}
    if f.exists():
        rec["sha256"] = hashlib.sha256(f.read_bytes()).hexdigest()
        try:
            with Image.open(f) as im:
                im.load()
                rec["width"], rec["height"] = im.size
                rec["phash"] = str(imagehash.phash(im.convert("RGB")))
                rec["ok"] = True
        except Exception as e:
            rec["ok"] = False
            rec["error"] = str(e)
    rows.append(rec)
    if i % 500 == 0:
        print(f"  hashed {i}/{len(paths)}", flush=True)
aud = pd.DataFrame(rows)
assert aud.image_path.is_unique and len(aud) == len(paths)
aud.to_csv("manifests/v2_image_audit.csv", index=False)
print("files:", len(aud), "| exist:", int(aud.exists.sum()), "| opened OK:", int(aud.ok.fillna(False).sum()))
print(aud[["width", "height"]].describe().round(0).to_string())

full = img.merge(aud, on="image_path", how="left", validate="many_to_one")
cases = full.drop_duplicates("case_id")[["case_id", "label"]].reset_index(drop=True)
parent = {c: c for c in cases.case_id}


def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb:
        parent[rb] = ra


# exact duplicates: same bytes shared by different cases
ex = full.groupby("sha256").case_id.apply(lambda s: sorted(set(s)))
ex_multi = ex[ex.apply(len) > 1]
for cs in ex_multi:
    for c in cs[1:]:
        union(cs[0], c)
print("\nexact-duplicate byte-groups shared by >1 case:", len(ex_multi))
same_bytes_diff_path = int((aud.sha256.value_counts() > 1).sum())
print("byte-identical files stored under different paths:", same_bytes_diff_path)

# near duplicates (pHash), only pairs that differ in bytes and belong to different cases
ph = aud.dropna(subset=["phash"]).reset_index(drop=True)
bits = np.array([[int(b) for b in format(int(h, 16), "064b")] for h in ph.phash], dtype=np.uint8)
file_cases = full.groupby("image_path").case_id.apply(set).to_dict()
n_pairs = n_cross = 0
for s in range(0, len(ph), 256):
    d = (bits[s:s + 256, None, :] != bits[None, :, :]).sum(-1)
    for a, b in zip(*np.where(d <= HAMMING)):
        i = s + a
        if i >= b:
            continue
        if ph.sha256[i] == ph.sha256[b]:
            continue
        n_pairs += 1
        ci, cj = file_cases[ph.image_path[i]], file_cases[ph.image_path[b]]
        if ci != cj:
            n_cross += 1
            for x in ci | cj:
                union(next(iter(ci)), x)
print(f"near-duplicate pairs (pHash <= {HAMMING}, different bytes): {n_pairs} | across different cases: {n_cross}")

cases["group"] = cases.case_id.map(find)
sizes = cases.group.value_counts()
print("\ncases:", len(cases), "| groups:", cases.group.nunique(), "| largest group:", int(sizes.max()), "cases")
print("groups with >1 case:", int((sizes > 1).sum()))
lab_mix = cases.groupby("group").label.nunique()
print("groups whose cases carry DIFFERENT labels:", int((lab_mix > 1).sum()))
cases.to_csv("manifests/v2_groups.csv", index=False)
print("saved manifests/v2_groups.csv and manifests/v2_image_audit.csv")