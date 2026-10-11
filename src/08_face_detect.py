import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

ROOT = Path("data/raw/scin")
MAX_SIDE = 640
PARAMS = dict(scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))  # fixed in advance, not tuned

frontal = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
profile = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")
assert not frontal.empty() and not profile.empty()


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(c - h, 2), round(c + h, 2)


cand = pd.read_csv("manifests/scin_candidate_images.csv")
paths = cand.image_path.drop_duplicates().tolist()
print("unique image files:", len(paths))

rows = []
for p in paths:
    im = Image.open(ROOT / p).convert("L")
    s = MAX_SIDE / max(im.size)
    if s < 1:
        im = im.resize((int(im.width * s), int(im.height * s)))
    g = np.array(im)
    nf = len(frontal.detectMultiScale(g, **PARAMS))
    npf = len(profile.detectMultiScale(g, **PARAMS)) + len(profile.detectMultiScale(cv2.flip(g, 1), **PARAMS))
    rows.append({"image_path": p, "n_frontal": nf, "n_profile": npf, "det_pos": (nf + npf) > 0})
det = pd.DataFrame(rows)
assert det.image_path.is_unique
det.to_csv("manifests/face_detect_haar.csv", index=False)
print("detector-positive images:", int(det.det_pos.sum()), "/", len(det))

# --- agreement with the manual reference (unique images, 'unclear' excluded) ---
ref = pd.read_csv("manifests/face_review_manual_annotated_v1.csv").drop_duplicates("image_path")
ref = ref[ref.judgement != "unclear"].merge(det, on="image_path", validate="one_to_one")
ref["truth_pos"] = ref.judgement.isin(["face", "partial_face"])
tp = int((ref.truth_pos & ref.det_pos).sum()); fn = int((ref.truth_pos & ~ref.det_pos).sum())
fp = int((~ref.truth_pos & ref.det_pos).sum()); tn = int((~ref.truth_pos & ~ref.det_pos).sum())
print(f"\nreference n={len(ref)} | TP {tp}  FN {fn}  FP {fp}  TN {tn}")
print("sensitivity (face+partial):", round(tp / max(tp + fn, 1), 2), "CI", wilson(tp, tp + fn))
print("specificity               :", round(tn / max(tn + fp, 1), 2), "CI", wilson(tn, tn + fp))
for j in ("face", "partial_face"):
    s = ref[ref.judgement == j]
    print(f"  detected among '{j}': {int(s.det_pos.sum())}/{len(s)}")

# --- lower-bound case counts per class if only detector-positive images are kept ---
m = cand.merge(det[["image_path", "det_pos"]], on="image_path", how="left", validate="many_to_one")
print("\ncases with >=1 detector-positive image (lower bound; partial faces missed):")
print(m.groupby("label").apply(lambda x: x[x.det_pos].case_id.nunique()).to_string())