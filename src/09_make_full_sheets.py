import math
import random
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

SEED = 2026
PER_SHEET, COLS, T = 20, 4, 320
N_REPEAT = 12
ROOT = Path("data/raw/scin")
OUTD = Path("outputs/figures")
OUTD.mkdir(parents=True, exist_ok=True)
CSV = Path("manifests/face_review_full.csv")
if CSV.exists():
    raise SystemExit(f"{CSV} already exists. Not overwriting your annotations.")

cand = pd.read_csv("manifests/scin_candidate_images.csv")
uniq = cand.image_path.drop_duplicates().tolist()

ref = pd.read_csv("manifests/face_review_manual_annotated_v1.csv").drop_duplicates("image_path")
ref = ref[ref.judgement.isin(["face", "partial_face", "no_face", "unclear"])]
ref_j = dict(zip(ref.image_path, ref.judgement))
todo = [p for p in uniq if p not in ref_j]
print("unique images:", len(uniq), "| already judged:", len(ref_j), "| to annotate:", len(todo))

rng = random.Random(SEED)
reps = rng.sample(sorted(ref_j), N_REPEAT)
items = [(p, False, "") for p in todo] + [(p, True, ref_j[p]) for p in reps]
rng.shuffle(items)

n_sheets = math.ceil(len(items) / PER_SHEET)
rows = []
for k, (p, rep, v1) in enumerate(items):
    rows.append({"item_id": k + 1, "sheet": k // PER_SHEET + 1, "pos": k % PER_SHEET + 1,
                 "image_path": p, "is_repeat": rep, "v1_judgement": v1, "judgement": ""})
df = pd.DataFrame(rows)
df.to_csv(CSV, index=False)

font = ImageFont.load_default(size=24)
for s in range(1, n_sheets + 1):
    chunk = df[df.sheet == s]
    nrows = math.ceil(len(chunk) / COLS)
    sheet = Image.new("RGB", (COLS * T, nrows * T), "white")
    d = ImageDraw.Draw(sheet)
    for r in chunk.itertuples():
        i = r.pos - 1
        x, y = (i % COLS) * T, (i // COLS) * T
        with Image.open(ROOT / r.image_path) as im:
            im = im.convert("RGB")
            im.thumbnail((T - 6, T - 6))
            sheet.paste(im, (x + 3, y + 3))
        d.rectangle([x, y, x + T - 1, y + T - 1], outline="gray")
        d.rectangle([x + 3, y + 3, x + 44, y + 34], fill="white")
        d.text((x + 8, y + 5), str(r.pos), fill="black", font=font)
    sheet.save(OUTD / f"face_full_sheet_{s:02d}.png")
print("sheets:", n_sheets, "| images per sheet:", PER_SHEET, "| repeats included:", N_REPEAT)
print("annotation file:", CSV)