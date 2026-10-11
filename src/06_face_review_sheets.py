import random
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

SEED, N, PER_SHEET, COLS, T = 42, 40, 20, 5, 260
ROOT = Path("data/raw/scin")
OUTD = Path("outputs/figures")
OUTD.mkdir(parents=True, exist_ok=True)

df = pd.read_csv("manifests/scin_candidate_images_audit.csv")
df = df[df.ok == True].reset_index(drop=True)

random.seed(SEED)
idx = sorted(random.sample(range(len(df)), N))
sample = df.loc[idx, ["image_path", "case_id", "shot_type"]].reset_index(drop=True)
sample.insert(0, "review_id", range(1, N + 1))
sample["judgement"] = ""  # you fill: face / partial_face / no_face / unclear
sample.to_csv("manifests/face_review_manual.csv", index=False)

for s in range(0, N, PER_SHEET):
    chunk = sample.iloc[s:s + PER_SHEET]
    rows = (len(chunk) + COLS - 1) // COLS
    sheet = Image.new("RGB", (COLS * T, rows * (T + 24)), "white")
    d = ImageDraw.Draw(sheet)
    for k, r in enumerate(chunk.itertuples()):
        with Image.open(ROOT / r.image_path) as im:
            im = im.convert("RGB")
            im.thumbnail((T, T))
            x, y = (k % COLS) * T, (k // COLS) * (T + 24)
            sheet.paste(im, (x, y + 24))
            d.text((x + 6, y + 6), f"#{r.review_id}", fill="black")
    out = OUTD / f"face_review_sheet_{s // PER_SHEET + 1}.png"
    sheet.save(out)
    print("saved", out)
print("annotation template: manifests/face_review_manual.csv")