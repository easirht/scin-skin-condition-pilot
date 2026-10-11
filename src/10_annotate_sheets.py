import pandas as pd

CSV = "manifests/face_review_full.csv"
KEYS = {"f": "face", "p": "partial_face", "n": "no_face", "u": "unclear"}
COLS = 4

d = pd.read_csv(CSV, dtype=str, keep_default_na=False)
d["sheet"] = d["sheet"].astype(int)
d["pos"] = d["pos"].astype(int)
sheets = sorted(d.sheet.unique())
done = [s for s in sheets if (d.loc[d.sheet == s, "judgement"] != "").all()]
print(f"{len(done)}/{len(sheets)} sheets done.  f=face  p=partial_face  n=no_face  u=unclear")

stop = False
for s in sheets:
    if s in done:
        continue
    m = (d.sheet == s)
    n = int(m.sum())
    while True:
        print(f"\nSheet {s}/{len(sheets)}: outputs/figures/face_full_sheet_{s:02d}.png  "
              f"({n} images, {COLS} per row, left to right, top to bottom)")
        raw = input(f"type {n} letters (spaces allowed), q = save & quit > ").strip().lower()
        if raw == "q":
            stop = True
            break
        t = raw.replace(" ", "")
        if len(t) != n or any(c not in KEYS for c in t):
            print(f"  need exactly {n} letters from f/p/n/u (you typed {len(t)})")
            continue
        for i in range(0, n, COLS):
            print("  " + "   ".join(f"{i + j + 1}:{t[i + j]}" for j in range(min(COLS, n - i))))
        if input("  Enter = confirm, r = redo > ").strip().lower() == "r":
            continue
        d.loc[m.values, "judgement"] = [KEYS[c] for c in t]  # d is ordered by sheet, then pos
        d.to_csv(CSV, index=False)
        print("  saved")
        break
    if stop:
        break

print("\nsheets done:", int((d.groupby("sheet").judgement.apply(lambda x: (x != "").all())).sum()), "/", len(sheets))