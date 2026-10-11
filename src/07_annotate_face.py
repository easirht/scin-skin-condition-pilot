import pandas as pd

CSV = "manifests/face_review_manual.csv"
KEYS = {"f": "face", "p": "partial_face", "n": "no_face", "u": "unclear"}

d = pd.read_csv(CSV, dtype=str, keep_default_na=False)
todo = d.index[d.judgement.str.strip() == ""].tolist()
print(f"{len(todo)} left. Keys: f=face  p=partial_face  n=no_face  u=unclear  q=save&quit")
print("Look at the number on the sheet image, then type a key here.\n")

for i in todo:
    r = d.loc[i]
    sheet = 1 if int(r.review_id) <= 20 else 2
    while True:
        k = input(f"Photo #{r.review_id} (in sheet {sheet}) > ").strip().lower()
        if k in KEYS or k == "q":
            break
        print("  type only: f / p / n / u / q")
    if k == "q":
        break
    d.loc[i, "judgement"] = KEYS[k]
    d.to_csv(CSV, index=False)

print(d.judgement.replace("", "(empty)").value_counts())