import pandas as pd

V = {"face", "partial_face", "no_face", "unclear"}
full = pd.read_csv("manifests/face_review_full.csv", dtype=str, keep_default_na=False)
rep = full.is_repeat.str.lower().eq("true")
v1 = pd.read_csv("manifests/face_review_manual_annotated_v1.csv", dtype=str, keep_default_na=False)
v1 = v1[v1.judgement.isin(V)].drop_duplicates("image_path")

j = pd.concat([v1[["image_path", "judgement"]], full[~rep][["image_path", "judgement"]]])
assert j.image_path.is_unique

aud = pd.read_csv("manifests/scin_candidate_images_audit.csv", dtype={"case_id": str})
m = aud.merge(j, on="image_path", how="left", validate="many_to_one")
assert m.judgement.notna().all()

defs = {
    "primary (face+partial)": {"face", "partial_face"},
    "lenient (+unclear)": {"face", "partial_face", "unclear"},
    "strict (face only)": {"face"},
}
rows = {}
for name, S in defs.items():
    c = m.assign(f=m.judgement.isin(S)).groupby("case_id").agg(label=("label", "first"), f=("f", "any"))
    rows[name] = c.groupby("label").f.sum().astype(int)
tab = pd.DataFrame(rows)
print("facial cases per class under three definitions:")
print(tab.to_string())
for name in tab:
    kept = tab.index[tab[name] >= 15].tolist()
    print(f"{name}: classes with >=15 facial cases = {kept} | fallback (<3 classes): {len(kept) < 3}")