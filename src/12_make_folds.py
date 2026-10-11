import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

SEED, K = 42, 5
case = pd.read_csv("manifests/face_review_final.csv", dtype={"case_id": str, "group": str})
print("cases:", len(case), "| groups:", case.group.nunique())
print(case.label.value_counts().to_string())

sgkf = StratifiedGroupKFold(n_splits=K, shuffle=True, random_state=SEED)
case["fold"] = -1
for f, (_, te) in enumerate(sgkf.split(case, case.label, groups=case.group)):
    case.iloc[te, case.columns.get_loc("fold")] = f

# checks: every case assigned, no duplicate-linked group split across folds
assert (case.fold >= 0).all()
assert (case.groupby("group").fold.nunique() == 1).all(), "a group spans several folds"

case.to_csv("manifests/folds_headneck_v1.csv", index=False)
print("\ncases per fold x class:")
print(pd.crosstab(case.label, case.fold).to_string())
print("\nfacial cases per fold x class:")
print(pd.crosstab(case[case.facial].label, case[case.facial].fold).to_string())
print("\nsaved manifests/folds_headneck_v1.csv  (seed", SEED, ")")