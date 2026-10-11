import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

SEED, K = 42, 5
g = pd.read_csv("manifests/v2_groups.csv", dtype={"case_id": str, "group": str})
cases = pd.read_csv("manifests/v2_cases.csv", dtype={"case_id": str})
g = g.merge(cases[["case_id"]], on="case_id", validate="one_to_one")  # same case set as the final manifest
print("cases:", len(g), "| groups:", g.group.nunique())
print(g.label.value_counts().to_string())

g["fold"] = -1
for f, (_, te) in enumerate(StratifiedGroupKFold(n_splits=K, shuffle=True, random_state=SEED)
                            .split(g, g.label, groups=g.group)):
    g.iloc[te, g.columns.get_loc("fold")] = f

assert (g.fold >= 0).all()
assert (g.groupby("group").fold.nunique() == 1).all(), "a duplicate-linked group spans several folds"
g.to_csv("manifests/folds_v2.csv", index=False)
print("\ncases per fold x class:")
print(pd.crosstab(g.label, g.fold).to_string())
print("\nsaved manifests/folds_v2.csv (seed", SEED, ")")