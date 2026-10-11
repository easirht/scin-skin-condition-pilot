import numpy as np
import pandas as pd

SEED, N_BOOT = 42, 2000
R = "outputs/reports/"
folds = pd.read_csv("manifests/folds_v2.csv", dtype={"case_id": str, "group": str})
classes = sorted(folds.label.unique())
y = folds.label.values
o = pd.read_csv(R + "oof_v2_finetune_mobilenet.csv", dtype={"case_id": str}).set_index("case_id").loc[folds.case_id]
assert (o["true"].values == y).all()
P = o[classes].values
pred = np.array(classes)[P.argmax(1)]
conf = P.max(1)
K = len(classes)
ci = {c: i for i, c in enumerate(classes)}

cm = pd.crosstab(pd.Series(y, name="true"), pd.Series(pred, name="pred")).reindex(index=classes, columns=classes, fill_value=0)
rn = cm.div(cm.sum(1), axis=0)
print("row-normalised confusion matrix (recall by true class):")
print(rn.round(2).to_string())

rng = np.random.default_rng(SEED)
grp = {g: np.where(folds.group.values == g)[0] for g in folds.group.unique()}
gl = list(grp)
boots = [np.concatenate([grp[g] for g in rng.choice(gl, len(gl))]) for _ in range(N_BOOT)]
print("\nrecall with group-bootstrap 95% CI:")
for c in classes:
    r = np.array([(pred[b][y[b] == c] == c).mean() for b in boots if (y[b] == c).any()])
    print(f"  {c:30s} {(pred[y == c] == c).mean():.3f} [{np.percentile(r, 2.5):.3f}, {np.percentile(r, 97.5):.3f}]  (n={int((y == c).sum())})")

off = [(cm.loc[a, b], a, b) for a in classes for b in classes if a != b]
print("\nlargest off-diagonal cells (count, true -> predicted, share of true class):")
for n, a, b in sorted(off, reverse=True)[:8]:
    print(f"  {n:4d}  {a} -> {b}  ({n / cm.loc[a].sum():.2f})")

e, a_ = "Eczema", "Allergic Contact Dermatitis"
print(f"\npair {e} / {a_}:")
print(f"  {e} predicted as {a_}: {cm.loc[e, a_]}/{cm.loc[e].sum()} = {cm.loc[e, a_] / cm.loc[e].sum():.2f}")
print(f"  {a_} predicted as {e}: {cm.loc[a_, e]}/{cm.loc[a_].sum()} = {cm.loc[a_, e] / cm.loc[a_].sum():.2f}")
merge = lambda v: np.where(np.isin(v, [e, a_]), "EA", v)
ym, pm = merge(y), merge(pred)
mc = [c for c in np.unique(ym)]
bal = lambda yy, pp: np.mean([(pp[yy == c] == c).mean() for c in mc])
Bm = np.array([bal(ym[b], pm[b]) for b in boots])
print(f"  balanced accuracy if the two classes are merged (7 classes, descriptive only): "
      f"{bal(ym, pm):.3f} [{np.percentile(Bm, 2.5):.3f}, {np.percentile(Bm, 97.5):.3f}]  "
      f"(unmerged 8-class: {np.mean([(pred[y == c] == c).mean() for c in classes]):.3f})")

ok = pred == y
print(f"\ntop-label probability: correct n={ok.sum()} median {np.median(conf[ok]):.3f} | "
      f"incorrect n={(~ok).sum()} median {np.median(conf[~ok]):.3f}")
for q in (0.25, 0.5, 0.75):
    print(f"  quantile {q}: correct {np.quantile(conf[ok], q):.3f} | incorrect {np.quantile(conf[~ok], q):.3f}")