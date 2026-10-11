import numpy as np
import pandas as pd

SEED, N_BOOT, MIN_N, MIN_CLS, MIN_TOTAL = 42, 2000, 5, 2, 30
R = "outputs/reports/"
d = pd.read_csv("manifests/v2_skintone_groups.csv", dtype={"case_id": str, "group": str})
classes = sorted(d.label.unique())
y = d.label.values


def load(name, probs):
    o = pd.read_csv(R + name, dtype={"case_id": str}).set_index("case_id").loc[d.case_id]
    assert (o["true"].values == y).all()
    return o[classes].idxmax(axis=1).values if probs else o["pred"].values


P = {"fine-tuned": load("oof_v2_finetune_mobilenet.csv", True),
     "meta_rich_w": load("oof_v2bal_meta_rich.csv", False),
     "majority": load("oof_v2_majority.csv", False)}


def bal(yy, pp, inc):
    rec = [(pp[yy == c] == c).mean() for c in inc if (yy == c).any()]
    return float(np.mean(rec))


rng = np.random.default_rng(SEED)
rows = []
for col in ("derm_fst_group", "monk_group"):
    print(f"\n===== {col} =====")
    for g in sorted(d[col].dropna().unique()):
        idx = np.where(d[col].values == g)[0]
        ys = y[idx]
        cnt = pd.Series(ys).value_counts()
        inc = sorted(cnt[cnt >= MIN_N].index)
        if len(idx) < MIN_TOTAL or len(inc) < MIN_CLS:
            print(f"{g}: n={len(idx)} too small to analyse")
            continue
        pg = {m: p[idx] for m, p in P.items()}
        gk = d.group.values[idx]
        grp = {k: np.where(gk == k)[0] for k in np.unique(gk)}
        gl = list(grp)
        bs = [np.concatenate([grp[k] for k in rng.choice(gl, len(gl))]) for _ in range(N_BOOT)]
        pt = {m: bal(ys, pg[m], inc) for m in pg}
        B = {m: np.array([bal(ys[b], pg[m][b], inc) for b in bs]) for m in pg}
        dd = B["fine-tuned"] - B["meta_rich_w"]
        ci = {m: np.percentile(B[m], [2.5, 97.5]) for m in pg}
        dlo, dhi = np.percentile(dd, [2.5, 97.5])
        excl = {c: int(n) for c, n in cnt.items() if n < MIN_N}
        print(f"\n{g}: n={len(idx)} | included classes ({len(inc)}): {inc}")
        print(f"  excluded (<{MIN_N} cases): {excl}")
        print(f"  fine-tuned accuracy (all {len(idx)} cases): {(pg['fine-tuned'] == ys).mean():.3f}")
        for m in pg:
            print(f"  balanced acc {m:12s} {pt[m]:.3f} [{ci[m][0]:.3f}, {ci[m][1]:.3f}]")
        print(f"  paired fine-tuned - meta_rich_w: {pt['fine-tuned'] - pt['meta_rich_w']:+.3f} [{dlo:+.3f}, {dhi:+.3f}]")
        rows.append({"grouping": col, "group": g, "n": len(idx), "k_classes": len(inc),
                     "acc_ft": round((pg['fine-tuned'] == ys).mean(), 3),
                     "bal_ft": round(pt["fine-tuned"], 3), "ft_lo": round(ci["fine-tuned"][0], 3),
                     "ft_hi": round(ci["fine-tuned"][1], 3),
                     "bal_meta": round(pt["meta_rich_w"], 3), "bal_majority": round(pt["majority"], 3),
                     "diff": round(pt["fine-tuned"] - pt["meta_rich_w"], 3),
                     "diff_lo": round(dlo, 3), "diff_hi": round(dhi, 3)})
pd.DataFrame(rows).to_csv(R + "subgroups_v2.csv", index=False)
print("\nsaved outputs/reports/subgroups_v2.csv")