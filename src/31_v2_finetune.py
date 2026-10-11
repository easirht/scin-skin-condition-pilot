import copy
import os
import time
import warnings
from pathlib import Path

os.environ["TORCH_HOME"] = str(Path(".torch").resolve())

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torchvision import models
from torchvision import transforms as T

SEED, N_BOOT = 42, 2000
EPOCHS, BATCH, LR, WD = 12, 16, 1e-4, 1e-2
ROOT = Path("data/raw/scin")
FOLD_DIR = Path("outputs/reports/v2_ft_folds")
FOLD_DIR.mkdir(parents=True, exist_ok=True)
torch.manual_seed(SEED)
np.random.seed(SEED)

folds = pd.read_csv("manifests/folds_v2.csv", dtype={"case_id": str, "group": str})
cand = pd.read_csv("manifests/v2_candidate_images.csv", dtype={"case_id": str})
classes = sorted(folds.label.unique())
cid = {c: i for i, c in enumerate(classes)}
img = cand[["case_id", "image_path"]].merge(
    folds[["case_id", "label", "fold", "group"]], on="case_id", validate="many_to_one")
img["y"] = img.label.map(cid)
assert len(img) == 2560 and img.case_id.nunique() == 1172

cache = {}
for p in img.image_path.unique():
    with Image.open(ROOT / p) as im:
        im = im.convert("RGB")
        im.thumbnail((320, 320))
        cache[p] = im.copy()
print("cached", len(cache), "images", flush=True)

W = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
train_tf = T.Compose([T.RandomResizedCrop(224, scale=(0.7, 1.0)), T.RandomHorizontalFlip(),
                      T.ColorJitter(0.2, 0.2, 0.1), T.ToTensor(), T.Normalize(MEAN, STD)])
eval_tf = W.transforms()


def batches(rows, tf, shuffle):
    idx = np.random.permutation(len(rows)) if shuffle else np.arange(len(rows))
    for i in range(0, len(rows), BATCH):
        r = rows.iloc[idx[i:i + BATCH]]
        yield torch.stack([tf(cache[p]) for p in r.image_path]), torch.tensor(r.y.values)


def make_model():
    m = models.mobilenet_v3_small(weights=W)
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, len(classes))
    return m


def epoch_loss(m, rows, lossf):
    m.eval()
    tot = 0.0
    with torch.no_grad():
        for x, y in batches(rows, eval_tf, False):
            tot += lossf(m(x), y).item() * len(y)
    return tot / len(rows)


oof, curves = [], []
for f in sorted(folds.fold.unique()):
    done = FOLD_DIR / f"fold{f}_probs.csv"
    if done.exists():  # resume: a finished fold is never recomputed
        oof.append(pd.read_csv(done, dtype={"case_id": str}).set_index("case_id"))
        print(f"fold {f}: loaded from disk", flush=True)
        continue
    t0 = time.time()
    tr_cases = folds[folds.fold != f].reset_index(drop=True)
    _, vi = next(StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
                 .split(tr_cases, tr_cases.label, tr_cases.group))
    val_ids = set(tr_cases.iloc[vi].case_id)
    tr = img[(img.fold != f) & ~img.case_id.isin(val_ids)].reset_index(drop=True)
    va = img[(img.fold != f) & img.case_id.isin(val_ids)].reset_index(drop=True)
    te = img[img.fold == f].reset_index(drop=True)
    assert not set(tr.case_id) & set(va.case_id) and not set(te.case_id) & set(tr.case_id)

    cnt = np.bincount(tr.y.values, minlength=len(classes)).astype(float)
    lossf = nn.CrossEntropyLoss(weight=torch.tensor(cnt.sum() / (len(classes) * cnt), dtype=torch.float32))
    model = make_model()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    best, state, best_ep = 1e9, None, 0
    for ep in range(1, EPOCHS + 1):
        model.train()
        tl, n = 0.0, 0
        for x, y in batches(tr, train_tf, True):
            opt.zero_grad()
            loss = lossf(model(x), y)
            loss.backward()
            opt.step()
            tl += loss.item() * len(y)
            n += len(y)
        vl = epoch_loss(model, va, lossf)
        curves.append({"fold": f, "epoch": ep, "train_loss": tl / n, "val_loss": vl})
        if vl < best:
            best, state, best_ep = vl, copy.deepcopy(model.state_dict()), ep
        print(f"  fold {f} epoch {ep}: train {tl / n:.3f} val {vl:.3f}", flush=True)
    model.load_state_dict(state)
    model.eval()
    probs = []
    with torch.no_grad():
        for x, _ in batches(te, eval_tf, False):
            probs.append(torch.softmax(model(x), 1).numpy())
    P = pd.DataFrame(np.concatenate(probs), columns=classes)
    P["case_id"] = te.case_id.values
    Pc = P.groupby("case_id").mean()
    Pc.to_csv(done)
    oof.append(Pc)
    pd.DataFrame(curves).to_csv(FOLD_DIR / f"fold{f}_curves.csv", index=False)
    curves = []
    print(f"fold {f}: best epoch {best_ep}, val loss {best:.3f}, {time.time() - t0:.0f}s", flush=True)

P = pd.concat(oof).loc[folds.case_id]
P.assign(true=folds.label.values).to_csv("outputs/reports/oof_v2_finetune_mobilenet.csv")
y, pred = folds.label.values, P.idxmax(axis=1).values
group = folds.group.values


def stat(yy, pp):
    rec = recall_score(yy, pp, labels=classes, average=None, zero_division=0)
    return np.r_[accuracy_score(yy, pp), balanced_accuracy_score(yy, pp),
                 f1_score(yy, pp, labels=classes, average="macro", zero_division=0), rec]


rng = np.random.default_rng(SEED)
grp = {g: np.where(group == g)[0] for g in np.unique(group)}
gl = list(grp)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    B = np.array([stat(y[b], pred[b]) for b in
                  (np.concatenate([grp[g] for g in rng.choice(gl, len(gl))]) for _ in range(N_BOOT))])
lo, hi = np.percentile(B, [2.5, 97.5], axis=0)
names = ["accuracy", "balanced_acc", "macro_F1"] + [f"recall_{c}" for c in classes]
print(f"\n=== fine-tuned MobileNetV3-Small, v2 (pooled out-of-fold, n={len(y)} cases) ===")
for n, p, a, b in zip(names, stat(y, pred), lo, hi):
    print(f"  {n:42s} {p:.3f}  [{a:.3f}, {b:.3f}]")
print(pd.crosstab(pd.Series(y, name="true"), pd.Series(pred, name="pred")).to_string())
print(f"claim rule (CI lower bound > 0.179): {'PASS' if lo[1] > 0.179 else 'FAIL'} (lower bound {lo[1]:.3f}) "
      f"| above richer-metadata upper bound 0.232: {'yes' if lo[1] > 0.232 else 'no'}")