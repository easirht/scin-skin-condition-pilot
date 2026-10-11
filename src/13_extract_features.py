import os
from pathlib import Path

os.environ["TORCH_HOME"] = str(Path(".torch").resolve())  # keep weights on E:, not C:

import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from torch import nn
from torchvision import models

torch.manual_seed(42)
ROOT = Path("data/raw/scin")
OUT = Path("data/interim")
OUT.mkdir(parents=True, exist_ok=True)

cand = pd.read_csv("manifests/scin_candidate_images.csv")
paths = cand.image_path.drop_duplicates().tolist()
print("unique images:", len(paths))
pd.DataFrame({"image_path": paths}).to_csv(OUT / "features_image_paths.csv", index=False)

SPECS = {
    "mobilenet_v3_small": (models.mobilenet_v3_small, models.MobileNet_V3_Small_Weights.IMAGENET1K_V1, "classifier"),
    "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1, "fc"),
    "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.IMAGENET1K_V1, "classifier"),
}

prov = [f"torch {torch.__version__} | torchvision {torchvision.__version__}", ""]
for name, (ctor, w, head) in SPECS.items():
    model = ctor(weights=w)
    setattr(model, head, nn.Identity())  # drop the ImageNet classification head
    model.eval()
    tf = w.transforms()
    prov += [f"{name}: {w}", f"  url: {w.url}", f"  transforms: {tf}", ""]

    feats = []
    with torch.no_grad():
        for i in range(0, len(paths), 16):
            batch = []
            for p in paths[i:i + 16]:
                with Image.open(ROOT / p) as im:
                    batch.append(tf(im.convert("RGB")))
            feats.append(model(torch.stack(batch)).numpy())
    X = np.concatenate(feats)
    assert X.shape[0] == len(paths) and np.isfinite(X).all()
    np.save(OUT / f"features_{name}.npy", X)
    print(f"{name}: features {X.shape}")

Path("outputs/reports").mkdir(parents=True, exist_ok=True)
Path("outputs/reports/pretrained_provenance.txt").write_text("\n".join(prov), encoding="utf-8")
print("saved features to", OUT, "and provenance log")