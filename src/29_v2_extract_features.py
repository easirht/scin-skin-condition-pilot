import os
from pathlib import Path

os.environ["TORCH_HOME"] = str(Path(".torch").resolve())

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torchvision import models

torch.manual_seed(42)
ROOT = Path("data/raw/scin")
OUT = Path("data/interim")
paths = pd.read_csv("manifests/v2_candidate_images.csv").image_path.drop_duplicates().tolist()
missing = [p for p in paths if not (ROOT / p).exists()]
assert not missing, f"{len(missing)} files missing"
print("unique images:", len(paths), flush=True)
pd.DataFrame({"image_path": paths}).to_csv(OUT / "v2_features_image_paths.csv", index=False)

SPECS = {
    "mobilenet_v3_small": (models.mobilenet_v3_small, models.MobileNet_V3_Small_Weights.IMAGENET1K_V1, "classifier"),
    "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1, "fc"),
    "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.IMAGENET1K_V1, "classifier"),
}
for name, (ctor, w, head) in SPECS.items():
    f_out = OUT / f"v2_features_{name}.npy"
    if f_out.exists():
        print(name, "already done, skipping")
        continue
    model = ctor(weights=w)
    setattr(model, head, nn.Identity())
    model.eval()
    tf = w.transforms()
    feats = []
    with torch.no_grad():
        for i in range(0, len(paths), 32):
            batch = []
            for p in paths[i:i + 32]:
                with Image.open(ROOT / p) as im:
                    batch.append(tf(im.convert("RGB")))
            feats.append(model(torch.stack(batch)).numpy())
            if (i // 32) % 10 == 0:
                print(f"  {name}: {min(i + 32, len(paths))}/{len(paths)}", flush=True)
    X = np.concatenate(feats)
    assert X.shape[0] == len(paths) and np.isfinite(X).all()
    np.save(f_out, X)
    print(f"{name}: features {X.shape}", flush=True)
print("done")