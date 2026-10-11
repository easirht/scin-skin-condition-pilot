import matplotlib.pyplot as plt
import pandas as pd

c = pd.read_csv("outputs/reports/finetune_curves_mobilenet.csv")
fig, axes = plt.subplots(1, 5, figsize=(18, 3.2), sharey=True)
for ax, (f, g) in zip(axes, c.groupby("fold")):
    ax.plot(g.epoch, g.train_loss, label="train")
    ax.plot(g.epoch, g.val_loss, label="inner validation")
    ax.set_title(f"fold {f}")
    ax.set_xlabel("epoch")
axes[0].set_ylabel("class-weighted CE loss")
axes[0].legend()
plt.tight_layout()
plt.savefig("outputs/figures/finetune_curves_mobilenet.png", dpi=150)
print("saved outputs/figures/finetune_curves_mobilenet.png")