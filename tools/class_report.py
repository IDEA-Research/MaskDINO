#!/usr/bin/env python
"""Per-class precision / recall / F1 + confusion matrix for a cropcls run.

Recall alone hides where errors go. If one class absorbs another, the absorbing class
shows high recall and LOW precision - invisible until you compute both.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import torch, torch.nn.functional as F
import maskdino  # noqa: F401
from cropcls.config import load_config
from cropcls.data import CropPipeline
from cropcls.detector import FrozenSingleClassDetector
from cropcls.modeling import build_classifier, load_weights

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--frames", type=int, default=256)
ap.add_argument("--batch", type=int, default=1)
ap.add_argument("--device", default="cuda")
a = ap.parse_args()

run = Path(a.run)
cfg = load_config(run / "config.json"); cfg.output_dir = str(run); cfg.ims_per_batch = a.batch
device = torch.device(a.device)
det = FrozenSingleClassDetector(cfg, device=a.device)
pipe = CropPipeline(cfg, det, cfg.val_dataset, is_train=False)
model = build_classifier(cfg, len(pipe.class_names), pipe.class_names).to(device)
load_weights(model, torch.load(run / "model_final.pth", map_location="cpu", weights_only=False), cfg)
model.eval()

ys, ps = [], []
with torch.no_grad():
    for s in range(0, min(a.frames, len(pipe)), cfg.ims_per_batch):
        b = pipe.build_batch(list(range(s, min(s + cfg.ims_per_batch, a.frames))), rotations=[0])
        if len(b) == 0:
            continue
        lg = model(b.crops.to(device), b.aux.to(device)).float()
        ys.append(b.labels); ps.append(lg.argmax(1).cpu())
y, p = torch.cat(ys), torch.cat(ps)
names = pipe.class_names
C = len(names)
cm = torch.zeros((C, C), dtype=torch.long)
for t, q in zip(y.tolist(), p.tolist()):
    cm[t][q] += 1

print(f"\n{run.name}   n={len(y)}   overall top-1 = {(y==p).float().mean():.4f}\n")
print(f"{'class':<18}{'support':>8}{'precision':>11}{'recall':>9}{'F1':>8}{'predicted':>11}")
print("-" * 65)
macro_p = macro_r = macro_f = 0.0
for c in range(C):
    tp = int(cm[c][c]); sup = int(cm[c].sum()); pred = int(cm[:, c].sum())
    prec = tp / pred if pred else 0.0
    rec = tp / sup if sup else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    macro_p += prec / C; macro_r += rec / C; macro_f += f1 / C
    print(f"{names[c]:<18}{sup:>8}{prec:>11.3f}{rec:>9.3f}{f1:>8.3f}{pred:>11}")
print("-" * 65)
print(f"{'macro avg':<18}{len(y):>8}{macro_p:>11.3f}{macro_r:>9.3f}{macro_f:>8.3f}")

print(f"\nconfusion matrix (rows = true, cols = predicted)\n")
short = [n.replace("_mm", "")[:9] for n in names]
print(f"{'':<11}" + "".join(f"{s:>10}" for s in short))
for c in range(C):
    row = "".join((f"{int(v):>10}" if v else f"{'.':>10}") for v in cm[c])
    print(f"{short[c]:<11}{row}")
