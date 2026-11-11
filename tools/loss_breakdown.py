#!/usr/bin/env python
"""Where does the validation cross-entropy actually come from?

Splits mean val CE by class and by correct/incorrect, to test whether the loss is
dominated by a small number of confident errors rather than by the bulk of the data.
"""
from __future__ import annotations
import argparse, json
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
ap.add_argument("--device", default="cuda")
ap.add_argument("--batch", type=int, default=None, help="frames per detector forward; lower = less VRAM")
a = ap.parse_args()

run = Path(a.run)
cfg = load_config(run / "config.json")      # the run's own record: right trunk, right flags
cfg.output_dir = str(run)
if a.batch:
    cfg.ims_per_batch = a.batch
device = torch.device(a.device)
det = FrozenSingleClassDetector(cfg, device=a.device)
pipe = CropPipeline(cfg, det, cfg.val_dataset, is_train=False)
model = build_classifier(cfg, len(pipe.class_names), pipe.class_names).to(device)
load_weights(model, torch.load(run / "model_final.pth", map_location="cpu", weights_only=False), cfg)
model.eval()

ce, lab, ok = [], [], []
with torch.no_grad():
    for s in range(0, min(a.frames, len(pipe)), cfg.ims_per_batch):
        b = pipe.build_batch(list(range(s, min(s + cfg.ims_per_batch, a.frames))), rotations=[0])
        if len(b) == 0:
            continue
        lg = model(b.crops.to(device), b.aux.to(device)).float()
        y = b.labels.to(device)
        ce.append(F.cross_entropy(lg, y, reduction="none").cpu())
        lab.append(b.labels); ok.append((lg.argmax(1) == y).cpu())
ce, lab, ok = torch.cat(ce), torch.cat(lab), torch.cat(ok)
tot = float(ce.sum())

print(f"\n{run.name}:  n={len(ce)}  mean CE={ce.mean():.4f}  top1={ok.float().mean():.4f}\n")
print(f"{'':<18}{'n':>6}{'acc':>8}{'mean CE':>10}{'sum CE':>10}{'% of loss':>11}")
print("-" * 63)
for c in range(len(pipe.class_names)):
    m = lab == c
    if not m.any():
        continue
    print(f"{pipe.class_names[c]:<18}{int(m.sum()):>6}{ok[m].float().mean():>8.3f}"
          f"{ce[m].mean():>10.3f}{float(ce[m].sum()):>10.1f}{100*float(ce[m].sum())/tot:>10.1f}%")
print("-" * 63)
for name, m in (("CORRECT", ok), ("WRONG", ~ok)):
    print(f"{name:<18}{int(m.sum()):>6}{'':>8}{ce[m].mean():>10.3f}"
          f"{float(ce[m].sum()):>10.1f}{100*float(ce[m].sum())/tot:>10.1f}%")
pair = (lab == pipe.class_names.index("forcep03_mm")) | (lab == pipe.class_names.index("forcep04_mm"))
print(f"\nforcep03+04: {int(pair.sum())}/{len(ce)} crops ({100*float(pair.float().mean()):.1f}%) "
      f"but {100*float(ce[pair].sum())/tot:.1f}% of the loss")
