#!/usr/bin/env python
"""List and render the val crops a trained classifier gets wrong.

    PYTHONPATH=. python tools/viz_misclassified.py --run runs/cropcls_phase1_seta

Writes `<run>/misclassified/crops.csv` plus one contact sheet per GT class. The rendered
image is the *model input* - blacked out, letterboxed, de-normalized - not the raw frame
patch, so what the sheet shows is all the evidence the classifier had.

Scored on matched masks only (the crops that pass the IoU floor and runner-up ceiling),
i.e. the same population as train_cropcls.py's `val_top1` monitor. That makes this a
diagnostic of the CLASSIFIER, not of the run's classaware_f1: masks the detector missed
never appear here.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

import maskdino  # noqa: F401  - registers the hdf5 datasets
from cropcls.config import load_config
from cropcls.data import CropPipeline, apply_truncated_filter, apply_val_stride
from cropcls.detector import FrozenSingleClassDetector
from cropcls.modeling import build_classifier, load_weights

COLS = [
    "frame", "pred_index", "gt_class", "pred_class", "p_pred", "p_gt",
    "runner_up_class", "p_runner_up", "margin", "best_iou", "runner_up_iou", "det_score",
]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="run dir; its config.json is authoritative")
    ap.add_argument("--weights", default=None, help="default <run>/model_final.pth")
    ap.add_argument("--out", default=None, help="default <run>/misclassified")
    ap.add_argument("--frames", type=int, default=None, help="cap val frames (smoke runs)")
    ap.add_argument("--per-sheet", type=int, default=24)
    ap.add_argument("--device", default="cuda")
    return ap.parse_args()


def denorm(crop: torch.Tensor, mean, std) -> torch.Tensor:
    m = torch.as_tensor(mean).view(-1, 1, 1)
    s = torch.as_tensor(std).view(-1, 1, 1)
    # A mask_channel run carries a 4th channel that was never normalized by these stats.
    crop = crop[: len(mean)]
    return (crop * s + m).clamp(0, 255).byte().permute(1, 2, 0)


@torch.no_grad()
def collect(model, pipe, device, max_frames=None):
    """Every misclassified crop, with its probabilities and the crop as the model saw it."""
    n = len(pipe) if max_frames is None else min(max_frames, len(pipe))
    step = pipe.cfg.ims_per_batch
    wrong, total = [], 0

    for start in range(0, n, step):
        batch = pipe.build_batch(list(range(start, min(start + step, n))), rotations=[0])
        if len(batch) == 0:
            continue
        x = batch.crops.to(device, non_blocking=True)
        aux = batch.aux.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(x, aux)
        prob = F.softmax(logits.float(), dim=1).cpu()
        pred = prob.argmax(1)
        total += len(batch)

        for i in (pred != batch.labels).nonzero(as_tuple=True)[0].tolist():
            p = prob[i]
            gt, pi = int(batch.labels[i]), int(pred[i])
            # Runner-up behind the prediction: with pred != gt it is often the GT class,
            # and when it is not, the error is not a near-miss between two look-alikes.
            order = p.argsort(descending=True)
            second = int(order[1])
            r = batch.records[i]
            wrong.append({
                "frame": Path(r.file_name).stem or f"idx{r.frame_index}",
                "pred_index": r.pred_index,
                "gt_class": pipe.class_names[gt],
                "pred_class": pipe.class_names[pi],
                "p_pred": round(float(p[pi]), 4),
                "p_gt": round(float(p[gt]), 4),
                "runner_up_class": pipe.class_names[second],
                "p_runner_up": round(float(p[second]), 4),
                "margin": round(float(p[pi] - p[second]), 4),
                "best_iou": round(r.best_iou, 4),
                "runner_up_iou": round(r.runner_up_iou, 4),
                "det_score": round(r.score, 4),
                "_crop": denorm(batch.crops[i], pipe.cfg.pixel_mean, pipe.cfg.pixel_std),
            })
    return wrong, total


def sheets(wrong, out: Path, per_sheet: int, cols: int = 6) -> list[Path]:
    """One contact sheet per GT class, most-confident error first."""
    by_gt = defaultdict(list)
    for w in wrong:
        by_gt[w["gt_class"]].append(w)

    written = []
    for gt, items in sorted(by_gt.items()):
        items.sort(key=lambda w: -w["p_pred"])
        for page, s in enumerate(range(0, len(items), per_sheet)):
            chunk = items[s : s + per_sheet]
            rows = (len(chunk) + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(2.3 * cols, 2.75 * rows), squeeze=False)
            for ax in axes.ravel():
                ax.axis("off")
            for i, w in enumerate(chunk):
                ax = axes[i // cols][i % cols]
                ax.imshow(w.pop("_crop").numpy())
                ax.set_title(
                    f"-> {w['pred_class']}  {w['p_pred']:.2f}\n"
                    f"p(gt)={w['p_gt']:.2f}  IoU {w['best_iou']:.2f}",
                    fontsize=7, color="#c62828",
                )
                for sp in ax.spines.values():
                    sp.set_color("#c62828")
                    sp.set_linewidth(2)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.axis("on")
            fig.suptitle(
                f"GT {gt}: {len(items)} misclassified crops"
                + (f"  (page {page + 1})" if len(items) > per_sheet else ""),
                fontsize=11,
            )
            fig.tight_layout(rect=(0, 0, 1, 0.96))
            path = out / f"{gt}{'' if len(items) <= per_sheet else f'_p{page + 1}'}.png"
            fig.savefig(path, dpi=110)
            plt.close(fig)
            written.append(path)
    return written


def main():
    args = parse_args()
    run = Path(args.run)
    cfg = load_config(run / "config.json")
    cfg.weights = args.weights or str(run / "model_final.pth")
    out = Path(args.out or run / "misclassified")
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    n_frames = apply_val_stride(cfg.val_dataset, cfg.val_stride)
    n_gt = apply_truncated_filter(cfg.val_dataset, cfg.exclude_truncated)
    print(f"val={cfg.val_dataset} stride={cfg.val_stride} -> {n_frames} frames, {n_gt} GT")

    det = FrozenSingleClassDetector(cfg, device=args.device)
    pipe = CropPipeline(cfg, det, cfg.val_dataset, is_train=False)
    model = build_classifier(cfg, len(pipe.class_names), pipe.class_names).to(device)
    state = torch.load(cfg.weights, map_location="cpu", weights_only=False)
    load_weights(model, state, cfg)
    model.eval()
    print(f"weights={cfg.weights} (iter {state.get('iteration')})  mask_dilate={cfg.mask_dilate}")

    wrong, total = collect(model, pipe, device, max_frames=args.frames)
    acc = 1 - len(wrong) / max(total, 1)
    print(f"\n{len(wrong)} wrong / {total} matched crops  top1={acc:.4f}")

    pairs = Counter((w["gt_class"], w["pred_class"]) for w in wrong)
    print("\nconfusions (gt -> pred, count):")
    for (gt, pr), c in pairs.most_common():
        print(f"  {c:4d}  {gt:<16} -> {pr}")

    paths = sheets(wrong, out, args.per_sheet)
    with (out / "crops.csv").open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=COLS)
        wr.writeheader()
        wr.writerows([{k: w[k] for k in COLS} for w in wrong])
    (out / "summary.json").write_text(json.dumps({
        "weights": cfg.weights,
        "dataset": cfg.val_dataset,
        "val_stride": cfg.val_stride,
        "matched_crops": total,
        "n_wrong": len(wrong),
        "top1": round(acc, 4),
        "confusions": {f"{gt}->{pr}": c for (gt, pr), c in pairs.most_common()},
    }, indent=2))
    print(f"\nwrote {out / 'crops.csv'}, summary.json and {len(paths)} sheets to {out}")


if __name__ == "__main__":
    main()
