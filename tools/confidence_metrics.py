"""Calibration and failure-prediction metrics for a trained crop classifier.

    PYTHONPATH=. python tools/confidence_metrics.py --run runs/cropcls_phase1_seta

Answers one question: does the classifier's confidence separate its correct predictions
from its wrong ones well enough to reject the wrong ones at inference?

Two properties, measured separately because they are independent (a model emitting its own
base accuracy for every input has ECE 0 and AUROC 0.5):

* ECE / reliability - do the stated probabilities match empirical accuracy?
* AUROC, risk-coverage - does confidence RANK errors below correct answers? Only this one
  decides whether a `p >= t` reject threshold can work.

Unlike `viz_misclassified.py`, which keeps only `pred != label`, every prediction is kept -
including the ones the assignment rule drops, because an unmatched prediction is a false
positive and thresholding removes those too. That makes the classaware projection below a
LOWER bound on the gain.

Writes `<run>/confidence/{crops.csv,summary.json,reliability.png,risk_coverage.png}`.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

import maskdino  # noqa: F401  - registers the hdf5 datasets
from cropcls.assign import assign_gt_classes
from cropcls.config import load_config
from cropcls.crops import letterbox_crop, mask_tight_bbox, scale_feats, to_model_input
from cropcls.data import CropPipeline, apply_truncated_filter, apply_val_stride
from cropcls.detector import FrozenSingleClassDetector
from cropcls.modeling import build_classifier, load_weights

# Scores to rank by. Every one is "higher = more trustworthy", so one AUROC sign convention
# covers all four. msp is what a `p_pred >= t` threshold would actually use; the other three
# are there to show whether a better-ordered score exists before anyone retrains for one.
SCORES = ("msp", "margin", "neg_entropy", "max_logit")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run dir holding config.json + model_final.pth")
    ap.add_argument("--weights", default=None, help="default <run>/model_final.pth")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--frames", type=int, default=None, help="cap frames (debug)")
    ap.add_argument("--frames-per-forward", type=int, default=None,
                    help="frames per detector forward; default cfg.ims_per_batch. Lower it "
                         "when the card is shared - the deformable-attention fallback asks "
                         "for ~1.6 GB per frame at crop 448.")
    ap.add_argument("--val-stride", type=int, default=None,
                    help="override cfg.val_stride; 1 = every val frame. The run's "
                         "own value is the 8-class baseline protocol, so metrics "
                         "compared against it must keep it.")
    ap.add_argument("--out", default=None, help="default <run>/confidence")
    ap.add_argument("--bins", type=int, default=15, help="ECE bins")
    return ap.parse_args()


@torch.no_grad()
def collect(model, pipe, device, max_frames=None, step=None):
    """Classify every detector prediction, assigned or not. One row per prediction."""
    n = len(pipe) if max_frames is None else min(max_frames, len(pipe))
    cfg = pipe.cfg
    step = step or cfg.ims_per_batch
    rows = []

    for start in range(0, n, step):
        idx = list(range(start, min(start + step, n)))
        mapped = pipe._map_frames(idx)
        dets = pipe.detector.detect([m["image"] for m in mapped], amp=cfg.amp)

        for m, det in zip(mapped, dets):
            if len(det) == 0:
                continue
            inst = m.get("instances", None)
            if inst is None or not inst.has("gt_masks"):
                continue
            gt_masks = inst.gt_masks
            if not torch.is_tensor(gt_masks):
                gt_masks = gt_masks.tensor
            pred_masks = det.masks.cpu()
            a = assign_gt_classes(
                pred_masks, gt_masks, inst.gt_classes,
                iou_floor=cfg.iou_floor, runner_up_ceiling=cfg.runner_up_ceiling,
            )

            image = m["image"].float()
            crops, auxs, keep, geom = [], [], [], []
            for k in range(len(det)):
                tight = mask_tight_bbox(pred_masks[k])
                if tight is None:
                    continue
                x0, y0, x1, y1 = tight
                out = letterbox_crop(
                    image, pred_masks[k], cfg.crop_size, box=tight, pad_frac=cfg.pad_frac,
                    mask_only=cfg.mask_only, blackout=cfg.blackout,
                    dilate=cfg.mask_dilate, close=cfg.mask_close,
                )
                if out is None:
                    continue
                crops.append(
                    to_model_input(out, cfg.pixel_mean, cfg.pixel_std, 0, cfg.mask_channel)
                )
                # Pre-letterbox geometry, read off the TIGHT box, not the padded one: the
                # letterbox throws the scale away (cropcls-scale-feature-notes.md section 1),
                # and reading it after pad_frac would inflate it by (1 + 2*pad_frac). Appended
                # here rather than above the `out is None` guard so it stays index-aligned with
                # `keep` - a dropped crop would otherwise shift every later row's scale columns
                # onto the wrong detection.
                geom.append((int(y1 - y0), int(x1 - x0), int(pred_masks[k].sum())))
                auxs.append(scale_feats(pred_masks[k]))
                keep.append(k)
            if not crops:
                continue

            x = torch.stack(crops).to(device)
            aux = torch.stack(auxs).to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(x, aux)
            logits = logits.float().cpu()
            prob = F.softmax(logits, dim=1)
            top2 = prob.topk(min(2, prob.shape[1]), dim=1)

            for j, k in enumerate(keep):
                p = prob[j]
                pred = int(top2.indices[j, 0])
                label = int(a.labels[k])
                g_h, g_w, g_area = geom[j]
                rows.append({
                    "frame": Path(str(m.get("file_name", ""))).stem,
                    "pred_index": k,
                    "verdict": a.verdict[k],
                    "gt_class": pipe.class_names[label] if label >= 0 else "",
                    "pred_class": pipe.class_names[pred],
                    # correct / wrong is only defined where the assignment rule gave a label;
                    # an unassigned prediction is neither, it is a false positive.
                    "correct": int(label == pred) if label >= 0 else -1,
                    "msp": float(top2.values[j, 0]),
                    "margin": float(top2.values[j, 0] - top2.values[j, 1]),
                    "neg_entropy": float((p * p.clamp_min(1e-12).log()).sum()),
                    "max_logit": float(logits[j].max()),
                    "p_gt": float(p[label]) if label >= 0 else -1.0,
                    "best_iou": float(a.best_iou[k]),
                    "runner_up_iou": float(a.runner_up_iou[k]),
                    "det_score": float(det.scores[k]),
                    "mask_h": g_h,
                    "mask_w": g_w,
                    "mask_long": max(g_h, g_w),
                    "mask_area": g_area,
                })
    return rows


def auroc(pos: list[float], neg: list[float]) -> float:
    """P(score of a correct prediction > score of a wrong one), ties counted as 0.5.

    Rank form of Mann-Whitney U, with mid-ranks for ties - no sklearn in this venv.
    """
    if not pos or not neg:
        return float("nan")
    xs = sorted([(s, 1) for s in pos] + [(s, 0) for s in neg])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(xs):                      # mid-rank every tie group
        j = i
        while j + 1 < len(xs) and xs[j + 1][0] == xs[i][0]:
            j += 1
        mid = (i + j) / 2.0 + 1.0
        for t in range(i, j + 1):
            ranks[t] = mid
        i = j + 1
    r_pos = sum(r for r, (_, lab) in zip(ranks, xs) if lab == 1)
    n_p, n_n = len(pos), len(neg)
    return (r_pos - n_p * (n_p + 1) / 2.0) / (n_p * n_n)


def ece(conf: list[float], correct: list[int], bins: int):
    """Equal-width-bin expected calibration error, plus the reliability table."""
    n = len(conf)
    table = []
    total = 0.0
    worst = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        sel = [i for i, c in enumerate(conf) if (c > lo or b == 0) and c <= hi]
        if not sel:
            table.append({"lo": lo, "hi": hi, "n": 0, "conf": None, "acc": None})
            continue
        mc = sum(conf[i] for i in sel) / len(sel)
        ma = sum(correct[i] for i in sel) / len(sel)
        gap = abs(ma - mc)
        total += len(sel) / n * gap
        worst = max(worst, gap)
        table.append({"lo": lo, "hi": hi, "n": len(sel), "conf": mc, "acc": ma})
    return total, worst, table


def risk_coverage(score: list[float], correct: list[int]):
    """Error rate among the predictions kept, as the least-confident ones are dropped."""
    order = sorted(range(len(score)), key=lambda i: -score[i])
    pts, errs = [], 0
    for rank, i in enumerate(order, start=1):
        errs += 1 - correct[i]
        pts.append((rank / len(order), errs / rank))
    aurc = sum(r for _, r in pts) / len(pts)
    return pts, aurc


def classaware_projection(rows, baseline, grid):
    """Project a confidence floor onto the Hungarian evaluator's classaware metrics.

    Exact, given the baseline counts, because rejecting a prediction moves known counts
    (`hungarian_evaluator_metrics.md`: classaware FP = false_pos + misclassified,
    FN = false_neg + misclassified):

      reject a MISCLASSIFIED match -> misclassified -1, false_neg +1. FN is unchanged
          (the two cancel), FP drops by 1, TP unchanged. Strictly better.
      reject a CORRECT match       -> TP -1, false_neg +1. Both precision and recall fall.
      reject an UNMATCHED pred     -> false_pos -1. Strictly better.

    Approximate only in that the crop population (IoU >= floor AND runner_up < ceiling) is
    not bit-identical to the evaluator's Hungarian one-to-one matching; the row counts
    printed by --run let you see how far apart they are.
    """
    tp0 = baseline["num_correct_class_total"]
    mis0 = baseline["misclassified_total"]
    fp0 = baseline["false_positives_total"]
    fn0 = baseline["false_negatives_total"]

    ok = [r["msp"] for r in rows if r["correct"] == 1]
    bad = [r["msp"] for r in rows if r["correct"] == 0]
    unm = [r["msp"] for r in rows if r["correct"] == -1]

    out = []
    for t in grid:
        d_ok = sum(1 for s in ok if s < t)
        d_bad = sum(1 for s in bad if s < t)
        d_unm = sum(1 for s in unm if s < t)
        tp = tp0 - d_ok
        fp = (fp0 - d_unm) + (mis0 - d_bad)
        fn = (fn0 + d_ok + d_bad) + (mis0 - d_bad)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        out.append({
            "thresh": t,
            "rejected_correct": d_ok,
            "rejected_misclassified": d_bad,
            "rejected_unmatched": d_unm,
            "classaware_precision": p,
            "classaware_recall": r,
            "classaware_f1": 2 * p * r / (p + r) if p + r else 0.0,
        })
    return out


def plots(table, rc, out: Path, title: str):
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.plot([0, 1], [0, 1], "--", color="#888", lw=1, label="perfect calibration")
    xs = [(b["lo"] + b["hi"]) / 2 for b in table if b["n"]]
    ax.bar(xs, [b["acc"] for b in table if b["n"]], width=1 / len(table) * 0.9,
           color="#1f77b4", edgecolor="white", label="accuracy")
    ax.plot(xs, [b["conf"] for b in table if b["n"]], "o-", color="#c62828",
            ms=4, lw=1.2, label="mean confidence")
    ax.set_xlabel("confidence (max softmax)")
    ax.set_ylabel("accuracy")
    ax.set_title(f"Reliability - {title}", fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "reliability.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    ax.plot([c for c, _ in rc], [100 * r for _, r in rc], color="#1f77b4", lw=1.6)
    ax.set_xlabel("coverage (fraction of matched predictions kept)")
    ax.set_ylabel("error rate among kept  [%]")
    ax.set_title(f"Risk-coverage - {title}", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "risk_coverage.png", dpi=120)
    plt.close(fig)


def main():
    args = parse_args()
    run = Path(args.run)
    cfg = load_config(run / "config.json")
    cfg.weights = args.weights or str(run / "model_final.pth")
    out = Path(args.out or run / "confidence")
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    stride = cfg.val_stride if args.val_stride is None else args.val_stride
    n_frames = apply_val_stride(cfg.val_dataset, stride)
    n_gt = apply_truncated_filter(cfg.val_dataset, cfg.exclude_truncated)
    print(f"val={cfg.val_dataset} stride={stride} -> {n_frames} frames, {n_gt} GT")

    det = FrozenSingleClassDetector(cfg, device=args.device)
    pipe = CropPipeline(cfg, det, cfg.val_dataset, is_train=False)
    model = build_classifier(cfg, len(pipe.class_names), pipe.class_names).to(device)
    state = torch.load(cfg.weights, map_location="cpu", weights_only=False)
    # strict: the only proof that this checkpoint belongs to this config's trunk. Several run
    # dirs were renamed on 2026-10-31 and one stale `weights` path survives in an artifact.
    load_weights(model, state, cfg)
    model.eval()
    print(f"trunk={cfg.trunk} crop={cfg.crop_size} mask_channel={cfg.mask_channel} "
          f"blackout={cfg.blackout}")
    print(f"weights={cfg.weights} (iter {state.get('iteration')})")

    rows = collect(model, pipe, device, max_frames=args.frames,
                   step=args.frames_per_forward)
    matched = [r for r in rows if r["correct"] >= 0]
    conf = [r["msp"] for r in matched]
    corr = [r["correct"] for r in matched]
    n_wrong = len(matched) - sum(corr)
    top1 = sum(corr) / max(len(matched), 1)
    print(f"\n{len(rows)} predictions: {len(matched)} matched "
          f"({n_wrong} wrong, top1={top1:.4f}), {len(rows) - len(matched)} unmatched")

    e, mce, table = ece(conf, corr, args.bins)
    mean_conf = sum(conf) / max(len(conf), 1)
    print(f"\nECE({args.bins} bins) = {e:.4f}   max bin gap = {mce:.4f}")
    print(f"mean confidence {mean_conf:.4f} vs accuracy {top1:.4f} "
          f"-> overconfident by {mean_conf - top1:+.4f}")

    print("\nAUROC (correct vs wrong), higher = confidence ranks errors lower:")
    aurocs = {}
    for s in SCORES:
        aurocs[s] = auroc(
            [r[s] for r in matched if r["correct"] == 1],
            [r[s] for r in matched if r["correct"] == 0],
        )
        print(f"  {s:<12} {aurocs[s]:.4f}")

    rc, aurc = risk_coverage(conf, corr)
    print(f"\nAURC = {aurc:.4f}  (area under risk-coverage; lower is better)")
    print("selective risk at fixed coverage:")
    for cov in (1.0, 0.95, 0.9, 0.8, 0.7):
        i = min(int(cov * len(rc)) - 1, len(rc) - 1)
        print(f"  keep {cov:.0%} -> error rate {100 * rc[i][1]:.2f}%  "
              f"(drop {len(rc) - i - 1} of {len(rc)})")

    grid = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999]
    proj = None
    bl_path = None
    for name in ("hungarian_eval_detector.json", "hungarian_eval_stride4.json",
                 "hungarian_eval.json"):
        if (run / name).is_file():
            bl_path = run / name
            break
    if bl_path:
        with open(bl_path) as f:
            baseline = json.load(f)
        need = ("num_correct_class_total", "misclassified_total",
                "false_positives_total", "false_negatives_total")
        if all(k in baseline for k in need):
            proj = classaware_projection(matched + [r for r in rows if r["correct"] < 0],
                                         baseline, grid)
            print(f"\nclassaware projection from {bl_path.name} "
                  f"(baseline f1 {baseline.get('classaware_f1', float('nan')):.4f}):")
            print("  thresh  rej_ok  rej_wrong  rej_unmatched   prec    recall      f1")
            for p in proj:
                print(f"  {p['thresh']:.3f}  {p['rejected_correct']:6d}  "
                      f"{p['rejected_misclassified']:9d}  {p['rejected_unmatched']:13d}  "
                      f"{p['classaware_precision']:.4f}  {p['classaware_recall']:.4f}  "
                      f"{p['classaware_f1']:.4f}")

    fields = [k for k in rows[0]] if rows else []
    with open(out / "crops.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    summary = {
        "run": str(run),
        "weights": cfg.weights,
        "iteration": state.get("iteration"),
        "trunk": cfg.trunk,
        "crop_size": cfg.crop_size,
        "mask_channel": cfg.mask_channel,
        "blackout": cfg.blackout,
        "dataset": cfg.val_dataset,
        "val_stride": cfg.val_stride,
        "n_frames": n_frames,
        "n_predictions": len(rows),
        "n_matched": len(matched),
        "n_unmatched": len(rows) - len(matched),
        "n_wrong": n_wrong,
        "top1": top1,
        "mean_confidence": mean_conf,
        "ece": e,
        "ece_bins": args.bins,
        "max_bin_gap": mce,
        "auroc": aurocs,
        "aurc": aurc,
        "reliability": table,
        "selective_risk": {
            f"{cov:.2f}": rc[min(int(cov * len(rc)) - 1, len(rc) - 1)][1]
            for cov in (1.0, 0.95, 0.9, 0.8, 0.7)
        },
        "classaware_projection": proj,
        "classaware_baseline": str(bl_path) if bl_path else None,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    plots(table, rc, out, run.name)
    print(f"\nwrote {out}/crops.csv, summary.json, reliability.png, risk_coverage.png")


if __name__ == "__main__":
    main()
