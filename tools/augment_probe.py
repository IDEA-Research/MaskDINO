"""Does the classifier's answer survive label-preserving perturbation, and which way does it fall?

    PYTHONPATH=. python tools/augment_probe.py --run runs/20261103122513_mask4_cxt_c448

For every validation instance of a chosen class pair, re-classify it under a grid of
perturbations that CANNOT change the true label, and report:

* flip rate - how often the prediction changes across the grid. An instance whose answer
  depends on the rotation or the crop padding was never reading a real cue.
* which way flips land - the bias question. If ambiguous instances fall preferentially to
  one member of the pair, the model has a default, and `msp` is reporting that default's
  prior rather than evidence.
* whether `msp` knows - median confidence on stable vs unstable instances. If they match,
  the softmax cannot see its own fragility, and test-time disagreement is strictly more
  information than `msp`.

The perturbation grid is built only from transforms this pipeline already trusts:
lossless 90-degree rotations, the `pad_frac` crop window, and `PhotometricJitter`. NO
horizontal flip - `cropcls/augment.py` records that surgical instruments have fixed
handedness, so a mirrored instrument is a shape that does not exist and the label would
not survive.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

import maskdino  # noqa: F401  - registers the hdf5 datasets
from cropcls.assign import assign_gt_classes
from cropcls.augment import PhotometricJitter
from cropcls.config import load_config
from cropcls.crops import letterbox_crop, mask_tight_bbox, scale_feats, to_model_input
from cropcls.data import CropPipeline, apply_truncated_filter, apply_val_stride
from cropcls.detector import FrozenSingleClassDetector
from cropcls.modeling import build_classifier, load_weights

ROTATIONS = (0, 90, 180, 270)
PAD_FRACS = (0.05, 0.10, 0.20)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--frames-per-forward", type=int, default=2)
    ap.add_argument("--classes", default="forcep03_mm,forcep04_mm,forcep01_mm,forcep02_mm",
                    help="only probe instances whose assigned GT is one of these")
    ap.add_argument("--photometric", type=int, default=2, help="jittered draws per cell")
    ap.add_argument("--pad-fracs", default=None,
                    help="comma-separated crop paddings to probe. Default: the run's OWN "
                         "pad_frac only, so every variant is in-distribution. Widening this "
                         "measures off-distribution sensitivity as well as ambiguity - a run "
                         "trained at pad_frac=0.0 has never seen 0.10.")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--val-stride", type=int, default=None,
                    help="override cfg.val_stride. 1 = every val frame (1024), no striding. "
                         "The run's own value (4, 256 frames) is the 8-class baseline's "
                         "protocol, so only metrics compared against it need it.")
    ap.add_argument("--out", default=None)
    return ap.parse_args()


@torch.no_grad()
def probe(model, pipe, device, keep_classes, n_photo, batch, max_frames=None, step=2):
    cfg = pipe.cfg
    n = len(pipe) if max_frames is None else min(max_frames, len(pipe))
    jit = PhotometricJitter(generator=torch.Generator().manual_seed(0))
    out = []

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
            gt = inst.gt_masks
            if not torch.is_tensor(gt):
                gt = gt.tensor
            pm = det.masks.cpu()
            a = assign_gt_classes(pm, gt, inst.gt_classes,
                                  iou_floor=cfg.iou_floor,
                                  runner_up_ceiling=cfg.runner_up_ceiling)
            image = m["image"].float()

            for k in torch.where(a.keep)[0].tolist():
                label = int(a.labels[k])
                if pipe.class_names[label] not in keep_classes:
                    continue
                tight = mask_tight_bbox(pm[k])
                if tight is None:
                    continue

                variants, tags = [], []
                for pf in PAD_FRACS:
                    for photo in range(n_photo + 1):      # 0 = clean, then jittered draws
                        crop = letterbox_crop(
                            image, pm[k], cfg.crop_size, box=tight, pad_frac=pf,
                            photometric=None if photo == 0 else jit,
                            mask_only=cfg.mask_only, blackout=cfg.blackout,
                            dilate=cfg.mask_dilate, close=cfg.mask_close,
                        )
                        if crop is None:
                            continue
                        for deg in ROTATIONS:
                            variants.append(to_model_input(
                                crop, cfg.pixel_mean, cfg.pixel_std, deg, cfg.mask_channel))
                            tags.append((pf, photo, deg))
                if not variants:
                    continue
                sf = scale_feats(pm[k])   # every variant is the same mask: one aux for all

                preds, msps = [], []
                for s in range(0, len(variants), batch):
                    x = torch.stack(variants[s:s + batch]).to(device)
                    aux = sf.expand(len(x), -1).to(device)
                    with torch.autocast("cuda", dtype=torch.float16,
                                        enabled=device.type == "cuda"):
                        lg = model(x, aux)
                    p = F.softmax(lg.float(), dim=1).cpu()
                    v, i = p.max(1)
                    preds += [int(t) for t in i]
                    msps += [float(t) for t in v]

                c = Counter(preds)
                maj, maj_n = c.most_common(1)[0]
                out.append({
                    "frame": Path(str(m.get("file_name", ""))).stem,
                    "pred_index": k,
                    "gt_class": pipe.class_names[label],
                    "n_variants": len(preds),
                    "n_distinct_preds": len(c),
                    "majority_class": pipe.class_names[maj],
                    "majority_frac": maj_n / len(preds),
                    "flipped": int(len(c) > 1),
                    "clean_pred": pipe.class_names[preds[0]],
                    "clean_msp": msps[0],
                    "mean_msp": sum(msps) / len(msps),
                    "min_msp": min(msps),
                    # share of variants landing on each member of the pair, for the bias read
                    **{"frac_" + pipe.class_names[ci]: c.get(ci, 0) / len(preds)
                       for ci in range(len(pipe.class_names))},
                })
    return out


def main():
    args = parse_args()
    run = Path(args.run)
    cfg = load_config(run / "config.json")
    cfg.weights = args.weights or str(run / "model_final.pth")
    out_dir = Path(args.out or run / "augprobe")
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    keep = set(args.classes.split(","))

    stride = cfg.val_stride if args.val_stride is None else args.val_stride
    n_frames = apply_val_stride(cfg.val_dataset, stride)
    n_gt = apply_truncated_filter(cfg.val_dataset, cfg.exclude_truncated)
    print(f"val={cfg.val_dataset} stride={stride} -> {n_frames} frames, {n_gt} scored GT")
    det = FrozenSingleClassDetector(cfg, device=args.device)
    pipe = CropPipeline(cfg, det, cfg.val_dataset, is_train=False)
    model = build_classifier(cfg, len(pipe.class_names), pipe.class_names).to(device)
    state = torch.load(cfg.weights, map_location="cpu", weights_only=False)
    load_weights(model, state, cfg)
    model.eval()
    print(f"weights={cfg.weights} trunk={cfg.trunk} crop={cfg.crop_size} "
          f"blackout={cfg.blackout} mask_channel={cfg.mask_channel}")

    global PAD_FRACS
    PAD_FRACS = (tuple(float(x) for x in args.pad_fracs.split(","))
                 if args.pad_fracs else (cfg.pad_frac,))
    print(f"grid: pad_frac {PAD_FRACS} x {args.photometric + 1} photometric x "
          f"{len(ROTATIONS)} rotations = {len(PAD_FRACS)*(args.photometric+1)*len(ROTATIONS)} "
          f"variants per instance (no flips: handedness is real); run trained at "
          f"pad_frac={cfg.pad_frac}")
    rows = probe(model, pipe, device, keep, args.photometric, args.batch,
                 args.frames, args.frames_per_forward)
    with open(out_dir / "augprobe.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\n{len(rows)} instances probed")

    for pair in (("forcep03_mm", "forcep04_mm"), ("forcep01_mm", "forcep02_mm")):
        sel = [r for r in rows if r["gt_class"] in pair]
        if not sel:
            continue
        flip = [r for r in sel if r["flipped"]]
        print(f"\n=== {pair[0]} / {pair[1]}  n={len(sel)}")
        print(f"  unstable under perturbation: {len(flip)} ({100*len(flip)/len(sel):.1f}%)")
        print(f"  clean-crop accuracy         : "
              f"{100*sum(1 for r in sel if r['clean_pred']==r['gt_class'])/len(sel):.1f}%")
        print(f"  majority-vote accuracy      : "
              f"{100*sum(1 for r in sel if r['majority_class']==r['gt_class'])/len(sel):.1f}%")
        if flip:
            print(f"  median msp  stable {statistics.median(r['mean_msp'] for r in sel if not r['flipped']):.5f}"
                  f"   unstable {statistics.median(r['mean_msp'] for r in flip):.5f}")
            # THE bias read: across unstable instances, where does the vote mass land?
            for c in pair:
                share = statistics.mean(r["frac_" + c] for r in flip)
                print(f"    unstable instances put {100*share:5.1f}% of their votes on {c}")
            for c in pair:
                sub = [r for r in flip if r["gt_class"] == c]
                if sub:
                    other = pair[1] if c == pair[0] else pair[0]
                    print(f"    GT {c}: n={len(sub)}  votes {100*statistics.mean(r['frac_'+c] for r in sub):5.1f}% "
                          f"correct / {100*statistics.mean(r['frac_'+other] for r in sub):5.1f}% to {other}")
    print(f"\nwrote {out_dir}/augprobe.csv")


if __name__ == "__main__":
    main()
