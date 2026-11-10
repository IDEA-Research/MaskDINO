#!/usr/bin/env python
"""Train the crop classifier on frozen class-agnostic MaskDINO masks.

A standalone loop, not a detectron2 Trainer and not a META_ARCHITECTURE: the frozen
detector is a plain object, the only trainable parameters are the classifier's trunk
and head. See spec_cropcls.md.

    train    python train_cropcls.py --config configs/cropcls/phase1_seta.yaml
    eval     python train_cropcls.py --config ... --eval-only              # model_final.pth
    eval     python train_cropcls.py --config ... --eval-only weights=<ckpt>
    gate     python train_cropcls.py --config ... --viz-only
    smoke    python train_cropcls.py --config ... --val-frames 4 max_iter=50

Any config field takes a `key=value` override on the command line. They are collected
with argparse.REMAINDER, so they must come LAST: a flag written after one is swallowed
into the override list and, having no "=", silently dropped.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import maskdino  # noqa: F401  - registers the hdf5 datasets
from cropcls.config import load_config
from cropcls.crop_folder import build_loader as build_crop_loader
from cropcls.data import (
    CropBatch,
    CropPipeline,
    StageTimer,
    apply_truncated_filter,
    apply_val_stride,
    dataset_class_names,
)
from cropcls.detector import FrozenSingleClassDetector
from cropcls.modeling import build_classifier

logger = logging.getLogger("cropcls")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--viz-only", action="store_true", help="step-4 gate: write contact sheets")
    p.add_argument("--viz-frames", type=int, default=12)
    p.add_argument("--val-frames", type=int, default=None, help="cap val frames (smoke runs)")
    p.add_argument("--hungarian", action="store_true",
                   help="score with HungarianInstanceEvaluator (step 8: comparable to the "
                        "8-class baseline's classaware_f1), not just crop top-1")
    p.add_argument("--device", default="cuda")
    p.add_argument("opts", nargs=argparse.REMAINDER, help="key=value config overrides")
    return p.parse_args()


def _parse_value(v: str):
    """JSON where it parses, plain string otherwise.

    The prefix test alone misreads paths: `weights=20261007_run/model.pth` and
    `output_dir=notes/x` both start with a character that opens a JSON literal, and
    json.loads then raises instead of leaving a perfectly good string alone.
    """
    if v[:1] not in "[{0123456789-tfn\"":
        return v
    try:
        return json.loads(v)
    except json.JSONDecodeError:
        return v


def _overrides(opts: list[str]) -> dict:
    out = {}
    for item in opts:
        if "=" not in item:
            continue
        k, v = item.split("=", 1)
        out[k.lstrip("-")] = _parse_value(v)
    return out


@torch.no_grad()
def evaluate(model, pipeline: CropPipeline, device, max_frames=None) -> dict:
    """Top-1 accuracy on matched masks + mean CE, over one pass of the val set.

    Top-1 on matched masks is the metric that is confound-free against the 8-class
    MaskDINO baseline (97.22% on `20261006204751_phase1_training_seta_v1-1`): both
    numbers condition on an instance having been localized, so a difference in the two
    detectors' recall cannot flatter either side.
    """
    model.eval()
    n = len(pipeline) if max_frames is None else min(max_frames, len(pipeline))
    step = pipeline.cfg.ims_per_batch
    correct = total = 0
    loss_sum = 0.0
    per_class_hit: dict[int, int] = {}
    per_class_tot: dict[int, int] = {}

    for start in range(0, n, step):
        batch = pipeline.build_batch(list(range(start, min(start + step, n))), rotations=[0])
        if len(batch) == 0:
            continue
        x = batch.crops.to(device, non_blocking=True)
        y = batch.labels.to(device, non_blocking=True)
        aux = batch.aux.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(x, aux)
            loss = F.cross_entropy(logits.float(), y)
        pred = logits.float().argmax(1)
        correct += int((pred == y).sum())
        total += int(y.numel())
        loss_sum += float(loss) * int(y.numel())
        for c, ok in zip(y.tolist(), (pred == y).tolist()):
            per_class_tot[c] = per_class_tot.get(c, 0) + 1
            per_class_hit[c] = per_class_hit.get(c, 0) + int(ok)

    model.train()
    acc = correct / max(total, 1)
    return {
        "val_top1": acc,
        "val_loss": loss_sum / max(total, 1),
        "val_n": total,
        "per_class": {
            pipeline.class_names[c]: per_class_hit[c] / per_class_tot[c]
            for c in sorted(per_class_tot)
        },
    }


def _log_eval(writer: SummaryWriter, m: dict, it: int) -> None:
    """Mirror one evaluate() result into tensorboard.

    Scalar names are the ones tools/backfill_tensorboard.py reconstructs, so the runs
    that predate this writer plot on the same axes as the ones that come after it.
    """
    writer.add_scalar("val/top1", m["val_top1"], it)
    writer.add_scalar("val/loss", m["val_loss"], it)
    for name, acc in m["per_class"].items():
        writer.add_scalar(f"val_per_class/{name}", acc, it)


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s %(name)s] %(message)s", datefmt="%H:%M:%S"
    )
    # An --eval-only run must rebuild the architecture the checkpoint was TRAINED with.
    # The base YAML does not know about CLI overrides used at launch (trunk, mask_only,
    # blackout, ...), so a run's own config.json is the authoritative record - loading
    # the YAML instead silently builds the wrong model and fails on a size mismatch.
    config_path = args.config
    ov = _overrides(args.opts)
    if args.eval_only:
        probe = Path(ov.get("output_dir", load_config(args.config).output_dir)) / "config.json"
        if probe.is_file():
            config_path = probe
            logger.info("eval: using the run's own config %s (not %s)", probe, args.config)

    cfg = load_config(config_path, ov)

    if not (args.eval_only or args.viz_only):
        base = Path(cfg.output_dir.rstrip("/"))
        cfg.output_dir = str(base.parent / f"{time.strftime('%Y%m%d%H%M%S')}_{base.name}")

    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    logger.info("run dir: %s", out)

    # --eval-only scores the checkpoint the run PRODUCED, so it ignores the `weights` its
    # config.json records: that field is the run's starting point, and a run initialized
    # from an earlier phase would otherwise silently re-score that phase's checkpoint.
    # Only an explicit weights= override picks a different one (e.g. model_0004000.pth).
    if args.eval_only:
        if "weights" not in ov:
            cfg.weights = str(out / "model_final.pth")
        if not Path(cfg.weights).is_file():
            raise FileNotFoundError(
                f"--eval-only has no checkpoint to score: {cfg.weights} does not exist. "
                "Pass weights=<ckpt> as a trailing override, or train the run first."
            )

    cfg_path = out / ("config_eval.json" if (args.eval_only or args.viz_only) else "config.json")
    cfg_path.write_text(json.dumps(cfg.to_dict(), indent=2))
    torch.manual_seed(cfg.seed)
    device = torch.device(args.device)

    # The label space is the train dataset's. Read before the detector so a bad
    # registration fails here, not after 50M parameters are on the GPU.
    class_names = dataset_class_names(cfg.train_dataset)
    logger.info(
        "classes=%d from %s (%s)  augment_frames=%s",
        len(class_names), cfg.train_dataset, ", ".join(class_names), cfg.augment_frames,
    )

    n_val = apply_val_stride(cfg.val_dataset, cfg.val_stride)
    n_gt = apply_truncated_filter(cfg.val_dataset, cfg.exclude_truncated)
    logger.info(
        "val=%s stride=%d -> %d frames, %d GT instances (exclude_truncated=%s)",
        cfg.val_dataset, cfg.val_stride, n_val, n_gt, cfg.exclude_truncated,
    )

    # Built on first use, not up front: training from `crop_folder` never needs it, and not
    # loading it is what makes "zero detector forwards" structural rather than hoped for. It
    # also leaves its ~177 MiB of VRAM free for a larger crops_per_batch.
    _det: dict[str, FrozenSingleClassDetector] = {}

    def get_detector() -> FrozenSingleClassDetector:
        if "d" not in _det:
            logger.info("loading frozen detector: %s", cfg.detector_checkpoint)
            _det["d"] = FrozenSingleClassDetector(cfg, device=args.device)
        return _det["d"]

    timer = StageTimer()

    # Built where it is consumed, because a pipeline is not free - it reads
    # DatasetCatalog and constructs a mapper. --viz-only reads train, --eval-only reads
    # val, --hungarian neither (evaluate_hungarian builds its own), training both.
    # Only the train split shares `timer`: its table is the training-loop report.
    def build_pipe(split: Literal["train", "val"]) -> CropPipeline:
        is_train = split == "train"
        name = cfg.train_dataset if is_train else cfg.val_dataset
        pipe = CropPipeline(
            cfg, get_detector(), name, is_train=is_train,
            timer=timer if is_train else StageTimer(),
        )
        logger.info("%s=%s (%d frames)", split, name, len(pipe))
        return pipe

    # visualization of crops on which are trained on
    if args.viz_only:
        from cropcls.viz import write_contact_sheets

        paths = write_contact_sheets(
            build_pipe("train"), out / "viz", list(range(args.viz_frames))
        )
        logger.info("wrote %d contact sheets to %s", len(paths), out / "viz")
        return

    model = build_classifier(cfg, len(class_names)).to(device)
    logger.info(
        "classifier: trunk=%s dim=%d trainable=%s",
        cfg.trunk, model.feature_dim, f"{model.n_trainable():,}",
    )
    if cfg.weights:
        state = torch.load(cfg.weights, map_location="cpu", weights_only=False)
        # The stem stride is a Conv2d ATTRIBUTE, not a parameter: a stride-2 and a stride-4
        # trunk have identical state_dict keys and identical tensor shapes, so load_state_dict
        # accepts the wrong one without error and then runs a 7x7 feature map where the weights
        # expect 14x14. mask_channel is caught only because its conv changes shape and torch
        # raises; nothing would raise here. Compare the EFFECTIVE stride, since stem_stride 0
        # and 4 are the same convnext_tiny. A checkpoint with no recorded stride predates the
        # field, which means the trunk's native stride - what every earlier run trained at.
        ckpt_stride = state.get("stem_stride", model.native_stem_stride)
        if ckpt_stride != model.stem_stride:
            raise ValueError(
                f"{cfg.weights} was trained with a first-conv stride of {ckpt_stride} but this "
                f"model is built with {model.stem_stride}. The weights would load silently and "
                "run at the wrong feature-map resolution. Set stem_stride to match, or point "
                "weights= at a checkpoint from a run with this stride."
            )
        if cfg.scale_film and not any(k.startswith("film.") for k in state["model"]):
            # Warm start from a plain checkpoint: FiLM starts at zero, so the loaded model is
            # exactly that checkpoint at iteration 0. Only film.* may be missing.
            missing, unexpected = model.load_state_dict(state["model"], strict=False)
            if unexpected or any(not k.startswith("film.") for k in missing):
                raise RuntimeError(f"checkpoint mismatch: missing {missing}, unexpected {unexpected}")
            logger.info("FiLM warm start: film.* initialized fresh (zero output)")
        else:
            model.load_state_dict(state["model"])
        logger.info("loaded weights from %s (iter %s, stem_stride %s)",
                    cfg.weights, state.get("iteration"), ckpt_stride)

    if args.eval_only and args.hungarian:
        from cropcls.export import evaluate_hungarian

        # Mode in the name so both modes can land in the run's own folder. A side folder
        # has no config.json, so the probe above cannot fire and a wrong crop_size is silent.
        m = evaluate_hungarian(
            cfg, get_detector(), model, cfg.val_dataset, device,
            stride=cfg.val_stride, output_dir=out / f"inference_{cfg.score_mode}",
            max_frames=args.val_frames,
        )
        base = {"classagnostic_f1": 0.9521, "classaware_f1": 0.9256,
                "classaware_recall": 0.9017, "classaware_precision": 0.9508}
        logger.info("--- Hungarian eval vs 8-class MaskDINO baseline ---")
        for k, b in base.items():
            v = m.get(k)
            if v is not None:
                logger.info("   %-24s %.4f   baseline %.4f   delta %+.4f", k, v, b, v - b)
        for k in sorted(m):
            if k.startswith(("classaware_recall_per_class/", "misclassified_per_class/")):
                logger.info("   %-44s %.4f", k, m[k])
        dst = out / f"hungarian_eval_{cfg.score_mode}.json"
        dst.write_text(json.dumps(m, indent=2))
        logger.info("wrote %s", dst)
        return

    if args.eval_only:
        m = evaluate(model, build_pipe("val"), device, max_frames=args.val_frames)
        logger.info("val_top1=%.4f val_loss=%.4f over %d crops", m["val_top1"], m["val_loss"], m["val_n"])
        for k, v in m["per_class"].items():
            logger.info("   %-18s %.4f", k, v)
        dst = out / "eval_evalonly.json"
        dst.write_text(json.dumps(m, indent=2))
        logger.info("wrote %s", dst)
        return

    # Validation stays on the live path either way: caching it would need a second folder and
    # an evaluate() taking a loader, and it would mean deleting the crops breaks re-scoring a
    # finished run.
    _pipes: dict[str, CropPipeline] = {}

    def get_val_pipe() -> CropPipeline:
        if "val" not in _pipes:
            _pipes["val"] = build_pipe("val")
        return _pipes["val"]

    train_pipe = None if cfg.crop_folder else build_pipe("train")
    opt = torch.optim.AdamW(model.param_groups(cfg.base_lr, cfg.weight_decay))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.base_lr, total_steps=cfg.max_iter,
        pct_start=min(0.3, cfg.warmup_iters / max(cfg.max_iter, 1)),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")
    model.train()

    writer = SummaryWriter(log_dir=str(out), flush_secs=30)
    writer.add_text("config", f"```json\n{json.dumps(cfg.to_dict(), indent=2)}\n```", 0)
    logger.info("tensorboard: tensorboard --logdir %s", out.parent)

    if cfg.crop_folder:
        loader = build_crop_loader(cfg, cfg.crop_folder, is_train=True, seed=cfg.seed)
        logger.info(
            "static crops: %d from %s | crops_per_batch=%d, num_workers=%d  "
            "(ims_per_batch=%d counts FRAMES and is used only by validation)",
            len(loader.dataset), cfg.crop_folder, cfg.crops_per_batch, cfg.num_workers,
            cfg.ims_per_batch,
        )

        def _crop_stream():
            while True:                      # a fresh iter() reshuffles
                for x, aux, y in loader:
                    yield CropBatch(crops=x, labels=y, aux=aux)

        stream = _crop_stream()
    else:
        stream = train_pipe.iter_batches(seed=cfg.seed)
    t_start = time.perf_counter()
    running = 0.0
    seen = 0
    last_log = 0
    for it in range(1, cfg.max_iter + 1):
        with timer("data", sync=False):
            batch = next(stream)
        if len(batch) == 0:
            continue
        x = batch.crops.to(device, non_blocking=True)
        y = batch.labels.to(device, non_blocking=True)
        aux = batch.aux.to(device, non_blocking=True)
        with timer("classifier", sync=True):
            with torch.autocast("cuda", dtype=torch.float16, enabled=scaler.is_enabled()):
                loss = F.cross_entropy(model(x, aux).float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        sched.step()
        running += float(loss)
        seen += len(batch)

        if it % 20 == 0 or it == cfg.max_iter:
            # Divide by the window actually accumulated, not a hardcoded 20: the last
            # window is short whenever max_iter is not a multiple of 20.
            mean_loss = running / max(it - last_log, 1)
            sec_per_iter = (time.perf_counter() - t_start) / it
            logger.info(
                "iter %6d/%d  loss %.4f  lr %.2e  crops %d  %.2f s/iter",
                it, cfg.max_iter, mean_loss, sched.get_last_lr()[0], seen, sec_per_iter,
            )
            writer.add_scalar("train/loss", mean_loss, it)
            writer.add_scalar("train/lr", sched.get_last_lr()[0], it)
            writer.add_scalar("train/sec_per_iter", sec_per_iter, it)
            running = 0.0
            last_log = it
        if cfg.eval_period and it % cfg.eval_period == 0:
            m = evaluate(model, get_val_pipe(), device, max_frames=args.val_frames)
            logger.info("iter %6d  val_top1=%.4f  val_loss=%.4f", it, m["val_top1"], m["val_loss"])
            _log_eval(writer, m, it)
        if cfg.checkpoint_period and it % cfg.checkpoint_period == 0:
            torch.save({"model": model.state_dict(), "iteration": it,
                        "stem_stride": model.stem_stride}, out / f"model_{it:07d}.pth")

    torch.save({"model": model.state_dict(), "iteration": cfg.max_iter,
                "stem_stride": model.stem_stride}, out / "model_final.pth")
    m = evaluate(model, get_val_pipe(), device, max_frames=args.val_frames)
    logger.info("FINAL val_top1=%.4f val_loss=%.4f  (8-class MaskDINO baseline: 0.9722)", m["val_top1"], m["val_loss"])
    # The headline comparison is at matched total crops seen, not matched max_iter: the live
    # arm emits 4 rotations per instance per forward, the static arm one per draw.
    m["total_crops_seen"] = seen
    for k, v in m["per_class"].items():
        logger.info("   %-18s %.4f", k, v)
    (out / "eval.json").write_text(json.dumps(m, indent=2))
    _log_eval(writer, m, cfg.max_iter)
    writer.close()
    logger.info("\n%s", timer.table())


if __name__ == "__main__":
    main()
