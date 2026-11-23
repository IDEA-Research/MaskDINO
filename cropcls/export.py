"""Score the cropcls pipeline with the SAME evaluator the baseline was scored with.

`train_cropcls.evaluate()` reports top-1 over crops that passed *our* assignment rule
(IoU >= 0.5 and runner-up < 0.2). That is the right primary metric - it conditions on an
instance having been localized, so the two detectors' differing recall cannot flatter
either side - but it is NOT comparable to the baseline's `classaware_f1` of 0.9256, which
comes from `HungarianInstanceEvaluator`'s one-to-one matching.

This module feeds the classifier's predictions straight into that evaluator, so every number
is produced by identical code. No JSON round-trip, so the contiguous-vs-canonical
category id trap never arises.

Two traps this handles explicitly:

* **The evaluator silently disables misclassification accounting** when
  `cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES <= 1` (`_check_classes` in
  `hungarian_instance_evaluation.py`). The detector's own cfg says 1, so it is cloned with
  NUM_CLASSES set to the classifier's count - otherwise every `misclassified_*` comes back
  0.0 and the comparison looks perfect for the wrong reason.
* **`apply_test_sample_stride` is not idempotent** - it removes the catalog entry and
  re-registers a plain lambda, so a second call strides the already-strided list (4 -> 16).
  It is called at most once per process here.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from detectron2.structures import Boxes, Instances

from maskdino.evaluation.hungarian_instance_evaluation import HungarianInstanceEvaluator

from .config import CropClsConfig
from .crops import letterbox_crop, mask_tight_bbox, scale_feats, to_model_input
from .data import CropPipeline, apply_val_stride

logger = logging.getLogger("cropcls.export")

__all__ = ["evaluate_hungarian"]


def _eval_cfg(detector_cfg, dataset_name: str, num_classes: int, stride: int):
    d2 = detector_cfg.clone()
    d2.defrost()
    d2.MODEL.SEM_SEG_HEAD.NUM_CLASSES = num_classes      # or _check_classes is False
    d2.DATASETS.TEST = (dataset_name,)
    d2.DATASETS.TEST_SAMPLE_STRIDE = stride
    d2.freeze()
    return d2


def _mask_boxes(masks: torch.Tensor) -> Boxes:
    """Each mask's own tight box, which is the frame the masks live in.

    NOT the detector's `pred_boxes`: with no "height"/"width" in the batched dict MaskDINO
    denormalizes those against the PADDED size while postprocessing masks to the un-padded
    one (maskdino/maskdino.py:411-413), so they sit in a different frame. The tight box is
    also what the GT-side filter tests and what the crop is cut from.
    """
    rows = []
    for m in masks:
        b = mask_tight_bbox(m)
        rows.append([0.0, 0.0, 0.0, 0.0] if b is None else [float(v) for v in b])
    return Boxes(torch.tensor(rows, dtype=torch.float32).reshape(-1, 4))


@torch.no_grad()
def evaluate_hungarian(
    cfg: CropClsConfig,
    detector,
    model,
    dataset_name: str,
    device,
    stride: int = 4,
    output_dir: str | Path | None = None,
    max_frames: int | None = None,
) -> dict:
    """Run detector -> crops -> classifier over the val set, score with the Hungarian evaluator.

    Returns its metric dict: classagnostic_/classaware_ precision, recall and f1, the
    per-class recalls, and the misclassification counts - the same keys the baseline run
    logged under `instance_matching/`.
    """
    # The head's own width, not a configured number: this is what the model predicts,
    # so the evaluator's NUM_CLASSES cannot disagree with it.
    n_classes = int(model.head.out_features)
    d2 = _eval_cfg(detector.d2_cfg, dataset_name, n_classes, stride)
    # The stride is normally already applied by train_cropcls from cfg.val_stride, before
    # anything reads the dataset. This is the fallback for calling evaluate_hungarian
    # directly; the shared guard makes the second call a no-op rather than a 4 -> 16 restride.
    apply_val_stride(dataset_name, stride)

    # Built AFTER the stride, so its GT index covers exactly the evaluated frames.
    evaluator = HungarianInstanceEvaluator(
        dataset_name, d2, distributed=False,
        output_dir=str(output_dir) if output_dir else None,
    )
    # No prediction-side wrapper: `exclude_truncated` now MARKS border-truncated GT
    # iscrowd=1 (apply_truncated_filter -> apply_truncated_instance_filter) instead of
    # deleting it, and HungarianInstanceEvaluator reads that flag itself - a cut tool is an
    # IGNORE region, so it is neither a false negative when missed nor a false positive when
    # detected. The old DropTruncatedPredictions existed only because deleting the GT left
    # correct detections with nothing to match (26 dropped GT raised false positives 19 -> 45);
    # ignore regions fix both sides at once, so there is nothing left for it to do.
    evaluator.reset()

    # is_train=False -> no crop-level augmentation; the mapper still yields GT.
    pipe = CropPipeline(cfg, detector, dataset_name, is_train=False)
    n = len(pipe) if max_frames is None else min(max_frames, len(pipe))
    model.eval()
    logger.info(
        "hungarian eval: %s stride=%d -> %d frames, NUM_CLASSES=%d (check_classes=%s)",
        dataset_name, stride, n, n_classes, n_classes > 1,
    )

    for i in range(n):
        mapped = pipe.mapper(copy.deepcopy(pipe.dicts[i]))
        image = mapped["image"]
        det = detector.detect([image], amp=cfg.amp)[0]
        h, w = image.shape[-2:]

        if len(det) == 0:
            inst = Instances((h, w))
            inst.pred_masks = torch.zeros((0, h, w), dtype=torch.bool)
            inst.pred_classes = torch.zeros((0,), dtype=torch.long)
            inst.scores = torch.zeros((0,))
            inst.pred_boxes = Boxes(torch.zeros((0, 4)))
            evaluator.process(
                [{"image_id": mapped["image_id"], "height": h, "width": w}],
                [{"instances": inst}],
            )
            continue

        masks = det.masks.cpu()
        img_f = image.float()
        # Same occluder union as the live path. This is the path that produces classaware_f1,
        # so a channel layout that disagrees with training here does not raise - it just
        # reports a worse number (to_model_input's own warning).
        occ_all = masks.any(0) if cfg.occluder_channel else None
        crops, auxs, keep_idx = [], [], []
        for k in range(len(det)):
            tight = mask_tight_bbox(masks[k])
            if tight is None:
                continue
            # box= is not optional. Without it letterbox_crop falls back to the DILATED tight
            # box, so this path cropped a wider field of view than training did for the same
            # mask_dilate - wider by 2*mask_dilate px per axis, and nothing raised; it just
            # scored a differently-framed crop than the model trained on. Measured over the
            # 4623 crops in crops/train, mean long-axis ratio: 1.04x at dilate 4, 1.08x at 8,
            # 1.16x at 16, 1.33x at 32. So it was negligible for the small radii and real for
            # dilate 32, where 78% of crops were framed >20% wider than in training.
            # Closing is exempt: close_mask does not grow the outer silhouette, so the dilated
            # tight box equals the undilated one. Training passes the undilated box; match it.
            out = letterbox_crop(
                img_f, masks[k], cfg.crop_size,
                box=tight, pad_frac=cfg.pad_frac,
                others=(occ_all & ~masks[k]) if occ_all is not None else None,
                mask_only=cfg.mask_only, blackout=cfg.blackout,
                dilate=cfg.mask_dilate, close=cfg.mask_close,
            )
            if out is None:
                continue
            crops.append(
                to_model_input(out, cfg.pixel_mean, cfg.pixel_std,
                               mask_channel=cfg.mask_channel,
                               occluder_channel=cfg.occluder_channel)
            )
            auxs.append(scale_feats(masks[k]))   # undilated full-frame mask, as in training
            keep_idx.append(k)

        if crops:
            x = torch.stack(crops).to(device)
            aux = torch.stack(auxs).to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(x, aux)
            prob = F.softmax(logits.float(), dim=1)
            cls_prob, pred_classes = prob.max(1)
            obj = det.scores.cpu()[keep_idx]
            if cfg.score_mode == "detector":
                scores = obj
            else:
                # MaskDINO parity: score = sigmoid(class_logit) * mask_score.
                scores = obj * cls_prob.cpu()
            inst = Instances((h, w))
            inst.pred_masks = masks[keep_idx]
            inst.pred_classes = pred_classes.cpu().long()
            inst.scores = scores
            inst.pred_boxes = _mask_boxes(inst.pred_masks)
        else:
            inst = Instances((h, w))
            inst.pred_masks = torch.zeros((0, h, w), dtype=torch.bool)
            inst.pred_classes = torch.zeros((0,), dtype=torch.long)
            inst.scores = torch.zeros((0,))
            inst.pred_boxes = Boxes(torch.zeros((0, 4)))

        # height/width size the evaluator's GT and ignore-region rasters for this frame.
        evaluator.process(
            [{"image_id": mapped["image_id"], "height": h, "width": w}],
            [{"instances": inst}],
        )

    results = evaluator.evaluate() or {}
    return dict(results.get("instance_matching", results))
