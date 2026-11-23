"""Frame -> augment -> detect -> assign -> crops, with per-stage timing.

Order matters: the detector runs on the *augmented* frame, so its masks and the
augmented `instances.gt_masks` share a coordinate frame. That also means the masks
cannot be cached - scale jitter changes the image every iteration.

Rotation is applied *after* cropping (0/90/180/270, lossless on a square canvas), so one
detector forward yields len(rotations) x the classifier samples. `INPUT.RANDOM_ROTATION`
is therefore switched off on the mapper - see spec_cropcls.md section 6, step 5.
"""

from __future__ import annotations

import copy
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch
from detectron2.config import CfgNode
from detectron2.data import DatasetCatalog, MetadataCatalog

from maskdino.data.dataset_mappers.hdf5_coco_instance_dataset_mapper import (
    Hdf5CocoInstanceDatasetMapper,
)
from maskdino.data.datasets.register_hdf5_instance import (
    apply_test_sample_stride,
    apply_truncated_instance_filter,
)

from .assign import assign_gt_classes
from .augment import PhotometricJitter
from .config import CropClsConfig
from .crops import (
    AUX_DIM,
    letterbox_crop,
    mask_tight_bbox,
    scale_feats,
    to_model_input,
)
from .detector import FrozenSingleClassDetector

__all__ = [
    "STRIDE_APPLIED",
    "TRUNCATED_FILTERED",
    "CropBatch",
    "CropPipeline",
    "CropRecord",
    "StageTimer",
    "apply_truncated_filter",
    "apply_val_stride",
    "dataset_class_names",
]

# Guard for the non-idempotent stride helper. It lives here rather than in export.py because
# the in-training monitor and --hungarian must share ONE guard: the helper re-registers the
# catalog entry with an already-strided list, so a second call strides 4 -> 16 and silently
# scores a quarter of the frames you think (cropcls/README.md).
STRIDE_APPLIED: set[str] = set()
# Same one-shot story for the truncation filter: it also re-registers the catalog entry, so
# a second call would filter an already-filtered list.
TRUNCATED_FILTERED: set[str] = set()


def apply_val_stride(dataset_name: str, stride: int) -> int:
    """Keep every `stride`-th val frame, once, in place under the same name. Returns the count.
    """
    if stride <= 1 or dataset_name in STRIDE_APPLIED:
        return len(DatasetCatalog.get(dataset_name))
    d2 = CfgNode({"DATASETS": CfgNode({"TEST": (dataset_name,), "TEST_SAMPLE_STRIDE": stride})})
    apply_test_sample_stride(d2)
    STRIDE_APPLIED.add(dataset_name)
    return len(DatasetCatalog.get(dataset_name))


def apply_truncated_filter(dataset_name: str, enabled: bool) -> int:
    """Mark frame-border-truncated GT as iscrowd=1, once. Returns the SCORED instance count.

    Marks, no longer deletes. `apply_truncated_instance_filter` changed under us on main
    (c4daffb): a cut tool is now an IGNORE region rather than a removed annotation, so it is
    neither a false negative when missed nor a false positive when detected, and
    `HungarianInstanceEvaluator` reads `iscrowd` itself. That is why the prediction-side
    `DropTruncatedPredictions` wrapper could be deleted - it existed only to repair the other
    half of the deletion, where a correct detection of a cut tool had no GT left to match
    (26 dropped GT raised false positives 19 -> 45).

    The returned count is therefore non-crowd annotations, not `len(annotations)`: the
    truncated ones are still in the dicts, and counting them would report 1227 where 1201 is
    scored. 1201 is the 8-class baseline's protocol ("stride 4, excl-truncated, 1201 GT"),
    and stride 4 plus this filter still reproduces it exactly on `val_seta_mm`
    (4802 GT -> 4693 untruncated -> 1201 at stride 4).

    `EXCLUDE_TRUNCATED_INSTANCES: true` is already set in the detector's own config.yaml, but
    it does nothing unless something calls this - the flag was inherited and inert.

    Frame striding and annotation filtering commute, so the order relative to
    `apply_val_stride` does not change any count.
    """

    def _scored(name: str) -> int:
        return sum(
            1
            for d in DatasetCatalog.get(name)
            for a in d["annotations"]
            if a.get("iscrowd", 0) != 1
        )

    if not enabled or dataset_name in TRUNCATED_FILTERED:
        return _scored(dataset_name)
    d2 = CfgNode({
        "DATASETS": CfgNode({"TEST": (dataset_name,)}),
        "INPUT": CfgNode({"EXCLUDE_TRUNCATED_INSTANCES": True}),
    })
    apply_truncated_instance_filter(d2)
    TRUNCATED_FILTERED.add(dataset_name)
    return _scored(dataset_name)


def dataset_class_names(dataset_name: str) -> list[str]:
    """`thing_classes` for a registered dataset - the only source of the class count.

    The head is sized from this and the label space IS this, so there is nothing to keep
    in sync. The two mistakes a stated num_classes used to catch are caught here.
    """
    names = list(getattr(MetadataCatalog.get(dataset_name), "thing_classes", []) or [])
    if not names:
        raise ValueError(
            f"{dataset_name} has no thing_classes - it is not registered. On a new host "
            "this is usually the missing `training/images` symlink, which makes every "
            "dataset registration silently skip."
        )
    if len(names) < 2:
        raise ValueError(
            f"{dataset_name} has {len(names)} class ({names}): the classifier needs a "
            "multiclass registration (train_seta_mm has 8), not a *_singleclass one - "
            "those exist for the class-agnostic DETECTOR."
        )
    return names


class StageTimer:
    """Wall-clock accumulator per pipeline stage, for the DoD's timing report."""

    def __init__(self) -> None:
        self.total: dict[str, float] = defaultdict(float)
        self.calls: dict[str, int] = defaultdict(int)

    @contextmanager
    def __call__(self, stage: str, sync: bool = False):
        if sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if sync and torch.cuda.is_available():
                torch.cuda.synchronize()
            self.total[stage] += time.perf_counter() - t0
            self.calls[stage] += 1

    def reset(self) -> None:
        self.total.clear()
        self.calls.clear()

    def table(self, n_frames: int | None = None) -> str:
        rows = sorted(self.total.items(), key=lambda kv: -kv[1])
        grand = sum(self.total.values()) or 1e-9
        out = [f"{'stage':<16}{'total s':>10}{'calls':>8}{'ms/call':>10}{'% time':>9}"]
        if n_frames:
            out[0] += f"{'ms/frame':>11}"
        for stage, tot in rows:
            n = self.calls[stage] or 1
            line = f"{stage:<16}{tot:>10.2f}{self.calls[stage]:>8}{1000 * tot / n:>10.1f}{100 * tot / grand:>8.1f}%"
            if n_frames:
                line += f"{1000 * tot / n_frames:>11.1f}"
            out.append(line)
        out.append(f"{'TOTAL':<16}{grand:>10.2f}")
        return "\n".join(out)


@dataclass
class CropRecord:
    """Provenance for one crop - drives the step-4 contact sheet and debugging."""

    frame_index: int
    file_name: str
    pred_index: int
    rotation: int
    label: int
    class_name: str
    best_iou: float
    runner_up_iou: float
    verdict: str
    score: float


@dataclass
class CropBatch:
    crops: torch.Tensor                       # (B, C, S, S) normalized, C = cfg.in_chans
    labels: torch.Tensor                      # (B,) long
    # (B, AUX_DIM) crops.scale_feats of each crop's frame mask. Always filled, whatever
    # cfg.scale_film says, so the crop paths cannot diverge by config (spec Step 12).
    aux: torch.Tensor
    records: list[CropRecord] = field(default_factory=list)
    verdict_counts: dict[str, int] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.crops.shape[0])


class CropPipeline:
    def __init__(
        self,
        cfg: CropClsConfig,
        detector: FrozenSingleClassDetector,
        dataset_name: str,
        is_train: bool = True,
        timer: StageTimer | None = None,
    ):
        self.cfg = cfg
        self.detector = detector
        self.dataset_name = dataset_name
        self.is_train = is_train
        self.timer = timer if timer is not None else StageTimer()

        self.dicts = DatasetCatalog.get(dataset_name)
        self.class_names = dataset_class_names(dataset_name)

        # is_train=True always: the mapper drops `annotations` entirely when False, and
        # the assignment step needs GT masks on val as much as on train. What varies is
        # whether any frame-level augmentation runs, which is `cfg.augment_frames`.
        if cfg.augment_frames and is_train:
            d2 = detector.d2_cfg.clone()
            d2.defrost()
            # Rotation moves to crop level: doing it here would cost a detector forward
            # per rotation instead of one forward for all four.
            d2.INPUT.RANDOM_ROTATION = False
            d2.freeze()
            self.mapper = Hdf5CocoInstanceDatasetMapper(d2, is_train=True)
        else:
            # Identity transform: the detector sees the clean frame, exactly as at
            # inference, and its masks are then deterministic per frame.
            self.mapper = Hdf5CocoInstanceDatasetMapper(
                is_train=True,
                tfm_gens=[],
                image_format=detector.d2_cfg.INPUT.FORMAT,
                min_visibility=detector.d2_cfg.INPUT.MIN_VISIBILITY,
            )

        # Crop-level augmentation replaces the frame-level pipeline; val gets none.
        self.photometric = (
            PhotometricJitter() if (is_train and cfg.photometric) else None
        )
    def __len__(self) -> int:
        return len(self.dicts)

    def _map_frames(self, indices: list[int]) -> list[dict]:
        # Index the pool with d[i]; iterating a LivePoolDataset with `for r in d` is slow.
        return [self.mapper(copy.deepcopy(self.dicts[i])) for i in indices]

    def build_batch(self, indices: list[int], rotations: list[int] | None = None) -> CropBatch:
        """One classifier batch from `len(indices)` frames."""
        rots = self.cfg.rotations if rotations is None else rotations
        t = self.timer

        with t("augment"):
            mapped = self._map_frames(indices)
        images = [m["image"] for m in mapped]

        with t("detector", sync=True):
            dets = self.detector.detect(images, amp=self.cfg.amp)

        crops: list[torch.Tensor] = []
        labels: list[int] = []
        auxs: list[torch.Tensor] = []
        records: list[CropRecord] = []
        counts: dict[str, int] = defaultdict(int)

        for frame_i, (m, det) in zip(indices, zip(mapped, dets)):
            inst = m.get("instances", None)
            if inst is None or not inst.has("gt_masks"):
                continue
            gt_masks = inst.gt_masks
            if not torch.is_tensor(gt_masks):          # PolygonMasks/BitMasks safety
                gt_masks = gt_masks.tensor
            gt_classes = inst.gt_classes

            with t("mask_to_cpu"):
                pred_masks = det.masks.cpu()

            with t("assign"):
                a = assign_gt_classes(
                    pred_masks,
                    gt_masks,
                    gt_classes,
                    iou_floor=self.cfg.iou_floor,
                    runner_up_ceiling=self.cfg.runner_up_ceiling,
                )
            for v, n in a.counts().items():
                counts[v] += n

            image = m["image"].float()
            # Union of every OTHER prediction in this frame, built once per frame and
            # subtracted per instance - O(K) instead of the O(K^2) a per-instance loop costs.
            occ_all = pred_masks.any(0) if self.cfg.occluder_channel else None
            with t("crop"):
                for k in torch.where(a.keep)[0].tolist():
                    tight = mask_tight_bbox(pred_masks[k])
                    if tight is None:
                        continue
                    others = (occ_all & ~pred_masks[k]) if occ_all is not None else None
                    out = letterbox_crop(
                        image,
                        pred_masks[k],
                        self.cfg.crop_size,
                        box=tight,
                        others=others,
                        pad_frac=self.cfg.pad_frac,
                        photometric=self.photometric,
                        mask_only=self.cfg.mask_only,
                        blackout=self.cfg.blackout,
                        dilate=self.cfg.mask_dilate,
                        close=self.cfg.mask_close,
                    )
                    if out is None:          # empty after the bbox round-trip
                        continue
                    label = int(a.labels[k])
                    sf = scale_feats(pred_masks[k])       # rotation-invariant: once per mask
                    for deg in rots:
                        auxs.append(sf)
                        crops.append(
                            to_model_input(
                                out, self.cfg.pixel_mean, self.cfg.pixel_std,
                                deg, self.cfg.mask_channel, self.cfg.occluder_channel,
                            )
                        )
                        labels.append(label)
                        records.append(
                            CropRecord(
                                frame_index=frame_i,
                                file_name=str(m.get("file_name", "")),
                                pred_index=k,
                                rotation=deg,
                                label=label,
                                class_name=self.class_names[label],
                                best_iou=float(a.best_iou[k]),
                                runner_up_iou=float(a.runner_up_iou[k]),
                                verdict=a.verdict[k],
                                score=float(det.scores[k]),
                            )
                        )

        with t("collate"):
            if crops:
                batch = torch.stack(crops)
                lab = torch.tensor(labels, dtype=torch.long)
                aux = torch.stack(auxs)
            else:
                s = self.cfg.crop_size
                batch = torch.zeros((0, self.cfg.in_chans, s, s))
                lab = torch.zeros((0,), dtype=torch.long)
                aux = torch.zeros((0, AUX_DIM))

        return CropBatch(
            crops=batch, labels=lab, aux=aux, records=records, verdict_counts=dict(counts)
        )

    def iter_batches(self, seed: int = 0, shuffle: bool | None = None):
        """Infinite (train) or single-pass (val) stream of CropBatch."""
        shuffle = self.is_train if shuffle is None else shuffle
        n = len(self.dicts)
        step = self.cfg.ims_per_batch
        g = torch.Generator().manual_seed(seed)
        if not shuffle:
            for start in range(0, n, step):
                yield self.build_batch(list(range(start, min(start + step, n))))
            return
        while True:
            order = torch.randperm(n, generator=g).tolist()
            for start in range(0, n - step + 1, step):
                yield self.build_batch(order[start : start + step])
