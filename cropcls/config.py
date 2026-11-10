"""Plain-dataclass config for the crop classifier.

No yacs, no detectron2 CfgNode, no registry: the detector side is driven by its own
run `config.yaml` (merged in `detector.py`), and everything here is classifier-only.
"""

from __future__ import annotations

import dataclasses
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# MaskDINO's own pixel statistics, which are ImageNet's scaled to 0-255 (RGB).
# Reused verbatim so the classifier's preprocessing cannot drift from the frontend's.
PIXEL_MEAN = [123.675, 116.28, 103.53]
PIXEL_STD = [58.395, 57.12, 57.375]


@dataclass
class CropClsConfig:
    # --- frozen detector -----------------------------------------------------
    detector_checkpoint: str
    # Defaults to <checkpoint dir>/config.yaml - the run's own resolved config.
    detector_config: str = ""
    # Predictions below this score are not turned into crops. 0.5 matches
    # MODEL.MaskDINO.TEST.HUNGARIAN_EVAL.SCORE_THRESH, so the crop population
    # matches the population the baseline f1 was measured on.
    detector_score_thresh: float = 0.5

    # --- data ----------------------------------------------------------------
    # No num_classes: it is len(thing_classes) of train_dataset, read by
    # cropcls.data.dataset_class_names.
    train_dataset: str = "train_seta_mm"
    val_dataset: str = "val_seta_mm"
    # Keep every Nth val frame. 4 matches the 8-class baseline's protocol, so classaware_f1
    # is computed over the same frames. Applied ONCE to the catalog entry, which is what makes
    # it reach the in-training monitor, --eval-only and --hungarian alike. 1 = no striding.
    val_stride: int = 4
    # Drop frame-border-truncated GT before scoring. The 8-class baseline was measured this
    # way ("stride 4, excl-truncated, 1201 GT"), and stride 4 + this filter reproduces 1201
    # exactly - so leaving it off scores 26 extra clipped instruments the baseline never saw.
    exclude_truncated: bool = True

    # --- augmentation --------------------------------------------------------
    # The detector is FROZEN, so augmenting its input regularizes nothing - it only
    # perturbs the masks, and the letterbox discards the scale/position variation
    # anyway. Keeping it off makes the detector's output deterministic per frame
    # (hence cacheable) and matches its inference condition. Augmentation happens at
    # crop level instead. Set True only to A/B against the original recipe.
    augment_frames: bool = False

    # Crop-level augmentation, standing in for what the frame-level pipeline did.
    photometric: bool = True     # brightness/contrast/saturation/lighting, pre-blackout
    mask_only: bool = False      # shape-only ablation: silhouette instead of RGB
    # Grow the mask by N pixels before cropping and blacking out. 0 = tight silhouette.
    # Sits between blackout (no context at all) and blackout=False (all of it, which acts
    # as a distractor): it recovers the boundary transition and any real instrument pixels
    # the detector's mask clipped, without admitting the surroundings.
    mask_dilate: int = 0
    # Use closing (dilate+erode) instead of plain dilation at that radius: repairs
    # fragmented masks without admitting any surrounding pixels.
    mask_close: bool = False

    blackout: bool = True        # zero everything outside the mask. False keeps the real
                                 # background - tests whether surrounding context helps,
                                 # which the spec's blackout throws away by design.

    # --- static crop folder (spec_static_crops.md) ---------------------------
    # Folder of per-class crop directories written by tools/extract_crops.py.
    # Empty = train from the live pipeline instead.
    crop_folder: str = ""
    # Crop region = the mask's tight box grown by this fraction of its own width/height on
    # each side. Unlike `mask_dilate` this widens the field of view and leaves the mask alone,
    # so the border is real background. Relative rather than pixels because the letterbox puts
    # every crop on the same canvas; see `pad_box`. For a static folder this is what the LOADER
    # crops to, bounded above by the `pad_frac` the folder was extracted at.
    pad_frac: float = 0.10
    # Batch size in CROPS. ims_per_batch counts FRAMES and keeps that meaning.
    crops_per_batch: int = 64
    num_workers: int = 4

    # Override the stride of the trunk's FIRST conv, 0 = leave the trunk's own. The early
    # reduction is what destroys thin structure: measured, forcep03/04 have a 0.40-cell median
    # half-width after convnext_tiny's 4x4/stride-4 patchify at crop_size 224, and they are the
    # two classes that collapse. Halving it doubles every stage (112/56/28/14 against
    # 56/28/14/7), which is crop_size 448's spatial resolution without upsampling and at an
    # identical parameter count. `output_stride` cannot do this - it moves only the last stage.
    stem_stride: int = 0

    # --- crops ---------------------------------------------------------------
    crop_size: int = 224
    # Append the instance mask as a 4th input channel, 0.0/1.0, after normalization - the
    # 0-255 RGB statistics mean nothing for an occupancy channel. 
    mask_channel: bool = False
    # Append the union of the OTHER predicted instances as a 5th channel, same 0.0/1.0. Tells
    # the trunk whether a silhouette is cut by another tool or ends in free space - a
    # distinction nothing in the input carries today (spec_cropcls.md Step 11). Predicted
    # masks, never GT: GT occluders would be a train-only oracle. Always last, so turning it
    # on does not renumber the channel a mask_channel checkpoint was trained with.
    occluder_channel: bool = False
    # FiLM the pooled trunk features with the instance's real size (crops.scale_feats), which
    # the letterbox throws away - the set-C classes differ only in length (spec Step 12). The
    # FiLM output layer starts at zero, so at iteration 0 the model equals scale_film=false.
    scale_film: bool = False
    # Lossless pixel permutations on a square canvas; applied after cropping so one
    # detector forward yields 4x the classifier samples (spec section 6, step 5). Live path
    # only - the static cache (spec_static_crops.md) samples one rotation per draw, because
    # bundling all four in one batch correlates the BatchNorm statistics.
    rotations: list[int] = field(default_factory=lambda: [0, 90, 180, 270])
    pixel_mean: list[float] = field(default_factory=lambda: list(PIXEL_MEAN))
    pixel_std: list[float] = field(default_factory=lambda: list(PIXEL_STD))

    # How the score fed to the evaluator is formed.
    #   "detector_x_class" - objectness * classifier softmax max. Exact analogue of
    #       MaskDINO's `sigmoid(class_logit) * mask_score`, so it is the comparable
    #       setting - but it couples localization to classifier confidence: a weak
    #       classifier pushes good masks below SCORE_THRESH and depresses even the
    #       class-AGNOSTIC metric.
    #   "detector" - objectness only. Diagnostic: classagnostic_f1 then reads out pure
    #       localization (should match the frontend's own 0.9763).
    score_mode: str = "detector_x_class"

    # --- GT class assignment -------------------------------------------------
    iou_floor: float = 0.5
    runner_up_ceiling: float = 0.2

    # --- model ---------------------------------------------------------------
    # The classifier's own trunk: any timm image-classification backbone, pluggable -
    # the segmentation map comes from MaskDINO, so there is no decoder of any kind here.
    # Named "trunk", not "encoder", because in this repo "encoder"/"decoder" already mean
    # MaskDINO's pixel decoder and transformer decoder (cf.
    # RECLASSIFY_FINETUNE.UNFREEZE_ENCODER) - and *those* are frozen forever.
    trunk: str = "resnet50"
    trunk_pretrained: bool = True
    freeze_trunk: bool = False
    # Classifier checkpoint to start from: an init/resume for training, the thing being
    # scored under --eval-only. Lives here and not on the command line for the same
    # reason detector_checkpoint does - a run's config.json is its own record, so the
    # checkpoint it was built on is reproducible from the file rather than from shell
    # history. Empty = train from the trunk's pretrained weights.
    weights: str = ""

    # --- solver --------------------------------------------------------------
    base_lr: float = 1e-4
    weight_decay: float = 0.05
    max_iter: int = 20000
    ims_per_batch: int = 4          # FRAMES per detector forward
    warmup_iters: int = 250
    checkpoint_period: int = 2000
    eval_period: int = 1000
    amp: bool = True
    seed: int = 0

    # --- io ------------------------------------------------------------------
    output_dir: str = "runs/cropcls_phase1_seta"

    @property
    def in_chans(self) -> int:
        """One definition of the channel count. The model and the crop pipeline disagreeing
        about it surfaces only as a conv shape error on the first batch, so neither side
        derives it independently. Not a dataclass field: `asdict` skips it, keeping
        config.json a record of inputs, and `load_config` keeps rejecting it as a YAML key."""
        return 3 + int(self.mask_channel) + int(self.occluder_channel)

    def __post_init__(self) -> None:
        if not self.detector_config:
            self.detector_config = str(
                Path(self.detector_checkpoint).parent / "config.yaml"
            )

    def validate(self) -> None:
        """Fail at startup rather than mid-training on a mistyped path."""
        paths = [
            ("detector_checkpoint", self.detector_checkpoint),
            ("detector_config", self.detector_config),
        ]
        if self.weights:  # empty is legal: training from the trunk's pretrained weights
            paths.append(("weights", self.weights))
        for label, p in paths:
            if not Path(p).is_file():
                raise FileNotFoundError(f"{label} does not exist: {p}")
        if self.crop_size <= 0:
            raise ValueError(f"crop_size must be positive, got {self.crop_size}")
        if not 0.0 <= self.pad_frac < 1.0:
            raise ValueError(f"need 0 <= pad_frac < 1, got {self.pad_frac}")
        if not 0.0 <= self.runner_up_ceiling <= self.iou_floor <= 1.0:
            raise ValueError(
                "need 0 <= runner_up_ceiling <= iou_floor <= 1, got "
                f"{self.runner_up_ceiling} / {self.iou_floor}"
            )
        bad = [r for r in self.rotations if r % 90 != 0]
        if bad:
            raise ValueError(
                f"rotations must be multiples of 90 (lossless on a square crop), got {bad}"
            )
        # Both of these would train for hours and measure nothing. Raising beats warning:
        # the cost of noticing at iteration 4000 is the whole run.
        if self.mask_channel and self.mask_only:
            raise ValueError(
                "mask_channel with mask_only: the RGB channels are already the silhouette, so "
                "channel 3 is an affine rescaling of channel 0 (mask*255 normalized to "
                "~+2.25/-2.12, against 1.0/0.0) and carries no information the trunk "
                "cannot already read. Pick one."
            )
        if self.scale_film and self.augment_frames:
            raise ValueError(
                "scale_film with augment_frames: frame-level augmentation rescales the frame "
                "before the detector, so training aux (crops.scale_feats, in frame pixels) "
                "would be jittered while val and export see native 1024 px. FiLM would learn "
                "that size is noise."
            )
        if self.occluder_channel and not self.mask_channel:
            raise ValueError(
                "occluder_channel without mask_channel: in_chans would be 4 with the OCCLUDER "
                "union at channel 3 and the instance's own mask never fed at all. That is not "
                "the documented layout (RGB, own mask, occluders) and nobody asked for it."
            )
        if self.occluder_channel and self.freeze_trunk:
            raise ValueError(
                "occluder_channel with freeze_trunk: the new input weights are initialized to "
                "zero and freezing pins them there, so the channel is a no-op for the whole "
                "run and the arm reproduces the 4-channel result exactly."
            )
        if self.mask_channel and self.freeze_trunk:
            raise ValueError(
                "mask_channel with freeze_trunk: the new input weights are initialized to zero "
                "and freezing pins them there, so the mask channel is a no-op for the whole "
                "run and the arm reproduces the 3-channel result exactly."
            )

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


# Fields that existed in a shipped config and no longer do. `to_dict` is `dataclasses.asdict`,
# so every `runs/*/config.json` carries whatever fields were live when it was written, and
# `tools/confidence_metrics.py` and `tools/viz_misclassified.py` re-load those to diagnose a
# finished run. Dropping a field would therefore make every earlier run unloadable, so a
# retired name is skipped with a warning while any other unknown key still raises - the typo
# guard is the point of that check and it keeps working.
RETIRED_KEYS = frozenset({"margin"})


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> CropClsConfig:
    """Read a YAML file into CropClsConfig. Unknown keys are an error, not ignored."""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    if overrides:
        raw.update(overrides)
    retired = sorted(RETIRED_KEYS & set(raw))
    if retired:
        warnings.warn(
            f"ignoring retired config key(s) {retired} in {path}. `margin` grew the stored PNG "
            "and never the crop; `pad_frac` replaces it and does widen the field of view, so a "
            "run predating the change trained at pad_frac=0.0 whatever its margin said.",
            stacklevel=2,
        )
        if "margin" in raw and "pad_frac" not in raw:
            # Not the dataclass default. A config written before `pad_frac` existed trained on
            # the tight box, so inheriting the new 0.10 default would re-diagnose a finished run
            # with a field of view it never saw - and silently, since nothing here would raise.
            raw["pad_frac"] = 0.0
        raw = {k: v for k, v in raw.items() if k not in RETIRED_KEYS}
    known = {f.name for f in dataclasses.fields(CropClsConfig)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown config keys in {path}: {sorted(unknown)}")
    cfg = CropClsConfig(**raw)
    cfg.validate()
    return cfg
