"""Crops on disk: one folder per class, one RGBA PNG per crop.

The folder name gives the class NAME, so there is no sidecar mapping each crop to a label.
The class ORDER still comes from `meta.json`'s `thing_classes`, not from `sorted()` - see
`CropFolder`. `tools/extract_crops.py` writes the folder; this reads it.

Alpha carries the segmentation mask rather than a blacked-out RGB, which is what keeps
photometric jitter usable: it has to run on the raw crop *before* blackout, because contrast
blends toward the image mean and a blacked crop's mean is near zero.

Patches are stored at native resolution, never pre-letterboxed. `mask_dilate` has to stay a
real-pixel operation - dilating an already-resized mask by 8 px would mean a different real
radius for a large instrument than a small one, which is not the operation that was measured.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .augment import PhotometricJitter
from .config import CropClsConfig
from .crops import letterbox_crop, mask_tight_bbox, pad_box, scale_feats, to_model_input
from .length_prior import require_border

__all__ = [
    "CropFolder",
    "build_loader",
    "load_crop",
    "patch_bbox",
    "save_crop",
    "unclipped_border",
    "write_frame_crops",
]


def patch_bbox(mask: torch.Tensor, pad_frac: float) -> tuple[int, int, int, int] | None:
    """The mask's tight box grown by `pad_frac` per axis, clipped to the frame.

    What gets STORED. The loader crops to its own `cfg.pad_frac` within this, so extracting at
    a larger fraction than you intend to train at costs disk and buys a sweep.
    """
    tight = mask_tight_bbox(mask)
    if tight is None:
        return None
    H, W = mask.shape
    return pad_box(tight, pad_frac, H, W)


def unclipped_border(
    tight: tuple[int, int, int, int], patch: tuple[int, int, int, int], H: int, W: int
) -> int | None:
    """Smallest stored border, over the sides that are NOT at the frame edge.

    `mask_close` leaks foreground inward from the array edge by up to its radius, so it needs
    real border to hide the leak in. A side sitting on the frame edge needs none: there the
    patch edge IS the frame edge, which is the convention `close_mask` already applies on a
    full frame. Returns None when every side is frame-clipped, i.e. nothing to promise.
    """
    b = []
    if patch[0] > 0:
        b.append(tight[0] - patch[0])
    if patch[1] > 0:
        b.append(tight[1] - patch[1])
    if patch[2] < W:
        b.append(patch[2] - tight[2])
    if patch[3] < H:
        b.append(patch[3] - tight[3])
    return min(b) if b else None


def occ_path(path: Path) -> Path:
    """Sidecar holding the occluder union for `path`. RGBA has no free channel left - alpha
    is already the instance mask - so the second raster is its own 1-bit file."""
    return path.with_suffix(".occ.png")


def save_crop(
    path: Path, rgb: torch.Tensor, mask: torch.Tensor, others: torch.Tensor | None = None
) -> None:
    """(3,h,w) uint8 + (h,w) bool -> RGBA PNG. Lossless; a float frame would quantize.

    `others` (the union of the OTHER instances over the same patch) is written beside it as
    a grayscale sidecar, not squeezed into the RGBA.
    """
    if rgb.dtype != torch.uint8:
        raise TypeError(f"expected uint8 RGB, got {rgb.dtype}: PNG would quantize it")
    a = mask.numpy().astype(np.uint8) * 255
    Image.fromarray(np.dstack([rgb.permute(1, 2, 0).numpy(), a]), "RGBA").save(
        path, optimize=True
    )
    if others is not None:
        Image.fromarray(others.numpy().astype(np.uint8) * 255, "L").save(
            occ_path(path), optimize=True
        )


def load_crop(
    path: Path, want_others: bool = False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Fixed 3-arity, third element None unless `want_others`. See `letterbox_crop` for why
    the None matters: a silently-zero occluder channel turns the arm into its own control."""
    arr = np.array(Image.open(path).convert("RGBA"))
    rgb = torch.from_numpy(arr[..., :3].copy()).permute(2, 0, 1)
    mask = torch.from_numpy(arr[..., 3].copy()) > 127
    others = None
    if want_others:
        op = occ_path(path)
        if not op.is_file():
            raise FileNotFoundError(
                f"{op} is missing, so this crop has no occluder raster. The folder predates "
                "occluder extraction; re-run tools/extract_crops.py. Defaulting to zeros "
                "would make an occluder_channel run numerically its own control."
            )
        others = torch.from_numpy(np.array(Image.open(op).convert("L")).copy()) > 127
        if others.shape != mask.shape:
            raise ValueError(f"{op} is {tuple(others.shape)}, patch is {tuple(mask.shape)}")
    return rgb, mask, others


def write_frame_crops(
    out: Path,
    stem: str,
    image: torch.Tensor,
    masks: torch.Tensor,
    asg,
    class_names: list[str],
    pad_frac: float,
    occluders: torch.Tensor | None = None,
) -> list[dict]:
    """Write one frame's kept crops into `out/<class>/`. Returns a row per DETECTION.

    Split out of the extraction script so it is testable without a detector: everything that
    can go wrong here - which detections get a crop, what the filename is, what the row says
    about the ones that do not - is decidable from an Assignment and a mask tensor.
    """
    rows = []
    for k in range(len(asg)):
        row = {
            "frame": stem, "pred_index": k, "verdict": asg.verdict[k],
            "best_iou": round(float(asg.best_iou[k]), 4),
            "runner_up_iou": round(float(asg.runner_up_iou[k]), 4),
            "class_name": "", "crop": "", "border_px": "",
        }
        if bool(asg.keep[k]):
            box = patch_bbox(masks[k], pad_frac)
            if box is None:
                # A kept detection can still have an empty mask. It gets a row, not silence.
                row["verdict"] = "dropped_empty_box"
            else:
                x0, y0, x1, y1 = box
                H, W = masks[k].shape
                bp = unclipped_border(mask_tight_bbox(masks[k]), box, H, W)
                row["border_px"] = "" if bp is None else bp
                cls = class_names[int(asg.labels[k])]
                (out / cls).mkdir(parents=True, exist_ok=True)
                rel = f"{cls}/{stem}_p{k:03d}.png"
                # Every OTHER instance over this patch. Built from the frame-level union so
                # the stored raster matches what the live path computes for the same crop.
                oth = None
                if occluders is not None:
                    oth = (occluders & ~masks[k])[y0:y1, x0:x1]
                save_crop(out / rel, image[:, y0:y1, x0:x1], masks[k][y0:y1, x0:x1], oth)
                row["class_name"], row["crop"] = cls, rel
        rows.append(row)
    return rows


class CropFolder(torch.utils.data.Dataset):
    """`root/<class_name>/*.png`. The folder gives the class NAME; `meta.json` gives the ORDER.

    The order must NOT come from `sorted()` of the directory names. For this data the two
    disagree: `thing_classes` is
    `[forcep01..04, scalpel01, hammer01, sharpspoon01, scarstick01]` while sorted() puts
    hammer before scalpel and scarstick before sharpspoon. The four forceps coincide, so a
    sorted order trains four classes correctly and silently mislabels the other four - which
    is exactly what it did: forceps scored 0.27-0.82 and the rest 0.00-0.03.
    """

    def __init__(
        self,
        root: str | Path,
        cfg: CropClsConfig,
        is_train: bool = True,
        seed: int = 0,
    ):
        self.root = Path(root)
        self.cfg = cfg
        dirs = sorted(p for p in self.root.iterdir() if p.is_dir())
        if not dirs:
            raise FileNotFoundError(f"{self.root} has no class folders; run tools/extract_crops.py")

        meta_path = self.root / "meta.json"
        if not meta_path.is_file():
            raise FileNotFoundError(
                f"{meta_path} is missing, so the class ORDER is unknown. It cannot be inferred "
                "from the directory names: sorted() and the dataset's thing_classes disagree "
                "for this data, and training on one order while evaluating against the other "
                "silently mislabels whichever classes the two orders disagree on. Re-extract "
                "with tools/extract_crops.py, which records thing_classes."
            )
        self.meta = json.loads(meta_path.read_text())
        self.class_names = list(self.meta["thing_classes"])
        idx = {n: i for i, n in enumerate(self.class_names)}
        if cfg.occluder_channel and not self.meta.get("occluders", False):
            raise ValueError(
                f"{meta_path} does not record occluders, so this folder was extracted before "
                "the occluder raster existed. Re-run tools/extract_crops.py. Reading it anyway "
                "would feed a constant-zero channel 4 and silently turn the arm into its own "
                "control (spec_cropcls.md Step 11)."
            )
        unknown = sorted(d.name for d in dirs if d.name not in idx)
        if unknown:
            raise ValueError(
                f"{self.root} has class folders {unknown} that are not in the recorded label "
                f"space {self.class_names}. The crops were extracted against a different "
                "dataset; re-extract."
            )
        self.items: list[tuple[Path, int]] = [
            (f, idx[d.name]) for d in dirs for f in sorted(d.glob("*.png"))
            if not f.name.endswith(".occ.png")
        ]
        if not self.items:
            raise FileNotFoundError(f"{self.root} has class folders but no crops")
        self._check_border()
        self.photometric = PhotometricJitter() if (is_train and cfg.photometric) else None
        self.rotations = list(cfg.rotations) if is_train else [0]
        # Own generator: the global RNG is also seeded for model init, so drawing rotations
        # from it would couple augmentation to unrelated RNG consumption.
        self.g = torch.Generator().manual_seed(seed + 7919)

    def _check_border(self) -> None:
        """The stored patch is a ceiling: it bounds `pad_frac`, and it bounds the close window.

        Two different things need the stored border, so two checks.

        `pad_frac` is the crop region, and the loader cannot invent pixels the PNG does not
        hold, so training above the extracted fraction is impossible rather than merely
        inadvisable.

        `mask_close` is subtler: `close_mask` is `~dilate_mask(~d, r)` and `max_pool2d` pads
        with -inf, so the array edge reads as FOREGROUND during the erode and foreground leaks
        inward by up to `radius`. On a full frame `letterbox_crop` widens its own window to
        `tight +- 2d` so the leak lands outside the box; here the array edge is the stored patch,
        so the leak is invisible only while the border covers it. Checked against the MEASURED
        `min_border_px` rather than a configured number: the patch is clipped at the frame, so a
        nominal fraction is not a promise. Plain dilation needs no border at all - every mask
        pixel is inside its own tight box - which `test_plain_dilation_needs_no_margin_at_all`
        pins.
        """
        cfg = self.cfg
        if "pad_frac" not in self.meta:
            raise ValueError(
                f"{self.root}/meta.json has no `pad_frac`, so it predates the crop-padding "
                "change and its PNGs are the mask's TIGHT box. Defaulting to 0 would train on "
                f"tight crops under a config asking for pad_frac={cfg.pad_frac}, which does not "
                "raise and only reports a worse number. Re-extract with tools/extract_crops.py."
            )
        stored = float(self.meta["pad_frac"])
        if cfg.pad_frac > stored + 1e-9:
            raise ValueError(
                f"pad_frac={cfg.pad_frac} but these crops were extracted at pad_frac={stored}. "
                "The border pixels are not in the PNGs and cannot be invented. Re-extract at "
                f"pad_frac >= {cfg.pad_frac}, or train at <= {stored}."
            )
        if cfg.length_prior:
            require_border(self.meta, str(self.root))
        if not (cfg.mask_close and cfg.mask_dilate > 0):
            return
        border = self.meta.get("min_border_px")
        if border is None:
            raise ValueError(
                f"mask_close with mask_dilate={cfg.mask_dilate} needs to know the stored border, "
                f"but {self.root}/meta.json has no `min_border_px`. Re-extract."
            )
        # `>=`, not `>`. While the crop was the tight box, a border equal to the radius was
        # enough: the dilated mask reached the patch edge but the close_mask leak landed in the
        # border, outside the crop. pad_frac extends the crop region to the patch edge, so that
        # leak is now INSIDE the crop. Measured: a 204x160 mask at pad_frac=0.10 stores a 16 px
        # border, and at mask_dilate=16 the folder crop differs from the live crop by 1053 of
        # 96x96 mask pixels while `>` let it through. Exact threshold is border >= dilate + 1.
        if cfg.mask_dilate >= border:
            raise ValueError(
                f"mask_close with mask_dilate={cfg.mask_dilate} needs every stored border > "
                f"{cfg.mask_dilate} px, but the thinnest in this folder is {border} px. "
                f"pad_frac={stored} is relative, so this is the SINGLE thinnest detection, not "
                "the typical one - a 22x3 px sliver gets a 0 px border at 0.10 and gates the "
                "whole folder. The closing window would be truncated at the patch edge and that "
                "crop would quietly stop matching the live path, so this raises rather than "
                "differing. Fixes, in order of cost: drop mask_close (it measured +0.4 and -0.2 "
                "in the sweep, cropcls-results.md section 4), re-extract at a pad_frac large "
                "enough for the thinnest detection, or check crops.csv `border_px` and whether "
                "a 3 px-tall detection should be passing the assignment rule at all."
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        path, label = self.items[i]
        rgb, mask, others = load_crop(path, want_others=self.cfg.occluder_channel)
        cfg = self.cfg
        tight = mask_tight_bbox(mask)
        if tight is None:
            raise ValueError(f"{path} has an empty mask; re-extract")
        # Same call as CropPipeline.build_batch, including box= the UNDILATED tight box and
        # the same pad_frac. A passed box overrides letterbox_crop's dilated one, so dilation
        # grows the mask inside the region but never widens it; only pad_frac widens it. Here
        # `mask` IS the stored patch, so pad_box's clip lands on the patch edge - which is why
        # cfg.pad_frac above the stored one has to raise rather than silently crop short.
        out = letterbox_crop(
            rgb.float(), mask, cfg.crop_size, box=tight, pad_frac=cfg.pad_frac,
            others=others,
            photometric=self.photometric, mask_only=cfg.mask_only, blackout=cfg.blackout,
            dilate=cfg.mask_dilate, close=cfg.mask_close,
        )
        if out is None:
            raise ValueError(f"{path} produced no crop; re-extract")
        deg = self.rotations[int(torch.randint(len(self.rotations), (), generator=self.g))]
        # The stored alpha is the undilated predicted mask at native resolution and the patch
        # holds all of it, so this equals scale_feats of the full-frame mask. Never the PNG's
        # own size: that is the pad_frac-grown box (spec Step 12).
        return to_model_input(
            out, cfg.pixel_mean, cfg.pixel_std, deg, cfg.mask_channel,
            cfg.occluder_channel,
        ), scale_feats(mask), label


def build_loader(cfg: CropClsConfig, root: str | Path, is_train: bool = True, seed: int = 0):
    ds = CropFolder(root, cfg, is_train=is_train, seed=seed)
    if is_train and len(ds) < cfg.crops_per_batch:
        raise ValueError(
            f"{len(ds)} crops but crops_per_batch={cfg.crops_per_batch} with drop_last: the "
            "loader would yield nothing and the training loop would spin silently"
        )
    return torch.utils.data.DataLoader(
        ds,
        batch_size=cfg.crops_per_batch,
        shuffle=is_train,
        drop_last=is_train,
        num_workers=cfg.num_workers,
        persistent_workers=cfg.num_workers > 0,
    )
