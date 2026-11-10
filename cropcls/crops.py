"""Letterboxed, background-blacked instance crops.

Pure tensor ops - no detectron2, no model, no config object - so this is unit-testable
on its own (spec_cropcls.md section 6, step 2).

The crop is taken from the **mask's tight bounding box**, not from the detector's learned
`pred_boxes`, so the crop and the blackout stay self-consistent - a learned box can violate
that in either direction, clipping the mask or padding it with slack.

`pad_frac` then grows that box by a fraction of its own size on each side. It is the only thing
here that widens the field of view: `dilate` grows the MASK within the region, so with
`blackout=True` every non-zero pixel is still inside the mask, while a padded border is real
background and shows only when `blackout=False`.
"""

from __future__ import annotations

import math

import cv2
import torch
import torch.nn.functional as F

__all__ = [
    "close_mask",
    "dilate_mask",
    "letterbox_crop",
    "mask_tight_bbox",
    "normalize_crop",
    "pad_box",
    "rotate_k90",
    "scale_feats",
    "to_model_input",
]

SCALE_FEAT_DIM = 4


def pad_box(
    box: tuple[int, int, int, int], pad_frac: float, H: int, W: int
) -> tuple[int, int, int, int]:
    """Grow a box by `pad_frac` of its OWN width/height on each side, clipped to HxW.

    Per-axis and relative, which is the whole point: the letterbox scales every crop onto the
    same canvas, so a fixed pixel border would land as a different fraction of that canvas for
    a 12 px instance than for a 505 px one (the range measured over `crops/train`). Relative
    growth puts the silhouette at 1/(1 + 2*pad_frac) of the canvas edge for every instance.
    """
    if pad_frac <= 0:
        return box
    x0, y0, x1, y1 = box
    dx = round((x1 - x0) * pad_frac)
    dy = round((y1 - y0) * pad_frac)
    return (max(0, x0 - dx), max(0, y0 - dy), min(W, x1 + dx), min(H, y1 + dy))


def dilate_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """Grow a (H, W) boolean mask by `radius` pixels (square structuring element).

    Separable: a square max-pool factors into a 1xk pass and a kx1 pass, turning O(k^2)
    per pixel into O(2k). At radius 16 on a 1024^2 mask that is the difference between
    ~1.1 billion comparisons and ~67 million - on CPU, between 16 s/iter and usable.
    Callers should also crop before dilating (see `letterbox_crop`).
    """
    if radius <= 0:
        return mask.bool()
    r = int(radius)
    k = 2 * r + 1
    x = mask[None, None].float()
    x = F.max_pool2d(x, kernel_size=(1, k), stride=1, padding=(0, r))
    x = F.max_pool2d(x, kernel_size=(k, 1), stride=1, padding=(r, 0))
    return x[0, 0].bool()


def close_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """Morphological closing: dilate by `radius`, then erode by `radius`.

    Fills holes and rejoins broken fragments **without growing the outer silhouette** -
    which separates "the detector's mask is fragmented" from "surrounding context helps".
    The predicted masks here are visibly broken (IoU 0.55-0.83, shafts rendered as dashed
    segments), so repair and context are two different hypotheses that plain dilation
    conflates.

    Erosion is the complement of the dilation of the complement. `max_pool2d` pads with
    -inf, so outside-frame is treated as foreground during the erode step and an object
    touching the border is not eaten away from it.
    """
    if radius <= 0:
        return mask.bool()
    d = dilate_mask(mask, radius)
    return ~dilate_mask(~d, radius)


def mask_tight_bbox(mask: torch.Tensor) -> tuple[int, int, int, int] | None:
    """Tight (x0, y0, x1, y1) half-open box of a (H, W) boolean mask.

    Returns None for a genuinely empty mask - callers must drop it rather than
    producing an all-black crop or a 0/0 NaN downstream.
    """
    if mask.ndim != 2:
        raise ValueError(f"expected a (H, W) mask, got shape {tuple(mask.shape)}")
    m = mask.bool()
    rows = torch.any(m, dim=1)
    cols = torch.any(m, dim=0)
    if not bool(rows.any()):
        return None
    y = torch.where(rows)[0]
    x = torch.where(cols)[0]
    return int(x[0]), int(y[0]), int(x[-1]) + 1, int(y[-1]) + 1


def letterbox_crop(
    image: torch.Tensor,
    mask: torch.Tensor,
    size: int,
    pad_value: float = 0.0,
    blackout: bool = True,
    dilate: int = 0,
    close: bool = False,
    box: tuple[int, int, int, int] | None = None,
    pad_frac: float = 0.0,
    photometric=None,
    mask_only: bool = False,
    others: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None] | None:
    """Crop to the mask's bbox, black out the background, uniform-scale and pad to square.

    Args:
        image: (3, H, W) float tensor in whatever range the caller uses (not normalized).
        mask:  (H, W) bool tensor, same H/W as `image`.
        size:  output edge length; the result is (3, size, size).
        blackout: zero every pixel outside the mask. False keeps the real background,
            which only the step-4 contact sheet wants (to show what the mask removed).
        close: use morphological closing (dilate then erode) instead of plain dilation,
            repairing the mask without growing the silhouette.
        dilate: grow the mask by this many pixels first, before the blackout. NOTE: this
            grows the MASK, not the crop region, whenever `box` is passed - and every real
            caller passes it. A passed `box` overrides the dilated box below, so dilation fills
            holes and joins fragments *inside* the crop region and never widens the field of
            view; `pad_frac` is the knob that widens it. Do not "fix" this to match an earlier
            version of this docstring: `test_a_passed_box_overrides_the_dilated_box` pins it,
            and the static crop folder is built on it.
        box: crop region override (e.g. a jittered bbox). Defaults to the mask's tight box.
        pad_frac: grow the crop region by this fraction of its own width/height on each side,
            after `box` is resolved and before the crop is taken. Unlike `dilate` this DOES
            widen the field of view, and it leaves the mask alone - the border is real
            background, kept or blacked by `blackout` like any other non-mask pixel. Pass the
            same value at every call site: training and scoring disagreeing here does not
            raise, it just reports a worse number.
        photometric: callable applied to the RAW crop before blackout. It must run
            pre-blackout - contrast blends toward the crop mean, and on a ~95%-black crop
            that mean is near zero, so post-blackout it would behave nothing like the
            frame-level op it replaces.
        others: (H, W) bool union of the OTHER instances' masks in this frame, carried
            through the SAME box, pad_frac, scale and centering as the RGB so it cannot
            misalign with the pixels it describes. Never blacked out, dilated or closed -
            `blackout`/`dilate`/`close` repair the instance's own silhouette and have no
            meaning for a neighbour's. None when the caller has no occluders to supply.
        mask_only: return the silhouette (255 inside the mask, 0 outside) instead of RGB.
            The shape-only ablation: if it scores near the RGB model, the task is shape
            matching and the RGB channels are carrying little.

    Returns:
        (crop, crop_mask, crop_others) where crop is (3, size, size), crop_mask is
        (size, size) bool, and crop_others is (size, size) bool or **None when `others` was
        not supplied**. The arity is fixed so a caller cannot silently unpack the wrong
        shape; the None is what makes a caller that forgot `others` fail loudly in
        `to_model_input` instead of training on an all-zero occluder channel.
        None (the whole result) if the mask is empty or its bbox is degenerate.

    Scaling is uniform on the longest axis - never skewed - and the content is centred so
    that `rotate_k90` keeps it centred.
    """
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"expected a (3, H, W) image, got shape {tuple(image.shape)}")
    if mask.shape != image.shape[-2:]:
        raise ValueError(
            f"mask shape {tuple(mask.shape)} does not match image {tuple(image.shape[-2:])}"
        )

    tight = mask_tight_bbox(mask)
    if dilate > 0:
        if tight is None:
            return None
        # Dilate only the neighbourhood we will crop, not the whole frame. Expanding the
        # tight box by `dilate` first makes this EXACT - every original mask pixel is
        # inside `tight`, so dilation by `dilate` reaches at most the expanded box - while
        # turning a 1024^2 op into one over roughly the instrument's own area.
        H, W = mask.shape
        # Closing needs a wider window than plain dilation, but not for the reason it looks:
        # the mask is empty outside `tight`, so there is no extra mask content to read. It is
        # the border convention in close_mask (`~dilate_mask(~d, r)`) - max_pool pads with
        # -inf, so the window edge reads as FOREGROUND during the erode and foreground leaks
        # inward by up to r. A 2r window puts that leak outside `tight` where it cannot show.
        # Measured consequence for a cropped patch: margin >= dilate is required, exactly.
        m = dilate * (2 if close else 1)
        x0 = max(0, tight[0] - m); y0 = max(0, tight[1] - m)
        x1 = min(W, tight[2] + m); y1 = min(H, tight[3] + m)
        op = close_mask if close else dilate_mask
        sub = op(mask[y0:y1, x0:x1], dilate)
        mask = torch.zeros_like(mask)
        mask[y0:y1, x0:x1] = sub
    tight = mask_tight_bbox(mask)
    if tight is None:
        return None
    x0, y0, x1, y1 = tight if box is None else box
    if pad_frac > 0:
        x0, y0, x1, y1 = pad_box((x0, y0, x1, y1), pad_frac, *mask.shape)
    h, w = y1 - y0, x1 - x0
    if h <= 0 or w <= 0:
        return None

    img = image[:, y0:y1, x0:x1].float()
    msk = mask[y0:y1, x0:x1].bool()
    oth = others[y0:y1, x0:x1].bool() if others is not None else None
    if not bool(msk.any()):          # a jittered box can only ever grow, but be safe
        return None

    if photometric is not None:
        img = photometric(img)
    if mask_only:
        img = msk.unsqueeze(0).float().expand(3, -1, -1).mul(255.0).contiguous()

    # Black out first, so the subsequent bilinear resize bleeds black into the instrument
    # edge (benign, reads as antialiasing) rather than bleeding real background in.
    if blackout:
        img = img * msk.unsqueeze(0)

    scale = size / max(h, w)
    new_h = max(1, min(size, int(round(h * scale))))
    new_w = max(1, min(size, int(round(w * scale))))

    img = F.interpolate(
        img.unsqueeze(0),
        size=(new_h, new_w),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    ).squeeze(0)
    msk = (
        F.interpolate(
            msk.unsqueeze(0).unsqueeze(0).float(),
            size=(new_h, new_w),
            mode="nearest",
        )
        .squeeze(0)
        .squeeze(0)
        .bool()
    )
    if oth is not None:
        oth = (
            F.interpolate(
                oth.unsqueeze(0).unsqueeze(0).float(),
                size=(new_h, new_w),
                mode="nearest",
            )
            .squeeze(0)
            .squeeze(0)
            .bool()
        )

    # Re-apply the resized mask so the resize leaves no soft halo outside the silhouette.
    if blackout:
        img = img * msk.unsqueeze(0)

    crop = img.new_full((3, size, size), float(pad_value))
    crop_mask = torch.zeros((size, size), dtype=torch.bool, device=img.device)
    top = (size - new_h) // 2
    left = (size - new_w) // 2
    crop[:, top : top + new_h, left : left + new_w] = img
    crop_mask[top : top + new_h, left : left + new_w] = msk
    crop_others = None
    if oth is not None:
        crop_others = torch.zeros((size, size), dtype=torch.bool, device=img.device)
        crop_others[top : top + new_h, left : left + new_w] = oth
    return crop, crop_mask, crop_others


def scale_feats(mask: torch.Tensor) -> torch.Tensor:
    """The instance's real size, which the letterbox discards (spec_cropcls.md Step 12).

    From the undilated mask in FRAME pixels, never the letterboxed crop. Uses the minimum-area
    rectangle, not the axis-aligned box: the box's long edge shrinks when the tool is rotated,
    the rectangle's does not, and it is unchanged by the 90-degree rotation augmentation. A
    mask split by an occluder still spans its full extent, so length is lost only when an end
    is hidden. Returns (4,) float32: log2(long/256), log2(short/32), log2(sqrt(area)/64),
    fill - scaled to roughly +-1 for the set-C instrument sizes at 1024 px.
    """
    ys, xs = torch.nonzero(mask.detach().cpu(), as_tuple=True)
    area = float(ys.numel())
    if area == 0:
        raise ValueError("scale_feats of an empty mask")
    pts = torch.stack([xs, ys], 1).numpy().astype("float32")
    (_, _), (w, h), _ = cv2.minAreaRect(pts)
    # minAreaRect spans pixel CENTRES, so an n-px side reads n-1; +1 gives the pixel extent
    # and keeps fill = area / (long*short) at ~1 for a solid bar instead of above it
    long_, short = max(w, h) + 1.0, min(w, h) + 1.0
    return torch.tensor(
        [
            math.log2(long_ / 256.0),
            math.log2(short / 32.0),
            math.log2(math.sqrt(area) / 64.0),
            min(area / (long_ * short), 1.0),
        ],
        dtype=torch.float32,
    )


def rotate_k90(x: torch.Tensor, degrees: int) -> torch.Tensor:
    """Rotate the last two dims by a multiple of 90 degrees.

    Lossless on a square canvas: a pure pixel permutation, no interpolation, so the crop
    and its mask stay exactly aligned. Applied after cropping, which is why one detector
    forward can yield four classifier samples (spec_cropcls.md step 5).
    """
    if degrees % 90 != 0:
        raise ValueError(f"only multiples of 90 are lossless, got {degrees}")
    k = (degrees // 90) % 4
    if k == 0:
        return x
    if x.shape[-1] != x.shape[-2]:
        raise ValueError(
            f"rotate_k90 needs a square canvas, got {tuple(x.shape[-2:])}"
        )
    return torch.rot90(x, k, dims=(-2, -1))


def normalize_crop(
    crop: torch.Tensor, pixel_mean: list[float], pixel_std: list[float]
) -> torch.Tensor:
    """(x - mean) / std with MaskDINO's 0-255 RGB statistics."""
    mean = torch.as_tensor(pixel_mean, dtype=crop.dtype, device=crop.device).view(-1, 1, 1)
    std = torch.as_tensor(pixel_std, dtype=crop.dtype, device=crop.device).view(-1, 1, 1)
    return (crop - mean) / std


def to_model_input(
    out: tuple[torch.Tensor, torch.Tensor],
    pixel_mean: list[float],
    pixel_std: list[float],
    degrees: int = 0,
    mask_channel: bool = False,
    occluder_channel: bool = False,
) -> torch.Tensor:
    """`letterbox_crop`'s output -> the exact tensor the classifier is fed.

    One function for all three crop paths (live, static folder, Hungarian export), because
    the training path and the scoring path disagreeing about the channel layout does not
    raise - it just reports a worse number.

    Takes the pair `letterbox_crop` returns rather than two tensors, so a crop cannot be
    paired with some other instance's mask.

    The mask is appended AFTER normalization: `pixel_mean`/`pixel_std` are MaskDINO's three
    0-255 RGB statistics and an occupancy channel has no business being shifted by them.
    0.0/1.0 keeps its gradient in the same range as normalized RGB (~+-2), so the
    zero-initialized input weights learn at a comparable rate. This also puts it after
    photometric jitter for free - that runs inside `letterbox_crop`, on RGB only, and its
    (3,1,1) tensors would not survive a 4-channel input.

    Rotation is applied once, to all channels together, so neither mask can drift out of
    alignment with the pixels it describes.

    Channel order is fixed: RGB, then the own-instance mask, then the occluder union. The
    occluder is last so that turning `occluder_channel` on does not renumber the channel a
    `mask_channel` checkpoint was trained with.
    """
    crop, crop_mask, crop_others = out
    x = normalize_crop(crop, pixel_mean, pixel_std)
    if mask_channel:
        x = torch.cat((x, crop_mask.unsqueeze(0).to(x.dtype)), dim=0)
    if occluder_channel:
        if crop_others is None:
            # letterbox_crop was called without `others`. Appending zeros here would train
            # and score an occluder channel that is constant 0 - the arm would silently
            # become its own control and report a worse number with no error. Raise instead.
            raise ValueError(
                "occluder_channel=True but letterbox_crop returned crop_others=None, i.e. it "
                "was called without `others=`. Every crop path must supply the occluder union."
            )
        x = torch.cat((x, crop_others.unsqueeze(0).to(x.dtype)), dim=0)
    return rotate_k90(x, degrees)
