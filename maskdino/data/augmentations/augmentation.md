# Copy-paste compositing — reference

`maskdino/data/augmentations/copy_paste.py` (`CopyPasteCompositor`). Enabled per config via
`cfg.INPUT.COPY_PASTE.ENABLED`, wired in by `Hdf5CocoInstanceDatasetMapper` (reclassify phase
only). Combines MBOI's (Badilla-Solorzano et al., IJCARS 2022) cut/rotate/paste with Cut,
Paste and Learn's (Dwibedi et al., arXiv:1708.01642) per-instance randomized blend mode — see
the module's own header comment for the full rationale on what's deliberately *not* copied
from either paper (no flip/rescale, no retry-to-avoid-overlap).

## Pipeline

**Once, in `__init__`** (`_build_instance_cache`): every eligible annotation (`iscrowd == 0`)
across all `source_dicts` is cut out exactly once — decode its RLE mask, crop the source frame
to that mask's bounding box **plus a margin** (`_crop_margin_px`, derived from
`blur_kernel_range[1] // 2 + 1`, clamped to the source frame's bounds), and cache
`(crop, crop_mask, category_id)`. The margin exists so `gaussian_blur_edge`/`box_blur` later
have genuine background pixels to blur with instead of hitting the crop's own edge (see
Limitations).

**Per call** (`__call__`, once per training sample): draw `k ~ U(min_instances, max_instances)`,
then pick `k` cache entries **class-balanced** via `_sample_instance_indices` — a shuffled class
list is walked rather than sampling uniformly over the flat pool, so a class's paste frequency
does not depend on how many renders it happens to have (40 vs 20 would otherwise be a 2x exposure
difference unrelated to the experiment). For `k <= num_classes` every pasted instrument is a
*different* class; beyond that the list wraps.

For each sampled instance: transform it (`_transform_instance` — rotation, optional projective
tilt, optional appearance jitter; never flipped, never rescaled), place it (`_place`, see
Placement below), composite it (`_blend`), then shrink every earlier layer's stored mask by
whatever the new paste's *hard* silhouette covers (`layer["visible"] &= ~full_mask`) so all
returned annotations stay modal (visible-region-only, never amodal).

## Functions

| function | does |
|---|---|
| `_read_colors` | reads an `.hdf5` frame's `colors` array, converts to `image_format`; returns `None` (logged) on a torn/missing read |
| `_build_instance_cache` | one-time cut-out of every eligible instance's crop + mask + category, margin-padded |
| `__call__` | per-sample entry point: sample instances, rotate/place/blend each, update occlusion, return `(composite, annotations)` |
| `_update_modal_annotation` | re-rasterizes one layer's segmentation/bbox/area/`visibility_fraction` from its current `visible` mask |
| `_sample_instance_indices` | picks `k` cache positions class-balanced (shuffled class walk), shuffled again so paste order — i.e. who occludes whom — is not grouped by class |
| `_transform_instance` | rotates crop + mask by `θ ~ U(-rotation_degrees, +rotation_degrees)`, optionally composes a projective tilt into the same matrix, then applies appearance jitter; same interpolation for image and mask (avoids boundary mismatch — see `mask_interp`); also returns `valid_footprint` marking real content vs. the warp canvas's synthetic black padding |
| `_perspective_matrix` / `_tilt_homography` | per-instance out-of-plane viewpoint change, **composed into the rotation matrix** so the crop is resampled once, not twice (see Perspective) |
| `_jitter_appearance` | per-instance brightness / contrast / saturation / hue / defocus — pixels only, never the mask (see Appearance jitter) |
| `_place` | `(py, px)` for the patch: uniform over the canvas, or Gaussian around a frame-specific centre (see Placement); `None` if the patch can't fit at all |
| `_cluster_sigma` / `_sample_cluster_center` | σ and μ of the Gaussian placement component |
| `_index_by_class` | `category_id` -> cache positions, the index `_sample_instance_indices` walks |
| `_normalize_jitter` / `_normalize_perspective` | config node (or dict) -> plain dict/tuple, returning `None` when every knob is a no-op so the hot path skips the feature with one identity check; both accept UPPERCASE (yacs) and lowercase keys |
| `_ensure_rngs` | lazily (re)seeds the private jitter/perspective streams, keyed on DataLoader worker index (see Random streams) |
| `_shift_hue` / `_luma_weights` | hue rotation via 8-bit HSV (0-179 half-degrees), and the Rec.601 luma weights in the configured channel order |
| `_blend` | picks a blend mode per instance, computes per-pixel `alpha`, composites `alpha*patch + (1-alpha)*region` |
| `_protect_core` | forces `alpha=1.0` more than `core_margin_px` inside the mask, so a wide blur can't erode the object's own interior |
| `_blend_poisson` | `cv2.seamlessClone`-based blend; opt-in only (not in the default `blend_modes`) |

## Blend modes (`_blend`)

| mode | mechanism |
|---|---|
| `none` | hard 0/1 mask, no ramp |
| `gaussian_blur_edge` / `box_blur` | blur the hard mask itself with a random odd kernel from `blur_kernel_range`; naturally continuous since it's one linear filter pass |
| `poisson` | gradient-domain seamless cloning; falls back to `none` if it can't be solved or the patch would spill past the canvas edge |

## Placement (`_place`, `_cluster_sigma`, `_sample_cluster_center`)

Positions come from a **two-component spatial mixture**, not a single uniform draw:

> With probability `p` a frame is *concentrated*: a frame-specific centre **μ** is drawn uniformly
> (inset by σ so most of the mass lands on canvas), and each instrument's position is drawn from an
> isotropic Gaussian **N(μ, σ²I)** with probability `1 − s`, or uniformly with probability `s`.
> With probability `1 − p` every position in the frame is uniform. The scale is relative to image
> size: **σ = r · min(H, W) / 2**, so `r` is the 2σ (≈95%) footprint radius as a fraction of the
> shorter side.

`p` = `CLUSTER_PROB`, `r` = `CLUSTER_RADIUS_FRACTION`, `s` = `CLUSTER_STRAY_FRACTION`. The
concentrated mode exists because validation frames show instruments gathered in a small region
rather than scattered; the uniform component and the stray fraction keep that from becoming a
single rigid layout.

There is still **no retry-to-avoid-overlap** in either mode (see the module header): occlusion is
a real scenario to learn, not an artifact to suppress.

Two deviations from a textbook Gaussian, both deliberate:

- out-of-canvas draws are **clamped, not rejected**, so a little probability mass accumulates at
  the border (`test_place_clustered_clamps_instead_of_rejecting` pins this);
- the centre is drawn uniformly *inset by σ*, so the centre distribution itself is mildly
  non-uniform near edges.

## Appearance jitter (`_jitter_appearance`)

Randomizes one pasted crop's own appearance — brightness, contrast, saturation, hue, defocus.
Motivation: the cache cuts each source instance out **once**, so without this the same crop is
pixel-identical in every frame it appears in. Whole-frame photometric augmentation shifts the
entire image together and so leaves each instrument's appearance *relative to its neighbours*
untouched; this does not.

Appearance only: it must never touch `patch_mask` or `valid_footprint`, or the modal-mask
bookkeeping in `_update_modal_annotation` would disagree with the pixels actually painted.

**Contrast pivots on the instrument's own pixels**, not on the patch mean. The patch rect also
contains the crop's mat background, which sits well below the instrument — pivoting there puts
every instrument pixel above the pivot, so raising contrast *brightens* the instrument and fights
the brightness knob (measured on real crops: at brightness 0.70, contrast 1.0 → 2.6 moved the
tool-pixel mean 114.8 → 183.8). It also made one contrast value mean different things for a
scalpel and a clamp, since the pivot drifts with how much background the bounding box contains.
With the instrument mean as pivot the two knobs are independent: brightness sets the level,
contrast sets internal texture.

## Perspective (`_perspective_matrix`, `_tilt_homography`)

A per-instance **plane-induced homography**, parameterized as a real out-of-plane viewpoint:
tilt ~ `U(0, TILT_DEGREES)` with azimuth ~ `U(0, 2π)`. The crop is treated as the z=0 plane,
rotated by that angle about an in-plane axis, and projected through a pinhole at
`2.5 · max(w, h)`; focal length equals that distance, so tilt 0 is exactly the identity. The local
anisotropy this produces matches the classical **1/cos(slant)** to 3-4 decimals.

Why an angle rather than free corner offsets: four freely-moving corners span the homography's
entire 8 DoF — including translation, scale, aspect and shear, which in-plane rotation already
covers — and are uniform in *displacement* but bell-shaped in *foreshortening*, so strong
viewpoints stay rare however the knob is set. Tilt + azimuth is 2 DoF, all of it viewpoint, and
uniform in the quantity the knob names.

The warped quad is rescaled about its centroid to **preserve area** (measured mean area ratio
1.009), because apparent size is class-informative here. The warp is composed into the rotation
matrix so the crop is resampled once rather than twice.

> Naming note: in the classical slant-tilt convention (shape-from-texture) the out-of-plane angle
> is the *slant* and "tilt" is the azimuth; ASIFT uses "tilt" for `1/cos(slant)`, a ratio rather
> than an angle. `TILT_DEGREES` here is the out-of-plane angle.

## Random streams

Appearance jitter and the projective tilt draw from **private RNGs**, not the module-level
`random` every other decision uses. Both are seeded once at construction from the global stream —
whether or not the feature is enabled, so two configs consume identical global draws — and then
mixed with the DataLoader **worker index**.

Two bugs this avoids: drawing from the global stream inside `_transform_instance` (which runs
before `_place`) would shift every subsequent placement, so enabling jitter would change the pile
layout too and confound any A/B against it; and seeding without the worker index would make every
DataLoader worker replay one identical augmentation sequence. The worker index rather than the pid
because it is 0 in the main process, which keeps `tools/augmentation/dump_training_samples.py` reproducible
across invocations.

## Known limitations

- **Thin structures can wash out under `gaussian_blur_edge`/`box_blur`.** `_protect_core`'s
  erosion can fully erase a thin object's "core" (nothing survives to force back to
  `alpha=1`), and a wide kernel then blurs across the *entire* object from both edges at once.
  The reference implementation (`debidatta/syndata-generation`, cited above) uses fixed 5×5
  (σ=2) / 3×3 kernels, well below this codebase's configurable range — keep
  `blur_kernel_range`'s upper bound modest for domains with thin instrument features (finger
  loops, blades).
- **Occlusion bookkeeping uses the hard silhouette, not the soft alpha actually painted.**
  `layer["visible"] &= ~full_mask` is a binary cut; at a blurred boundary pixel the
  underlying layer may still be mostly visible pixel-wise, but the GT mask treats it as fully
  occluded. Believed intentional (blending is photometric camouflage only; GT should stay
  geometrically exact) rather than a bug.

## Config (`cfg.INPUT.COPY_PASTE`)

| key | default | meaning |
|---|---|---|
| `ENABLED` | `False` | attach the compositor (force-disabled at eval time regardless) |
| `MIN_INSTANCES` / `MAX_INSTANCES` | `0` / `10` | range for `k`, the number of paste attempts per sample |
| `ROTATION_DEGREES` | `30.0` | `θ ~ U(-this, +this)`; MBOI itself uses `U(-90, 90)` |
| `MASK_INTERP` | `["nearest"]` | list of eligible modes, randomized per instance like `BLEND_MODES`: `"nearest"` (blocky, bit-exact mask), `"linear"`/`"bicubic"`/`"lanczos"` (anti-aliased, mask thresholded at 0.5). `"bicubic"`/`"lanczos"` genuinely ring (~15-20% brightness overshoot at a high-contrast edge in testing) unlike `"linear"`'s pure convex combination — saturates safely into `uint8`, doesn't jaggy up the mask boundary, but is a real visible artifact tradeoff |
| `BLEND_MODES` | `["none", "gaussian_blur_edge", "box_blur"]` | randomized per pasted instance; `"poisson"` is implemented but deliberately excluded by default (hurt in Cut-Paste-and-Learn's own ablation). `"alpha_feather"` was removed in #64 and now raises `ValueError` |
| `BLUR_KERNEL_RANGE` | `[3, 7]` | odd kernel size range for `gaussian_blur_edge`/`box_blur`; also sets the cache's crop margin (`upper // 2 + 1`) |
| `FEATHER_WIDTH_RANGE` | removed (#64) | registered as a deprecated key: a saved config that sets it still loads. A strict merge drops the value with a warning; a loader that calls `set_new_allowed(True)` first keeps it as an unused stray key, without a warning |
| `CORE_MARGIN_PX` | `2` | how far inward from any boundary `_protect_core` forces `alpha=1`; `0` disables it (restores the old fully-symmetric ramp) |
| `CLUSTER_PROB` | `0.0` | `p` — probability a frame uses the Gaussian (concentrated) placement component instead of uniform; `0` restores pure uniform scatter |
| `CLUSTER_RADIUS_FRACTION` | `0.25` | `r` — the Gaussian's 2σ (≈95%) footprint radius as a fraction of the shorter side; `σ = r · min(H,W) / 2` |
| `CLUSTER_STRAY_FRACTION` | `0.2` | `s` — on a concentrated frame, the fraction of instances still placed uniformly, so a pile is never a rigid layout |
| `INSTANCE_JITTER.BRIGHTNESS` / `.CONTRAST` / `.SATURATION` | `[1.0, 1.0]` | per-pasted-instance multiplicative `[lo, hi]`. Ranges are **not** required to be symmetric about 1.0 — centring them elsewhere is how a systematic render-vs-real offset gets corrected, which a symmetric range (mean 1.0) cannot do |
| `INSTANCE_JITTER.HUE_DEGREES` | `0.0` | per-instance hue shift `~ U(-this, +this)`, for white-balance drift |
| `INSTANCE_JITTER.BLUR_SIGMA` | `[0.0, 0.0]` | per-instance defocus `σ ~ U(lo, hi)`, applied to the patch before blending |
| `PERSPECTIVE.PROB` | `0.0` | probability a pasted instance gets a projective tilt |
| `PERSPECTIVE.TILT_DEGREES` | `0.0` | max out-of-plane viewing angle; the tilt is drawn `~ U(0, this)` with a uniformly random azimuth. `0` disables regardless of `PROB` |

Not a config knob: `image_format` is always `cfg.INPUT.FORMAT` (kept in sync with the rest of
the mapper, not copy-paste-specific). Every kwarg above is passed explicitly from this config
block, so this table is the single source of truth; don't expect (or add) a second default
elsewhere. The two exceptions are `instance_jitter` and `instance_perspective`, which default to
`None` in the constructor so the compositor stays constructible outside the mapper (tests, tools)
— both are normalized to `None` when every knob is a no-op, which is what keeps the hot path free
for configs that do not opt in.
