# ------------------------------------------------------------------------
# Copy-paste compositing augmentation for the reclassify phase.
#
# Combines two papers' ideas:
#   - MBOI (Badilla-Solorzano et al., IJCARS 2022, "Deep-learning-based instrument
#     detection for intra-operative robotic assistance"): cut an instrument out of
#     one image via its GT mask, apply a random rotation/translation, and paste it
#     onto a different image - combining several sparse single-instrument renders
#     into one busier multi-instrument composite.
#   - Cut, Paste and Learn (Dwibedi et al., arXiv:1708.01642; reference impl at
#     github.com/debidatta/syndata-generation/blob/master/dataset_generator.py):
#     randomizing the paste-boundary blend mode per instance ("All Blend") beat
#     every single fixed mode in their ablation. Poisson blending alone hurt
#     (58.4 vs 65.9 mAP for no-blend) - it's implemented here (see
#     CopyPasteCompositor._blend_poisson) since a small forced color-shift might
#     still help some training runs, but it is NOT in the default blend_modes;
#     it's opt-in for the user's own A/B experiments.
#
# Unlike MBOI, pasted instances are never flipped or rescaled: this codebase treats
# surgical instruments as having fixed handedness (see the RandomFlip comment in
# coco_instance_new_baseline_dataset_mapper.build_transform_gen), and apparent size
# is a real, class-informative cue under a fixed camera setup, so only rotation +
# translation are used.
#
# Also unlike MBOI (which resamples the placement up to 20x to dodge overlap with
# existing content), placement here is a single uniform-random draw with no
# retry-to-avoid-overlap: occluded instruments are a real scenario the network
# needs to learn, not an artifact to suppress, so overlap is left to whatever
# chance produces rather than being systematically minimized.
# ------------------------------------------------------------------------
import logging
import random

import cv2
import h5py
import numpy as np
from pycocotools import mask as mask_util
from sgdata import schema

__all__ = ["CopyPasteCompositor"]

logger = logging.getLogger(__name__)


def _read_colors(file_name, image_format):
    try:
        with h5py.File(file_name, "r") as f:
            image = f[schema.COLORS][()]
    except (OSError, KeyError) as exc:
        logger.warning(
            "[CopyPasteCompositor] dropping unreadable source frame %s: %s",
            file_name,
            exc,
        )
        return None

    assert image.ndim == 3 and image.shape[2] == 3, (
        "expected an (H, W, 3) RGB 'colors' array"
    )
    if image_format == "BGR":
        image = image[:, :, ::-1]
    return np.ascontiguousarray(image)


# "nearest" is handled separately below (already discrete, no threshold needed).
# The rest all follow the same recipe - warp both image and mask with the same
# flag, then threshold the (float, 0-1) mask at 0.5 - only the flag differs.
_SUBPIXEL_INTERP_FLAGS = {
    "linear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC,
    "lanczos": cv2.INTER_LANCZOS4,
}


# Rec.601 luma weights in RGB order, reversed for BGR input - same convention as
# maskdino.data.augmentations.photometric (duplicated rather than imported to keep
# this module's import surface to cv2/numpy/pycocotools).
_LUMA_RGB = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def _luma_weights(image_format):
    return _LUMA_RGB if image_format != "BGR" else _LUMA_RGB[::-1]


def _shift_hue(img, degrees, image_format):
    """Rotate hue by `degrees`. OpenCV's 8-bit HSV packs hue into 0-179 (half-degrees),
    so the shift is wrapped there, and uint8 wraparound is exactly the cyclic behavior
    hue needs."""
    code_to, code_from = (
        (cv2.COLOR_BGR2HSV, cv2.COLOR_HSV2BGR)
        if image_format == "BGR"
        else (cv2.COLOR_RGB2HSV, cv2.COLOR_HSV2RGB)
    )
    hsv = cv2.cvtColor(img, code_to)
    hsv[..., 0] = (hsv[..., 0].astype(np.int16) + int(round(degrees / 2.0))) % 180
    return cv2.cvtColor(hsv, code_from)


def _normalize_jitter(cfg_node):
    """Config node (or dict) -> plain dict, or None when every knob is a no-op, so
    the hot path can skip the whole thing with one identity check."""
    if cfg_node is None:
        return None

    def get(key, default):
        # yacs CfgNode is a dict subclass whose keys are UPPERCASE, while a plain
        # dict built in a test or by hand is likely lowercase - accept both, and
        # never silently fall through to the default for a key that IS set, or an
        # arm config would quietly train as if jitter were off.
        for k in (key.upper(), key):
            if isinstance(cfg_node, dict):
                if k in cfg_node:
                    return cfg_node[k]
            elif hasattr(cfg_node, k):
                return getattr(cfg_node, k)
        return default

    j = {
        "brightness": tuple(float(v) for v in get("brightness", (1.0, 1.0))),
        "contrast": tuple(float(v) for v in get("contrast", (1.0, 1.0))),
        "saturation": tuple(float(v) for v in get("saturation", (1.0, 1.0))),
        "hue_degrees": float(get("hue_degrees", 0.0)),
        "blur_sigma": tuple(float(v) for v in get("blur_sigma", (0.0, 0.0))),
    }
    is_noop = (
        j["brightness"] == (1.0, 1.0)
        and j["contrast"] == (1.0, 1.0)
        and j["saturation"] == (1.0, 1.0)
        and j["hue_degrees"] == 0.0
        and j["blur_sigma"][1] <= 0.0
    )
    return None if is_noop else j


def _tilt_homography(cw, ch, tilt_deg, azimuth_rad, cam_distance_factor=2.5):
    """Homography for viewing a flat crop from `tilt_deg` off its normal.

    Parameterized by a real out-of-plane TILT rather than by free corner offsets,
    because tilt is the quantity worth controlling and the quantity worth sampling
    uniformly. Independent per-corner jitter is uniform in corner displacement but
    NOT in foreshortening: eight independent uniforms concentrate around the middle
    (central limit), so extreme viewpoints stay rare no matter how the knob is set.
    Sampling the angle directly gives rotation-like uniform coverage.

    The construction is the literal physical one: treat the crop as the z=0 plane
    centered on the origin, rotate it by `tilt_deg` about the in-plane axis pointing
    at `azimuth_rad`, and project through a pinhole at distance
    cam_distance_factor * max(cw, ch). Focal length is set equal to that distance so
    tilt 0 is exactly the identity.

    The result is then rescaled about its centroid to preserve area: a tilt
    foreshortens, and apparent size is a class cue in this dataset, so the warp must
    change shape without quietly shrinking the instrument.
    """
    w, h = cw - 1.0, ch - 1.0
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    centered = np.column_stack(
        [corners[:, 0] - w / 2.0, corners[:, 1] - h / 2.0, np.zeros(4)]
    )

    phi = np.deg2rad(tilt_deg)
    axis = np.array([np.cos(azimuth_rad), np.sin(azimuth_rad), 0.0])
    # Rodrigues' rotation of the plane about an in-plane axis.
    K = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )
    R = np.eye(3) + np.sin(phi) * K + (1.0 - np.cos(phi)) * (K @ K)
    rotated = centered @ R.T

    distance = cam_distance_factor * max(w, h)
    denom = distance + rotated[:, 2]
    denom = np.where(np.abs(denom) < 1e-6, 1e-6, denom)
    projected = np.column_stack(
        [
            distance * rotated[:, 0] / denom + w / 2.0,
            distance * rotated[:, 1] / denom + h / 2.0,
        ]
    )

    def _area(q):
        x, y = q[:, 0], q[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    src_area, dst_area = _area(corners), _area(projected)
    if dst_area > 1e-9:
        centroid = projected.mean(axis=0)
        projected = centroid + (projected - centroid) * np.sqrt(src_area / dst_area)

    return cv2.getPerspectiveTransform(corners, projected.astype(np.float32))


def _worker_index():
    """This DataLoader worker's index, or 0 in the main process.

    torch is imported lazily so this module stays importable (and unit-testable)
    without it, and so the import cost is not paid on every call site that never
    touches a DataLoader.
    """
    try:
        import torch.utils.data

        info = torch.utils.data.get_worker_info()
        return 0 if info is None else int(info.id) + 1
    except Exception:
        return 0


def _normalize_perspective(cfg_node):
    """Config node (or dict) -> (prob, distortion_scale), or None when disabled."""
    if cfg_node is None:
        return None

    def get(key, default):
        for k in (key.upper(), key):
            if isinstance(cfg_node, dict):
                if k in cfg_node:
                    return cfg_node[k]
            elif hasattr(cfg_node, k):
                return getattr(cfg_node, k)
        return default

    prob = float(get("prob", 0.0))
    tilt = float(get("tilt_degrees", 0.0))
    return None if prob <= 0.0 or tilt <= 0.0 else (prob, tilt)


def _random_odd(low, high):
    lo, hi = int(low) | 1, int(high) | 1
    if lo > hi:
        lo, hi = hi, lo
    n = (hi - lo) // 2 + 1
    return lo + 2 * random.randint(0, n - 1)


class CopyPasteCompositor:
    """Pastes 1-k instrument crops onto a destination image. Call before any
    geometric/photometric augmentation runs, on the raw 'colors' array and the
    raw (untransformed) annotation dict list.

    Every eligible instance's raw (un-rotated) crop + mask is cut out once, up
    front in __init__, from the .hdf5 frames named by `source_dicts` - not
    re-read from disk on every paste attempt. This trades a one-time startup
    cost (proportional to the number of distinct source frames) for a per-call
    cost that no longer touches the filesystem at all: __call__ just samples
    from the in-memory instance cache and applies a fresh random
    rotation/placement/blend each time, same as before."""

    def __init__(
        self,
        source_dicts,
        *,
        min_instances,
        max_instances,
        rotation_degrees,
        mask_interp,
        blend_modes,
        blur_kernel_range,
        core_margin_px,
        image_format,
        cluster_prob,
        cluster_radius_fraction,
        cluster_stray_fraction,
        instance_jitter=None,
        instance_perspective=None,
    ):
        self.min_instances = min_instances
        self.max_instances = max_instances
        self.rotation_degrees = rotation_degrees
        self.mask_interp = tuple(mask_interp)
        self.blend_modes = tuple(blend_modes)
        if "alpha_feather" in self.blend_modes:
            raise ValueError(
                "copy-paste blend mode 'alpha_feather' was removed in #64; "
                "drop it from INPUT.COPY_PASTE.BLEND_MODES"
            )
        self.blur_kernel_range = tuple(blur_kernel_range)
        self.core_margin_px = core_margin_px
        self.image_format = image_format
        self.cluster_prob = cluster_prob
        self.cluster_radius_fraction = cluster_radius_fraction
        self.cluster_stray_fraction = cluster_stray_fraction
        self.instance_jitter = _normalize_jitter(instance_jitter)
        # Appearance jitter draws from its OWN stream, not the module-level `random`
        # every other decision here uses. Otherwise enabling jitter would consume
        # draws inside _transform_instance - which runs before _place() - and shift
        # every subsequent placement, so a jittered arm would differ from its control
        # in pile layout as well as appearance, confounding the very comparison these
        # arms exist to make. Seeded from the global stream exactly once, at
        # construction (whether or not jitter is on), so runs stay reproducible under
        # random.seed(...) and dataloader workers still diverge from each other.
        self._rng_base = random.randrange(2**32)
        self._rng_worker = None
        self.instance_perspective = _normalize_perspective(instance_perspective)
        self._crop_margin_px = max(self.blur_kernel_range[1] // 2 + 1, 0)
        self._instance_cache = self._build_instance_cache(source_dicts)
        self._cache_by_class = self._index_by_class(self._instance_cache)

    def _build_instance_cache(self, source_dicts):
        """Extract every eligible instance's raw crop + mask once, reading each
        distinct source frame from disk exactly one time (rather than once per
        paste attempt per instance, as a naive on-the-fly version would).

        Named "cache", not "pool", to avoid colliding with the unrelated
        LivePoolDataset/pool_dir sense of "pool" elsewhere in maskdino.data
        (sgdata's continuously-replenished directory of rendered frames) -
        this is a fixed, in-memory set of already-cropped instances, built
        once from the (possibly pool-backed) `source_dicts` snapshot handed
        to __init__."""
        anns_by_file = {}
        for d in source_dicts:
            file_name = d.get("file_name")
            anns = d.get("annotations")
            if not file_name or not anns:
                continue
            anns_by_file.setdefault(file_name, []).extend(
                a for a in anns if a.get("iscrowd", 0) == 0
            )

        cache = []
        for file_name, anns in anns_by_file.items():
            image = _read_colors(file_name, self.image_format)
            if image is None:
                continue
            img_h, img_w = image.shape[:2]
            for ann in anns:
                mask = mask_util.decode(ann["segmentation"]).astype(bool)
                ys, xs = np.where(mask)
                m = self._crop_margin_px
                y0, y1 = max(0, ys.min() - m), min(img_h, ys.max() + 1 + m)
                x0, x1 = max(0, xs.min() - m), min(img_w, xs.max() + 1 + m)
                crop = np.ascontiguousarray(image[y0:y1, x0:x1])
                crop_mask = np.ascontiguousarray(mask[y0:y1, x0:x1])
                cache.append((crop, crop_mask, ann["category_id"]))

        logger.info(
            "[CopyPasteCompositor] instance cache built: %d instance(s) from %d frame(s)",
            len(cache),
            len(anns_by_file),
        )
        return cache

    @staticmethod
    def _index_by_class(cache):
        """category_id -> list of its positions in `cache`, for class-balanced
        paste sampling (see _sample_instance_indices)."""
        by_class = {}
        for i, (_, _, category_id) in enumerate(cache):
            by_class.setdefault(category_id, []).append(i)
        if by_class:
            counts = sorted(len(v) for v in by_class.values())
            logger.info(
                "[CopyPasteCompositor] %d class(es) in the paste pool, "
                "%d-%d render(s) each - sampling is class-balanced, so the "
                "%dx spread in render count no longer biases paste frequency",
                len(by_class),
                counts[0],
                counts[-1],
                max(1, round(counts[-1] / max(counts[0], 1))),
            )
        return by_class

    def _sample_instance_indices(self, k):
        """Pick k cache positions with every class equally likely, instead of
        uniformly over the flat pool.

        Uniform-over-pool sampling makes a class's paste frequency proportional
        to how many renders it happens to have - with 40 renders for some
        classes and 20 for the rest that is a 2x exposure difference that has
        nothing to do with the experiment. Here a shuffled class list is walked
        instead, so for k <= num_classes every pasted instrument is a *different*
        class (maximal in-frame class diversity, which is also the contrastive
        pressure that fine-grained pairs like forcep03/forcep04 need), and for
        larger k the list wraps and a class may recur.

        Within a chosen class the render is drawn uniformly, without replacement
        across repeats of that class, so a frame never shows the same crop twice
        (unreachable in practice: it needs k > num_classes * renders_per_class).
        The result is shuffled because paste order decides who occludes whom -
        returning it grouped by class would make later-drawn classes
        systematically the occluders.
        """
        classes = list(self._cache_by_class)
        if not classes:
            return []
        random.shuffle(classes)

        wanted = {}
        for i in range(k):
            cid = classes[i % len(classes)]
            wanted[cid] = wanted.get(cid, 0) + 1

        picked = []
        for cid, count in wanted.items():
            idxs = self._cache_by_class[cid]
            if count <= len(idxs):
                picked.extend(random.sample(idxs, count))
            else:  # more repeats than renders - fall back to with-replacement
                picked.extend(random.choices(idxs, k=count))
        random.shuffle(picked)
        return picked

    def __call__(self, image, annotations):
        """Returns (composite_image, updated_annotations). `image` is never
        mutated in place (a copy is composited and returned); `annotations` dicts
        may have "segmentation"/"bbox"/"area"/"visibility_fraction" updated in
        place if a later paste occludes them - masks stay modal (visible-region
        only), never amodal, throughout - and the returned list may contain new
        dicts appended for newly pasted instances."""
        h, w = image.shape[:2]
        composite = image.copy()

        layers = []  # each: {"ann": dict, "visible": (H,W) bool, "nominal_area": float}
        for ann in annotations:
            mask = mask_util.decode(ann["segmentation"]).astype(bool)
            nominal_area = float(mask.sum())
            if nominal_area <= 0:
                continue
            layers.append({"ann": ann, "visible": mask, "nominal_area": nominal_area})

        k = random.randint(self.min_instances, self.max_instances)

        # Per-frame mode pick: with probability cluster_prob this whole call
        # uses clustered "pile" placement (a fresh center + sigma, computed
        # once here)
        use_cluster = random.random() < self.cluster_prob
        cluster_center = None
        cluster_std = None
        if use_cluster:
            sigma = self._cluster_sigma(h, w)
            cluster_center = self._sample_cluster_center(h, w, sigma)
            cluster_std = (sigma, sigma)

        pasted = 0
        for idx in self._sample_instance_indices(k):
            crop, crop_mask, category_id = self._instance_cache[idx]

            patch, patch_mask, valid_footprint = self._transform_instance(
                crop, crop_mask
            )

            if use_cluster and random.random() >= self.cluster_stray_fraction:
                # Piled: Gaussian-around-center draw.
                placement = self._place(
                    patch_mask, h, w, center=cluster_center, std=cluster_std
                )
            else:
                placement = self._place(patch_mask, h, w)
            if placement is None:
                continue
            py, px, full_mask = placement

            self._blend(composite, patch, patch_mask, valid_footprint, py, px)

            rle = mask_util.encode(np.asfortranarray(full_mask.astype(np.uint8)))
            rle["counts"] = rle["counts"].decode("ascii")
            new_ann = {
                "bbox": mask_util.toBbox(rle).tolist(),
                "bbox_mode": 1,  # BoxMode.XYWH_ABS
                "category_id": category_id,
                "segmentation": rle,
                "area": float(mask_util.area(rle)),
                "iscrowd": 0,
                "visibility_fraction": 1.0,
            }
            new_layer = {
                "ann": new_ann,
                "visible": full_mask,
                "nominal_area": float(full_mask.sum()),
            }

            for layer in layers:
                layer["visible"] &= ~full_mask
                self._update_modal_annotation(layer)

            layers.append(new_layer)
            pasted += 1

        return composite, [layer["ann"] for layer in layers]

    @staticmethod
    def _update_modal_annotation(layer):
        """Re-rasterize `layer["ann"]`'s segmentation/bbox/area from its current
        `layer["visible"]` mask, and refresh `visibility_fraction`. Masks here must
        stay modal (visible-region-only, not amodal/pre-occlusion): whenever a
        later paste in this call covers part of an earlier instance (destination-
        original or already-pasted), that instance's stored mask must shrink to
        match what's actually still visible - otherwise the GT would claim area
        that's now a different, painted-on-top instrument."""
        visible = layer["visible"]
        ann = layer["ann"]
        ann["visibility_fraction"] = float(visible.sum()) / layer["nominal_area"]
        rle = mask_util.encode(np.asfortranarray(visible.astype(np.uint8)))
        rle["counts"] = rle["counts"].decode("ascii")
        ann["segmentation"] = rle
        ann["area"] = float(mask_util.area(rle))
        ann["bbox"] = mask_util.toBbox(rle).tolist()

    def _transform_instance(self, crop, crop_mask):
        ch, cw = crop.shape[:2]
        theta = random.uniform(-self.rotation_degrees, self.rotation_degrees)

        abs_cos = abs(np.cos(np.deg2rad(theta)))
        abs_sin = abs(np.sin(np.deg2rad(theta)))
        bound_w = int(np.ceil(ch * abs_sin + cw * abs_cos)) + 2
        bound_h = int(np.ceil(ch * abs_cos + cw * abs_sin)) + 2

        rot_mat = cv2.getRotationMatrix2D((cw / 2.0, ch / 2.0), theta, 1.0)
        rot_mat[0, 2] += bound_w / 2.0 - cw / 2.0
        rot_mat[1, 2] += bound_h / 2.0 - ch / 2.0

        # Optional per-instance projective warp, COMPOSED into the rotation rather
        # than applied as a second pass: one resampling step instead of two, so the
        # crop is not softened twice, and the image/mask/footprint still go through
        # exactly one shared matrix (the invariant the big comment below is about).
        # When disabled, the original warpAffine path runs untouched.
        matrix = (
            self._perspective_matrix(cw, ch, rot_mat)
            if self.instance_perspective
            else None
        )
        if matrix is not None:
            matrix, bound_w, bound_h = matrix

        # Same interpolation for the image and its mask - see the module header
        # comment: mixing a smooth flag (image) with NEAREST (mask) makes their
        # boundaries disagree at non-90 deg angles, producing spurious notches.
        # "nearest": blocky/staircased edges, but the mask is always a bit-exact
        # copy of a source pixel - simplest, matches the reference implementation.
        # "linear"/"bicubic"/"lanczos": smoother, anti-aliased edges - both are
        # warped with the same flag, then the (float, 0-1) mask is thresholded at
        # 0.5 for a sub-pixel-accurate (not blocky) boundary. Unlike "linear"
        # (a pure convex combination, never overshoots [0,1]), "bicubic"/"lanczos"
        # have negative side-lobes and genuinely ring - up to ~15-20% brightness
        # overshoot right at a high-contrast edge - but that's saturated back into
        # uint8 range by cv2 (no wraparound), and stays well clear of the 0.5
        # threshold in practice, so it doesn't jaggy up the mask boundary; it's a
        # tradeoff picked per instance (like blend_modes) via self.mask_interp,
        # not a fixed choice for the whole run.
        interp = random.choice(self.mask_interp)

        def warp(src, flags):
            if matrix is None:
                return cv2.warpAffine(
                    src, rot_mat, (bound_w, bound_h), flags=flags, borderValue=0
                )
            return cv2.warpPerspective(
                src, matrix, (bound_w, bound_h), flags=flags, borderValue=0
            )

        if interp == "nearest":
            patch = warp(crop, cv2.INTER_NEAREST)
            patch_mask = warp(crop_mask.astype(np.uint8), cv2.INTER_NEAREST).astype(
                bool
            )
        elif interp in _SUBPIXEL_INTERP_FLAGS:
            flag = _SUBPIXEL_INTERP_FLAGS[interp]
            patch = warp(crop, flag)
            patch_mask = warp(crop_mask.astype(np.float32), flag) > 0.5
        else:
            raise ValueError(f"unknown copy-paste mask_interp: {interp!r}")
        # The output canvas (bound_w x bound_h) is larger than the rotated crop
        # itself, so the corners outside the rotated rectangle are synthetic
        # borderValue=0 (black) fill, not real image content. patch_mask already
        # excludes that zone (it's warped with the same borderValue=0), but a
        # blur alpha ramp doesn't know "real background fading out" from
        # "synthetic black padding" - without this, it can pull genuine black into
        # the blend near a thin object's tip, where the padding sits close to the
        # boundary. valid_footprint marks exactly which pixels are real (rotated
        # crop) content, so _blend can clip alpha to 0 everywhere else.
        valid_footprint = warp(
            np.full(crop.shape[:2], 255, np.uint8), cv2.INTER_NEAREST
        ).astype(bool)

        # Appearance-only, and deliberately last: patch_mask and valid_footprint are
        # already final, and _jitter_appearance must not touch either of them (see
        # the module header and INPUT.COPY_PASTE.INSTANCE_JITTER in config.py).
        patch = self._jitter_appearance(patch, patch_mask)
        return patch, patch_mask, valid_footprint

    def _ensure_rngs(self):
        """Appearance jitter and the projective warp draw from their OWN streams, not
        the module-level `random` every other decision uses. Two reasons:

        - enabling either would otherwise consume draws inside _transform_instance,
          which runs before _place(), shifting every subsequent placement - so an arm
          testing jitter would also differ from its control in pile layout, confounding
          exactly the comparison the arms exist to make;
        - the seed is derived once from the global stream (at construction, whether or
          not the feature is on, so arms consume identical global draws) and then mixed
          with the DataLoader WORKER INDEX, because a worker inherits this object by
          fork: without that, every worker would replay one identical augmentation
          sequence. The worker index - not the pid - because it is 0 in the main
          process, which keeps tools/augmentation/dump_training_samples.py reproducible across
          invocations while still giving each worker its own stream.
        """
        worker = _worker_index()
        if self._rng_worker != worker:
            self._rng_worker = worker
            mix = (worker + 1) * 0x9E3779B1
            self._jitter_rng = random.Random(self._rng_base ^ mix)
            self._perspective_rng = random.Random((self._rng_base + 1) ^ mix)

    def _perspective_matrix(self, cw, ch, rot_mat):
        """(3x3 matrix, bound_w, bound_h) for a rotation composed with a centered,
        area-neutral projective warp - or None when this instance does not get one.

        the
        frame-level version has to sample its quad inward to avoid empty corners and
        therefore always zooms in, but a crop can simply grow its canvas, and whatever
        lands outside the instrument's mask is never pasted anyway. Apparent size
        therefore stays class-informative.
        """
        self._ensure_rngs()
        prob, tilt_degrees = self.instance_perspective
        rng = self._perspective_rng
        if rng.random() >= prob:
            return None

        # Tilt uniform over its range, azimuth uniform over the full circle: the
        # instrument is equally likely to be viewed from any direction, at any
        # off-axis angle up to the cap. A signed tilt would be redundant - azimuth
        # already covers the opposite direction.
        tilt = rng.uniform(0.0, tilt_degrees)
        azimuth = rng.uniform(0.0, 2.0 * np.pi)
        homography = _tilt_homography(cw, ch, tilt, azimuth)

        corners = np.float32([[0, 0], [cw - 1, 0], [cw - 1, ch - 1], [0, ch - 1]])

        rot3 = np.vstack([rot_mat, [0.0, 0.0, 1.0]])
        matrix = homography @ rot3

        # The composed warp can push content outside the rotation-derived bounds, so
        # re-derive them from where the crop's own corners actually land, and shift
        # the result back to the origin. Straight lines stay straight under a
        # homography, so the four corners bound the whole warped crop.
        src = np.concatenate([corners, np.ones((4, 1), np.float32)], axis=1)
        dst = src @ matrix.T
        dst = dst[:, :2] / np.where(np.abs(dst[:, 2:3]) < 1e-12, 1e-12, dst[:, 2:3])
        min_xy = dst.min(axis=0)
        max_xy = dst.max(axis=0)
        shift = np.array(
            [
                [1.0, 0.0, -min_xy[0] + 1.0],
                [0.0, 1.0, -min_xy[1] + 1.0],
                [0.0, 0.0, 1.0],
            ]
        )
        bound_w = int(np.ceil(max_xy[0] - min_xy[0])) + 2
        bound_h = int(np.ceil(max_xy[1] - min_xy[1])) + 2
        return shift @ matrix, bound_w, bound_h

    def _jitter_appearance(self, patch, patch_mask=None):
        """Randomize ONE pasted crop's own appearance - brightness, contrast,
        saturation, hue, focus - so the same cached source crop does not look
        pixel-identical in every frame it is pasted into.

        Phase-2's training set has ~2 distinct appearances per class, which a
        fine-tuned model simply memorizes (0.90 on composited train vs 0.26 on real
        val, see trainset_confusion_check_forcep03_results.md). Global photometric
        augmentation moves the whole frame and so leaves each crop's appearance
        relative to its neighbours untouched; this does not.

        Operates on the whole patch rect, not just the masked pixels: the unmasked
        remainder is the crop's own mat background, which _blend only pulls in at
        the alpha ramp, and keeping it consistent with the instrument avoids a
        visible seam of un-jittered background right at the boundary.
        """
        j = self.instance_jitter
        if j is None:
            return patch

        self._ensure_rngs()
        rng = self._jitter_rng
        out = patch.astype(np.float32)
        lo, hi = j["brightness"]
        if lo != 1.0 or hi != 1.0:
            out *= rng.uniform(lo, hi)
        lo, hi = j["contrast"]
        if lo != 1.0 or hi != 1.0:
            # Pivot on the INSTRUMENT's own mean, not the whole patch's. The patch
            # rect also contains the crop's mat background, which sits well below the
            # instrument: pivoting there puts every instrument pixel above the pivot,
            # so raising contrast *brightens* the instrument and fights the brightness
            # knob (measured: at brightness 0.70, contrast 1.0 -> 2.6 moved the tool
            # mean 114.8 -> 183.8). It also made the same contrast value mean different
            # things for a scalpel and a clamp, since the pivot moves with however much
            # background a given instrument's bounding box happens to contain.
            # With the instrument mean as pivot the two knobs are independent:
            # brightness sets the level, contrast sets internal texture.
            pivot = (
                out[patch_mask].mean()
                if patch_mask is not None and patch_mask.any()
                else out.mean()
            )
            out = pivot + rng.uniform(lo, hi) * (out - pivot)
        lo, hi = j["saturation"]
        if lo != 1.0 or hi != 1.0:
            gray = (out * _luma_weights(self.image_format)).sum(axis=2, keepdims=True)
            out = gray + rng.uniform(lo, hi) * (out - gray)
        out = np.clip(out, 0, 255).astype(np.uint8)

        if j["hue_degrees"] > 0:
            out = _shift_hue(
                out, rng.uniform(-j["hue_degrees"], j["hue_degrees"]), self.image_format
            )
        lo, hi = j["blur_sigma"]
        if hi > 0:
            sigma = rng.uniform(lo, hi)
            if sigma > 0:
                out = cv2.GaussianBlur(out, (0, 0), sigmaX=sigma, sigmaY=sigma)
        return out

    def _cluster_sigma(self, h, w):
        """2D Gaussian std for clustered placement, derived from
        cluster_radius_fraction: the pile's ~2-sigma (95%) footprint radius is
        that fraction of the image's shorter side. E.g. 0.25 -> a circular
        2-sigma footprint covering ~1/5 of the image area."""
        return self.cluster_radius_fraction * min(h, w) / 2.0

    def _sample_cluster_center(self, h, w, sigma):
        """Fresh, uniformly-random cluster center for ONE clustered __call__ -
        never a fixed location. Biased inward by `sigma` so the pile's mass
        mostly lands on canvas even when the center draw is near an edge;
        collapses toward the canvas midpoint on any axis where sigma already
        covers half the canvas."""
        margin_y = min(sigma, h / 2.0)
        margin_x = min(sigma, w / 2.0)
        cy = random.uniform(margin_y, h - margin_y)
        cx = random.uniform(margin_x, w - margin_x)
        return cy, cx

    def _place(self, patch_mask, h, w, center=None, std=None):
        """Single random placement - no retry to reduce overlap with
        already-occupied pixels. Occlusion between instruments is a real
        scenario the network needs to learn, not an artifact to suppress, so
        whatever the one random draw lands on (however much it overlaps
        existing content, or spills toward a canvas edge) is used as-is -
        clamped into the valid placement range, never re-drawn/rejected.

        center=None (default): uniform placement, exactly the original
        behavior - py, px ~ U(0, h-ph) x U(0, w-pw).

        center=(cy, cx), std=(sigma_y, sigma_x): clustered "pile" placement -
        the patch's CENTER (not its top-left corner) is drawn from a 2D
        Gaussian around (cy, cx), converted to a top-left corner, and clamped
        into [0, h-ph] x [0, w-pw] rather than discarded.
        """
        ph, pw = patch_mask.shape[:2]
        if ph > h or pw > w:
            return None

        if center is None:
            py = random.randint(0, h - ph)
            px = random.randint(0, w - pw)
        else:
            cy, cx = center
            sigma_y, sigma_x = std
            y = random.gauss(cy, sigma_y) - ph / 2.0
            x = random.gauss(cx, sigma_x) - pw / 2.0
            py = int(round(min(max(y, 0.0), h - ph)))
            px = int(round(min(max(x, 0.0), w - pw)))

        full_mask = np.zeros((h, w), dtype=bool)
        full_mask[py : py + ph, px : px + pw] = patch_mask
        return py, px, full_mask

    def _blend(self, composite, patch, patch_mask, valid_footprint, py, px):
        mode = random.choice(self.blend_modes)

        if mode == "poisson":
            if self._blend_poisson(composite, patch, patch_mask, py, px):
                return
            mode = "none"  # seamlessClone declined (e.g. too close to the edge)

        ph, pw = patch_mask.shape[:2]
        region = composite[py : py + ph, px : px + pw]

        if mode == "none":
            alpha = patch_mask.astype(np.float32)
        elif mode == "gaussian_blur_edge":
            k = _random_odd(*self.blur_kernel_range)
            alpha = cv2.GaussianBlur(patch_mask.astype(np.float32), (k, k), 0)
            alpha = np.clip(alpha, 0.0, 1.0)
            alpha = self._protect_core(alpha, patch_mask)
        elif mode == "box_blur":
            k = _random_odd(*self.blur_kernel_range)
            alpha = cv2.blur(patch_mask.astype(np.float32), (k, k))
            alpha = np.clip(alpha, 0.0, 1.0)
            alpha = self._protect_core(alpha, patch_mask)
        else:
            raise ValueError(f"unknown copy-paste blend mode: {mode}")

        # Never let a blur ramp pull in the synthetic black borderValue
        # fill from outside the rotated rectangle (see _transform_instance's
        # valid_footprint comment) - only real (rotated crop) pixels may blend.
        alpha = np.where(valid_footprint, alpha, 0.0).astype(np.float32)

        alpha = alpha[..., None]
        region[...] = (alpha * patch + (1.0 - alpha) * region).astype(region.dtype)

    def _protect_core(self, alpha, patch_mask):
        """Force alpha=1.0 more than core_margin_px inside the mask. gaussian_blur
        _edge/box_blur both ramp symmetrically around the true
        boundary, softening up to ~half the kernel width INTO the object's
        own interior - for a thin instrument that can meaningfully blur/erode its
        real appearance. Capping the inward reach at a small fixed margin (default
        2px) keeps the softening confined to the true boundary transition band,
        regardless of how large a kernel was sampled for the outward
        (into-background) side."""
        if self.core_margin_px <= 0:
            return alpha
        core = cv2.erode(
            patch_mask.astype(np.uint8),
            np.ones((3, 3), np.uint8),
            iterations=self.core_margin_px,
        ).astype(bool)
        return np.where(core, 1.0, alpha).astype(np.float32)

    def _blend_poisson(self, composite, patch, patch_mask, py, px):
        """Gradient-domain (seamless) cloning via cv2.seamlessClone - can shift the
        pasted instrument's colors toward the destination's palette. Cut, Paste and
        Learn's own ablation found this hurts on average (58.4 vs 65.9 mAP for
        no-blend), so it is NOT in the default blend_modes; it's opt-in, for the
        user's own A/B experiments on possible small color-shift robustness gains.
        Returns False (caller falls back to a hard paste) if the padded patch would
        spill past the canvas edge, or if cv2 can't solve the blend."""
        h, w = composite.shape[:2]
        pad = 2  # cv2.seamlessClone requires the mask to not touch the src's border
        padded_patch = cv2.copyMakeBorder(
            patch, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0
        )
        padded_mask = cv2.copyMakeBorder(
            patch_mask.astype(np.uint8) * 255,
            pad,
            pad,
            pad,
            pad,
            cv2.BORDER_CONSTANT,
            value=0,
        )
        ph, pw = padded_mask.shape[:2]
        cy, cx = py + patch_mask.shape[0] // 2, px + patch_mask.shape[1] // 2
        if cx - pw // 2 < 0 or cy - ph // 2 < 0 or cx + pw // 2 > w or cy + ph // 2 > h:
            return False
        try:
            composite[...] = cv2.seamlessClone(
                padded_patch, composite, padded_mask, (cx, cy), cv2.NORMAL_CLONE
            )
        except cv2.error:
            return False
        return True
