"""Unit tests for maskdino/data/augmentations/copy_paste.py.

The module is loaded by file path (like tests/test_class_mapping.py) so the tests
don't trigger ``import maskdino`` (which registers datasets from absolute data
paths). It has no maskdino-internal imports of its own, so this is purely to stay
consistent with the rest of the test suite.
"""

import importlib.util
import os
import random
import sys

import cv2
import h5py
import numpy as np
import pytest
from pycocotools import mask as mask_util

_HERE = os.path.dirname(__file__)
_MOD_PATH = os.path.join(
    _HERE, "..", "maskdino", "data", "augmentations", "copy_paste.py"
)
_spec = importlib.util.spec_from_file_location("copy_paste", _MOD_PATH)
cp_mod = importlib.util.module_from_spec(_spec)
sys.modules["copy_paste"] = cp_mod
_spec.loader.exec_module(cp_mod)

CopyPasteCompositor = cp_mod.CopyPasteCompositor

# config.py has no maskdino-internal imports of its own (only detectron2), so like
# copy_paste.py above it can be loaded by file path without triggering `import
# maskdino` - needed below to check the actual default BLEND_MODES now that
# CopyPasteCompositor's constructor no longer has its own (see config.py's
# INPUT.COPY_PASTE, the single source of truth for these defaults).
_CONFIG_PATH = os.path.join(_HERE, "..", "maskdino", "config.py")
_config_spec = importlib.util.spec_from_file_location("maskdino_config", _CONFIG_PATH)
config_mod = importlib.util.module_from_spec(_config_spec)
sys.modules["maskdino_config"] = config_mod
_config_spec.loader.exec_module(config_mod)

add_maskdino_config = config_mod.add_maskdino_config


def _encode(mask):
    rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def _make_ann(mask, category_id):
    rle = _encode(mask)
    return {
        "bbox": mask_util.toBbox(rle).tolist(),
        "bbox_mode": 1,
        "category_id": category_id,
        "segmentation": rle,
        "area": float(mask_util.area(rle)),
        "iscrowd": 0,
    }


def _write_source(tmp_path, name, image, mask, category_id):
    path = str(tmp_path / f"{name}.hdf5")
    with h5py.File(path, "w") as f:
        f.create_dataset("colors", data=image)
    return {"file_name": path, "annotations": [_make_ann(mask, category_id)]}


def _l_shape(size=24, arm=8):
    """An asymmetric L-tromino-style corner: a vertical bar + a horizontal bar
    meeting at the bottom-left. Its mirror image ("J") is not reachable by any
    rotation, so it can detect an accidental flip."""
    mask = np.zeros((size, size), dtype=bool)
    mask[:, :arm] = True  # vertical bar (full height, left side)
    mask[-arm:, :] = True  # horizontal bar (bottom, full width)
    return mask


def _solid_image(size, value):
    return np.full((size, size, 3), value, dtype=np.uint8)


def _solid_rect_mask(size=40, margin=4):
    mask = np.zeros((size, size), dtype=bool)
    mask[margin:-margin, margin:-margin] = True
    return mask


def _signed_orientation(mask):
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    largest = max(contours, key=cv2.contourArea)
    return cv2.contourArea(largest, oriented=True)


@pytest.fixture
def compositor_factory(tmp_path):
    def make(n_sources=3, **kwargs):
        sources = [
            _write_source(tmp_path, f"src{i}", _solid_image(24, 200), _l_shape(), 5 + i)
            for i in range(n_sources)
        ]
        defaults = dict(
            min_instances=1,
            max_instances=1,
            rotation_degrees=90.0,
            mask_interp=("nearest",),
            blend_modes=("none",),
            blur_kernel_range=(3, 7),
            core_margin_px=2,
            image_format="BGR",
            cluster_prob=0.0,
            cluster_radius_fraction=0.25,
            cluster_stray_fraction=0.2,
        )
        defaults.update(kwargs)
        return CopyPasteCompositor(sources, **defaults), sources

    return make


def test_paste_appends_annotation_with_matching_category(compositor_factory):
    random.seed(0)
    compositor, _ = compositor_factory()
    dest = _solid_image(64, 0)
    new_image, new_anns = compositor(dest, [])

    assert new_image.shape == dest.shape
    assert new_image.dtype == dest.dtype
    assert len(new_anns) == 1
    assert new_anns[0]["category_id"] in {5, 6, 7}
    mask = mask_util.decode(new_anns[0]["segmentation"])
    assert mask.sum() > 0
    ys, xs = np.where(mask)
    assert ys.min() >= 0 and ys.max() < 64
    assert xs.min() >= 0 and xs.max() < 64


def test_no_paste_leaves_destination_unaffected(compositor_factory):
    random.seed(1)
    compositor, _ = compositor_factory(min_instances=0, max_instances=0)
    dest = _solid_image(64, 0)
    dest_ann = _make_ann(np.ones((64, 64), dtype=bool), category_id=1)

    new_image, new_anns = compositor(dest, [dest_ann])

    assert len(new_anns) == 1
    assert new_anns[0].get("visibility_fraction", 1.0) == 1.0
    assert np.array_equal(new_image, dest)


def test_paste_reduces_destination_visibility_on_overlap(compositor_factory):
    random.seed(2)
    # destination mask covers the whole canvas -> any placement overlaps it fully.
    compositor, _ = compositor_factory()
    dest = _solid_image(64, 0)
    dest_ann = _make_ann(np.ones((64, 64), dtype=bool), category_id=1)

    _, new_anns = compositor(dest, [dest_ann])

    original = next(a for a in new_anns if a["category_id"] == 1)
    pasted = next(a for a in new_anns if a["category_id"] != 1)
    assert original["visibility_fraction"] < 1.0
    pasted_mask = mask_util.decode(pasted["segmentation"]).astype(bool)
    assert pasted_mask.sum() > 0

    # Modal, not amodal: the original instance's stored mask/bbox/area must shrink
    # to match what's actually still visible, not stay at its pre-occlusion extent.
    original_mask = mask_util.decode(original["segmentation"]).astype(bool)
    assert original_mask.sum() < 64 * 64
    assert original["area"] == float(original_mask.sum())
    assert not np.any(original_mask & pasted_mask)  # no double-claimed pixels


def test_modal_masks_never_overlap_across_paste_chronology(compositor_factory):
    # With several pastes in one call, later pastes routinely cover part of
    # earlier ones (destination-original or already-pasted). Every returned
    # mask must reflect only what's still visible after the whole chronology -
    # modal, not amodal - so no two instances' masks may share a pixel, and
    # each annotation's "area" must match its own mask's pixel count.
    compositor, _ = compositor_factory(n_sources=5, min_instances=3, max_instances=3)
    dest = _solid_image(64, 0)

    random.seed(9)
    _, new_anns = compositor(dest, [])

    masks = [mask_util.decode(a["segmentation"]).astype(bool) for a in new_anns]
    for ann, mask in zip(new_anns, masks):
        assert ann["area"] == float(mask.sum())
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            assert not np.any(masks[i] & masks[j])


def test_cluster_prob_zero_matches_uniform_path(compositor_factory):
    # cluster_prob=0.0 (the fixture default) must be bit-identical to the
    # pre-clustering uniform placement for the same seed - the one extra
    # random.random() draw __call__ now always makes must not itself change
    # *which* pixels get chosen, only that a fixed extra draw is consumed.
    compositor, _ = compositor_factory(n_sources=5, min_instances=3, max_instances=3)
    dest = _solid_image(64, 0)

    random.seed(20)
    _, anns_a = compositor(dest, [])
    random.seed(20)
    _, anns_b = compositor(dest, [])

    for a, b in zip(anns_a, anns_b):
        assert a["bbox"] == b["bbox"]


def test_cluster_prob_one_concentrates_placements_near_center(compositor_factory):
    # cluster_prob=1.0, cluster_stray_fraction=0.0 -> every instance is piled,
    # so every pasted mask's centroid should land within a small multiple of
    # sigma of the group's own mean centroid (the true center isn't directly
    # observable from the public API, so we check mutual concentration).
    compositor, _ = compositor_factory(
        n_sources=6,
        min_instances=6,
        max_instances=6,
        cluster_prob=1.0,
        cluster_radius_fraction=0.15,
        cluster_stray_fraction=0.0,
    )
    dest = _solid_image(200, 0)

    random.seed(21)
    _, anns = compositor(dest, [])
    centroids = []
    for ann in anns:
        mask = mask_util.decode(ann["segmentation"]).astype(bool)
        ys, xs = np.where(mask)
        centroids.append((ys.mean(), xs.mean()))

    sigma = compositor._cluster_sigma(200, 200)
    mean_y = sum(c[0] for c in centroids) / len(centroids)
    mean_x = sum(c[1] for c in centroids) / len(centroids)
    for y, x in centroids:
        assert abs(y - mean_y) < 4 * sigma
        assert abs(x - mean_x) < 4 * sigma


def test_cluster_center_moves_across_calls(compositor_factory):
    # The Gaussian center must be freshly randomized per __call__, never
    # fixed. Two different seeds should (with overwhelming probability, given
    # a small std relative to a large canvas) produce different pile
    # locations.
    compositor, _ = compositor_factory(
        n_sources=4,
        min_instances=4,
        max_instances=4,
        cluster_prob=1.0,
        cluster_radius_fraction=0.1,
        cluster_stray_fraction=0.0,
    )
    dest = _solid_image(256, 0)

    def _mean_centroid(seed):
        random.seed(seed)
        _, anns = compositor(dest, [])
        pts = []
        for ann in anns:
            mask = mask_util.decode(ann["segmentation"]).astype(bool)
            ys, xs = np.where(mask)
            pts.append((ys.mean(), xs.mean()))
        return sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)

    c1 = _mean_centroid(30)
    c2 = _mean_centroid(31)
    assert c1 != c2


def test_cluster_stray_fraction_one_behaves_like_uniform(compositor_factory):
    # cluster_prob=1.0 but cluster_stray_fraction=1.0 -> every instance takes
    # the stray (old uniform) branch despite the frame being "clustered",
    # i.e. functionally equivalent to the plain uniform path.
    compositor, _ = compositor_factory(
        n_sources=5,
        min_instances=3,
        max_instances=3,
        cluster_prob=1.0,
        cluster_stray_fraction=1.0,
    )
    dest = _solid_image(64, 0)

    random.seed(22)
    _, anns = compositor(dest, [])
    assert len(anns) == 3  # just a sanity check that placement still succeeds


def test_place_clustered_clamps_instead_of_rejecting(compositor_factory):
    # A center near the edge with std=0 must clamp the patch fully on-canvas
    # rather than returning None - preserving the "no retry, clamp instead"
    # guarantee for the clustered path.
    compositor, _ = compositor_factory()
    patch_mask = _l_shape(size=24)
    placement = compositor._place(patch_mask, 64, 64, center=(0.0, 0.0), std=(0.0, 0.0))
    assert placement is not None
    py, px, full_mask = placement
    assert 0 <= py <= 64 - 24
    assert 0 <= px <= 64 - 24


def test_transform_never_flips(compositor_factory):
    compositor, _ = compositor_factory()
    original = _l_shape()
    original_sign = np.sign(_signed_orientation(original))

    random.seed(3)
    for _ in range(30):
        patch, patch_mask, _valid = compositor._transform_instance(
            _solid_image(original.shape[0], 200), original
        )
        assert np.sign(_signed_orientation(patch_mask)) == original_sign


def test_transform_uses_consistent_interpolation(compositor_factory):
    # Regression test: an earlier version used INTER_LINEAR for the image but
    # INTER_NEAREST for the mask, so their boundaries disagreed at non-90-degree
    # angles - visible as spurious notches biting into a pasted instrument's
    # silhouette. Both must use INTER_NEAREST, so every warped image pixel is an
    # exact copy of an original pixel (or the border fill) - never a blended
    # in-between value - and wherever the mask says "object", the image is
    # guaranteed to be real object color, never border-fill leaking in.
    compositor, _ = compositor_factory()
    original_mask = _l_shape()
    original_image = _solid_image(original_mask.shape[0], 200)
    allowed_values = set(np.unique(original_image).tolist()) | {0}  # 0 = border fill

    random.seed(6)
    for _ in range(40):
        patch, patch_mask, _valid = compositor._transform_instance(
            original_image, original_mask
        )
        assert set(np.unique(patch).tolist()) <= allowed_values
        assert set(np.unique(patch[patch_mask]).tolist()) <= {200}


@pytest.mark.parametrize("interp", ["linear", "bicubic", "lanczos"])
def test_transform_subpixel_interp_produces_smooth_edges(compositor_factory, interp):
    # Each subpixel mode warps both image and mask with the same flag (mask
    # thresholded at 0.5 afterward) - unlike "nearest" (which only ever copies
    # exact source/border values, see test above), this must produce intermediate
    # blended pixel values at the boundary, confirming the edge is actually
    # anti-aliased/smooth rather than blocky.
    compositor, _ = compositor_factory(mask_interp=(interp,))
    original_mask = _l_shape()
    original_image = _solid_image(original_mask.shape[0], 200)

    random.seed(12)
    saw_intermediate_value = False
    for _ in range(20):
        patch, patch_mask, valid_footprint = compositor._transform_instance(
            original_image, original_mask
        )
        assert patch_mask.dtype == bool
        assert patch_mask.shape == valid_footprint.shape
        if np.any((patch > 5) & (patch < 195)):
            saw_intermediate_value = True
    assert saw_intermediate_value


def test_blend_never_bleeds_border_fill(compositor_factory):
    # Regression test: the rotation canvas is larger than the rotated crop, so its
    # corners are synthetic borderValue=0 (black) padding, not real image content.
    # patch_mask already excludes that zone, but a blur alpha ramp doesn't - it
    # can pull the synthetic black into the blend near a thin/elongated object's
    # tip, where the true boundary sits close to those padded corners. Only real
    # (rotated crop) pixels may ever contribute to the blend.
    compositor, _ = compositor_factory(
        blend_modes=("gaussian_blur_edge",),
        blur_kernel_range=(21, 21),
        rotation_degrees=45.0,
    )
    thin_mask = np.zeros((60, 60), dtype=bool)
    for i in range(60):
        thin_mask[i, i] = True
        thin_mask[i, min(i + 1, 59)] = True  # a thin diagonal line corner-to-corner
    crop = _solid_image(60, 200)
    dest_value = 100

    random.seed(11)
    for _ in range(20):
        patch, patch_mask, valid_footprint = compositor._transform_instance(
            crop, thin_mask
        )
        ph, pw = patch_mask.shape[:2]
        composite = _solid_image(ph + pw + 100, dest_value)
        py, px = 50, 50

        compositor._blend(composite, patch, patch_mask, valid_footprint, py, px)

        region = composite[py : py + ph, px : px + pw]
        assert region.min() >= min(dest_value, 200)


def test_blend_none_is_hard_edged(compositor_factory):
    compositor, _ = compositor_factory(blend_modes=("none",))
    dest = _solid_image(64, 0)

    random.seed(4)
    new_image, new_anns = compositor(dest, [])
    mask = mask_util.decode(new_anns[0]["segmentation"]).astype(bool)

    pixel_values = set(new_image[..., 0][mask.astype(bool)].tolist())
    background_values = set(new_image[..., 0][~mask].tolist())
    assert pixel_values <= {200}
    assert background_values <= {0}


def test_blend_gaussian_softens_edge(compositor_factory):
    compositor, _ = compositor_factory(
        blend_modes=("gaussian_blur_edge",),
        blur_kernel_range=(9, 9),
    )
    dest = _solid_image(64, 0)

    random.seed(5)
    new_image, new_anns = compositor(dest, [])
    mask = mask_util.decode(new_anns[0]["segmentation"]).astype(bool)
    boundary = cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) & (
        ~mask
    ).astype(np.uint8)

    boundary_values = new_image[..., 0][boundary.astype(bool)]
    assert np.any((boundary_values > 5) & (boundary_values < 195))


def test_blend_box_blur_softens_edge(compositor_factory):
    compositor, _ = compositor_factory(
        blend_modes=("box_blur",),
        blur_kernel_range=(9, 9),
    )
    dest = _solid_image(64, 0)

    random.seed(7)
    new_image, new_anns = compositor(dest, [])
    mask = mask_util.decode(new_anns[0]["segmentation"]).astype(bool)
    boundary = cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) & (
        ~mask
    ).astype(np.uint8)

    boundary_values = new_image[..., 0][boundary.astype(bool)]
    assert np.any((boundary_values > 5) & (boundary_values < 195))


def test_core_margin_protects_object_interior(compositor_factory):
    # gaussian_blur_edge/box_blur both ramp symmetrically, so a large
    # enough kernel would otherwise blend a chunk of the object's own
    # interior with the destination background. core_margin_px must force alpha=1
    # more than that many px inside the mask, regardless of kernel size.
    compositor, _ = compositor_factory(
        blend_modes=("gaussian_blur_edge",),
        blur_kernel_range=(15, 15),
        core_margin_px=2,
    )
    patch_mask = _solid_rect_mask(size=40, margin=4)  # true region is rows/cols [4,36)
    patch = _solid_image(40, 200)
    composite = _solid_image(64, 0)
    valid_footprint = np.ones_like(patch_mask, dtype=bool)
    py, px = 10, 10

    compositor._blend(composite, patch, patch_mask, valid_footprint, py, px)

    # eroded by 2px -> core is [6,34); [8,32) is safely inside it regardless of
    # the (much larger) 15px blur kernel used for the outward side.
    deep_interior = np.zeros_like(patch_mask)
    deep_interior[8:-8, 8:-8] = True
    region = composite[py : py + 40, px : px + 40]
    assert set(np.unique(region[deep_interior]).tolist()) == {200}


def test_core_margin_zero_disables_protection(compositor_factory):
    compositor, _ = compositor_factory(
        blend_modes=("gaussian_blur_edge",),
        blur_kernel_range=(31, 31),
        core_margin_px=0,
    )
    patch_mask = _solid_rect_mask(size=40, margin=4)
    patch = _solid_image(40, 200)
    composite = _solid_image(64, 0)
    valid_footprint = np.ones_like(patch_mask, dtype=bool)
    py, px = 10, 10

    compositor._blend(composite, patch, patch_mask, valid_footprint, py, px)

    # without protection, a wide enough blur reaches even fairly deep pixels
    deep_interior = np.zeros_like(patch_mask)
    deep_interior[8:-8, 8:-8] = True
    region = composite[py : py + 40, px : px + 40]
    assert not np.all(region[deep_interior] == 200)


def test_poisson_not_in_default_blend_modes():
    # "poisson" must be implemented but opt-in only (Cut, Paste and Learn's own
    # ablation found it hurts on average) - a regression guard on cfg.INPUT.COPY_PASTE
    # (CopyPasteCompositor's constructor has no default of its own - see its __init__)
    # so an experiment addition never silently becomes the shipped default.
    from detectron2.config import get_cfg

    cfg = get_cfg()
    add_maskdino_config(cfg)
    default_blend_modes = cfg.INPUT.COPY_PASTE.BLEND_MODES
    assert "poisson" not in default_blend_modes
    assert "gaussian_blur_edge" in default_blend_modes
    assert "box_blur" in default_blend_modes


def test_alpha_feather_blend_mode_is_removed(compositor_factory):
    """alpha_feather was removed in #64. A config that still lists it must fail when
    the compositor is built, not paste with a silently different blend mix."""
    with pytest.raises(ValueError, match="#64"):
        compositor_factory(blend_modes=("none", "alpha_feather"))


def test_saved_config_with_feather_width_range_still_merges(tmp_path):
    """Saved run configs carry FEATHER_WIDTH_RANGE, removed in #64. A strict merge
    must skip it instead of raising KeyError, and still merge the keys next to it."""
    from detectron2.config import get_cfg

    saved = tmp_path / "config.yaml"
    saved.write_text(
        "VERSION: 2\n"
        "INPUT:\n"
        "  COPY_PASTE:\n"
        "    FEATHER_WIDTH_RANGE: [3, 5]\n"
        "    BLUR_KERNEL_RANGE: [3, 5]\n"
    )
    cfg = get_cfg()
    add_maskdino_config(cfg)
    add_maskdino_config(cfg)  # yacs asserts on registering a deprecated key twice
    cfg.merge_from_file(str(saved))

    assert "FEATHER_WIDTH_RANGE" not in cfg.INPUT.COPY_PASTE
    assert cfg.INPUT.COPY_PASTE.BLUR_KERNEL_RANGE == [3, 5]
    assert cfg.key_is_deprecated("INPUT.COPY_PASTE.FEATHER_WIDTH_RANGE")


def test_blend_poisson_applies_when_selected(compositor_factory):
    compositor, _ = compositor_factory(blend_modes=("poisson",))
    composite = _solid_image(128, 50)
    patch = _solid_image(24, 200)
    patch_mask = _l_shape()
    py, px = 40, 40  # well inside the 128x128 canvas - padded patch won't spill

    applied = compositor._blend_poisson(composite, patch, patch_mask, py, px)

    assert applied
    assert not np.array_equal(composite, _solid_image(128, 50))


def test_blend_poisson_falls_back_near_edge(compositor_factory):
    compositor, _ = compositor_factory(blend_modes=("poisson",))
    composite = _solid_image(32, 50)
    patch = _solid_image(24, 200)
    patch_mask = _l_shape()
    py, px = 0, 0  # padded patch spills past the top-left edge

    applied = compositor._blend_poisson(composite, patch, patch_mask, py, px)

    assert not applied
    assert np.array_equal(composite, _solid_image(32, 50))  # untouched


def test_poisson_blend_mode_falls_back_gracefully_in_full_call(compositor_factory):
    # Regardless of where random placement lands (including right at an edge,
    # where _blend_poisson declines), _blend must fall back to a hard paste
    # rather than raising.
    compositor, _ = compositor_factory(blend_modes=("poisson",))
    dest = _solid_image(64, 0)

    random.seed(8)
    new_image, new_anns = compositor(dest, [])

    assert len(new_anns) == 1
    assert mask_util.decode(new_anns[0]["segmentation"]).sum() > 0


def test_returned_annotations_match_evaluator_gt_contract(compositor_factory):
    """Pins what tools/eval_composited_confusion.py hands to
    HungarianInstanceEvaluator._load_gt_masks: every annotation must decode to a
    full-frame (H, W) mask and carry a category_id from the source class space,
    with no conversion step in between."""
    random.seed(7)
    compositor, _ = compositor_factory(min_instances=3, max_instances=3)
    dest = _solid_image(64, 0)
    dest_ann = _make_ann(_solid_rect_mask(size=64, margin=20), category_id=5)

    _, anns = compositor(dest, [dest_ann])

    assert len(anns) >= 2
    for ann in anns:
        assert ann["category_id"] in {5, 6, 7}
        assert ann["iscrowd"] == 0
        assert ann["bbox_mode"] == 1
        assert 0.0 <= ann["visibility_fraction"] <= 1.0
        mask = mask_util.decode(ann["segmentation"])
        assert mask.shape == (64, 64)
        assert mask.dtype == np.uint8


# --- per-instance appearance jitter (INPUT.COPY_PASTE.INSTANCE_JITTER) --------
#
# The contract is narrow and worth pinning: jitter changes PIXELS ONLY. If it ever
# moved geometry, every returned annotation would disagree with what was painted,
# and the modal-mask bookkeeping in _update_modal_annotation would be silently wrong.

_LIVE_JITTER = {
    "brightness": (0.7, 1.3),
    "contrast": (0.8, 1.2),
    "saturation": (0.8, 1.2),
    "hue_degrees": 10.0,
    "blur_sigma": (0.0, 1.5),
}


def test_normalize_jitter_treats_identity_config_as_off():
    assert cp_mod._normalize_jitter(None) is None
    assert cp_mod._normalize_jitter({}) is None
    assert (
        cp_mod._normalize_jitter(
            {
                "brightness": (1.0, 1.0),
                "contrast": (1.0, 1.0),
                "saturation": (1.0, 1.0),
                "hue_degrees": 0.0,
                "blur_sigma": (0.0, 0.0),
            }
        )
        is None
    )


def test_normalize_jitter_reads_uppercase_config_keys():
    # yacs CfgNode is a dict subclass with UPPERCASE keys. Reading only lowercase
    # would make every arm config silently train with jitter OFF.
    from yacs.config import CfgNode as CN

    cfg = CN()
    cfg.BRIGHTNESS = [0.8, 1.2]
    cfg.CONTRAST = [1.0, 1.0]
    cfg.SATURATION = [1.0, 1.0]
    cfg.HUE_DEGREES = 0.0
    cfg.BLUR_SIGMA = [0.0, 0.0]
    assert cp_mod._normalize_jitter(cfg)["brightness"] == (0.8, 1.2)


def test_default_config_leaves_instance_jitter_off():
    from detectron2.config import get_cfg

    cfg = get_cfg()
    add_maskdino_config(cfg)
    assert cp_mod._normalize_jitter(cfg.INPUT.COPY_PASTE.INSTANCE_JITTER) is None


def test_jitter_off_is_bit_identical(compositor_factory):
    random.seed(7)
    plain, _ = compositor_factory()
    out_plain, anns_plain = plain(_solid_image(64, 0), [])

    random.seed(7)
    explicit_noop, _ = compositor_factory(
        instance_jitter={"brightness": (1.0, 1.0), "blur_sigma": (0.0, 0.0)}
    )
    out_noop, anns_noop = explicit_noop(_solid_image(64, 0), [])

    assert np.array_equal(out_plain, out_noop)
    assert [a["segmentation"] for a in anns_plain] == [
        a["segmentation"] for a in anns_noop
    ]


def test_jitter_changes_pixels_but_not_geometry(compositor_factory):
    random.seed(11)
    plain, _ = compositor_factory()
    out_plain, anns_plain = plain(_solid_image(64, 0), [])

    random.seed(11)
    jittered, _ = compositor_factory(instance_jitter=_LIVE_JITTER)
    out_jit, anns_jit = jittered(_solid_image(64, 0), [])

    # Same draws for placement/rotation (the jitter draws come last, after every
    # geometric decision), so the masks must match exactly...
    assert len(anns_plain) == len(anns_jit) == 1
    for a, b in zip(anns_plain, anns_jit):
        assert a["segmentation"] == b["segmentation"]
        assert a["bbox"] == b["bbox"]
        assert a["area"] == b["area"]
        assert a["visibility_fraction"] == b["visibility_fraction"]
    # ...while the painted pixels must not.
    assert not np.array_equal(out_plain, out_jit)


def test_jitter_only_repaints_inside_the_pasted_footprint(compositor_factory):
    random.seed(3)
    plain, _ = compositor_factory()
    out_plain, anns = plain(_solid_image(64, 0), [])

    random.seed(3)
    jittered, _ = compositor_factory(instance_jitter=_LIVE_JITTER)
    out_jit, _ = jittered(_solid_image(64, 0), [])

    pasted = mask_util.decode(anns[0]["segmentation"]).astype(bool)
    changed = np.any(out_plain != out_jit, axis=2)
    # Blending softens a few pixels just outside the mask, so allow the difference
    # to touch the dilated footprint - but never the far background.
    allowed = cv2.dilate(pasted.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    assert not np.any(changed & ~allowed)


def test_jitter_produces_a_different_appearance_each_paste(compositor_factory):
    # The whole point: one cached source crop must not look identical every time.
    # Same global seed on both calls -> identical placement, so any pixel difference
    # is the private jitter stream advancing, which is exactly what should vary.
    jittered, _ = compositor_factory(instance_jitter=_LIVE_JITTER)
    random.seed(21)
    first, anns_first = jittered(_solid_image(64, 0), [])
    random.seed(21)
    second, anns_second = jittered(_solid_image(64, 0), [])
    assert anns_first[0]["segmentation"] == anns_second[0]["segmentation"]
    assert not np.array_equal(first, second)


# --- per-instance projective warp (INPUT.COPY_PASTE.PERSPECTIVE) -------------
#
# Unlike INSTANCE_JITTER this one DOES change geometry, so the contract is the
# opposite: the mask must move exactly with the pixels, and the returned annotation
# must describe where the instrument actually landed.

_LIVE_PERSPECTIVE = {"prob": 1.0, "tilt_degrees": 20.0}


def test_normalize_perspective_off_unless_both_knobs_set():
    assert cp_mod._normalize_perspective(None) is None
    assert cp_mod._normalize_perspective({"prob": 0.0, "tilt_degrees": 15.0}) is None
    assert cp_mod._normalize_perspective({"prob": 1.0, "tilt_degrees": 0.0}) is None
    assert cp_mod._normalize_perspective({"prob": 0.5, "tilt_degrees": 15.0}) == (
        0.5,
        15.0,
    )


def test_default_config_leaves_instance_perspective_off():
    from detectron2.config import get_cfg

    cfg = get_cfg()
    add_maskdino_config(cfg)
    assert cp_mod._normalize_perspective(cfg.INPUT.COPY_PASTE.PERSPECTIVE) is None


def test_perspective_off_is_bit_identical(compositor_factory):
    random.seed(17)
    plain, _ = compositor_factory()
    out_plain, anns_plain = plain(_solid_image(64, 0), [])

    random.seed(17)
    explicit_off, _ = compositor_factory(
        instance_perspective={"prob": 0.0, "tilt_degrees": 15.0}
    )
    out_off, anns_off = explicit_off(_solid_image(64, 0), [])

    assert np.array_equal(out_plain, out_off)
    assert [a["segmentation"] for a in anns_plain] == [
        a["segmentation"] for a in anns_off
    ]


def test_perspective_changes_the_pasted_shape(compositor_factory):
    random.seed(23)
    plain, _ = compositor_factory()
    _out_plain, anns_plain = plain(_solid_image(64, 0), [])

    random.seed(23)
    warped, _ = compositor_factory(instance_perspective=_LIVE_PERSPECTIVE)
    _out_warp, anns_warp = warped(_solid_image(64, 0), [])

    assert anns_plain[0]["segmentation"] != anns_warp[0]["segmentation"]


def test_perspective_keeps_mask_on_the_pasted_pixels(compositor_factory):
    # The decisive property. The sources are bright L-shapes on a black destination,
    # so every pixel the compositor actually painted must be inside the returned
    # mask - if the warp moved image and mask apart, painted pixels would fall
    # outside it.
    random.seed(29)
    warped, _ = compositor_factory(
        instance_perspective=_LIVE_PERSPECTIVE, blend_modes=("none",)
    )
    out, anns = warped(_solid_image(64, 0), [])

    mask = mask_util.decode(anns[0]["segmentation"]).astype(bool)
    painted = out.max(axis=2) > 0
    assert painted.sum() > 0
    stray = np.logical_and(painted, ~mask).sum()
    assert stray == 0, f"{stray} painted pixels fell outside the instance mask"


def test_perspective_is_area_neutral_on_average(compositor_factory):
    # This per-instance version expands its canvas and
    # rescales the tilted quad back to the source area, so it must not systematically
    # grow or shrink instruments - apparent size is a class cue here.
    areas_plain, areas_warped = [], []
    for seed in range(40):
        random.seed(seed)
        plain, _ = compositor_factory()
        _o, a = plain(_solid_image(64, 0), [])
        areas_plain.append(a[0]["area"])

        random.seed(seed)
        warped, _ = compositor_factory(instance_perspective=_LIVE_PERSPECTIVE)
        _o, a = warped(_solid_image(64, 0), [])
        areas_warped.append(a[0]["area"])

    ratio = np.mean(areas_warped) / np.mean(areas_plain)
    assert 0.93 < ratio < 1.07, f"systematic area change {ratio:.3f}"


def test_perspective_does_not_shift_placement_draws(compositor_factory):
    # Perspective draws come from a private RNG, so enabling it must not consume
    # global draws and move every subsequent paste - otherwise an arm testing
    # perspective would also differ in pile layout.
    random.seed(31)
    plain, _ = compositor_factory(min_instances=3, max_instances=3)
    _o, anns_plain = plain(_solid_image(96, 0), [])

    random.seed(31)
    warped, _ = compositor_factory(
        min_instances=3, max_instances=3, instance_perspective=_LIVE_PERSPECTIVE
    )
    _o, anns_warp = warped(_solid_image(96, 0), [])

    assert len(anns_plain) == len(anns_warp)
    # Same instruments, in the same order, at roughly the same spots.
    assert [a["category_id"] for a in anns_plain] == [
        a["category_id"] for a in anns_warp
    ]


def test_tilt_is_sampled_uniformly_in_angle():
    # The point of parameterizing by angle: coverage must be flat across the range,
    # not concentrated in the middle the way independent per-corner offsets are.
    import random as _random

    rng = _random.Random(0)
    tilts = [rng.uniform(0.0, 18.0) for _ in range(6000)]
    counts, _ = np.histogram(tilts, bins=6, range=(0.0, 18.0))
    assert counts.min() / counts.max() > 0.85


def test_tilt_homography_is_area_preserving():
    # A real tilt foreshortens; the helper rescales it back, because apparent size
    # is class-informative in this dataset.
    corners = np.float32([[0, 0], [99, 0], [99, 99], [0, 99]])

    def area(q):
        x, y = q[:, 0], q[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    for tilt in (5.0, 15.0, 25.0):
        h = cp_mod._tilt_homography(100, 100, tilt, 0.7)
        pts = np.concatenate([corners, np.ones((4, 1), np.float32)], axis=1) @ h.T
        assert area(pts[:, :2] / pts[:, 2:3]) == pytest.approx(area(corners), rel=0.02)


def test_zero_tilt_is_the_identity():
    h = cp_mod._tilt_homography(100, 80, 0.0, 1.3)
    assert np.allclose(h / h[2, 2], np.eye(3), atol=1e-6)


def test_contrast_pivots_on_instrument_not_patch_background():
    """Contrast must pivot on the instrument's own pixels, not the whole patch.

    The patch rect also holds the crop's mat background, which sits well below the
    instrument. Pivoting there puts every instrument pixel above the pivot, so
    raising contrast brightens the instrument and fights the brightness knob
    (measured on real crops: at brightness 0.70, contrast 1.0 -> 2.6 moved the tool
    mean 114.8 -> 183.8). It also makes a given contrast value mean different things
    for different instrument shapes, since the pivot drifts with how much background
    the bounding box contains.
    """

    patch = np.zeros((40, 40, 3), np.uint8)
    patch[:] = 40  # dark "mat" background
    mask = np.zeros((40, 40), bool)
    mask[10:30, 10:30] = True
    patch[mask] = 160  # bright "instrument"

    comp = CopyPasteCompositor.__new__(CopyPasteCompositor)
    comp.image_format = "BGR"
    comp._rng_base = 7
    comp._rng_worker = None
    comp.instance_jitter = {
        "brightness": (1.0, 1.0),
        "contrast": (2.0, 2.0),
        "saturation": (1.0, 1.0),
        "hue_degrees": 0.0,
        "blur_sigma": (0.0, 0.0),
    }

    before = patch[mask].mean()
    after = comp._jitter_appearance(patch, mask)[mask].mean()
    # Pivoting on the instrument mean leaves that mean where it was; only the
    # spread around it changes.
    assert abs(after - before) < 3.0

    # ...and without a mask it must fall back to the old whole-patch pivot rather
    # than crashing, which is what a caller with an empty mask would hit.
    comp._rng_worker = None
    fallback = comp._jitter_appearance(patch, None)[mask].mean()
    assert fallback > before + 10


def test_contrast_still_changes_internal_spread():
    # Decoupling must not mean "contrast does nothing": it should still scale the
    # instrument's internal texture, which is the statistic being matched to real.
    patch = np.full((40, 40, 3), 40, np.uint8)
    mask = np.zeros((40, 40), bool)
    mask[10:30, 10:30] = True
    rng = np.random.default_rng(0)
    patch[mask] = rng.integers(120, 200, size=(mask.sum(), 3), dtype=np.uint8)

    comp = CopyPasteCompositor.__new__(CopyPasteCompositor)
    comp.image_format = "BGR"
    comp._rng_base = 7
    comp._rng_worker = None
    comp.instance_jitter = {
        "brightness": (1.0, 1.0),
        "contrast": (1.8, 1.8),
        "saturation": (1.0, 1.0),
        "hue_degrees": 0.0,
        "blur_sigma": (0.0, 0.0),
    }
    out = comp._jitter_appearance(patch, mask)
    assert out[mask].std() > patch[mask].std() * 1.4
