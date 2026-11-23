"""Crops on disk: a folder crop must equal the crop the live pipeline builds.

The fake mask here is deliberately FRAGMENTED - dashed bars plus a blob, like the predicted
masks `crops.py:52-54` describes. An earlier version of these tests used a solid rectangle,
which made every dilation assertion vacuous: a solid mask equals its own tight box, and with
`box=tight` the crop region never grows, so `mask_dilate=16` produced a crop bit-identical to
`mask_dilate=0`. The test passed with dilation deleted.
"""

from __future__ import annotations

import csv
import json

import pytest
import torch

from cropcls.assign import assign_gt_classes
from cropcls.config import CropClsConfig
from cropcls.crop_folder import (
    CropFolder,
    build_loader,
    load_crop,
    patch_bbox,
    save_crop,
    unclipped_border,
    write_frame_crops,
)
from cropcls.crops import (
    letterbox_crop,
    mask_tight_bbox,
    normalize_crop,
    to_model_input,
)

# Deliberately NOT sorted: thing_classes puts scalpel before hammer, sorted() does the
# reverse. A fixture in sorted order would make the real label-order bug invisible.
CLASSES = ["forcep01_mm", "scalpel01_mm", "hammer01_mm"]
assert CLASSES != sorted(CLASSES), "the fixture must exercise the order mismatch"
H = W = 256


def _frame(seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (3, H, W), dtype=torch.uint8, generator=g)


def _fragmented(seed: int) -> torch.Tensor:
    """A dashed shaft plus a jaw blob: holes that dilation closes and erosion reopens."""
    m = torch.zeros((H, W), dtype=torch.bool)
    y, x = 100 + (seed % 3), 60 + (seed % 5)
    for i in range(0, 110, 14):                     # dashed horizontal shaft
        m[y : y + 6, x + i : x + i + 9] = True
    m[y - 18 : y + 24, x + 46 : x + 52] = True      # crossbar
    m[y - 8 : y + 14, x + 104 : x + 126] = True     # jaws
    return m


def _cfg(**kw) -> CropClsConfig:
    base = {
        "detector_checkpoint": "/fake/det.pth", "detector_config": "/fake/cfg.yaml",
        "crop_size": 64, "photometric": False, "rotations": [0],
        "crops_per_batch": 2, "pad_frac": 0.10, "num_workers": 0,
    }
    base.update(kw)
    return CropClsConfig(**base)


def _folder(tmp_path, n=5, pad_frac=0.10):
    """Write n crops the way the extraction script does. Returns (root, truth)."""
    root = tmp_path / "crops"
    root.mkdir()
    truth = []
    borders = []
    for i in range(n):
        frame, mask = _frame(i), _fragmented(i)
        cls = CLASSES[i % len(CLASSES)]
        box = patch_bbox(mask, pad_frac)
        x0, y0, x1, y1 = box
        H, W = mask.shape
        b = unclipped_border(mask_tight_bbox(mask), box, H, W)
        if b is not None:
            borders.append(b)
        (root / cls).mkdir(exist_ok=True)
        save_crop(root / cls / f"f{i:03d}_p000.png", frame[:, y0:y1, x0:x1], mask[y0:y1, x0:x1])
        truth.append((f"f{i:03d}_p000.png", cls, frame, mask))
    (root / "meta.json").write_text(json.dumps({
        "pad_frac": pad_frac,
        "min_border_px": min(borders) if borders else None,
        "thing_classes": CLASSES,
    }))
    return root, truth


def _live(frame, mask, cfg, photometric=None):
    """What `data.py` builds: box= the UNDILATED tight box, grown by cfg.pad_frac."""
    out = letterbox_crop(
        frame.float(), mask, cfg.crop_size, box=mask_tight_bbox(mask),
        pad_frac=cfg.pad_frac,
        photometric=photometric, mask_only=cfg.mask_only, blackout=cfg.blackout,
        dilate=cfg.mask_dilate, close=cfg.mask_close,
    )
    return to_model_input(out, cfg.pixel_mean, cfg.pixel_std,
                          mask_channel=cfg.mask_channel)


# --- the folder crop equals the live crop -------------------------------------


@pytest.mark.parametrize("blackout", [True, False])
@pytest.mark.parametrize("mask_channel", [False, True])
@pytest.mark.parametrize(
    ("dilate", "close"), [(0, False), (8, False), (16, False), (8, True), (16, True)]
)
def test_folder_crop_equals_live_crop(tmp_path, dilate, close, mask_channel, blackout):
    """Over 5 frames, every mask setting, at `pad_frac` equal to the stored one.

    The live path clips the padded box to the FRAME and the folder path clips it to the stored
    patch; identity here is what says those two agree, including on the frame-clipped sides.
    `close=True` is the case where the stored border is load-bearing beyond the crop region -
    with `close=False` every mask pixel is inside its own tight box, so the border is unused.

    `blackout=False` and `mask_channel=True` are the settings spec_cropcls.md Step 10 ships,
    and the matrix reached neither before: `_cfg` inherits `blackout: True`, so folder-vs-live
    identity had never been exercised at the only blackout setting those arms use.
    """
    # pad_frac is relative, and these masks are 126x42 like the real shafts (aspect ~1.7-1.8
    # measured), so the SHORT axis sets the border: 0.10 gives 4 px, below the closing radii.
    # Closing cases therefore run at a fraction that actually covers them.
    pf = 0.40 if close else 0.10
    cfg = _cfg(mask_dilate=dilate, mask_close=close, pad_frac=pf,
               blackout=blackout, mask_channel=mask_channel)
    root, truth = _folder(tmp_path, n=5, pad_frac=pf)
    ds = CropFolder(root, cfg, is_train=False)
    assert len(ds) == 5
    by_name = {p.name: i for i, (p, _) in enumerate(ds.items)}
    for name, cls, frame, mask in truth:
        got, _, label = ds[by_name[name]]
        assert ds.class_names[label] == cls
        assert torch.equal(got, _live(frame, mask, cfg)), (
            f"{name} dilate={dilate} close={close} "
            f"mask_channel={mask_channel} blackout={blackout}"
        )


def test_dilating_changes_the_crop(tmp_path):
    """The assertion the old suite was missing, and the one that caught the fake fixture.

    If this passes with `mask_dilate` ignored, the fixture mask is solid again.
    """
    root, _ = _folder(tmp_path)
    a = CropFolder(root, _cfg(mask_dilate=0), is_train=False)[0][0]
    b = CropFolder(root, _cfg(mask_dilate=8), is_train=False)[0][0]
    assert not torch.equal(a, b), "dilation changed nothing; the fixture mask is not fragmented"


def test_closing_differs_from_plain_dilation(tmp_path):
    root, _ = _folder(tmp_path, pad_frac=0.40)   # closing needs border >= its radius
    cfg = {"pad_frac": 0.40, "mask_dilate": 8}
    d = CropFolder(root, _cfg(mask_close=False, **cfg), is_train=False)[0][0]
    c = CropFolder(root, _cfg(mask_close=True, **cfg), is_train=False)[0][0]
    assert not torch.equal(d, c)


def test_blackout_off_keeps_background(tmp_path):
    """blackout=False must actually change the crop, or the ablation measures nothing."""
    root, _ = _folder(tmp_path)
    on = CropFolder(root, _cfg(blackout=True), is_train=False)[0][0]
    off = CropFolder(root, _cfg(blackout=False), is_train=False)[0][0]
    assert not torch.equal(on, off)


# --- the 4th channel is what keeps photometric usable -------------------------


def test_photometric_runs_before_blackout(tmp_path):
    """The reason the mask is stored in alpha instead of baked into the RGB.

    Contrast blends toward the image mean; on a blacked crop that mean is near zero, so
    jitter applied after blackout behaves nothing like the frame-level op it replaces. Here
    it must change pixels inside the mask while leaving the background exactly 0.
    """
    cfg = _cfg(photometric=True)
    root, _ = _folder(tmp_path)
    ds = CropFolder(root, cfg, is_train=True, seed=3)
    torch.manual_seed(0)
    a, _, _ = ds[0]
    torch.manual_seed(1)
    b, _, _ = ds[0]
    assert not torch.equal(a, b), "photometric did not fire"

    plain = CropFolder(root, _cfg(photometric=False), is_train=False)[0][0]
    bg = plain == normalize_crop(
        torch.zeros(1), cfg.pixel_mean, cfg.pixel_std
    ).view(-1, 1, 1).expand_as(plain)
    assert torch.equal(a[bg], plain[bg]), "jitter leaked into the blacked-out background"


# --- what the extraction script writes ----------------------------------------


def test_write_frame_crops_writes_only_kept_detections(tmp_path):
    """A dropped detection gets a CSV row and no file - never silence."""
    frame = _frame(0)
    gt = torch.stack([_fragmented(0), _fragmented(1)])
    good = _fragmented(0)
    contaminated = gt[0] | gt[1]                      # straddles two GTs -> dropped
    masks = torch.stack([good, contaminated])
    asg = assign_gt_classes(masks, gt, torch.tensor([0, 1]))

    out = tmp_path / "crops"
    out.mkdir()
    rows = write_frame_crops(out, "f000", frame, masks, asg, CLASSES, pad_frac=0.10)

    assert len(rows) == 2, "one row per detection, kept or not"
    written = [r for r in rows if r["crop"]]
    assert len(written) == sum(bool(x) for x in asg.keep)
    assert len(list(out.rglob("*.png"))) == len(written)
    for r in rows:
        if not r["crop"]:
            assert r["verdict"].startswith("dropped"), r


def test_written_folder_loads_back(tmp_path):
    """The round trip the script depends on: what it writes, CropFolder reads."""
    frame, mask = _frame(0), _fragmented(0)
    asg = assign_gt_classes(mask[None], mask[None], torch.tensor([1]))
    out = tmp_path / "crops"
    out.mkdir()
    rows = write_frame_crops(out, "f000", frame, mask[None], asg, CLASSES, pad_frac=0.10)
    assert rows[0]["crop"] == f"{CLASSES[1]}/f000_p000.png"
    # the script writes meta.json alongside; CropFolder needs it for the label order
    (out / "meta.json").write_text(json.dumps(
        {"pad_frac": 0.10, "min_border_px": 4, "thing_classes": CLASSES}
    ))

    ds = CropFolder(out, _cfg(), is_train=False)
    assert len(ds) == 1
    got, _, label = ds[0]
    assert ds.class_names[label] == CLASSES[1]
    assert torch.equal(got, _live(frame, mask, _cfg()))


def test_stored_crop_is_lossless(tmp_path):
    """PNG round-trip must be exact, or bit-identity to the live path is luck."""
    frame, mask = _frame(0), _fragmented(0)
    x0, y0, x1, y1 = patch_bbox(mask, 0.10)
    p = tmp_path / "c.png"
    save_crop(p, frame[:, y0:y1, x0:x1], mask[y0:y1, x0:x1])
    rgb, m, _ = load_crop(p)
    assert torch.equal(rgb, frame[:, y0:y1, x0:x1])
    assert torch.equal(m, mask[y0:y1, x0:x1])


def test_float_frame_is_refused(tmp_path):
    """A float frame would quantize through PNG and silently break bit-identity."""
    with pytest.raises(TypeError, match="uint8"):
        save_crop(tmp_path / "c.png", torch.zeros((3, 8, 8)), torch.ones((8, 8), dtype=torch.bool))


# --- loading and batching -----------------------------------------------------


def test_class_order_comes_from_thing_classes_not_sorted_dirs(tmp_path):
    """The label ORDER must come from meta.json, never from sorted() directory names.

    This is the bug the first sanity run hit. thing_classes is
    [forcep01, scalpel01, hammer01, sharpspoon01, scarstick01, ...] while sorted() puts
    hammer before scalpel and scarstick before sharpspoon. The forceps coincide, so a sorted
    order trains those four correctly and silently mislabels the rest - measured as forceps
    0.27-0.82 and every other class 0.00-0.03.
    """
    root, truth = _folder(tmp_path, n=3)
    ds = CropFolder(root, _cfg(), is_train=False)
    assert ds.class_names == CLASSES, "order must be thing_classes, not sorted()"
    assert ds.class_names != sorted(CLASSES), "the fixture stopped exercising the mismatch"
    by_name = {p.name: i for i, (p, _) in enumerate(ds.items)}
    for name, cls, _frame, _mask in truth:
        _, _, label = ds[by_name[name]]
        assert ds.class_names[label] == cls, f"{name} labelled {ds.class_names[label]}, not {cls}"


def test_missing_meta_raises_rather_than_guessing_the_order(tmp_path):
    """Without meta.json the order is unknowable, and guessing it mislabels silently."""
    root, _ = _folder(tmp_path, n=3)
    (root / "meta.json").unlink()
    with pytest.raises(FileNotFoundError, match="class ORDER is unknown"):
        CropFolder(root, _cfg(), is_train=False)


def test_a_folder_outside_the_label_space_raises(tmp_path):
    """Crops extracted against a different dataset must not load as if they belonged."""
    root, _ = _folder(tmp_path, n=3)
    (root / "retractor99_mm").mkdir()
    (root / "retractor99_mm" / "x.png").write_bytes((root / CLASSES[0]).glob("*.png").__next__().read_bytes())
    with pytest.raises(ValueError, match="not in the recorded label space"):
        CropFolder(root, _cfg(), is_train=False)


def test_every_batch_has_the_configured_crop_count(tmp_path):
    """Constant batch size is what makes BatchNorm statistics comparable between steps."""
    root, _ = _folder(tmp_path, n=9)
    sizes = [x.shape[0] for x, _, _ in build_loader(_cfg(crops_per_batch=4), root, True, seed=0)]
    assert sizes == [4, 4], sizes


def test_successive_passes_draw_in_a_different_order(tmp_path):
    """Asserted on the ORDER: a full shuffled pass yields every crop exactly once, so the
    multiset is identical by construction and a multiset assertion could not fail."""
    root, _ = _folder(tmp_path, n=12)
    cfg = _cfg(crops_per_batch=2)
    runs = [[tuple(y.tolist()) for _, _, y in build_loader(cfg, root, True, seed=s)] for s in (0, 1)]
    assert sorted(x for b in runs[0] for x in b) == sorted(x for b in runs[1] for x in b)
    assert runs[0] != runs[1], "two passes drew in the same order"


def test_one_crop_gets_more_than_one_rotation_across_draws(tmp_path):
    """Rotation is sampled per draw, so no instance is pinned to one orientation."""
    root, _ = _folder(tmp_path, n=1)
    ds = CropFolder(root, _cfg(rotations=[0, 90, 180, 270]), is_train=True, seed=5)
    # Fingerprint the WHOLE crop. The first 32 values are the canvas corner, which is padding
    # for every rotation once pad_frac pushes the silhouette off the edge - so slicing them
    # made four distinct rotations look identical.
    seen = {hash(ds[0][0].numpy().tobytes()) for _ in range(40)}
    assert len(seen) >= 2


def test_eval_loading_is_unrotated_and_deterministic(tmp_path):
    """Augmentation must not leak into validation, or val numbers wander between passes."""
    root, _ = _folder(tmp_path, n=3)
    ds = CropFolder(root, _cfg(rotations=[0, 90, 180, 270], photometric=True), is_train=False)
    assert ds.rotations == [0] and ds.photometric is None
    assert all(torch.equal(ds[i][0], ds[i][0]) for i in range(len(ds)))


def test_eval_loader_keeps_the_short_final_batch(tmp_path):
    """drop_last on val would silently score fewer crops than the folder holds."""
    root, _ = _folder(tmp_path, n=5)
    loader = build_loader(_cfg(crops_per_batch=2), root, is_train=False)
    assert sum(x.shape[0] for x, _, _ in loader) == 5


def test_too_few_crops_for_one_batch_raises(tmp_path):
    """drop_last would yield nothing and the training loop would spin silently."""
    root, _ = _folder(tmp_path, n=2)
    with pytest.raises(ValueError, match="would yield nothing"):
        build_loader(_cfg(crops_per_batch=8), root, is_train=True)


def test_empty_folder_raises(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="no class folders"):
        CropFolder(tmp_path / "empty", _cfg())


def test_csv_rows_cover_every_detection(tmp_path):
    """The assignment rate is read off this file; a missing row would inflate it."""
    frame = _frame(0)
    gt = torch.stack([_fragmented(0), _fragmented(1)])
    masks = torch.stack([_fragmented(0), gt[0] | gt[1], torch.zeros((H, W), dtype=torch.bool)])
    asg = assign_gt_classes(masks, gt, torch.tensor([0, 1]))
    out = tmp_path / "crops"
    out.mkdir()
    rows = write_frame_crops(out, "f000", frame, masks, asg, CLASSES, pad_frac=0.10)
    path = out / "crops.csv"
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with path.open() as fh:
        assert len(list(csv.DictReader(fh))) == len(asg) == 3


def test_closing_wider_than_the_stored_border_raises(tmp_path):
    """Closing needs the stored border to cover its radius, checked against the MEASURED border.

    `pad_frac` is relative and clipped at the frame, so a nominal fraction promises no
    particular pixel count - which is why the check reads `min_border_px` rather than
    recomputing it from the fraction.
    """
    root, _ = _folder(tmp_path, pad_frac=0.02)
    with pytest.raises(ValueError, match="thinnest in this folder"):
        CropFolder(root, _cfg(mask_dilate=16, mask_close=True, pad_frac=0.02), is_train=False)


def test_closing_within_the_stored_border_is_allowed(tmp_path):
    root, _ = _folder(tmp_path, pad_frac=0.50)
    ds = CropFolder(root, _cfg(mask_dilate=16, mask_close=True, pad_frac=0.50), is_train=False)
    assert len(ds) == 5


def test_plain_dilation_needs_no_border_at_all(tmp_path):
    """pad_frac=0 must still serve the whole dilation sweep, or it forecloses +4.2."""
    root, truth = _folder(tmp_path, pad_frac=0.0)
    cfg = _cfg(mask_dilate=16, mask_close=False, pad_frac=0.0)
    ds = CropFolder(root, cfg, is_train=False)
    by_name = {p.name: i for i, (p, _) in enumerate(ds.items)}
    for name, _cls, frame, mask in truth:
        assert torch.equal(ds[by_name[name]][0], _live(frame, mask, cfg)), name


# --- the stored patch is a ceiling --------------------------------------------


def test_training_below_the_stored_pad_frac_still_matches_live(tmp_path):
    """One folder serves every pad_frac at or below the stored one, bit for bit.

    This is what makes the tight control free: extract once at 0.20, then train 0.0/0.10/0.20
    off the same PNGs with no re-extraction and no frame-to-frame mismatch against the control.
    """
    root, truth = _folder(tmp_path, pad_frac=0.20)
    for pf in (0.0, 0.05, 0.10, 0.20):
        cfg = _cfg(pad_frac=pf)
        ds = CropFolder(root, cfg, is_train=False)
        by_name = {p.name: i for i, (p, _) in enumerate(ds.items)}
        for name, _cls, frame, mask in truth:
            assert torch.equal(ds[by_name[name]][0], _live(frame, mask, cfg)), f"{name} pf={pf}"


def test_training_above_the_stored_pad_frac_raises(tmp_path):
    """The border pixels are not in the PNG, so cropping short would be silent."""
    root, _ = _folder(tmp_path, pad_frac=0.10)
    with pytest.raises(ValueError, match="cannot be invented"):
        CropFolder(root, _cfg(pad_frac=0.20), is_train=False)


def test_a_folder_predating_pad_frac_raises_rather_than_assuming_zero(tmp_path):
    """`crops/train` records `margin: 0` and no pad_frac; its PNGs are tight.

    Defaulting to 0 would train on tight crops under a padded config, which does not raise and
    only reports a worse number.
    """
    root, _ = _folder(tmp_path, pad_frac=0.10)
    meta = json.loads((root / "meta.json").read_text())
    del meta["pad_frac"]
    meta["margin"] = 0
    (root / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="predates the crop-padding change"):
        CropFolder(root, _cfg(pad_frac=0.10), is_train=False)


def test_pad_frac_actually_widens_the_field_of_view(tmp_path):
    """The whole point of the change: a padded crop is not the tight crop.

    Guards the failure mode that cost the first attempt - growing the stored PNG while the
    loader re-tightened from alpha, so the model saw byte-identical input.
    """
    root, _ = _folder(tmp_path, pad_frac=0.20)
    tight = CropFolder(root, _cfg(pad_frac=0.0), is_train=False)
    padded = CropFolder(root, _cfg(pad_frac=0.20), is_train=False)
    assert not torch.equal(tight[0][0], padded[0][0])


def test_pad_frac_keeps_the_background_out_of_the_mask(tmp_path):
    """Padding widens the view and leaves the MASK alone - unlike dilate, which grows the mask.

    Under blackout the border is therefore black, and the two knobs stay separable.
    """
    root, _ = _folder(tmp_path, pad_frac=0.20)
    cfg = _cfg(pad_frac=0.20, blackout=True, mask_channel=True)
    x, _aux, _label = CropFolder(root, cfg, is_train=False)[0]
    occupancy = x[3]
    assert occupancy.sum() > 0
    # every RGB pixel outside the mask is the pad value, normalized
    rgb, m = x[:3], occupancy.bool()
    assert torch.allclose(rgb[:, ~m].std(dim=1), torch.zeros(3), atol=1e-5)


def test_closing_at_exactly_the_stored_border_raises(tmp_path):
    """The off-by-one. `border == dilate` was safe while the crop was the tight box.

    Then the dilated mask reached the patch edge but the `close_mask` leak landed in the border,
    outside the crop. `pad_frac` extends the crop region to the patch edge, so the leak is now
    inside it: measured 1053 differing mask pixels on a 204x160 instance at pad_frac=0.10, which
    stores exactly a 16 px border. A `>` guard let that through.
    """
    root = tmp_path / "crops"
    (root / CLASSES[0]).mkdir(parents=True)
    H = W = 512
    frame = (torch.rand(3, H, W) * 255).to(torch.uint8)
    mask = torch.zeros(H, W, dtype=torch.bool)
    mask[150:310, 150:354] = True          # 204x160 -> round(160*0.10) = 16 px short-axis border
    mask[200:240, 200:240] = False         # a hole, so closing has something to do
    box = patch_bbox(mask, 0.10)
    border = unclipped_border(mask_tight_bbox(mask), box, H, W)
    assert border == 16, border
    x0, y0, x1, y1 = box
    save_crop(root / CLASSES[0] / "f000_p000.png", frame[:, y0:y1, x0:x1], mask[y0:y1, x0:x1])
    (root / "meta.json").write_text(json.dumps(
        {"pad_frac": 0.10, "min_border_px": border, "thing_classes": CLASSES}
    ))
    cfg = {"pad_frac": 0.10, "mask_close": True}
    with pytest.raises(ValueError, match="thinnest in this folder"):
        CropFolder(root, _cfg(mask_dilate=16, **cfg), is_train=False)
    # one less still matches the live path exactly, so the guard must not over-reach
    ds = CropFolder(root, _cfg(mask_dilate=15, **cfg), is_train=False)
    live = letterbox_crop(
        frame.float(), mask, ds.cfg.crop_size, box=mask_tight_bbox(mask), pad_frac=0.10,
        dilate=15, close=True, blackout=ds.cfg.blackout,
    )
    expected = to_model_input(live, ds.cfg.pixel_mean, ds.cfg.pixel_std,
                              mask_channel=ds.cfg.mask_channel)
    assert torch.equal(ds[0][0], expected)


def test_identity_holds_for_an_instance_clipped_at_the_frame_edge(tmp_path):
    """The padded box is clipped to the FRAME live and to the STORED PATCH in the folder.

    When the instance sits on the image border those two clips are the same cut, which is the
    only reason the asymmetric-border case stays bit-identical. Every other fixture here keeps
    its mask well inside the frame, so this branch of `pad_box` was never executed.
    """
    root = tmp_path / "crops"
    (root / CLASSES[0]).mkdir(parents=True)
    H = W = 256
    frame = (torch.rand(3, H, W) * 255).to(torch.uint8)
    mask = torch.zeros(H, W, dtype=torch.bool)
    mask[0:60, 0:90] = True                # flush against the top-left corner
    box = patch_bbox(mask, 0.20)
    assert box[0] == 0 and box[1] == 0, box     # clipped on two sides
    assert unclipped_border(mask_tight_bbox(mask), box, H, W) is not None
    x0, y0, x1, y1 = box
    save_crop(root / CLASSES[0] / "f000_p000.png", frame[:, y0:y1, x0:x1], mask[y0:y1, x0:x1])
    (root / "meta.json").write_text(json.dumps(
        {"pad_frac": 0.20, "min_border_px": 12, "thing_classes": CLASSES}
    ))
    for pf in (0.0, 0.10, 0.20):
        cfg = _cfg(pad_frac=pf)
        ds = CropFolder(root, cfg, is_train=False)
        assert torch.equal(ds[0][0], _live(frame, mask, cfg)), f"pad_frac={pf}"


def test_a_retired_margin_key_loads_as_pad_frac_zero(tmp_path):
    """All 53 `runs/*/config.json` carry `margin`, and two tools re-load them to diagnose a run.

    Dropping the field outright would break every one of them; inheriting the new 0.10 default
    would re-diagnose them with a field of view they never trained on. Neither fails loudly, so
    pin both halves.
    """
    import yaml

    from cropcls.config import load_config
    # load_config validates, so these have to be real files
    det, dcfg = tmp_path / "det.pth", tmp_path / "det.yaml"
    det.touch(); dcfg.touch()
    path = tmp_path / "config.json"
    path.write_text(yaml.safe_dump({
        "detector_checkpoint": str(det), "detector_config": str(dcfg),
        "margin": 16, "crop_size": 64,
    }))
    with pytest.warns(UserWarning, match="retired config key"):
        cfg = load_config(path)
    assert cfg.pad_frac == 0.0
    assert not hasattr(cfg, "margin")

    # an explicit pad_frac alongside a stale margin is respected, not overwritten
    path.write_text(yaml.safe_dump({
        "detector_checkpoint": str(det), "detector_config": str(dcfg),
        "margin": 16, "pad_frac": 0.10, "crop_size": 64,
    }))
    with pytest.warns(UserWarning):
        assert load_config(path).pad_frac == 0.10

    # a genuine typo must still raise - that is what the unknown-key check is for
    path.write_text(yaml.safe_dump({
        "detector_checkpoint": str(det), "detector_config": str(dcfg),
        "pad_frakk": 0.10,
    }))
    with pytest.raises(ValueError, match="unknown config keys"):
        load_config(path)
