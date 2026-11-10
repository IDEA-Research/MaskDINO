"""Step 12 of spec_cropcls.md: scale awareness via FiLM.

As with the channel arms, every way this can go wrong is silent: aux taken from the
letterboxed or dilated mask, a crop path that passes a different aux than training, or a FiLM
that perturbs the model at init. These tests pin each of those.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from cropcls import data as data_mod
from cropcls import export as export_mod
from cropcls.config import CropClsConfig
from cropcls.crop_folder import CropFolder, patch_bbox, save_crop
from cropcls.crops import SCALE_FEAT_DIM, mask_tight_bbox, scale_feats
from cropcls.data import CropPipeline, StageTimer
from cropcls.detector import Detections
from cropcls.modeling.classifier import CropClassifier

H = W = 256
CLASSES = ["a_nominal", "a_10", "b_nominal"]


def _cfg(**kw) -> CropClsConfig:
    base = {
        "detector_checkpoint": "/fake/det.pth", "detector_config": "/fake/cfg.yaml",
        "crop_size": 64, "photometric": False, "rotations": [0, 90, 180, 270],
        "crops_per_batch": 2, "pad_frac": 0.10, "num_workers": 0, "mask_channel": True,
        "blackout": False,
    }
    base.update(kw)
    return CropClsConfig(**base)


def _rotated_bar(length=150, width=20, angle=30.0, cx=128, cy=128, gap=False):
    """A rotated rectangle, optionally cut in two by a gap - an occluder crossing it."""
    pts = cv2.boxPoints(((cx, cy), (length, width), angle)).astype(np.int32)
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [pts], 1)
    if gap:
        g = cv2.boxPoints(((cx, cy), (12, 60), angle)).astype(np.int32)
        cv2.fillPoly(m, [g], 0)
    return torch.from_numpy(m.astype(bool))


def _long_short(sf):
    return 256 * 2 ** float(sf[0]), 32 * 2 ** float(sf[1])


# --- the feature measures real size ----------------------------------------------------


def test_axis_aligned_rectangle_gives_its_sides():
    m = torch.zeros((H, W), dtype=torch.bool)
    m[50:70, 30:180] = True                       # 150 x 20
    long_, short = _long_short(scale_feats(m))
    assert abs(long_ - 150) < 1e-3 and abs(short - 20) < 1e-3
    assert abs(float(scale_feats(m)[3]) - 1.0) < 1e-3   # a solid bar fills its rect exactly


def test_fill_separates_a_hollow_shape_from_a_solid_one():
    solid = torch.zeros((H, W), dtype=torch.bool)
    solid[50:90, 30:180] = True
    ring = solid.clone()
    ring[55:85, 35:175] = False
    assert float(scale_feats(ring)[3]) < 0.5 < float(scale_feats(solid)[3])


def test_scale_film_with_frame_augmentation_is_rejected(tmp_path):
    det, dcfg = tmp_path / "det.pth", tmp_path / "cfg.yaml"
    det.touch(); dcfg.touch()
    cfg = _cfg(scale_film=True, augment_frames=True,
               detector_checkpoint=str(det), detector_config=str(dcfg))
    with pytest.raises(ValueError, match="augment_frames"):
        cfg.validate()


def test_rotated_rectangle_keeps_its_length_unlike_the_axis_aligned_box():
    """The reason for minAreaRect: the box's long edge changes with rotation."""
    a, b = _rotated_bar(angle=0.0), _rotated_bar(angle=37.0)
    la, lb = _long_short(scale_feats(a))[0], _long_short(scale_feats(b))[0]
    assert abs(la - lb) <= 2
    x0, y0, x1, y1 = mask_tight_bbox(b)
    assert abs(max(x1 - x0, y1 - y0) - 150) > 10   # the box would have moved


def test_invariant_to_the_rotation_augmentation():
    m = _rotated_bar(angle=23.0)
    ref = scale_feats(m)
    for k in (1, 2, 3):
        assert torch.allclose(scale_feats(torch.rot90(m, k)), ref, atol=1e-5)


def test_occluder_gap_does_not_shorten_the_length():
    whole, cut = _rotated_bar(angle=15.0), _rotated_bar(angle=15.0, gap=True)
    assert abs(_long_short(scale_feats(cut))[0] - _long_short(scale_feats(whole))[0]) <= 2


def test_empty_mask_raises():
    with pytest.raises(ValueError):
        scale_feats(torch.zeros((8, 8), dtype=torch.bool))


# --- the model ------------------------------------------------------------------------


def _models():
    torch.manual_seed(0)
    plain = CropClassifier(num_classes=5, trunk="resnet18", pretrained=False, mask_channel=True)
    film = CropClassifier(num_classes=5, trunk="resnet18", pretrained=False, mask_channel=True,
                          scale_film=True)
    missing, unexpected = film.load_state_dict(plain.state_dict(), strict=False)
    assert not unexpected and all(k.startswith("film.") for k in missing)
    return plain.eval(), film.eval()


def test_film_model_equals_the_plain_model_at_init():
    plain, film = _models()
    x = torch.randn(3, 4, 64, 64)
    aux = torch.randn(3, SCALE_FEAT_DIM)
    with torch.no_grad():
        assert torch.equal(plain(x), film(x, aux))


def test_one_step_moves_the_film_output_layer_off_zero():
    _, film = _models()
    film.train()
    opt = torch.optim.SGD(film.parameters(), lr=0.1)
    x, aux = torch.randn(4, 4, 64, 64), torch.randn(4, SCALE_FEAT_DIM)
    loss = torch.nn.functional.cross_entropy(film(x, aux), torch.tensor([0, 1, 2, 3]))
    loss.backward()
    opt.step()
    assert film.film[-1].weight.abs().sum() > 0


def test_film_model_without_aux_raises():
    _, film = _models()
    with pytest.raises(ValueError, match="without aux"):
        film(torch.randn(1, 4, 64, 64))


def test_plain_model_ignores_aux():
    plain, _ = _models()
    x = torch.randn(2, 4, 64, 64)
    with torch.no_grad():
        assert torch.equal(plain(x), plain(x, torch.randn(2, SCALE_FEAT_DIM)))


# --- every crop path feeds the same aux ------------------------------------------------


def test_static_folder_aux_equals_the_full_frame_mask(tmp_path):
    """The patch is cut with a pad_frac border; its alpha must give the full mask's aux."""
    root = tmp_path / "crops"
    (root / CLASSES[0]).mkdir(parents=True)
    frame = torch.randint(0, 256, (3, H, W), dtype=torch.uint8)
    mask = _rotated_bar(angle=31.0)
    x0, y0, x1, y1 = patch_bbox(mask, 0.10)
    save_crop(root / CLASSES[0] / "f000_p000.png", frame[:, y0:y1, x0:x1], mask[y0:y1, x0:x1])
    (root / "meta.json").write_text(json.dumps(
        {"pad_frac": 0.10, "min_border_px": 10, "thing_classes": CLASSES}))
    ds = CropFolder(root, _cfg(), is_train=False)
    _, aux, _ = ds[0]
    assert torch.allclose(aux, scale_feats(mask), atol=1e-5)


class _Inst(SimpleNamespace):
    def has(self, k):
        return hasattr(self, k)


def _frame_with_two_tools():
    m1, m2 = _rotated_bar(angle=10.0, cx=100), _rotated_bar(length=110, angle=70.0, cx=170)
    image = torch.randint(0, 256, (3, H, W), dtype=torch.uint8)
    inst = _Inst(gt_masks=torch.stack([m1, m2]), gt_classes=torch.tensor([0, 2]))
    det = Detections(masks=torch.stack([m1, m2]), scores=torch.tensor([0.9, 0.8]))
    return image, inst, det


def _stub_pipe(cfg, image, inst, det):
    pipe = CropPipeline.__new__(CropPipeline)
    pipe.cfg, pipe.is_train, pipe.timer, pipe.photometric = cfg, False, StageTimer(), None
    pipe.class_names = CLASSES
    pipe.dicts = [{"image_id": 0}]
    pipe.mapper = lambda d: {"image": image, "instances": inst, "image_id": 0, "file_name": "f"}
    pipe.detector = SimpleNamespace(detect=lambda imgs, amp=False: [det], d2_cfg=None)
    return pipe


def test_live_batch_aux_is_scale_feats_of_each_frame_mask_once_per_rotation():
    cfg = _cfg(mask_dilate=8)            # dilation must NOT leak into aux
    image, inst, det = _frame_with_two_tools()
    b = _stub_pipe(cfg, image, inst, det).build_batch([0])
    assert b.aux.shape == (len(b), SCALE_FEAT_DIM) and len(b) == 2 * 4
    for i, r in enumerate(b.records):
        assert torch.equal(b.aux[i], scale_feats(det.masks[r.pred_index]))


def test_export_path_feeds_the_model_scale_feats_of_the_frame_mask(monkeypatch):
    cfg = _cfg(mask_dilate=8, score_mode="detector")
    image, inst, det = _frame_with_two_tools()
    pipe = _stub_pipe(cfg, image, inst, det)
    seen = {}

    class _Model(torch.nn.Module):
        head = torch.nn.Linear(1, len(CLASSES))

        def forward(self, x, aux=None):
            seen["aux"] = aux.cpu()
            return torch.zeros((len(x), len(CLASSES)))

    class _Eval:
        def __init__(self, *a, **k):
            pass

        def reset(self):
            pass

        def process(self, *a):
            pass

        def evaluate(self):
            return {}

    monkeypatch.setattr(export_mod, "_eval_cfg", lambda *a: None)
    monkeypatch.setattr(export_mod, "apply_val_stride", lambda *a: None)
    monkeypatch.setattr(export_mod, "HungarianInstanceEvaluator", _Eval)
    monkeypatch.setattr(export_mod, "CropPipeline", lambda *a, **k: pipe)
    detector = SimpleNamespace(detect=lambda imgs, amp=False: [det], d2_cfg=None)
    export_mod.evaluate_hungarian(cfg, detector, _Model(), "x", torch.device("cpu"))
    want = torch.stack([scale_feats(m) for m in det.masks])
    assert torch.equal(seen["aux"], want)


def test_crop_batch_has_aux_field():
    assert "aux" in data_mod.CropBatch.__dataclass_fields__
