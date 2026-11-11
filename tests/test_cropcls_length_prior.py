"""Step 13 of spec_cropcls.md: fixed length and width prior on the logits.

Each way this can go wrong is silent: a prior in a permuted class order, an edge flag that
differs between the static folder and the frame, an fp16 NaN, or an arm-A checkpoint that
loads without the prior. These tests pin each of those.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from cropcls.config import CropClsConfig
from cropcls.crop_folder import CropFolder, patch_bbox, save_crop
from cropcls.crops import AUX_DIM, scale_feats
from cropcls.length_prior import compute_stats, floored_loglik, load_stats, prior_term
from cropcls.modeling.classifier import CropClassifier, build_classifier, load_weights

CLASSES = ["a_nominal", "a_10", "b_nominal"]
MU = torch.tensor([[300.0, 320.0, 300.0], [100.0, 100.0, 50.0]])
SD = torch.tensor([[3.0, 4.0, 3.0], [1.5, 1.5, 2.0]])


def _aux(L, W, edge=0.0):
    return torch.tensor([[np.log2(L / 256), np.log2(W / 32), 0.0, 1.0, edge]], dtype=torch.float32)


def _posthoc(v, mu, sd, floor=0.02):
    """film_step12/compare.py posthoc, verbatim apart from the edge line."""
    lik = np.exp(-0.5 * ((v[:, None] - mu) / sd) ** 2) / sd
    lik = lik / lik.max(1, keepdims=True)
    return (1 - floor) * lik + floor


def test_length_term_equals_the_posthoc_likelihood_and_width_adds_its_own():
    v = np.array([305.0, 318.0, 290.0])
    want = np.log(_posthoc(v, MU[0].numpy(), SD[0].numpy()))
    got = floored_loglik(torch.tensor(v), MU[0], SD[0], 0.02).numpy()
    assert np.allclose(got, want, atol=1e-6)
    aux = _aux(305.0, 99.0)
    both = prior_term(aux, MU, SD, 0.02)[0].numpy()
    wl = np.log(_posthoc(np.array([305.0]), MU[0].numpy(), SD[0].numpy()))[0]
    ww = np.log(_posthoc(np.array([99.0]), MU[1].numpy(), SD[1].numpy()))[0]
    assert np.allclose(both, wl + ww, atol=1e-4)


def _model(prior=True, film=False):
    torch.manual_seed(0)
    return CropClassifier(num_classes=3, trunk="resnet18", pretrained=False, mask_channel=True,
                          scale_film=film, length_prior=(MU, SD) if prior else None).eval()


def test_prior_argmax_is_the_posthoc_product_argmax():
    m = _model()
    x = torch.randn(5, 4, 64, 64)
    aux = torch.cat([_aux(L, W) for L, W in [(300, 100), (321, 100), (299, 51), (310, 100), (330, 70)]])
    with torch.no_grad():
        m.use_length_prior = False
        head = m(x, aux)
        m.use_length_prior = True
        out = m(x, aux)
    lik = prior_term(aux, MU, SD, 0.02).exp()
    assert torch.equal(out.argmax(1), (head.softmax(1) * lik).argmax(1))


def test_prior_is_float32_and_finite_with_fp16_inputs_under_fp16_autocast():
    """fp16 aux and stats, inside fp16 autocast: z^2 here is ~1e6, past fp16's 65504, so a
    term computed in half precision would be inf and the row NaN."""
    aux = torch.cat([_aux(5000.0, 2000.0), _aux(2.0, 1.0), _aux(300.0, 100.0)]).half()
    with torch.autocast("cpu", dtype=torch.float16):
        g = prior_term(aux, MU.half(), SD.half(), 0.02)
    assert g.dtype == torch.float32 and torch.isfinite(g).all()
    # far from every class: all classes equally unlikely, the floor caps the penalty
    assert g[0].max() <= 0 and g[0].min() >= 2 * np.log(0.02) - 1e-4


def test_edge_crop_gets_no_prior():
    assert torch.equal(prior_term(_aux(305, 99, edge=1.0), MU, SD, 0.02), torch.zeros(1, 3))


def test_switching_the_prior_off_returns_the_head():
    m, plain = _model(), _model(prior=False)
    plain.load_state_dict({k: v for k, v in m.state_dict().items() if not k.startswith("len_")})
    x, aux = torch.randn(2, 4, 64, 64), torch.cat([_aux(300, 100), _aux(320, 100)])
    m.use_length_prior = False
    with torch.no_grad():
        assert torch.equal(m(x, aux), plain(x, aux))


def test_prior_without_aux_raises():
    with pytest.raises(ValueError, match="without aux"):
        _model()(torch.randn(1, 4, 64, 64))


def test_prior_stats_are_buffers_not_parameters():
    m = _model()
    assert {"len_mu", "len_sd"} <= set(m.state_dict())
    names = {n for n, _ in m.named_parameters()}
    assert not any(n.startswith("len_") for n in names)
    opt_params = {id(p) for g in m.param_groups(1e-3, 0.0) for p in g["params"]}
    assert id(m.len_mu) not in opt_params


# --- the stats file --------------------------------------------------------------------


def _stats_file(tmp_path, names=CLASSES):
    d = {"class_names": names, "mu_length": MU[0].tolist(), "sd_length": SD[0].tolist(),
         "mu_width": MU[1].tolist(), "sd_width": SD[1].tolist()}
    p = tmp_path / "stats.json"
    p.write_text(json.dumps(d))
    return p


def test_stats_in_a_different_class_order_raise(tmp_path):
    p = _stats_file(tmp_path, names=[CLASSES[1], CLASSES[0], CLASSES[2]])
    with pytest.raises(ValueError, match="order"):
        load_stats(p, CLASSES)


def test_compute_stats_excludes_edge_crops_and_floors_sd():
    aux = np.concatenate([_aux(300.0, 100.0).numpy()] * 5 + [_aux(500.0, 100.0, edge=1.0).numpy()]
                         + [_aux(320.0, 100.0).numpy()] * 5 + [_aux(300.0, 50.0).numpy()] * 5)
    labels = np.array([0] * 6 + [1] * 5 + [2] * 5)
    s = compute_stats(aux, labels, CLASSES)
    assert s["n_edge_excluded"] == 1 and s["n"][0] == 5
    assert abs(s["mu_length"][0] - 300.0) < 1e-3          # the 500 px edge crop is not in it
    assert s["sd_length"][0] == 3.0 and s["sd_width"][0] == 1.5


# --- loading checkpoints ----------------------------------------------------------------


def _cfg(tmp_path, **kw):
    base = {"detector_checkpoint": "/fake/det.pth", "detector_config": "/fake/cfg.yaml",
            "crop_size": 64, "mask_channel": True, "trunk": "resnet18", "trunk_pretrained": False,
            "length_prior": True, "length_stats": str(_stats_file(tmp_path))}
    base.update(kw)
    return CropClsConfig(**base)


def _state(model):
    return {"model": model.state_dict(), "stem_stride": model.stem_stride}


def test_plain_checkpoint_loads_into_a_prior_model_with_the_files_stats(tmp_path):
    cfg = _cfg(tmp_path)
    m = build_classifier(cfg, 3, CLASSES)
    load_weights(m, _state(_model(prior=False)), cfg)
    assert torch.equal(m.len_mu, MU) and torch.equal(m.len_sd, SD)


def test_plain_checkpoint_loads_into_a_film_plus_prior_model(tmp_path):
    cfg = _cfg(tmp_path, scale_film=True)
    load_weights(build_classifier(cfg, 3, CLASSES), _state(_model(prior=False)), cfg)


def test_checkpoint_with_a_different_prior_raises(tmp_path):
    cfg = _cfg(tmp_path)
    other = CropClassifier(num_classes=3, trunk="resnet18", pretrained=False, mask_channel=True,
                           length_prior=(MU + 5, SD))
    with pytest.raises(ValueError, match="differs"):
        load_weights(build_classifier(cfg, 3, CLASSES), _state(other), cfg)


def test_any_other_missing_key_raises(tmp_path):
    cfg = _cfg(tmp_path)
    st = _state(_model(prior=False))
    del st["model"]["head.weight"]
    with pytest.raises(RuntimeError, match="mismatch"):
        load_weights(build_classifier(cfg, 3, CLASSES), st, cfg)


def test_build_classifier_without_class_names_raises(tmp_path):
    with pytest.raises(ValueError, match="class names"):
        build_classifier(_cfg(tmp_path), 3)


# --- the edge flag on every path --------------------------------------------------------


def _bar(H=256, W=256, x0=40, x1=200, y0=100, y1=120):
    m = torch.zeros((H, W), dtype=torch.bool)
    m[y0:y1, x0:x1] = True
    return m


def test_edge_flag_marks_a_mask_on_the_frame_border():
    assert float(scale_feats(_bar())[4]) == 0.0
    assert float(scale_feats(_bar(x0=0))[4]) == 1.0
    assert float(scale_feats(_bar(y1=256))[4]) == 1.0
    assert scale_feats(_bar()).shape == (AUX_DIM,)


@pytest.mark.parametrize("at_frame", [False, True])
def test_static_patch_edge_flag_equals_the_frame_masks(tmp_path, at_frame):
    """The stored patch is the padded tight box clipped at the frame: a mask on the patch
    border must mean 'cut by the frame', and an unclipped patch must not read as cut."""
    mask = _bar(x0=0 if at_frame else 40)
    frame = torch.randint(0, 256, (3, 256, 256), dtype=torch.uint8)
    root = tmp_path / "crops"
    (root / CLASSES[0]).mkdir(parents=True)
    x0, y0, x1, y1 = patch_bbox(mask, 0.10)
    save_crop(root / CLASSES[0] / "f_p000.png", frame[:, y0:y1, x0:x1], mask[y0:y1, x0:x1])
    (root / "meta.json").write_text(json.dumps(
        {"pad_frac": 0.10, "min_border_px": 2, "thing_classes": CLASSES}))
    cfg = _cfg(tmp_path, pad_frac=0.10, num_workers=0)
    _, aux, _ = CropFolder(root, cfg, is_train=False)[0]
    assert float(aux[4]) == float(scale_feats(mask)[4]) == float(at_frame)


def test_static_folder_without_min_border_raises_under_the_prior(tmp_path):
    root = tmp_path / "crops"
    (root / CLASSES[0]).mkdir(parents=True)
    mask = _bar()
    x0, y0, x1, y1 = patch_bbox(mask, 0.10)
    save_crop(root / CLASSES[0] / "f_p000.png",
              torch.zeros((3, 256, 256), dtype=torch.uint8)[:, y0:y1, x0:x1], mask[y0:y1, x0:x1])
    (root / "meta.json").write_text(json.dumps({"pad_frac": 0.10, "thing_classes": CLASSES}))
    with pytest.raises(ValueError, match="min_border_px"):
        CropFolder(root, _cfg(tmp_path, pad_frac=0.10), is_train=False)


def test_length_stats_tool_raises_without_min_border(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("length_stats", "tools/length_stats.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    (tmp_path / "meta.json").write_text(json.dumps({"pad_frac": 0.1, "thing_classes": CLASSES,
                                                    "min_border_px": 0}))
    with pytest.raises(ValueError, match="min_border_px"):
        mod.folder_stats(tmp_path, workers=1)


# --- config ------------------------------------------------------------------------------


def test_validate_rejects_frame_augmentation_and_a_missing_stats_file(tmp_path):
    det, dcfg = tmp_path / "det.pth", tmp_path / "cfg.yaml"
    det.touch(); dcfg.touch()
    paths = {"detector_checkpoint": str(det), "detector_config": str(dcfg)}
    with pytest.raises(ValueError, match="augment_frames"):
        _cfg(tmp_path, augment_frames=True, **paths).validate()
    with pytest.raises(ValueError, match="length_stats"):
        _cfg(tmp_path, length_stats=str(tmp_path / "nope.json"), **paths).validate()


def test_checkpoint_with_the_same_prior_loads_onto_another_device(tmp_path):
    """The checkpoint is loaded to CPU while the model may sit on the GPU already."""
    cfg = _cfg(tmp_path)
    m = build_classifier(cfg, 3, CLASSES)
    st = _state(build_classifier(cfg, 3, CLASSES))
    if torch.cuda.is_available():
        m = m.cuda()
    load_weights(m, st, cfg)


def test_checkpoint_with_half_a_prior_raises(tmp_path):
    cfg = _cfg(tmp_path)
    st = _state(build_classifier(cfg, 3, CLASSES))
    del st["model"]["len_sd"]
    with pytest.raises(ValueError, match="differs"):
        load_weights(build_classifier(cfg, 3, CLASSES), st, cfg)


def test_validate_rejects_a_zero_floor(tmp_path):
    det, dcfg = tmp_path / "det.pth", tmp_path / "cfg.yaml"
    det.touch(); dcfg.touch()
    with pytest.raises(ValueError, match="floor"):
        _cfg(tmp_path, length_prior_floor=0.0, detector_checkpoint=str(det),
             detector_config=str(dcfg)).validate()
