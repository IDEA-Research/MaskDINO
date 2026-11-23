"""Fixed length and width prior on the logits (spec_cropcls.md Step 13).

Per class, a Gaussian likelihood of the instance's measured length L and width W (the long and
short side of its minimum-area rectangle, from `crops.scale_feats`), scaled to peak 1 and mixed
with a floor, added to the logits as `k * (g_L + g_W)`. With k = 1 that is the post-hoc product
`softmax(head) * lik_L * lik_W` up to a per-row constant.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

# robust sd floors: a class whose masks all measure the same would otherwise get sd ~ 0 and
# turn the prior into a hard gate
SD_FLOOR = {"length": 3.0, "width": 1.5}


def lengths_from_aux(aux: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Inverse of scale_feats' encoding: (L, W) in frame px and the edge flag."""
    aux = aux.float()
    return 256.0 * torch.exp2(aux[:, 0]), 32.0 * torch.exp2(aux[:, 1]), aux[:, 4] > 0.5


def floored_loglik(v: torch.Tensor, mu: torch.Tensor, sd: torch.Tensor, floor: float) -> torch.Tensor:
    """(B,) values against (C,) Gaussians -> (B, C) log of the peak-1, floored likelihood."""
    q = -0.5 * ((v[:, None] - mu[None]) / sd[None]) ** 2 - torch.log(sd)[None]
    q = q - q.max(1, keepdim=True).values
    return torch.log((1.0 - floor) * torch.exp(q) + floor)


def prior_term(aux: torch.Tensor, mu: torch.Tensor, sd: torch.Tensor, floor: float) -> torch.Tensor:
    """(B, C) float32 `g_L + g_W`, zero for crops whose mask touches the frame.

    mu, sd are (2, C): row 0 length, row 1 width. Always float32 and outside autocast: in fp16,
    z^2 overflows a few hundred px from every class and `-inf - -inf` is NaN.
    """
    with torch.autocast(device_type=aux.device.type, enabled=False):
        L, W, edge = lengths_from_aux(aux)
        mu, sd = mu.float(), sd.float()
        g = floored_loglik(L, mu[0], sd[0], floor) + floored_loglik(W, mu[1], sd[1], floor)
        return torch.where(edge[:, None], torch.zeros_like(g), g)


def compute_stats(aux: np.ndarray, labels: np.ndarray, class_names: list[str]) -> dict:
    """Per-class median and robust sd (1.4826 * MAD) of L and W over non-edge crops."""
    L = 256.0 * np.exp2(aux[:, 0])
    W = 32.0 * np.exp2(aux[:, 1])
    ok = aux[:, 4] < 0.5
    out = {"class_names": list(class_names), "n": [], "n_edge_excluded": int((~ok).sum())}
    for key, v in (("length", L), ("width", W)):
        mus, sds = [], []
        for c in range(len(class_names)):
            x = v[ok & (labels == c)]
            if len(x) == 0:
                raise ValueError(f"no non-edge crops of class {class_names[c]}: no {key} prior for it")
            m = float(np.median(x))
            mus.append(m)
            sds.append(max(1.4826 * float(np.median(np.abs(x - m))), SD_FLOOR[key]))
        out[f"mu_{key}"], out[f"sd_{key}"] = mus, sds
    out["n"] = [int((ok & (labels == c)).sum()) for c in range(len(class_names))]
    return out


def load_stats(path: str | Path, class_names: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    """(2, C) mu and sd, rows length/width, in the MODEL's class order - or raise.

    The order is checked, not assumed: for set C `sorted()` and `thing_classes` disagree, and a
    permuted prior raises nothing, it just pushes every class toward its neighbour's length.
    """
    d = json.loads(Path(path).read_text())
    if list(d["class_names"]) != list(class_names):
        raise ValueError(
            f"{path} was computed for classes {d['class_names']}, but the model's order is "
            f"{list(class_names)}. Re-run tools/length_stats.py on this model's crop folder."
        )
    mu = torch.tensor([d["mu_length"], d["mu_width"]], dtype=torch.float32)
    sd = torch.tensor([d["sd_length"], d["sd_width"]], dtype=torch.float32)
    return mu, sd


def require_border(meta: dict, where: str) -> None:
    """The static folder's edge flag is exact only if every unclipped side kept >= 1 px."""
    b = meta.get("min_border_px")
    if b is None or b < 1:
        raise ValueError(
            f"{where}: meta.json min_border_px is {b!r}. The edge flag of a stored patch equals "
            "'touches the frame' only when every side not clipped at the frame kept >= 1 px of "
            "border; otherwise a mask that merely fills its patch reads as frame-cut and loses "
            "its length prior. Re-extract with a larger pad_frac."
        )
