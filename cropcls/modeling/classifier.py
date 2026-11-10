"""The classifier: a CNN trunk + global average pool + linear head.

Classification only, no dense output. The segmentation already comes from MaskDINO
and all this stage adds is a class, so there is no expanding path and nothing
U-Net-shaped about it (spec_cropcls.md section 2) - it is an ImageNet image
classifier applied to a cropped, blacked-out instance.

The trunk is a timm backbone so it can start from ImageNet weights, and it is
pluggable: `resnet50` is the default, not the architecture. That pretraining is
also why the crop size is 224.

Nothing in this file touches MaskDINO. Its pixel decoder and transformer decoder are
frozen for the lifetime of the run; the only trainable parameters in the system are
this trunk and this head.
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn

from ..config import CropClsConfig
from ..crops import SCALE_FEAT_DIM

__all__ = [
    "CropClassifier",
    "add_mask_input_channel",
    "build_classifier",
    "first_conv",
    "set_stem_stride",
]


def first_conv(trunk: nn.Module) -> tuple[str, nn.Conv2d]:
    """Locate the trunk's input conv via `pretrained_cfg["first_conv"]`.

    `conv1`, `conv1.0`, `stem.0` for resnet50 / resnet50d / convnext_tiny. Never a per-trunk
    table: that goes stale the first time someone passes a different `trunk=`.
    """
    path = getattr(trunk, "pretrained_cfg", {}).get("first_conv")
    if not isinstance(path, str) or not path:
        raise ValueError(
            f"{type(trunk).__name__} declares no pretrained_cfg['first_conv'], so its input "
            "conv cannot be located"
        )
    conv = trunk.get_submodule(path)
    if not isinstance(conv, nn.Conv2d):
        raise TypeError(f"first_conv {path!r} is {type(conv).__name__}, not nn.Conv2d")
    return path, conv


def set_stem_stride(trunk: nn.Module, stride: int) -> int:
    """Re-stride the trunk's input conv, keeping the kernel and its pretrained weights.

    The point is the EARLY reduction, which `output_stride` cannot reach: dilating stage 4
    moves only the last feature map (7 -> 14 at c224) while conv1/maxpool/stages 1-3 are
    untouched, whereas halving the stem stride doubles every stage. Measured on convnext_tiny
    at 224: stem 56 -> 112 and stages 56/28/14/7 -> 112/56/28/14, which is c448's spatial
    resolution with no upsampling and an identical parameter count.

    Padding moves with it, `ceil((kernel - stride) / 2)`, which is the "same"-padding
    convention: it holds the stem output at `input / stride`. The floor form `(k - s) // 2`
    looks equivalent and is not - it is only correct for an EVEN kernel. Measured at 224:
    convnext_tiny (4x4) agrees either way, but resnet50 (7x7) native is s2/p3 -> 112 while the
    floor form gives p2 -> 111, and resnet50d (3x3) native is s2/p1 -> 112 against p0 -> 111.
    So the floor form silently breaks the `stem_stride == native stride is a no-op` guarantee
    on both resnets, and the checkpoint guard compares effective STRIDES, so it would not catch
    it. 4x4/s2 still lands on p1 -> 112, which is what the convnext arm needs.

    Only `.stride` and `.padding` change - the kernel keeps its shape, so nothing is resampled
    or re-tiled and the ImageNet weights stay bit-identical. Returns the effective stride.
    """
    _, conv = first_conv(trunk)
    if stride <= 0:
        return int(conv.stride[0])
    k = int(conv.kernel_size[0])
    if stride > k:
        raise ValueError(
            f"stem_stride={stride} exceeds the kernel size {k}: the conv would skip pixels "
            "entirely, which is a different operation from re-striding it"
        )
    if stride == int(conv.stride[0]):
        return stride            # a stated no-op must not touch padding either
    pad = (k - stride + 1) // 2  # ceil((k - stride) / 2)
    conv.stride = (stride, stride)
    conv.padding = (pad, pad)
    return stride


def add_mask_input_channel(trunk: nn.Module, n: int = 1) -> nn.Module:
    """Give the trunk's first conv `n` extra, ZERO-initialized input channels.

    NOT `timm.create_model(in_chans=4)`. That routes the pretrained weights through
    `adapt_input_conv` (timm/models/_manipulate.py), which tiles the RGB kernels to
    (R,G,B,R) and rescales all four by 3/4 - so the pretrained RGB path is altered and the
    new channel starts as a copy of red. Zeros instead leave the 4-channel model
    bit-identical to the 3-channel one at step 0, which is what makes the arm a measurement
    of the channel rather than of a perturbed trunk.

    The conv is located via `pretrained_cfg["first_conv"]` - `conv1`, `conv1.0`, `stem.0`
    for resnet50 / resnet50d / convnext_tiny - because a per-trunk table goes stale the
    first time someone passes a different `trunk=` on the command line.
    """
    path, old = first_conv(trunk)

    new = nn.Conv2d(
        old.in_channels + n,
        old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        dilation=old.dilation,
        groups=old.groups,
        bias=old.bias is not None,
        padding_mode=old.padding_mode,
        device=old.weight.device,
        dtype=old.weight.dtype,
    )
    with torch.no_grad():
        new.weight[:, : old.in_channels] = old.weight
        new.weight[:, old.in_channels :] = 0.0
        if old.bias is not None:      # convnext_tiny's stem.0 has one; the resnets do not
            new.bias.copy_(old.bias)
    parent, _, name = path.rpartition(".")
    setattr(trunk.get_submodule(parent) if parent else trunk, name, new)
    return trunk


class CropClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        trunk: str = "resnet50",
        pretrained: bool = True,
        freeze_trunk: bool = False,
        mask_channel: bool = False,
        occluder_channel: bool = False,
        stem_stride: int = 0,
        scale_film: bool = False,
    ):
        super().__init__()
        # num_classes=0 + global_pool="avg" gives pooled features and no head, so the
        # head stays explicit and swappable (linear / mlp capacity is a reported finding,
        # not an implementation detail).
        # No in_chans= here on purpose: it is the only route to timm's adapt_input_conv,
        # which rescales the pretrained RGB kernels. See add_mask_input_channel.
        self.trunk = timm.create_model(
            trunk,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        # BEFORE add_mask_input_channel: that rebuilds the conv copying `stride=old.stride`
        # and `padding=old.padding`, so it inherits these. The reverse order would discard
        # them and nothing would raise.
        # Both recorded: `stem_stride` is what this model runs at, `native_stem_stride` is
        # what the trunk ships with, which is how a checkpoint predating the field is read.
        self.native_stem_stride = int(first_conv(self.trunk)[1].stride[0])
        self.stem_stride = set_stem_stride(self.trunk, stem_stride)

        extra = int(mask_channel) + int(occluder_channel)
        if extra:
            # Before the freeze below, so a frozen trunk freezes the new conv too rather
            # than leaving it the one trainable tensor in a "frozen" run. One rebuild for
            # both channels: two successive rebuilds would copy the first new channel's
            # zeros forward correctly but double the surgery for no reason.
            add_mask_input_channel(self.trunk, extra)
        self.feature_dim = int(self.trunk.num_features)
        self.head = nn.Linear(self.feature_dim, num_classes)
        nn.init.trunc_normal_(self.head.weight, std=0.01)
        nn.init.zeros_(self.head.bias)

        self.film = None
        if scale_film:
            self.film = nn.Sequential(
                nn.Linear(SCALE_FEAT_DIM, 64), nn.GELU(),
                nn.Linear(64, 64), nn.GELU(),
                nn.Linear(64, 2 * self.feature_dim),
            )
            # zero gamma and beta: the arm starts as exactly the plain model
            nn.init.zeros_(self.film[-1].weight)
            nn.init.zeros_(self.film[-1].bias)

        self.freeze_trunk = freeze_trunk
        if freeze_trunk:
            for p in self.trunk.parameters():
                p.requires_grad_(False)

    def train(self, mode: bool = True):
        """Keep a frozen trunk in eval mode so its BatchNorm running stats cannot drift.

        timm backbones use real BatchNorm2d (unlike detectron2's FrozenBN), so a frozen
        trunk left in train mode would still update running_mean/var on every forward -
        silently changing a module whose weights are pinned.
        """
        super().train(mode)
        if self.freeze_trunk:
            self.trunk.eval()
        return self

    def forward(self, x: torch.Tensor, aux: torch.Tensor | None = None) -> torch.Tensor:
        f = self.trunk(x)
        if self.film is not None:
            if aux is None:
                # a call site that forgot aux would otherwise score a worse number silently
                raise ValueError("scale_film model called without aux (crops.scale_feats)")
            gamma, beta = self.film(aux.to(f.dtype)).chunk(2, dim=1)
            f = f * (1 + gamma) + beta
        return self.head(f)

    def param_groups(self, base_lr: float, weight_decay: float):
        """Decay on weights, none on norms and biases."""
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (no_decay if p.ndim <= 1 else decay).append(p)
        return [
            {"params": decay, "lr": base_lr, "weight_decay": weight_decay},
            {"params": no_decay, "lr": base_lr, "weight_decay": 0.0},
        ]

    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_classifier(cfg: CropClsConfig, num_classes: int) -> CropClassifier:
    return CropClassifier(
        trunk=cfg.trunk,
        num_classes=num_classes,
        pretrained=cfg.trunk_pretrained,
        freeze_trunk=cfg.freeze_trunk,
        mask_channel=cfg.mask_channel,
        occluder_channel=cfg.occluder_channel,
        stem_stride=cfg.stem_stride,
        scale_film=cfg.scale_film,
    )
