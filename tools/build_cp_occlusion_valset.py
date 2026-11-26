"""Build a copy-paste validation set with exact per-instance visibility.

Base frames and paste sources are a single-tool set, so every source tool is unoccluded
and the compositor's visibility_fraction (visible / full area) is exact. Edge-cut tools
are excluded from both, since their full area is unknown.

PYTHONPATH=. python tools/build_cp_occlusion_valset.py --src DIR --out DIR --frames 1000
"""
import argparse
import glob
import json
import os
import random

import h5py
import numpy as np
import yaml
from detectron2.config import get_cfg

from maskdino import add_maskdino_config
from maskdino.data.augmentations import CopyPasteCompositor
from maskdino.data.datasets.register_hdf5_instance import _is_truncated

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--det-config", required=True, help="segmenter config.yaml; its INPUT.COPY_PASTE is reused")
ap.add_argument("--frames", type=int, default=1000)
ap.add_argument("--min-paste", type=int, default=2)
ap.add_argument("--max-paste", type=int, default=9)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--cluster-radius", type=float, default=None, help="override CLUSTER_RADIUS_FRACTION; smaller = tighter piles")
ap.add_argument("--stray", type=float, default=None, help="override CLUSTER_STRAY_FRACTION")
a = ap.parse_args()
random.seed(a.seed)
np.random.seed(a.seed)

src = []
for p in sorted(glob.glob(os.path.join(a.src, "*.hdf5"))):
    with h5py.File(p, "r") as f:
        h, w = f["colors"].shape[:2]
        anns = [x for x in json.loads(f["coco_annotations"][()]) if not _is_truncated(x, h, w)]
    if anns:
        src.append({"file_name": p, "height": h, "width": w, "annotations": anns})
print(f"{len(src)} usable single-tool frames")

cfg = get_cfg()
add_maskdino_config(cfg)
cfg.set_new_allowed(True)
cfg.merge_from_file(a.det_config)
cp = cfg.INPUT.COPY_PASTE
comp = CopyPasteCompositor(
    src, min_instances=a.min_paste, max_instances=a.max_paste,
    rotation_degrees=cp.ROTATION_DEGREES, mask_interp=tuple(cp.MASK_INTERP),
    blend_modes=tuple(cp.BLEND_MODES), blur_kernel_range=tuple(cp.BLUR_KERNEL_RANGE),
    core_margin_px=cp.CORE_MARGIN_PX,
    image_format="RGB", cluster_prob=cp.CLUSTER_PROB,
    cluster_radius_fraction=a.cluster_radius if a.cluster_radius is not None else cp.CLUSTER_RADIUS_FRACTION,
    cluster_stray_fraction=a.stray if a.stray is not None else cp.CLUSTER_STRAY_FRACTION,
    instance_jitter=cp.INSTANCE_JITTER, instance_perspective=cp.PERSPECTIVE,
)

os.makedirs(a.out, exist_ok=True)
vis = []
for i in range(a.frames):
    base = src[i % len(src)]
    with h5py.File(base["file_name"], "r") as f:
        image = f["colors"][()]
        classes = f["instrument_classes"][()]
    anns = json.loads(json.dumps(base["annotations"]))
    image, anns = comp(image, anns)
    for x in anns:
        x.setdefault("visibility_fraction", 1.0)
        vis.append(x["visibility_fraction"])
    stem = f"cpocc_{i:05d}"
    with h5py.File(os.path.join(a.out, stem + ".hdf5"), "w") as f:
        f.create_dataset("colors", data=image, compression="gzip", compression_opts=4)
        f.create_dataset("coco_annotations", data=json.dumps(anns).encode())
        f.create_dataset("instrument_classes", data=classes)
        f.attrs["base_frame"] = os.path.basename(base["file_name"])
    if i % 100 == 0:
        print(f"{i}/{a.frames}", flush=True)

with open(os.path.join(a.out, "_config.yaml"), "w") as f:
    yaml.safe_dump({"built_by": "tools/build_cp_occlusion_valset.py", "args": vars(a),
                    "copy_paste": yaml.safe_load(cp.dump())}, f)
v = np.array(vis)
print(f"{len(v)} instances; visibility quantiles 10/25/50/75/90%:",
      np.round(np.quantile(v, [.1, .25, .5, .75, .9]), 3), f"; fully visible {np.mean(v >= 0.999):.2f}")
