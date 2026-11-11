"""Per-class length/width statistics of a static crop folder, for the Step 13 prior.

PYTHONPATH=. python tools/length_stats.py --folder crops/setc_train_cp --out length_stats.json

Reads each crop's stored alpha (the undilated predicted mask, whole, at native resolution), so
`scale_feats` of it equals that of the full-frame mask. Edge crops are excluded from the stats.
"""

from __future__ import annotations

import argparse
import csv
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from cropcls.crop_folder import load_crop
from cropcls.crops import scale_feats
from cropcls.length_prior import compute_stats, require_border


def _aux(path: str) -> np.ndarray:
    # load_crop's own decode, so the stats see exactly the mask CropFolder trains on
    return scale_feats(load_crop(Path(path))[1]).numpy()


def folder_stats(folder: Path, workers: int = 16) -> dict:
    meta = json.loads((folder / "meta.json").read_text())
    require_border(meta, str(folder))
    names = list(meta["thing_classes"])
    with open(folder / "crops.csv") as fh:
        rows = [r for r in csv.DictReader(fh) if r["crop"]]
    labels = np.array([names.index(r["class_name"]) for r in rows])
    paths = [str(folder / r["crop"]) for r in rows]
    if workers > 1:
        with Pool(workers) as pool:
            aux = np.stack(pool.map(_aux, paths, chunksize=64))
    else:
        aux = np.stack([_aux(p) for p in paths])
    out = compute_stats(aux, labels, names)
    out["folder"] = str(folder.resolve())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    out = folder_stats(Path(a.folder), a.workers)
    Path(a.out).write_text(json.dumps(out, indent=1))
    for c, n in enumerate(out["class_names"]):
        print(f"{n:<32} n={out['n'][c]:>5}  L {out['mu_length'][c]:7.1f} +- {out['sd_length'][c]:4.1f}"
              f"  W {out['mu_width'][c]:6.1f} +- {out['sd_width'][c]:4.1f}")
    print(f"edge crops excluded: {out['n_edge_excluded']}  -> {a.out}")


if __name__ == "__main__":
    main()
