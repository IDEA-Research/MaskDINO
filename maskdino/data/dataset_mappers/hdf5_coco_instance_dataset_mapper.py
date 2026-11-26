# ------------------------------------------------------------------------
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
import copy
import logging

import h5py
import numpy as np
import torch
from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.structures import BitMasks
from sgdata import schema

from ..augmentations import CopyPasteCompositor
from .coco_instance_new_baseline_dataset_mapper import build_transform_gen

__all__ = ["Hdf5CocoInstanceDatasetMapper"]


def _tighten_bbox_to_mask(anno):
    """Re-derive the box from the transformed mask.

    detectron2 transforms a box by mapping its four CORNERS and taking their AABB.
    That is exact for the 90-degree rotations and the resize/crop this pipeline used
    to be limited to, but a homography maps a rectangle to a quadrilateral, so the
    corner AABB is only an upper bound - measured at DISTORTION_SCALE 0.05 it runs
    ~4% loose on average and up to ~15% on a long diagonal instrument. Feeding that
    to the box loss while the mask loss sees the true extent trains the two heads
    toward different answers for the same object.

    Mask and box are already consistent on the way in (CopyPasteCompositor derives
    both from the same modal mask), so this just keeps them consistent on the way out.
    A no-op for the transforms where the corner AABB is already exact.
    """
    seg = anno.get("segmentation")
    if not isinstance(seg, np.ndarray):
        return anno  # polygons: apply_polygons already tracks the true outline
    ys, xs = np.nonzero(seg)
    if len(ys) == 0:
        # Warped entirely out of frame; a zero box makes filter_empty_instances
        # drop it rather than leaving a box around nothing.
        anno["bbox"] = [0.0, 0.0, 0.0, 0.0]
    else:
        anno["bbox"] = [
            float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)
        ]
    return anno


class Hdf5CocoInstanceDatasetMapper:
    """
    A callable which takes a dataset dict pointing at a BlenderProc-style .hdf5 frame
    (see sgdata.schema for the `coco_annotations` schema) and maps it
    into the format used by MaskDINO for instance segmentation.

    Unlike COCOInstanceNewBaselineDatasetMapper, the image ("colors") is read directly
    out of the .hdf5 file named by dataset_dict["file_name"] — there is no separate
    image file. Annotations are expected to already be present in
    dataset_dict["annotations"] (populated from the file's `coco_annotations` key by
    register_hdf5_instance.list_hdf5_dicts at dataset-listing time), same as a regular
    COCO-format dataset dict — there is no separate COCO JSON either.
    """

    @configurable
    def __init__(
        self,
        is_train=True,
        *,
        tfm_gens,
        image_format,
        min_visibility=0.0,
        copy_paste_enabled=False,
        copy_paste_kwargs=None,
    ):
        self.tfm_gens = tfm_gens
        logging.getLogger(__name__).info(
            f"[Hdf5CocoInstanceDatasetMapper] Full TransformGens used in training: {self.tfm_gens!s}"
        )

        self.img_format = image_format
        self.is_train = is_train
        self.min_visibility = min_visibility

        # Copy-paste compositing (see set_copy_paste_sources): needs the full list of
        # dataset dicts to sample paste sources from, which isn't known until the
        # dataloader is built (train_net.py), so construction is deferred.
        self._copy_paste_enabled = copy_paste_enabled
        self._copy_paste_kwargs = copy_paste_kwargs or {}
        self._copy_paste = None

    @classmethod
    def from_config(cls, cfg, is_train=True):
        if is_train:
            tfm_gens = build_transform_gen(cfg, is_train)
        else:
            # build_transform_gen only supports train-time LSJ augmentation; at eval
            # time just resize like Detectron2's own default DatasetMapper does.
            tfm_gens = [
                T.ResizeShortestEdge(
                    cfg.INPUT.MIN_SIZE_TEST,
                    cfg.INPUT.MAX_SIZE_TEST,
                    sample_style="choice",
                )
            ]

        ret = {
            "is_train": is_train,
            "tfm_gens": tfm_gens,
            "image_format": cfg.INPUT.FORMAT,
            "min_visibility": cfg.INPUT.MIN_VISIBILITY,
            # Force-disabled at eval time regardless of the flag - only training
            # frames get composited.
            "copy_paste_enabled": is_train and cfg.INPUT.COPY_PASTE.ENABLED,
            "copy_paste_kwargs": {
                "min_instances": cfg.INPUT.COPY_PASTE.MIN_INSTANCES,
                "max_instances": cfg.INPUT.COPY_PASTE.MAX_INSTANCES,
                "rotation_degrees": cfg.INPUT.COPY_PASTE.ROTATION_DEGREES,
                "mask_interp": tuple(cfg.INPUT.COPY_PASTE.MASK_INTERP),
                "blend_modes": tuple(cfg.INPUT.COPY_PASTE.BLEND_MODES),
                "blur_kernel_range": tuple(cfg.INPUT.COPY_PASTE.BLUR_KERNEL_RANGE),
                "core_margin_px": cfg.INPUT.COPY_PASTE.CORE_MARGIN_PX,
                "image_format": cfg.INPUT.FORMAT,
                "cluster_prob": cfg.INPUT.COPY_PASTE.CLUSTER_PROB,
                "cluster_radius_fraction": cfg.INPUT.COPY_PASTE.CLUSTER_RADIUS_FRACTION,
                "cluster_stray_fraction": cfg.INPUT.COPY_PASTE.CLUSTER_STRAY_FRACTION,
                # Per-pasted-instance appearance jitter; a no-op node (the default)
                # is normalized to None inside the compositor, so the hot path is
                # untouched for every config that does not opt in.
                "instance_jitter": cfg.INPUT.COPY_PASTE.INSTANCE_JITTER,
                # Per-instance projective warp - geometry, so kept separate from
                # INSTANCE_JITTER (which is contractually appearance-only).
                "instance_perspective": cfg.INPUT.COPY_PASTE.PERSPECTIVE,
            },
        }
        return ret

    def set_copy_paste_sources(self, dataset_dicts):
        """Wire the full training dataset-dict list in as copy-paste sources. Call
        once, right after the dataset list is resolved in train_net.py's"""
        if not self._copy_paste_enabled:
            return
        self._copy_paste = CopyPasteCompositor(dataset_dicts, **self._copy_paste_kwargs)

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below

        try:
            with h5py.File(dataset_dict["file_name"], "r") as f:
                image = f[schema.COLORS][()]
        except (OSError, KeyError) as exc:
            # Frame evicted from a live pool between listing and now, or a torn
            # read. Returning None makes MapDataset.__getitem__ retry with another
            # sample instead of crashing the loader.
            logging.getLogger(__name__).warning(
                "[Hdf5CocoInstanceDatasetMapper] dropping unreadable frame %s: %s",
                dataset_dict["file_name"],
                exc,
            )
            return None

        assert image.ndim == 3 and image.shape[2] == 3, (
            "expected an (H, W, 3) RGB 'colors' array"
        )
        if self.img_format == "BGR":
            image = image[:, :, ::-1]
        image = np.ascontiguousarray(image)

        if self.is_train and self._copy_paste is not None:
            image, dataset_dict["annotations"] = self._copy_paste(
                image, dataset_dict["annotations"]
            )

        utils.check_image_size(dataset_dict, image)

        image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        image_shape = image.shape[:2]  # h, w

        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            return dataset_dict

        annos = [
            _tighten_bbox_to_mask(
                utils.transform_instance_annotations(obj, transforms, image_shape)
            )
            for obj in dataset_dict.pop("annotations")
            if obj.get("iscrowd", 0) == 0
            and obj.get("visibility_fraction", 1.0) >= self.min_visibility
        ]
        instances = utils.annotations_to_instances(
            annos, image_shape, mask_format="bitmask"
        )
        if not instances.has("gt_masks"):  # image has no (visible) instances
            instances.gt_masks = BitMasks(
                torch.zeros((0, *image_shape), dtype=torch.uint8)
            )
        instances = utils.filter_empty_instances(instances)
        # MaskDINO's prepare_targets expects gt_masks to be a plain (N, H, W) tensor.
        instances.gt_masks = instances.gt_masks.tensor
        dataset_dict["instances"] = instances

        return dataset_dict
