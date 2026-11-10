# Specification: Two-Stage Instance Classification (frozen segmenter + crop classifier)

Named `cropcls` throughout: the role, not the architecture. The trunk is a pluggable timm
backbone (`resnet50` by default), so neither "unet" nor "resnet" belongs in the name.

## 1. Context & Goals
* **Problem:** The current MaskDINO Model which does Classification and Segmentation shows problems with distinguishing similar objects, further it is not know how it will handle huge amount of classes (e.g. 1000).
* **Objective:** A separate classifier should now do the classification part, it is to be analyzed if it performs better than MaskDINO on its own.
* **User Story:** MaskDINO "single_class" model provides the segmentation Mask from a given image during inference. The Network then has to analyze the image with segmentation mask to find the classification for the segmented parts.

## 2. Technical Architecture & Data Flow
MaskDINO provides a instance Level Segmentation Mask, it always predicts the same class "surgical_instrument". It should be fed to the Classifier cropped along the bbox and also segmented, such that only the segmented pixel are valid and the background is black. The CLassifier then classifies this image.

### Components impacted:
- New top-level package `cropcls/` (answering "find a suitable location"): a plain Python
  package, sibling to `maskdino/`, which is **not modified**. The classifier network itself
  lives in `cropcls/modeling/classifier.py` as `CropClassifier`. Full layout in section 6.

### Data Models / Schema:
- use this phase 1 model to get the segmentation masks: /home/janick.bilang/dev/MaskDINO/runs/20261007130733_single_class_v2/model_final.pth
- use these images to train: `/home/janick.bilang/training/images_lambdablade-6000/train_phase1_pool_set_a_mm/`
- use these images to validate: `/home/janick.bilang/training/images_lambdablade-6000/val_set_a_mm/`
  (1024 frames each. The old `training/images` symlink is gone;
  `register_hdf5_instance.py` resolves the root through `images_root.py`, which globs
  `~/training/images_*` and **raises** when several match - as they do on `dbe-cia-sl19-01`,
  so `SURGICAL_IMAGES_ROOT` must point at `images_lambdablade-6000` or `import maskdino`
  fails before any dataset is registered. A folder name does not identify its data, so
  `crops/train/meta.json` pins `pool_realpath` and the frame-name digest. The digest no
  longer separates the two roots - their phase-1 pools are byte-identical copies as of
  2026-11-03 - it detects pool replenishment.)

**Classes:** 8 - `forcep01_mm`, `forcep02_mm`, `forcep03_mm`, `forcep04_mm`, `scalpel01_mm`,
`hammer01_mm`, `sharpspoon01_mm`, `scarstick01_mm`. (Not the 1-class `*_singleclass`
registrations - those are the detector's label space, not the classifier's.)

**Dataset names changed.** Registration is now by folder name, so the registered sets are
`train_phase1_pool_set_a_mm` and `val_set_a_mm`. The names `train_seta_mm` / `val_seta_mm`
used throughout this file and in `cropcls-results.md` are the **old** aliases: they name the
same data but are no longer registered, so a config naming them raises
`Dataset 'train_seta_mm' is not registered!`. Read them as historical labels below; the
Step 10 configs already use the new names.

**Crop / classifier input contract:**

- crop size **224 x 224**, letterboxed: uniform scale on the longest axis, zero padding, no skew
- classifier input tensor **`(B, 3, 224, 224)`**, float
- channels **3 (RGB)**. The mask is implicit in the blackout - non-zero pixels are the instrument.
  A 4th explicit mask channel is an ablation, not the default: `mask_channel: true`
  (`requirements_mask_channel.md`) appends the letterboxed mask as channel 3, `0.0`/`1.0`, after
  normalization - the 0-255 RGB statistics have no meaning for an occupancy channel. The trunk's
  first conv then gains one **zero-initialized** input channel, so at step 0 the 4-channel model
  is numerically the 3-channel model and the arm measures the channel rather than a perturbed
  trunk. Checkpoints do not cross the flag in either direction; the first-conv size mismatch is
  the intended guard. Only meaningful with `blackout: false`, where the mask is the sole cue to
  which pixels are the instance - under `blackout: true` channel 3 is near-redundant with the
  already-zeroed background. `blackout: true` nonetheless stays **legal** in code - near-redundant is a
  weak arm, not a broken one - but it is **not an arm in this series**: the user was offered it
  and declined (`requirements_mask_channel.md`). Two combinations are rejected outright in `validate()`
  because they cannot measure anything: `mask_only` makes the RGB channels the silhouette, so
  channel 3 becomes an exact duplicate of channel 0; and `freeze_trunk` pins the new weights at
  the zero they were initialized to, so the channel is a literal no-op for the whole run. Both
  raise rather than warn - the cost of noticing at iteration 4000 is a wasted run.
- normalization: reuse **MaskDINO's own pixel statistics**, which are ImageNet's scaled to 0-255 -
  `PIXEL_MEAN [123.675, 116.28, 103.53]`, `PIXEL_STD [58.395, 57.12, 57.375]`, `FORMAT: RGB`.
  Divide by 255 first if the trunk expects 0-1 inputs.

**Why 224:** ImageNet classifiers are pretrained at 224, so a pretrained trunk operates at the object
scale / receptive-field ratio it was trained for. It also sits at native scale for this data -
median long side is 201 px over 4802 `val_seta_mm` instances, so crops are only mildly resampled
either way, and just 1.8% of instances fall under 16 px on the short axis (at 128 it would be 6.1%,
which is where forcep03/forcep04 discrimination dies).

Two structural facts, not tunable via crop size: pad fraction is ~0.51 (median bbox aspect ratio
2.03) and median mask fill is 0.103, so the typical input tensor is **~95% zeros**. Run the
**mask-only ablation** (binary mask, no RGB) early: if it scores near the RGB model the task is
shape matching and forcep03/04 may be near-indistinguishable by construction - their median area
(1421 vs 1469 px), aspect ratio (1.80 vs 1.69) and fill (0.072 vs 0.070) agree within 6%.

## 3. Core Requirements & Logic
- [ ] **BBOX Cropping:** The segmented instances from MaskDINO have to be cropped to its bounding box, scale it along its largest axis, and pad the image. Do not skew it, only uniform scale is acceptable.
- [ ] **Segmenting of Image:** The segmented instances are used to segment the rgb pixels, everything outside the segmentation mask stays black.
- [ ] **Class-Type** Use really the segmentation of MaskDINO and use the class type from the groundtruth. Assign the Classtype according to IoU so if the segmentation occurs over two classes in the groundtruth take the class which segmentation map has the higher IoU. The IoU with the underlying groundtruth has to be >0.5 at least though. Check second highest overlap to be lower than < 0.2 in order to have a distinct Groundtruth for this segmented mask. If the second highest overlap is over 0.2, drop this sample from classifying. For doing these calculations, upsample the segmap, do not downsample the image to predict!
- [ ] **Edge Cases:** If the classifier is unsure, it should output a uncertainity state. For instance also when the follower class is close to the confidence of the best classification.
- [ ] **Training Image View**Output to me the Images which are used for Training in the classifier, I want to see them with and without segmentation maps and also overlapped segmentation map from the original 
- [ ] **Phase 1 Train** analogous to the MaskDINO training, lets train this Classifier in a first step on Phase 1 Training images. 
- [ ] **Phase 2 Train** this is not part of this features, leave it out
- [ ] **Augmentation Pipeline** Use the same augmentation *content* as MaskDINO phase-1, but
      applied **after** the detector, not before it. The detector is frozen, so augmenting its
      input regularizes nothing - it only perturbs the masks, and the bbox crop + letterbox
      discards the scale and position variation anyway. Running it on clean frames also matches
      its inference condition (where its 97.9% class-agnostic recall was measured) and makes its
      output deterministic per frame, hence cacheable. Crop-level equivalents:
      rotation 0/90/180/270 (lossless on a square crop, and 4x the classifier samples per
      detector forward - this 4x-per-forward emission describes the **live** path only; the
      static-cache path in [spec_static_crops.md](spec_static_crops.md) samples one rotation
      per draw instead, which costs no throughput because the forward is already amortised); `RandomBrightness/Contrast/Saturation/Lighting` at the same ranges as
      `build_transform_gen`, applied pre-blackout; optional mask dilate/erode in place of the mask-shape
      variation frame jitter used to produce incidentally. No flipping - instruments have fixed
      handedness. Controlled by `augment_frames` (default `false`); set it `true` to A/B against
      the original recipe.

## 4. API Endpoints / Interface Definitions
- The Training Data will be also Pool based as in MaskDINO training, adapte the Dataloader accordingly

## 5. Definition of Done (DoD) & Testing

- [x] **Pipeline works correctly.** Measured order, which differs from the original sketch -
      the detector runs on the **clean** frame and augmentation happens after cropping
      (see section 3 and step 5 for why):

      Image -> SingleClassV2 (instance segmentation map) -> assign GT class via IoU
            -> bbox crop + blackout + letterbox -> crop-level augmentation -> Train

      Verified on 20 real frames: 108 assigned, 1 dropped contaminated, 1 dropped low-IoU
      (98.2% assignment rate, consistent with the detector's 2.1% class-agnostic FN).
      Confirmed over the **whole 1024-frame pool** by `tools/extract_crops.py` (2026-10-28,
      `dbe-cia-sl19-01`, GPU 0 idle while 7 of 8 were at 94-100%): **4623 assigned, 26
      dropped low-IoU, 1 dropped contaminated = 99.4%**, 4.51 crops/frame, 751 ms/frame,
      12.8 min, 186 MB.

- [x] **Unit tests** cover bbox cropping, segmentation and class assignment - 139 passing in the
      `test_cropcls_*` set, 305 in the whole suite (measured 2026-11-02; the suite excludes
      `tests/test_fixed_size_crop_seg_padding.py`, which fails to *collect* on `origin/main`
      too - it loads a mapper by file path and upstream gave that mapper a relative import):
      `tests/test_cropcls_crops.py` (square output, aspect ratio preserved 10:1, background
      exactly 0, empty mask rejected not NaN, rotations lossless),
      `tests/test_cropcls_assign.py` (assigned / low-IoU / contaminated, **zero-GT frame**,
      single-GT, predictions upsampled rather than GT downsampled),
      `tests/test_cropcls_config.py`, `tests/test_cropcls_detector.py`
      (`class_embed (1,256)`, frozen, eval-mode, rejects an 8-class checkpoint).

- [x] **Short run: val loss reduces.** 250 iterations, `ims_per_batch 2`, 8,856 crops:
      train loss 2.0820 -> 0.3396 (chance is ln(8) = 2.079), val_loss 0.4791,
      val_top1 0.7711 over 201 crops. Plumbing only - not a result.

- [ ] **Comparison to `20261006204751_phase1_training_seta_v1-1`** (8-class MaskDINO,
      `val_seta_mm`, stride 4, excl-truncated, 1201 GT). Needs a full run.
      - primary, confound-free: **top-1 accuracy on matched masks**, baseline **97.22%**
        (1083/1114). Both sides condition on an instance having been localized, so the two
        detectors' differing recall cannot flatter either.
      - secondary: `classaware_f1`, baseline **0.9256**; report `classagnostic_f1`
        (baseline 0.9521) beside it, since the frontend supplies 1176 matched masks against
        the baseline's 1114 - worth up to +7.75 points of classaware recall on its own.
      - per class, because 30 of the baseline's 31 errors are forcep01-04. The 250-iteration
        smoke run already reproduces that shape: every non-forceps class 0.88-1.00,
        forcep02 0.62, forcep04 0.42, forcep03 0.35.

- [x] **Per-stage timings reported - and yes, the detector forward is too slow.**
      Measured (`ims_per_batch 4`, Quadro RTX 6000):

      | stage | ms/frame | % |
      |---|---|---|
      | detector | 602.0 | 72.0% |
      | augment (frame-level, now removed) | 189.0 | 22.6% |
      | assign | 24.9 | 3.0% |
      | crop | 14.4 | 1.7% |
      | collate + mask_to_cpu | 5.7 | 0.7% |

      Batching does not help: flat at ~630 ms/frame from batch 2 to 12. Within the detector,
      backbone is 23.8 ms and deformable attention is ~740 ms (pixel decoder 211.6, transformer
      decoder 531.2) - the MSDeformAttn CUDA op is not compiled and there is no `nvcc` on this
      machine, so it runs the pure-PyTorch fallback. At 20,000 iterations that is ~18.6 h of
      data generation alone.

- [ ] **The adaptation** (follows from the timing result, implemented in part).
      Superseded in design by [spec_static_crops.md](spec_static_crops.md) - read that
      for what is actually being built; the sub-items below are the original sketch:
      - [x] drop frame-level augmentation so the detector sees clean frames -> its output is
            deterministic per frame (`augment_frames: false`; removes the 189 ms/frame stage)
      - [ ] cache crops keyed by frame filename: one detector forward per frame, ever.
            ~12 min for the static 1024 frames. If the live pool is running, the cache must
            **mirror the pool's FIFO eviction** - delete a frame's crops when the frame is
            evicted - or retired scenes leak back into training and the cache grows ~2.8 GB/day.
      - [ ] cache the crop **before** blackout, thresholding and rotation, so photometric
            augmentation can run pre-blackout and `iou_floor` / `runner_up_ceiling` become
            load-time filters rather than precompute decisions.

## 6. Implementation Checklist (The TODO)
*The step-by-step roadmap for the AI.*

Sequenced so that **everything is verifiable before GPU hours are spent**: steps 1-4 need no
training, step 4 is a human inspection gate, and step 5 answers the timing question in the DoD
before any schedule is chosen.

No new `META_ARCHITECTURE` is registered - the multi-arch config machinery (`build_base_cfg`,
`arch_ns`, `_ARCH_CONFIG_NS`) exists only on branch `41-train-on-mask2former` and is unmerged, so
registering an arch would block this feature on that merge. `maskdino/` is not modified at all.

**Target file layout**

```
cropcls/
  __init__.py
  config.py               dataclass + yaml.safe_load (no yacs, no registry)
  detector.py             frozen single_class_v2 wrapper -> masks + scores
  crops.py                letterbox crop + mask blackout        (step 2)
  assign.py               IoU -> GT class, runner-up guard      (step 3)
  augment.py              crop-level photometric jitter
  viz.py                  the "Training Image View" deliverable (step 4)
  data.py                 LIVE loader: detect -> assign -> crop -> augment (step 5)
  crop_folder.py          STATIC loader: crops/<class>/*.png   (spec_static_crops.md)
  export.py               Hungarian eval against the baseline's own evaluator
  modeling/classifier.py  CNN trunk + global pool + linear head    (step 6)
tools/extract_crops.py  writes the static crop folder, one forward per frame
train_cropcls.py        own training loop, --eval-only, --viz-only
configs/cropcls/phase1_seta.yaml
configs/cropcls/mask4_{r50,r50d,cxt}_c{224,448}.yaml   # Step 10 arms
tests/test_cropcls_{assign,augment,classes,config,crop_folder,crops,detector,eval_cfg,mask_channel,occluder_channel}.py

No `metrics.py`: top-1 lives in `train_cropcls.evaluate`, the timing table in
`data.StageTimer`. No `test_cropcls_timing.py` either - see step 5.
```

- [x] **Step 0 - scaffolding.** Package + config dataclass + `configs/cropcls/phase1_seta.yaml`.
      Config carries: detector checkpoint path, train/val dataset names, `CROP_SIZE: 224`, `NUM_CLASSES: 8`,
      trunk name, LR, iters, IoU floor (0.5) and runner-up ceiling (0.2). The detector path goes
      in the YAML, not in code. **Verify:** config loads, paths assert-exist at startup.

- [x] **Step 1 - frozen detector wrapper** (`detector.py`). Bootstrap via `get_cfg()` +
      `add_deeplab_config` + `add_maskdino_config` + merge the run's own `config.yaml`, then
      `build_model` -> `.eval()` -> `DetectionCheckpointer(model).load(...)` - the pattern at
      `tools/benchmark_inference.py:60-70`. Wrap the forward in `no_grad` + `autocast`, pin to
      `.eval()` permanently. Returns per-frame `(masks, scores)` - the learned `pred_boxes`
      are deliberately not carried, since step 2 crops from each mask's tight bbox instead.
      **Verify:** load the checkpoint, run one `val_seta_mm` frame, assert `class_embed` is `(1, 256)`,
      print mask count and shapes, print the forward time.

- [x] **Step 2 - crops** (`crops.py`). Pure function, no model, no detectron2. bbox crop ->
      zero every pixel outside the mask -> uniform scale on the longest axis -> pad to square.
      No skew. **Verify:** `tests/test_cropcls_crops.py` - output is square and the requested size;
      aspect ratio preserved within a pixel; background is exactly 0; a 40x400 bbox is not stretched;
      an empty mask is rejected rather than producing a NaN or all-black crop.

- [x] **Step 3 - class assignment** (`assign.py`). Reuse
      `maskdino.evaluation.hungarian_instance_evaluation.mask_iou_matrix` (exact, box-prefiltered).
      Upsample the predicted segmap to GT resolution - never downsample the GT. Rule:
      `best >= 0.5` AND `runner_up < 0.2` -> assign `gt_classes[argmax]`; otherwise drop.
      **Verify:** `tests/test_cropcls_assign.py` - clean single-GT mask assigned correctly;
      a mask straddling two GTs at 0.55/0.38 dropped; `best < 0.5` dropped; a frame with **zero GT**
      returns empty instead of throwing (such frames exist in this data, e.g.
      `012676FD8_01-008.hdf5`); correct behaviour when only one GT is present.

- [x] **Step 4 - training image view** (`viz.py`, `--viz-only`). GATE: no training before this is
      inspected. For N sampled frames write a contact sheet per instance: raw crop, crop with
      background blacked, and the predicted mask overlaid on the original full frame. Label each
      tile with assigned class, best IoU, runner-up IoU, and verdict (assigned / dropped-low-IoU /
      dropped-contaminated). Also write the dropped bucket, so the thresholds can be calibrated by
      eye. **Verify:** human review - the dropped pile should look genuinely ambiguous. Adjust 0.5 /
      0.2 here, not later.

- [x] **Step 5 - pool dataloader + timing** (`data.py`). Order is **detect -> assign -> crop ->
      augment**: the detector runs on the clean frame, so its masks and `instances.gt_masks` share
      a coordinate frame trivially and its output is deterministic per frame. Reuse
      `Hdf5CocoInstanceDatasetMapper` with `is_train=True` and `tfm_gens=[]` - `is_train=True` is
      required because the mapper **drops `annotations` entirely** when False, and the assignment
      step needs GT on val as much as on train; `tfm_gens=[]` makes it an identity transform. It
      returns `instances.gt_masks` as an `(N,H,W)` bitmask tensor and `gt_classes`, so no mapper
      change is needed. Index the pool with `d[i]`, not `for r in d`. Instrument every stage.
      **Verify:** the per-stage table printed by `data.StageTimer` at the end of every run -
      measured below. `tests/test_cropcls_timing.py` was named here but never written; the
      table is the evidence, and a unit test cannot produce it.
      (scene gen / augment / detector forward / upsample+IoU / crop / collate). **Report the
      timings before step 7 picks `MAX_ITER`.** If the detector forward dominates, the likely cause
      is MSDeformAttn running its pure-PyTorch fallback (`CUDA_HOME` is empty here, there is no
      `nvcc` on this machine, so the CUDA op cannot be built).

      **Measured, and it is too slow**: 602 ms/frame of a 836 ms/frame pipeline (72%), flat under
      batching from batch 2 to 12, with deformable attention accounting for ~740 ms of the 767 ms
      detector forward (backbone is only 23.8 ms). See section 5 for the full table and the
      adaptation. Dropping frame-level augmentation removes a further 189 ms/frame and makes the
      detector output cacheable, which is what actually resolves it.

      **All augmentation moves to crop level** (`augment_frames: false`). Rotation: emit each crop
      at 0/90/180/270 after cropping. On a square letterboxed crop
      those four are lossless pixel permutations and the axis-aligned bbox is symmetric under them,
      so the result is identical to rotating the frame - but one detector forward then yields 4x the
      classifier samples instead of one rotation's worth. Since the detector forward is the
      expensive stage, this is where the throughput is. (Live path only - the static cache in
      [spec_static_crops.md](spec_static_crops.md) samples one rotation per draw, because
      bundling all four in a batch is what correlates the BatchNorm statistics.) Keep the frame-level photometric augs
      (`RandomBrightness/Contrast/Saturation/Lighting`) as they are - they survive the crop
      untouched. Do **not** add flipping: instruments have fixed handedness, so a mirrored scalpel
      is an invalid shape (see the comment in `build_transform_gen`). Note the four rotations of one
      instance are correlated, so effective diversity grows well under 4x - this buys throughput,
      not a substitute for more frames.

- [x] **Step 6 - trunk + head** (`modeling/classifier.py`). A plain image-classification CNN -
      **no decoder of any kind**, since the masks come from MaskDINO. Trunk -> global average
      pool -> `nn.Linear(C, 8)`. Trunk
      pluggable (`detectron2_resnet` / `timm`); `timm 1.0.28` is present. Stock PyTorch ops only.
      If the trunk is frozen and uses real `BatchNorm2d`, pin it to `.eval()`.
      **Verify:** `tests/test_cropcls_mask_channel.py::test_the_classifier_accepts_3_channels`
      (forward shape); parameter count printed by `train_cropcls.py`.
      (The `mask_channel` conv surgery also lives in this file, but its obligations are Step
      10's - this step stays ticked on the 3-channel contract it was signed off against.)

- [x] **Step 7 - training loop** (`train_cropcls.py`). Own loop: AMP, checkpoint save/load, LR via
      `maskdino/solver`, detectron2 `EventStorage` + writers, `--eval-only`. Cross-entropy over 8
      classes, argmax at inference (no reject state, per spec). Log train loss, **val loss**, and
      **top-1 accuracy on matched masks**. **Verify:** 20 iterations at batch 2 on 1 GPU - loss
      decreases, checkpoint is small, memory sane.

- [ ] **Step 8 - short run + comparison.** Train, then report against
      `runs/20261006204751_phase1_training_seta_v1-1` (8-class, `val_seta_mm`, stride 4,
      excl-truncated, 1201 GT):
      - primary, confound-free: **top-1 accuracy on matched masks**, baseline **97.22%** (1083/1114)
      - secondary: `classaware_f1`, baseline **0.9256** - report `classagnostic_f1` beside it, since
        the frontend supplies 1176 matched masks vs the baseline's 1114 and that is worth up to
        +7.75 points of classaware recall on its own (see `cropcls-baseline-comparison.md`)
      - per-class breakdown, because 30 of the baseline's 31 errors are forcep01-04
      **Verify:** DoD items 3 and 4 satisfied.

- [ ] **Step 9 - full run** at the `MAX_ITER` chosen from the step-5 timings.

- [ ] **Step 10 - mask-channel arm** (`requirements_mask_channel.md`). Six configs: `resnet50` /
      `resnet50d` / `convnext_tiny` x crop 224 / 448, all `blackout: false mask_channel: true`,
      all `mask_dilate: 0` (under `blackout: false` dilation no longer touches RGB at all, so it
      becomes a channel-3-only knob and a second variable; pinned here, not swept).
      The first conv is rebuilt with one extra zero input channel, located via
      `pretrained_cfg["first_conv"]`. Never `timm.create_model(in_chans=4)`: that routes the
      weights through `adapt_input_conv`, which tiles RGB to (R,G,B,R) and rescales all four by
      3/4 - altering the pretrained path and seeding channel 3 with a copy of red.

      Each arm needs a **`blackout: false mask_channel: false` control at the same trunk, crop
      size and LR**, launched from the same config with `mask_channel=false` and **its own
      `output_dir=`** - both arms of a pair otherwise inherit one `output_dir` and the run folder
      does not say which is which (that exact mix-up already cost two renames, see
      `cropcls-results.md` section 5). No existing run is a control: the `20261031*` arms are all
      `blackout: true`, and the one `blackout: false` measurement that exists (r5, 0.7982, **-7.8
      points** - worst of twelve) predates the static crop folder and used bbox jitter, which is
      no longer a config field. That -7.8 is the level all six arms start from.

      **Verify:** `tests/test_cropcls_mask_channel.py` - the 4-channel model's output is
      bit-identical to the 3-channel model's at init on all three trunks, AND one optimizer step
      moves the channel-3 weights off zero (the equivalence alone is satisfied by a channel of
      garbage, since zero weights ignore any input). `tests/test_cropcls_crops.py` -
      `test_mask_channel_is_the_letterboxed_mask_not_zeros`: channel 3 takes only `{0.0, 1.0}`,
      equals `rotate_k90(crop_mask, deg)` for each of 0/90/180/270, and is non-degenerate.
      `tests/test_cropcls_crop_folder.py` - folder-vs-live bit identity holds at
      `blackout: false` and with the channel on, since that is the only setting these arms use.
      Results land in `cropcls-results.md` as one table of twelve rows, each naming its run id,
      trunk, crop size and channel count. Seed variance on `classaware_f1` is 0.82 points
      (`cropcls-results.md` section 5), so a gap under ~1.6 points is not rankable - report it as
      unresolved, not as a result.

- [ ] **Step 11 - occluder-channel arm** (`requirements_occlusion_channel.md`). One extra input
      channel holding the union of the OTHER predicted instance masks in the same frame, so the
      trunk can tell a silhouette that is cut by another tool from one that ends in free space.

      **Why, measured.** Instability under label-preserving perturbation (`tools/augment_probe.py`,
      1024 val frames, 24 in-distribution variants) is driven by occlusion, not by mask quality.
      For the fragile forcep03 instances against the stable ones: `overlap_frac` 5.36x,
      `censored_frac` 2.99x, instances-per-frame 1.33x, while `best_iou` (0.97x), `mask_long`
      (0.96x) and `det_score` (0.98x) are flat. The model is blind to this today: median `msp` is
      0.99998 in **every** `censored_frac` band while the error rate moves 1.1% -> 6.7%
      (`tools/occlusion_stats.py`). And it does not leak class on the pairs that matter -
      forcep03-vs-forcep04 from `censored_frac` alone is AUROC 0.4973, forcep01/02 0.4893 - so
      the channel is a censoring cue here and not a class shortcut. Re-check that leak on any new
      scene set before trusting a gain.

      **Design.** `occluder_channel: bool = False`. Base is `blackout: false mask_channel: true`,
      so the layout is ch0-2 RGB, ch3 own instance, **ch4 occluder union**, and `in_chans` becomes
      `3 + mask_channel + occluder_channel`. Both extra channels are 0.0/1.0 appended after
      normalization, for the reason `to_model_input` already records: the 0-255 RGB statistics
      mean nothing for an occupancy channel. Rotation stays a single `rotate_k90` over all
      channels, so the occluder cannot drift out of alignment with the pixels it describes.

      **The crop region must be shared, not just the append.** `letterbox_crop` currently takes
      one `mask` and returns `(crop, crop_mask)`; it gains an optional `others` mask and returns
      `(crop, crop_mask, crop_others)`, with `crop_others` carried through the **same** `box`,
      `pad_frac`, uniform scale and centering as the RGB. Appending a separately-cropped raster
      would misalign it with the pixels it describes, and that failure is silent - `to_model_input`
      already records why ("the training path and the scoring path disagreeing about the channel
      layout does not raise - it just reports a worse number"). `to_model_input` takes the tuple
      rather than loose tensors for the same reason it already does: so a crop cannot be paired
      with some other instance's masks.

      **Four call sites change together**, and missing any one is silent:
      `cropcls/data.py` (live), `cropcls/crop_folder.py` (static), **`cropcls/export.py`** - the
      Hungarian scoring path that produces the `classaware_f1` this step is ranked by, whose
      `to_model_input` call at `export.py:162` passes only `mask_channel=` today - and
      `cropcls/modeling/classifier.py` (the first-conv rebuild). `cropcls/README.md` records
      "all three crop paths emit the same channel layout" as a tested invariant; the test must
      extend to the new channel or the invariant is no longer enforced.

      **Not a third value in ch3.** `{background 0, mine 1, occluder 0.5}` was the user's first
      suggestion and is rejected: a conv is linear in channel values, so every filter would
      respond to an occluder at exactly half its response to the instance, asserting a false
      ordinal relation between three categorical labels. Worse, after the trunk's 32x
      downsampling a half-dense region of the OWN mask averages to the same 0.5, making the
      encoding ambiguous under the very pooling it must survive. Two binary channels have no
      such collision.

      **`add_mask_input_channel` generalizes to n channels**, keeping the zero-init: at iteration
      0 the 5-channel model must be numerically the 4-channel one, or the arm measures a
      perturbed trunk rather than the channel. Same `pretrained_cfg["first_conv"]` lookup, never
      `timm.create_model(in_chans=5)`.

      **`validate()` gains two guards**, both raising rather than warning, as the existing
      `mask_channel` pair does (the cost of noticing at iteration 4000 is the whole run):
      `occluder_channel` with `freeze_trunk` - the new input weights start at zero and freezing
      pins them there, so the channel is a literal no-op; and `occluder_channel` **without**
      `mask_channel` - under `in_chans = 3 + mask_channel + occluder_channel` that silently
      yields a 4-channel model whose channel 3 is the occluder union with the instance's own
      mask never fed at all, which is not the arm anyone asked for and which the layout
      "ch0-2 RGB, ch3 own instance, ch4 occluder union" does not describe.

      **Static folder needs re-extraction and a version guard.** The stored PNG is RGB + alpha
      and the occluder union is not recoverable from it, so `crops/` must be re-extracted with a
      second single-channel raster per crop. `meta.json` gains a field recording that the folder
      carries occluders, and a folder without it must **raise** - a missing raster silently read
      as zeros turns this arm into its own control and the run reports a worse number with no
      error (the hazard `to_model_input` already warns about for channel layout). The new folder
      must be extracted at `pad_frac >= 0.10`, the value both arms train at, since `CropFolder`
      treats the extraction `pad_frac` as a ceiling and raises below it.

      **Pin `pad_frac`.** An occluder lying over the instance is inside the crop regardless, but
      an adjacent one reaches it only through the padded rim, so the channel's usefulness is a
      function of `pad_frac` and the two are entangled variables otherwise.

      **Control: a fresh run, NOT `runs/20261103122513_mask4_cxt_c448`.** That run resolves to
      `pad_frac: 0.0` off `crops/train` (its `config.json` carries `margin: 0` and no `pad_frac`,
      which `load_config` maps to 0.0), while `configs/cropcls/mask4_cxt_c448.yaml` now trains at
      `pad_frac: 0.10` off `crops/train_pad20`. Using it would move `pad_frac` AND the crop folder
      alongside the channel - three variables, and the 0.9474 it scored is not a baseline for this
      config. Launch both arms from the same config, `occluder_channel=true/false`, each with
      **its own `output_dir=`** - the Step 10 discipline, for the reason recorded there (one
      shared `output_dir` already cost two renames).

      **Out of scope for Step 11:** the soft-target loss. The user chose channel-alone first.
      Record the expectation up front so a null result is not read as a surprise: under hard-label
      cross-entropy the network is told its evidence is censored while still being required to
      emit a one-hot answer, so it gains one more feature to be confident with rather than a
      reason to hedge. If Step 11 comes back inside seed noise, that is the predicted outcome and
      the follow-up is the loss, not a bigger channel.

      **Rank by `classaware_f1`** from `HungarianInstanceEvaluator` with `score_mode=detector`, at
      the run's own `val_stride: 4` (the 8-class baseline protocol, 1201 GT). Seed variance is
      0.82 points, so a gap below ~1.6 points is not rankable at one seed per arm. Crop top-1 is
      a monitor only.

      **Verify** - `tests/test_cropcls_occluder_channel.py` (new),
      `tests/test_cropcls_crops.py`, `tests/test_cropcls_crop_folder.py`: the 5-channel model is
      bit-identical to the 4-channel one at init and one
      optimizer step moves the ch4 weights off zero; ch4 takes only {0.0, 1.0} and equals
      `rotate_k90(occluder_mask, deg)` for each of 0/90/180/270; ch4 is zero for an instance
      alone in its frame and non-zero for one with a neighbour overlapping its crop;
      folder-vs-live bit identity holds with the channel on; a `crops/` folder lacking the
      occluder raster raises rather than yielding zeros.

- [x] **Step 12 - scale awareness via FiLM** (`requirements_scale.md`). Give the classifier
      back the instance's real size, which the letterbox throws away, by modulating the pooled
      trunk features with a small network fed per-instance geometry.

      **Why, measured** (set C, 2026-11-09/10, `/data/janick.bilang/runs/occ_analysis`). The
      set-C classes inside a family differ only in length (nominal, +10 ... +50 mm), and
      `letterbox_crop` scales every crop to the same long edge. The long side of the predicted
      mask's minimum-area rectangle grows ~20 px per +10 mm with a 4-9 px spread inside a class;
      length alone picks the right level 90-98% of the time on single tools. A post-hoc product
      of the frozen classifier with a length likelihood raised exact top-1 0.539 -> 0.610 on
      `val_set_c_mm-1.0x` and 0.826 -> 0.894 on `val_set_c_single_mm-1.0x`, with a learned
      per-detection weight 0.64 / 0.92. This step tests whether learning it end to end does the
      same or better. The camera is fixed (user, 2026-11-10), so pixel size is a stable cue on
      these sets; `cropcls-scale-feature-notes.md` section 4's shortcut warning still applies
      to any other camera setup.

      **Design - FiLM on the pooled features** (`cropcls-scale-feature-notes.md` section 5,
      FiLM variant). `f = trunk(x)` (pooled, `feature_dim`), `gamma, beta = film(aux)`,
      `f = f * (1 + gamma) + beta`, then the unchanged head. `film` is `Linear(4,64)-GELU-
      Linear(64,64)-GELU-Linear(64, 2*feature_dim)` with the **last layer zero-initialized**, so
      at iteration 0 the model is numerically the `scale_film: false` model and the arm
      measures the feature, not a perturbed head. Config flag `scale_film: bool = False`.

      **Features** (`scale_feats(mask)` in `cropcls/crops.py`), from the instance's own
      **undilated** mask in frame pixels, before any letterbox: long and short side of
      `cv2.minAreaRect` over the mask pixels, `sqrt(area)`, and `fill = area / (long*short)`,
      as `log2(long/256), log2(short/32), log2(sqrt(area)/64), fill`. Not the axis-aligned box
      the design note proposed: its long edge shrinks with rotation, and measured on set C it
      separates the length levels at 38-52% where the rectangle reaches 90-98%. The rectangle is
      invariant to the 90-degree rotation augmentation, so `aux` does not change with `deg`.
      Edge-cut instances get the same features (their length is short); the trunk sees the cut.

      **Plumbing - aux is computed on every path, always.** Not only when `scale_film` is on:
      that keeps the three crop paths from diverging by config, which is exactly the silent
      failure `to_model_input` warns about. `CropBatch` gains `aux (B,4)`; the static
      `CropFolder.__getitem__` returns `(x, aux, label)`; `cropcls/export.py` (the path that
      produces `classaware_f1`) and `train_cropcls.py` (train loop and `evaluate`) pass
      `model(x, aux)`. `CropClassifier.forward(x, aux=None)` **raises** when `scale_film` is on
      and `aux` is None, so a missed call site fails loudly instead of scoring a worse number.
      Tools that call the model (`confidence_metrics`, `class_report`, `loss_breakdown`,
      `viz_misclassified`, `augment_probe`) pass `aux` too.

      **Resolution assumption.** `aux` is in frame pixels, and the static folder stores patches
      at native resolution while the val mapper feeds the detector native 1024x1024 frames, so
      train and score agree. A resize in either mapper would shift every feature; not guarded
      beyond this note.

      **Static folder needs no re-extraction.** Alpha is the undilated predicted mask and the
      stored patch contains all of it, so the rectangle of the patch mask equals the rectangle
      of the full-frame mask. Do NOT read size off the PNG dimensions (design note section 6
      point 6).

      **Arms.** Base `configs/cropcls/setc_cp_cxt_c448.yaml`. Control: same file,
      `scale_film=false`; FiLM arm: `scale_film=true`; each with its own `output_dir=`, seeds 0
      and 1 (seed variance 0.82 points). The existing `runs/20261109083120_setc_cp_cxt_c448` is
      NOT reused as control: it was trained before this code change.

      **Rank by `classaware_f1`** from `HungarianInstanceEvaluator`, `score_mode=detector`, on
      `val_set_c_mm-1.0x` (the config's own val). Also reported, not ranked:
      `val_set_c_single_mm-1.0x` and `val_set_c_cpocc_mm-1.0x`, and exact / family / +-10 mm
      top-1 on matched detections.

      **Verify** - `tests/test_cropcls_scale_film.py` (new) and `tests/test_cropcls_crop_folder.py`:
      `scale_feats` of a rasterized W x H rectangle gives long/short within 1 px of max/min(W,H)
      and is identical for the mask and each `rot90` of it; the `scale_film` model's logits equal
      the plain model's at init with the shared weights copied over, and one optimizer step moves
      the FiLM output layer off zero; `forward(x)` without `aux` raises when `scale_film` is on;
      the static folder's `aux` equals `scale_feats` of the full-frame mask the patch was cut
      from, and the live path (`CropPipeline.build_batch`) and the scoring path
      (`cropcls/export.py`) feed the model exactly `scale_feats` of the same undilated
      full-frame mask - checked by capturing the `aux` each path passes to a stub model, so an
      `aux` taken from the letterboxed or dilated mask fails; `CropBatch.aux` has one row per
      crop.

      **Decision rule, fixed before the runs.** FiLM wins if its mean `classaware_f1` over seeds
      0 and 1 on `val_set_c_mm-1.0x` beats the control's mean by more than 1.6 points (2x seed
      variance), and loses if it is more than 1.6 points worse; in between is a tie. To compare
      against the post-hoc fusion, which has no `classaware_f1`, report exact top-1 on matched
      detections from `pred_length_pass.py`-style output for both arms, against the post-hoc
      0.610 (fixed product) and 0.641 (learned weight) on piles and 0.894 / 0.924 on single.

      **Out of scope for Step 12:** detector score or other reliability inputs to `film` (not
      available in the static folder without re-extraction); additive-logit fusion; a
      scale-preserving crop.

      **Outcome, 2026-11-10: no gain. By the pre-registered rule FiLM loses; with the measured
      variance it is a tie.** The verdict word is left to the user. Runs in
      `/data/janick.bilang/runs/film_step12`, compared with `compare.py` there; results in
      `cropcls-results.md` section 11. Piles `classaware_f1`, seeds 0/1: control 52.78 / 58.66
      (mean 55.72), FiLM 48.40 / 56.20 (mean 52.30), so -3.4, past the rule's 1.6. But the 0.82
      seed variance behind that threshold was wrong for this config: the control seeds differ by
      5.9. Seeds 2 and 3 were therefore added for both arms. Over four seeds: control 54.60
      (sd 2.8), FiLM 52.68 (sd 3.3), a gap of -1.9, inside one sd. Paired by seed, FiLM is lower
      at all four (-4.4, -2.5, -0.1, -0.8; sign test p ~0.125). Mechanism: a test-time shift of
      the long-side feature by +-0.1 log2 (~1.3 length steps) changes 0 of 1127 single-set
      argmaxes for `film_s1` and 1 for `filmwarm_s0`. These are the only two runs probed, and
      only on the single set. The `film` output layer ends near zero (abs max 0.013-0.034). Train
      loss reaches 0.002-0.005 on appearance alone, so length gets little gradient; that cause is
      inferred, not measured. The post-hoc length product adds +6.6 points of exact top-1 on
      piles (8-run mean, every run positive, min +4.6). **Added beyond the Arms item:** one
      diagnostic warm-start arm, n=1 (`filmwarm_s0`: FiLM from
      `20261109083120_setc_cp_cxt_c448`, 2000 it). It needs the `strict=False` film.*-only load in
      `train_cropcls.py`. **Added beyond Resolution assumption:** `validate()` rejects
      `scale_film` with `augment_frames`, because the frame augmentation would change pixel size.
      **Features as built:** each minAreaRect side has +1 px added (pixel extent, not centres),
      and fill is clipped to 1. Verify: `tests/test_cropcls_scale_film.py` and
      `tests/test_cropcls_crop_folder.py` pass (65 tests; full suite 412).

**Out of scope (explicit):** phase 2 / set B, COCO AP as a headline metric (argmax cannot hedge
the way `instance_inference`'s `topk(100)` does), any modification to `maskdino/` or `train_net.py`. The owner made a change to register_hdf5_instance.py, in order to register the dataset correctly, this change is approved. (That change has since been superseded by `maskdino/data/datasets/images_root.py`, which resolves the images root from `$SURGICAL_IMAGES_ROOT` or a single `~/training/images_*` match; same approval, same carve-out.)

Out of scope for **Step 10** specifically (`requirements_mask_channel.md`): adapting a checkpoint
across `mask_channel` in either direction; a `blackout: true` 4-channel arm; and feeding an
**undilated** mask in channel 3 while the blackout uses the dilated one. That last one is the
item a later reader will try to "fix": channel 3 is whatever `letterbox_crop` returns, which is
the **post-dilation** mask, and decoupling them would need `letterbox_crop` to return a second
mask. The six configs pin `mask_dilate: 0`, where the two coincide and the question does not
arise.
