# cropcls — invariants and traps

The classifier half of Arm C: a frozen class-agnostic MaskDINO supplies instance masks,
and a CNN **trunk** + global pool + linear head assigns the class to each cropped,
blacked-out instance. See `../spec_cropcls.md`.

Nothing here registers a detectron2 `META_ARCHITECTURE`, config namespace or dataset.
`maskdino/` is **not modified**. The multi-arch config machinery (`build_base_cfg`,
`arch_ns`, `_ARCH_CONFIG_NS`) exists only on branch `41-train-on-mask2former` and is
unmerged, so registering an architecture would have blocked this work on that merge.

## What is reused from `maskdino/`

| what | why it is safe to reuse |
|---|---|
| `Hdf5CocoInstanceDatasetMapper` | `@configurable`, so plain-kwargs construction works |
| `mask_iou_matrix` | exact, box-prefiltered; returns float32 on CPU |
| `HungarianInstanceEvaluator` | `distributed=False` kills all collectives |
| `apply_test_sample_stride` | pure function over `DatasetCatalog` |
| dataset registrations | imported via `import maskdino` |

## Invariants, and the test that enforces each

| invariant | why | test |
|---|---|---|
| background is **exactly** 0 after blackout | the model must not learn a background cue | `test_background_is_exactly_zero` |
| photometric runs **pre**-blackout | `RandomLighting` adds a constant colour shift; post-blackout it lifts the background off zero. `RandomContrast` blends toward the crop mean, which on a ~95%-black crop is near zero | `test_photometric_runs_before_blackout_so_background_is_still_zero` |
| rotations are **lossless** | 0/90/180/270 on a square canvas are pixel permutations: no interpolation, so crop and mask stay aligned | `test_rotations_are_lossless_permutations` |
| aspect ratio preserved, never skewed | proportions are discriminative, and p90 bbox AR is 6.0 on set A | `test_aspect_ratio_preserved_not_skewed` |
| an empty mask is **dropped**, not returned | otherwise an all-black crop trains as a real class | `test_empty_mask_is_rejected_not_nan` |
| the eval cfg carries the **classifier's** class count | `HungarianInstanceEvaluator` sets `_check_classes = NUM_CLASSES > 1`; the detector's cfg says 1, so misclassification would be silently reported as 0.0 | `test_eval_cfg_clone_restores_the_class_count` |
| a zero-GT frame returns empty, never throws | such frames exist in this data (e.g. `012676FD8_01-008.hdf5`) and `(K,0).max(1)` raises | `test_zero_gt_frame_returns_empty_instead_of_throwing` |
| an 8-class checkpoint is **rejected** | nothing would crash: the detector would just emit 8 logits and the class-agnostic premise would break invisibly | `test_multiclass_checkpoint_is_rejected` |
| predictions are upsampled; GT is **never** downsampled | at fill 0.070 forcep03/04 are filigree and would thin away | `test_predictions_at_lower_resolution_are_upsampled_not_gt_downsampled` |
| under `mask_channel`, the 4-channel model **is** the 3-channel model at init | the new first-conv weights are zero, so the arm measures the channel rather than a perturbed trunk | `test_the_4ch_model_is_the_3ch_model_at_init` |
| ...and the channel is still **learnable** | zero weights ignore channel 3 entirely, so the equivalence above passes just as well on a frozen or detached channel | `test_one_step_moves_the_mask_weights_off_zero` |
| channel 3 **is** the co-rotated letterboxed mask, in {0,1} | same reason: equivalence at init is satisfied by a channel of zeros or of noise, so the content needs its own assertion | `test_mask_channel_is_the_letterboxed_mask_not_zeros` |
| all three crop paths emit the **same** channel layout | live, static-folder and `--hungarian` disagreeing does not raise - it just reports a worse number | `test_folder_crop_equals_live_crop` (parametrized over `mask_channel`) |
| under `scale_film`, the FiLM model **is** the plain model at init | the film output layer is zero, so the arm measures scale rather than a perturbed head | `test_film_model_equals_the_plain_model_at_init` |
| `aux` is `scale_feats` of the **undilated full-frame** mask on all three paths | the letterbox discards size, and a dilated or letterboxed mask gives a wrong size that raises nothing | `test_live_batch_aux_is_scale_feats_of_each_frame_mask_once_per_rotation`, `test_export_path_feeds_the_model_scale_feats_of_the_frame_mask`, `test_static_folder_aux_equals_the_full_frame_mask` |
| a FiLM model called without `aux` **raises** | a missed call site would otherwise score a worse number silently | `test_film_model_without_aux_raises` |

## Traps that cost real time

**`apply_test_sample_stride` is not idempotent.** It removes the catalog entry and
re-registers a plain lambda, so a second call strides the already-strided list (4 -> 16)
and you silently evaluate a quarter of the frames you think you do. `export.py` guards
with a module-level `STRIDE_APPLIED` set.

**The mapper drops `annotations` entirely when `is_train=False`.** The assignment step
needs GT on val as much as on train, so the pipeline always builds it with
`is_train=True` and controls augmentation separately via `cfg.augment_frames`. Using
`is_train=False` would have produced an empty val metric with no error.

**`score = objectness * class_prob` couples localization to classifier confidence.**
It is MaskDINO parity (`sigmoid(class_logit) * mask_score`), but an undertrained
classifier emits softmax ~0.35, so `0.9 * 0.35 = 0.32` falls below `SCORE_THRESH 0.5`
and good masks vanish — depressing even the **class-agnostic** metric by 21 points
(0.9758 -> 0.7619, measured). Use `score_mode="detector"` to read out localization alone.

**The single-class checkpoint was trained by branch-41 code.** Its dumped `config.yaml`
carries `MODEL.MASK_FORMER` keys that this branch does not register, so the merge needs
`set_new_allowed(True)`. The keys are inert for building a MaskDINO.

**`timm.create_model(in_chans=4)` is NOT a zero-initialized 4th channel.** It routes the
pretrained weights through `adapt_input_conv`, which tiles the RGB kernels to (R,G,B,R) and
rescales **all four by 3/4** - so the pretrained RGB path is altered and channel 3 starts as a
copy of red. `add_mask_input_channel` does the surgery by hand instead, and `in_chans` was
deleted from `CropClassifier.__init__` so that route is unreachable. `test_timm_in_chans_4_is_not_zero_init`
pins the trap itself.

**The mask channel's parameter cost is per-trunk, not a constant.** It is one input channel
on the first conv, so the count is `out_channels x kh x kw`: **3,136** for `resnet50`
(64x7x7), **288** for `resnet50d` (32x3x3 deep stem) and **1,536** for `convnext_tiny`
(96x4x4). Quoting 3,136 as "the" cost is wrong for two of the three trunks in the series.

**Checkpoints do not cross `mask_channel`, in either direction.** A 3-channel checkpoint loaded
into a 4-channel model raises `size mismatch for trunk.conv1.weight` (or `trunk.conv1.0.weight`
/ `trunk.stem.0.weight`). That is the intended guard, not a bug to paper over - and
`--eval-only` reads the run's own `config.json`, so the normal path is safe automatically. The
hazard is passing `weights=<other run>/model_final.pth` on the command line.

**GPU contention dominates the timing.** The same detector forward is 629 ms/frame on a
95%-contended Quadro RTX 6000 and 125 ms/frame on an idle RTX 2080 Ti — a nominally
slower card. Any timing claim must name the machine and its load.
