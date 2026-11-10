# Crop classifier — results

Run 2026-10-10/11 on `dbe-cia-sl23-01` (4x RTX 2080 Ti, GPUs 1–3 idle). All numbers
from `val_seta_mm`; the baseline is `20261006204751_phase1_training_seta_v1-1` (8-class
MaskDINO, stride 4, excl-truncated, 1201 GT).

Six runs. Measured **endpoint** seed variance (r2 vs r4, identical but for the seed) is
**0.09 points** on crop top-1 and **0.82 points** on `classaware_f1`. An earlier 2.3-point
figure in this file came from iteration 500 and was wrong for final numbers.

## 1. What ran

Common to all six: 4000 iterations, `ims_per_batch 4`, `base_lr 3e-4`, crop 224, frozen
`single_class_v2` detector run on **unaugmented** frames, augmentation applied at crop level.

| run | trunk | augmentation | input | seed |
|---|---|---|---|---|
| r1 | resnet50 | rotations only | RGB, blacked | 0 |
| r2 | resnet50 | rotations + photometric + bbox jitter | RGB, blacked | 0 |
| r3 | resnet50 | full | **silhouette only** | 0 |
| r4 | resnet50 | full | RGB, blacked | **1** |
| r5 | resnet50 | full | RGB, **background kept** | 0 |
| r6 | **resnet18** | full | RGB, blacked | 0 |
| dil3 | resnet50 | full | RGB, **mask dilated 3 px** | 0 |
| dil8 | resnet50 | full | RGB, **mask dilated 8 px** | 0 |
| dil16 | resnet50 | full | RGB, **mask dilated 16 px** | 0 |

## 2. Headline

**All numbers are `classaware_f1` from the baseline's own `HungarianInstanceEvaluator`
(`score_mode=detector`). Crop top-1 is reported only as a training monitor — it has
misled in both directions (section 5) and should not be used to rank runs.**

| run | mask handling | classagnostic_f1 | **classaware_f1** | vs base | (crop top-1) |
|---|---|---|---|---|---|
| **8-class MaskDINO** | — | 0.9521 | **0.9256** | — | 0.9722 |
| **dil16** | dilate 16 px | 0.9812 | **0.9175** | **−0.0081** | 0.9195 |
| dil3 | dilate 3 px | 0.9812 | 0.9077 | −0.0179 | 0.9115 |
| dil8 | dilate 8 px | 0.9812 | 0.9003 | −0.0253 | 0.9044 |
| r1 | tight, rotations only | 0.9812 | 0.8824 | −0.0432 | 0.9000 |
| close16 | closing 16 px | 0.9812 | 0.8799 | −0.0457 | 0.9124 |
| r2 | tight, full aug | 0.9812 | 0.8758 | −0.0498 | 0.8947 |
| close3 | closing 3 px | 0.9812 | 0.8734 | −0.0522 | 0.8956 |
| r4 | tight, seed 1 | 0.9812 | 0.8676 | −0.0580 | 0.8956 |
| r6 | tight, resnet18 | 0.9812 | 0.8538 | −0.0718 | 0.8805 |
| r3 | tight, silhouette only | 0.9812 | 0.8235 | −0.1021 | 0.8965 |
| **dil32** | dilate 32 px | 0.9812 | **0.8105** | −0.1151 | 0.9195 |
| r5 | no blackout | 0.9812 | 0.7982 | −0.1274 | 0.8876 |

**Arm C reaches parity with end-to-end MaskDINO while localizing 2.9 points better:**

```
classagnostic_f1   0.9812  vs  0.9521    +2.91
classaware_f1      0.9175  vs  0.9256    -0.81
```

`classagnostic_f1` is 0.9812 in **all twelve** runs — every run shares the same frozen
detector, and the exported masks are the detector's own (dilation affects only the crop fed
to the classifier). Both required and a clean consistency check.

### The decisive variable: padding the predicted mask

```
dilate  0  (tight)      0.8734 - 0.8824      <- 8 configurations, nothing escapes this band
dilate  3               0.9077   +2.5
dilate  8               0.9003   +1.8
dilate 16               0.9175   +3.5        <- best
dilate 32               0.8105   -6.5        <- cliff
no blackout (box)       0.7982   -7.8        <- worst
closing 3 / 16          0.8734 / 0.8799      <- no benefit at either radius
```

Three things this says:

1. **Dilation helps in a window, roughly 3–16 px**, and is worth more than trunk size,
   augmentation recipe, input modality, seed and background handling *combined*.
2. **Closing never helps.** It fills gaps and merges limbs without moving the silhouette,
   and gains nothing at either radius. So the benefit is *growth* — recovering real
   instrument pixels the detector clipped — not repairing a fragmented mask.
3. **There is a cliff past ~16 px.** At 32 px, adjacent instruments in a pile dilate into
   one another and a crop contains its neighbour.

**Upstream implication.** The frontend's class-agnostic recall is 0.98, but recall counts an
instance found at `IoU >= 0.5` — it is blind to mask *quality*. On the forceps the masks
arrive at IoU 0.55–0.83, under-covering the instrument. Dilation is a workaround; the
principled fix is better masks from `single_class_v2`.

**Caveat: every dilation point is n=1** and the 3/8/16 ordering is non-monotonic. A seed
repeat at 16 px is running.

## 3. The finding: forcep03/forcep04 collapse into one class

| run | config | forcep03 | forcep04 | **pair sum** |
|---|---|---|---|---|
| r1 | rotations only | 0.6142 | 0.7926 | 1.407 |
| r2 | full aug, seed 0 | 0.9213 | 0.4148 | 1.336 |
| r3 | silhouette only | 0.9134 | 0.5259 | 1.439 |
| r4 | full aug, seed 1 | 0.9370 | 0.3704 | 1.307 |
| r5 | no blackout | 0.5984 | 0.8222 | 1.421 |
| r6 | resnet18 | 0.9291 | 0.3185 | 1.248 |
| **baseline** | — | 0.9322 | 0.9237 | **1.856** |

**Six configurations — varying trunk size, augmentation, input modality and background
handling — all produce a pair sum of 1.25–1.44. None approaches 1.856.** Whichever member
wins, the pair is worth ~0.45 less than MaskDINO extracts from it.

**Padding the predicted mask is the one thing that moves it.** dil3/dil8/dil16 reach
1.563 / 1.542 / 1.567 — clear of the entire six-run range, closing ~38% of the gap to the
baseline. The crops explain why: the predicted masks arrive **fragmented**, IoU 0.55–0.83,
with shafts rendered as dashed segments and holes through the middle
(`runs/forceps_crops.png`). The discriminating detail between two ring-handled forceps is
at the jaws — the thin end a mask drops first. That is the real,
config-independent deficit, and it accounts for essentially the whole gap: every non-forceps
class lands within 0.2–2.3 points of the baseline.

The model can classify forcep03 at 0.93, matching MaskDINO — it simply cannot do both at
once. The capability is there; the discriminating information is not reaching it.

**Which member wins is NOT seed noise.** Both seeds of the full-augmentation config favour
forcep03 (0.9213 / 0.9370) with only 1.6 points of wobble. The 31-point swing between r1 and
r2/r4 tracks the *configuration*, not the seed. (n=1 per config, so association, not cause.)

Consistent with their geometry: median area 1421 vs 1469 px, aspect ratio 1.80 vs 1.69, fill
0.072 vs 0.070 — all within 6%.

## 4. What mattered, ranked (`classaware_f1`, against r2 tight = 0.8758)

> **The plain-dilation rows below are confounded — see section 8.** Until 2026-11-05
> `cropcls/export.py` passed no `box=`, so the scoring path cropped the *dilated* tight box
> while training cropped the undilated one: wider by `2 * mask_dilate` px per axis. Measured
> over the 4623 crops in `crops/train`, the mean long-axis ratio is 1.04x at dilate 4, 1.08x at
> 8, 1.16x at 16 and 1.33x at 32. The model was therefore scored on a wider framing than it
> trained on, by an amount that grows with the radius, so the larger radii carried the bigger
> handicap. `dilate 32 = -6.5` is the row to distrust; `dilate 16 = +4.2` was achieved *despite*
> the mismatch and is if anything understated. The baseline, both closing rows and the
> blackout/RGB rows are unaffected: the baseline and those ablations run at `mask_dilate: 0`,
> and `close_mask` does not grow the outer silhouette, so its tight box is unchanged.

| change | effect | reading |
|---|---|---|
| **dilate the mask 16 px** | **+4.2** | largest effect found, by a wide margin |
| dilate 3 px | +3.2 | your first guess; already most of the benefit |
| dilate 8 px | +2.5 | — |
| rotations-only vs full aug (r1) | +0.7 | inside seed noise |
| closing 16 px | +0.4 | not a real gain |
| closing 3 px | −0.2 | nothing |
| seed (r2 vs r4) | ±0.8 | — |
| resnet50 → resnet18 | −2.2 | capacity is not the binding constraint |
| remove RGB | −5.2 | texture is load-bearing despite ~95%-zero crops |
| **dilate 32 px** | **−6.5** | over-dilation merges neighbours in piles |
| remove blackout | −7.8 | a rectangle of real background is the worst input |

**Overfitting is present but is not the binding constraint.** r1 ends at train loss 0.0095
against val loss 0.3492, val loss rising from iteration 1500, val accuracy flat from 2500.
Yet r2, which barely memorizes, lands in the same place, and reducing capacity (r6) makes
things worse. More regularization alone would not have closed the gap — padding the mask did.

## 5. Methodology notes, including what I got wrong

**Crop top-1 is unusable for ranking runs — it misleads in BOTH directions.** It scores only
crops that passed the assignment rule (`IoU >= 0.5` AND `runner_up < 0.2`), i.e. clean,
isolated instances; `classaware_f1` scores everything, including crowded and occluded cases,
and counts false positives. Three times it *understated* an effect ~10x (mask-only, no
blackout). Once it *flattered* one badly: dil32 is tied-best on crop top-1 at 0.9195 and
second-worst on `classaware_f1` at 0.8105 — because 32 px of padding only hurts in crowded
scenes, which crop top-1 filters out by construction. close16 likewise looked like a gain
(0.9124) and was not (0.8799). I ranked runs by it repeatedly and had to retract each time.

**Crop top-1 and `classaware_f1` are not interchangeable.** r1 and r3 differ by 0.35 points on
crop top-1 but **5.9 points** on `classaware_f1`. Crop top-1 only scores crops that passed the
assignment rule (`IoU >= 0.5` and `runner_up < 0.2`) — clean, isolated instances. The Hungarian
protocol scores everything, including crowded and occluded instances, and counts false
positives. An occluded *silhouette* is far more ambiguous than an occluded *textured* crop, so
the filtered metric flatters mask-only by excluding exactly the cases where it fails.
**Report `classaware_f1`; treat crop top-1 as a training monitor only.**

**Seed variance at convergence is small.** r2 vs r4 differ only in seed: 0.8947 vs 0.8956 on
crop top-1 (**0.09 points**) and 0.8758 vs 0.8676 on `classaware_f1` (**0.82 points**). The
gap was 2.3 points at iteration 500 and shrank monotonically — an early-training figure is
not a variance estimate for final numbers, and I quoted one as such for several hours.

**Per-class forceps numbers ARE reproducible across seeds** (1.6 points on forcep03, 4.4 on
forcep04) but swing 31+ points across configurations. I initially attributed the swing to
"initialization and data order"; the seed repeat disproved that.

**Two bugs worth keeping fixed.** (a) `--eval-only` shared the training run's output dir and
overwrote its `config.json` with the eval's defaults, destroying the provenance of all three
round-1 runs. (b) The eval then rebuilt the model from the base YAML rather than the run's
own config, so evaluating r6 built a resnet50 and died on a size mismatch against resnet18
weights — and `hungarian.sh` piped stderr through a grep filter, so the traceback matched
nothing and the log came out empty. Silence looked like success.

**The score formula couples localization to classifier confidence.**
`score = objectness * class_prob` is MaskDINO parity, but an undertrained classifier emits
softmax ~0.35, pushing good masks below `SCORE_THRESH 0.5` and depressing even the
*class-agnostic* metric by 21 points. At convergence both score modes agree to four decimals,
so this only bites on early checkpoints. `score_mode="detector"` gives the clean localization
readout.

**GPU contention dominated the first timing estimates.** The same detector forward is
629 ms/frame on a 95%-contended Quadro RTX 6000 and 125 ms/frame on an idle RTX 2080 Ti. My
original "18.6 hours, caching is a prerequisite" conclusion was mostly an artifact of the
shared card; the real figure is ~5.4 h uncached, and caching is an optimization.

**convnext_tiny beats resnet50 by 5.68 points at crop 448.** On `dbe-cia-sl19-01`,
2026-10-31. The run now at `runs/20261031121447_cxt_c448_dil0_lr1e4` was originally created as
`..._r50_c448_dil0_lr1e4` despite being configured `trunk: convnext_tiny`; the real control was
re-run as `..._REAL_r50_...`. **Both directories were renamed on 2026-10-31 to match their
configs** (`cxt` / `r50`, `REAL_` dropped) and their stored `output_dir` patched to match, so
the names below are the current ones. Both configs are otherwise identical (`base_lr 1e-4`,
crop 448, `dilate_px 0`, `freeze_trunk false`, 4000 iter, seed 0,
`train_seta_mm` / `val_seta_mm`):

| run dir | `trunk` | `classaware_f1` | `classagnostic_f1` |
|---|---|---|---|
| `20261031121447_cxt_c448_dil0_lr1e4` | `convnext_tiny` | 0.9391 | 0.9808 |
| `20261031140925_r50_c448_dil0_lr1e4` | `resnet50` | 0.8823 | 0.9808 |

Both `score_mode=detector`. The identical `classagnostic_f1` confirms the same frozen detector
and the same masks, so the whole gap is the trunk. n=1: this sits outside the six-run round
above and has not been seed-repeated.

**Which chain drove which run.** The sweeps were run as chained background commands that wrote
`TRAINED` / `SCORED` / `ALLDONE` progress markers to `runs/*.out`, while each step's real stdout
went to its own `runs/*.log`. Both sets were transient and were cleaned up on 2026-10-31; the
table below is transcribed from them and is now the only record of the grouping:

| chain | runs it drove |
|---|---|
| `sweep_dil0` / `dil8` / `dil32` / `dil64` | the four `20261030122636_c448_dil*` |
| `stem_r50_224` / `stem_r50d_224` / `stem_r50d_448` | the three `20261031115041_*` |
| `ctrl_chain` (the only one with an `ALLDONE`) | scored the convnext run above, then trained the r50 control |

`c448_dil4` and `c448_dil16` predate the sweep batch and had no marker file.

## 6. What to try next

1. **Seed-repeat the winning radius.** Every dilation point is n=1 and 3/8/16 is
   non-monotonic. One repeat at the best radius settles whether 16 is genuinely better than
   3 or they are the same within noise.
2. **Round 4, running:** `close3` / `close16` (morphological closing — repairs the mask
   without moving the silhouette, verified bbox-identical) against `dil32` (does the curve
   turn over?). close-vs-dilate at matched radius separates *repair* from *growth*, which
   decides whether the fix belongs in the mask pipeline (nearly free) or is a crop-design
   parameter to tune.
3. **Better masks at source.** If repair is the mechanism, the real fix is upstream: the
   frontend's masks are fragmented at IoU 0.55–0.83 on this pair even though its
   class-agnostic recall is 0.98. Recall counts an instance as found at IoU >= 0.5; it says
   nothing about mask *quality*, and quality is what the classifier needs.
4. **Look at `runs/forceps_crops.png`.** forcep03 and forcep04 are both ring-handled forceps
   with identical topology. Whether they are separable in principle, by someone who knows
   the instruments, bounds everything above.

## 7. Needs a decision

1. **Regenerate the scene pool at 5–10 instruments?** Your generator, and it would evict the
   current 1024 frames. I would now say **don't, yet**: six configurations produce the same
   forceps collapse, so this is an information problem and more frames of an indistinguishable
   pair will not fix it. Resolve the pair question first.
2. **Snapshot the current 1024 frames** to a static directory before any regeneration.
3. **Nothing is committed** — all `cropcls/` work is untracked on branch 42.

## 8. Crop padding (`pad_frac`), 2026-11-05

`pad_frac` grows the crop region by a fraction of the mask's tight box on each side, per axis.
Unlike `mask_dilate` it widens the **field of view** and leaves the mask alone, so under
`blackout: false` the border is real background — the pixels the detector's mask clipped.
Ground truth for the ask: `requirements_crop_padding.md`. Spec: `spec_static_crops.md` section 3.

### 8.1 The folder

`crops/train_pad20`, extracted at `pad_frac: 0.20`, which is a **ceiling**: the loader crops to
its own `cfg.pad_frac` within it, so one folder serves every fraction at or below 0.20,
including the tight control at 0.0. Both measured on `dbe-cia-sl19-01`:

| folder | `pad_frac` | size | crops | frames | time |
|---|---|---|---|---|---|
| `crops/train` (old, tight) | — (`margin: 0`) | 186 MB | 4623 | 1024 | 769 s, 751 ms/frame, load unrecorded |
| `crops/train_pad20` | 0.20 | **312 MB** | 4623 | 1024 | 1417 s, **1384 ms/frame, GPUs 5-6 at ~100% from another job** |

Same `frame_digest` (`0d90fc4eb1bf`) and the same 4623 crops, as expected from a frozen
detector and an unchanged assignment rule. 312 MB against 268 MB projected from 1.44x the
pixels — PNG compresses a mostly-background border better than pixel count implies, so scaling
bytes by area ran 15% high. `min_border_px` is 1; no crop has a 0 px border.

**`crops/train` is no longer loadable** — it predates `pad_frac` and `CropFolder` raises rather
than defaulting to 0 and training tight under a padded config. The `pad_frac: 0.0` control
therefore comes off `crops/train_pad20` too. Verified byte-identical first: 16 crops sampled
across classes, `crops/train`'s PNGs equal `crops/train_pad20` re-tightened to its own alpha,
and the filename sets match — which is what makes the two pre-existing tight runs valid
controls instead of needing a fresh pair.

### 8.2 Verdict: 0.10 is nothing; 0.05 on resnet50 is borderline and unresolved

`classaware_f1`, `score_mode=detector`, c224, `blackout: false`, `mask_channel: true`,
4000 iters, all off `crops/train_pad20`. The `pad_frac: 0.0` rows are controls; the two
convnext and two resnet50 controls at seeds 0-1 are pre-existing `mask4_*` runs, not re-run.

| trunk | `pad_frac` | seeds | runs | mean | vs 0.0 | arm spread |
|---|---|---|---|---|---|---|
| resnet50 | 0.0 | 0,1,2,3 | 0.9140 / 0.9182 / 0.9115 / 0.9224 | 0.9165 | — | **1.09 pt** |
| resnet50 | 0.05 | 0,1,2,3 | 0.9257 / 0.9215 / 0.9282 / 0.9281 | **0.9259** | **+0.94 pt** | 0.67 pt |
| resnet50 | 0.10 | 0,1 | 0.9115 / 0.9182 | 0.9149 | −0.17 pt | 0.67 pt |
| convnext_tiny | 0.0 | 0,1 | 0.9249 / 0.9240 | 0.9245 | — | 0.08 pt |
| convnext_tiny | 0.05 | 0,1 | 0.9249 / 0.9341 | 0.9295 | +0.50 pt | 0.92 pt |
| convnext_tiny | 0.10 | 0,1 | 0.9299 / 0.9124 | 0.9211 | −0.33 pt | 1.75 pt |

**0.10 buys nothing on either trunk. 0.05 on resnet50 is borderline: +0.94 pt at p = 0.057,
4 seeds against 4.** Everything else overlaps its control.

**This number moved twice, and the history matters more than the endpoint.** At 2-vs-2 the
control was 0.9140/0.9182 and the arm 0.9257/0.9215 — complete separation, +0.75 pt, recorded
here as "suggestive". Control seeds 2 and 3 then came in at 0.9115 and **0.9224**, widening the
control from 0.42 pt to 1.09 pt and putting 0.9224 above the arm's 0.9215; that was recorded as
a retraction, "the effect was a narrow control, not the padding". Arm seeds 2 and 3 then scored
0.9282 and 0.9281, lifting the mean to 0.9259 and the gap to +0.94 pt, exact two-sided
permutation **p = 0.0571** on 4 vs 4.

So the retraction was too strong. The honest reading is **borderline and unresolved**: short of
the 0.05 threshold, one overlapping pair (control max 0.9224 against arm min 0.9215), but a
consistent direction across all four seeds and the strongest candidate anywhere in this
investigation. Two more seeds per arm would settle it either way; nothing smaller will.
The lesson is the one section 5 already records — at 2 seeds per arm, neither a separation nor
an overlap is evidence.

**n=2 cannot estimate its own spread, and this is the demonstration.** The convnext controls
look reassuringly tight at 0.08 pt, but convnext `pad_frac=0.10` spans 1.75 pt on the same two
seeds. A 0.08 pt spread is not evidence of low variance; it is one draw. Treat anything under
~1.1 pt at c224 as noise, which is wider than the 0.82 pt this file records in section 5.

**Both trunks agree on the direction at 0.10**: −0.17 pt and −0.33 pt, consistent with the
resolution trade — the silhouette drops from 100% to 83% of the canvas edge — but inside noise
either way.

Best single number in the table is convnext `pad_frac=0.05` seed 1 at 0.9341. It is one seed,
and its sibling scored 0.9249, exactly the control. Do not quote it.

`classagnostic_f1` is 0.9808 on all 14 runs, identical to four decimals. The detector is frozen
and `score_mode=detector` keeps scores classifier-independent, so localization cannot move and
only class assignment can. It is the check that the eval isolates classification.

Crop top-1 ranked the resnet50 arms in a different order than `classaware_f1` did, as usual.

### 8.3 Per class: one candidate, not significant, and not the one that matters

The evaluator emits no per-class F1 or precision — `misclassified_per_class` is keyed by **GT**
class, so it cannot give precision. These numbers come from
`inference_detector/instance_matching_confusion.csv` (GT rows, predicted columns, plus a
false-negative column and false-positive row), which does: `TP = M[c][c]`,
`FN = rowsum - TP`, `FP = colsum - TP`.

Pooled across both trunks: 6 control runs against 8 padded (0.05 and 0.10 together), exact
two-sided permutation test on the difference of means.

| class | control | padded | delta | control seed spread | p |
|---|---|---|---|---|---|
| scarstick01_mm | 0.9452 | 0.9614 | **+1.62 pt** | 2.71 pt | **0.0147** |
| hammer01_mm | 0.9964 | 0.9908 | −0.56 pt | 0.92 pt | 0.0586 |
| forcep04_mm | 0.8269 | 0.8339 | +0.70 pt | 4.76 pt | 0.49 |
| sharpspoon01_mm | 0.9753 | 0.9795 | +0.42 pt | 1.80 pt | 0.34 |
| forcep03_mm | 0.8151 | 0.8190 | +0.39 pt | 5.83 pt | 0.66 |
| forcep01_mm | 0.9045 | 0.9078 | +0.33 pt | 4.13 pt | 0.73 |
| scalpel01_mm | 0.9658 | 0.9638 | −0.20 pt | 1.29 pt | 0.64 |
| forcep02_mm | 0.9119 | 0.9115 | −0.04 pt | 3.99 pt | 0.97 |

**Per-class noise dwarfs aggregate noise.** The control's own seed spread is 5.83 pt on
forcep03 and 4.76 on forcep04, against 1.09 pt on aggregate `classaware_f1`. Any per-class
claim under ~4 pt on the forceps is unreadable at these seed counts.

**scarstick01_mm is the only candidate, and it does not survive correction.** p = 0.0147 is
nominal; with 8 classes tested the Bonferroni threshold is 0.00625, and the chance of seeing
something this strong somewhere among 8 classes is about 11%. What makes it more than a lucky
p-value is that it is consistent in sign in all four trunk x fraction cells (+0.48 to +2.13 pt)
and 6 of 8 padded runs beat the best of 6 control runs. What makes it less than a result is
that nothing else moves and the aggregate is null.

**Its mechanism is specific and legible.** The gain is entirely precision — recall was already
0.9975, i.e. essentially every scarstick is found with or without padding:

| | control | padded | delta |
|---|---|---|---|
| precision | 0.8984 | 0.9283 | **+2.99 pt** |
| recall | 0.9975 | 0.9971 | −0.03 pt |
| things wrongly called scarstick | 14.8 | 10.1 | **−4.7** |

Broken down, the predictions that stop being called scarstick are `scalpel01` (5.8 -> 4.2),
`sharpspoon01` (5.8 -> 4.6) and spurious detections matching no GT (1.8 -> 0.2). So padding
helps the classifier stop calling *other elongated instruments* scarstick. That fits the
geometry: scarstick is by far the most elongated class (mean aspect **5.33**, next is scalpel at
3.15 and sharpspoon at 2.78), and surrounding context is what separates a bare thin shaft from
a blade or a spoon. Note the obvious alternative story is **not** supported — if padding worked
by recovering clipped instrument, scarstick's silhouette should sit on its tight-box edge more
than other classes, and it sits there *least* (0.025 of box perimeter, against 0.046-0.055 for
the forceps).

**The errors move, they do not disappear.** `false_positives_total` (20), `false_negatives_total`
(26) and `num_mask_matched_total` (1175) are identical to two decimals across all 14 runs, as
they must be with a frozen detector and `score_mode=detector`. `misclassified_total` goes
73.83 -> 70.12, a 5% relative reduction worth +0.32 pt of class-aware recall — and a permutation
test on it gives **p = 0.4376** (control 67-83, padded 56-83). Not a result.

**Padding does not touch the actual problem.** Section 3's finding is the forcep03/forcep04
collapse, and those two sit at 0.815 and 0.827 F1 — far the worst classes — with deltas of
+0.39 and +0.70 pt against seed spreads of 5.83 and 4.76. Whatever padding does, it does not
help the pair that is costing the most.

### 8.4 Pending and not done

- resnet50 `pad_frac=0.05` seeds 2 and 3 are training, to make that arm 4-vs-4 against its
  control. They can only narrow the conclusion above, not reverse the retraction: the control
  already reaches 0.9224.
- `pad_frac` untried at c448, where the resolution trade should be cheaper and where convnext
  scores highest (0.9474/0.9499 tight).
- The dilation sweep in section 4 should be re-run now that `export.py` frames crops the way
  training does, since `pad_frac` is a cleaner way to widen the view than dilation was.
- Nothing is committed.

## 9. Stem stride, 2026-11-05

`stem_stride` re-strides the trunk's FIRST conv, keeping the kernel and its pretrained weights.
Ground truth: `requirements_stem_stride.md`. Spec: `spec_stem_stride.md`.

Why this and not `output_stride`: the user's objection, verified. `output_stride=16` changes
only `layer4` (7 -> 14 at c224) while `conv1`, `maxpool` and stages 1-3 are untouched, whereas
c448 doubles every one of them. So `output_stride` cannot test the early-detail hypothesis at
all. On convnext_tiny `stem_stride: 2` turns the 4x4/stride-4 patchify into overlapping
4x4/stride-2: stages 56/28/14/7 -> 112/56/28/14, which is c448's spatial resolution with **no
upsampling** and an **identical parameter count**.

### 9.1 It works, and it is the first thing that moves forcep03/04

convnext_tiny, c224, `pad_frac: 0.0`, `blackout: false`, `mask_channel: true`, 4000 iters.
`classaware_f1`, `score_mode=detector`. Control is the pre-existing tight pair, not re-run.

| arm | n | runs | mean | vs control |
|---|---|---|---|---|
| stride 4 (control) | 2 | 0.9240 / 0.9249 | 0.9245 | — |
| **stride 2** | 2 | **0.9382 / 0.9407** | **0.9395** | **+1.50 pt** |
| stride 4, c448 (reference) | 2 | 0.9474 / 0.9499 | 0.9487 | +2.42 pt |

**On the p-value: do not quote it.** The exact two-sided permutation test returns 0.3333, which
is the **floor** at 2-vs-2 (`2 / C(4,2)`), reached whenever the arms separate completely. It is
the test having no resolution at this seed count, not evidence of absence. What carries the
claim instead: both arm runs sit above both control runs, the gap is 1.50 pt against a control
spread of **0.08 pt** and an arm spread of 0.25 pt, and it clears the ~1.1 pt c224 noise floor
from section 8.2. Two more seeds per arm would make it testable; the spec's own DoD asks for
that and it is **not yet met**.

### 9.2 Per class, the mechanism's own prediction

The hypothesis was that the early 4x reduction destroys thin structure, so the **thinnest**
classes should gain most. Half-width is the median distance transform of the silhouette in
feature-map cells after the stem at c224 (section 8.3 method). Four classes separate completely
(every arm run above every control run); the other four overlap.

| class | half-width | control | stride 2 | delta | separated | c448 for comparison |
|---|---|---|---|---|---|---|
| **forcep04_mm** | 0.41c | 0.8296 | 0.8694 | **+3.98 pt** | yes | +5.51 pt (72%) |
| **forcep03_mm** | 0.40c | 0.8148 | 0.8534 | **+3.86 pt** | yes | +5.68 pt (68%) |
| scarstick01_mm | 0.53c | 0.9472 | 0.9756 | +2.84 pt | yes | +4.33 pt (66%) |
| scalpel01_mm | 1.06c | 0.9660 | 0.9848 | +1.87 pt | yes | +2.03 pt (92%) |
| forcep02_mm | 0.95c | 0.9236 | 0.9259 | +0.23 pt | no | +1.07 pt (21%) |
| sharpspoon01_mm | 0.32c | 0.9831 | 0.9848 | +0.17 pt | no | +0.68 pt (25%) |
| forcep01_mm | 0.79c | 0.9199 | 0.9196 | −0.02 pt | no | +0.99 pt (−3%) |
| hammer01_mm | 0.88c | 0.9985 | 0.9938 | −0.46 pt | no | −0.15 pt |

**forcep03 and forcep04 are the two largest gains, +3.86 and +3.98 pt, both fully separated.**
Section 3's collapse pair is the whole problem on this dataset and nothing in sections 4 or 8
moved it: the dilation sweep, blackout, the mask channel and `pad_frac` all left it at
0.815/0.827. This is the first intervention that does, and it is the one the geometry predicted.

The prediction is supported but **not cleanly monotone**, and the exceptions are informative
rather than noise: sharpspoon is the thinnest class of all (0.32c) and gains only +0.17, but it
starts at 0.9831 with almost no headroom; hammer and scalpel are the two thickest and go −0.46
and +1.87, so thickness alone does not order the outcome. Read it as "the classes that were
being destroyed recover", not as a law.

Every per-class gain is smaller than c448's on the same class (the bracketed share), and the
four separated classes capture 66-92% of it while the four unseparated ones capture 21% or
less. Consistent with both being the same mechanism, with c448 buying more of it.

**Correction, 2026-11-05:** the c448 column and the aggregate c448 row were first written from
three runs, 0.9474/0.9499/0.9499. Only **two** convnext c448 runs exist; a glob matched the
seed-1 directory twice, so seed 1 was double-counted. Corrected here: mean 0.9487 not 0.9491,
+2.42 pt not +2.46, and six of the eight per-class c448 deltas moved (forcep04 +5.51 not +5.18,
forcep02 +1.07 not +1.53, forcep01 +0.99 not +1.19, hammer01 −0.15 not −0.26, scarstick01 +4.33
not +4.39, scalpel01 +2.03 not +2.08). The control and stride-2 columns were unaffected.

### 9.3 Cost

Measured on `dbe-cia-sl19-01`, 80 iterations each, one run per Quadro RTX 6000. The two ran
**concurrently on GPUs 2 and 3**, not sequentially, with two eval passes also on the box — so
they share a measurement window, though not a GPU. An earlier 1.14 s/iter figure for the
control came from a six-way-contended run and was discarded as not comparable.

| | s/iter (iter 80/80) | parameters |
|---|---|---|
| stride 4 | 0.22 | 27,827,816 |
| stride 2 | 0.47 | 27,827,816 |

**~2.1x the time at an identical parameter count**, so this is a compute change, not a capacity
change — which matters because `cropcls-results.md` section 4 records resnet18-vs-resnet50 at
only −2.2 points, i.e. capacity was never the binding constraint.

Against c448's **+2.42 pt**, stride 2 buys **62% of the gain** (+1.50 of +2.42), and 66-92% of
it per class on the four classes that separate. What the extra cost *ratio* is cannot be stated:
no matched-condition c448 timing exists in this repo, so the earlier "roughly 4x, about 45% of
the extra cost" was an unmeasured assertion and is withdrawn. Measuring c448 the same way is
cheap and not yet done.

**Corrections, 2026-11-05.** Two numbers here were wrong when first written. The s/iter figures
were quoted as 0.29 and 0.52, which are the **iteration-40** cumulative-average lines read while
the runs were still going, not the final figures above; the profiler's own TOTAL/calls averages
are 0.205 and 0.456, so the ratio is stable at ~2.1-2.2x but the absolute numbers were not the
run's own. The memory figures, 8,962 MiB and 16,268 MiB, came from a **single `nvidia-smi`
snapshot taken ~20 s after launch** — before the stride-4 run had logged its first 20
iterations — so they are an early sample, not a peak, and are withheld pending a
`torch.cuda.max_memory_allocated` read. The ~1.8x memory ratio quoted from them should not be
relied on.

### 9.4 What this does and does not settle

- **Not settled: significance.** Two seeds per arm, and the permutation floor at 2-vs-2 is
  p = 0.333. The spec requires two seeds minimum and they are met; it also wants a rankable
  gap, and that needs n >= 3 per arm.
- **Not settled: peak memory**, and the c448 cost it would be compared against. Both withdrawn
  above rather than estimated.
- **Open: does it compose with c448?** If the mechanism is the early reduction, `stem_stride: 2`
  at c448 should give stages 224/112/56/28 and go further still. Untested, and the memory at
  c224 is already 16 GB, so c448 at stride 2 may not fit at `crops_per_batch: 64`.
- **Open: `stem_stride: 2` with `pad_frac: 0.05`.** Both now look positive and they are
  independent mechanisms (field of view vs sampling density); nothing says they add.
- **Untested on the resnets by design** — the ask was convnext only. The code is trunk-agnostic
  and `ceil((k - stride) / 2)` was verified to reproduce native padding on resnet50 and
  resnet50d, so those arms are launchable without further code.

## 10. What has been tested, and what is now stale

Inventory taken 2026-11-05: **70 run directories**, 57 carrying a Hungarian eval. All use a
static crop folder; no surviving run trains from the live pipeline.

### 10.1 Test families

| family | what it varied | state |
|---|---|---|
| dilation sweep | `mask_dilate` 3/4/8/16/32/64, c224 and c448 | **confounded**, see 10.2 |
| closing | `mask_close` at radius 3 and 16 | **runs gone**, see 10.2 |
| blackout | `blackout` true/false, 45 runs at false | live |
| mask channel | `mask_channel` 4th input channel, the `mask4_*` family | live |
| trunk | resnet50 / resnet50d / convnext_tiny, `.a1_in1k` and `.fb_in1k` pinned | live |
| crop size | 224 vs 448 | live |
| crop padding | `pad_frac` 0.0 / 0.05 / 0.10, sections 8.1-8.3 | live |
| stem stride | `stem_stride` 2 vs native 4, convnext only, section 9 | live |
| freeze trunk | 1 run | live |
| forcep03/04 subset | `crops/train_f0304`, 7 runs | never comparable, see 10.2 |

### 10.2 Stale, and why

- **The 9 plain-dilation runs are confounded.** Until 2026-11-05 `cropcls/export.py` passed no
  `box=`, so scoring cropped the *dilated* tight box while training cropped the undilated one -
  wider by `2 * mask_dilate` px per axis, a mean long-axis ratio of 1.04x at dilate 4 rising to
  1.33x at 32. The model was scored on a framing it never trained on, by an amount growing with
  the radius. Section 4's `dilate 32 = -6.5` is the row to distrust; `dilate 16 = +4.2` was
  achieved despite the handicap and is if anything understated. Re-running the sweep is the only
  way to recover those numbers.
- **Section 4's closing rows (+0.4, -0.2) and its `remove RGB` row (-5.2) have no artifacts
  left.** All 70 surviving `config.json` files carry `mask_close: false` and `mask_only: false`,
  and the key is present in every one of them, so this is deletion rather than a missing field.
  `runs/` is gitignored shared state that other sessions reorganize. Those three numbers are
  now unverifiable and should be re-run before being quoted again. Closing is separately
  **not** affected by the export.py confound - `close_mask` does not grow the outer silhouette,
  so its tight box is unchanged.
- **The 2 batch-24 c448 runs are not comparable to the other 68.** `crops_per_batch` is 24
  against 64 everywhere else, because c448 with `stem_stride: 2` OOMs at 64, 48 and 32 (23,590
  of 24,576 MiB at 64). They are matched to each other at 256,008 crops seen, so the
  stride-2-vs-stride-4 contrast inside the pair is valid; the absolute numbers are not
  comparable to the 0.9487 c448 winner.
- **The 46 runs on `crops/train` keep their results but cannot be re-run.** That folder predates
  `pad_frac` and `CropFolder` now raises on it. The results stand because `crops/train` was
  verified byte-identical to `crops/train_pad20` re-tightened to its own alpha; reproduction
  needs the new folder plus `pad_frac=0.0`.
- **The 7 `crops/train_f0304` runs were never comparable** - a 2-class subset, so their
  `classaware_f1` is not on the 8-class scale.
- **The 40 `detector_x_class` evals are not the ranking metric.** `score_mode=detector` is
  (section 5); 53 runs have one.

### 10.3 Not stale

Everything at `mask_dilate: 0`, `crops_per_batch: 64`, scored `score_mode=detector`: the
`mask4_*` mask-channel family, the trunk and crop-size comparisons, the `pad_frac` arms of
section 8, and the `stem_stride` arms of section 9. That is the set the section 9.1 ranking is
drawn from, and the current leader within it is convnext_tiny c448 at **0.9487** (n=2).


## 11. Scale awareness via FiLM (spec Step 12), 2026-11-10

Set-C copy-paste classifier (`configs/cropcls/setc_cp_cxt_c448.yaml`), control vs `scale_film=true`,
four seeds each. Runs and scripts: `/data/janick.bilang/runs/film_step12`. Rank metric:
`classaware_f1`, Hungarian, `score_mode=detector`. Host dbe-cia-sl19-01, one GPU per run.

| arm | piles f1 | single f1 | cp-occ f1 | piles exact | piles exact + post-hoc length |
|---|---|---|---|---|---|
| control s0/s1/s2/s3 | 52.78 / 58.66 / 54.27 / 52.70 | 82.61 / 87.49 / 82.43 / 84.83 | 75.02 / 78.88 / 76.07 / 76.92 | 0.539 / 0.599 / 0.554 / 0.538 | 0.610 / 0.645 / 0.627 / 0.610 |
| control mean | **54.60** (sd 2.8) | 84.34 | 76.72 | 0.558 | 0.623 |
| FiLM s0/s1/s2/s3 | 48.40 / 56.20 / 54.19 / 51.91 | 77.82 / 86.51 / 84.65 / 82.43 | 71.21 / 78.10 / 76.99 / 75.66 | 0.494 / 0.574 / 0.553 / 0.530 | 0.575 / 0.637 / 0.604 / 0.601 |
| FiLM mean | **52.68** (sd 3.3) | 82.85 | 75.49 | 0.538 | 0.604 |
| FiLM warm start (from 20261109083120, 2000 it), s0, diagnostic n=1 | 57.43 | 87.40 | 79.72 | 0.586 | 0.643 |

Exact / family / +-10 mm top-1 on matched detections, four-seed means:

| arm | piles | single | cp-occ |
|---|---|---|---|
| control | 0.558 / 0.744 / 0.711 | 0.843 / 0.968 / 0.956 | 0.785 / 0.961 / 0.938 |
| FiLM | 0.538 / 0.733 / 0.699 | 0.829 / 0.966 / 0.949 | 0.773 / 0.963 / 0.936 |

Reference points from before this step (frozen `20261109083120` classifier, `occ_analysis`):
post-hoc fixed product 0.610 piles / 0.894 single; learned per-detection weight 0.641 / 0.924.
No FiLM arm reaches either. The same fixed product applied to these runs gives 0.604-0.623 on
piles (means above).

- **No gain.** By the pre-registered rule (seeds 0/1, +-1.6) FiLM loses by 3.4. That rule assumed
  0.82 seed variance; the measured control sd is 2.8. Over four seeds the gap is -1.9, inside one
  sd, which makes it a tie under any threshold based on the measured variance. Paired by seed,
  FiLM is lower at all four seeds (-4.4 / -2.5 / -0.1 / -0.8), which is not significant (sign test
  p ~0.125).
- **The trained FiLM barely uses the scale input.** A test-time shift of the length feature by
  +-0.1 log2 (about 1.3 length steps) changed 0 of 1127 single-set argmaxes for `film_s1` and 1
  for `filmwarm_s0` (`probe_report.py`). Only these two runs were probed, and only on the single
  set. The film output layer ends near zero (abs max 0.013-0.034). Final train loss is
  0.002-0.005, so appearance alone fits the training crops and length gets little gradient. That
  causal link is inferred, not measured.
- **Seed variance on this config is large.** Control piles f1 sd is 2.8 points (range 52.7-58.7),
  not the 0.82 measured on earlier configs. A 2-point claim on set C needs at least four seeds.
- **Post-hoc length fusion is the mechanism that works.** It adds +6.6 points of exact top-1 on
  piles (mean of 8 runs, every run positive, min +4.6). Length statistics come from each run's
  single-set pass, so the single-set post-hoc numbers are in-sample; piles and cp-occ are not.
