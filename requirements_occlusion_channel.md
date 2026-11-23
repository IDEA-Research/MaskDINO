# requirements: occlusion channel

The user's ask, pasted verbatim and unedited (typos intact). This file is the reviewers'
source of truth per `CLAUDE.md`. Nothing below this heading is the session's words.

---

## 2026-11-04, on whether to feed the occluders

> should i also feed the masks which occlude my instrument? also in layer 1 as channel 4 but maybe with another value?

## 2026-11-04, the go-ahead

> lets setup a new test with a new channel for occlusion

---

## Context the user supplied earlier in the same conversation (verbatim)

These set the goal the channel is meant to serve. They are the user's words, not a summary.

> does cropcls have a reject class? or how can i reject detection where the classifier is not sure

> but how would i do this fully automatically? imagine i have three scissors, excactly the same type but one is 1 cm longer the other one 2 cm.

> and i have only 20 different images of the scissors (rotational from side a and b)

> how about we use mixcut, such that the network learns it could also predict two classes of 0.5 to reduce its confidence on undistiguishable aprts?

> but does precissison increase but recall reduce? lets see the 101 fragile forceps03

---

## Decisions the user made by selecting from offered options (2026-11-04)

Not free text, so not quoted as prose. The session asked two multiple-choice questions and
these were the selections. Recorded here because the reviewers' source of truth is this file,
and a decision described only in the spec proves nothing.

**Q: "Which base config should the occlusion-channel arm build on?"**
Selected: *"blackout:false + mask_channel (recommended)"* - occluders as channel 5 on top of
the mask-channel line, rather than a `blackout: true` base or a 2x2 of both.

**Q: "Should the arm ship the channel alone, or the channel plus soft targets?"**
Selected: *"Channel alone first"* - no soft-target loss in this step, over
"channel + censoring-driven soft targets" and "both as separate arms".

The user's ternary suggestion ("maybe with another value") was answered in the session with a
technical argument for two binary channels instead; the user did not restate it after that and
chose the channel-5 option above, which presupposes a separate channel. If that reading is
wrong, this arm is built on a misreading and should be stopped.

## Assumption the session made that the ask does not settle

"the masks which occlude my instrument" is implemented as the union of the other **predicted**
instance masks in the frame, not GT masks. Rationale: predictions are available identically at
training and inference time, so the channel cannot become a train-only oracle. GT occluders
would not exist at inference. Flagged because the verbatim ask does not distinguish them.

---

## 2026-11-11, branch 59: the channel on set C

> Here i want to test out the impact of the occlusion channel. see if this improves something or not. test it on set c our currently worst set. I want to see if the network learns that legnth parameters are not reliable if the object ends in a pile.

## 2026-11-12, scope of the first step

> how about we first just test if the additional channel helps, on the normal training, see its impact then continue with the hyptohesis that length estimated helps when occlusion channel is available. The current hypothesis would be, classifier profits from occlusion knowledge, since we saw that the classisifer learns some kind of size parameter already on its own, lets see if occlusion channel helps to furhter improve.
