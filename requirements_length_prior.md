# Requirements: length prior in the classifier (spec Step 13)

The user's words, verbatim and in order (2026-11-11), unedited. Assistant turns in between are
summarised in brackets for context only; they are not requirements.

> i would run these tests on totally three seeds, since most other runs also are on three seeds, then we calculate sd and just mention there was no significant change observed since it seems that the classifier already learns this parameter quite well

> nevermind back to our scaling problem. i think improtant are length and width, the rest i dont know. but how about we just have two trainable weigths like this f=f*(alpha*length+beta*width) where the weights are initialized with random number and thus force the network to use them somehow

[Assistant: a single scalar times every pooled feature only rescales all logits, so it cannot
change which class wins; proposed instead a per-class Gaussian length term on the logits,
`logit_c - k * ((L - mu_c)/sd_c)^2`, with mu/sd from the training crops.]

> whats a gaussianterm

> how is floor calculted though? it seems to be like a confidence parameter of length/ geometric measurments

[Assistant: the floor is a fixed 2% mixture (98% the measurement is good, 2% it is
uninformative), plus length is switched off when the mask touches the frame edge; a learned
per-detection weight w(x) from reliability features gave piles exact 0.641 vs 0.610 for the
fixed floor.]

> i would assume if we fix k before hand we would take away the burden of the classifier to extract the length, sucht that it can focus on appearance more.

[Assistant: proposed arms A = control + post-hoc product, B = joint training with the fixed
length term (k = 1, 2% floor, edge off), C = B + 30% length dropout.]

> but should control + post-hoc product then not also use edge off?

> I would leave out step C for the moment, since it would be less relevant if B works right?

> one more thing, i also have an occlusion channel in the pipeline (next to the mask channel) which shows how much occlusion a instrument has to work with. we could  use that to get rid of some of the handcrafted parameters right?

[Assistant: yes for the floor and the hand-picked reliability features (a learned gate from the
trunk features with the occluder channel on, fit on held-out data); proposed doing Step 13
(A vs B, no occluder channel) first and the occluder gate as a later Step 14.]

> yes start with it, report back when finished, only prompt me if you hit a roadblock
