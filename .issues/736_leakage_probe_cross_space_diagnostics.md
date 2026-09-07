# Issue 736: leakage_probe + cross_space diagnostics — modelless open primitives (fusion idea, novelty TBD)

**Status:** Fusion idea — novelty TBD on the probe's *defense-audit* framing; feasibility verified (Research 540). Source: arXiv:2505.12540 "Harnessing the Universal Geometry of Embeddings" (vec2vec, NeurIPS 2025).

## Pinned claim (pre-search, per the TTPO rule)

"A modelless kNN attribute-transfer **leakage probe** that, given unpaired foreign-space vectors + labeled anchors in OUR space, scores the attribute-transferability of a stored vector store — for surfaces `riir-neuron-db` embedding indexes + `BonsaiEmbedder` outputs, consuming unpaired foreign samples, distinguished from `latent_confounder_audit` (single-space, counterfactual slices over OUR directions) by operating across TWO spaces with no encoder access."

## What ships today (signal-diff verified, Research 540)

- `mag::TransferMetric::{CkaLinear, RbfMmd, Wasserstein1d}` — scores a *direction*, not a two-space *map*.
- `latent_confounder_audit` — audits OUR directions via counterfactual slices in ONE space; not foreign-unpaired-across-two.
- `FaithfulnessProbe`, `gaussianity_probe`, `effective_rank`/`within_class_effective_rank`, `ica_lens` — all single-space.
- **Nothing ships unpaired cross-space alignment or a leak audit** (repo-wide grep for `vec2vec|2505.12540|universal geometry` = zero hits; CHaRS R389's unpaired-correspondence design never landed).

## Proposed modules

- [ ] T1 — `katgpt-core/src/leakage_probe/` (feature `leakage_probe`, opt-in): kNN attribute-transfer leak score; GOAT gate in the Bench-194 shape (planted-leak recovery, monotone-in-coefficient, G2 latency, G4 alloc)
- [ ] T2 — `katgpt-core/src/cross_space/`: unpaired alignment diagnostics (kNN-overlap / rank-agreement), reusing MAG `TransferMetric`s
- [ ] T3 — `katgpt-core/src/unpaired_transport/` (opt-in): whitening → PCA correspondence → Sinkhorn self-labeling → orthogonal Procrustes; **G1c pins ≈random on real cross-backbone pairs as an honest negative** (paper evidence + Bench 426/427 say the modelless ceiling is the OT tier)
- [ ] T4 — novelty deep-search BEFORE T1 lands: "embedding inversion detection", "vector database leakage audit", "membership inference on embeddings" (the searches run so far covered the alignment landscape, not the defense-audit framing)
- [ ] T5 — consumer: riir-neuron-db Issue 614 (E2) adopts T1 as its security test

## Refs

`katgpt-rs/.research/540` (full feasibility table) · MAG Plan 418 · SipIt Plan 561 (fusion endgame: probe → translate → invert) · CD-LAM Issue 194 · Bench 194 · arXiv:2505.12540
