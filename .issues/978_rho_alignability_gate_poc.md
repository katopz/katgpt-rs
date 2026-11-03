# Issue 978: ρ alignability gate — planted-ρ fixture + pre-registered falsification arms (arXiv:2610.39302 Thm 3 distillation PoC)

**Status:** Open — PoC pending; promotion consumer-gated
**Date:** 2026-11-03
**Research:** [katgpt-rs/.research/646_Platonic_Representation_Good_Views_PRH.md](../.research/646_Platonic_Representation_Good_Views_PRH.md)
**Source:** [arXiv:2610.39302](https://arxiv.org/abs/2610.39302) Thm 3 + Prop 2 (Balestriero & LeCun)

## Background

The PRH paper's ρ diagnostic — state-explained variance fraction over grouped multi-view embeddings, ρ = tr(Cov(E[f(x)|z])^{1/2})/k on whitened spaces — lower-bounds best-rotation cross-space alignment error by 2k(1−min ρ) with **no pairs, no labels, no training**. Workspace signal-diffs (Research 646 §5): nothing ships a one-space pre-pair alignability predictor; `leakage_probe` simulates the full two-space attack, `gw_alignment` aligns, `basis_split` targets one nuisance in one eigenspace. Prior-art verdict: the measurement itself is the paper's own result (Schnaus et al. 2610.09411 covers cross-set predictability) — the surviving delta is OPERATIONAL: a go/no-go gate that refuses to spend Procrustes/transport compute when the floor already proves misalignment, and embedder ranking by ρ. This issue is the engineering adoption, not a novelty claim.

## Scope

New opt-in measurement primitive in `katgpt-core` (proposed name `nuisance_fraction` — re-run the substrate-first grep at implementation time: `grouped_variance`, `between_group`, `icc`, `rho_space`, `scene_fraction` are candidate pre-existing spellings), re-exported at the root. Zero-alloc scratch pattern per `gaussianity_probe`.

- [ ] **T1. Core primitive.** `rho_fraction(groups: &[&[f32]], scratch) -> RhoReport { rho, between_eigenvalues, k, whitening_convention }`: whiten over the population, per-group means, between-group covariance **corrected for finite-group spread** (subtract the mean within-group covariance / n_g — the paper's Table 6 estimator subtracts a fifth of the mean within-image covariance for its 5-view groups; the uncorrected form overstates Cov(E[f|z]) by ≈ within-variance/n_g and at n_g = 2, e.g. healer format-pairs, that is HALF the within-variance), ρ = Σᵢ√λᵢ(C)/k. Loud refusal (not a zero) when any group has < 2 members or n_groups < 2 — ρ is undefined on single-view data, never silently computed (Research 646 §8).
- [ ] **T2. The floor.** `alignment_floor(rho_a, rho_b, k) -> f64` = 2k(1 − min ρ) in the whitened RMS² metric, returned WITH the convention documented — consumers compare post-Procrustes RMS² against it, never raw cosine.
- [ ] **T3. Planted-ρ fixture + falsification arms** (the pre-registered proof path; every arm is a failing-loud test):
  - planted fixture **`f = ρ·s + √(1−ρ²)·n`** with controlled groups (Cov(s) = Cov(n) = I, independent): measured ρ recovers the planted value within tolerance at ρ ∈ {0.2, 0.5, 0.8, 0.95}. ⛔ The naive `f = √ρ·s + √(1−ρ)·n` planting is WRONG — with that form C_f = ρI and tr(C_f^{1/2})/k = **√ρ** (planted 0.5 reads 0.707) — carry it as an explicit tripwire test asserting the wrong planting reads √ρ, so the sign of the convention can never silently flip;
  - shuffled-group control: relabeling views across groups collapses ρ toward the floor value (the non-redundancy pin, `gaussianity_probe`/Bench 673 pattern);
  - bound arm: synthetic paired spaces with planted nuisance share — post-Procrustes RMS² must respect 2k(1−min ρ) at every planted level;
  - rotated-copy arm: one space = rotation of the other ⇒ floor slack ≈ noise, ρ equal both sides;
  - degenerate-group arm: single-member groups refuse loud.
- [ ] **T4. GOAT gate.** G1 = the arms above; G2 ≤ ~10 µs/report at d=256, n=256 (scratch, release); G3 default-feature count unchanged; G4 zero alloc steady-state. **Promotion stays opt-in pending a concrete consumer measurement** (`latent_confounder_audit` precedent, Bench 194): the gate is real when a consumer shows refusing a doomed transport/rotation SAVES measurable compute or prevents a wrong verdict — candidate consumers: `leakage_probe` front-end (skip `transport` when the floor proves misalignment), `katgpt-canon` pre-check (ProcrustesAdapter vs GL(k) objective-class rule, Research 646 Law 2), MAG TransferMetric headline swap. File the consumer measurement under this issue before any promotion.
- [ ] **T5. Cross-repo consumers (their own lanes, not this repo):** riir-refine — ρ per embedder over `assets/format_pairs_v1.jsonl` grouped by rule family (rule-explained fraction; ranks embedder candidates beside the FLINT architecture axis, Research 309); riir-reflex — ρ per suite embedding grouped by label + ρ-gated cross-lane Procrustes probe. Recorded here so the consumers are findable; each lands behind its own repo's issue discipline.

## Disclosures

- ρ needs grouping labels sharing one world state with independent nuisances — on single-view corpora it refuses, it does not read 0.
- Whitening convention is part of the report; ρ values across differently-whitened spaces are incomparable.
- Concurrent art (Schnaus et al. 2610.09411) covers cross-set predictability — cite it in any consumer doc; the gate here is adoption engineering.
