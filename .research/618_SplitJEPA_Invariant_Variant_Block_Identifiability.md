# Research 618: SplitJEPA — Reconstruction-Free Invariant/Variant Block Identification

> **Source:** [SplitJEPA: Learning Invariant and Variant Latent Worlds without Reconstruction](https://arxiv.org/abs/2610.12349) — Hua, Liu, Zhao, Zheng, arXiv:2610.12349 (2026-10-08). Project page: https://splitjepa.github.io/ . Code not yet public at distill time (the provided GitHub URL hosts the project site only; the estimation recipe is fully specified in the paper's Appendix C.1).
> **Date:** 2026-10-11
> **Status:** Active — GOAT verdict (modelless track: Gain, plan + issues filed) + training-track Gain (riir-train plan filed). NOT Super-GOAT: the estimator family ships (TILR, Plan 425, DEFAULT-ON) and is classical (SFA / within-class scatter); the paper's contribution — and ours — is the identifiability theorem, the refuse-loud gates, and the certification instruments. Landed: katgpt-rs `1e35e7bb6019` (2026-10-11, Claude verdict session 782f282b — round 1 REVISE 6 fixes, round 2 AGREE). **Phase 0+1+2+3 COMPLETE 2026-10-12** (`acbe6450d` T0 measured/T0.4 flag, `85a0ad8a5` Phase-1 primitive `basis_split`, `3ca8a1470` bench+G4, `76fc7cba1` README row; G2 CLOSED at the quiet window: **1.923 ms @ d=64/N=2048, 5.2× under the 10 ms bar**, preflight-PASSED load 3.98 — `.benchmarks/935_basis_split_g2_latency.md`, plan 625 zero open rows); TILR-first verdict: TILR passes G1 exactly on planted blocks (report-only posture may compose TILR alone); the f64/zero-alloc/gated path justified by the measured residual gaps (228 allocs/call, τ-heuristic vs eigengap gate, no whitening form).
> **Related Research:** 138 (LeJEPA / Klindt et al. 2605.26379 — the paper's direct theoretical base; our verdict there was "no new algorithm, we don't train JEPAs" — this paper closes exactly that gap with a closed-form head), 498 (LeVLJEPA → the shipped SIGReg/gaussianity_probe), 232 (task-relevant identifiability), 408 (TILR, arXiv:2606.29164 — the closest SHIPPED cousin, Plan 425), 357/397 (contrastive difference sources), 114 (`subspace_ratios`)
> **Related Plans:** 425 (`tilr_invariant_subspace`, DEFAULT-ON), 301 (`subspace_phase_gate` SVD primitive), 152 (`river_valley::subspace_ratios`) — this paper → **Plan 625** (`basis_split`, this repo) + **riir-train Plan 457** (training program, evidence-gated)
> **Classification:** Public

---

## TL;DR

SplitJEPA proves that a JEPA trained with (a) predictive alignment, (b) matched-pair block consistency, and (c) isotropy normalization identifies the **invariant and variant subspaces** of a latent state with **no observation decoder** (Thm 2, up to within-block isometries, under full-rank variation `Σ_ΔL ≻ 0`), and that any downstream predictor consuming only the invariant block is **provably insensitive to variant perturbations** (Thm 3: `∂Y/∂z_L = 0`). The estimation recipe is closed-form on a FROZEN encoder: ZCA-whiten (or solve the generalized eigenproblem `M v = λ Σ v`), accumulate the second moment of matched-pair representation differences `M = (1/N)Σ Δr Δrᵀ`, bottom eigenspace = invariant block, complement = variant block. **Metric honesty (verdict-round-1 correction):** the paper's reported numbers — no-recovery leakage ≈ 0.33, full recovery < 1.5e-3 — are **ground-truth block metrics** (`‖A₁₂‖_F`, purity; ground-truth latents used for evaluation only). The held-out ratio `E‖P_inv Δ‖²/‖Δ‖²` we use as an instrument is a **generalization residual of the fitted null space**, not purity — it approaches 0 by construction once N ≫ d on pairs from the same generator. Purity is only measurable where variant ground truth exists (planted-block tests, deterministic sims).

**Verdict: GOAT — Gain (modelless track).** Not Super-GOAT, on two independent grounds: the estimator is **classical** — bottom-eigenspace-of-whitened-difference-covariance is Slow Feature Analysis (Wiskott & Sejnowski 2002) when pairs are temporal, within-class scatter / Fisher LDA when pairs are matched groups — and it **ships** in this workspace as TILR (Plan 425, `tilr_invariant_subspace`, DEFAULT-ON, whose `discover_invariant_subspace` accepts ANY difference vectors; pair semantics and orientation are caller choices). What does NOT ship — and is worth filing — is the **package around the operator**: the identifiability premise as a refuse-loud eigengap gate, whitening-as-default (generalized-eigenform), an f64/zero-alloc accumulation path for armed read use, the certification instruments (held-out `pair_leakage` residual, ground-truth purity harness, Thm-3 perturbation certificate), and the **free labeled-pair sources** this workspace already produces as exhaust (§4). Training track: separate **Gain** — a riir-train plan whose every run is gated on beating the closed-form head.

**One-line summary of the operational unlock: this workspace is a matched-pair generator that does not know it, and it ships no instrument that measures where variation lives inside a frozen embedding — the theorem tells us when the measurement is trustworthy, the instruments are the new code.**

---

## 1. Paper core

### 1.1 Setup

Latent `z = [z_H; z_L]` (invariant/shared vs variant/varying factors); observations `h = f(z)` through an unknown smooth nonlinear map; encoder `r = g(z) = [r_H; r_L]`. Two correspondence structures over pairs:

- **Predictive pairs** (`D_pred`): `z⁽²⁾ = ρ·z⁽¹⁾ + √(1−ρ²)·ε`, stationary Gaussian (OU), `0 < ρ < 1`.
- **Invariant/matched pairs** (`D_inv`): pairs sharing `z_H` exactly, differing `z_L` (same task content, different camera/execution/realization).

### 1.2 The three theorems

| # | Statement | Role |
|---|---|---|
| Thm 1 | Observations alone cannot identify the split: `z̃_H = z_H + ψ(z_L)` is observationally equivalent while leaking variant into the invariant block | The negative result — information preservation ≠ organization |
| Prop 2 | Predictive recovery (LeJEPA-class) yields `g(z) = Qz`, `Q` orthogonal — any rotation mixes the blocks | The residual ambiguity after JEPA training |
| **Thm 2** | If `r_H` agrees across matched pairs AND `Σ_ΔL = E[Δz_L Δz_Lᵀ] ≻ 0`, then `Q` is **block-diagonal**: `r_H = A₁₁z_H`, `r_L = A₂₂z_L`, blocks orthogonal. Identification is up to within-block isometries — subspaces, never individual axes | The payload |
| **Thm 3** | A downstream predictor using only `r_H` has `∂Y/∂z_L = 0` | The operational law: consume-only-invariant-block interfaces |

### 1.3 Estimation recipe (as implemented, Appendix C.1) — with metric attribution

1. Train encoder with `L = L_pred + λ_inv·L_inv + λ_var·L_var` (SIGReg-style anti-collapse).
2. **Freeze** the encoder. ZCA-whiten representations over ~30k unlabeled samples (the whitening is what makes `Q` square-orthogonal and the bottom-eigenspace reading valid — without it, low-total-variance directions masquerade as "invariant").
3. Construct N=30k matched pairs; `Δr̄ᵢ = r̄⁽¹⁾ᵢ − r̄⁽²⁾ᵢ`; `M̂ = (1/N)Σ Δr̄ᵢ Δr̄ᵢᵀ`.
4. Under exact block structure `Δz_H = 0`, the null space of the difference moment is the invariant subspace: bottom `d_H` eigenvectors = invariant block, orthogonal complement = variant block.
5. **Evaluation uses ground-truth latents** (available in the synthetic setting): Global/block R², cross-block leakage `‖A₁₂‖_F`, `‖A₂₁‖_F`, and block purity `1 − (‖A₁₂‖²+‖A₂₁‖²)/‖A‖²`. The robot setting has no latent ground truth; there the paper validates blocks by **differential response ratios** (r_H vs r_L distance under matched interventions: 8.4×/14.0× viewpoint, 2.1×/3.1× trajectory).

Measured facts worth adopting as laws: **the invariant nullspace identifies with ~1k pairs while the complementary block needs ~30k head-fit samples** (pair-budget law); shuffled pairs destroy recovery at unchanged global R² (the built-in adversarial control — note that under whitening, shuffled pairs give `M ≈ 2I` with no eigengap, which is exactly what the gap gate must catch); no-diversity (`Δz_L = 0`) collapses the method to identity — full-rank variation is not optional.

---

## 2. Path 0 decomposition (math, not training loop — per component)

| Component | Paper role | Modelless analog in this workspace | Status |
|---|---|---|---|
| Predictive alignment (JEPA training) | trains the encoder | latent-prediction SSL + TILR-style difference collection ship; encoder *training* is track (c) | training track → Plan 457 |
| `L_inv` matched-pair block consistency | bakes block structure into weights | **post-hoc equivalent exists**: matched-pair spectral head needs NO training (the paper's own recipe freezes the encoder first) | **extracted** → Plan 625 |
| ZCA whitening / generalized eigenform | makes the bottom-eigenspace reading valid | ships (`ica_lens::whiten_into`, `ns_inv_sqrt_psd_into`) | covered; **default**, not optional (round-1 fix) |
| Difference second-moment → eigensplit | the identification operator | **classical estimator** (SFA / within-class scatter family); **ships** as TILR for arbitrary difference vectors — real gaps are f32 vs f64, allocation, τ-rank vs gated rank, no whitening form | compose/extend → Plan 625 (Phase 0 measures TILR-first) |
| Sufficient-variation rank gate | identifiability precondition | **does not ship** — no refuse-loud eigengap gate on pair degeneracy | **extracted** → Plan 625 |
| Certification instruments | validation | **do not ship**: held-out `pair_leakage` residual, ground-truth purity harness, Thm-3 perturbation certificate (closest: nuisance-response audit + shortcut-leakage — §3.4) | **extracted** → Plan 625 + consumer issues |
| Thm 3 gradient isolation | robustness law | architectural analog ships (think-brain/info-brain separation); executable perturbation-certificate harness does not | extracted (harness) → Plan 625 + consumer issues |
| Per-block persistence ρ → decay λ | OU byproduct | global confidence-decay sigmoid ships; per-subspace λ does not (the consecutive-window fast/slow split is exactly SFA — named, not new) | follow-up (issue-level, P3) |

**Modelless-validable: YES — no riir-train deferral for the head.** The training loop remains available and is filed separately with an honest gate (must beat the closed-form head).

---

## 3. Signal-diffs (coverage-dismissal defense — each cousin read at the formula level; corrected in verdict round 1)

### 3.1 TILR (Plan 425, DEFAULT-ON) — the load-bearing diff, stated honestly

`tilr::discover_invariant_subspace(differences: &[&[f32]], tau)` accepts **any** difference vectors; "pair semantics" (model-state good/bad vs content-matched observations) is a CALLER choice, and the "orientation" delta is just `I − VVᵀ`. The honest residual gaps are engineering + instruments, not the operator:

| Axis | TILR (`tilr.rs`) | What Plan 625 adds |
|---|---|---|
| Accumulator | f32, allocating | f64 deterministic accumulation, zero-alloc scratch (BLAKE3-committable basis for armed read paths) |
| Rank selection | τ variance threshold (r≈12 at τ=0.90, heuristic) | eigengap gate with refuse-loud `Unidentified` + k-from-gap selection (theorem-backed) |
| Geometry | no whitening form | whitening default / generalized eigenproblem `M v = λ Σ v` (quiet-axis defense) |
| Instruments | none (validated downstream) | held-out `pair_leakage` residual; ground-truth purity harness; Thm-3 perturbation certificate |
| Consumer contract | γ-gated correction at inference | read-path projector + certificates; report-only first |

**Phase 0 consequence (round-1 fix):** before any new code, measure TILR's discovery + complement against the G1 planted-block bar (principal angles). Report-only consumers may be satisfiable by TILR composition alone; the f64/zero-alloc/gated path is justified only where that fails G1 or G4 (armed read paths, committable bases).

### 3.2 PEIRA (Research 115 / `peira.rs`)

Matched view pairs → EMA cross/within second moments → top-eigenvector alignment SCORE. Cross-products not differences; no split, no certificates. The fixed-pair accumulation discipline transfers.

### 3.3 SVCCA (`data_probe/cca.rs`) + `ica_lens` + `linalg/`

Owns eigen-denoise → whitening → canonical spectrum and the fixed-probe law; `linalg::symmetric_eig` is the deterministic f64 solver (Householder+QL, documented for d ≥ 256) and `karc::jacobi_eigen` covers small d — **Plan 625 must not build a fifth Jacobi** (three private copies already exist in `katgpt-spectral`). The machinery is the composition substrate.

### 3.4 `latent_confounder_audit` + `subspace_intervention` — the validation rig already exists

R_shift (nuisance-pair response ≈ 0) + shortcut-leakage audits; aligned/random/**residual** three-arm reads through a frozen head; principal-angle similarity. These are the Thm-3 certificate's measurement half. They neither construct a split nor quantify a claimed split's residual. Plan 625 composes them as its G1 harness rather than re-implementing.

### 3.5 Named negatives + the classical prior art (round-1 fix)

Published prior art the estimator reduces to — **the paper's novelty is the identifiability theorem, not the estimator**:
- **Slow Feature Analysis** (Wiskott & Sejnowski 2002): bottom eigenspace of the temporal-difference covariance — the `D_pred`/consecutive-window instance.
- **Within-class scatter / Fisher LDA**: bottom eigenspace of the matched-group difference covariance — the `D_inv` instance.

Workspace negatives (vocabulary-translation sweep, 2026-10-11, zero hits): difference/delta covariance as a named quantity · block purity metric · eigengap refuse-loud gate · whitened-difference SFA or within-class LDA (nearest: one bench comment; `fpcg_modelless` is mean-difference only) · null-space-of-differences invariant block · augmentation/cross-trajectory/cross-view correspondence pairs. (`ZCA` exists only in `ica_lens`.)

---

## 4. The free labeled-pair sources (the operational unlock)

| Frozen embedding | Free matched-pair source | Pair type | Consumer issue |
|---|---|---|---|
| Decision-server hashed-feature embed | option-permutation probe pairs (K deterministic orderings, content-identical) | matched | reflex issue 086 |
| Healer dense-embed (`dense_embed`) | **compile-gated mechanical fix pairs** — before/after spans of verified edits are semantics-preserving by construction | matched | refine issue 162 |
| Game style vectors (`style_weights[64]`, affect direction vectors) | consecutive trajectory windows (predictive/OU — the SFA instance); same temperament across weapon-move mixes (matched) | both | riir-ai issue 1056 |
| Laya encoder outputs | same perm/paraphrase pairs routed through the encoder; G5 cross-device same-input pairs | matched | follow-up |
| `NeuronShard` rows across freeze snapshots | consecutive envelope versions (consolidation drift = the dynamics) | predictive | follow-up (MAPE-K drift basis) |
| Belief-state vectors | consecutive ticks — literally the OU process | predictive | follow-up (per-block decay λ) |

Deterministic sims give **exact** matched groups (re-run one seeded trajectory, change one variant factor) with ground truth by construction — the one setting where purity (not just the residual) is measurable.

---

## 5. Fusion map

1. **Certification instrument (primary, repo-agnostic):** `basis_split` = whitening-default (generalized-eigenform) → pair-difference second moment (f64, zero-alloc) → eigengap refuse-loud gate → split → `pair_leakage` residual; ground-truth purity harness where truth exists. Consumes `linalg::symmetric_eig`/`karc::jacobi_eigen` + Plans 301/152 substrate; validated through the existing audit/intervention rigs. Phase 0 measures TILR-composition-first.
2. **Decision serving (public):** perm-pair presentation-subspace certificate — upgrade the permutation probe from a black-box spread gate to a subspace *explanation* (where does ordering-flip energy live, how much reaches decision scores), opt-in projection at the routing seam.
3. **Healer (investment #2):** formatting-variant basis of dense-embed space from compile-gated fix pairs → formatting-invariant retrieval, instrumented by a top-1 flip-rate-under-formatting-mutants metric.
4. **Game runtime (#1 investment):** temperament/state split on the shipped style/affect projections — audit that transient-synced scalars live in the fast block and stable reads in the slow block (the SFA instance, named as such); later (separate gated act) per-block empirical decay λ feeding the confidence sigmoid.
5. **Training program (track c, evidence-gated):** riir-train Plan 457 — game-perception JEPA encoder, grammar-head `L_inv` aux, block-structured QAT as the pre-registered Q4 re-open; every run gated on beating the closed-form head.

---

## 6. Verdict per track

| Track | Verdict | Tier rationale | Outputs |
|---|---|---|---|
| (a) modelless | **Gain — GOAT** | Q1 no twice over (estimator classical AND TILR ships it); Q2 yes (certification-instrument class + theorem-backed gates are new here); Q4 yes (≥2 pillars) — strict Super-GOAT fails on Q1, so no private guide; the plan + issues carry it | Plan 625 + issues 1056/162/086 |
| (c) model-based | **Gain** | Path 0.5: applicable, actively pursued; serving-envelope fit puts the closed-form head FIRST — training is the evidence-gated escalation | riir-train Plan 457 |

GOAT gates (Plan 625): **G1** planted-block recovery by principal angles (< 1e-3) + Phase-0 TILR-composition arm + shuffled-pairs adversarial canary (post-whitening `M ≈ 2I` ⇒ the gap gate must emit `Unidentified`) + degenerate-pair canary (gate must fire); **G2** `--release`, split < 10 ms at d=64/N=2k, preflight provenance quoted; **G3** consumer suites unchanged (report-only posture); **G4** fixed scratch, zero heap, counting-allocator canary. Promote to default only on a passed GOAT with a wired consumer; demote per the standing loser law.

---

## 7. Priority

- **P0** — Plan 625 Phase 0 (TILR-first measurement) then Phase 1. Everything else consumes it.
- **P1** — the three consumer issues (1056 split audit, 162 formatting-invariant retrieval, 086 presentation certificate), report-only first.
- **P2** — riir-train Plan 457 T1/T2 (cheap training runs, gated on the P0 head).
- **P3** — per-block decay λ (the SFA instance, named); freeze-snapshot drift basis; laya G5 drift-direction readout.

---

## 8. Weakest point (named; official framing after verdict round 1)

The operator is classical (SFA / within-class scatter) and ships as TILR, whose discovery function already accepts arbitrary difference vectors — pair semantics and orientation are caller choices, not contributions. The defensible delta is narrow and specific: the identifiability theorem converted into refuse-loud engineering (eigengap gate, whitening-default geometry), the certification instruments (held-out `pair_leakage` residual, ground-truth purity harness, Thm-3 perturbation certificate), the f64/zero-alloc committable path for armed read use, and the enumerated free pair sources. Plan 625's Phase 0 measures TILR-composition-first; if TILR + complement passes the G1 bar for the report-only consumers, the new accumulator is justified only by G4/armed-path needs. Secondary weakness: Thm 2 presumes JEPA-trained encoders; on arbitrary frozen encoders the agreement premise is approximate — the law is to publish the agreement residual beside every split rather than hide it.
