# Research 686: The Content–Geometry Tradeoff in World Models — SVReg, Slow Channels, Gated Residual

> **Status:** Done — per-track verdicts locked (**PASS** modelless track · **Gain** training track → riir-train Plan 507). Claude verdict session `e5a64289` (claude_code reviewer, 3 rounds): R1 REVISE (6 must-fixes — Plan 507's Bench-531 scale error, missing Mode-B falsifier, vague labeled set, wrong incumbent posture, no ground truth for the geometry bar, broken cost figure; + 3 minors), R2 REVISE (3 narrow — Mode B is a weaker sibling of Prop A.1, not an instance; Phase 0 surface split → game arm; pre-registered expectations + arm (e)), R3 **AGREE**.
> **Source:** "The Content–Geometry Tradeoff in World Models: When JEPA Learns Latent Maps" — Mengye Ren, Ying Wang, Christopher Hoang, Yilun Kuang, Yann LeCun (NYU / AMI Labs), Technical Report AGL-TR-2026-04, Nov 2026. doi:10.5281/zenodo.24356797 · [lab page](https://agenticlearning.ai/content-geometry-tradeoff/) · [PDF](https://cdn.agenticlearning.ai/25c41dcbc4b7a11e/content-geometry-tradeoff-2026.pdf) · alphaXiv `2611.jepa-latent-maps-content-geometry`. No arXiv ID (tech report); no code repo at distill time.
> **Date:** 2026-11-20
> **Related Research:** 138 (LeJEPA — SIGReg lineage), 498 (LeVLJEPA → `gaussianity_probe`, the closest diagnostic cousin), 504 (Orthogonal JEPA — the sibling geometry-recipe distill; #687 + Plan 351), 426 (Temporal Straightening — **the same authors; the paper's "SG + straightening" baseline IS this paper**), 358/462/465 (the JEPA-world-model PASS family), 513 (LpWM — slowness runtime duals), 618 (SplitJEPA — SFA/TILR), 661 (SimpleHC — gated-residual/identity-transition precedent)
> **Related Plans:** riir-train **507** (SVReg recipe arm for NextLat — the training half, this note's Gain), riir-train 351 (orthogonal factorized NextLat — coupling cell), riir-train 493 (SimpleHC arm — coupling cell), katgpt-rs 579 (#687 hinge primitives — landed)
> **Classification:** Public (this note)
> **Domain:** katgpt-rs (this note, public). Modelless track: no new primitive — PASS. Training recipe: riir-train Plan 507 (private).

---

## TL;DR

The LeCun-lab JEPA world-model line continues: a **one-sided variance floor along random directions (SVReg)** trained **without stop-gradient** beats both SG and full-distribution regularizers (VICReg/SIGReg) on latent GEOMETRY (smooth paths, distance ≈ progress), but its predictability pressure can erase slowly-changing hard-to-predict CONTENT (a resting pushed block); **slow channels** (fixed dims, variance-normalized slowness) + a **gated residual predictor** (identity + per-dim gated change) recover it. The paper defines the content–geometry tradeoff, proves the linear-encoder objective forms (Theorem 2.1), signal collapse (Prop A.1: latent covariance cannot identify signal recovery — only a held-out probe can), and content competition (Prop A.2: mode-entry growth rate over a shared latent coordinate).

**Distilled for katgpt-rs (modelless, inference-time):** nothing new stands alone. The paper's entire evaluation layer (participation ratio, jaggedness, distance rank correlation, per-direction variance floors) is covered by shipped substrate; its slowness/gated-residual content is covered by runtime duals; its latent-map/geometry findings validate the explicit-belief/DEC design. The one genuinely actionable content is the **training recipe** — a directly testable refinement of riir-train NextLat (`loss_nextlat.rs`), which trains belief-state heads with **detached (SG) targets** (`SmoothL1(sg[h], ĥ)`; uniformity opt-in — `lambda_unif` defaults 0.0, Plan 254). The recipe routes to riir-train Plan 507, whose plan-literal constants are σ-scaled to the MEASURED healthy profile (per-dim σ ≈ 0.01–0.03, Bench 531) and whose cheap first falsifier re-plants Bench 531's Mode A/B collapse modes on the NextLat game arm (Bench 531's own Mode harness lives on the edge_lora surface, which has no healthy σ anchor — its surface split is the plan's Phase-0 caveat).

---

## 1. Paper Core Findings

| # | Finding | Statement | Track |
|---|---|---|---|
| 1 | SVReg | `R = E_u[max(0, 1 − std_t(z_t·u))]²` over random unit directions; one-sided floor; trained WITHOUT SG | training |
| 2 | SG → poor geometry | One-sided SG objective doesn't ask consecutive latents to lie on a smooth path; jagged latent routes break latent-distance planning (hard-goal success SG 21–42% vs Slow SVReg 92–96% on first-person mazes) | training |
| 3 | Content–geometry tradeoff | Predictability pressure erases slow, action-unexplained factors (resting block R² < mean-predictor under SVReg; SG keeps them); content decided EARLY — the random initial encoder decides which runs lose the factor; SG→SVReg switch at 20k steps keeps acquired content | training |
| 4 | Signal collapse (Prop A.1) | With SG + finite data the encoder can read noise while latent covariance looks healthy (differs from faithful only by a global scale — invisible to variance/shape stats); held-out probe R² = 1/(1+σ²) vs 0 is the only detector | analysis (modelless-readable law) |
| 5 | Content competition (Prop A.2) | Two modes sharing one latent coordinate: entry growth `γ_no-sg = 2ηv_j[δ_i − δ_j − (φ_i−φ_j)²]`, `γ_sg = 2ηv_j·φ_i(φ_j−φ_i)` — without SG an action-driven mode can displace a passive persistent one; first-to-fill can keep the slot (both endpoints stable) | analysis |
| 6 | Theorem 2.1 | Without SG the reduced loss = tr of residual covariance (VAMP-like, penalizes unexplained variance incl. noise); with SG the target is fixed per update so noise directions are unpenalized | analysis |
| 7 | Slow channels | Fixed dims (last 1/8), variance-normalized slowness `R_slow = E_t‖b_{t+1}−b_t‖²/tr Var(b)` — normalization is shrink-proof | training (+ monitor dual) |
| 8 | Gated residual predictor | `ẑ = z_t + c⊙f(z_t, a_t)`, per-dim learned gate init 0.1 (LayerScale over time); unchanged factors cost nothing; collapses under SG (copy = trivial) | training |
| 9 | Geometry metrics | Jaggedness `J = Σ‖Δz‖/Σ‖Δz̄‖` (9-step MA; smooth ≈ 1); distance rank ρ (Spearman); participation ratio `PR = (Σμ)²/Σμ²` | analysis (all ship) |
| 10 | Empirical | Best planning in hard settings of 5 visual tasks; maze + real-city (Gaussian-splat) layouts appear in top-2 PCs, from first-person frames + actions, no poses; **J tracks planning, global ρ does not** | empirical |

## 2. Path 0 Decomposition — component → coverage + signal-diff → verdict

| Component | Coverage (analog ships?) | Signal-diff vs cousin (formula read) | Verdict |
|---|---|---|---|
| SVReg floor as OBJECTIVE | training-only — the incumbent postures it competes against ship in NextLat: **SG targets by default** (`lambda_unif: 0.0`, `SmoothL1(sg[h], ĥ)`), uniformity opt-in (SMT, arXiv:2606.06479, Plan 254, e.g. 0.1) | The paper's loser classes are BOTH testable here: arm (a0) = SG-only default, arm (a1) = SG + uniformity (the "SG + distributional regularizer" comparator), vs SVReg no-SG. Scale caveat is load-bearing: Bench 531 measured NextLat per-dim σ ≈ 0.01–0.03 and showed plan-literal VICReg-scale γ/λ is a scale mismatch — any floor term must be σ-scaled | **riir-train Plan 507** (training) |
| SVReg floor as DIAGNOSTIC | **COVERED** — `gaussianity_probe` (16 fixed directions: 4 axis + 12 BLAKE3 Rademacher, KS shape tests, Bench 673 GOAT) + #687 per-coordinate activity hinges `max(0, γ−σ)` (Bench 676 GOAT, landed `2ff1e9768`) | Probe checks SHAPE per random direction; #687 floors SPREAD per coordinate. A std≥1 floor without the training target is scale-arbitrary ("1" is the training's chosen scale) — no diagnostic meaning standalone. Both axes (shape, per-coordinate spread) already covered | covered — sub-delta |
| Participation ratio | **SHIPS** — `katgpt-spectral::spectral::participation_ratio` (used in `kv_cache_flatten_bench`) | identical formula | covered |
| Jaggedness J | **COVERED at capability level** — `mean_curvature` (riir-engine `latent_functor`, Plan 342): `arccos(v_t·v_{t+1}/‖‖)` path-curvature diagnostic | J = path-length ratio raw/moving-averaged (amplitude-sensitive zigzag); mean_curvature = angle between consecutive velocities (scale-free). Different quantities, same capability (latent path-smoothness diagnostic); no consumer distinguishes them — Plan 342's router-integration is the named re-open trigger | covered — variant without consumer |
| Distance rank correlation | generic Spearman (rank-transform + pinned tie rule) | no standalone primitive content | covered |
| Slow channels (runtime dual) | **COVERED** — `decay_confidence(Δt,λ)=σ(−λΔt)` (riir-games-shared), `persistence_eta` low-pass, `block λ` per-axis decay (measured ρ → λ per axis, compile-time pinned), EMA sites (R513 Path 0 row) | Block λ gives per-axis slow/fast structure; the variance-NORMALIZED drift-audit ratio is a monitor variant with no documented consumer (the 5 synced HLA scalars have no drift-budget gate to cheat) | covered; monitor variant = audited discard |
| Gated residual predictor | **COVERED at shape level** — every EMA site IS identity + gated delta: `b' = αb + (1−α)·drive ≡ b + (1−α)(drive − b)`; `evolve_belief_additive` (per-kind retention α), Plan 639 input-derived retention, SimpleHC identity transitions (Plan 493), LoRA zero-init A·B | The paper's delta is the gate being per-dim LEARNED (training-time). Runtime: gate is fixed (1−α) or input-derived. Shape fully shipped; learned gate → training | shape covered; learned gate → Plan 507 |
| Prop A.1 probe law | embodied in practice — every latent-health gate consumer has a probe downstream (reflex harness suites, refine score-bench, G5 parity fixtures, BLAKE3 digests) | The sharp residue — a noise-reading encoder passes BOTH per-coordinate floors (#687) and shape probes (gaussianity) — is real but mitigated by defense-in-depth (accuracy probes sit behind every floor consumer). Doc-note below, no file | doc-note (§5) |
| Prop A.2 displacement law | no capacity-contention consumer — grep found none: HLA axes are 5 fixed scalars (no shared-slot competition), Elo/adoption contention resolves by evidence counts not latent geometry, rule-index fan-out has no per-coordinate capacity | The ordering law `D_ij = δ_i − δ_j − (φ_i−φ_j)²` is modelless-computable IF δ/φ are measured per coordinate, but there is no site where a shared latent slot is allocated by it | **audited discard** — no documented capacity-contention site; re-open on the first one |
| Latent maps (maze/city in top-2 PCs) | validates the explicit design — two-brain spatial model (raw `MapPos` info brain + `SpatialBelief` think brain), DEC zone geometry ops (`hodge_decompose`, codifferential), zone attention dot-product+sigmoid | Our spatial navigation doesn't use learned latents; the paper is evidence FOR explicit-state design, not a primitive source | validation |
| Whitening overlay (closed-form floor satisfaction `w(z)=QΛ^{−1/2}Qᵀz`) | cousins ship — Bonsai2 Hadamard rotation lane (deterministic incoherence transform), `spectral_pre_rotate`, edge_lora overlays | Deterministic whitening of frozen embeddings has no documented consumer need; no metric-structure failure is recorded on any retrieval/centroid consumer | **audited discard** — re-open when a consumer reports embedding metric-structure failure |
| First-to-fill lock-in law | the house first-index tie rules already pin this everywhere | codification, not new math | doc-note |

**Reverse-grep before PASS (done):** `jagged|path smoothness|distance quality|rank correlation` across `*.rs` + `.docs`/`.benchmarks` — zero documented gaps naming these; the one insufficiency class the paper's Prop A.1 exposes (aggregate stats hide degenerate encoders) was already filed and **filled** (R504's reverse-grep found `gaussianity.rs:8-9`; #687 landed the per-coordinate hinge, GOAT Bench 676).

## 3. Adversarial panel record (§3.5)

Two advocates spawned (one round, same batch as the prior-art search). Coordinator dispositions — every discard carries its mechanism-level reason:

| Advocate finding | Disposition | Reason |
|---|---|---|
| No-GD 1 (Jaggedness + multiscale spectrum J_w) | discard | Capability covered by `mean_curvature` (Plan 342); no consumer distinguishes amplitude-ratio from angle diagnostics; the paper's "J tracks planning" is a claim about LEARNED latents for planning — ours aren't learned |
| No-GD 2 (distance ρ) | discard | Generic Spearman; no standalone primitive |
| No-GD 3 (PR) | covered | ships verbatim (`katgpt-spectral`) |
| No-GD 4 (directional variance floor spectrum) | discard | std≥1 scale-arbitrary without the training target (see §2 row 2) |
| No-GD 5 (whitening overlay) | discard | no documented consumer need; deterministic incoherence transforms ship (Bonsai2 Hadamard); re-open trigger recorded |
| No-GD 6+7 (probe+bound collapse detector; scale-degeneracy doctrine) | doc-note | The law is real (Prop A.1) and its hole is not covered by floors alone, but every floor-gate consumer already pairs a downstream probe; folding as §5 note, no new file |
| No-GD 8+9 (displacement forecaster + δ/φ estimators) | discard | no capacity-contention site ships (grep: none); estimators are generic AR(1) stats |
| No-GD 10 (slow-channel slowness monitor) | discard | no drift-budget gate exists to cheat; block λ covers per-axis structure |
| No-GD 11 (gated-residual overlay shape) | covered | EMA-lerp ≡ identity + gated delta with gate (1−α); learned-per-dim gate is training-only → Plan 507 |
| No-GD 12 (PCA-2 + Procrustes map audit) | discard | no documented zone-geometry gap; DEC covers zone-space ops |
| No-GD 13 (first-to-fill law) | doc-note | codifies existing tie rules |
| No-GD 14 (encoder selection by offline triplet) | discard | embedder swaps already gate on accuracy windows (the probe IS the selection); geometry triplet adds nothing without a documented failure |
| No-GD 15–17 (floor semantics, latent_audit substrate family, frontier tracer) | discard | infrastructure for the above discards; creating a new katgpt-core module family with zero landed consumers violates the ungoat discipline (smallest artifact that carries the verdict) |
| MB 1 (SVReg replaces `lambda_unif` on NextLat) | **FILED** | riir-train Plan 507 Phase 2 — the central recipe cell |
| MB 2 (no-SG posture vs EMA target on Plan 351) | FILED (fold) | Plan 507 Phase 3 coupling cell — 2×2 {target} × {SVReg} on the factorized arm |
| MB 3 (slow channels on NextLat/493) | FILED (fold) | Plan 507 Phase 2 — the paired lane (SVReg + slow channels ship TOGETHER: Prop A.2 predicts the SVReg-only arm loses passive factors; the paper's own theory makes these one experiment) |
| MB 4 (gated residual, gate-init sweep) | FILED (fold) | Plan 507 Phase 3 — coupled to the no-SG posture (collapses under SG by the paper's own Table 4) |
| MB 5 (analysis instruments: probe VAMP-r², φ/δ census) | FILED (Phase 1 first) | cheap telemetry; calibrated against the repo's documented collapse history (the labeled set) before any recipe cell reads them |
| MB 6 (SG→SVReg mid-training switch) | FILED (fold) | rides Plan 507's fresh SVReg arm + the resume lane |
| MB 7 (seed-dependent content loss census) | FILED (fold) | rides Phase 2's seed sweep — the mechanism check |
| MB 8 (norm/history/predictor details) | discard | confounding architecture changes; NextLat keeps its posture, only the loss/posture axes move |
| MB 9 (quest_grammar LoRA / TernaryDraftModel) | discard | no latent-dynamics surface (token-supervised); LoRA zero-init already embodies identity-default |
| MB 10 (laya encoder re-distillation) | discard | frozen BLAKE3-pinned artifacts behind G5 parity — a cross-repo re-baseline with no documented geometry failure; park until a consumer reports one |

## 4. Published prior-art search (§4 — run via subagent, 6 rounds)

- Headline "sliced variance regularization": **not found** — the term and the recipe are original to this Nov-2026 report.
- Closest adjacent: **LeWM** (arXiv:2603.19312 — no-SG JEPA WM + SIGReg, full Gaussian matching), **SIGReg/LeJEPA** (arXiv:2511.08544 — our R498/R138), **VISReg** (arXiv:2606.02572 — sliced sketching + variance, distribution-shape matching), **VICReg** (arXiv:2105.04906 — per-coordinate one-sided hinge, the floor's ancestor), **MotionJEPA** (arXiv:2609.23881 — the reverse polarity: predicts change to keep fast features), **Temporal Straightening** (arXiv:2603.12231 — our R426, the paper's own baseline).
- Combination check (floor-only anti-collapse + no SG + slow channels + gated residual, pre-Nov-2026): **not found**. No pre-emption; the recipe is novel — which routes it to the training track as a real recipe Gain, not a replication.

## 5. The Prop A.1 law (doc-note — the one modelless-readable residue)

**A variance floor certifies nothing alone — and this repo has PARTIAL in-house evidence.** Prop A.1: a degenerate encoder's latent covariance differs from the faithful one by a single global scale — it passes per-coordinate activity hinges (#687), shape probes (`gaussianity_probe`), AND erank simultaneously; only a held-out probe detects it. In-house partial evidence: Bench 531 Mode B shows per-coordinate floors are blind to rank-1-with-spread collapse (erank still catches it — "the detector keeps its job"), so Mode B is a WEAKER sibling of Prop A.1, not an instance; the floor misses it, erank sees it. Prop A.1's case is strictly worse, because it also passes erank. (Mode A, constant targets: the hinge fires but "cannot CREATE directions" — Bench 531's mechanism verdict.) Known floor-gate consumers pair downstream probes: reflex harness suites and refine score-bench are labeled probes; G5 parity fixtures and BLAKE3 digests gate the frozen-artifact surfaces. Standing rule for any future latent-health gate: name its paired probe, or print `unprobed`. This is the timed-region/`black_box` lesson one abstraction up: absent signal satisfies a floor exactly as absent work satisfies a timed bar. No new file — this paragraph is the record.

## 6. Verdict

**Modelless track (a+b): PASS.** No new standalone primitive. The paper's evaluation layer ships (PR: `katgpt-spectral`; shape-per-random-direction: `gaussianity_probe` Bench 673; per-coordinate floors: #687 Bench 676; path smoothness: `mean_curvature` Plan 342); the slowness/gated-residual runtime duals ship (`decay_confidence`, `persistence_eta`, block λ, EMA-lerp-as-gated-residual); the geometry findings validate the explicit-belief/DEC design. 6th instance of the canonical JEPA-world-model PASS class (358 SMWM, 360 AdaJEPA, 426 Straightening, 462 INTACT, 465 ODEWorld precedents; 513/618 were Gains because they carried documented-gap-filling primitives — this one carries none: the gap class it exposes was already filled by #687).

**Training track (c): Gain → riir-train Plan 507.** NextLat trains belief-state heads for speculative decoding with detached SG targets by default (uniformity opt-in, `lambda_unif: 0.0` in `Default`) — the SG posture the paper shows loses latent geometry, with the distributional-regularizer posture available as the (a1) comparator arm. The recipe (SVReg no-SG floor + slow channels + gated residual predictor, instruments-first on the game arm where true state and actions give the geometry bar ground truth) is a directly testable refinement, coupled to Plan 351's factorized cells and Plan 493's gate cells. Proof path: Phase 0's planted Mode A/B falsifier (re-planted on the game arm per Bench 531's surface split, with pre-registered expectations) + pre-registered paired gates in the plan; CPU-fixture lane primary (Bench 531 precedent ~80 s per A/B), 4090 LLM capture gate-demanded only.

**Two-sided priority-ladder check (fusion #1):** rulewright — nothing actionable: `ModellessEmbedder`/rule embeddings are deterministic hashed features, no encoder-training posture exists to change; score-bench is the probe. Decision-serving self-evolve — nothing actionable: the ESC gate is distance-only on a FROZEN encoder with per-suite fitted ρ (the calibration fit IS the probe; the gate is exact/stationary — a dynamics signal has no mutated state to read). Both honest no-action rows; the value lands in the training lane (#5 on the ladder), filed there.

**MOAT gates:** katgpt-rs — PASS rows don't claim the engine moat; the diagnostics cited are its standing substrate (cited, not extended). riir-train — active moat: Plan 507 is a training-method recipe with a falsifiable experiment, exactly the tier's charter.

**Fusion:** paper × Plan 351 (orthogonal factorized heads) × Plan 493 (identity-transition gates): slow channels are the CONTENT-side twin of SimpleHC's MECHANISM-side (identity transitions + gates reserve dims; slow channels give those dims a reason to hold persistent factors); SVReg + no-SG is the shared posture cell both arms need first. Prop A.2 supplies the experiment's prediction: the SVReg-only arm should lose passive factors on action-heavy data — confirm, don't confound (items 1+3 run as one paired lane).

**Weakest points, named:** (1) the "floor-as-diagnostic is scale-arbitrary" dismissal is a judgment call — a reviewer could ask for a sliced-floor margin readout as a cheap diagnostic; the defense: without the training objective there is no meaningful "1" (and on THIS trainer even the scale is measured, not assumed — Bench 531's σ ≈ 0.01–0.03 vs the paper's visual std≈1), and both coverage axes (shape per random direction via gaussianity_probe's 12 Rademacher directions, spread per coordinate via #687 hinges) already ship. (2) Plan 507's value depends on NextLat training being live — its primary surface is the 0-GPU game-arm fixture lane (Plan 351 precedent), sequenced to ride the next Plan-351 window, which keeps it honest if the lane is dormant. (3) Prop A.2's competition law is linear-encoder analysis — transfer to NextLat's MLP predictor is exactly what Plan 507's mechanism check (T2.5) tests before any recipe claim.

## PASS-Redirects (recorded in the 3 closest cousin notes)

498 (gaussianity_probe — SVReg is SIGReg's floor-only sibling; the distilled-diagnostic reading extends: shape + per-coordinate activity floors cover the floor's diagnostic reading), 426 (same-author baseline; SVReg corroborates the runtime-constructs-straightness PASS class; recipe → riir-train Plan 507), 504 (sibling geometry-recipe distill; slow channels + gated residual + SVReg extend the same NextLat recipe axis — Plan 507 couples to 351's cells).

## References

- Ren, Wang, Hoang, Kuang, LeCun. "The Content–Geometry Tradeoff in World Models: When JEPA Learns Latent Maps." AGL-TR-2026-04, 2026.
- Maes et al. "LeWorldModel." arXiv:2603.19312 · Balestriero & LeCun "LeJEPA." arXiv:2511.08544 · Wu et al. "VISReg." arXiv:2606.02572 · Bardes et al. "VICReg." arXiv:2105.04906 · Wang et al. "Temporal Straightening." arXiv:2603.12231 (Research 426).
- Workspace: Research 498 (gaussianity_probe / Bench 673), 504 (#687 / Bench 676 / Plan 351), 513 (slowness duals), 661 (SimpleHC / Plan 493), katgpt-rs Plan 579.
