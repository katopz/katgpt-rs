# Research 504: Orthogonal JEPA — Factorized Predictive States for Latent World Models

> **Source:** [Orthogonal JEPA: Factorized Predictive States for Latent World Models](https://arxiv.org/abs/2608.20065) — Taoyong Cui, Pheng-Ann Heng, Wanli Ouyang (CUHK), arXiv:2608.20065v1, 2026-08-20
> **Date:** 2026-08-24
> **Status:** DISTILLED — pending owner decision (katgpt-rs #687 primitives + riir-train Plan 351)
> **P3 cross-track note (2026-08-26, riir-train Bench 532):** the EMA m-sweep's 0.996-wins condition is NOT met — m=0.99 ties online on healthy data, higher m taxes quality monotonically, EMA alone cannot prevent planted collapse, and the only regime where EMA helps (damping the hinge's quality-tax oscillation) prefers m=0.999. No calibration evidence for `LatentSkillEvolution`'s EMA decay constant from this study.
> **Related Research:** 138 (LeJEPA — linear-identifiability precedent), 360 (AdaJEPA — runtime analog = `ReestimationScheduler`, PASS), 498 (LeVLJEPA/SIGReg — gaussianity probe #681 + dist-guard lineage), 288 (KARC delay basis — closest fixed-basis forecaster), 475 (ICA lens — orthogonality ≠ independence), 291 (cross-resolution Parseval precedent), 214 (spectral irrep channels)
> **Related Plans:** riir-train 351 (orthogonal factorized NextLat — the trained half)
> **Related Issues:** katgpt-rs #687 (orthogonal factorization primitives — the modelless half); riir-train Issue 743 (dist_guard — the hinge A/B closes its deferred Phase 2 gate)
> **Domain:** katgpt-rs (open primitives: orthogonalization, hinges, certificates) + riir-ai (affect directions, k_selector, rollout certificates) + riir-neuron-db (blend interference gate) + riir-train (trained factorized heads)

---

## TL;DR

Orthogonal JEPA replaces a JEPA's monolithic prediction target with **K learned basis matrices** `B_k ∈ R^{d×r}` (Kr=d) that analyze a stop-gradient EMA target into K orthogonal factors, each predicted by a **dedicated branch** from a shared context, then synthesized back via Moore–Penrose pseudoinverse. Losses: per-factor regression + within/cross-factor orthogonality + **factor-activity variance hinges** `max(0, γ−σ)` + VICReg-style encoder-variance hinge. Proposition 1 is elementary Parseval: orthonormal-complete B gives `‖B^Tz‖² = Σ_k‖B_k^Tz‖² = ‖z‖²` and perfect reconstruction `z = Σ_k B_k B_k^T z`. Gains over monolithic JEPA across vision/single-cell/health/control/molecular (e.g. Walker2d CEM 45.1 vs 4.9; better 100-step MD rollout RMSD).

**Verdict: GAIN (not Super-GOAT).** Published prior art is crowded (MoP-JEPA multi-branch predictors Jul 2026; FLAM/C-JEPA/IFactor/FWM factorized world models; VICReg hinges classical; pseudoinverse synthesis = textbook analysis-synthesis) — Q1 fails, Q3 (product selling point) fails. But the §3.5 Path 0 decomposition splits cleanly into a **dual-track contribution**:

1. **Modelless (katgpt-rs #687):** the paper's *structure* — orthonormal-by-construction bases on FIXED direction sets, per-coordinate activity hinges, Parseval runtime invariants, exact energy-budget truncation certificates, blend-interference gates — is closed-form linear algebra with NO gradient anywhere, and it fills **documented gaps**: production affect/motivation direction vectors are never orthogonalized (the orthogonality is a TEST-ONLY assumption, `civ/emotion/tests.rs:803`), and no per-coordinate activity floor exists beside the aggregate erank / distribution-shape gaussianity audits.
2. **Trained (riir-train Plan 351):** the *learned* half — data-adaptive orthogonal bases + dedicated per-factor heads + hinges + EMA target encoder — docks onto **NextLat** (`loss_nextlat.rs`), our shipped latent world model with a monolithic predictor and a **documented collapse history** (Plan 254 `lambda_unif` exists because h-space collapse was observed). ≤1.5 GPU-hrs for the falsifiable experiment.

---

## Key Mechanisms

| # | Mechanism | Formal statement | Training-only? |
|---|---|---|---|
| 1 | Orthogonal predictive factorization | `z_t^(k) = B_k^T sg(z_t)`, K factors, Kr=d | B learned → training; **fixed-set orthogonalization is modelless** |
| 2 | Dedicated per-factor branches | `ẑ_t^(k) = q_k(z_c, s_t)` from shared context | Heads trained → training; **per-factor ridge w/ per-factor λ is modelless (offline)** |
| 3 | Synthesis | `ẑ_t = (B^T)^† û_t`; orthonormal limit `= Σ_k B_k ẑ^(k)` | Classical; orthonormal B ⇒ `(B^T)^† = B` exactly (κ=1) |
| 4 | Orthogonality objective | `Σ‖B_k^TB_k−I‖²_F + Σ_{i<j}‖B_i^TB_j‖²_F` | As loss → training; **as defect-score diagnostic → modelless** |
| 5 | Factor-activity hinge | `(1/Kr) Σ max(0, γ_fac − σ̂_{k,j})` per projected coordinate | **Modelless monitor** (order statistics) |
| 6 | Encoder-variance hinge | `(1/d) Σ max(0, γ_enc − σ̂_j)` (VICReg) | **Modelless monitor** |
| 7 | EMA stop-gradient target | `θ̄ ← mθ̄ + (1−m)θ`, no grads to target | EMA update class ships (6+ sites); in-loop target encoder = training |
| 8 | Prop. 1 (Parseval) | orthonormal-complete ⇒ energy preservation + perfect reconstruction | **Modelless invariant/certificate** |

Paper's own caveats worth keeping: synthesis depends on **conditioning of B** (monitor singular values under autoregressive rollout); orthogonality ≠ statistical independence / disentanglement; variance hinges do NOT guarantee full-rank covariance.

---

## Path 0 Decomposition (component → coverage + signal-diff → extraction)

| Component | Coverage (analog ships?) | Signal-diff vs cousin | Extraction (modelless?) | Verdict |
|---|---|---|---|---|
| Orthogonal basis on latent state | **RICH** — `hodge_decompose` (exact⊕harmonic⊕coexact, d∘d=0), octopus triplet decompose+recompose norm-preservation, `cross_resolution.rs` (Parseval literally documented, R291), `rank_k.rs` PCA eigenbasis (orthogonal by Jacobi construction), KARC Chebyshev, `hla_eigenbasis`, SpectralQuant pre-rotate | All shipped orthogonality is either **domain-fixed** (cochains/polynomial bases) or **data-window eigenvectors**. NONE enforces orthogonality on **semantic direction SETS** — the 5 affect directions (`neuron_vessel_runtime.rs` `affect_directions`), 14 planner drive `dir_vec`s, archetype fields. Orthogonality there is a **test-only assumption** (`civ/emotion/tests.rs:803` "so that each direction captures exactly one axis"); production extraction is contrastive-mean and may correlate | **YES** — modified Gram–Schmidt (twice-reorthogonalized) at construction; byproduct residual = one-shot redundancy audit of the original set | **Issue #687** |
| Dedicated per-factor prediction branches | **PARTIAL** — `rank_k.rs` is the closest cousin (sigmoid partition-of-unity basis + PCA bases + **shared** rank-k operator + ridge damping); `neuron_vessel_runtime.rs` 5 affect projections are dedicated but **linear dot-products, no independent capacity/λ**; KARC = monolithic ridge over expanded features | rank_k: shared operator vs per-factor capacity — the "dominant signal drowns weak signal" case needs **per-factor λ/whitening** (scale separation), else per-head OLS ≡ monolithic solve algebraically. Honest equivalence condition recorded | **YES (offline)** — per-factor ridge heads refit at consolidation sleep, frozen during play; win condition = heteroscale targets (scale ratio ≥10²) | Fusion in #687 + Plan 351 |
| Energy-complete synthesis + Parseval invariant | **PARTIAL** — octopus triplet `test_norm_preservation`, DEC recombination α/β/γ, `cross_resolution` band-limited exact transport, phase-rotation/spherical-steering norm-preservation GOAT gates | Shipped norm-preservation is per-op (steer/rotate/decompose); no **runtime structural check** `‖z‖² ≟ Σ_k‖B_k^Tz‖²` on factorized semantic state, and no **exact truncation certificate** (dropped error = Σ dropped E_k, an identity by Parseval) | **YES** — O(d) check; truncation budget is exact, not approximate | **Issue #687** |
| Factor-activity hinge (per-coordinate variance deficit) | **EMPTY** — erank (aggregate), gaussianity probe (distribution shape per random direction), `spectral_flatness` (spectral), `VARIANCE_FLOOR=1e-8` (numeric guard only). **No `max(0, γ−σ)` activity floor anywhere** | erank = one scalar over the population (a dead channel hides in a full-rank aggregate — `gaussianity.rs:8-9` documents exactly this insufficiency class); hinge = **per-(factor,coordinate) attribution**, bounded [0,γ], 1-Lipschitz, cheaper than gaussianity | **YES** — Welford accumulators over a population window; γ schedule above the std-estimator's own sampling noise (`γ ≥ max(γ_min, c/√n)`) | **Issue #687** |
| Encoder-variance hinge | same class, different population (online latent coordinates) | — | **YES** (shared harness) | **Issue #687** |
| EMA stop-gradient target encoder | **SPLIT** — sg ships 5× training-side (CISPO `detach`, loss_asft, collimation, sdpg, d2f x̂₀); EMA direction updates ship 6+ sites (`self_evolve/evolve.rs`, offline_scored_wake α_f/α_s, curiosity_bridge, metamemory); **JEPA-style target encoder: zero hits** | The EMA *update class* is covered; the *in-loop frozen-target* pattern is the genuinely-trained half | Monitor half covered; trained half → **Plan 351** (m-sweep {0, 0.99, 0.996, 0.999}) | Plan 351 |
| Conditioning / singular-value rollout monitor | **RICH** — `multi_tick_band.rs` drift-onset slope test + spectral radius; `spectral_pencil` eigengap γk + Davis–Kahan pin | Shipped monitors watch the **dynamics** eigenvalues; the paper's caveat watches the **synthesis basis** conditioning. With orthonormal-by-construction B, κ(B)=1 — the caveat *converts to a certificate*: per-head amplification ≤ ‖W_k‖₂ = √λ_max(W_k^T W_k), **one spectral_pencil call per head**; composite rollout bound = Π_t max_k‖W_k‖₂ | **YES** — construction-time certificate, committed as metadata | **Issue #687** (cert) |
| Pseudoinverse synthesis | classical analysis-synthesis (dictionary learning / POD) | orthonormal ⇒ `(B^T)^† = B` (plain matmul, bit-identical under fixed op order); non-orthonormal fixed bases ⇒ ridge-regularized pinv via Cholesky with the **Batch-54 coupling window** (λ above f32 formation noise ~1e-7·scale, below damping budget, asserted) | **YES** | #687 / Plan 351 |

**Reverse-grep (documented gaps this paper fills → Gain confirmed):**
- `riir-ai/.../hla/multi_tick_band.rs:109-122` — "Plan 458 T4.2 **deferred**: ships opt-in until either (a) a QR-based non-symmetric eigensolver lands … **orthogonal predictive signal**. If NO → keep opt-in permanently as a diagnostic".
- `katgpt-core/src/data_probe/gaussianity.rs:8-9` — full-rank population can pass erank + flatness while marginals are degenerate (the per-coordinate hinge is the missing attribution axis).
- `riir-games-civ/src/civ/emotion/tests.rs:803` — orthogonal affect directions **assumed in tests only**; production never enforces.
- `cgsp_runtime/interpolation_geometry.rs:16` — open question: does the midpoint between emotion clusters collapse off-manifold.

---

## Novelty Gate (published prior art — mandatory searches, all run)

- **MoP-JEPA** (arXiv:2607.05238, Jul 2026) — K independent JEPA predictor heads (for stochastic multimodality; proves single deterministic regression collapses to the conditional mean). **Direct prior art for multi-branch JEPA predictors**, 6 weeks earlier.
- **Factorized Latent Dynamics for Video JEPA** (arXiv:2605.17165), **C-JEPA** (arXiv:2602.11389, object-centric), **FLAM** (arXiv:2602.16229 — its own motivation: "the inherent limitation of non-factored/monolithic world models"), **IFactor** (NeurIPS'23), attribute-factored WM (NeurIPS'23) — the factorized-target territory is crowded 2023–2026.
- VICReg/Barlow hinges: classical and cited by the paper. Pseudoinverse synthesis: dictionary learning / POD lineage. Prop 1: elementary Parseval (paper admits: "exact-orthogonality limit").
- **Eigenoptions / proto-value functions / NEO** (NeurIPS'25) + **successor features** (Barreto'18) — orthogonal-basis factorized prediction in RL, uncited by the paper.
- **FAR-JEPA** (preprints 202607.2043, Jul 2026) uses "the orthogonal JEPA solution class" as a term for a different thing (identifiability up to orthogonal rotation) — **name-collision risk**, concurrent work within 8 weeks.

**Verdict:** integration/engineering contribution, not a new primitive. The specific *orthogonal-dictionary* instantiation (learned orthonormal B_k + cross-factor Gram penalties + activity hinges on projected coordinates) was not found published elsewhere — plausibly novel as a formulation, crowded as a concept. Not Super-GOAT.

---

## Distillation & Fusion

1. **Orthogonalized affect/motivation directions (riir-ai, flag-gated).** GS the 5 HLA affect directions (and the 14 planner drive `dir_vec`s) at construction; defect score as a load-time diagnostic; measured cross-activation `|cos(q_i,q_j)|` before/after. Fills the test-only assumption gap. **Promotion is a gameplay owner call** (changes hand-tuned semantics — the CLR precedent). Consumer issue in riir-ai after #687 GOAT.
2. **Energy-budget certificate × `KSelectionBandit`.** k_selector picks k by coherence − α·latency (UCB1, no certificate). Parseval truncation gives the **exact** dropped-error identity `Σ_dropped E_k` — a principled reward axis for the bandit (energy-captured per µs), composable without changing the bandit.
3. **Activity hinge × dist-guard family (riir-train Issue 743 / Bench 494).** The hinge is the per-coordinate axis the erank/gaussianity audits lack; Plan 351's planted-collapse A/B (hinged vs baseline on the Bench 494 T3 harness) is exactly the "measured real-run collapse" gate Issue 743 set for entering its SIGReg Phase 2.
4. **Blend interference gate (riir-neuron-db).** `CommittedFieldBlend` sums sigmoid-gated archetype fields with no cross-archetype orthogonality check. Leakage `‖B_i^TB_j‖²_F` as (a) a refuse/downweight gate before committing a blend and (b) a **merge criterion** for `ShardCompactor` (merge most-aligned pairs first — redundancy as the signal).
5. **Factor-horizon allocation × confidence decay (two-brain).** Per-factor prediction horizons d_k with `sigmoid(−λ_k·Δt)` staleness shaping at readout — grounds "factors = distinct state components" in our shipped confidence-decay substrate (Plan 351 item; game-trajectory teacher is where heterogeneous factor timescales actually exist).
6. **Muon × orthogonal representations (training fusion, neither paper has it).** NS-orthogonalized *updates* preserving conditioning of orthogonal *representations* during training — hypothesis: drops the λ needed for ‖B_k^TB_k−I‖_F < 0.1 by 10× (Plan 351).
7. **Per-factor quantization with exact additive MSE.** Orthonormal B ⇒ quantization noise adds across blocks: total MSE = `Σ_k r·Δ_k²/12` **exactly**. Generalizes the q8kv per-32-block scale + outlier-sidecar family (Bench 691) from "block" to "orthogonal block". Hadamard skeleton: d=64=2⁶ (both `style_weights[64]` and HLA latent) ⇒ 384 add/subs, **zero multiplies**, dyadic 1/8 scale = exact integer core ⇒ cross-platform bit-identity.

---

## Dual-Track Outputs (this session)

- **katgpt-rs #687** — `orthogonal_factorization` primitives: `orthonormalize_into` (modified GS + defect score), `factor_activity_hinge` (Welford), `parseval_energy_check`, conditioning certificate via `spectral_pencil`. Feature-gated, GOAT (bit-identity, planted-collision + planted-dead-channel negative controls, µs/alloc).
- **riir-train Plan 351** — orthogonal factorized NextLat: K=8/r=8 heads, hinges vs `lambda_unif`, EMA target m-sweep, Muon-on-B, `CompressionMlp` dominance overlay, factor-horizon allocation; ≤1.5 GPU-hrs; GOAT vs monolithic AND vs KARC (floor rule if intervals claimed).
- **Panel discards (auditable):** No-GD #19 LatCal CORDIC 2×2 orthogonalization — no consumer pull, low confidence; No-GD #20 crowd-opinion energy accounting — diagnostic framing only, owner call; Model-based #9 quest-grammar factorized head — GOAT-FAILED family prior (bench_285 G1/G2, bench_287 G2), must beat a failed bar, ranked last and dropped.

## Honest Caveats

- **Orthogonality ≠ independence** (paper's own caveat; our ICA lens R475). All fusion items above claim geometry, never disentanglement.
- Fixed/constructed bases do **no feature discovery** — GS preserves span, Hadamard is data-agnostic. The learned half is what Plan 351 buys.
- Per-factor heads ≡ monolithic solve under shared features + shared λ — the win lives in **per-factor λ/whitening** (scale separation) or budget selection; every claim must carry the matched-param control.
- Concurrent-work risk is high (MoP-JEPA, FAR-JEPA both ~8 weeks); cite carefully.
- Any interval/coverage claim from per-factor conformal (Plan 351 optional item) is UQ-bearing → **Report-the-Floor rule** (`ConformalIntervalCalibrator<SeasonalNaiveForecaster>`, Plan 340); honest prior: may lose on periodic slices (KARC+overlay precedent).

> **PASS-Redirects (synthesis):** Ren, Wang, Hoang, Kuang, LeCun [AGL-TR-2026-04 "The Content–Geometry Tradeoff in World Models: When JEPA Learns Latent Maps"] — the sibling geometry-recipe distill on the same NextLat training axis this note opened with Plan 351: SVReg (one-sided variance floor, no SG) + slow channels + gated residual predictor extend the recipe family; the coupling cells absorb them (riir-train Plan 507). Modelless track PASS — the per-coordinate floor gap this note's reverse-grep found is FILLED by #687 (Bench 676), so the new paper carries no unfilled modelless gap. See [Research 686](686_Content_Geometry_Tradeoff_JEPA_Latent_Maps.md).
