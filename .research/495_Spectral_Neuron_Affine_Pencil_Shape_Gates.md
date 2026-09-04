# Research 495: The Spectral Neuron — Affine Matrix Pencil as a Shape-Guaranteed, Coefficient-Transparent Scalar Gate

> **Source:** Alex Shtoff (TII) — "The Spectral Neuron" — [arXiv:2608.08003](https://arxiv.org/abs/2608.08003) [stat.ML], Aug 2026, 32 pp.
> **Code:** [github.com/alexshtf/spectral_neuron_paper](https://github.com/alexshtf/spectral_neuron_paper) (Python/PyTorch, BSD-3, 9★ — experiments only; `src/paper/models.py` holds the model; uv-locked, Python 3.14).
> **Date:** 2026-08-21
> **Status:** DISTILLED — pending owner decision (Gain verdict; Super-GOAT re-gate armed on the riir-ai PoC, Issue 736)
> **Classification:** Public — open-primitive layer (generic symmetric-eigenvalue math; no game/chain/shard IP)
> **Related Research:** [466](466_CPd_minus_1_Hopfield_Top_Eigenvector_Recall.md) (cp_hopfield — closest structural cousin: matrix from memories, top-eigenvector output vs matrix from input, eigenvalue output), [451](451_Delta_Lattice_Tunneling_Transfer_Matrix_Band_Structure.md) (band-structure analyzer — the verdict-discipline template this note follows), [039](039_SpectralQuant_Calibrated_Eigenbasis_KV_Compression.md) (calibrated eigenbasis, dataset-level), [246 MPI router](246_Manifold_Power_Iteration_MoE_Router.md) (Rayleigh-quotient ascent), [053](053_CNA_Contrastive_Neuron_Attribution.md) / [491](491_Sterling_Additive_Concept_Attribution_Steering.md) / riir-ai [009](../../riir-ai/.research/009_MDA_Mechanistic_Data_Attribution.md) (attribution family), [269](269_Variable_Width_Shape_Adapter_Fusion.md) (width-shape adapters — different "shape"), [307](307_FNO_Practical_Perspective_Spectral_Primitives_Survey.md)
> **Related shipped code:** `katgpt-core/src/cp_hopfield/` (`hermitian_top_eigenvector` — power iteration + Rayleigh quotient, reusable solver substrate), `katgpt-core/src/committed_field_blend.rs` (`ArchetypeFieldSource::lipschitz_bound` — hand-constant Lipschitz certificates this paper would upgrade), `katgpt-spectral/src/hla_eigenbasis.rs` (window eigenbasis), `katgpt-core/src/conformal/floor_harness.rs` (the UQ floor the eigengap-confidence items must beat), riir-mmorpg-examples Issue 078 / Bench 027 (hero attribution ledger + FusionArm rig — the consumer-side PoC surface)
> **Issues filed:** katgpt-rs `676` (open primitive), katgpt-rs `678` (Lean 4 package — blocked on 676), katgpt-rs `679` (owner ruling: LP-anchor compile), riir-ai `736` (consumer PoC + Super-GOAT re-gate), riir-train `472` (training-track recipe fragments). The Sturm-at-chain-seam item stays deliberately unfiled (speculative — no Glacial-rate consumer exists; reopen from this note if one materializes).

---

## TL;DR

The paper proposes a scalar model **f(x) = λk(A₀ + Σᵢ₌₁ⁿ xᵢ·Aᵢ)** — the k-th smallest eigenvalue of an affine symmetric matrix pencil. The input enters linearly into a matrix; the nonlinearity is reading one ordered eigenvalue. The claimed middle ground: expressivity that grows with matrix dimension d **while retaining linear-model-style transparency**:

1. **Global feature-influence bound** (Weyl): `|f(x+δ)−f(x)| ≤ Σ|δᵢ|·‖Aᵢ‖₂`. Per-feature worst case, closed-form from coefficients — the "coefficient magnitude" of the matrix age.
2. **Shape control by construction**: k=1 → concave, k=d → convex (Rayleigh–Ritz); `Aᵢ ⪰ 0` → non-decreasing in xᵢ, `Aᵢ ⪯ 0` → non-increasing (Loewner/eigenvalue monotonicity). Mixable per feature; composable with convexity/concavity via k.
3. **Exact local attribution** (Hellmann–Feynman): at simple eigenvalues `∂f/∂xᵢ = vᵀ Aᵢ v`; at repeated eigenvalues a bound `‖Vᵀ Aᵢ V‖₂` (Clarke subdifferential).
4. **Initialization theory** (the closed-form crown jewel): `A₀ = Qᵀdiag(−1,…,−1, 0@k, 1,…,1)Q` (Q random orthogonal), `Aᵢ = αᵢI + diag(εᵢ)`, α~U(±1/√n), ε~U(±1/(20n)) ⇒ **eigengap γk ≥ ½ proven by Weyl whenever ‖x‖∞ ≤ 5**, and the jitter makes A₀Aᵢ ≠ AᵢA₀ a.s. (avoids the simultaneous-diagonalization trap where gradient methods keep the model a piecewise-linear order statistic forever).
5. Universality (interior k) via Cook et al. 2025 PMM; DC decomposition λk = Λk − Λk+1 (Ky Fan); orthogonal invariance ⇒ one matrix diagonal w.l.o.g.; invertible monotone element `g(x)=λk(A+xB)`, `B ≻ 0`, with closed-form inverse `g⁻¹(z) = λ_{d−k+1}(B^{−1/2}(zI−A)B^{−1/2})` (verified independently by the No-GD panel via an inertia argument).
6. Trains with plain Adam + autograd-through-`eigh`; scales on synthetic splines, Criteo (competitive with FMs), HIGGS (competitive-with-behind MLPs). Not SOTA — the value is the transparency/shape/scaling triple.

**Verdict: Gain** (not Super-GOAT — see §6). The modelless content is dense (10 of the paper's 11 content blocks need no gradient step) and genuinely novel to this stack (verified: zero in-workspace prior art for the pencil model, its shape DSL, Hellmann–Feynman attribution, eigengap init, Sturm counting, sym-packing, squareplus). But the **model family itself is published prior art** — Cook et al., *Parametric Matrix Models*, Nature Comms 2025 ([arXiv:2401.11694](https://arxiv.org/abs/2401.11694)) contains the construction; spectrahedral regression (O'Reilly & Chandrasekaran 2023) covers extremal-k; maxout (2013) IS the commuting/diagonal case. Our novelty lives in the **compositions** (per-NPC seeded personality genomes, certificate-backed Lipschitz safety composition, Sturm integer predicates at the chain seam), whose behavioral value (Q2/Q3) is unproven until the Issue 736 PoC — exactly the Research 451 downgrade pattern.

---

## 1. Paper core (compressed)

- **Continuity/Lipschitz**: eigenvalues are 1-Lipschitz in spectral norm (Stewart–Sun). Corollary 1 gives the per-feature global bound. Suggested Lasso-analog: regularize `Σ‖Aᵢ‖₂` for feature selection.
- **Convexity**: λmin concave / λmax convex (min/max of linear forms in A); extremal-k universality for convex/concave on compacta (O'Reilly & Chandrasekaran 2023); interior-k universality for ALL continuous functions (Cook et al. 2025).
- **Orthogonal invariance**: pencil simultaneously conjugated represents the same function ⇒ canonical gauge with one matrix diagonal.
- **Latent-variable reading** (Courant): λk = max_C min_{u⊥C,‖u‖=1} uᵀA(x)u — a min–max game; also a "recurrent" reading (Claim 4: each eigenvector constrains the next eigenvalue's variational problem).
- **Monotonicity**: PSD perturbations push every λk up (Loewner). Parametrization: `Aᵢ = LᵢLᵢᵀ` (PSD) or diagonal `diag(squareplus(v))` — squareplus(x) = (x+√(1+x²))/2 chosen over softplus because its gradient decays polynomially (1/(4x²)) not exponentially, so deeply-negative parameters stay trainable.
- **sym(v) packing**: off-diagonals ×1/√2 ⇒ ‖sym(v)‖_F = ‖v‖₂ exactly (and ⟨sym(u),sym(v)⟩_F = ⟨u,v⟩) — avoids an accidental basis-dependent preconditioner under Euclidean optimizers.
- **Differentiation**: Clarke subdifferential ∂λk = conv{xxᵀ : x ∈ eigenspace, ‖x‖=1}; gradient vvᵀ at simple eigenvalues — autograd-compatible (PyTorch `eigh`), or SciPy single-eigenvalue solvers + hand differentiation.
- **Costs**: inference nd² + 4/3·d³ FLOPs (eigenvalues); training nd² + 14/3·d³ (eigenvectors). d ≈ 5–30 in experiments.
- **Hyper-network application**: encoder net predicts (A, B⪰0) from context; `λk(A + bid·B)` is a monotone-in-bid CDF for auction bid shading by construction (generalizes Zhou et al. 2021).
- **Future work it names**: symmetric tridiagonal pencils (O(d) params, fast eigensolvers), multi-PSD parametrizations, monotone invertible flow elements, spectral-norm regularization.

## 2. §3.5 Path 0 decomposition (two questions per component)

| # | Component | Coverage (ships?) | Extraction (modelless?) |
|---|---|---|---|
| 1 | Eigen-solver machinery (top eigenvalue) | **YES** — `cp_hopfield::hermitian_top_eigenvector`, `beta_fitter` power iteration, `zone_manifold` deflate chain, `hla_eigenbasis`, DEC `hodge` eigendecomposition, SpectralQuant calibration eigh | YES (shipped) |
| 2 | **Interior-k single eigenvalue kernel** (bisection/QL/Jacobi) | **NO** — power-iteration substrate is extremal-biased; deflation to interior k is O(k) matvec chains; nothing ships Sturm/QL/bisection | **YES** — new kernel; tridiagonal + Sturm-count bisection ≈ 50·d ops/eigenvalue, O(d) exact integer counts below threshold |
| 3 | **The pencil model** λk(A₀+ΣxᵢAᵢ) as a decision function | **NO** (verified: grep `pencil\|spectral neuron\|A0 +` → zero model hits) | **YES** — pure evaluation of constructed matrices |
| 4 | **Global influence bounds** ‖Aᵢ‖₂ | **PARTIAL** — `CommittedFieldBlend::lipschitz_bound` ships hand-constant per-field bounds (FAME Lemma 1 safety composition); signal-diff: hand-claimed constants on FIXED vector fields vs per-feature bounds DERIVED from coefficients of an adaptive scalar function | **YES** — closed-form from construction |
| 5 | **Shape DSL** (k⇒convex/concave; Aᵢ⪰0⇒monotone) | **NO as a primitive** — monotonicity is desired + empirically test-pinned (`quantize_mood_monotone`, `hla_bucket_monotone_in_curiosity_drive`, `zone_mood_projection_monotone_in_dot`, `borrow_monotone_in_movement`) but achieved by hand-tuned sigmoid ladders, not by construction | **YES** — definiteness constraints are construction-time |
| 6 | **Hellmann–Feynman attribution** ∂f/∂xᵢ = vᵀAᵢv | **NO** — hero attribution ledger (Bench 027) explains goal selection post-hoc; signal-diff in §4.3 | **YES** — one quadratic form per feature per eval |
| 7 | **Eigengap confidence** γk (Davis–Kahan ⇒ attribution trust + decision stability ~1/γ) | **NO** (`spectral_gap` exists only in SpectralQuant CalibrationResult — a data-spectrum diagnostic, different signal) | **YES** — runtime certificate from the same solve. **UQ-bearing ⇒ conformal-floor gate mandatory** (floor ships at `katgpt-core/src/conformal/floor_harness.rs`) |
| 8 | **Certified box→interval** (monotone pencils: A(lo)⪯A(x)⪯A(hi) ⇒ 2 solves bound f) | **NO** | **YES** — UQ-bearing ⇒ floor gate |
| 9 | **Init constructor w/ eigengap ≥ ½ + non-commutativity certificate** | **NO** (`new_spectral` shard init is unrelated vocabulary) | **YES** — THE zero-training extraction: seeded construction is a provably-conditioned nonlinear function generator |
| 10 | sym(v) 1/√2 isometric packing | **NO** | **YES** — makes every attribution/similarity query a plain SIMD dot on packed vectors |
| 11 | squareplus positivity parametrization | **NO** (softplus ships: KDA decay gates, WallDiagonalGate) | **YES** — also a training-side drop-in (riir-train 472) |
| 12 | Loewner laws + mirror duality (λk(−A)=−λ_{d−k+1}(A): convex↔concave for free) | **NO** (Courant/Ky Fan/Loewner name-grep: zero) | **YES** |
| 13 | Courant–Fischer certificate ladder (anytime one-sided bounds) | **NO** (anytime-refinement PATTERN exists in MCTS substrate) | **YES** |
| 14 | Invertible monotone warp + closed-form inverse | **NO** (bridge functions are dot+sigmoid projections / clamps, not bijections) | **YES** — both directions one eigen-solve each |
| 15 | DC decomposition / Ky Fan | **NO** | YES (metadata; used by #8's tighter variants) |
| 16 | Tridiagonal pencil variant (O(d) params) | **NO** | **YES** — the paper's future work is immediately licensable (norm bounds are sparsity-blind; Weyl argument survives; rescale ε) |
| 17 | Training loop (Adam through eigh), scaling experiments | NO (irrelevant — track c) | NO — riir-train 472 owns the recipe |
| 18 | Universality theorems | N/A | existence metadata — licenses the family, no code |

**Funnel result: rows 2–16 have no shipped analog ⇒ MODELLESS-VALIDABLE core.** Row 17 routes to riir-train (Path 0.5) — the recipe is applicable at trivial cost (~8–15 GPU-h total program, M3-first) but is NOT a blocker for anything modelless. This is the rare paper where Path 0 and Path 0.5 BOTH land.

## 3. Adversarial panel synthesis (mandatory — training-adjacent abstract)

Two advocates ran in parallel (No-GD 41-item extraction; Model-based 14-item extraction). Merged and curated below; discards carry one-line auditable reasons.

### 3.1 Admitted — modelless track (katgpt-rs → consumers)

**P0 (one module, one feature flag `spectral_pencil`, one bench — Issue 676):**
- `sym` isometric packing (+ inner-product preservation ⇒ every query is a SIMD dot on packed fixed-size arrays).
- Dense small-d single-eigenvalue kernel (pinned cyclic-Jacobi / fixed-iteration, deterministic) + **tridiagonal pencil + Sturm-count bisection** (any eigenvalue to f32 precision ≈ 50·d ops; **exact integer eigenvalue-counts below threshold in O(d)**).
- Seeded init constructor (`A₀` = conjugated ladder, `Aᵢ = αᵢI + diag(εᵢ)`) with the γk ≥ ½ guarantee property-tested + non-commutativity certificate `‖[A₀,Aᵢ]‖_F > 0`.
- Global bound helpers (‖Aᵢ‖₂, linear growth envelope |f(x)| ≤ ‖A₀‖+Σ|xᵢ|‖Aᵢ‖).
- Hellmann–Feynman attribution `vᵀAᵢv` (+ subdifferential interval at repeated eigenvalues; γk flags low-trust attribution).
- Deterministic pinned-evaluation policy (fixed-iteration bisection for any committed readout — no library QR rotation-order variance).
- Shape DSL constructors (PSD/NSD per-feature, k index; mirror duality).
- Rank-one fast path (`Aᵢ = βᵢdᵢdᵢᵀ` over shipped BLAKE3 direction vectors — the matrix lift of the existing dot-projection idiom; gradient = βᵢ(vᵀdᵢ)²).

**P1 (differentiating compositions — after P0):**
- Committed Lipschitz composition: spectral gate as `ArchetypeFieldSource` with **certificate-backed** `lipschitz_bound()` (upgrades FAME Lemma 1 inputs from hand constants to derived bounds).
- Monotone decay-to-baseline dynamics: decayed inputs + PSD Aᵢ ⇒ provably monotone glide to the temperament baseline λk(A₀) — "beliefs fade, never deleted" with an overshoot proof (the Issue-070 fear-lock class structurally impossible for these gates).
- Per-NPC seeded gate genomes (seed = BLAKE3(npc_id ‖ world_seed)); canonical-gauge stable bytes ⇒ BLAKE3-committable; fixed-size Pod layout beside the neuron-db spectral-init module.
- Genome similarity = packed Frobenius inner product ⇒ social KG-triple proximity (the social-domain rule instantiated on gate genomes).
- Deterministic genome merge (elementwise mean) + Weyl health certificate: γk(avg) ≥ γk(p₁) − ‖p₂−p₁‖₂.
- Invertible monotone warp family (`g(x)=λk(A+xB)`, B = I + Σβᵢdᵢdᵢᵀ; closed-form inverse) — the first *provably bijective* bridge element in the stack.
- Certified box→interval readout + Lipschitz tamper check (committed (x,f) pairs; violation = physically-impossible transition) — **both UQ-bearing: conformal-floor gate**.
- Curiosity-at-kinks: `sigmoid(−γk/τ)` as a zero-training exploration signal (cgsp consumer; A/B vs flat-curiosity).
- Temperament k-index: k=1 pessimist (any-direction veto) … k=d optimist; runtime re-index = explainable mood swap with exact attribution delta.

- **P2 (owner-ruling / later):**
- Sturm integer predicates at the chain seam ("≥ j modes below θ" — exact, platform-stable, tamper-evident; consumed via riir-dapps at Glacial rates only; most outcomes settle nothing — `Settlement::None` stays the default) — **deliberately unfiled** (no Glacial consumer exists; reopen from this note if one materializes).
- LP anchor interpolation on the commuting subclass (order-statistic-of-affines fitting is quantile-curve LP) — **RULED NOT LEGAL for the modelless track** (owner ruling, 2026-08-24, Issue 679): an LP fit is still a fitting procedure producing embedded data, and pinned-solver bit-determinism is fragile across platforms; the conservative route is doc-only — any real fitting demand routes to riir-train `472` (trained heads). Non-commuting pencils by any solver remain riir-train territory under all rulings. No code on this route.
- Seeded property-test search (sample 676 constructions until one passes a property test; deterministic given seed + test) — unambiguous (same seed → same bytes), unaffected by the ruling.
- Lean 4 package: sym-isometry, Weyl 1-Lipschitz, Loewner monotonicity, **the constructive eigengap bound** — static matrix algebra, fits the FV doctrine (public items → `KatgptProof`; bump `EXPECTED_THEOREMS`; paired SpecTests + Rust spec_match + negative perturbations) — **Issue 678**.

### 3.2 Admitted — model-based track (riir-train 472; frozen-Pod shipping complies with the mandate's sanctioned weight states)

- sym(v) packing for ANY symmetric-matrix parameters under Euclidean optimizers (anisotropic implicit trust-region fix; ~0 GPU-h).
- squareplus for every positivity site (variance heads, GRPO temps, PSD diagonals; polynomial gradient decay keeps deep-negative params trainable; ~0.1 GPU-h).
- Eigengap-guarding init + non-commutativity jitter as general eigh-differentiation insurance (any future backprop-through-spectrum code: spectralquant calibration, shard spectral init).
- Hero fusion 4th arm on the EXISTING Issue 078 / Bench 027 rig: σ(λk(A₀+Σ goalᵢ Aᵢ)) — k is a continuous learnable interpolation between max-like (k=d) and mean-like (mid-k) aggregation, the design space Issue 078 could only probe at 3 discrete points.
- Monotone auction/bid-shading CDF head (the paper's flagship application; A_bid ⪰ 0 ⇒ CDF by construction) — floor-rule-adjudicated.
- Optimizer canary (closed-form-verifiable gradients, degenerate-crossing landscape; sub-second runs).
- Tridiagonal trainable edge variant (params below dense d=16 at d=32; flagged unproven in paper — honest risk).

### 3.3 Discarded (auditable reasons)

- **GPU eigh + vvᵀ backward kernel family** (model-based #7) — deferred: nothing admitted trains eigenvalues on-device yet; promoted to a line inside Issue 472 rather than a work item (kernel arrives with the first real consumer).
- **KARC spectral re-gate** (model-based #9) — discarded as a standing item: the T7 structural scope-limit (Chebyshev/ridge can't fit periodic) plausibly transfers to piecewise-analytic eigenvalue functions; the floor re-gate is already mandatory separately. One-line probe may ride along in 472; not a work item.
- **Shard-spectral readout heads** (model-based #14) — discarded: speculative (no paper evidence for the composition); the modelless genome-Pod (P1) delivers the shard bridge without training.
- **MLP→spectral distillation lane** (model-based #13) — discarded for now: no identified teacher head in-stack justifying it; revisit if a trained black-box scalar head ships.
- **No-GD #25 deterministic committee ensembles** — folded into P1 genomes (a population IS the committee).
- **No-GD #29/#30 curiosity/zone consumers** — kept but demoted to PoC arms inside Issue 736 (need G8 evidence before primitive status).
- **Courant ladder as a standalone primitive** (P2 in the brief) — folded into the kernel module as an anytime API on the existing MCTS-style refinement pattern; not separately gated.

## 4. Signal-diff checks (§3.6 — every "partial coverage" row)

1. **vs `cp_hopfield` (466)**: cp_hopfield builds `K = Σ_μ O_μ|ξ_μ⟩⟨ξ_μ|` from STORED MEMORIES (fixed kernel) and outputs the top **eigenvector** (state alignment; capacity via BBP gap). Spectral neuron builds `A(x)` from the LIVE INPUT and outputs the k-th **eigenvalue** (scalar + attribution + shape). Consumed signal: memory overlaps vs live features. Product: recall state vs decision scalar. **Different mechanisms; complementary.** Solver reuse: `hermitian_top_eigenvector` covers k∈{1,d} on small d; interior k needs the new Sturm/QL kernel (power iteration is extremal-biased).
2. **vs `CommittedFieldBlend::lipschitz_bound`**: shipped = hand-claimed constants (`scale.abs()`, `1.0`, `0.0`) on FIXED archetype vector fields, composed via FAME Lemma 1 into a safety bound. Paper = per-feature bounds DERIVED from coefficient matrices of an input-adaptive scalar function. Fusion upgrades the composition's inputs from claims to certificates; the composition law itself is reused as-is.
3. **vs hero attribution ledger (Bench 027)**: ledger decomposes the goal-selection arithmetic post-hoc (which goal won, fractional contributions) for ONE hand-built motivation ladder. Hellmann–Feynman gives closed-form per-INPUT-FEATURE influence for ANY function in the class, valid at every input, with a trust certificate (γk). Ledger explains goal choice; spectral explains feature influence on a scalar. Complementary — the ledger is the consumer surface a spectral head would plug into.
4. **vs `hla_eigenbasis`**: spectrum of an affect WINDOW (data-level personality characterization, energy ratios). Spectral neuron: eigenvalue AS the decision function of live inputs. Fusion hook: window spectra can calibrate Aᵢ (personality-informed construction).
5. **vs SpectralQuant `spectral_gap`**: gap of a calibration-time covariance spectrum (data diagnostic). Eigengap γk here: gap of the live input-conditioned pencil at the current decision — a runtime stability/trust certificate. Different signal, different time constant.
6. **vs monotone test-pins (`quantize_mood_monotone` etc.)**: tests assert a property of hand-tuned maps on sampled grids. The DSL makes the property structural (PSD ⇒ monotone everywhere, proof carries). The tests remain as regression guards for the legacy maps; the DSL removes the need to hand-verify new maps.

## 5. Fusion (the novel combination)

**Spectral personality genomes × CommittedFieldBlend safety composition × hero attribution ledger × freeze/thaw shards:**

A per-NPC gate genome (seeded construction, γk ≥ ½ certificate, canonical-gauge stable bytes) becomes an `ArchetypeFieldSource` whose `lipschitz_bound()` is a derived certificate (‖Aᵢ‖ envelope) rather than a hand constant; its decisions feed the existing attribution-ledger UI with exact per-feature signed influences (vᵀAᵢv) + trust flags (γk); the genome persists as a fixed-size Pod through the existing freeze/thaw envelope; population diversity ships as a BLAKE3-seeded generator instead of hand-tuned constant tables; crowd behavior gets the temperament ladder k=1..d (pessimist→optimist over shared evidence matrices — an eigen-structural CLR sibling: ΣPSD evidence raises every λk by Loewner). None of the four pillars alone produces "every NPC decision provably attributable, shape-guaranteed, and tamper-checkable"; the pencil is the missing connective tissue. Additionally the tridiagonal+Sturm form gives the chain seam **exact integer predicates** (eigenvalue-count-below-θ) — the only quorum readout class with zero float cross-platform drift.

**Cost arithmetic** (advocate-verified): dense d=16 ≈ 8K FLOPs/eval; tridiagonal d=16 ≈ 800; 10,000 NPCs × 20 Hz ⇒ 160 MFLOP/s–1.6 GFLOP/s — SIMD-trivial; attribution = n packed dots. Placement: Plasma/Hot (eval + attribution), Warm (genome refits), Cold (genome archive), Glacial (committed integer predicates only).

## 6. Novelty gate (§1.5) — honest scoring

- **Q1 (no prior art?): PARTIAL.** In-workspace: verified zero (dual-vocab greps: pencil/spectral neuron/eigengap init/Sturm/Courant/Ky Fan/Loewner/squareplus/sym-packing; every Rayleigh/power-iteration hit is a different mechanism — §4). Literature: **the model family is published** — PMM (Cook et al. 2025, the paper's own acknowledged frame), spectrahedral regression (2023), maxout/order-statistics (2013/2014), monotonic networks/lattices/ICNN (the shape-constrained lineage the paper itself surveys). Our claims must therefore be compositions, not the model. No art found for: seeded eigengap-guaranteed construction as a function GENERATOR, Sturm integer predicates for quorum, certificate-backed Lipschitz composition into a committed blend, per-NPC personality genomes. (Two web searches run; a Super-GOAT claim would need per-fusion searches.)
- **Q2 (new behavior class?): TBD.** Explainable-by-construction + shape-guaranteed + k-temperament is a plausible new class, but like Research 451's honest call: until a consumer converts the structure into measured emergent behavior (G8 A/B), it is a capability *hypothesis*. Issue 736 is the converter.
- **Q3 (product selling point?): TBD.** "Every NPC decision ships a per-feature signed influence certificate and a certified worst-case bound; fear→flee is monotone by construction, not by tuning" is finishable — but unproven demand; GM-tool value is the strongest near-term form.
- **Q4 (force multiplier?): YES.** Connects katgpt-spectral solvers, CommittedFieldBlend, direction vectors, attribution ledger, cgsp curiosity, freeze/thaw shards, conformal floor, riir-train recipe lanes. ≥2 pillars trivially.

**Not all 4 unambiguous YES ⇒ Gain, Super-GOAT re-gate armed.** Trigger: Issue 736 PoC demonstrates (a) structural properties hold at population scale (G1 eigengap/monotone/attribution-exactness property tests), and (b) at least one G8 behavioral differentiation vs the incumbent dot+sigmoid gates. If both land → re-run the gate with per-fusion prior-art searches; expect Q1 (compositions), Q2, Q3 to flip.

## 7. Routing + priorities

| Priority | Item | Repo | Vehicle |
|---|---|---|---|
| P0 | `spectral_pencil` open primitive (packing, dense+tridiag+Sturm kernel, seeded init, bounds, attribution, shape DSL, rank-one path, pinned evaluation) | katgpt-rs (katgpt-core, opt-in flag) | Issue 676 |
| P1 | Consumer PoC: personality genomes + certificate Lipschitz + ledger integration + curiosity/temperament A/Bs + Super-GOAT re-gate | riir-ai (riir-poc → consumer) | Issue 736 |
| P1 | Training recipes: sym packing, squareplus, eigengap init, hero 4th fusion arm, auction CDF probe | riir-train | Issue 472 |
| P2 | Sturm chain predicates (via riir-dapps, Glacial-only, `Settlement::None` default — unfiled by design), seeded property-test search, Lean package (Issue 678). LP-anchor compile **ruled out of the modelless track** (Issue 679, 2026-08-24 — fitting demand → riir-train 472) | katgpt-rs / riir-chain / .proofs | 678 |

## 8. Validation protocol (GOAT sketch — full gates live in the issues)

- **G1 correctness**: property tests over frozen seed sets — γk ≥ ½ on the input box; monotone sweeps per PSD/NSD feature; attribution vs central finite-difference (test-only FD) at simple eigenvalues; subdifferential interval encloses one-sided limits at constructed degeneracies; rank-one path == dense path bitwise; Sturm count == full-solve count on 10⁶ random tridiags; warp round-trip g⁻¹(g(x)) == x.
- **G2 perf**: ns-scale per eval at d ∈ {8,16,32} (dense vs tridiag); tick-budget headroom at 10k NPCs × 20 Hz; construction amortized at spawn/freeze.
- **G3 no-regression**: opt-in flag; default builds untouched; boundary-guard clean (scalar-only outputs; think-brain side only).
- **G4 alloc-free**: fixed-size packed arrays, caller-owned scratch, zero steady-state allocs (counting allocator).
- **UQ floor rule** (mandatory for γk-confidence and box→interval items): beat `ConformalIntervalCalibrator<SeasonalNaiveForecaster>` on CRPS/coverage/Winkler, or scope-limit honestly (the KARC precedent — calibrated coverage with a documented scope-limit is acceptable; an unmeasured confidence claim is not).
- **Determinism**: pinned-algorithm evaluation for any committed readout; canonical gauge for stable commitment bytes; cross-architecture equality for integer Sturm predicates.
- **Domain discipline**: all gates consume think-brain beliefs; only scalar outputs cross any seam; no latent gate ever validates movement claims (Lipschitz tamper check reads committed raw pairs only).

## 9. What does NOT extract (honest boundaries)

1. The gradient training loop + scaling results (track c — riir-train 472; nothing modelless needs them).
2. Universality as code (existence theorem; licenses the family, produces nothing).
3. Any "fits data better" claim — untested here; §3.6 PoC owns that if ever made. Everything claimed above is structure/bounds/exactness/determinism — architecturally checkable without a training baseline.
4. Curse-of-dim caveat: if anyone routes this onto high-dim shard embeddings via DEC-style boundary arguments, the R296 rule applies — boundary-vs-volume wins only d ≤ 3. The pencil is per-input small-d by design; keep it that way.
5. Cross-platform float bit-identity for eigenvalue readouts is NOT claimed (pinned-algorithm determinism per-binary + lattice quantization for committed floats; integer Sturm counts are the exact class).

---

## PoC Addendum (Issue 736 Phase A, 2026-08-22)

**Status: RECORD** — Phase A defend-wrong PoC landed (riir-ai
`crates/riir-poc/src/spectral_neuron_poc.rs`, 11 gates green,
debug-marked behavioral set). Raw numbers + per-axis calls:

### Toy

100-NPC cluster (Bench-010 shape), 480 ticks, 32 world seeds, one scripted
threat (waypoint walk). Belief x ∈ [0,5]⁴: threat proximity, evidence EMA,
safety (NSD), fatigue (NSD). Shaped genome (family M): A₀ = seeded ladder
(the 676 init; 0@k simple eigenvalue, γk ≈ 1 at neutral) + rank-one-PSD
threat feature (β ∈ [0.6, 1.2] + αI) + weaker rank-one evidence + NSD
diagonals. Matched operating point: 85th-percentile threshold over a
shared reference belief distribution (same protocol for every score arm).

### A2 structural — 5/5 CONFIRMED, zero violations

| Axis | Gate | Result |
|---|---|---|
| Eigengap ≥ ½ at population scale | 10⁴ family-S seeds × 17 box points | 0 violations (the 0@k mechanism holds at consumer scale) |
| Monotone sweeps | 300 genomes × 4 features × 101 pts | 0 violations (by construction, verified) |
| Attribution = FD | 10⁴ probes, trusted-gap only | 0 violations |
| Lipschitz envelope | 10⁴ random pairs vs Σ|δᵢ|‖Aᵢ‖ | 0 violations |
| Curvature (interactions) | interaction deviation in score space | pencil interior-k > 1e-3 median; dot EXACTLY 0 (logit-linear) |

### A3 behavioral — 2 of 3 axes CONFIRMED

**(a) Temperament ladder — CONFIRMED, graded, 32/32 seeds monotone.**
Tipping medians along the strong-signal axis by k (0-indexed):
`[6.0, 6.0, 6.0, 6.0, 6.0, 5.0, 3.8, 2.2]` (6.0 = never-fires sentinel).
Exactly the interlacing prediction: rank-one PSD evidence lifts only the
top eigenvalues, so low-k NPCs attend to diffuse evidence only, high-k to
single strong signals. A GRADED behavioral axis from one genome constructor
— not a cliff.

**(b) Curiosity-at-kinks — REFUTED at toy scale (honest wash).** 16/32
wins at pre-registered τ=0.5, medians equal (138 vs 138); τ=0.25 also wash
(12/32, kink mean slightly WORSE: 128.97 vs 130.56). Mechanism: at
interior k the rank-one response is O(t²), so spatial γk variation is
dominated by the mild NSD town-distance gradient — the 4 lookahead
directions barely differ in γk. The axis may re-open with
rank-one-dominated genomes or a k-D curiosity readout; at this scale it is
refuted.

**(c) Decay-to-baseline — CONFIRMED as a recovery-lag asymmetry.** Pencil
return after threat-vanish: **0 ticks** at every T_on ∈ {20,60,120,240}
(memoryless ⇒ structurally immediate). Raw accumulator: 45 ticks flat
(the ln 2 / 0.015 fear-decay timescale, as predicted). Patience patch
(Issue-054): only helps long chases — T_on=20 → 42 ticks (patience
unexpired at vanish), T_on ≥ 60 → 6–8. No hard locks at production
constants in ANY arm (honest: at production constants the fear-lock
disease is lag, not permanent lock).

### A1 headline — the Q2/Q3 evidence

32-seed totals, arms [pencil k=4, pencil k=7, dot, accum, frozen]:

| Metric | pencil k=4 | pencil k=7 | dot (hand-tuned) | accum | frozen |
|---|---|---|---|---|---|
| damage events | 3683 | **0** | 0 | 3200 | 25700 |
| forage score | 92110 | **110466** | 110566 | 67168 | 132324 |
| flee rate | 0.255 | **0.184** | 0.204 | 0.194 | 0 |

**The load-bearing result:** the SEEDED, UNTUNED k=D−1 pencil matches the
hand-tuned dot+sigmoid incumbent on damage (0 vs 0) and forage (110466 vs
110566, within 0.1%) at a LOWER flee rate — while carrying monotone-by-
construction, γk confidence, exact attribution, and the Lipschitz
certificate, none of which the incumbent has. And temperament selection is
load-bearing behavior: the same genome family at k=4 takes 3683 damage.
Zero hand-tuning reached incumbent parity — the paper's promise, measured.

### Verdict effect

The Research 451-style Q2 capability hypothesis now has measured evidence
on 2 of 3 axes (ladder + recovery) plus parity-with-certificates on the
headline. Per Issue 736 Phase C: the Super-GOAT re-gate (per-fusion
prior-art searches: seeded eigengap personalities; certificate-Lipschitz
composition; Sturm quorum predicates) is ARMED — owner decision point.
Curiosity-at-kinks is recorded refuted at toy scale and does NOT proceed
to Phase B wiring.

### Re-gate EXECUTED — PASS (2026-08-29)

The armed re-gate ran same-window as T3's Bench 794 landing (the trigger
had long fired: structural 5/5 + 2/3 behavioral). Per-fusion prior-art
searches (arXiv full-text + DBLP + citation graphs + dual-vocab workspace
grep; Scholar unsearchable until the 2026-09-15 quota reset — caveat
recorded): all three fusion compositions ABSENT (F1 seeded eigengap
personalities; F2 certificate-Lipschitz composition through committed
gates; F3 Sturm quorum predicates — application-layer novelty only), and
the composed selling point (attributable + shape-guaranteed + personality-
certified + tamper-checkable) has no published match. §1.5 all-4-YES ⇒
**Super-GOAT PASS**. Mandatory outputs landed in the same session: the
private guide [riir-ai/.research/356](../../riir-ai/.research/356_Spectral_Certified_Decision_Stack_Guide.md)
+ rollout [riir-ai Plan 555](../../riir-ai/.plans/555_spectral_certified_decision_rollout.md).
Issue 736 resolved + removed (noise-reduction); this note's §PoC Addendum
remains the canonical raw-number record.

---

## PASS-Redirects

N/A — Gain verdict (files created: this note + Issues 676/736/472).
