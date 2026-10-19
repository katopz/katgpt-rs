# Research 625: Path Kernel — Every GD-Learned Model Is an Exemplar Superposition (Gradient-Space Similarity Lens)

> **Source:** Pedro Domingos, "Every Model Learned by Gradient Descent Is Approximately a Kernel Machine" [arXiv:2012.00152](https://arxiv.org/abs/2012.00152) (UW, 2020-11-30; 12 pp, theory, no code)
> **Date:** 2026-10-19
> **Status:** Active
> **Related Research:** 624 (Associative Memory Capacity Laws — the store side), 455 (Hebbian Kernel Memory — the construct side), 454 (HOPE capacity kernel), 024 (δ-Mem online associative memory), 324 (Trajectory Geometry — intra-trajectory geometry, signal-diffed), 238 (LoRA/Muon — trained-asset consumer side)
> **Cross-ref (riir-ai):** [Research 009 MDA](../../riir-ai/.research/009_MDA_Mechanistic_Data_Attribution.md) (EK-FAC training-data attribution — the trained-weight instance of this lens)
> **Classification:** Public

---

## TL;DR

Domingos' Theorem 1: any model learned by gradient descent (lr → 0, gradient flow), *regardless of architecture*, is approximately a kernel machine `y = Σᵢ aᵢ·K(x, xᵢ) + b` under the **path kernel** `K(x,x') = ∫_c(t) ∇_w y(x)·∇_w y(x') dt` (integrand = the neural tangent kernel), with `aᵢ` = path-averaged loss derivative and `b` = the initial model. For this workspace the paper is a **bridging lens, not a new mechanism**: our kernel-memory family (455/454/024/624) already ships the store, the construction, and the capacity math — Domingos adds that **every GD-trained asset we own (ternary `.bits`, LoRA adapters, quest-grammar heads, Bonsai) IS a member of that family**, hence readable and attributable as an exemplar superposition. The one genuinely modelless extraction is the **degenerate path kernel**: at a zero-length path (no training at all) it collapses to the tangent kernel `K(x,x') = ∇_w f(x)·∇_w f(x')` evaluated at *authored* parameters — a positive-semidefinite **response-space similarity** computable on any of our deterministic scorers by finite differences (no GD, no loss, just calculus). We do not ship that similarity channel (signal-diffed: `similarity.rs` is input-space; `subspace_phase_gate::participation_ratio` runs over latent directions, not knob-sensitivity directions). Class-level published prior art exists (Fisher kernels 1998; influence functions 2017; TracIn 2020; NTK 2018) → **verdict: Gain** (issues, not plans).

**Distilled for katgpt-rs (modelless, inference-time):**
The path kernel of an *untrained, authored* scorer is a closed-form, training-free similarity: perturb the scorer's knobs, take the resulting response vectors, dot them. Points far in input space can be near in response space (the paper's sine-maxima example, amplitude-knob form: for `f(x) = a·sin(kx)` the fingerprint is `sin(kx)`, so all maxima — `2π/k` apart in input space — have fingerprint-cosine 1; the frequency-knob form has ZERO gradient at maxima and is the zero-norm guard's fixture, not the demo). This is the exact space our exemplar-comparison machinery (retrieval KNN, corpus routing, abstention density) should ALSO be reading — currently it reads only input space. ⛔ Collapse caveat: the channel is signal-bearing only where the scorer is nonlinear in its knobs (§2).

---

## 1. Paper core findings

**Theorem 1 (path-kernel identity).** For `y = f_w(x)` learned by GD with loss `L = Σᵢ L(y*ᵢ, yᵢ)`:

```
lim(ε→0)  y = Σᵢ₌₁ᵐ aᵢ·K(x, xᵢ) + b
K(x, xᵢ)  = ∫_c(t) ∇_w y(x)·∇_w y(xᵢ) dt          (path kernel; c(t) = the GD trajectory)
aᵢ        = path-average of −∂L/∂yᵢ weighted by the tangent kernel
b         = y₀ (the initial model)
```

Proof is one page of calculus on the gradient-flow ODE `dw/dt = −∇_w L` — the chain rule rearranged. Remarks worth keeping:

- **R1:** `aᵢ`, `b` depend on the query; still SVM-like ("examples the loss was more sensitive to get more weight").
- **R4:** regularization adds a closed-form term to `b` only.
- **R5:** SGD → stochastic integrals; extension to Adam/momentum is open.

**The seven implications the author draws** (the lens): (1) weights = superposition of training examples in gradient space → interpretability = exemplar matching; (2) gradient-space similarity combats the curse of dimensionality; (3) superposition storage — Gram matrix and exemplar store both eliminated, storage/matching cost independent of example count; (4) GD = boosting with tangent-kernel weak learners (weights = loss derivatives, model weight = learning rate); (5) every GD-learned probabilistic model is kernel density estimation; convex solutions are kernel machines; (6) architecture knowledge (CNN translation invariance, transformer attention) ports into kernels; (7) brittleness away from training instances is *expected* kernel-estimator behavior.

**Caveats (load-bearing for honesty, from the follow-up literature + the paper's own regime):**
- The identity is exact only in the infinitesimal-lr regime; at practical lr with Adam/mini-batches it is a lens, not a proof (paper R5).
- The kernel-vs-feature-learning line shows deep learning is NOT merely its NTK at practical widths: Fort, Dziugaite, et al., "Deep Learning versus Kernel Learning: An Empirical Study of Loss Landscape Geometry and the Time Evolution of the Neural Tangent Kernel" (NeurIPS 2021) measures kernel evolution; Baratin et al., "Neural Networks as Kernel Learners: The Silent Alignment Effect" [arXiv:2111.00034]; and Hui, Belkin, Mallinar, "The Impossibility Theorem of Kernel Ridge Regression" (2020) shows a *fixed* kernel cannot both interpolate and generalize. Consequence for us: the superposition reading of our *trained* assets is approximate, and a run pinned at the kernel frontier is telling us data is the lever, not epochs (§4).

---

## 2. Distillation — Path 0 inventory

The paper has no training loop to steal; the value is the math. Decomposition + modelless coverage (§3.6 signal-diff on every "covered" row):

| # | Paper component | Shipped analog (signal it consumes) | Signal diff verdict | Path-0 disposition |
|---|---|---|---|---|
| C1 | Path kernel identity | — (no response-space kernel anywhere) | gap | **EXTRACTED** → E1/E4 (issue 949) |
| C2 | Weights = exemplar superposition | Hebbian-MLP 455 (`MLP(z) ≡ Σᵢ vᵢ·K(kᵢ,z)`, construct-side); δ-Mem 024 (online write-side); NeuronShard style_weights (store-side) | coverage is the CONSTRUCTION direction; Domingos is the recognition direction ("GD constructs it too") → extends the family to trained assets, no new mechanism | Lens note (this file); MDA 009 owns the trained-weight attribution tooling |
| C3 | Gradient-space similarity (anti-curse) | `similarity.rs` cosine/dot — **input space**; `indicator_similarity.rs` — input-space indicators | different signal (inputs vs response-variation) | **EXTRACTED** → E4 (rides issue 949) |
| C4 | Sensitivity spectrum / effective rank | `subspace_phase_gate::participation_ratio` — PR over **latent subspace** directions; `latent_trajectory_geometry.rs` — intra-trajectory geometry | PR machinery ships, but over a different object (latent Gram vs knob-sensitivity Gram); cross-query gradient alignment vs intra-path self-geometry both absent | **EXTRACTED** → E3 (new object, reused math; rides issue 949) |
| C5 | GD = boosting (loss-derivative re-weighting) | `KernelExpertRouter::pick_domain` routing weights; sigmoid calibration — but weights are hand/EMA-set, never loss-derivative-derived | partial (same shape, different coefficient source) | Recorded → E5/E11 (small closed-form orderings; folded into issue 949 scope, optional arm) |
| C6 | GD probabilistic model = KDE | `CorpusDistanceGate` (reflex) = KDE over **input embeddings** | coverage in input space; response-space density is the gap | Recorded → E6 (NOT filed: gated on riir-reflex Issue 093's measured drift evidence — filing an unmitigated-need gate now would be speculative; re-file if 093's drift read lands) |
| C7 | Brittleness = kernel reach | reflex abstention threshold; no nearest-anchor distance in any space | partial | Recorded → E8 (same gating as C6) |
| C8 | Architecture → kernel porting | simhash corpus centroid (455/A3 signed fold), zone embeddings | no concrete consumer for orbit-factored kernels beyond formatting (which `basis_cert` already measured NEGATIVE for transferable formatting subspaces, Bench 115) | **Discarded** — auditable reason: the one concrete consumer class (formatting-orbit kernels) was measured not to transfer (riir-refine Bench 115); no second consumer named |

**The degenerate-path trick (why this is modelless at all).** Set `c(t) ≡ w₀` (zero-length path — no GD ever ran): the integral collapses to its integrand. `b = y₀` is the authored model and appears as the base term anyway. Everything below lives at "step zero": differentiating a frozen deterministic function is analysis, not training; for sigmoid-gated scorers the derivative is closed-form (`σ'` computed with the SAME `fast_sigmoid` the scorer itself uses; house law: sigmoid, never softmax — this extraction is sigmoid-native); finite differences (central, pinned `h`, sequential `dot_f32_ordered` fold) cover opaque scorers.

⛔ **The collapse caveat (verdict round 1, load-bearing).** For any scorer LINEAR in its knobs, `f = σ(w·φ(x))`, the normalized fingerprint kernel is algebraically the input cosine: `∇_w f = σ'(w·φ(x))·φ(x)` and the scalar `σ'` cancels on normalization. This is an identity, not noise. New signal exists ONLY where knobs enter nonlinearly — stacked gates, a knob inside a composition, multiplicative knob coupling, per-stage gains. Issue 949 is scoped accordingly: the linear case is retained as a G1 correctness oracle ("fingerprint cosine == input cosine" is the known answer that proves the kernel code), and every signal fixture is knob-nonlinear. Companion guard: a zero-norm fingerprint (e.g. a frequency-knob `sin(wx)` scorer at its maxima, where `∇_w = x·cos(wx) = 0` — the sine example is only valid under the amplitude knob `a·sin(kx)`) must abstain with a sentinel, never NaN.

### The extracted primitive set (E-numbers from the No-GD advocate, merged)

- **E1+E4 — sensitivity-fingerprint kernel.** `S(x) = ∇_w f_w(x)|_{w₀} ∈ ℝ^p` over the scorer's `p ≤ 64` authored knobs; `K(x,x') = ⟨Ŝ(x), Ŝ(x')⟩` (normalized). PSD by construction. Consumers: a second-axis similarity beside cosine in `similarity.rs`; KNN retrieval re-rank; `pick_domain` regime signal. Includes E4's diagnostic: Spearman ρ between input-space distance and fingerprint similarity — low ρ = hidden symmetry classes (function-space duplicate rules in the healer: token-disjoint spans with identical sensitivity profiles).
- **E3 — sensitivity-spectrum phase read.** Gram over an anchor slice → participation ratio / effective rank via the *shipped* PR math over the new object; fast spectral decay = behavior lives in few knob-directions; flat = the paper's implication-7 brittleness, measured pre-deployment.
- **E2/E12 — influence + prior/exemplar split.** Closed-form ridge solve `(K+λI)α = y` over a small anchor set → `aᵢ`-style attribution at the authored point (Koh–Liang at step zero, no Hessian); `η(x)` = authored-prior share vs exemplar-driven share of the answer. Consumers: trajectory-store explanations, decision receipts, belief attribution.
- **E5/E11 — fit-first ordering (optional arm).** Loss-derivative-shaped one-pass priority ordering over candidate sets (mining queue, corpus-synthesis candidates).

**GOAT gate shape for the strongest candidate (feature `path_kernel_sim`, katgpt-core):**
G1: PSD on randomized fixtures (Gram eigenvalues ≥ −ε·‖G‖); the linear-in-knobs identity oracle (fingerprint cosine == input cosine — known answer, proves the code); the amplitude-knob sine fixture (distinct maxima of `a·sin(kx)` at maximal input distance → fingerprint cosine ≈ 1); the planted-class fixture with an INDEPENDENT label generator (class `z` from a seeded RNG, class-conditional `x` independent of the scorer's construction — a fixture whose labels derive from the scorer's own response passes by construction and proves nothing); and a TWO-baseline battery: fingerprint must beat input cosine AND the gradient-free **output-vector baseline** `r(x) = [f_{w₁}(x), …, f_{w_m}(x)]` — if the output vector ties the Jacobian fingerprint, the tangent kernel adds nothing and the primitive stays off. G2: `--release`, p ≤ 64 knobs / 256 anchors, µs-class per-pair bars, `black_box` args + receiver (timed-region loud-zero defense). G3: every cosine/dot consumer byte-identical with the flag off (hash-pinned corpus). G4: zero-alloc hot path, scratch buffers warm. E6's KDE gate, if ever filed, additionally binds Report-the-Floor (conformal-naive floor, Plan 340). Scope note: the POC keeps every fingerprint output LOCAL to the computing process; if a future consumer routes fingerprints into SYNCED belief attribution, the determinism law binds (sequential fold, first-index ties, `freeze_version_hash` coverage).

---

## 3. Fusion — closest cousins and the novel combination

1. **riir-ai Research 009 (MDA, arXiv:2601.21996)** — EK-FAC training-data attribution for interpretable circuits. Domingos is the identity *underneath* TracIn/EK-FAC influence (TracIn = discretized, loss-weighted path kernel); MDA rejected EK-FAC as too heavy for our LoRA stack and proposed an activation-overlap proxy. Signal diff vs our training-track candidate (§4): MDA's proxy consumes **activations**; the TracIn-lite candidate consumes **cached loss-grad dot products across checkpoints** — materially lighter than the EK-FAC MDA declined (no Hessian, no per-sample curvature), so MDA's discard does not cover it.
2. **Research 624 (capacity laws)** — the I/S ratio of similarity-scored stores as they grow. Orthogonal axis, same object: 624 says what happens to a store as exemplars accumulate; Domingos says trained weights ARE such a store. Combination: read 624's growth diagnostics (interference tail, log-M threshold law) as **trained-model health metrics** — e.g., an over-trained/over-fit adapter as a store past its capacity knee.
3. **Research 455 (Hebbian kernel memory)** — the construct-then-swap direction (closed-form fact-storing MLP). With Domingos: train-is-construct. The concrete novel combination: **use 455's closed-form Hebbian construction as the kernel-regime limit of a trained ternary head** — if a GD-trained ternary model and its Hebbian reconstruction from the same exemplar set agree, the run sat at the kernel frontier (data was the lever); if they diverge, feature learning happened. One controlled comparison, no new machinery, falsifiable, pre-registerable (§4).

## 4. Three-track verdicts (per-track, no cascade)

**Track a — modelless inference: GOAT-tier candidate, not Super-GOAT.** Q1 fails on class-level prior art (Fisher kernels, Jaakkola & Haussler 1998 = similarity via a model's parameter-gradients; influence functions, Koh & Liang 2017; TracIn, Pruthi et al. 2020 = checkpointed gradient-dot influence; NTK, Jacot et al. 2018). The pinned claim is therefore the **consumption pattern**: *response-space (tangent-kernel) similarity and attribution of our own zero-training deterministic scorers, as a second channel beside input-space cosine, across retrieval/routing/abstention/belief surfaces* — that pattern has no shipped analog and is exactly the class internal greps returned zero hits for. Since Q1 (mechanism) fails while Q3/Q4 (selling point, force-multiplier) hold, the gate lands at GOAT-candidate → **`.issues/949` POC** (feature `path_kernel_sim`, gates as §2), not a Super-GOAT guide.

**Track b — self-adaptive runtime: recorded, no separate file.** The NPC angle ("which experiences shaped this behavior") is E2's attribution primitive consumed game-side (`evolve_belief` belief deltas, `hla` style_weights as the superposition store, `vibe` KG triples as the attributed-experience ledger). No new game-side mechanism beyond 949's primitive → auditable discard: identical mechanism, different consumer; cross-repo consumption follows the open primitive.

**Track c — model-based (training): Gain → riir-train issue (DEFERRED FILE, content below).** The model-based advocate's merge, ranked by value-per-GPU-hour:
- **R8-lite — TracIn-lite influence on synthesized corpora** (quest-grammar corpus; 0.4B test-arch model per the test-only rule): cache per-example loss-grad embeddings at 4–8 checkpoints of an already-scheduled run; rank influence on a probe set; drop/downweight the negative tail; **control = random pruning at equal token budget**; plus an init-only-kernel-dot control (tests whether the *path* matters — a theory-discriminating control). ≤5 GPU-h on the 4090 (issue protocol), G1 = paired LB95 > 0 vs random-prune at equal budget, 3 seeds, G3 = no downstream quant-ppl regression. Byproduct R9: patch the *generator recipe*, not the model — the corpus is code, attribution edits code.
- **R5 — drift thermometer** on every scheduled run: alignment(`K_init`, `K_end`) over a fixed 512-pair probe. Low drift = the run was kernel regression (data is the lever); high = schedule/optimizer is the lever. <0.5 GPU-h/run, marginal.
- **R2/R15 — KRR-at-init oracle + brittleness map**: predicts run feasibility/length before burning hours; buckets eval accuracy by grad-embedding nearest-neighbor distance → a measured abstention threshold for served small models. <5 GPU-h total.
- **Pre-registration (from cousin-fusion 3):** "the ternary track sits nearer the kernel frontier than an FP full-FT control at the same scale" — falsifiable with the R5 drift scalar + the 455-Hebbian-limit comparison.
- Advocate's honest too-speculative list (accepted, with reasons): path-kernel-exact training (lazy-regime solutions underperform at finite width — refuted direction), boosting-instead-of-GD literally, KDE sampling for generation heads, direct architecture→kernel design, full sign-mask spectrum optimization (keep the free measurement), EK-FAC preconditioning (implementation-heavy; revisit only on drift-starved evidence).
- ⛔ **Filing state:** RESOLVED at verdict round 1 — filed via a detached worktree at `origin/develop` (`7e84f0ec`) so the sibling's uncommitted WIP on `.issues/635` was never touched (rebase would refuse on the unstaged tracked change, and stashing files this session did not modify is forbidden): **riir-train `.issues/638_path_kernel_influence_diagnostics_poc.md`, commit `b6921703`, pushed to `origin/develop`.** HW advanced 637 → 638; dual-allocation gate 0 collisions at allocation time; worktree removed after the push.

## 5. §4 prior-art searches actually run

1. Workspace-internal first (GRAM/058 rule): grep of all 16 `.research/` + `.plans/` + `riir-ai/.docs` for `2012.00152` (zero hits) and for `neural tangent|path kernel|kernel machine|influence function|tracin|gradient space|superposition` (hits read: 009 MDA, 624, 455, 127, 146 — the four cousins above; the quant-plan hits were "kernel" false positives).
2. Web: "TracIn influence functions training example attribution gradient dot product checkpoints" → TracIn (arXiv:2002.08484, 962 citations), TRAK, empirical influence functions — confirms the attribution class is densely published.
3. Web: Domingos follow-up/critique landscape → the paper itself dominates; the kernel-vs-feature-learning caveats pinned from the citation graph (Fort et al. NeurIPS 2021; silent alignment arXiv:2111.00034).
4. Codebase signal-diffs (reads, not just greps): `similarity.rs`, `subspace_phase_gate.rs` (PR over latent directions), `latent_trajectory_geometry.rs` (intra-trajectory geometry — the doc's own "what this is NOT" section separates it from pairwise cross-query kernels), `faithfulness/smear.rs` (mass-distribution classifier, unrelated), reflex `embed.rs`/`engine.rs` (input-space hashed features).

## 6. Verdict

**Tiers:**

| Tier | Criteria | This paper |
|---|---|---|
| Super-GOAT | All 4 YES | **No** — Q1 fails (Fisher/TracIn/NTK class prior art) |
| GOAT | Provable gain, no new class | **Candidate** — E1/E4 fingerprint kernel + E3 spectrum, pending issue-949 POC (the gain is a hypothesis until the planted-class fixture measures it) |
| **Gain** | Incremental, useful | **← VERDICT: Gain.** One-line reasoning: the lens bridges our saturated kernel-memory family to every trained asset we own and extracts one genuinely unshipped modelless similarity channel, but the mechanism class is published (Fisher/TracIn/NTK), so this files issues, not a guide. |
| Pass | Ships/irrelevant | — |

**MOAT gate (katgpt-rs):** in scope — fundamental-principle lens + open-primitive candidate (`path_kernel_sim` is generic math, no game/chain/shard IP); public-safe. The game-side consumer (attribution receipts) and the healer consumer (function-space rule dedup) are recorded as consumers of the open primitive, not separate moats.

**Files:** this note; `katgpt-rs/.issues/949_response_space_similarity_kernel_poc.md`; cross-ref lines added to 624, 455 (this repo) and riir-ai 009 (committed file-scoped). riir-train issue 638: deferred file, content in §4.

## 7. T9 verdict addendum (2026-10-20) — KEEP-OFF

The issue-949 POC landed (feature `path_kernel_sim`, opt-in; Bench 947 is the
record). The planted-class battery at release read a **THREE-WAY TIE**
(fingerprint 1.000 · input-cosine 1.000 · output-vector 1.000) → T5d's own
law fires: the tangent kernel adds nothing over gradient-free response
probing at matched information → **promoted nowhere**.

The structural reason generalizes the collapse caveat (§2): for any scorer,
`Ŝ(x)` is a deterministic transform of `x`, so a full-rank probe battery
(m ≥ p informative settings) is at-least-as-informative as the gradient
direction — gradient information ⊆ local-response information. The two
residual edge regimes (probe coverage m ≪ p; probe degradation via
saturation) are consumption-side preconditions, priced out of POC scope.
Re-open bar: a consumer whose probe budget is genuinely ≪ p (the heal-seat
rule-dedup consumer would need its span-fingerprint dimension to
substantially exceed its affordable probe count).

The kernel itself is CORRECT (T5a proven the collapse identity, T1–T4 green,
G2 79–132 ns/call, G4 zero-alloc) — it ships opt-in as a diagnostic
instrument, never a default. This upgrades §6's GOAT-candidate tier to a
measured **Pass-tier negative**: the lens produced a real primitive, the
primitive's promotion hypothesis measured false.
