# Research 667: Contracting Belief Observers — KKL Certificates for Latent Residual Updates

> **Source:** "Provable State Estimation with Recurrent Models" — Vincent Andrieu, Pauline Bernard, Lucas Brivadis, Laurent Praly, Christian Wolf (LAGEPP/CAS-Mines Paris/L2S/Naver Labs Europe), [arXiv:2611.09217](https://arxiv.org/abs/2611.09217) (v1 2026-11-06), NeurIPS 2026. 52 pp, theory paper.
> **Date:** 2026-11-12
> **Status:** Active — verdict **Gain** (modelless track) via Claude ping-pong 2026-11-12, 2-round negotiation (R1 REVISE — Jacobian sign/space error, C1 parameterisation mix, F3 ℓ=0 untestability, S2 backwards semantics; R2 AGREE with the F4/S2 fade-rate condition, applied), reviewer session `420f2e77-f1ec-4fa0-9d74-e75058073f9f`; training track = audited discard with re-open trigger (§8)
> **Related Research:** 266 (FPRM damped fixed-point halting — the iteration-map cousin), 546 (fractal basins / λF loop-map measurement), 152 (KARC delay embedding — riir-ai), 556 (FLYNN — the additive belief kernel's own lineage)
> **Cross-ref (riir-ai / riir-train):** riir-engine `cwm_runtime::ns_csg_bridge::ObservationGate` (the consumer seam); riir-train `.issues/688` (brain-lane probes)
> **Classification:** Public

---

## TL;DR

The paper proves that a recurrent filter whose latent flow is **uniformly monotone** (Jacobian symmetric part ≤ −2I) is a *provable observer*: it exponentially forgets its initialization (|Δz(t)| ≤ e^{−κt}|Δz₀|), embeds the observable physical state **injectively** for large enough gain κ (KKL observer theory), is **tunable** (error → ε by t_o via a closed-form gain law κ ≈ t_o⁻¹·ln(1/(L̄‖δ‖)) with a κ^ℓ noise floor), and its discrete residual-RNN form is stable iff **δκL_z² ≤ 1**. Distilled here: the shipped `evolve_belief_additive` kernel **is** the paper's contractive discrete cell — its continuous residual field has Jacobian exactly −(1−α_m)·I (uniformly monotone; the discrete map's contraction factor is ρ = α_m), with a bounded sigmoid drive — and therefore carries all of these guarantees by construction — none of which any surface currently asserts. The actionable extraction is a **certificate + derived-schedule primitive family** (per-cell monotonicity, settling-tick / disturbance-fade laws, κ* gain selection with typed SNR refusal) replacing three hand-pinned constants (`belief_retention` 0.9, `ObservationGate` λ 0.05 / threshold 0.3) with derived forms, plus a fixture kit proving them.

**Distilled for katgpt-rs (modelless, inference-time):**
The paper's theorems are about what an architecture *can represent*, not how to train it — the entire certificate machinery is deterministic config arithmetic + property fixtures. No gradient descent anywhere in the extraction (§3, §4).

---

## 1. Paper Core Findings (compact)

1. **Contraction ⇒ fading memory** (Prop 3.3): a residual cell satisfying Assumption 3.2 (uniform monotonicity; the paper's example cell `ż_i = κ(−α_i z_i + λ_i tanh(z_i + U_i o))` satisfies it iff **α_i − |λ_i| ≥ 1**) keeps the ball Z = {|z| ≤ 2ḡ} invariant and contracts all trajectory pairs at rate e^{−κt}. The scalar κ is an explicit tunable time-scale-separation gain.
2. **Injective state embedding** (Thms 3.4/3.7): for κ > 2L_f, under differential observability (order ℓ) + a generic filter rank condition (C_ℓ = [B AB … A^ℓ B] left-invertible at per-observation equilibria; Prop K.1 proves genericity via a Vandermonde argument), a C¹ Lipschitz-injective map T_κ: x ↦ z exists with a globally Lipschitz left-inverse. Remark 3.8: T_κ is a **filtered** version of (o, ċo, …, o^(ℓ)) — the i-th derivative enters at scale κ^{−i} — "instead of differentiating o ℓ times, which is ill-posed under noise, the latent state carries the same information through a stable filter whose bandwidth is set by κ". This is the noise-robust cousin of delay embedding.
3. **Tunability + noise floor** (Thm 4.1, Prop 4.3): |x̂−x| ≤ c′κ^ℓe^{−κt}|z₀−T_κ(x₀)|; under bounded observation disturbance δ the steady-state floor is 2c′κ^ℓ L̄_go‖δ‖∞ and past disturbances fade at rate κ/2. The **useful-gain formula**: κ ≈ t_o⁻¹·log(1/(L̄_go‖δ‖∞)) — beyond it, raising κ only raises the floor. If L̄‖δ‖ ≥ 1, **no useful gain exists** (log ≤ 0).
4. **Decoder complexity** (Thm 4.2): an MLP decoder of W ≤ C(κ^ℓ/ε)^m·log(κ^ℓ/ε) weights suffices.
5. **Discrete stability** (Thm 5.3): the residual RNN z_{k+1} = z_k + δκ·g(z_k,o_k) is tunable iff **δκL_z² ≤ 1**, with per-step contraction factor ρ = 1 − δκ/2 and global Euler error ≤ Mκδ.
6. **Contracting GRU** (App H.2): gated residual cell `ż = −αz + r⊙n` with spectral-norm ≤ 1 on W_r, W_n + α = 2.5 certifies λ_max(sym ∂g/∂z) ≤ −1.25 a priori (measured −2.18 via autodiff); stacks stay monotone under the metric P = diag(μ²I, I) with κ rescaling.
7. **Emergent contraction**: unconstrained GRUs trained on a real 27 km pose-correction task converge to contracting dynamics anyway — contraction is an emergent property of well-trained sequence models; the constrained variant trains more stably across seeds (5/5).

## 2. Published prior art (§4 sweep result — class is covered)

Two-source verdict, live-verified this session (~20 sources: arXiv API, Semantic Scholar, Crossref):

- **Textbook:** uniform exponential forgetting = echo state property (Jaeger 2001) / monotonicity consequence (Lohmiller–Slotine 1998); the speed-vs-noise high-gain trade-off is Khalil's SIAM book Ch. 8 (plus Ball–Khalil 2008 nonlinear-gain, Ahrens–Khalil 2009 switched-gain); injective KKL immersion = Andrieu–Praly 2006, nonlinear contracting targets = Pachy et al. 2024 (the paper's own lineage, same groups); the Euler-monotone step condition is classical numerics (the gradient-step condition's twin).
- **Recent but published:** trainable contraction-certified RNNs (Revay–Manchester arXiv:1912.10402; REN arXiv:2104.05942, TAC 2023 — includes data-driven observer demos); contracting Neural ODEs (Zakwan 2022/2025; NodeREN CDC 2023); learned KKL incl. contraction-enforced (Deep KKL CDC 2021; Galimberti et al. IFAC 2026); PINN-KKL with ISS noise guarantees (arXiv:2611.13411 — landed 2 days *before* this paper); empirical noise-aware KKL gain tuning (Buisson-Fenet 2022); reservoir injective embeddings + the RC=KKL bridge (Hart–Hook–Dawes 2020; Grigoryeva 2022; Bizyaeva arXiv:2610.04343).
- **Genuinely new in the paper:** the *conjunction and quantitative packaging* — one trainable uniformly-monotone class carrying the theorem chain from forgetting → injective embedding with an explicit large-gain threshold, the closed-form tunability law with κ^ℓ floor (no published instance found for learned filters — a one-line consequence of exponential decay + polynomial floor, so novelty is packaging), the gain-linked decoder bound, the discrete extension, the GRU-contraction census. Class novelty: low-to-moderate.

Consequence for the tier verdict: the CLASS is published; only workspace-level novelty is claimable (§3 signal-diffs), which caps this at **Gain**.

## 3. Path 0 inventory + workspace signal-diffs

| Paper component | Workspace analog? | Signal-diff (what it consumes vs ours) | Disposition |
|---|---|---|---|
| Monotonicity certificate (α_i − \|λ_i\| ≥ 1) | **No ship.** Closest: `GainCostLoopHalter` `HaltReason::NonContraction` (bench-observed step-decay halting, not a Jacobian certificate); Research 546's λF loop-map measurement (training-lane diagnostics over looped-LM iterations) | Halter consumes *step-size decay + direction reversal* of an iteration sequence; λF consumes Jacobian-product growth of a *loop map*; neither is a per-cell static certificate on a belief update | **Extract** → katgpt-rs issue 1000 T1/T2 |
| Fading bound \|Δz\| ≤ e^{−κt}\|Δz₀\| | **Partial.** `ObservationGate` (riir-engine `cwm_runtime/ns_csg_bridge.rs`) decays confidence `sigmoid(−λ·Δtick)`, λ = 0.05 hand-pinned; `evolve_belief_additive` decays as α^t by construction | Gate consumes **time-since-observation** (staleness → re-observe cadence); paper's bound consumes **initial-condition distance** (trajectory-pair forgetting). Different contracts; both "fading", neither asserts the other | **Extract** (settling-tick law couples them — §4) |
| Derived gain law κ* + noise floor + SNR refusal | **No ship.** Shipped constants hand-pinned: `belief_retention: [0.9; 6]` (κ = −ln 0.9 ≈ 0.105/tick, half-life ≈ 6.6 ticks — 0.33 s at a 20 Hz cognition tick, 0.66 s at seal's 10 Hz serving rate), ObservationGate λ = 0.05 / threshold 0.3 (re-observe at Δt ≈ 17 ticks = 16.9), `additive_drive_scale` 4.0. House law bans magic thresholds beside the FSM | No surface derives a rate from a noise bound or a settling target; `spsa_direction_with_floor`'s `noise_floor` is a numerical-difference guard, not an observer gain | **Extract** (the κ* law + typed SNR refusal) — the strongest single item |
| Injective embedding T_κ (filtered observability data) | **Cousin, not coverage.** KARC (riir-ai Research 152) reconstructs state via **delay embedding** — delay coordinates are noise-fragile at high orders; the paper's T_κ is the **filtered** version (bandwidth κ, noise floor κ^ℓ) | KARC consumes the delay-coordinate map; T_κ consumes a low-pass filter bank with a tunable bandwidth. The noise-robustness axis is the delta; KARC unaffected | **Fusion note only** — no runtime surface needs the map itself (it is existential, not constructive) |
| Iteration-map contraction / spiraling | **Covered.** Research 266 FPRM: damped update g_η,θ = η·f_θ + (1−η)·z kills \|λ_i\| ≥ 1, ℜ(λ_i) < 1 spiraling in looped-transformer fixed-point halting | FPRM is about **iteration-to-fixed-point** convergence (inner loop); the paper is about **observation-filter** forgetting (outer time). Orthogonal axes that share the monotonicity instrument | Compose, never merge — cited in 266 |
| Convergence census of belief loops | **Covered, negative.** katgpt-rs Issue 991 (closed NEGATIVE 2026-11-11): `ConvergenceDetector` residual census; at caller postures (K=3) nothing converges before K | Census measures *dynamic* iteration-to-convergence at runtime; the certificate is the *static by-construction* guarantee the census could not find dynamically | Compose — the stationary-surface test (skill §1 step 4a): the certificate is config arithmetic (no mutated state ⇒ no dynamics signal), the census was the dynamics readout; they answer different questions |
| Discrete step law δκL_z² ≤ 1 | **Partial.** GDN/linear-attention recurrences in riir-infer are stability-engineered per-lane; belief kernels carry no step law (their "step" is the clamp + α) | Inference lanes tune kernels per-arch; the law here is a config-validation for any residual latent update | Annotation only (no consumer lacks it today) |
| Contracting GRU training recipe | **No failing consumer** — see §8 | — | **Audited discard** (training track) |

## 4. Distillation — the modelless primitive family

### 4.1 The shipped kernels under the paper's lens (the load-bearing observation)

`evolve_belief_additive` (katgpt-sense `reconstruction.rs` L986-1005, `modality_additive` DEFAULT-ON since Issue 777 T4):

```
h_{t+1,i} = α_{m[i]}·h_{t,i} + (1−α_{m[i]})·(2σ(η·k_{m[i]})−1)   (clamped to [−1,1])
```

This **is** the paper's contractive discrete cell: linear dissipation with a state-independent bounded sigmoid drive. Two parameterisations, kept apart — the paper's analysis lives in the continuous residual field; the kernel iterates the discrete map (review R1 correction; the original draft conflated them):

- **Discrete map:** h_{t+1} = α_m·h_t + (1−α_m)·drive ⇒ map Jacobian **+α_m·I**; trajectory-pair distance decays as ρ^t with **ρ = α_m exact**; rate κ_m = −ln α_m ≈ 0.105 at the shipped α = 0.9 (half-life ≈ 6.6 ticks — 0.33 s at a 20 Hz cognition tick, 0.66 s at seal's 10 Hz serving rate).
- **Continuous residual field:** ġ = −(1−α_m)(h − drive) ⇒ field Jacobian **−(1−α_m)·I**, symmetric part negative definite ⇒ the paper's uniform monotonicity holds *exactly*; effective step δκ = 1 − α_m = 0.1 at ship.
- **Bridge (so C1 and S4 agree on one kernel):** exact Euler of the linear dissipative field gives discrete factor ρ = 1 − δκ; the paper's general nonlinear bound ρ = 1 − δκ/2 (under δκL_z² ≤ 1) *specializes* to the exact ρ = 1 − δκ = α_m here. One kernel, one number: ρ = α_m.

Every paper guarantee (fading, invariant ball, tunability-in-the-κ sense) applies verbatim. **No surface asserts or exploits this today** — the certificate is a documentation + fixture + config-validation gap, not a rewrite.

By contrast, the divisive kernel `leaky_step` (katgpt-types `leaky_core.rs`, behind `evolve_belief`): delta = `clamp(lr·min(total,1)/total·(input − total/2), ±max_delta)` is **state-independent** — the update is z ↦ clamp(z + c(e)) — so two trajectories differing by a constant interior offset never converge; its fading is **clamp-boundary-driven**, not flow contraction. NOT a contraction in the paper's sense — recorded as a measured distinction, **not a defect**: the divisive kernel is GOAT-adopted for superposition scaling reasons (modality_superposition_bench; ratio 5.17 additive-vs-divisive axes are orthogonal). If a consumer ever needs a *certified* belief filter, the additive kernel is the one that carries it.

### 4.2 Extraction family (feasible, no GD — the issue-1000 scope)

**Certificates (config arithmetic, exact for the shipped family):**
- **C1 per-cell monotonicity (spaces named):** *discrete-space* certificate: assert α_m ∈ (0, 1) ⇒ ρ = α_m, κ_m = −ln α_m (exact for the additive family). *Continuous-flow-space* certificate (the paper's iff): residual cell ż = κ(−α_i z_i + λ_i tanh(·)) is monotone iff α_i − |λ_i| ≥ 1. Bridge between spaces: δκ ↔ 1 − α_m; exact-linear fields give ρ = 1 − δκ; the paper's general bound ρ = 1 − δκ/2 specializes to it (§4.1). Admission-gate shape: an imported/configured cell failing its space's certificate is refused by name.
- **C2 invariant ball:** for the clamped kernels R_z = √8 is structural (clamp domain [−1,1]⁸ ⇒ |z| ≤ √8) — and **conservative**: the drive is 2σ(η·max(0,k)) − 1 ∈ [0,1), so the reachable set is tighter still; the slack is tolerance budget for F4, never slack in the law. Generic family: R_z = 2ḡ. Bound |Δz₀| ≤ 2√8 feeds the settling law with no unknown constants.
- **C3 (deferred):** spectral-norm certificate for W-carrying cells (power iteration at load) — no current consumer ships a W in its belief cell; filed as the generic extension, not built.

**Derived schedules (one-liners, replacing pinned constants):**
- **S1 settling ticks:** ticks until |Δz| ≤ ε = ln(2√8/ε)/(−ln α_m). At α = 0.9, ε = 0.1: ≈ 38 ticks. The c′-free worst-case wait; c′ replaced by a fixture-measured surrogate (F1).
- **S2 disturbance-fade after observations resume:** Prop 4.3's d(t) bounds how fast *past disturbances* wash out of the estimate once clean observations flow again: after n clean ticks, residual disturbance influence ≤ e^{−κ_fade·n}·‖δ‖∞, so an episode of bound ‖δ‖ stops mattering within ≈ ln(‖δ‖/ε)/κ_fade ticks of resumption. **Fade rate (mirrors the §4.1 ρ bridge):** κ_fade = κ/2 is the paper's generic nonlinear bound; on the exact-linear additive kernel the disturbance-driven state difference obeys Δh_{t+1} = α_m·Δh_t once the disturbance stops, so the fade is at the FULL rate κ_m exactly. It is **not** a statement about what a total observation gap costs — during a gap no new information arrives and error vs a moving truth grows unchecked (that side is governed by the physical flow, which is exactly what the re-observe cadence exists to bound). Consumer shape for ObservationGate: "after a stale/noisy episode, the estimate is trustworthy again n ticks after clean resumption, n derived from (κ, ‖δ‖, ε)" — not a pinned λ.
- **S3 useful gain + typed refusal:** κ*(t_o, δ, L̄) = t_o⁻¹·ln(1/(L̄·δ)) — note ℓ does NOT enter κ* (it enters the floor 2c′κ^ℓL̄‖δ‖); `None` (typed SNR refusal) when L̄·δ ≥ 1 — fail-closed naming the deficit; feasible window [κ_track(ε), κ*]; a diminishing-returns lint for κ > κ* — the lint is meaningful only when ℓ ≥ 1 (F3: at ℓ = 0 the floor is flat in κ, so "more gain raises your floor" has no observable on the additive kernel).
- **S4 step law:** δκL_z² ≤ 1 validation + ρ = 1 − δκ/2 per-tick confidence multiplier for any residual latent update (generic, annotation-level for current consumers).

**What does NOT extract (honest limits):** T_κ itself (existential — ship the filtered-ladder *idea* only as an architectural note); c/c′ existence constants (fixture-measured surrogates, disclosed as evidence not proof); differential-observability order ℓ for nonlinear cells (structural property; sampled rank checks are heuristics); the decoder bound (capacity sanity lint at most).

### 4.3 Fusion candidates

- **× ObservationGate (riir-engine):** the strongest consumer seam. The gate's λ = 0.05 and threshold = 0.3 are independent pins; the paper couples forgetting-rate and observation-noise into one law (S2/S3). The re-observe cadence should be ≥ the settling horizon of whatever filter feeds the gate (currently two constants that don't know about each other). **Gated on a noise-input decision**: no surface measures a per-channel observation-noise bound δ today — the derived form needs δ from a declared sensor spec or an F1-measured surrogate. Filed as the named follow-up consumer, not wired blind.
- **× FPRM (Research 266):** compose, never merge — 266 owns iteration-to-fixed-point (inner loop) contraction; this note owns observation-filter (outer time) contraction. The shared instrument is the monotonicity/Jacobian-symmetric-part check; 266's `NonContraction` halt generalizes from step-decay evidence to a per-cell certificate.
- **× KARC (riir-ai Research 152):** delay embedding vs filtered embedding — the paper's Remark 3.8 is the noise-robust reformulation of the same "latent carries the observability data" claim. KARC's delay coordinates are the unfiltered (noise-fragile at high orders) pole; bandwidth-tunable filtering is the delta. Architectural cross-ref only.
- **× DEC (katgpt-dec):** contraction = negative divergence; the symmetric-Jacobian trace is computable directly. DEC framing **skipped by the house curse-of-dim law** (belief vectors are 8-dim; boundary-vs-volume wins only d ≤ 3).
- **× sigmoid_calibration (katgpt-core, Benches 807/808):** calibration of gate *confidences* (ECE/Brier axis); orthogonal to decay-rate derivation. No composition.

## 5. Proof path (pre-registered fixtures — issue 1000 T2)

| Fixture | Asserts | Falsification arm |
|---|---|---|
| **F1 contraction** | paired rollouts of `evolve_belief_additive` from distinct inits: decay ≤ α^t per dim; regression-fitted rate matches −ln α within tolerance | plant α → 1−10⁻⁶ near-unity: fixture must show non-contraction at realistic tick horizons |
| **F2 invariance** | boundary-seeded rollouts stay in the clamp ball | feed a thawed/garbage state → clamp + telemetry fire |
| **F3 floor sweep (ℓ-split, pre-registered after review R1)** | **primary arm — synthetic ℓ = 1 observed-integrator cell** (position observed, velocity latent): steady-state error floor **rises** for κ > κ* — the paper's κ^ℓ law made observable. **Consistency arm — the additive kernel at ℓ = 0:** its floor is FLAT in κ (identity readout carries no derivative information) — asserted so the law is never misapplied | primary: floor(2κ*) > floor(κ*); consistency: additive floor flat in κ across the sweep; L̄·δ ≥ 1 arm → typed SNR refusal fires |
| **F4 disturbance-fade settling** | after a disturbance episode ends, estimate error crosses ε at the S2-predicted resumption tick ± tolerance. Rate split (mirrors the ρ bridge): the additive kernel fades at the FULL rate κ_m exactly — F4's additive arm asserts the crossing at the κ_m-rate tick; the generic κ/2 tick is an UPPER BOUND that must also hold. S1's settling horizon bounds the init-mismatch side | releasing earlier than predicted shows error > ε (law is load-bearing, not decorative); the ± tolerance exists because C2's √8 radius is conservative — never mistaken for law slack |
| **F5 stability edge** | δκL_z² just over 1 diverges; just under contracts at ρ | proves S4's condition is live |

## 6. Published-prior-art + workspace-cousin cross-refs (grep hygiene)

arXiv ID + title greps across the workspace returned **zero** prior distillations of 2611.09217. Cousins verified this session and diffed in §3: Research 266 (FPRM), Research 546 (λF + `HaltReason::{Oscillation,NonContraction,Concavity}`), Issue 991 (closed NEGATIVE, census instrument retained), Research 152 (KARC, riir-ai), `ObservationGate` + `evolve_belief_additive` + `leaky_step` (shipped code), `sigmoid_calibration` (Bench 807/808).

## 7. Verdict — modelless track: **Gain**

| Gate | Score | Reason |
|---|---|---|
| Q1 no prior art | workspace YES / class NO | §2: class published (KKL lineage + contracting-RNN industrialized 2019-2023 + the trade-off textbook); §3: the certificate + derived-schedule combination has no workspace ship or note |
| Q2 new behavior class | NO (weak) | certified-decay is an upgrade of existing belief substrate, not a capability class the stack lacks |
| Q3 product selling point | moderate | "NPC beliefs are provably fading-memory observers with derived staleness-vs-noise tuning" — real, niche |
| Q4 force multiplier | YES | belief kernels (sense) + ObservationGate (engine) + any residual latent update (core) + future W-cells |

Not all 4 YES → **not Super-GOAT**. Gain (actionable, small, fixture-provable) — filed as katgpt-rs `.issues/1000` (T1 primitive, T2 fixtures, T3 consumer seam). Per ungoat discipline: smallest artifact that carries the verdict; no plan file (the issue's task list is the plan).

**MOAT gate (katgpt-rs):** PASS — base primitive for belief/latent-residual updates, generic math, no game semantics in the primitive itself; the game wiring is the named riir-ai consumer, kept out of this repo.

## 8. Training track: audited discard + two probes filed (riir-train `.issues/688`)

The Contracting GRU recipe (spectral norm + α on gated residual cells) targets *training stability of unconstrained recurrent heads*. Discard reasoning at mechanism level: (a) the stack's trained recurrent heads (frozen-brain ternary modules, riir-engine `sense::brain::NpcBrain`; drafter heads) show **no documented seed-instability gate failure** — nothing the recipe fixes today; (b) the paper's own finding is that contraction **emerges** in well-trained GRUs, so the expected catch rate of a retrofit is low; (c) a standing detector already exists (λF loop-map measurement, Research 546 addendum / riir-train Issue 532) and the compose side is deterministic enough to probe cheaply. **Re-open trigger:** first demonstrated seed instability in a trained recurrent head, or the next brain re-train cycle running the two ~zero-cost probes filed in riir-train 688 (contraction-emergence probe in the training report; freeze-gate certificate asserting the discrete stability condition at the lane's serving tick rate + recording the measured contraction rate beside the freeze digest, AnchorSidecar precedent). The trajectory-integration loss (windowed kinematic drift) is the one genuine *capability* lever for fog-of-war belief tracking — recorded inside 688 as evidence-gated, not filed as a plan: it needs a state-supervised brain lane that does not exist today.

## 9. Weakest point (named)

The κ* law's **inputs do not exist yet**: no surface measures an observation-noise bound δ, a readout Lipschitz constant L̄, or an observability order ℓ (and at the additive kernel's identity readout ℓ = 0, so the floor-rise half of the law needs a synthetic ℓ ≥ 1 cell — F3's split). The near-term deliverable is therefore C1/C2 + S1/S2 + the typed-refusal *shape* — all computable from shipped constants alone — while S3's full form lands only after a noise-input decision (declared sensor spec or F1-measured surrogate). The "replaces magic constants" claim is **partial**: the derived formula exists, the input pipeline does not. Secondary: this note certifies the additive kernel's contraction from its formula (exact) but the fixture kit (F1) is what makes that a measured claim — the two must land together or the certificate is prose.
