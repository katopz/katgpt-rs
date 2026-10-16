# Research 620: Mamba-3 Oscillatory Dynamics — Damping Regimes, GDN-Is-Overdamped, and the Rotation/Erase Accumulation Duality

**Status:** Active — GAIN verdict; monitor issue (riir-infer `.issues/046`), training probe plan (riir-train `.plans/459`), phase-carrying belief issue (katgpt-rs `.issues/939`) filed this session.

> **Source:** "Understanding Mamba-3 Through Oscillatory Dynamics" — Shutian Liang, Yue Song (Tsinghua), [arXiv:2610.13341](https://arxiv.org/abs/2610.13341), 2026-10-09. Code: shutian-liang.github.io/oscillationmamba. CC-BY 4.0.
> **Date:** 2026-10-13
> **Related Research:** 623 (QRR quotient recurrent reasoner — the CONVERGENCE half of this axis: rate-inheritance theorem, symmetry-modulo halting, delta-sequence mode attribution; this note owns the capacity/regime half), 169 (LinOSS oscillatory SSM — closest cousin on the oscillation axis), 371 (low-rank RNN oscillatory DMFT — the mean-field cousin), 070 (GDN2 — the GDN-lane cousin; its Mamba-3(MIMO) tables are what this paper's framework explains), 243 (temporal derivative kernel), 301 (Subspace Phase-Gate)
> **Related Plans:** 189 (LinOSS, shipped from 169), 104 (GDN2, shipped), riir-train Plan 459 (this research's training probe — filed this session)
> **Classification:** Public

---

## TL;DR

The paper proves Mamba-3's decay-rotation transition `h_t = ρ_t·R(θ_t)·h_{t-1}` is **exactly** the flow of an underdamped second-order oscillator (`ω₀=√(a²+ν²)`, `ζ=a/ω₀<1` for any ν≠0), then uses that framework for the first controlled damping-regime comparison in linear-attention recurrences: **underdamped beats critically-damped and overdamped on state tracking by 30–300×** (Parity 100 vs 3.27/3.22; Mod-3 95.26 vs 0.58/0.32), and Appendix A **proves our exact shipped GDN recurrence is overdamped** (ζ=(2γ+b)/(2√(γ(γ+b)))>1, discrete spectrum {g(1−β), g}). The workspace consequence: our Bonsai fleet sits in the paper's control arm on the oscillation axis (with the coarse delta-rule erase as partial mitigation, and hybrid attention layers absorbing part of the load — an upper bound on exposure), and the paper's closed-form machinery (ζ diagnostic, mode-persistence trichotomy, constructed phase codes, the additive-write accumulation bound) is modelless-extractable as monitors, spec tests, and eval fixtures — no training required for the diagnostic half.

**Distilled for katgpt-rs (modelless, inference-time):**
Everything quantitative in the paper is closed-form algebra over two scalars per channel `(ρ, θ)`. The trained model's own optimum (E.1) is a *writable deterministic function* — input-conditional phase rotation implementing a modular counter — so the code extracts without the optimizer: `φ += 2π/m·1[event]`, cosine readout = count mod m.

---

## 1. Paper Core Findings

1. **Exact-flow equivalence (§3):** `h_t = ρ R(θ) h_{t-1}` with `a=-ln ρ/Δt`, `ν=θ/Δt` is the exact flow of `ẍ+2aẋ+(a²+ν²)x=0` (uses `J²=-I`, `e^{φJ}=cos φ·I+sin φ·J=R(φ)`; `I,J` commute). Damping ratio `ζ=a/√(a²+ν²)` — any nonzero rotation ⇒ underdamped.
2. **Damping-regime controlled comparison (§4):** the κ-controller `A(κ)=[[0,1],[-(a²-κ),-2a]]` holds second-order state structure fixed and moves only the spectral family: underdamped (κ=-ν²) vs critically damped (κ=0) vs overdamped (κ=r²). From-scratch state-tracking: **100/95.26/95.68 vs 3.27/3.22/81.60 vs 3.22/0.32/1.97** (Parity/Mod-3/Arith-w/o-brackets). Same state size, same inputs — the oscillatory *regime* is the active ingredient. Data-dependent phase also matters: fixed-rotation Mamba-3 collapses (4.05/0.85/25.49).
3. **Mode persistence (§4.2, C.2):** overdamped → fast mode exponentially suppressed → **mode collapse** to the slow eigendirection; critically damped → fastest attenuation, asymptotically **rank-1**; underdamped → both modes decay equally, **relative geometry preserved**.
4. **Accumulation bound (C.3):** with rotation, steady state under persistent forcing is `‖z∞‖=‖u‖/|1-ρe^{iθ}|` instead of `‖u‖/(1-ρ)` — near-unit retention ρ→1 does NOT imply state blow-up when θ keeps write directions varying. Measured: phase removal raises history–update cosine alignment (0.29→0.47 at 64K) and state RMS norm (~10%) in pretrained 1.5B Mamba-3.
5. **Phase interventions (§5.3):** phase removal/reversal/shuffle degrade PG-19 PPL up to **41×/74×/7.25× at 64K** (length-compounding) vs decay permutation ~6× (flat) — **phase error compounds with context; decay error does not**. Write–read phase consistency: removing rotation from B or C alone hurts (9×/12×), removing from BOTH partially restores (2×) — write and read must share the phase frame.
6. **Task-aligned phase codes (E.1):** trained pairs implement modular counters — input 1 rotates phase ≈180° (parity) / ≈120° (mod-3), input 0 ≈0°. The learned solution is the ideal phase code for counting ones mod m.
7. **Linear-attention spectral taxonomy (Appendix A, Table 8):** Linear Attention = no decay; DeltaNet = decay + preserved mode (eigenvalues -b, 0); **Gated DeltaNet = two-rate decay, PROVEN overdamped** (ζ>1 above); KDA = modal decay, overdamped; **Complex KDA** (β∈[0,2], signed α_i∈[-1,1], Householder composition `H_k·H_{e1}=R(θ)`) **can admit oscillatory modes** (window: α<0 ∧ |cos θ|<2√(-α)/(1-α)); **Momentum DeltaNet** admits underdamped modes when discriminant D<0. Stated OPEN: how to *guarantee and control* underdamped modes within linear-attention updates.
8. **Pretraining (§5.2):** oscillatory Mamba-3 wins length generalization at both scales (PG-19 PPL beyond 2K context; LongBench-E 13.22 vs 10.61 at 444M, a wash at 187M: 9.53 vs 9.76). Standard downstream comparable at 187M, modestly better at 444M. **The advantage grows with scale and context.**

---

## 2. Path 0 Inventory (training-target decomposition)

| Component | Value type | Analog in workspace? | Modelless extraction? | Verdict |
|---|---|---|---|---|
| Second-order equivalence (§3) | closed-form identity | No (new analysis of OUR recurrences) | YES — spec-test provable (Lean one-sitter + paired Rust spec-match) | Extract → spec test (riir-infer issue 046) |
| Damping-regime comparison (§4) | empirical (trained) | No | No — needs from-scratch training | Training track → riir-train Plan 459 |
| Mode-persistence trichotomy (§4.2) | closed-form | No | YES — applies to any OVERDAMPED propagation operator, ours included (App. A.4 classifies the GDN lane itself); report-only rank/geometry-collapse readout, one small SVD cached per window | Extract → riir-infer issue 046 (probe half) |
| Accumulation bound (C.3) | closed-form | No shipped consumer — **verified at verdict round 2**: every recurrent accumulator in the workspace is bounded by construction (GDN = delta rule with keyed erase, steady state `β‖v‖/(1−g+gβ) ≤ ‖v‖`; `evolve_belief_additive` = convex EMA, steady state = drive; refine `ema_step` = convex interpolation with clamp, steady state = fix_dir) — none has the unnormalized `h = ρh + u` write form the bound requires | The bound is **available math for future plain linear-attention (additive-write) lanes** — recorded in the note, no issue | Record (no file) |
| Phase interventions (§5.3) | diagnostic methodology | Not actionable — fleet has no rotating model (Bonsai GDN is scalar-decay, θ≡0) | The *shape* extracts: phase-error-compounds/decay-error-flat ordering law | Recorded (ordering law for future precision allocation) |
| Phase codes (E.1) | constructed mechanism | Partial — 169/371 cover *free/learned* oscillation; **no constructed input-conditional phase counter ships** (signal-diff below) | YES — `φ += 2π/m·1[event]`, `sigmoid(cos φ)` readout | Fusion idea → katgpt-rs issue 939 |
| Linear-attention taxonomy (App. A) | closed-form classification | Our lanes' rows derivable on paper: Bonsai GDN = scalar decay = paper's A.4 two-rate row exactly | YES — assert our recurrence's spectral class as a spec test | Extract → riir-infer issue 046 |
| Underdamped linear attention (open problem) | architecture/training | Published partial answers exist (Complex KDA, MDN, Selective RoPE) — the *class* is not novel; the **GDN2-instrumentation datapoint** (do decoupled erase/write gates admit/avoid underdamped modes?) is unpublished | Guarantee-half is by construction (CKDA window); control-half needs training | Training track → riir-train Plan 459 Phase 1 |
| Ternary × oscillation interaction | risk (uncovered by paper) | Our PQ2 lane is unique here — rotary channels carry phase; ternary {-1,0,1} plausibly distorts angles | No — measurement task | riir-train Plan 459 Phase 3 |

**Signal-diff on the "covered by 169/371" dismissal (§3.6):** Research 169 (LinOSS) ships *free* oscillation (imaginary-axis eigenvalues) as architecture; Research 371 covers *mean-field regime classification* of low-rank RNNs with activity-dependent adaptation. Neither provides (a) the damping-regime knob ζ as a design axis with the controlled comparison, (b) *input-conditional* constructed phase codes (E.1's conditional rotation), or (c) the accumulation complex-gap bound. The delta survives one read of both cousins' mechanisms.

**No-GD advocate (19 candidates) merged, with one correction:** the advocate's "complex-plane KV-cache eviction gain" consumer claim is WRONG for our attention KV — `kv_evict` (Issue 012) works on post-softmax attention rows, which carry **no per-entry (ρ,θ)**; the bound applies to *recurrence channels* with known (ρ,θ): GDN heads, belief channels, future SSM-state caches. Corrected consumers: GDN state budget + belief kernels. All other candidates retained with gates in the issues/plans.

---

## 3. Distillation

### 3.1 The rotation/erase duality (the fusion finding)

Research 070's tables show Mamba-3(MIMO) beating GDN on some RULER cells (S-NIAH-3 @2K: 72.4 vs 54.2) while GDN2's erase gate recovers it (89.8) **without oscillation**. This paper explains both sides with one mechanism: collinear state accumulation. Two remedy families exist:

- **Rotate** (Mamba-3): the complex gap `|1-ρe^{iθ}|` moderates constructive accumulation — retained history and incoming updates stop adding collinearly.
- **Erase**: remove accumulated mass before it compounds. **Plain GDN already carries the COARSE form of this remedy** — the delta term `(I − βk̃k̃ᵀ)` is a rank-1 keyed erase along the current key, which is exactly why GDN does not accumulate collinearly the way plain linear attention does (correction at verdict round 1: an earlier draft said the fleet "has neither" remedy — false; the delta rule IS an erase). GDN2 refines it into channel-wise erase plus a separate write gate.

Same disease, two cures, orthogonal axes (phase vs write-selectivity), with plain GDN holding the crudest erase. Design consequence: our Bonsai fleet carries the coarse erase (one remedy, the crude one) and **no rotation** — it is exposed on the oscillation axis only, not doubly. Caveat bounding even that exposure: our deltanet lane is a **hybrid** (GDN + full-attention layers, Qwen3.5-style per `riir-infer/src/deltanet/mod.rs`) — the attention layers absorb part of the state-tracking load, so "the paper's control arm" is an **upper bound** on the fleet's exposure (Bonsai's exact layer mix unverified at verdict time).

### 3.2 Modelless extraction candidates (merged, gates in filed files)

| # | Candidate | Consumer | Gate sketch |
|---|---|---|---|
| 1 | ζ/ω₀ diagnostic + spectral-class spec test | GDN lane, belief kernels, HLA | Spec test asserts Bonsai GDN row: **discrete spectrum `{g(1−β), g}`** (real, non-oscillatory — the App. A.4 two-rate row; ⚠ NOT `{g, 0}`, which is the A.3 DeltaNet row); closed-form ζ on the continuous-time form; classifier arms fire on positive AND negative fixtures |
| 2 | State-accumulation bound monitor | **No shipped consumer** (verdict round 2: GDN = delta-rule bounded; belief kernels = convex EMA bounded; refine EMA = convex interpolation bounded — all verified against source) | Held as available math for future plain additive-write (linear-attention-class) lanes; bound-vs-simulation gate on a (ρ,θ) grid applies WHEN a consumer lane exists. GDN gets the spectral-class spec test + report-only `‖S‖/max‖v‖` ratio probe instead (issue 046) |
| 3 | Rank-collapse readout (mode persistence) | GDN state S | Injected synthetic streams of known ζ recovered within tolerance |
| 4 | Phase-code counters (constructed) | FSM/drive stack, belief channels | cos(φ_t) matches count mod m over 10⁶ random streams; FSM-interaction test asserts PROGRESS (no ringing-induced flip-flop — the house real-boot rule) |
| 5 | Underdamped κ-block belief channel | katgpt-sense `evolve_belief` | Parity/mod-3 fixture win over the leaky kernel at matched state size (gate ≥20 pts; paper gap ~30×); G3 θ=0 bit-identity with incumbent |
| 6 | Write-read phase-consistency law | EMA direction lane, belief write/read seams | Mismatched-frame projection degrades monotonically in relative phase drift; negative-control fixture fires |
| 7 | Parity/mod-3 fixture suite | eval harness for all recurrent surfaces | Count-pinned floors; a-priori selector: counter-shaped workload ⇒ underdamped before building |
| 8 | Resonance predictor (forced-response curve) | belief budgets vs fixed-cadence game systems | Peak location matches closed form across cadence sweep; report-only first |

### 3.3 Game-context reframe

Per-NPC belief kernels are leaky integrators — magnitude-only memory. The underdamped κ-block converts a belief channel into **phase-carrying belief: memory of *when*, not just *how much*** (last-encounter phase, cadence, periodicity); constructed phase counters give mod-m event-count drives (`sigmoid(cos φ)` — smooth, house-compliant, no hard edges) for combo cadence / aggro cycles / spawn patterns; the accumulation bound is a saturation early-warning for long-memory NPCs and fixed-cadence systems that resonate with a belief channel (issue 939 carries the first two).

### 3.4 Healer-context reframe (priority #2, same evidence bar)

(a) Retrieval index/corpus: the accumulation bound applies to EMA-updated latent direction vectors **only in the unnormalized `h = ρh + u` form** — correction at verdict round 2: the shipped refine `ema_step` (evolve.rs:94) is the convex interpolation `dir + α(fix−dir)` with clamp, steady state = fix_dir, bounded by construction; the bound is recorded as available math should any future lane accumulate unnormalized updates. The MDN discriminant over the EMA hyperparameter grid would give a regime map artifact (recorded, no file). (b) Fix-trajectory memory/selection: no rotation concept — no application found. (c) Score-bench/eval: the parity/mod-3 fixtures could stress-test any future recurrent component of the healer, none exists today. Honest summary: weak; the regime-map is the one transferable piece.

### 3.5 Closest cousins (2–3) and why the claim survives

1. **Research 169 (LinOSS):** free oscillation as architecture; no damping-regime axis, no constructed phase codes, no accumulation bound. Complementary — 189 shipped the oscillator; this adds the *regime placement* discipline.
2. **Research 371 (oscillatory DMFT):** mean-field theory of low-rank RNN oscillation with adaptation; population-level regime classification, not per-channel closed-form diagnostics of delta-rule recurrences.
3. **Research 070 (GDN2):** the erase gate; its Mamba-3 tables are what this paper's framework explains (§3.1 duality). No spectral taxonomy.
4. **Published:** Complex KDA (arXiv:2609.24797 — rotation realizability in KDA), Momentum DeltaNet (arXiv:2605.05838), Selective RoPE (arXiv:2511.17388), Grazzi negative eigenvalues (arXiv:2411.12537), Stuffed Mamba (arXiv:2410.07145 — state-oversizing pathology, no complex-gap bound). The oscillation-in-linear-attention *architecture* class is dense — this kills any architecture-novelty claim and is why the training plan is probe-shaped, not architecture-claiming.

---

## 4. Verdict

**GAIN** (not Super-GOAT, not GOAT-as-new-primitive).

One-line reasoning: the oscillation axis is saturated internally (169/371) and published (CKDA/MDN/Selective RoPE) — Q1 fails for the architecture class; the paper's own novel content (damping-regime framework, GDN-overdamped proof) is *analysis we consume*, and our extractables are diagnostics/monitors/fixtures (engineering gain, real but not a new capability class) plus one unpublished *measurement* datapoint (GDN2 underdamped-mode instrumentation) that rides the training plan.

Per-track (no cascade):
- **Track a (modelless):** GAIN — riir-infer issue 046 (monitor + spec test), katgpt-rs issue 939 (phase-carrying belief + counters).
- **Track b (self-adaptive):** recorded — phase-trajectory multi-timescale observation + EMA regime map; no file (no consumer pull today).
- **Track c (model-based):** GAIN — riir-train Plan 459 (Phase 1 cheap probe + GDN2 instrumentation; Phases 2–3 owner-gated rungs). LoRA post-hoc oscillation explicitly killed: the recurrence generator is structural, unreachable by low-rank deltas.

**§3.6 PoC check:** no "already ships"/parity/quality-parity claims are made — PoC not mandatory. The monitor issue carries its own negative-control gate; the training plan's Phase 1 IS the cheapest falsifiable PoC (≤10M params, hours).

### MOAT gate per domain

| Domain | Fit | Call |
|---|---|---|
| katgpt-rs | Belief-kernel + phase-code primitives; the note itself | Fits (HLA/attention/recurrent-state factory rows; 169/70 precedent) — issue 939 filed here |
| riir-infer | GDN inference substrate owns the monitor + spec test | Fits (public substrate moat; the deltanet lane runs the fleet) — issue 046 filed here |
| riir-train | From-scratch probe + GDN2 instrumentation + ternary-retention | Fits (active training moat; deltanet backward lane already hosts) — Plan 459 filed here |
| riir-ai | HLA consumer of the taxonomy; belief kernels sense-side | Downstream consumer only — no file |
| riir-refine | Weak-to-moderate (EMA regime map) | Recorded only, per §3.4 |

### PASS-Redirects

N/A (Gain verdict) — cousin cross-refs added instead: one-line pointer added to Research 169 and Research 70 this session.

---

## 5. Provenance

- **Verdict round 2 (2026-10-13, Claude reviewer): REVISE (minor) → 2 corrections applied.** (1) §3.2 candidate-1 row still carried the stale `{g,0}` spectrum — corrected to `{g(1−β), g}`. (2) The accumulation-bound consumer claim moved one layer down and was wrong again: `evolve_belief_additive` is a convex EMA (reconstruction.rs:1072: `(α·belief + (1−α)·drive).clamp(-1,1)`, steady state = drive) — verified against source, as was refine's `ema_step` (evolve.rs:94: convex interpolation + clamp). Verdict: **NO shipped consumer for the additive-write bound** — every recurrent accumulator in the workspace is bounded by construction; the bound is recorded as available math for future plain linear-attention lanes only. Path 0 rows, §3.2 candidate 2, and §3.4 rescooped accordingly.
- **Verdict round 1 (2026-10-13, Claude reviewer): REVISE → 5 corrections applied.** (1) Issue 046's `1/(1−g)` hazard premise was FALSE for the delta rule — key-direction steady state `β‖v‖/(1−g+gβ) ≤ ‖v‖` (the delta write IS a keyed erase); issue rewritten, monitor demoted to report-only probe, spectrum corrected to `{g(1−β), g}`. (2) §3.1 "fleet has neither remedy" overstated — the delta term is the COARSE keyed erase; duality restated + hybrid-layer caveat added. (3) Plan 459 T1.4 rescoped to the windowed-product spectrum (single-step NO is algebra, not a measurement). (4) Issue 939 P1 reformulated to the exact integer counter (f32 phase accumulation drifts); P2 demoted to sanity check (parity win is by construction); consumer seam corrected to `reconstruction.rs` + defer-until-pull posture. (5) Path 0 accumulation-bound row rescooped (superseded by round 2's full no-consumer verdict).
- Panel: No-GD advocate + Model-based advocate (spawn_agent, 2026-10-13) — briefs merged into §2/§3.2 with the KV-eviction correction noted; discards: none silent (the one consumer overstatement is corrected in-line rather than discarded).
- Prior-art web sweep (spawn_agent, 2026-10-13): 9 searches across 4 families; verdict — **no pre-emption** of the damping-regime framework or GDN-overdamped classification; landscape dense (cited above); OpenReview discussion of 2610.13341 not mined (residual risk: reviewer-surfaced prior art).
- Internal-first grep: arXiv ID 2610.13341 + "Oscillatory Dynamics|Mamba-3" across all workspace `.research/` — no prior distillation of THIS paper; cousins 169/371/070 read in full.
- Dual-allocation gate: katgpt-rs 0 collisions, riir-train 0 collisions (run at allocation time, 2026-10-13).
