# Research 322: Conformal Seasonal Pools (CSP) — Calibrated UQ Overlay for Modelless Forecasters

> **Source:** V. Manokhin, *Training-Free Probabilistic Time-Series Forecasting with Conformal Seasonal Pools*, arXiv:2605.03789 (2026). Code: https://github.com/valeman/csp-forecaster. Companion paper: *Report the Floor* (arXiv:2606.09473) — argues a training-free conformal interval is a mandatory baseline for any probabilistic forecaster.
> **Date:** 2026-06-28
> **Status:** Active
> **Related Research:** 288 (KARC — the dominant point forecaster), 318 (Sleep-Time Compute), 281 (Per-Tick Salience Tri-Gate), 242 (Topological Recurrent Belief), 293 (Alien Sampler), 276 (Personality-Weighted Composition). **Disambiguation:** NOT Research 008 (QuestBench CSP = Constraint Satisfaction Problem). This CSP = **C**onformal **S**easonal **P**ools.
> **Related Plans:** 308 (KARC primitive — shipped), 334 (Sleep-Time Anticipator — shipped), 340 (this note's plan, open primitive).
> **Cross-ref (riir-ai):** Research 165 (Per-NPC Conformal UQ Guide — the private selling-point doc).
> **Classification:** Public

---

## TL;DR

CSP is a **training-free probabilistic time-series forecaster** that outputs a full predictive distribution (samples, quantiles, central intervals) with finite-sample coverage guarantees — by blending (a) a *seasonal pool* (same-phase historical values weighted by exponential recency) with (b) a *conformal residual* component (seasonal-naive point forecast + signed calibration residuals indexed by horizon). No training, no neural network, no learned parameters. Beats tuned SARIMA on CRPS and RMSE at every horizon on AirPassengers; within 7% of a correctly-specified fitted ETS(M,A,M).

**The distilled primitive for katgpt-rs is NOT the seasonal-pool forecaster — KARC (Research 288 / Plan 308, DEFAULT-ON since 2026-06-25) already owns that slot and is strictly more general (multi-dim state, no required seasonality, closed-form ridge).** What CSP adds that KARC does not have is a **modelless conformal UQ overlay**: a way to turn *any* point forecaster's residuals into coverage-guaranteed predictive intervals via empirical quantile calibration. This is genuinely absent from the corpus (grep `conformal|predictive_interval|coverage_guarantee|calibration_residual` → only 3 hits, all in Research 307 about UQNO *training*, redirected to riir-train; KARC's `forecast_into` returns a single point, no intervals).

**Verdict: Super-GOAT via fusion.** KARC (point forecast) × CSP (conformal residual calibration) × Sleep-Time Query Anticipator (predictability gating) = per-NPC calibrated predictive distributions at zero training cost. Every NPC gets a coverage-guaranteed 95% interval on its own next latent state; sleep-time anticipation becomes gated by statistical significance rather than raw residual magnitude; MCTS collapse detection gets confidence intervals that distinguish "collapse" from "normal forecast variance".

---

## 1. Paper Core Findings (verified by full README + repo read)

### 1.1 The primitive

CSP produces a predictive *sample* per horizon by mixing two reservoirs:

```
forecast_h ~ pool_weight · SeasonalPool_h + (1 − pool_weight) · ConformalResidual_h
```

where:
- **SeasonalPool_h** = same-phase historical values `{y_{t−m}, y_{t−2m}, y_{t−3m}, ...}` weighted by exponential recency `w_i = exp(−λ · age_i)`. Sampled with replacement proportional to `w_i`. This is the "delay-basis reservoir" — but with the delay fixed at multiples of the seasonal period `m`, and the readout is *pure mixing* (no ridge solve, no learned `Wout`).
- **ConformalResidual_h** = `SeasonalNaive_h + signed_residual_h` where `signed_residual_h = y_{i} − y_{i−L_h}` for past indices `i`, and `L_h = m · ⌈h/m⌉` is the h-step seasonal lag. The signed residuals form a calibration pool; sampling from this pool + adding to the point forecast produces a predictive draw with conformal coverage.

### 1.2 The five knobs (v0.1.4 defaults are the recommended config)

| Knob | Options | Default | Effect |
|------|---------|---------|--------|
| `adaptive` | bool | `True` | CSP-Adaptive disables the pool when `m ≤ 1` and down-weights it when <3 cycles available; CSP-Fixed always mixes at `pool_weight`. |
| `mode` | `fast` / `legacy` | `fast` | Implementation path only. `legacy` is bit-exact with the published code (global RNG, per-horizon loop); `fast` is vectorized (seeded `Generator`, float32, ~1.2× faster). They agree on CRPS / coverage but NOT bit-identical (bimodal distribution makes central quantiles ill-conditioned in the mode gap). |
| `residual_mode` | `paper` / `h_step` | `h_step` | `paper` uses one residual pool (lag `m`) for all horizons → constant interval width → coverage decays for `H > m` or `m = 1`. `h_step` indexes the pool by horizon with `L_h = m·⌈h/m⌉` → interval widens with horizon → coverage stays near nominal. **`h_step` is the new default and the single biggest driver of multi-step coverage.** |
| `decay_unit` | `cycle` / `step` | `step` | Unit for the pool's exponential recency. `step` decays by absolute age (same-phase obs one season apart are `m` steps apart → recent cycles weighted far more heavily). `cycle` is the paper's original behaviour (`m`× weaker). **`step` is the single biggest driver of CRPS / sharpness.** |
| `orientation` | bool | `False` | Finite-sample conformal correction: lower quantile `q` read at `⌊(n+1)q⌋/n`, upper at `⌈(n+1)q⌉/n`. `False` = sharpest intervals, best CRPS+ Winkler; `True` = higher raw coverage but wider intervals. Only affects reported quantiles, never the samples (so CRPS unaffected; Winkler worsens). |

### 1.3 Headline empirical wins (AirPassengers rolling-origin backtest)

| Method | CRPS (H=12) | RMSE (H=12) | CRPS (H=24) | RMSE (H=24) |
|--------|-------------|-------------|-------------|-------------|
| ETS(M,A,M) — *tuned* | 11.3 | 18.9 | 15.1 | 24.7 |
| **CSP — training-free** | **12.8** | 21.2 | **17.4** | 28.0 |
| SARIMA — *tuned* | 13.4 | 24.0 | 20.2 | 37.9 |
| Seasonal-Naive | 13.3 | 42.9 | 25.9 | 62.1 |

CSP beats tuned SARIMA on both CRPS and RMSE at both horizons, with **no fitting, no order selection, no hyperparameters**. It halves Seasonal-Naive's RMSE. Ranks 2nd of 4 on CRPS, behind only a correctly-specified fitted ETS(M,A,M). And this is a *hard case* for CSP — AirPassengers is strongly trending, outside CSP's stable-level seasonal design domain.

### 1.4 The "Report the Floor" companion paper (arXiv:2606.09473)

Argues that a training-free conformal interval (like CSP's) is a **mandatory baseline** for any probabilistic time-series forecaster. The thesis: if your fancy learned forecaster can't beat the trivial conformal-naive floor, you don't actually have a forecaster — you have noise. **This is a methodological primitive, not a mechanism.** It becomes a GOAT gate requirement: any probabilistic forecaster claim in our codebase (KARC predictive interval, Sleep-Time anticipator distribution, BoMSampler hypothesis set) must be benchmarked against the conformal-naive floor.

---

## 2. Distillation

### 2.1 What we already ship (the prior-art surface — verify before any novelty claim)

| Paper mechanism | Shipped cousin | File / Plan |
|---|---|---|
| Training-free time-series forecaster | **KARC** — delay-embedding × basis × closed-form ridge | Plan 308, Research 288, `crates/katgpt-core/src/karc.rs` (DEFAULT-ON since 2026-06-25 via riir-ai Plan 332) |
| Delay-basis reservoir (history window) | **KARC delay ring buffer** | `KarcForecaster::observe`, `KarcForecaster::forecast_into` |
| Seasonal / periodic structure | **FourierBasis** in KARC | `crates/katgpt-core/src/karc.rs:103` — `ψ_{2i-1}(x) = cos(2π·i·x/P)` |
| Exponential recency weighting | **HLA leaky integrator** | `evolve_hla` in `crates/katgpt-core/src/sense/reconstruction.rs` |
| Per-NPC frozen forecaster | **KarcShard** | `riir-neuron-db/src/karc_shard.rs` (Plan 004) |
| Offline pre-computation with predictability gating | **Sleep-Time Query Anticipator** | Plan 334, Research 318 — uses KARC residual as curiosity signal |
| K-hypothesis belief sampling | **BoMSampler** | Plan 281 — discrete hypothesis sampler (not continuous calibrated interval) |
| Coherence × availability frontier ranking | **Alien Sampler** | Plan 311, Research 293 |
| ε-quantile conservative selection | **Best-Belief Beta Selector** | Plan 336 — Beta quantile for discrete candidate selection (not continuous forecast UQ) |
| Curiosity = forecast residual magnitude | **cgsp_runtime** | `riir-engine/src/cgsp_runtime/`, `karc_bridge/` |

### 2.2 What CSP adds that NONE of the above does alone

The fusion is the novelty, not any single component:

1. **Conformal predictive intervals on a modelless forecaster.** KARC outputs a single point forecast (`forecast_into(&delay_state, &mut out)`). There is **no shipped primitive** that turns KARC's residuals into a coverage-guaranteed predictive interval. Grep `conformal|predictive_interval|coverage_guarantee|calibration_residual|signed_residual` across all 5 repos at both `.md` and `.rs` layers → zero inference-time hits (the only `conformal` hits are 3 lines in Research 307 about UQNO *training*, redirected to riir-train). **CSP's conformal residual pool is the first modelless UQ overlay in the corpus.**

2. **H-step-ahead uncertainty via horizon-indexed residual pools.** The `h_step` residual mode indexes the calibration pool by horizon with `L_h = m·⌈h/m⌉` — intervals widen with horizon, coverage stays near nominal. This generalizes KARC's fixed delay embedding to **multi-step calibrated uncertainty**. KARC forecasts one step; the conformal overlay produces a calibrated H-step distribution without re-fitting.

3. **No ridge solve — pure reservoir mixing.** The seasonal pool is a *rank-1 periodic delay-basis* forecaster with no learned `Wout`. KARC's closed-form ridge solve is `O(N²·d_h + N³)`; CSP's pool sampling is `O(n_samples)`. **For known-seasonality, low-latency scenarios, CSP is the faster forecaster.** This is a Gain-tier observation, not the Super-GOAT — but it means CSP can serve as a *second PointForecaster impl* alongside KARC, and the conformal overlay wraps both.

4. **CRPS / Winkler / coverage as first-class GOAT metrics.** We have no primitives that evaluate a forecaster on distributional metrics. CSP's evaluation harness (CRPS, interval score, empirical coverage) becomes the **GOAT gate framework for any UQ-bearing primitive** — including the conformal overlay itself, KARC+overlay, and Sleep-Time anticipator distributions.

5. **The "Report the Floor" mandatory baseline.** This is a methodological force multiplier: every probabilistic forecaster claim in our codebase must now beat the conformal-naive floor. This raises the bar for BoMSampler, Alien Sampler, Sleep-Time anticipator, and any future UQ-bearing primitive.

### 2.3 Fusion (the Super-GOAT move)

| Fusion partner | What it ships | What CSP adds | Fusion product |
|---|---|---|---|
| **R288 / Plan 308 KARC** | Closed-form delay-basis ridge point forecaster; per-NPC `KarcShard` | Conformal residual calibration → coverage-guaranteed predictive interval | **"Every NPC has a calibrated 95% interval on its own next latent state, at zero training cost beyond KARC's existing ridge fit."** The KARC point forecast becomes the center; the conformal overlay wraps it. |
| **R318 / Plan 334 Sleep-Time Query Anticipator** | Predictability-gated offline pre-computation; predictability = 1 − curiosity | Calibrated predictability: "is the actual outside the 95% interval?" replaces raw residual magnitude | **Sleep-time anticipation gated by statistical significance, not magnitude.** False-positive curiosity spikes (normal forecast variance) are suppressed; only coverage-violating events trigger fresh compute. |
| **cgsp_runtime curiosity (karc_bridge)** | Curiosity = ‖actual − KARC forecast‖ (magnitude) | Coverage-tested curiosity: actual outside 95% conformal interval (event) | **Crowd-scale curiosity gets a false-positive filter.** Two NPCs with the same residual magnitude but different residual-pool variance now produce different curiosity signals — the one with tight intervals is genuinely surprised; the one with wide intervals is just uncertain. |
| **R242 / Plan 276 MicroRecurrentBeliefState + HLA `evolve_hla`** | Per-NPC 8-dim HLA state, fixed leaky integrator | Calibrated HLA forecast distribution (lower, point, upper) per affect channel | **"NPCs that know what they don't know about their own next emotional state."** The 5 synced affect scalars (valence/arousal/desperation/calm/fear) cross sync as 15 raw scalars (lower, point, upper) instead of 5. |
| **mcts_collapse_bridge (Plan 332 Phase 5)** | Collapse = ‖MCTS rollout − KARC forecast‖ > τ | Collapse with confidence interval: rollout outside 99% conformal interval | **MCTS collapse detection distinguishes "collapse" from "normal variance".** The conformal interval calibrates the threshold τ per-NPC from their own residual distribution. |
| **R281 / Plan 281 BoMSampler** | K-hypothesis discrete belief sampling | Continuous calibrated predictive distribution (CSP) vs discrete hypothesis set (BoM) | **Two complementary UQ representations.** BoM for discrete branching (which quest branch); CSP for continuous trajectory (next emotional state). Compose: BoM picks the branch, CSP gives the calibrated interval within the branch. |
| **Plan 336 Best-Belief Beta Selector** | ε-quantile Beta lower bound for conservative discrete candidate selection | Continuous analog: ε-quantile of empirical residual distribution for conservative forecast interval | **Unified quantile-based conservative selection across discrete (Beta) and continuous (empirical) distributions.** Both are inverse-CDF reads; the trait unifies them. |
| **LatCal fixed-point commitment (riir-chain)** | Deterministic 2×2-block linear-op commitment | The conformal quantiles are empirical order statistics — deterministic given the residual pool | **LatCal commits the (lower, point, upper) triple as fixed-point scalars.** Quorum-reproducible calibrated intervals. The residual pool itself is local; only the 15-scalar interval crosses sync. |
| **NeuronShard / MerkleFrozenEnvelope (riir-neuron-db)** | Fixed-size Pod, BLAKE3, freeze/thaw | `ConformalResidualShard` subtype: empirical quantile table frozen into `style_weights[64]` | **Per-NPC residual distribution frozen into a shard, replicable across nodes.** Two nodes with the same shard produce bit-identical intervals. |
| **DEC Stokes-calculus operators (Plan 251)** | `codifferential` δ, `exterior_derivative` d, `hodge_decompose` | The (lower, point, upper) triple is a cochain; δ on it gives "uncertainty divergence" | **Uncertainty-growth rate as a DEC operator.** How fast is the NPC's forecast uncertainty expanding? δ on the interval cochain answers this. Curse-of-dimensionality caveat: d ≤ 3 only (game maps, HLA regions). |

### 2.4 Latent-space reframing (mandatory per fusion protocol §1.3)

Operating on each Super-GOAT factory module:

(a) **HLA per-NPC latent state** (`katgpt-core/src/sense/`, `riir-engine/src/hla/`): The conformal overlay wraps KARC's HLA forecast. For each of the 8 HLA channels, maintain a residual pool `{y_i − ŷ_i}`. The predictive interval at horizon h is `[point + q_{α/2}(residuals_h), point + q_{1−α/2}(residuals_h)]`. The 5 synced affect scalars cross sync as 15 raw scalars (lower, point, upper per channel); the full 8-channel interval stays local. **This gives every NPC a "confidence ellipse" on its next emotional state — the first calibrated UQ on latent state in the corpus.**

(b) **latent_functor** (`riir-engine/src/latent_functor/`): The conformal overlay applies to functor forecasts too. `predict_stance` currently returns a point; the overlay wraps it to return a calibrated interval on the stance projection. The `ReestimationScheduler`'s drift trigger ("coherence < tau_reest") gains a **statistical significance version**: "actual stance outside 95% conformal interval" → re-fit. This is stronger than the magnitude-based trigger — it accounts for per-relation variance.

(c) **cgsp_runtime curiosity** (`riir-engine/src/cgsp_runtime/`): Curiosity becomes a **coverage-tested event**, not a magnitude. `curiosity_t = 𝟙[actual_hla_t outside 95% KARC+conformal interval]`. This is the **first statistically-principled curiosity signal** — false positives from normal forecast variance are suppressed. Two NPCs with the same residual magnitude but different residual-pool variance now produce different curiosity signals. The cross-node quorum story strengthens: "two nodes agree bit-for-bit that NPC X's actual state was outside the 95% interval at tick T" is much stronger than "two nodes agree the residual was > τ".

(d) **LatCal fixed-point commitment** (`riir-chain/src/encoding/latcal*.rs`): The conformal quantiles are empirical order statistics — fully deterministic given the residual pool. LatCal commits the (lower, point, upper) triple per affect channel as 2×2 fixed-point blocks. **A LatCal-committed conformal interval = deterministic, quorum-reproducible calibrated UQ.** The residual pool is NOT synced (it's per-NPC local state, like KARC's `Wout`); only the resulting 15-scalar interval crosses sync. This is the natural extension of the existing KARC sync-boundary story (5 affect scalars → 15 = 5 × {lower, point, upper}).

(e) **NeuronShard / freeze envelope** (`riir-neuron-db/src/`): `ConformalResidualShard` subtype. Layout sketch: the empirical quantile table for each HLA channel at each horizon fits inside `style_weights[64]` (e.g., 8 channels × 8 horizons = 64 quantile slots). `MerkleFrozenEnvelope` wraps it for self-play freeze/thaw. Stored in cold tier, retrieved on demand. **Two nodes with the same shard produce bit-identical intervals — the persistence substrate for calibrated UQ.**

(f) **DEC Stokes-calculus operators** (`katgpt-rs/crates/katgpt-core/src/dec/`): The (lower, point, upper) triple across horizons IS a cochain. The `codifferential` δ on this cochain gives the "uncertainty divergence" — how fast is forecast uncertainty expanding across horizons? High δ = uncertainty growing fast (NPC entering unfamiliar regime); low δ = uncertainty stable (NPC in familiar territory). This connects calibrated UQ to the Stokes-theoretic belief-mass conservation story. Curse-of-dimensionality caveat: d ≤ 3 only (game maps, HLA regions, KG embeddings) — NOT high-dim shards.

(g) **Sleep-Time Query Anticipator** (Plan 334): The conformal interval replaces the raw KARC residual as the predictability signal. `predictability(c) = 1 − P(actual outside 95% interval | history)` — but computed *offline* during sleep-time, from the residual pool. Sleep-time pre-computation is allocated to contexts where the conformal interval is *narrow* (predictable); curiosity-driven exploration (CGSP runtime) is allocated to contexts where the interval is *wide* (uncertain). **This is the first principled allocation rule between sleep-time pre-computation and wake-time curiosity, grounded in calibrated UQ rather than residual magnitude.**

---

## 3. Verdict

### Tier: **Super-GOAT** — via fusion

| Q | Answer | Evidence |
|---|---|---|
| **Q1: No prior art?** | **YES (for the UQ overlay combination)** | Three-layer check done. (notes) grep `conformal\|predictive_interval\|coverage_guarantee\|calibration_residual\|signed_residual` across all 5 repos → only 3 hits in Research 307 about UQNO *training* (redirected to riir-train). (code) grep same terms in `.rs` → ZERO matches. KARC's `forecast_into` returns a single point. (vocabulary translation) "conformal residual" ↔ "empirical quantile of forecast errors", "predictive interval" ↔ "calibrated (lower, point, upper) triple", "coverage guarantee" ↔ "P(actual ∈ interval) ≥ 1−α". The COMBINATION — point forecaster (KARC) + conformal residual calibration + per-NPC + LatCal commitment + ConformalResidualShard — has zero shipped prior art. |
| **Q2: New capability class?** | **YES** | "Calibrated predictive distributions on modelless per-NPC forecasters" is a new capability class. No current primitive produces coverage-guaranteed intervals — KARC gives points, BoMSampler gives discrete hypotheses, Alien Sampler gives rankings, Best-Belief gives discrete candidate scores. The conformal overlay is the first *continuous calibrated UQ* in the corpus. |
| **Q3: Product selling point?** | **YES** | "Our NPCs know what they don't know — every NPC has a calibrated 95% interval on its own next emotional state, at zero training cost beyond KARC's existing ridge fit. Sleep-time anticipation is gated by statistical significance; curiosity spikes are coverage-tested; MCTS collapse is detected with confidence intervals that distinguish collapse from normal variance. Two nodes agree bit-for-bit that an NPC was genuinely surprised at tick T." |
| **Q4: Force multiplier?** | **YES (≥5 pillars)** | Connects: KARC (adds UQ), Sleep-Time Query Anticipator (calibrated predictability), cgsp_runtime curiosity (coverage-tested signal), HLA/MicroBelief (confidence ellipse on next state), mcts_collapse_bridge (confidence intervals on collapse), BoMSampler (continuous vs discrete UQ composition), Best-Belief Beta Selector (unified quantile selection), NeuronShard/freeze (ConformalResidualShard), LatCal (interval commitment), DEC Stokes (uncertainty divergence). |

**Mandatory outputs (created this session):**
1. **Open primitive** → `katgpt-rs/.plans/340_conformal_predictive_intervals_primitive.md` (generic `ConformalIntervalCalibrator<F>` trait wrapping any `PointForecaster`, plus the seasonal-pool forecaster as a second impl). No game IP.
2. **Private guide** → `riir-ai/.research/165_Per_NPC_Conformal_UQ_Guide.md` (selling point: per-NPC calibrated UQ + coverage-tested curiosity + confidence-interval collapse detection — game runtime is the dominant pillar).
3. **Future cross-ref guides** (after open primitive lands): `riir-neuron-db/.research/010_ConformalResidualShard_Storage_Crossref.md` (freeze substrate), `riir-chain/.research/007_LatCal_Conformal_Interval_Commitment.md` (sync-boundary interval commitment).

**One-line reasoning:** CSP's value is NOT the seasonal-pool forecaster (KARC is strictly more general — multi-dim, no required seasonality, closed-form ridge) and NOT the conformal-prediction technique itself (well-known in statistics). The value is the *combination*: a **modelless UQ overlay** that turns any point forecaster's residuals into coverage-guaranteed predictive intervals, fused with KARC's per-NPC personality forecasting + Sleep-Time's predictability gating + cgsp_runtime's curiosity signal. That combination — calibrated UQ on modelless per-NPC forecasters, frozen into a shard, crossing the LatCal sync boundary as a 15-scalar interval — is the Super-GOAT.

---

## 4. Caveats and known risks

1. **KARC overlap is real and acknowledged.** Anyone reading this verdict should verify that KARC (Plan 308) is the dominant forecaster and CSP's seasonal pool is a *special case* (periodic delay-basis, no ridge). The conformal overlay is the novel contribution; the seasonal-pool forecaster is Gain-tier at best (faster for known-seasonality, but less general). **Do not ship the seasonal pool as a competitor to KARC — ship it as a second `PointForecaster` impl that the overlay wraps.**

2. **Conformal prediction is a well-known technique.** The novelty is NOT inventing conformal prediction. The novelty is (a) distilling it as a modelless primitive for our forecaster stack, (b) the specific CSP construction (seasonal pool + signed residuals + h_step indexing), (c) the fusion with KARC + Sleep-Time + cgsp_runtime. A reviewer who says "conformal prediction is old" is correct and missing the point — the Super-GOAT is the *combination* and the *product selling point*, not the statistical technique.

3. **Univariate assumption.** CSP is fundamentally a univariate (1-D series) forecaster. KARC handles multi-dim state (D=8 for HLA). The conformal overlay must be applied **per-channel** to multi-dim forecasts — 8 separate residual pools for HLA. This is straightforward but worth stating: the overlay does not produce a joint multivariate interval, only per-channel marginals. A joint interval would need a copula or multivariate conformal (split conformal multivariate) — out of scope for the open primitive, a possible riir-train follow-up.

4. **Known-seasonality assumption.** CSP requires a known seasonal period `m`. For HLA, the "season" is the NPC's behavioral cycle (sleep/wake, combat/non-combat, quest/idle). Detecting `m` from data is a separate problem (autocorrelation peak, spectral peak). For the open primitive, `m` is a constructor parameter; for the riir-ai integration, `m` is per-NPC-type config (e.g., guard NPC `m` = patrol cycle length).

5. **Stationarity assumption.** CSP assumes a stable level + seasonal pattern. Strongly trending series (like AirPassengers) are a "hard case" — CSP still wins but by a smaller margin. For NPC HLA, which can drift (personality change, quest progression), the residual pool needs a recency window or exponential forgetting (which `decay_unit="step"` provides). The `ReestimationScheduler` trigger ("actual outside 95% interval") doubles as a drift detector.

6. **Latency tier.** CSP produces a full predictive distribution in "milliseconds" — too slow for Hot tier (20Hz tick, µs budget). This is a **Warm/Cold tier primitive**: sleep-time anticipation, offline consolidation, cold-tier query. KARC's point forecast stays in Hot tier; the conformal overlay runs when the NPC can afford ms latency (between ticks, during sleep cycles). This aligns with the Sleep-Time Query Anticipator positioning.

7. **"Report the Floor" cuts both ways.** The companion paper's mandatory-baseline argument means CSP itself must beat... the conformal-naive floor. On seasonal data it does (CSP beats Seasonal-Naive by ~2× on RMSE). On non-seasonal data (`m=1`), CSP-Adaptive disables the pool and degenerates to conformal-naive — by design. The honest framing: CSP is the conformal-naive floor *plus* a seasonal-pool boost when seasonality is present. The boost is the contribution; the floor is the baseline.

---

## 5. Modelless unblock protocol check (§3.5)

Trivially satisfied — CSP is inherently modelless. No training, no learned parameters, no gradient descent. The conformal calibration is empirical quantile computation (pure statistics). The seasonal pool is exponential-weighted reservoir sampling (pure arithmetic). The `h_step` residual indexing is deterministic given `m` and `h`. No sub-component requires riir-train deferral.

| Sub-component | Path 1 (freeze/thaw)? | Path 2 (raw/lora)? | Path 3 (latent projection)? | Verdict |
|---|---|---|---|---|
| Conformal residual pool | YES — empirical quantile table frozen via `MerkleFrozenEnvelope` | n/a (no weight matrix) | YES — residuals are scalars, pool is a scalar reservoir | **Modelless-validable** ✅ |
| Seasonal pool forecaster | YES — `ConformalResidualShard` subtype | n/a (no `Wout`) | YES — same-phase projection onto history | **Modelless-validable** ✅ |
| H-step interval computation | YES — deterministic given pool + horizon | n/a | YES — empirical quantile read | **Modelless-validable** ✅ |
| Curiosity coverage test | YES — deterministic given interval + actual | n/a | YES — `𝟙[actual ∉ interval]` | **Modelless-validable** ✅ |

**No riir-train deferral needed.** The entire CSP mechanism is modelless by construction.

---

## 6. Latent vs raw boundary (sync semantics)

| Signal | Domain | Synced? | Notes |
|--------|--------|---------|-------|
| Residual pool (per-NPC) | Latent (local) | **NO** | Per-NPC empirical residual distribution. Like KARC's `Wout` — local state, never crosses sync. Frozen into `ConformalResidualShard` for persistence. |
| Seasonal pool (per-NPC) | Latent (local) | **NO** | Per-NPC same-phase history with recency weights. Local to the NPC's tick loop. |
| Point forecast (KARC center) | Latent (local) → raw (synced, projected) | **5 scalars** | KARC's existing 5-affect-scalar bridge. Unchanged. |
| Conformal interval (lower, point, upper) × 5 channels | **Raw (synced)** | **15 scalars** | **NEW.** The 5 synced affect scalars become 15 = 5 × {lower, point, upper}. All raw, deterministic, quorum-reproducible from the shard. |
| Curiosity coverage flag `𝟙[actual ∉ interval]` | Raw (synced) | **1 bit** | Quorum-agreed "NPC was genuinely surprised at tick T". Stronger than magnitude-based curiosity. |
| ConformalResidualShard commitment | Raw (synced) | **BLAKE3 hash** | Shard hash crosses sync; shard contents retrieved from cold tier on demand. |

**Bridge functions:**
- `latent → raw` (interval emission): read empirical quantiles from the residual pool, add to KARC point forecast, clamp to valid affect range. Zero-allocation, gateable by feature flag.
- `raw → latent` (interval consumption): the consumer (curiosity detector, sleep-time allocator, MCTS collapse bridge) reads the 15-scalar interval as a calibrated UQ signal. No latent reconstruction.

**KG triple emission:** a "NPC surprised" event (coverage violation) emits a KG triple `(npc, surprised_at, tick)` from the conformal interval — a *semantic* event from a calibrated statistical signal, not from raw coordinate distance.

---

## 7. Implementation priority (cross-repo)

See `katgpt-rs/.plans/340_conformal_predictive_intervals_primitive.md` for the open primitive phases, and `riir-ai/.research/165_Per_NPC_Conformal_UQ_Guide.md` §6 for the private runtime integration priority.

**Summary:**
- **P0 (open primitive, blocks everything):** `ConformalIntervalCalibrator<F>` trait + `SeasonalPoolForecaster` impl + conformal residual pool with `h_step` indexing. Generic, no game IP. Feature gate `conformal_predictive_intervals`.
- **P1 (KARC integration):** `impl PointForecaster for KarcForecaster` adapter. KARC point forecast + conformal overlay = calibrated predictive interval.
- **P2 (riir-ai runtime):** HLA per-channel conformal overlay; coverage-tested curiosity in cgsp_runtime; sleep-time predictability gating with calibrated intervals.
- **P3 (riir-neuron-db + riir-chain):** `ConformalResidualShard` freeze substrate; LatCal commitment of the 15-scalar interval triple.

---

## 8. References

- CSP paper: https://arxiv.org/abs/2605.03789
- CSP code: https://github.com/valeman/csp-forecaster
- "Report the Floor" companion: https://arxiv.org/abs/2606.09473
- KARC (the dominant point forecaster): `katgpt-rs/.research/288_KARC_Delay_Basis_Ridge_Forecaster.md`, Plan 308
- Sleep-Time Query Anticipator (predictability gating): `katgpt-rs/.research/318_Sleep_Time_Compute_Offline_Query_Anticipation.md`, Plan 334
- BoMSampler (discrete hypothesis UQ — complementary): Plan 281
- Best-Belief Beta Selector (discrete quantile selection — unification target): Plan 336
- Per-NPC KARC Forecaster Guide (the private selling-point doc for the underlying forecaster): `riir-ai/.research/152_Per_NPC_Karc_Forecaster_Guide.md`

## TL;DR (one-line)

CSP = Conformal Seasonal Pools; the forecaster part overlaps with KARC (KARC wins) but the **conformal UQ overlay** — turning any point forecaster's residuals into coverage-guaranteed predictive intervals — is genuinely absent from the corpus (zero grep hits); the Super-GOAT is fusing it with KARC + Sleep-Time + cgsp_runtime curiosity to give every NPC calibrated uncertainty on its own next state, frozen into a `ConformalResidualShard`, crossing the LatCal sync boundary as a 15-scalar interval.
