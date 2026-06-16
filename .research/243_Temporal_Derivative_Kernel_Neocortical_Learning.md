# Research 243: Temporal Derivative Kernel — Neocortical Learning Distilled

> **Source:** Randall C. O'Reilly, "This is how the Neocortex Learns", arXiv:2606.08720 (Jun 2026).
> Companion experimental paper: Jang, Flores, Zito & O'Reilly, "Synaptic Plasticity as a Function of the Temporal Derivative", bioRxiv 2026.06.05.730489.
> Blog primer: https://arxiviq.substack.com/p/this-is-how-the-neocortex-learns
> **Date:** 2026-06-16
> **Status:** Active
> **Related Research:** 242 (Topological State Tracking / `evolve_hla` prior art), 192 (NextLat belief-state), 024 (δ-Mem — Plan 053 prediction-error memory), 240 (SGS curiosity self-play), 212 (collapse-aware thinking)
> **Related Plans:** 276 (`MicroRecurrentBeliefState` — attractor variant of HLA), 053 (δ-Mem — prediction-error-driven memory writes), 212 (collapse detection), 274 (CGSP curiosity)
> **Classification:** Public
> **Classification note:** Paper's weight-update rule (kinase LTP/LTD) is training → riir-train. **This note distills ONLY the inference-time transferable primitive: the dual fast/slow temporal-derivative signal.**

---

## TL;DR

O'Reilly 2026 argues the neocortex approximates error backpropagation by representing the error gradient *implicitly* as the temporal derivative between two activation states (prediction "minus" phase → outcome "plus" phase) over a 200ms theta cycle. Biologically, the derivative is computed at every synapse as the difference between a fast and a slow leaky integrator of the same calcium-calmodulin driver (`I_fast − I_slow`, `τ_fast ≪ τ_slow`), mapped to a CaMKII-vs-DAPK1 kinase competition.

**Distilled for katgpt-rs (modelless, inference-time):**

The transferable primitive is **not** the weight update (that is training → riir-train). It is the **dual fast/slow temporal-derivative kernel** — a zero-allocation, branch-free, sigmoid-compatible operator that turns any streaming latent scalar into a signed "surprise" signal: positive when the signal is rising, negative when falling, zero when flat. Applied per-dimension per-entity, it gives every NPC / every decode head / every bandit arm an *intrinsic prediction-error channel* computed from its own state dynamics, with no external supervisor and no backprop.

This primitive is currently **absent from the shipped code**. Every EMA in the codebase is a *single* integrator (`simd_fused_decay_write`, `AdaptiveTraceCompactor::ema_entropy`, `BreakevenTracker::update_ema`, `evolve_hla` itself). The dual `(I_fast − I_slow)` band-pass derivative is a new, minimal, composable kernel.

**Fusion targets (why this matters here):**

1. **`evolve_hla` (Plan 221, Research 242)** — HLA tracks *what is*; the temporal derivative tracks *how fast it is changing*. The two compose into a full "neocortical-style" per-NPC state channel: belief + rate-of-belief.
2. **`DeltaMemoryState` (Plan 053)** — currently writes on *spatial* prediction error (query vs stored). Dual-fast/slow derivative gives a *temporal* write gate: consolidate to long-term memory only when the derivative norm is large (something surprising just happened). This is the kinase-competition analog — write is gated by the fast/slow mismatch.
3. **Collapse-aware thinking (Plan 212)** — entropy collapse is one signal; *prediction-derivative collapse* (all dims → 0 derivative) is an orthogonal one. The two together give a more robust detector.
4. **CGSP curiosity (Plan 274)** — CGSP uses `(1 − solve_rate) · guide_score`, which requires an external Solver. The temporal derivative is an *intrinsic* curiosity signal — large derivative = surprising = worth exploring — with no Solver call.

**Verdict: GOAT.** Novel primitive (no prior art in code or notes), provable gain (does adding the surprise signal improve collapse recovery, memory write quality, or curiosity diversity?), force-multiplies ≥4 existing pillars. NOT Super-GOAT: it upgrades existing systems rather than creating a new capability class, and the selling-point sentence ("our NPCs get surprised") is real but not yet a moat. See §3.

---

## 1. Paper Core Findings

### 1.1 The temporal derivative model

The paper's central claim (computational + algorithmic + implementational, per Marr):

- **Computational:** the only learning mechanism that has demonstrated scaling to human-level intelligence is error backpropagation. Everything else (Hebbian, STDP, BCM) is computationally insufficient for deep networks.
- **Algorithmic:** backprop can be implemented *without* dedicated error neurons by representing the error gradient **implicitly** as the temporal difference between two activation phases of the *same* neurons:
  - **Minus phase (prediction):** bidirectional constraint satisfaction settles to a predicted state `a_minus`.
  - **Plus phase (outcome):** a strong driver input overrides the prediction with the actual outcome `a_plus`.
  - **Error ≈ `a_plus − a_minus`** (same neurons, two time slices).
  - Synaptic update: `Δw_ij ∝ (a_j,plus − a_j,minus) · a_i,minus`.
- **Implementational:** the temporal derivative is computed locally at each synapse as the difference of two leaky integrators of the *same* calcium-calmodulin driver, with different time constants:
  - `dI_fast/dt = −I_fast/τ_fast + Ca(t)`  (tracks the recent outcome)
  - `dI_slow/dt = −I_slow/τ_slow + Ca(t)`  (retains a trace of the earlier prediction)
  - `Δw ∝ I_fast − I_slow`  (signed temporal derivative)
  - `τ_fast ≪ τ_slow`.
  - Mapped to competitive kinases: CaMKII (fast, drives LTP on positive derivative) vs DAPK1 (slow, drives LTD on negative derivative).

The biological clock is a 200ms theta cycle (5 Hz): one prediction+outcome learning cycle per ~5 Hz burst of layer-5b intrinsic-bursting neurons into the pulvinar/mediodorsal thalamus.

### 1.2 The three empirical signatures (Jang et al. 2026)

Stimulating pre/post pyramidal neurons with 25 Hz / 50 Hz split across the two 100 ms halves of a theta cycle:

| Pattern | Net Ca²⁺ | Result |
|---|---|---|
| 25 → 50 Hz (rising) | moderate | **LTP** |
| 50 → 25 Hz (falling) | moderate | **LTD** |
| 25 → 25 Hz (flat) | low | no change |
| 50 → 50 Hz (flat) | **highest** | **no change** |

The 50→50 case is the killer: classical Hebbian/BCM predicts maximum LTP (most co-activity), but zero plasticity is observed. Only the *temporal derivative* explains this. This is direct experimental evidence that biological synapses are sensitive to the derivative of activity, not its magnitude.

### 1.3 Why explicit-error models fail

Predictive coding (Rao & Ballard 1999) and target-prop (LeCun 1986) require *structurally segregated* populations of neurons coding for prediction vs outcome vs error, with separate feedforward/feedback pathways. The neocortex is not wired this way — it is pervasively bidirectional and redundantly encodes positive representations at every layer. The temporal-derivative framework keeps the error **implicit** (difference across time, not across neurons), which matches the anatomy.

### 1.4 BTSP — the rapid-readout auxiliary system

Behavioral Timescale Synaptic Plasticity (Magee 2026) is a separate, faster mechanism restricted to output neurons (CA1, layer-5 pyramidal). Distal dendritic plateau potentials establish seconds-long eligibility traces that let later motor/reinforcement signals drive plasticity on earlier representations. The paper frames this as complementary: slow temporal-derivative learning builds deep statistical representations; fast BTSP rapidly reads them out to satisfy current behavioral demands. This maps cleanly onto our two-layer split: `evolve_hla` (slow statistical state) vs `DeltaMemoryState` write gate (fast behavioral readout).

---

## 2. Distillation

### 2.1 What is transferable (modelless)

Strip away the weight update (→ riir-train) and the neurochemistry. What remains is a **signal-processing primitive** with four properties:

1. **Two leaky integrators on the same input, different time constants.** `τ_fast ≪ τ_slow`.
2. **The derivative is their difference:** `surprise(t) = I_fast(t) − I_slow(t)`.
3. **Sign matters:** positive surprise = "signal is rising" (outcome exceeded prediction); negative = "falling". Zero = steady-state.
4. **Local and streaming:** the derivative is computed from the scalar's own time series. No external target, no backprop, no second network.

This is mathematically a **band-pass filter** — a standard DSP primitive. The novelty is *not* the math. The novelty is:
- (a) **It does not exist anywhere in our shipped code** (every EMA is single-integrator; see §2.4 prior-art check).
- (b) **It composes with our existing latent-state stack** (HLA, δ-Mem, collapse, curiosity) to give each of them a *prediction-error channel they currently lack*.
- (c) **It is the correct bridge-function shape** for our Latent-vs-Raw rules: operates on latent state, emits a bounded scalar, zero-allocation, gateable.

### 2.2 What is NOT transferable (training → riir-train)

- The CaMKII/DAPK1 kinase competition as a *weight update rule*. This is biological backprop approximation → training method → riir-train.
- The phase-alternation (200 ms theta) as a *training schedule*. That's a training-time concern.
- Equilibrium propagation / GeneRec as a *learning algorithm*. Training.

If we ever want biological-plausibility-inspired *training*, that goes to `riir-train/.research/` as a separate note. This note is exclusively about the inference-time signal.

### 2.3 The primitive

```rust
/// Dual fast/slow leaky integrator — signed temporal derivative of a scalar.
///
/// Distilled from O'Reilly 2026 (arXiv:2606.08720) §Implementational.
/// Mathematically a band-pass filter: I_fast tracks recent signal (plus phase),
/// I_slow retains earlier trace (minus phase). Their difference is the
/// implicit prediction-error / "surprise" signal the neocortex uses for
/// credit assignment, computed locally with zero external target.
///
/// Storage: 2 × N × sizeof(f32) for N scalars. No allocations.
/// Per-step cost: 2 EMA updates + 1 subtract per scalar. SIMD-friendly.
#[derive(Clone, Debug)]
pub struct TemporalDerivativeKernel<const N: usize> {
    /// Fast integrator state — tracks the "outcome" (plus phase).
    pub fast: [f32; N],
    /// Slow integrator state — retains the "prediction" (minus phase).
    pub slow: [f32; N],
    /// EMA weight for fast path. Higher = shorter memory.
    /// Paper: τ_fast ≪ τ_slow, so α_fast > α_slow.
    /// Typical: α_fast = 0.3 (3-sample memory), α_slow = 0.03 (30-sample).
    pub alpha_fast: f32,
    pub alpha_slow: f32,
}

impl<const N: usize> TemporalDerivativeKernel<N> {
    /// Push a new sample, return the per-dimension signed surprise vector.
    ///
    /// Output interpretation:
    /// - large positive  → signal is rising fast (outcome > prediction)
    /// - large negative → signal is falling fast (prediction > outcome)
    /// - near zero       → steady state, no surprise
    ///
    /// For a bounded [0, 1] curiosity / attention gate, project via
    /// `sigmoid(β · surprise)` (never softmax — per AGENTS.md constraint).
    #[inline]
    pub fn observe(&mut self, signal: &[f32; N]) -> [f32; N] {
        let mut out = [0.0f32; N];
        for i in 0..N {
            self.fast[i] = self.alpha_fast * signal[i] + (1.0 - self.alpha_fast) * self.fast[i];
            self.slow[i] = self.alpha_slow * signal[i] + (1.0 - self.alpha_slow) * self.slow[i];
            out[i] = self.fast[i] - self.slow[i];
        }
        out
    }

    /// L2 norm of the surprise vector — scalar "how surprising was this step?"
    #[inline]
    pub fn surprise_norm(&self) -> f32 {
        let mut s = 0.0f32;
        for i in 0..N {
            let d = self.fast[i] - self.slow[i];
            s += d * d;
        }
        s.sqrt()
    }

    /// Reset integrators (e.g. on entity respawn).
    pub fn reset(&mut self) {
        self.fast.fill(0.0);
        self.slow.fill(0.0);
    }
}
```

**Why this shape is right for us:**

- **Fixed-size `[f32; N]`** — fits our "fixed-size arrays for bounded domains" rule. NPC HLA has N=8; decode-head entropy has N=1; bandit arm preference has N=k.
- **Sigmoid-compatible output** — the raw difference is signed; downstream consumers project via `sigmoid(β · surprise)` for gating. Never softmax (per AGENTS.md).
- **Zero-allocation** — all stack. Per-NPC cost at N=8 is 64 bytes (2 × 8 × 4) + 8 bytes config = 72 bytes. 1000 NPCs = 72 KB. Fits in L1.
- **Gateable** — feature-flagged; when disabled, the struct is not stored and `observe()` is not called.

### 2.4 Prior-art check (the part that decides GOAT vs Super-GOAT)

Per the skill's mandatory two-layer check (notes + shipped code):

**Notes layer:**
- Research 242 (`Topological_State_Tracking_Recurrent_Belief`) + Plan 276 (`MicroRecurrentBeliefState`) — explicitly cover `evolve_hla` as Family C (delta-rule SSM recurrent belief state). **Verdict was downgraded Super-GOAT → GOAT precisely because `evolve_hla` already implements the recurrent belief kernel.** The temporal-derivative *output channel* is not in scope of 242/276 — they extend HLA with *attractor dynamics* (a different update rule), not with a derivative observer.
- Research 192 (NextLat) + Plan 217 (BeliefDrafter) — predicts next hidden states for speculative decoding. This is a *learned MLP predictor*, not a streaming derivative observer. Different mechanism.
- Plan 053 (δ-Mem) — uses prediction error to gate memory writes. The error is computed as `query − stored` (a *spatial* difference, new input vs memory content). It is **not** a temporal derivative of the memory's own state. The two are complementary, not duplicates.
- Plan 212 (Collapse-aware) — detects collapse via an entropy ring-buffer. Single-signal, not derivative-based.
- Plan 274 (CGSP) — curiosity via `(1 − solve_rate) · guide_score`. Requires an external Solver.

**Code layer (mandatory — the layer that caught the 242 overclaim):**
- `simd_fused_decay_write(dst, decay, src, write)` — single EMA. Used in `tiled_attention_inner`, `parallax_attn`, `mobius_add_into`, `gegelu`, `silu`, `BreakevenTracker::update_ema`. **Every EMA in the codebase is single-integrator.**
- `AdaptiveTraceCompactor::ema_entropy` — single EMA on entropy with `DEFAULT_EMA_ALPHA = 0.1`.
- `BreakevenTracker::update_ema` — single fixed-point EMA on cost.
- `evolve_hla` (reconstruction.rs L625-648) — single-step additive update from current evidence. Does NOT compute the derivative of `self.hla` over time. Confirmed by reading the implementation.
- `simd_outer_product_ema_f64` (peira.rs) — single EMA on covariance.

**Grep for `fast_integrator|slow_integrator|tau_fast|tau_slow|Ifast|Islow` in `.rs` → NO MATCHES.**

**Conclusion:** The dual-fast/slow temporal-derivative primitive is novel as shipped code. It is also novel as a research note (no prior note frames the prediction-error signal this way). The closest cousins are δ-Mem (Plan 053, spatial error) and evolve_hla (single-step state tracking) — neither covers the streaming temporal-derivative signal.

### 2.5 Fusion (the GOAT-tier combination)

The primitive alone is Gain-tier (a useful signal). The GOAT-tier claim comes from fusing it with four existing systems:

| Fusion | Existing system | What the derivative adds | Gate |
|---|---|---|---|
| **F1: HLA + derivative** | `evolve_hla` (Plan 221, per-NPC 8-dim latent state) | A per-NPC 8-dim surprise vector — "which emotions are changing fast right now". HLA answers *what is*; derivative answers *how is it shifting*. | Does the surprise vector predict emotionally-significant game events (combat onset, encounter, loot) better than raw HLA magnitude? |
| **F2: δ-Mem write gate** | `DeltaMemoryState::write_segment` (Plan 053) | Temporal write gate: consolidate to long-term memory only when `surprise_norm()` > θ. Currently writes on every query; derivative-gated writes happen only on surprising events. | Does derivative-gated writing reduce memory noise without losing salient events? Target: ≥30% write reduction with ≤5% recall loss. |
| **F3: Collapse detector fusion** | `CollapseDetector` (Plan 212, entropy ring-buffer) | Orthogonal signal: prediction-derivative collapse (all dims → 0) vs entropy collapse (distribution → one-hot). The two together catch collapse modes the other misses. | Does adding the derivative signal reduce false-negative collapse events by ≥20% on a synthetic suite? |
| **F4: Intrinsic curiosity** | CGSP (Plan 274, `(1 − solve_rate) · guide_score`) | CGSP needs a Solver; the derivative does not. For each NPC / decode head / bandit arm, `sigmoid(β · surprise_norm())` is a zero-cost curiosity signal. | Does derivative-driven curiosity match or beat CGSP on exploration diversity (G2-style collapse recovery) at ≤10% of the cost? |

The Super-GOAT candidate (explicitly **not claimed** — flagged as a future possibility) would be a *unified surprise bus* that drives all four systems from one signal. That is a capability class — "every NPC has intrinsic neocortex-style motivation" — but we cannot honestly claim it until F1–F4 individually pass their gates. Per the skill's `candidate` rule, this is logged as fusion-potential, not as a Super-GOAT claim.

---

## 3. Verdict

**GOAT.** Novel primitive (no prior art in notes or code), provable gain (4 fusion gates above), force-multiplies ≥4 existing pillars (HLA, δ-Mem, collapse, curiosity). NOT Super-GOAT because:

- **Q1 (no prior art):** YES — confirmed by both-layer grep. The dual-fast/slow temporal derivative does not exist as shipped code or framed as a note.
- **Q2 (new capability class):** NO — it is a *better signal* for existing capabilities (curiosity, collapse, memory), not a new capability class. The "unified surprise bus" would be a new class, but that is fusion-potential, not a proven claim.
- **Q3 (selling point):** WEAK — "our NPCs get surprised" is real but not a moat until F1–F4 validate on a game benchmark.
- **Q4 (force multiplier):** YES — connects to HLA, δ-Mem, collapse, curiosity, KG triple emission (surprise events → triples).

Since Q2 fails, the Super-GOAT mandatory outputs (riir-ai guide, open primitive + plans) are not triggered. Outputs: this research note + `katgpt-rs/.plans/244_*.md` (open primitive). If F1–F4 pass and the "unified surprise bus" fusion proves out, escalate to Super-GOAT in a follow-up note (create `.issues/` to track).

**One-line reasoning:** The dual-fast/slow temporal-derivative kernel is a novel, zero-alloc, sigmoid-compatible signal that upgrades four existing pillars — exactly the shape of a GOAT-tier primitive. It does not create a new capability class on its own.

### Verdict tiers reference

| Tier | Criteria | Routing |
|---|---|---|
| Super-GOAT | Novel mechanism + new capability class + selling point + force multiplier | Open primitive + riir-ai guide + plans |
| **GOAT (this)** | **Provable gain over existing approach, not a new class. Promote if it wins.** | **Plan + implement + benchmark (this note + Plan 244).** |
| Gain | Incremental, useful but not headline | Plan only, behind flag |
| Pass | Not relevant, OR training-only | One-line note. (The weight-update half of this paper is Pass → riir-train.) |

---

## 4. Implementation Sketch (delegates to Plan 244)

1. **`TemporalDerivativeKernel<const N>`** in `crates/katgpt-core/src/temporal_deriv.rs` — generic, no game semantics. Feature flag `temporal_deriv`.
2. **SIMD variant** `observe_simd` reusing `simd_fused_decay_write` for the two EMA passes.
3. **`surprise_norm()`** — L2 norm of the derivative vector.
4. **Bridge helper** `sigmoid_surprise_gate(derivative, beta)` — the canonical downstream projection (sigmoid, never softmax).
5. **Fusion consumers** (Plan 244 phases 2–5): HLA companion, δ-Mem write gate, collapse-detector fusion, curiosity intrinsic signal. Each is independently feature-gated.
6. **Benchmarks** (Plan 244 GOAT gates G1–G4): kernel overhead (<10ns per N=8 observe), HLA companion accuracy on synthetic emotional-event traces, δ-Mem write reduction vs recall, collapse false-negative reduction.

**Latent vs raw boundary:** The derivative operates on latent state (HLA vector, entropy, bandit Q-values). Its output (`surprise_norm`, a scalar) may cross the sync boundary as a raw f32 — it is a *summary statistic*, not a per-entity truth, so it follows the flock-centroid rule (sync as-is because it's a summary). The full N-dim derivative vector stays local (per-entity latent, never synced).

---

## 5. Open Questions / Risks

1. **`τ_fast` / `τ_slow` tuning.** The paper uses 100ms/200ms (theta cycle). Our tick is 20Hz (50ms) for game, token-by-token for inference. The ratio `α_fast / α_slow` matters more than absolute values — paper suggests ~10×. Plan 244 sweeps α_fast ∈ {0.2, 0.3, 0.5}, α_slow ∈ {0.02, 0.03, 0.05}.
2. **Stationarity assumption.** The dual-EMA derivative assumes the signal's statistics are roughly stationary over the slow window. For non-stationary game events (sudden combat), the slow integrator lags — this is *desired* (that's what makes it a derivative), but consumers must understand the lag.
3. **Scale invariance.** Raw derivative magnitude depends on signal magnitude. Consumers should normalize (e.g. divide by `|slow| + ε`) before thresholding, or use the sigmoid projection which is scale-tolerant.
4. **Does the "unified surprise bus" actually cohere?** F1–F4 each use the derivative differently. It is possible that the right α for HLA-companion is wrong for δ-Mem-gate. If so, the "unified bus" Super-GOAT claim dies and we ship 4 independent consumers of the same primitive — still GOAT, but not the moat.
