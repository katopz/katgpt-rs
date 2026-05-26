# Plan 141: Data Probe Diagnostics — Typical-Set Regime Classification

**Research:** 102 (Data Probes — Synthetic Sequence Diagnostics)
**Status:** ✅ COMPLETE (D1-D4 + GOAT G1-G6 done)
**Feature Gate:** `data_probe` (off by default, opt-in GOAT proof)
**Depends On:** None (pure modelless, no bandit dependency)
**Domain:** katgpt-rs (modelless core)

---

## Motivation

Research 102 distills ICML 2026 paper on data probes — synthetic sequences from known random processes that enable systematic study of how data characteristics affect model behavior. The key distillable component is the **typical-set regime classifier**: given a known reference distribution with entropy rate H, classify model output quality as over-conservative / typical / uncertain based on NLL relative to H±ε.

Our existing infrastructure already computes Shannon entropy (`token_entropy`), tracks entropy anomalies (`EntropyAnomalySummary`), and scores underspecification. What's missing:
1. **Markov chain probe generator** — controlled synthetic data from known distributions
2. **Typical-set regime classification** — three-way label instead of binary high/low
3. **NLL computation against known distributions** — ground truth quality metric
4. **Formal claim card infrastructure** — structured IV/EV tracking for GOAT proofs

This is a **methodology upgrade** that improves all future GOAT proofs by providing formal C1–C4 validation criteria. Not a GOAT pillar itself (no game-specific knowledge, paper is public).

---

## Architecture

```
src/data_probe/
├── mod.rs              — Public API + re-exports
├── markov.rs           — Dirichlet-sampled transition matrix generator
├── typical_set.rs      — Three-regime classifier (conservative/typical/uncertain)
├── nll.rs              — NLL computation against known Markov chain
└── claim.rs            — Claim card struct (C1–C4 criteria, IV/EV verdicts)
```

### Dependency Graph

```
markov.rs ──→ nll.rs ──→ typical_set.rs
                                    ↓
                              claim.rs (uses all above)
```

---

## Distillation Tasks

- [x] **D1: Markov Chain Probe Generator (`markov.rs`)**

Generate transition matrices from Dirichlet distribution, compute entropy rate, select by target entropy.

```rust
/// A Markov chain with known transition matrix and computed properties.
pub struct MarkovChain {
    /// Transition matrix P[i][j] = Pr(next=j | current=i).
    transition: Vec<Vec<f32>>,
    /// Stationary distribution π.
    stationary: Vec<f32>,
    /// Computed entropy rate H(P) = -Σᵢ πᵢ Σⱼ Pᵢⱼ log Pᵢⱼ.
    entropy_rate: f32,
    /// Number of states (= vocabulary size for probe-LLM).
    num_states: usize,
}

/// Generate a Markov chain with entropy rate closest to `target_h`.
///
/// Samples `n_candidates` transition matrices from Dirichlet(α, ..., α),
/// computes entropy rate for each, returns the one closest to `target_h`.
pub fn generate_markov_chain(
    num_states: usize,
    target_h: f32,
    alpha: f32,
    n_candidates: usize,
    rng: &mut impl Rng,
) -> MarkovChain;

/// Sample a sequence of length `n` from the Markov chain.
pub fn sample_sequence(chain: &MarkovChain, n: usize, rng: &mut impl Rng) -> Vec<usize>;
```

**GOAT test:** Generated chain's empirical entropy (measured from 10K sampled sequences) is within 5% of computed `entropy_rate`.

- [x] **D2: NLL Computation (`nll.rs`)**

Compute negative log-likelihood of a sequence against the known Markov chain distribution.

```rust
/// Average NLL of sequence against Markov chain: -log p(xⁿ)/n.
pub fn average_nll(chain: &MarkovChain, sequence: &[usize]) -> f32;

/// Full NLL profile: per-position log-probabilities.
pub fn nll_profile(chain: &MarkovChain, sequence: &[usize]) -> Vec<f32>;
```

**GOAT test:** NLL of 10K sequences from `sample_sequence` converges to `chain.entropy_rate` within ε=0.1.

- [x] **D3: Typical-Set Regime Classifier (`typical_set.rs`)**

Three-way classification based on NLL relative to entropy rate.

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Regime {
    /// NLL < H - ε: over-conservative, repetitive, mode-collapsed.
    Conservative,
    /// H - ε ≤ NLL ≤ H + ε: well-calibrated, meaningful output.
    Typical,
    /// NLL > H + ε: uncertain, hallucinated, off-distribution.
    Uncertain,
}

/// Classify a sequence's regime against a known reference distribution.
pub fn classify_regime(
    chain: &MarkovChain,
    sequence: &[usize],
    epsilon: f32,
) -> Regime;

/// Batch regime classification with statistics.
pub fn regime_distribution(
    chain: &MarkovChain,
    sequences: &[Vec<usize>],
    epsilon: f32,
) -> RegimeDistribution;

/// Summary of regime labels across many sequences.
pub struct RegimeDistribution {
    pub n_conservative: usize,
    pub n_typical: usize,
    pub n_uncertain: usize,
    pub mean_nll: f32,
}
```

**GOAT test:**
1. Greedy sampling from a chain with low-entropy transitions → Conservative regime > 80%.
2. Sampling with T≈1 from same chain → Typical regime > 60%.
3. Random uniform sampling → Uncertain regime > 80%.

- [x] **D4: Claim Card Infrastructure (`claim.rs`)**

Structured claim tracking for formal C1–C4 validation.

```rust
/// A formal claim card following the data-probe protocol.
pub struct ClaimCard {
    /// Human-readable claim description.
    pub claim: String,
    /// C1: Known process (reference to probe generator).
    pub process_description: String,
    /// C2: Intervention knob and contrast values.
    pub intervention: Intervention,
    /// C3: Diagnostic metric name.
    pub diagnostic: String,
    /// C4: Pre-declared falsification condition.
    pub falsification_condition: String,
    /// Result: IV verdict (probe-side).
    pub internal_validity: Option<ValidityVerdict>,
    /// Result: EV verdict (real-side transfer).
    pub external_validity: Option<ValidityVerdict>,
}

pub struct Intervention {
    /// Name of the knob being varied.
    pub knob: String,
    /// Baseline value.
    pub baseline: String,
    /// Intervention value.
    pub treatment: String,
    /// Expected direction: +1 (increase) or -1 (decrease).
    pub expected_direction: i8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidityVerdict {
    /// IV(h)=1, EV(h)=1 — transfer supported.
    TransferAccepted,
    /// IV(h)=1, EV(h)=0 — probe-local result only.
    ProbeLocal,
    /// IV(h)=0 — claim rejected under declared criterion.
    Rejected,
}

impl ClaimCard {
    /// Compute the overall transfer verdict.
    pub fn verdict(&self) -> ValidityVerdict;
}
```

**GOAT test:** Round-trip test — create claim card, set IV+EV=1, assert verdict is TransferAccepted.

---

## GOAT Proof Summary (Target: 6/6)

| # | Test | Threshold | Pass Condition |
|---|------|-----------|----------------|
| G1 | Markov entropy accuracy | ±5% | Empirical entropy ≈ computed entropy_rate |
| G2 | NLL convergence | ε=0.1 | Mean NLL of 10K samples → entropy_rate |
| G3 | Regime classification: greedy→Conservative | >80% | Conservative label on greedy samples |
| G4 | Regime classification: T=1→Typical | >60% | Typical label on T=1 samples |
| G5 | Regime classification: uniform→Uncertain | >80% | Uncertain label on random samples |
| G6 | Claim card round-trip | exact | IV+EV=1 → TransferAccepted |

---

## Feature Gate

```toml
# In katgpt-rs/Cargo.toml
[features]
data_probe = []  # Data probe diagnostics — Markov probes + typical-set regime classification (Research 102)
```

No dependencies on `bandit`, no game-specific code. Pure modelless diagnostics.

### Feature Gate Rationale

**Why off by default?** Data probes are a methodology/research tool, not needed for production inference. Users who want formal C1–C4 GOAT proofs or regime-based diagnostics opt in.

**Why in katgpt-rs not riir-ai?** The probe generator and classifier are generic (no game knowledge). The microscope is MIT; the slides are private. Game-specific probe generators (Bomber FSM, Go state) would go in riir-ai, but they're not part of this plan.

---

## Integration Points

### Existing Code Touched

1. **`src/lib.rs`** — add `pub mod data_probe;` behind `#[cfg(feature = "data_probe")]`
2. **`Cargo.toml`** — add `data_probe = []` feature

### Future riir-ai Integration (NOT this plan)

1. **Bomber FSM probe** — use Bomber FSM transition matrix as known reference distribution
2. **Go position entropy** — compute typical-set thresholds for Go state sequences
3. **LoRA calibration** — train probe-LLM with LoRA, measure regime distribution shift

These would be riir-ai plans referencing this infrastructure.

---

## Scope Estimate

| Task | Lines | Time |
|------|-------|------|
| D1: Markov chain generator | ~200 | 2 hours |
| D2: NLL computation | ~100 | 1 hour |
| D3: Typical-set regime classifier | ~80 | 1 hour |
| D4: Claim card infrastructure | ~150 | 2 hours |
| GOAT tests (6) | ~250 | 2 hours |
| Example (`data_probe_demo`) | ~100 | 1 hour |
| **Total** | **~880** | **~9 hours** |

---

## What This Plan Does NOT Do

- Train a probe-LLM (research methodology, not production code)
- Replace existing `EntropyAnomalySummary` or `token_entropy()` (complementary)
- Add game-specific probe generators (those go in riir-ai)
- Replace PPoT's `identify_high_entropy_positions()` (but regime labels could augment it)
- Implement PCFG or hierarchical probes (future work per paper authors)

---

## Cross-References

- **Research 102:** This plan
- **Research 061 (Entropy Anomaly):** Data probes extend entropy anomaly with known-reference comparisons
- **Research 037 (REAP Modelless):** Same modelless/model-based taxonomy
- **Research 076 (SR²AM):** Probe-calibrated entropy thresholds for configurator
- **27_mmo_goat_pillars_decision_matrix.md:** NOT a pillar — pure diagnostics, no game IP
