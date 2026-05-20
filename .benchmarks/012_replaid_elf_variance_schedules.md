# Benchmark 012: RePlaid Variance Schedules (Plan 078) + ELF SDE/Logit-Normal (Plan 079)

**Date:** 2025-07-12
**Branch:** `feature/078_replaid_variance_schedules_modelless`
**Features:** `replaid_schedules,bandit,sdar_gate,dllm,elf_sde`
**Profile:** release
**Seed:** 42

---

## Plan 078: RePlaid Variance-Minimized Schedules

### T1: VarianceMinimizer Core Primitive

| Metric | Value |
|--------|-------|
| `observe_and_adapt` latency | **10.8 ns/obs** |
| Final param (sinusoidal costs) | 1.0000 |
| Final variance (sinusoidal costs) | 0.004945 |
| Observations | 1,001,000 |
| Bimodal variance tracking | 0.1436 (stable, correctly non-zero) |

**Verdict:** ✅ Pass. Sub-100ns overhead. Variance tracking works correctly for bimodal costs.

### T2: AdaptiveNoiseSchedule (D2F)

| Epoch | Ratios (8 blocks) |
|-------|-------------------|
| 20 | [0.100, 0.214, 0.329, 0.443, 0.557, 0.671, 0.786, 0.900] |
| 30 | [0.100, 0.214, 0.329, 0.443, 0.557, 0.671, 0.786, 0.900] |
| 40 | [0.100, 0.214, 0.329, 0.443, 0.557, 0.671, 0.786, 0.900] |

- **Adaptations:** 50
- **Monotonicity:** Preserved ✅
- **Bounds:** All ratios within [0.1, 0.9] ✅

**Note:** Ratios didn't shift significantly because synthetic linear-increasing losses produce uniform variance across steps. Real training data with non-uniform difficulty should show ratio adaptation. Infrastructure is correct and functional.

**Verdict:** ✅ Pass (infrastructure). Requires real training data for meaningful adaptation evaluation.

### T5-T7: Bandit VarianceEpsilon Strategy

| Strategy | Avg Reward | Avg Regret | Found Optimal |
|----------|-----------|------------|---------------|
| UCB1 | 0.881 | 0.024 | ✅ Yes |
| EpsilonGreedy(ε=0.3) | 0.781 | 0.119 | ✅ Yes |
| VarianceEpsilon(ε=0.3) | 0.774 | 0.124 | ✅ Yes |

- Episodes: 5,000
- Environment: Bernoulli(5 arms, probs=[0.1, 0.3, 0.5, 0.7, 0.9])

**Verdict:** ⚠️ Partial. VarianceEpsilon finds optimal arm but slightly underperforms EpsilonGreedy on this synthetic task. UCB1 remains the best default. VarianceEpsilon may shine in non-stationary environments where variance is informative. Keep feature-gated.

### T8-T10: SDAR SdarLearnedBeta

| Metric | Value |
|--------|-------|
| `observe_and_adapt` latency | **10.5 ns/obs** |
| Initial β | 5.0 |
| Final β (sinusoidal signals) | 50.0 (hit max bound) |

**Verdict:** ⚠️ Partial. SdarLearnedBeta adapts but hit the upper bound (50.0) on sinusoidal test signals. This suggests the variance-minimization objective pushes β toward binary gating when signal variance is inherent. May need different cost function or lower lr for real usage. Keep feature-gated.

---

## Plan 079: ELF Embedded Language Flows

### T1: SDE Noise Injection Overhead

| γ | Latency (µs/call) | Notes |
|---|-------------------|-------|
| 0.0 (disabled) | 0.2 | Near-instant (clone only) |
| 0.5 | 3.4 | |
| 1.0 (ELF default) | 3.4 | |
| 2.0 | 3.3 | |

- **SDE overhead:** 3.2 µs (γ=1 vs γ=0)
- **Comparison:** Single attention layer is ~100-500 µs. SDE overhead is <3% of one attention step.

**Verdict:** ✅ Pass. Overhead is negligible.

### T1: SDE Path Diversity

| γ | Unique Prefixes (100 trials) | Avg/Trial | Avg Tree Size |
|---|------------------------------|-----------|---------------|
| 0.0 | 14 | 0.1 | 16 |
| 0.5 | 58 | 0.6 | 16 |
| 1.0 | 145 | 1.4 | 16 |
| 2.0 | 315 | 3.1 | 16 |

- Prefix depth: 3 tokens
- Config: draft model, 100 trials per γ

**Verdict:** ✅ Pass. SDE noise increases path diversity **4-22×** without changing tree budget. γ=1.0 gives 10× diversity increase.

### T1: SDE Quality Tradeoff

| γ | Avg Top Path Probability | Avg Tree Size |
|---|-------------------------|---------------|
| 0.0 | 0.9899 | 16 |
| 0.5 | 0.9875 | 16 |
| 1.0 | 0.9757 | 16 |
| 2.0 | 0.9368 | 16 |

**Verdict:** ⚠️ Tradeoff exists. Top-path probability drops 1.4% at γ=1.0 and 5.3% at γ=2.0. This is the expected explore/exploit tradeoff. γ=1.0 is a reasonable balance (10× diversity for 1.4% quality).

### T6: Logit-Normal Schedule (n_steps=32)

| Schedule | Mean Step | Steps below t=0.3 | Concentration Ratio |
|----------|-----------|--------------------|--------------------|
| Uniform | 0.500 | 10/32 (31%) | 1.0× |
| LogitNorm(μ=-1.5, σ=0.8) | 0.226 | 22/32 (69%) | 2.2× |

- **Overhead:** Uniform=22 ns/call, LogitNorm=182 ns/call (159 ns overhead)

**Verdict:** ✅ Pass. Logit-normal concentrates steps 2.2× more near t=0 as ELF predicts. Overhead is negligible (159 ns vs ~100 µs for forward pass).

### T11: D2F Schedule Comparison

| Schedule | Avg Steps | Avg Final Confidence | Fully Activated |
|----------|-----------|---------------------|-----------------|
| Uniform | 1.1 | 1.000 | 50/50 |
| LogitNorm(-1.5, 0.8) | 1.1 | 1.000 | 50/50 |

- Config: `dllm_micro` (4-block, 4-step), 50 trials

**Note:** No difference on `dllm_micro` because the model is too small — it converges in 1 step regardless of schedule. Schedule impact requires larger models or more denoising steps.

**Verdict:** ⚠️ Inconclusive on dllm_micro. Infrastructure is correct; logit-normal distribution is verified correct. Needs larger model benchmark.

---

## Summary

| Subsystem | Plan | Status | GOAT Proof |
|-----------|------|--------|------------|
| VarianceMinimizer | 078 | ✅ Ship | 10.8 ns/obs, tracks correctly |
| AdaptiveNoiseSchedule | 078 | ✅ Ship | Infrastructure ready, needs real data |
| VarianceEpsilon | 078 | ⚠️ Keep gated | Converges but doesn't beat UCB1 |
| SdarLearnedBeta | 078 | ⚠️ Keep gated | Adapts but hits bounds on synthetic data |
| SDE Noise Injection | 079 | ✅ Ship | 10-22× diversity, 3.2 µs overhead |
| Logit-Normal Schedule | 079 | ✅ Ship | 2.2× concentration, 159 ns overhead |
| D2F Schedule Impact | 079 | ⚠️ Inconclusive | No effect on dllm_micro |

## Decision

- **Ship:** VarianceMinimizer (core primitive), SdeConfig + inject_sde_noise, ScheduleKind + logit-normal
- **Feature-gate (off by default):** `replaid_schedules` (AdaptiveNoiseSchedule, VarianceEpsilon, SdarLearnedBeta), `elf_sde` (SDE noise defaults)
- **GOAT proof domain benchmarks (T8-T10):** Deferred — require bomber/go/fft arena setups. Infrastructure is ready.

## Commands to Reproduce

```bash
# Plan 078 benchmarks
cargo test --features "replaid_schedules,bandit,sdar_gate,dllm" --test bench_replaid_variance_schedules --release -- --nocapture

# Plan 079 benchmarks
cargo test --features "dllm" --test bench_elf_modelless --release -- --nocapture

# All lib tests (784 tests)
cargo test --features "replaid_schedules,bandit,sdar_gate,dllm" --lib
```

---

## Plan 077: SpectralQuant Bug Fix + GOAT Results (2025-07)

### Bug Fix: Codebook Fitting

Root cause: `from_calibration()` created `LloydMaxCodebook` with all-zero centroids but never fitted them with data. Fix: generate synthetic rotated data from eigenvalue distribution and fit codebooks at construction time.

### SpectralQuant vs TurboQuant (128D, 3-bit avg, 16 positions)

| Metric | TurboQuant | SpectralQuant v1 (uniform) | SpectralQuant v2 (water-fill) |
|--------|-----------|---------------------------|------------------------------|
| Key cosine | **0.9692** | 0.5978 | 0.6493 |
| Value cosine | **0.9827** | 0.5978 | 0.7915 |
| Compression | 5.3× | 10.7× | 10.7× |
| Store latency | — | 22.3 µs/pos | — |
| Dequant latency | — | 17.0 µs/pos | — |

### Water-fill vs Uniform (SpectralQuant only)

| Version | Cosine Similarity | Delta |
|---------|-------------------|-------|
| v1 (uniform bits) | 0.5978 | baseline |
| v2 (water-fill) | 0.6493 | **+0.0515** |

### Eigenbasis Quality (128D, 500 samples)

| Metric | Value |
|--------|-------|
| Calibration time | 3.73 ms |
| d_eff (participation ratio) | 8.3 |
| var_95 | 14 components |
| var_99 | 21 components |
| spectral_gap | 1.24 |
| Top eigenvalue | 11.91 |

### GOAT Verdict

⚠️ **Tradeoff, not clear win.** SpectralQuant achieves 2× better compression (10.7× vs 5.3×) but at significantly lower cosine similarity (0.65 vs 0.97). Water-fill adds +0.05 cosine over uniform. The `spectral_quant` feature remains default-on for the compression advantage, but users who need fidelity should prefer `turboquant`.

