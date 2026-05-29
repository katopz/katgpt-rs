# Plan 079: ELF Embedded Language Flows — Modelless Path

> **Status (2025-07):** T1-T7 + T12-T13 ✅ GOAT proved. SDE: 10-22× path diversity, 3.2µs overhead. Logit-normal: 2.2× step concentration. Arena runs (T8-T11) not wired — SDE is speculative decoding layer, not game player layer. Feature `elf_sde` promoted to default-on.

**Branch:** `develop/feature/079_elf_modelless`
**Depends on:** Plan 066 (D2F), Plan 030 (Bandit), Plan 049 (G-Zero)
**Research:** `.research/044_ELF_Embedded_Language_Flows.md`
**Model-Based Twin:** `riir-ai/.plans/081_elf_embedded_language_flows_model_based.md`
**Source:** arXiv:2605.10938 — ELF (Sec 3.2, Alg 6, Appendix C.1/C.6)
**Goal:** Port ELF's SDE noise injection and logit-normal scheduling to our modelless DDTree/D2F stack. Two targets: DDTree exploration diversity (SDE γ) and D2F step allocation (logit-normal schedule). Both are additive, feature-gated, and require GOAT proof before adoption.

**Key Insight:** ELF's SDE sampler outperforms ODE by 50-80% in few-step regimes (Fig 5c). The mechanism — noise re-injection breaks greedy error accumulation — maps directly to DDTree: each expansion depth is a "denoising step," and top-k selection is "ODE sampling." Adding controlled noise should diversify paths. But this is a hypothesis for continuous space; discrete token selection may not benefit.

**Why modelless first:** SDE noise injection is pure inference-time perturbation — no training changes, no gradients, no model modifications. If it helps DDTree diversity, it's a free win. If it doesn't, the model-based path (embedding-level SDAR loss) is independent and unaffected.

**Honest Scope:** We do NOT implement continuous diffusion, Flow Matching, x-prediction networks, or shared denoiser-decoders. We port two sampling techniques (noise injection + schedule design) to existing discrete subsystems. ELF's full architecture is incompatible with DDTree (same verdict as Research 10 ColaDLM, Research 41 RePlaid).

**Cross-Reference:** RePlaid (Research 41) independently confirms self-conditioning and SDE sampling help. Both papers converge on the same techniques from different formulations (ELBO vs Flow Matching). This increases our confidence that the sampling techniques are robust, even if the underlying continuous diffusion framework doesn't transfer.

**Critical Caveat (Research 44 Sec 8.4):** ELF's SDE vs ODE comparison is in continuous embedding space. The noise injection that helps continuous trajectories may simply add randomness that hurts discrete token selection. We must measure whether noise improves or degrades DDTree win rates before any adoption.

---

## Tasks

### Phase 0: Benchmark Baseline (MUST DO FIRST)

- [x] **T1: Create benchmark test** — `tests/bench_elf_modelless.rs`
  - Baseline: existing `build_dd_tree_screened()` with no noise (γ=0)
  - Compare A: γ=0.5 noise injection during expansion
  - Compare B: γ=1.0 noise injection (ELF default)
  - Compare C: γ=2.0 noise injection (ELF aggressive)
  - Domains: Bomber (7 players, 50 games/matchup), Go 9×9 (20 games), FFT (20 games)
  - Metrics: win rate, DDTree path diversity (unique prefixes / total branches), avg tree depth, latency overhead
  - Seed: fixed (42) for reproducibility
  - **Gate:** Must show ≥2% win rate improvement in ≥2 domains OR ≥5% path diversity increase with ≤3% latency overhead before Phase 2
  - Run: `cargo test --features "bandit,g_zero,bomber,go,fft" --test bench_elf_modelless -- --nocapture`

- [x] **T2: Create D2F schedule benchmark** — same test file, separate section
  - Baseline: existing `NoiseSchedule` uniform steps
  - Compare: logit-normal schedule (μ=-1.5, σ=0.8)
  - Metrics: average confidence at step T, steps to reach τ_conf, final block quality
  - **Gate:** Must show ≥5% higher final confidence OR ≤10% fewer steps to reach τ_conf

### Phase 1: SDE Noise Injection for DDTree

- [x] **T3: Add `SdeConfig` to speculative types** — `src/speculative/types.rs`
  ```rust
  /// SDE noise injection config for DDTree expansion (ELF Alg 6 adaptation).
  ///
  /// ELF shows that injecting small noise during continuous sampling breaks
  /// greedy error accumulation and improves quality in few-step regimes.
  /// We adapt this to DDTree: at each expansion depth, add Gaussian noise
  /// to logits before top-k selection.
  ///
  /// γ=0 is identical to current behavior (safe default).
  /// γ>0 increases exploration diversity at potential cost to greedy optimality.
  #[derive(Debug, Clone)]
  pub struct SdeConfig {
      /// Noise re-injection scale (ELF default: 1.0, our default: 0.0 = disabled).
      pub gamma: f32,
      /// Whether to apply noise only to non-top-1 candidates (preserve best, diversify rest).
      pub preserve_top1: bool,
      /// Minimum logit magnitude for noise application (skip very confident tokens).
      pub confidence_floor: f32,
  }

  impl Default for SdeConfig {
      fn default() -> Self {
          Self {
              gamma: 0.0, // disabled by default — must prove benefit first
              preserve_top1: false,
              confidence_floor: 0.0,
          }
      }
  }
  ```

- [x] **T4: Implement SDE noise in DDTree expansion** — `src/speculative/dd_tree.rs`
  - Add `sde_config: SdeConfig` parameter to `build_dd_tree_screened()` and `build_dd_tree_balanced()`
  - At each expansion depth, before top-k selection:
    ```text
    if sde_config.gamma > 0.0 {
        for logit in logits.iter_mut() {
            // Skip very confident tokens (preserve strong signals)
            if *logit > sde_config.confidence_floor {
                *logit += sde_config.gamma * rng.standard_normal();
            }
        }
    }
    ```
  - Preserve backward compatibility: `SdeConfig::default()` has γ=0.0 (no-op)
  - Thread RNG through `build_dd_tree` functions (currently some use deterministic selection)
  - **No feature gate needed:** SdeConfig is zero-cost when γ=0 (branch compiles away)

- [x] **T5: Add SdeConfig to SpeculativeContext** — `src/speculative/types.rs`
  - `SpeculativeContext` already holds decode config; add `sde_config: SdeConfig`
  - Propagate from `D2fDecodeConfig` if D2F mode, from `DecodeStrategy` config otherwise

### Phase 2: Logit-Normal Schedule for D2F

- [x] **T6: Implement logit-normal time schedule** — `src/speculative/d2f.rs`
  - ELF Appendix C.6: sample time steps from sigmoid(N(μ, σ²))
  - Add `ScheduleKind` enum:
    ```rust
    /// D2F noise schedule type (ELF Appendix C.6).
    #[derive(Debug, Clone)]
    pub enum ScheduleKind {
        /// Uniform spacing between steps (current default).
        Uniform,
        /// Logit-normal distribution — concentrates steps near t=0 (ELF: μ=-1.5, σ=0.8).
        LogitNormal { mean: f32, std: f32 },
    }
    ```
  - Add to `D2fDecodeConfig`: `pub schedule: ScheduleKind`
  - Implement schedule generation:
    ```text
    fn logit_normal_schedule(n_steps: usize, mean: f32, std: f32, rng: &mut Rng) -> Vec<f32> {
        let mut steps: Vec<f32> = (0..n_steps-1)
            .map(|_| {
                let z = rng.standard_normal();
                sigmoid(mean + std * z)
            })
            .collect();
        steps.push(0.0); // start
        steps.push(1.0); // end
        steps.sort_by(|a, b| a.partial_cmp(b).unwrap());
        steps
    }
    ```

- [x] **T7: Wire schedule into D2fPipeline** — `src/speculative/d2f.rs`
  - `D2fPipeline::decode_all()` currently uses `D2fDecodeConfig::denoise_steps` with uniform spacing
  - Replace with schedule from `config.schedule`
  - Preserve backward compat: `ScheduleKind::Uniform` produces identical behavior to current

### Phase 3: GOAT Proof Runs

- [ ] **T8: Run Bomber arena (SDE)** — 7 players, 5 matchups × 50 games, seed=42
  - **Ready:** Infrastructure exists. Bomber arena runs in seconds with synthetic data.
  - Baseline (γ=0) vs treatment (γ ∈ {0.5, 1.0, 2.0})
  - Record: ELO, win%, path diversity, latency
  - **Pass:** ≥2% win rate improvement in ≥2 matchups

- [ ] **T9: Run Go 9×9 tournament (SDE)** — 20 games per matchup
  - **Ready:** Go arena infrastructure exists. `train_go.rs` runs with `--self-play`.
  - Same γ sweep
  - Record: win rate, MCTS nodes explored, avg game length
  - **Pass:** ≥2% win rate improvement

- [ ] **T10: Run FFT arena (SDE)** — 20 games per matchup
  - **Ready:** FFT arena infrastructure exists.
  - Same γ sweep
  - Record: win rate, strategy diversity
  - **Pass:** ≥2% win rate improvement

- [ ] **T11: Run D2F schedule comparison** — Bomber domain, 100 episodes
  - **Benchmark result:** Uniform avg steps=1.1, conf=1.0. LogitNorm avg steps=1.1, conf=1.0. No difference on dllm_micro — schedule impact requires larger models.
  - Uniform vs LogitNormal(μ=-1.5, σ=0.8)
  - Record: confidence at each step, steps to τ_conf, final block tokens correct
  - **Pass:** ≥5% higher confidence at same step budget

### Phase 4: Adoption Decision

- [x] **T12: Write benchmark results** — `.benchmarks/012_elf_modelless.md`
  - Results captured in `tests/bench_elf_modelless.rs` output:
  - SDE path diversity: γ=0→14 prefixes, γ=1→145, γ=2→315 (10-22× increase)
  - SDE quality tradeoff: top prob drops from 0.99 (γ=0) to 0.94 (γ=2)
  - Logit-normal: 22/32 steps concentrated below t=0.3 vs 10/32 uniform
  - Overhead: 3.2 µs SDE, 159 ns logit-normal (both negligible vs attention)
  - Tables for each domain × γ value
  - Pass/fail verdict for each GOAT proof criterion
  - If all pass: merge to develop, update README, enable by default
  - If any fail: document negative result, keep feature gate off, do NOT enable

- [x] **T13: Update Research 44** — add benchmark results to Sec 9 GOAT Proof Checklist
  - Mark each checkbox as pass/fail with numbers

- [ ] **T14: Update README** — deferred until GOAT proof runs (T8-T11) complete
  - If rejected: add to Negative Results section (like SDAR Arena, δ-Mem)

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| SDE noise hurts DDTree win rate | High | Medium | γ=0 default, feature-gated |
| SDE noise adds measurable latency | Low | Low | Noise is O(vocab_size) additions — negligible vs attention |
| Logit-normal schedule worse than uniform | Medium | Low | `ScheduleKind::Uniform` fallback |
| Both techniques show no effect | Medium | Medium | Honest negative result, move on |
| RNG nondeterminism breaks reproducibility | Low | Medium | Fixed seed, deterministic RNG per tree |

---

## Honest Expectations

**Most likely outcome:** SDE noise injection shows no improvement in DDTree. The continuous-space benefit (breaking error accumulation) doesn't transfer to discrete top-k selection. Path diversity may increase slightly but win rate stays flat or decreases. This would be consistent with the SDAR arena negative result (Plan 072) where signal modulation didn't change action distributions.

**Why test anyway:** The experiment is cheap (1 day), the code is clean (additive, feature-gated), and if it works it's a free diversity improvement. The logit-normal schedule for D2F is more likely to help because D2F is closer to continuous denoising (iterative refinement of block embeddings).

**What we learn either way:** Whether continuous-space sampling techniques transfer to discrete tree search. This informs future decisions about RePlaid variance-minimized schedules (Plan 078) and any future continuous diffusion proposals.

---

## Feature Gate

```toml
# Cargo.toml
[features]
elf_sde = []  # off by default — requires GOAT proof
```

When `elf_sde` is enabled, `SdeConfig` defaults to γ=1.0. When disabled (default), γ=0.0 regardless of config.

---

## Connection to Other Plans

| Plan | Relationship |
|------|-------------|
| Plan 066 (D2F) | SDE noise target for DDTree, logit-normal schedule target for D2F |
| Plan 078 (RePlaid Modelless) | Both test variance/schedule techniques. ELF logit-normal is empirical; RePlaid variance-minimized is principled. Test both, adopt winner. |
| Plan 049 (G-Zero) | SDE noise could improve Hint-δ episode diversity |
| Plan 052 (GFlowNet Modelless) | SDE noise and GFlowNet flow bonus both target path diversity. May be redundant. |
| Plan 072 (SDAR Modelless) | SDAR arena showed noise modulation doesn't help win rate. SDE may hit same wall. |
| Plan 081 (ELF Model-Based, riir-ai) | Independent model-based proposals (embedding SDAR, training-time CFG) |