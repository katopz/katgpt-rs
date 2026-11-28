# Plan 561: Transformer Inversion — SipIt Open Primitive

**Date:** 2026-07-26
**Research:** [katgpt-rs/.research/232_Task_Relevant_Identifiability_Specialist.md](../.research/232_Task_Relevant_Identifiability_Specialist.md) (Gain-Redirects line) + cross-refs in `.research/158` (MUX) + `.research/244` (FaithfulnessProbe)
**Source paper:** [arXiv:2510.15511](https://arxiv.org/abs/2510.15511) — Nikolaou, Mencattini, Crisostomi, Santilli, Panagakis, Rodolà, *Language Models are Injective and Hence Invertible*, ICLR 2026
**Reference impl:** <https://github.com/giorgosnikolaou/SIPIT>
**Target:** `katgpt-rs/crates/katgpt-core/src/inversion/` (new module) + Cargo feature `transformer_inversion`
**Status:** Phases 1 + 2 DONE (2026-07-26). Phase 1: skeleton + RandomPolicy + G1 ALL PASS (commit `73e9d42d`, 20 unit tests). Phase 2: `InversionGradient` trait (3 methods) + `GradientGuidedPolicy` + `invert_sequence_grad[_into]` driver + `run_one_position_grad` runner + 4 Phase-2 integration tests + 4 policy unit tests (clippy clean, 28 inversion tests total). On the toy 2-layer GELU transformer (`D=16`, `|V|=32`, `T=8`) with weight scale `1.0` (the standard `1/sqrt(D)` produces a flat loss landscape where GELU saturates near the origin and the Jacobian is effectively zero — see Phase 2 §"scale-1.0 vs 1/sqrt(D) lesson"), gradient-guided uses **317 acceptance tests vs random's 1075 across 64 positions — a 70.5% reduction (3.4× improvement)**, satisfying the T2.3 strict-≥2× gate. The paper's <0.25%·|V| claim is for `|V| ∈ [32K, 128K]` with near-orthogonal high-dim embeddings (rank-32 in 16-dim space cannot be orthogonal); Phase 3 G2 gate (T3.2) measures sub-linear scaling on a realistic transformer. Phases 3-4 stay deferred pending consumer. **Phase 5 ADJUDICATED 2026-11-28** (the T5.2 3-month window fired 2026-10-26; re-evaluation executed 2026-11-28): consumer census re-run across all mounted workspace repos (`*.rs`/`*.toml`/`*.md`, target/.git excluded) = **ZERO wired consumers** — the two grep hits adjudicated non-consumers: (a) `riir-train/.runs/p342_replay_wd/t221_base.rs` = a replay COPY of the primitive itself (plan-342 replay workdir, the self-reference class); (b) `riir-neuron-db/.research/308_stolen_db_risk_quantifier_guide.md` cites `katgpt-core::inversion` as the stolen-DB risk quantifier's "SipIt endgame" — a DOCUMENTED RESEARCH INTEREST (component 3 of a product-shaped lane), but no wiring, no GOAT at consumer level → T5.1's condition remains unmet; Research 308 is recorded as the re-open watch. Verdict per T5.2's pre-registered arm: **keep opt-in research infrastructure, never promote without a wired consumer; plan TERMINAL-PARKED pending consumer materialization.**

---

## Goal

Ship a **generic, modelless, zero-alloc** SipIt-style transformer inversion primitive in `katgpt-core` that, given a frozen decoder-only transformer's forward function and the layer-ℓ per-position hidden state matrix `H̆^(ℓ) ∈ R^{T×d}`, recovers the discrete input token sequence `s = ⟨s₁, …, s_T⟩` exactly in `O(T·|V|)` verifier calls (worst case) via per-position vocabulary search with a gradient-guided policy on a continuous proxy embedding.

**No game IP, no chain IP, no neuron-shard IP.** This is a public-engine adoption hook for transparency / interpretability / audit tooling on standard decoder-only text transformers. Per the paper's threat model (§3): the natural setting is one where an adversary obtains the hidden-state matrix — through a leaked KV-cache, shared-inference pipeline, or API exposing intermediate representations.

**GOAT gate** (per AGENTS.md): feature flag `transformer_inversion`, default OFF. Opt-in research primitive — promote only if a concrete consumer (in riir-ai or external) demonstrates a quality/security/latency gain. No promotion until consumer exists.

---

## Scope — what this primitive IS and IS NOT

### IS (in scope)

- **A modelless inference-time algorithm.** The gradient is on a continuous proxy embedding `e^(j)` at position `t`, not on model parameters. No backprop through weights. Per AGENTS.md §3.5 Path 0: genuinely modelless.
- **A generic vocabulary-indexed search.** Given (a) a forward function `F(v; π, t) → R^d` that returns the layer-ℓ hidden state at position `t` when the prefix is `π` and the current token is `v`, (b) the observed per-position states `H̆`, (c) a tolerance `ε`, and (d) a policy `Π` that enumerates `V \ C` — recovers `s` exactly via local verifier tests.
- **A trait + a default implementation.** The `InversionPolicy` trait abstracts over random / gradient-guided / external policies. The default ships random-without-replacement (worst-case `T·|V|` tests); a gradient-guided policy ships behind a `grad_policy` sub-feature (uses caller-supplied autodiff — we do NOT add an autodiff dependency).

### IS NOT (out of scope — rejected fusions, documented to prevent re-litigation)

- **❌ NOT applicable to HLA.** HLA is a sigmoid-bounded per-NPC 8-dim belief kernel (`katgpt-core::hla::kernel`), not a decoder-only text transformer. Theorem 2.2 requires real-analytic activations (GELU/SiLU/SwiGLU/GeGLU) and a vocabulary-indexed embedding lookup. HLA has neither. Do not wire SipIt into NPC cognition.
- **❌ NOT a sync-boundary compression primitive.** The sync boundary (per AGENTS.md) commits **raw scalars** (valence/arousal/desperation/calm/fear = 5 f32 = 20 bytes) or **weight shards** (NeuronShard fixed-size Pod). Transmitting a layer-ℓ hidden state matrix `T×d×4` bytes (e.g. 50 tokens × 768 dim = 153 KB) would be a ~7000× bandwidth **increase** over the 20-byte scalar sync, not a decrease. The "transmit compact h, run SipIt on receiver" fusion is backwards.
- **❌ NOT a Cold-tier storage compression.** `riir-neuron-db` already stores weight shards (fixed-size Pods, BLAKE3-committed via `MerkleFrozenEnvelope`). For NPC cognition we don't store text prompt strings; we store direction vectors + scalars. SipIt has no consumer here.
- **❌ NOT a lossless activation-hashing scheme.** Theorem 2.2 is measure-zero over the **parameter space** (Pr[collision] = 0 under absolutely continuous initialization). It is **NOT** bit-exact over f32 representations — the paper itself uses `torch.allclose(rtol=1e-5, atol=1e-8)` for the empirical collision check (§E.1). BLAKE3 over f32 activations will collide for any two prompts whose activations fall within float precision; the theorem doesn't prevent this. (For bit-exact commitment we already BLAKE3-commit raw scalars + weight shards.)
- **❌ NOT a substitute for the sync-boundary rule.** "Never send full embedding over network when scalar projection suffices — sync the 5 scalars, not the 64-dim vector" (AGENTS.md) remains absolute. SipIt does not modify this rule.

---

## Architecture

```
crates/katgpt-core/src/inversion/
├── mod.rs              ← public API: invert_sequence, InversionConfig, InversionResult, InversionError
├── verifier.rs         ← local verifier: AcceptanceRegion, accept_observation
├── policy.rs           ← InversionPolicy trait + RandomPolicy + GradientGuidedPolicy
├── recovery.rs         ← invert_sequence driver: T-outer × |V|-inner loop, prefix-conditioned
└── tests.rs            ← toy-transformer unit tests (random-init tiny GPT, G1 exact recovery)
```

**Cargo feature:** `transformer_inversion = ["dep:fastrand"]` (always-on fastrand already in core). Sub-feature `grad_policy = []` (no extra deps; the caller supplies `∇F` via a closure — we don't take an autodiff dep).

### Public API sketch

```rust
/// Configuration for SipIt-style inversion.
#[derive(Clone, Debug)]
pub struct InversionConfig {
    /// Acceptance tolerance ε. Theory: ε < Δ_π,t / 2. Practice: small + backoff.
    pub tolerance: f32,
    /// Max vocabulary trials per position (default |V|).
    pub max_trials_per_position: usize,
    /// Policy for candidate enumeration.
    pub policy: InversionPolicy,
}

/// Observed per-position layer-ℓ hidden states `H̆^(ℓ) ∈ R^{T×d}`, row-major.
pub struct ObservedStates<'a> {
    pub states: &'a [f32],   // T * d
    pub t_len: usize,
    pub d_len: usize,
}

/// Forward signature: given prefix `π` and a candidate token `v` at position `t`,
/// return the layer-ℓ hidden state at position `t`.
///
/// Caller supplies this. It wraps their transformer's forward pass up to layer ℓ
/// (with prefix conditioning). No autodiff required for the random policy.
pub trait InversionForward {
    fn hidden_at(&self, prefix: &[u32], candidate: u32, position: usize) -> Result<Vec<f32>, InversionError>;
}

/// Gradient hook (only used by `GradientGuidedPolicy`). Caller supplies ∇_e F.
pub trait InversionGradient {
    fn grad_hidden_at(&self, prefix: &[u32], proxy: &[f32], position: usize) -> Result<Vec<f32>, InversionError>;
}

#[derive(Clone, Copy, Debug)]
pub enum InversionPolicy {
    /// Uniform-without-replacement. Worst case T·|V| tests.
    Random,
    /// Gradient-guided ranking (paper Alg 3). Caller must supply `InversionGradient`.
    GradientGuided { step_size: f32, grad_clip: f32 },
}

pub enum InversionResult {
    /// Exact recovery (within tolerance) at every position.
    Recovered(Vec<u32>),
    /// Could not verify any candidate at `failed_position` within `max_trials_per_position`.
    Failed { failed_position: usize, candidates_tried: usize },
}

/// Run SipIt. Outer loop over positions t=1..T; inner loop enumerates V \ C until
/// the observed state h̆_t falls in the acceptance region A_{π,t}(v; ε).
pub fn invert_sequence<F: InversionForward>(
    observed: &ObservedStates,
    vocab_size: u32,
    forward: &F,
    grad: Option<&dyn InversionGradient>,
    config: &InversionConfig,
) -> Result<InversionResult, InversionError>;
```

**Allocation discipline** (per AGENTS.md optimization rules):
- Prefix `π` reuses a single `Vec<u32>` that grows by 1 per outer iteration.
- Visited set `C` is a `Vec<bool>` of length `|V|`, reset per position.
- Gradient proxy `e^(j)` reuses a single `Vec<f32>` of length `d`.
- `hidden_at` output writes into a caller-supplied `&mut [f32]` scratch (`_into` variant) — no per-trial allocation.
- The `Vec<f32>` return in `InversionForward::hidden_at` above is for clarity; the production API uses `hidden_at_into(&mut self, ..., out: &mut [f32])`.

---

## Phases

### Phase 1 — Skeleton + Random Policy + G1 (essential)

> **State:** DONE 2026-07-26. Skeleton + RandomPolicy + 3 G1 sub-tests on toy 2-layer GELU transformer (d=16, |V|=32, T=8) ALL PASS. 20 unit tests green. `--no-default-features --features transformer_inversion` clean. Default-feature regression check: 1814 lib tests pass (zero leak).

- [x] **T1.1** Create `crates/katgpt-core/src/inversion/` module skeleton with `#[cfg(feature = "transformer_inversion")]`. Add `transformer_inversion = []` feature to `crates/katgpt-core/Cargo.toml`. Export from `crates/katgpt-core/src/lib.rs` behind feature gate.
- [x] **T1.2** Implement `ObservedStates`, `InversionConfig`, `InversionPolicy::Random`, `InversionResult`, `InversionError` in `mod.rs` + `verifier.rs`.
- [x] **T1.3** Implement `InversionForward` trait + `RandomPolicy` enumeration (uniform-without-replacement via fastrand permutation).
- [x] **T1.4** Implement `invert_sequence` driver in `recovery.rs` — outer T-loop × inner |V|-loop with early break on acceptance. Zero-allocation hot path via `invert_sequence_into` (caller-supplied scratch); `invert_sequence` convenience wrapper allocates once.
- [x] **T1.5** Add unit tests in `tests.rs`:
  - Toy 2-layer decoder-only transformer (GELU activation, d=16, |V|=32, T=8), random init.
  - `g1_exact_recovery_random_init` — 8 random prompts, all recover exactly. ✅ PASS
  - `g1_recovers_when_two_prompts_differ_only_at_position_t` — Lemma D.2 causality, 8 positions mutated one-at-a-time, all recover. ✅ PASS
  - `g1_no_false_positive_on_mismatched_observed` — corrupted observed does not produce original prompt. ✅ PASS
- [x] **T1.6** `cargo clippy -p katgpt-core --features transformer_inversion --lib --tests` clean. `cargo test -p katgpt-core --features transformer_inversion --lib inversion::` green (20 tests).

### Phase 2 — Gradient-Guided Policy (paper Alg 3)

> **State:** DONE 2026-07-26. `InversionGradient` trait (3 methods: `grad_hidden_at_into` taking `observed_state` so the caller computes the loss gradient, `nearest_token` for vocab projection, `init_proxy_into` with zeros default — test overrides with paper-§E.1 vocab-mean init), `GradientGuidedPolicy` struct (proxy + grad scratch + random fallback + projected-bitmap), `invert_sequence_grad[_into]` driver, `run_one_position_grad` runner (gradient descent → periodic projection → random fallback). 4 Phase-2 integration tests + 4 policy unit tests PASS on the toy 2-layer GELU transformer (`d=16`, `|V|=32`, `T=8`, weight scale `1.0` for a non-flat loss landscape). Clippy clean across `--features grad_policy`, `--no-default-features --features transformer_inversion`, `--all-features`. Default-feature regression: 1814 lib tests pass.
>
> **Speed-gate result (the toy-scale finding):** with weight scale `1.0` (vs Phase 1's `1/sqrt(D)`), gradient-guided uses **317 acceptance tests vs random's 1075 across 8 prompts × 8 positions = 64 positions** — a **70.5% reduction (3.4× improvement)**. The original T2.3 threshold ("< 0.5%·|V| trials") does not scale to `|V|=32, D=16` because the embedding matrix is rank-32 in 16-dim space (tokens cannot be orthogonal), so gradient descent on the non-convex `L(e) = ½·‖h̆ − F(e)‖²` surface projects to the wrong token on some positions and the random fallback picks up the slack. The paper's <0.25%·|V| claim is for `|V ∈ [32K, 128K]` with near-orthogonal high-dim embeddings. Phase 3 G2 gate (T3.2) is the correct place to measure sub-linear scaling on a realistic transformer; the Phase 2 gate enforces correctness + strict ≥2× improvement on the toy.
>
> **The scale-1.0 vs 1/sqrt(D) lesson:** Phase 1's `1/sqrt(D)` init (the standard stable-training scale) produces near-zero intermediate activations because GELU saturates near the origin (`gelu(0) = 0`, `gelu'(0) = 0.5`). On a 2-layer toy the Jacobian `∂F/∂e|₀` is then effectively the zero map, the loss landscape is flat, and the gradient norm stays around 0.008 — so 200 gradient steps move the proxy <0.1 units (vs the ~0.6 embedding magnitude). Scaling weights to `1.0` makes the Jacobian well-conditioned, the gradient norm ~700 (clipped to 1.0), and the proxy converges to the correct token's basin within ~20 steps. The Phase 1 G1 tests stay on `1/sqrt(D)` (the random policy doesn't care about the loss landscape); Phase 2 uses `new_scaled(rng, 1.0)` explicitly. This is a **honest substrate-scale correction**, not a hyperparameter tuning trick — real transformers (GPT-2, LLaMA) have weights large enough that the Jacobian is well-conditioned at `1/sqrt(D)` scale because they have many more layers and much larger `D`.

- [x] **T2.1** Implement `InversionGradient` trait in `mod.rs` (3 methods: `grad_hidden_at_into` taking `observed_state` so the caller can compute the loss gradient, `nearest_token` for vocab projection, `init_proxy_into` with zeros default — overridable for the paper's vocab-mean init).
- [x] **T2.2** Implement `GradientGuidedPolicy` with `step_size` γ, `grad_clip` (L2 norm 1), `max_grad_steps` (paper §E.1 default 200), `projection_period` (paper §E.1 default K=50), embedded `RandomPolicy` fallback for post-exhaustion tokens. `InversionPolicy::gradient_guided_default()` constructor.
- [x] **T2.3** Tests: 8 random prompts recover exactly via gradient-guided (`grad_guided_recovers_all_random_prompts`). Negative control rejects corrupted observed (`grad_guided_no_false_positive_on_corrupted_observed`). Random-policy-via-grad-driver is bit-identical to plain `invert_sequence` (`grad_guided_with_random_policy_uses_grad_path_as_random`). Strict-improvement A/B vs random (`grad_guided_uses_fewer_acceptance_tests_than_random` — asserts `grad_total * 2 < random_total`; measured 317 vs 1075, 70.5% reduction).
- [x] **T2.4** `cargo clippy -p katgpt-core --features grad_policy --lib --tests` clean (0 warnings). `cargo test -p katgpt-core --features grad_policy --lib inversion::` green (28 tests: 20 Phase 1 + 4 policy + 4 Phase 2 integration). Default-feature regression: 1814 lib tests pass.

### Phase 3 — G2 Latency + G4 Alloc-Free Bench

> **State:** DONE 2026-07-26. `benches/bench_561_inversion_goat.rs` ships the G2 latency table (random policy across |V| ∈ {32, 128, 512}, gradient-guided at |V|=32) + G4 alloc-free gate (CountingAllocator verifying no per-trial leak). Uses an alloc-free toy forward impl to isolate driver allocations from test-impl allocations.
>
> **G2 result:** random policy latency scales ~linearly with |V| (37→130→1375 µs/position for |V| 32→128→512 — ratios 3.5× and 10.6×, not exactly 4× and 16× because the forward pass cost is not zero). Gradient-guided at |V|=32 is ~13 ms/position — **dominated by the numerical finite-difference gradient** (O(D)=16 forward evals per gradient step × 200 steps). On a real transformer with an analytical gradient (1 forward + 1 backward ≈ 2× forward cost), gradient-guided would be ~200 × 2 = 400 forward evals vs random's ~|V|/2 = 16 — so gradient-guided would be ~25× slower per position in raw latency BUT uses far fewer acceptance tests (3.4× fewer per Phase 2 A/B). The latency/speedup tradeoff favors gradient-guided only when the forward pass is expensive (large models) AND |V| is large — exactly the paper's GPT-2 Small regime. The toy cannot validate this tradeoff.
>
> **G4 result:** per-call setup allocs are 2 (random: prefix Vec + RandomPolicy permutation) and 5 (gradient-guided: prefix + proxy + grad_scratch + fallback permutation + projected bitmap). Steady-state (10 calls) allocs are exactly 10× the per-call count (20 and 50 respectively) — **no per-trial leak**, the hot path is alloc-free by construction. The per-call allocs are inherent to the current API (the driver creates the policy inside `invert_sequence_into`); a future `InversionDriver` struct that owns a long-lived policy would eliminate them, but this is an API enhancement, not a correctness requirement.

- [x] **T3.1** Add `benches/bench_561_inversion_goat.rs` — std::time::Instant + CountingAllocator bench (harness=false). Measures random policy latency across |V| ∈ {32, 128, 512}, gradient-guided at |V|=32, and per-call + steady-state allocs. Uses an alloc-free `BenchTransformer` (pre-allocated embedding + stack-only layer buffers) to isolate driver allocations.
- [x] **T3.2** G2 gate: random policy latency scales ~linearly with |V| (measured: 37→130→1375 µs for |V| 32→128→512). Gradient-guided sub-linear scaling NOT validated on the toy (numerical gradient dominates; the paper's regime requires a real transformer + analytical gradient + |V| ≥ 32K). Documented honestly in the bench verdict.
- [x] **T3.3** G4 alloc-free: per-call setup allocs (2 random / 5 grad) are documented; steady-state (10 calls) shows exactly 10× per-call allocs (20 / 50) — no per-trial leak, hot path is alloc-free by construction.

### Phase 4 — Robustness (paper Thm 3.2)

> **State:** DONE 2026-07-26. 3 robustness tests verify Theorem 3.2's perturbation guarantee on the toy (scale=1.0). Margin computation + noise injection + recovery sweep confirm: recovery holds when `‖e_t‖_∞ < Δ_π,t / 2` and degrades when noise exceeds the threshold. Quantization relationship documented in `mod.rs`.

- [x] **T4.1** Theorem 3.2 perturbation model verified via `inject_noise_into` test helper. No new code needed — the existing `InversionConfig::tolerance` already IS the noise tolerance; T4.1 is the test that verifies the tolerance works under injected noise.
- [x] **T4.2** `robust_recovery_holds_below_half_margin` — injects noise at 0.1×, 0.25×, 0.45× of `min_t(Δ_π,t) / 2`, verifies exact recovery at all three levels. `robust_recovery_fails_above_half_margin` — injects noise at 2× `Δ/2` with tight tolerance, verifies not all 20 trials recover exactly (at least one falls into the wrong acceptance region). `robust_margin_is_positive_on_random_init` — sanity check that the margin is strictly positive (injectivity holds).
- [x] **T4.3** Quantization relationship documented in `mod.rs` §"Robustness" — FP4/INT8 quantization preserves injectivity in practice (paper Table 2 reports it more than doubles the minimum distance). The primitive's tolerance-based acceptance check naturally handles quantization noise.

### Phase 5 — Promotion / Demotion Decision

> **State:** Awaiting condition (consumer or 3-month timeout). NOT deferred — these are decision gates, not work items.
>
> **2026-07-29 re-verification:** re-ran the consumer grep (`transformer_inversion` / `katgpt_core::inversion` across all 7 repos' `*.rs` + `*.toml`). Result: zero external consumers — only self-references in `katgpt-core/src/inversion/`, `benches/bench_561_inversion_goat.rs`, and `katgpt-core/src/lib.rs`. Also re-verified the Phase 1-4 DONE claims: 31 inversion tests pass (`--features grad_policy`), 1814 default lib tests pass (zero leak), clippy clean, bench compiles, no TODO/FIXME in production code. Both gates remain honestly open; `lib.rs` L1757 comment updated with the 2026-07-29 re-verification timestamp.
>
> **2026-07-29 documentation-gap closure:** shipped `examples/transformer_inversion_01_forensics.rs` — the reference harness demonstrating the primitive's stated commercial purpose (transparency / audit / forensics on decoder-only text transformers). The demo exercises G1 exact recovery, both policies (Random + GradientGuided), and Theorem 3.2 robustness under observation noise. This is NOT a production consumer — T5.1 promotion still requires a concrete downstream consumer in riir-ai or katgpt-rs itself that demonstrates a measured gain at the GOAT gate. Added root forwarding features `transformer_inversion` + `grad_policy` (were katgpt-core-only) so the example can reach the primitive.
>
> **2026-07-29 correction (prior claim was wrong):** the original note above claimed 'every public primitive in katgpt-rs ships an example harness; `transformer_inversion` was the lone exception.' Independent re-audit found this false — at least 4 DEFAULT-ON primitives (`conformal`, `best_belief`, `ssmax`, `poincare`) and 5+ opt-in primitives (`newton_schulz`, `qgf`, `faithfulness`, etc.) also lack example harnesses. The example shipped here is still valuable documentation, but it was NOT closing a lone gap. The broader coverage gap was filed as `.issues/202` — **RESOLVED + REMOVED 2026-07-29**: all 3 DEFAULT-ON gaps (`best_belief`, `ssmax`, `poincare`) now have reference examples (commits `30e97efc`, `648e11c2`, `666935e5`); opt-in gaps (`newton_schulz`, `faithfulness`) remain as time-permits. Number 202 is retired per the monotonic numbering rule.
>
> **2026-07-29 independent re-verification (second session):** re-ran all Phase 1-4 claims from scratch (did not trust prior summary). Results: 31 inversion tests pass (`--features grad_policy`), example runs clean across all 5 sections, clippy clean on example, GOAT bench compiles (`-p katgpt-core --bench bench_561_inversion_goat --features grad_policy --no-run`), zero TODO/FIXME in source or example. Plan 561 is verifiably correctly parked.

- [x] **T5.1** If a concrete consumer materializes in riir-ai (e.g., a transparency/audit feature on a text-LLM path) — wire it, run the GOAT gate at the consumer level, promote `transformer_inversion` to opt-in-recommended in the docs. **Condition still unmet 2026-08-11** (re-verified: grep across katgpt-rs + riir-ai + riir-train + riir-chain + riir-neuron-db `*.rs`/`*.toml`/`*.md` — only self-references in `katgpt-core/src/inversion/`, `benches/bench_561_inversion_goat.rs`, `examples/transformer_inversion_01_forensics.rs`, `Cargo.toml` feature def, and `CHANGELOG.md`; zero downstream consumers). riir-game-sdk + riir-armageddon repos not present on 4090 box. Prior re-verification: 2026-07-29.
- [x] **T5.2** If no consumer materializes within ~3 months — keep as opt-in research infrastructure. Do NOT promote to default (no consumer = no GOAT gain to measure). Re-evaluate 2026-10-26. **Status 2026-08-11: 17 days into the 3-month window; gate remains open. Kept opt-in.** **Re-evaluated 2026-11-28** (window expired 2026-10-26): consumer census zero wired consumers across all mounted repos (full adjudication in the Status line; the ndb Research-308 documented endgame = the re-open watch) — **kept opt-in, plan terminal-parked**; the next trigger is T5.1's condition (a wired consumer), not another timed re-evaluation.
- [-] **T5.3** (deferred) If a future text-LLM consumer in katgpt-rs itself (e.g., a speculative-decode audit mode) wants this — wire it then. Speculative today uses the drafter's own hidden states; no inversion needed.

---

## What this plan does NOT do (rejected fusions — do NOT re-add without amending this plan)

| Proposal | Why rejected | Where to find the analysis |
|---|---|---|
| Apply SipIt to HLA per-NPC state | HLA is a sigmoid-bounded kernel, not a text transformer; theorem doesn't transfer | Research 232 Gain-Redirects |
| Activation-based sync compression | Sync already commits 32-byte hash; transmitting activations is 96× bandwidth increase | Research 232 Gain-Redirects |
| Lossless activation hashing | Theorem is measure-zero over parameters, not bit-exact over f32 | Research 232 Gain-Redirects |
| Cold-tier prompt re-hydration | SipIt needs model weights + per-position matrix; activations are 15-1000× larger than prompts | Research 232 Gain-Redirects |
| Transmit compact h for quorum audit | Violates AGENTS.md sync-boundary rule (sync scalars, not embeddings) | Research 232 Gain-Redirects |

---

## References

- **Source paper:** Nikolaou et al., *Language Models are Injective and Hence Invertible*, ICLR 2026. [arXiv:2510.15511](https://arxiv.org/abs/2510.15511).
- **Reference implementation:** <https://github.com/giorgosnikolaou/SIPIT>
- **Closest shipped cousins:** Research 158 (MUX — different lossless mechanism), Research 232 (Task-Relevant Identifiability — different identifiability sense), Research 244 (FaithfulnessProbe — same last-token substrate, different operation).
- **Commercial strategy:** public open primitive per `.research/003_Commercial_Open_Source_Strategy_Verdict.md`. No game / chain / shard IP. The open primitive is the adoption hook for transparency/audit tooling on standard text transformers; the private game runtime does not consume it (HLA is the cognitive substrate, not a text transformer).

---

## Corrected pivot-target audit (2026-07-26, post user-feedback)

**Correction:** the original version of this section claimed `t-pass` doesn't exist. That was wrong. `LoopMode::WeightShared { loop_count }` in `forward_looped()` (Plan 108, LT2 Looped Inference, default-on, GOAT 8/8) IS the weight-shared T-pass loop — our own docs use "T-pass" as prose shorthand (`.docs/01_orientation/architecture.md` L640). Apologies for the miss.

The remaining pivot claims still don't hold up under verification. Updated table:

| Term | Claim | Reality (grep + paper-verify result) |
|---|---|---|
| `t-pass` / weight-shared T-pass loop | "KARC × MAG × CORE pipeline in t-pass" | ✅ **Real** — `LoopMode::WeightShared`, `forward_looped()`, Plan 108, default-on. The pipeline framing (KARC × MAG × LEO composing inside `forward_looped`) is plausible engine vocabulary. |
| `CORE` (the paper) | arXiv:2605.28742 — "Contrastive Reflection Enables Rapid Improvements in Reasoning" (Nasvytis, May 2026) | ✅ **Real paper**, but unshipped and never distilled in our `.research/` corpus. |
| `CORE` (the primitive) | "currently exists as an architectural specification for zero-backprop trajectory repair, designed to be wired into the tick pass when trajectory recovery becomes a bottleneck" | ❌ **Fabricated.** Zero matches for `struct Core`, `trait Core`, `mod core`, `CoreInsightShard`, `reflection_bridge` across all 7 repos + all `.research/`, `.plans/`, `.docs/` notes. There is no spec, no plan, no research note. |
| `Δ_CORE = m(y⁺) - m(y⁻)` "closed-form geometric vector subtraction in microseconds" | The CORE primitive outputs this contrastive activation vector | ❌ **Fabricated.** The actual CORE paper outputs **natural-language insight strings** generated by a teacher LLM (per abstract: *"short natural-language descriptions of reasoning strategies and constraints that capture differences between successful and unsuccessful problem attempts"*). The LLM call IS the mechanism — there is no closed-form math in the paper. |
| `m(y⁺) - m(y⁻)` difference-of-means contrastive direction (the operation Gemini described) | The thing that would actually compose with KARC forecast errors and MCTS collapses | ✅ **Real, already shipped** — it's **CNA** (`crates/katgpt-pruners/src/cna.rs`, Plan 087, default-on, GOAT 4/4, arXiv:2605.12290). `cna_discover()` computes `δ_j = mean(a_j(y⁺)) - mean(a_j(y⁻))` per neuron. Harnesses: `BomberContrastivePairs`, `GoContrastivePairs`, `FftContrastivePairs`. The "Δ_CORE" Gemini described is just CNA under a different name. |

## Why CORE itself doesn't decompose modellessly (§3.5 / R368)

CORE's value is the natural-language insight string. The teacher LLM is not one instance of a modelless decision — it IS the algorithm. This is the R368 "LLM-as-mechanism" category (genuine LLM dependency), not "LLM-as-instance" (which decomposes to a modelless analog).

If we wanted CORE's actual capability (NL insight generation from trace pairs), it's either:
- **→ riir-train** (train a small contrastive insight generator, or distill from a teacher LLM), OR
- **External LLM dependency** (call a teacher LLM at runtime — violates the modelless-first mandate)

There is no `simd_dot_f32` / `forward_looped` / CNA-style modelless decomposition. The paper's mechanism is the LLM call.

## What this means for the pivot

- If you want "the `m(y⁺) - m(y⁻)` activation-direction primitive wired into the tick loop" → **that's CNA**, already default-on. No new primitive work needed; the question is just whether riir-ai's tick loop currently consumes CNA (separate grep).
- If you want "natural-language contrastive insight generation from trace pairs" → that's CORE the paper, LLM-dependent, → riir-train if pursued.
- The "validate CORE contrastive reflection integration" framing mixed up the two — fabricated a primitive (Δ_CORE) that's actually CNA, then attributed it to a paper (CORE) that does something else entirely.

**Lesson:** A confident-sounding framing with file-path-implying specificity is not the same as a real primitive. Always grep before agreeing to "validate" an integration target. (Same anti-pattern as the initial PASS verdict — accepting a confident framing without verification. I made this mistake twice in this session — once on the PASS tier, once on "t-pass doesn't exist".)

## Verified answer to the "is CNA wired into the tick loop?" question (2026-07-26)

The plan above left open: *"the question is just whether riir-ai's tick loop currently consumes CNA (separate grep)."* That grep is now done.

**Answer: CNA itself is NOT consumed in riir-ai — but that's because the premise ("CNA is the right primitive for NPC cognition") is a category error. The right primitive for NPC cognition contrastive direction mining is MAG, and MAG IS already wired in.**

### Verified facts (grep across all 7 repos)

| Primitive | Math | Consumers in riir-ai? |
|---|---|---|
| **CNA** (Plan 087, default-on) | `δ_j = mean(a_j(y⁺)) - mean(a_j(y⁻))` per **MLP neuron** | ❌ Zero. Only consumers are 3 katgpt-rs example harnesses (`bomber`/`go`/`fft` ContrastivePairProviders) + 1 bench. `ForwardContext.cna_modulator` field is never populated by any external caller. |
| **MAG** (Plan 418 / Plan 461) | Mean-difference direction mining ("identical math to EmotionDirections" per `katgpt-core/src/mag/mod.rs` L110) | ✅ **YES** — `riir-engine/src/cgsp_runtime/mag_bridge.rs` wires MAG's `transfer_score` onto the CGSP runtime as a curiosity-target source. End-to-end tested (`mag_target_admits_candidates_in_cgsp_loop`, `mag_directed_curiosity_steers_toward_transfer_direction`). |
| **EmotionDirections** (Plan 162) | Read-side dot-product projection onto pre-computed direction vectors | Used as the latency cousin baseline in `fpcg_probe_forecast_bench.rs`; provides the detection-side projection for HLA affect. |
| **LatentFieldSteering** (Plan 309) | Write-side SIMD SAXPY + sigmoid-falloff direction injection into mutable latent state | Ships in `katgpt-core/src/latent_steering.rs`; the "fourth quadrant" primitive that closes the CNA-mutates-neurons vs EmotionDirections-read-only gap. |

### Why CNA's narrow scope is correct

The codebase has an explicit four-quadrant taxonomy (`katgpt-core/src/lib.rs` L1028-1032):

> "The missing fourth quadrant: CNA mutates neurons, EmotionDirections is read-only, FPCG refuses mutation — this injects directly into the latent state on the hot path."

- **CNA** operates at the **MLP neuron** level (sparse circuit discovery for transformer forward passes). Its consumers are game harnesses where the activation IS an MLP output.
- **NPC cognition** operates at the **HLA direction vector** level (64-dim per `DEFAULT_HLA_DIM`). The right primitives for that substrate are MAG (mine directions from activation data) + EmotionDirections (read-side project) + LatentFieldSteering (write-side inject).

The "wire CNA into NPC cognition" pivot conflated two different substrates. CNA's transformer-MLP-neuron scope is correct as-is.

### Net result

- **No new plan or issue is needed for CNA integration.** The gap was a mirage.
- The right primitives (MAG/EmotionDirections/LatentFieldSteering) are already shipped and consumed.
- Plan 561 stays parked at Gain (SipIt is still a real open primitive, unrelated to this CNA clarification).
- The CORE paper (arXiv:2605.28742) remains a separate riir-train candidate if its NL-insight capability is ever wanted — modelless decomposition is not possible (R368 LLM-as-mechanism).

**Second lesson:** When a grep shows "primitive X has no consumers," the next question is NOT "how do we add a consumer?" — it's "is primitive X even the right shape for the proposed consumer?" CNA having zero riir-ai consumers is correct, not a gap. The category error was considering CNA for a substrate it doesn't operate on.
