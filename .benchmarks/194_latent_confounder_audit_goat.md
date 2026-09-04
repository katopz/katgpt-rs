# Benchmark 194 — LatentConfounderAudit GOAT Gate

> **Issue:** `katgpt-rs/.issues/194_latent_confounder_audit_primitive.md` *(removed per noise-reduction rule — this file is the durable home)*
> **Research:** [katgpt-rs/.research/460_CD_LAM_Latent_Confounder_Audit_Diagnostics.md](../.research/460_CD_LAM_Latent_Confounder_Audit_Diagnostics.md)
> **Source paper:** [arXiv:2607.09185](https://arxiv.org/abs/2607.09185) — Wei et al., *Causally Debiased Latent Action Model for Embodied Action Conditioned World Models* (CD-LAM §III-B + Appendix A)
> **Date:** 2026-07-28
> **Feature gate:** `latent_confounder_audit` (opt-in, NOT default — diagnostic primitive)
> **Status:** ✅ **G1–G4 PASS** modellessly. **Stays opt-in** — diagnostic primitives are promoted only when a concrete consumer benchmarks a quality gain (no consumer yet).

---

## Goal

Prove the three CD-LAM §III-B diagnostics (zero-transition response, shift-invariance
response, shortcut leakage) ship as a modelless, alloc-free, sub-µs primitive that
correctly identifies confounders in a synthetic encoder with a known confounder
coefficient `c`.

## The three diagnostics (recap)

| Diagnostic | Formula | Clean value | What it tests |
|---|---|---|---|
| Zero-transition response | `R₀ = RMS(‖E(x, x)‖) / D` | ≈ 0 | No-op input pair should produce near-zero latent |
| Shift-invariance response | `R_shift = RMS(‖E(x, T(x))‖) / D` | ≈ 0 | Nuisance transform should produce near-zero latent |
| Shortcut leakage | `mean_cos(diff-action) − mean_cos(same-action)` | < 0 | Action similarity should dominate context similarity |

Where `D = RMS(‖E(x, x′)‖) + ε` over ordinary transitions.

## GOAT gate summary

| Gate | Criterion | Result | Status |
|---|---|---|---|
| **G1** correctness | Diagnostics correctly identify confounders on synthetic encoder `E(x,x') = A(x,x') + c·confounder(x)`; monotone in `c` | 12 unit tests + 1 doctest. Clean (c=0): R₀<1e-5, R_shift<1e-5, L<0. Confounded (c=2.0): R₀>0.1, R_shift>0.1, L>-0.5. Monotone across c∈{0, 0.5, 1, 2, 5}. | ✅ PASS |
| **G2** perf | Sub-µs per audit at belief d=8 | **292 ns/call** at d=8 (3.4× under 1µs target). Sweep: d=32 = 750 ns, d=64 = 1.38 µs. | ✅ PASS |
| **G3** no-regression | New module, feature-gated, no existing code touched | `cargo check -p katgpt-core --all-features` clean. Default test count unchanged: 1814 → 1814. With feature on: 1814+12=1826. | ✅ PASS |
| **G4** alloc-free | Pre-allocated `AuditScratch`, zero steady-state allocation | `g4_audit_confounders_zero_alloc_steady_state`: 0 allocations across 100 audit calls. TrackingAllocator sentinel-verified (skips cleanly in binaries without it installed). | ✅ PASS |

## Perf details (G2)

Median of timed runs, Apple Silicon arm64 release build. Bench source:
`benches/bench_194_latent_confounder_audit_goat.rs`.

| Shape | latent dim | per-audit time |
|---|---|---|
| **belief (G2 target)** | **8**  | **292 ns** ✅ |
| Shard style_weights | 32 | 750 ns |
| Full shard          | 64 | 1.38 µs |

The audit is O(d) per check (norm + cosine). At belief scale it is essentially free;
at shard scale (d=64) it is still sub-2µs — comfortable for an offline pre-deployment
gate, which is the intended consumer pattern.

## What this proves

The three diagnostics correctly identify confounder purity on a synthetic encoder
where the ground-truth confounder coefficient is known. The signal is monotone in
`c` (more confounder → larger R₀, R_shift, less-negative L), so the audit can be
used as a quantitative purity score, not just a binary pass/fail.

## What this does NOT prove

1. **Does not prove the audit catches real bugs in production-mined direction
   vectors.** The G1 synthetic encoder has a known, injected confounder; real
   mined directions (MAG Plan 418, TILR Plan 425, Latent Field Steering Plan 309)
   could have subtler confounders that the three diagnostics miss. The consumer
   adoption gate (below) is where that gets tested.
2. **Does not prove the conformal-naive floor is beaten.** The "Report the
   Floor" rule (Research 322 / Plan 340) does NOT apply — the three metrics are
   raw geometric measurements (norm ratios, cosine gaps), NOT probabilities /
   confidence scores / predictive intervals. There is no distributional claim.
3. **Does not prove a quality gain in a downstream consumer.** Promotion to
   default-on requires a consumer (MAG/TILR/Steering/Blend) to benchmark a
   real-bug-caught gain. No consumer has adopted the audit yet.

## Design decisions (captured for future maintainers)

- **Encoder API:** `Fn(&[f32], &[f32], &mut [f32])` — output buffer as 3rd arg.
  Sidesteps the lifetime issues with `Fn(...) -> &[f32]` (HRTB constraints) and
  works with any closure.
- **Sign convention:** `shortcut_leakage < 0 = clean` (matches the issue spec).
  Formula is `mean_cos(diff-action, same-context) − mean_cos(same-action,
  diff-context)` so action-dominance makes the value negative.
- **RMS normalization:** R₀ and R_shift use `sqrt(mean(‖·‖²))` (matching D's
  form), not raw mean.
- **Scratch:** `AuditScratch::new(latent_dim)` pre-sizes two `Vec<f32>` buffers;
  `resize()` handles multi-dim audits. Zero steady-state allocation.
- **G4 sentinel logic:** the test allocates a known sentinel `Vec<u8>` first; if
  the counter didn't increase, the TrackingAllocator isn't installed — skip.
  Otherwise the audit truly is alloc-free. (Distinguishes "0 allocs = PASS" from
  "allocator not installed = unmeasurable".)
- **Test-fixture pitfall:** the clean encoder `clean(x, x')[i] = (x'[i] - x[i]) -
  mean(...)` mean-subtracts the displacement. A *constant* displacement
  `[0.5, 0.5, ...]` mean-subtracts to zero → cosine undefined → treated as 0 →
  shortcut_leakage = 0 instead of < 0. Fix: use **non-constant** displacements
  (a zero-mean ramp like `[-0.28, -0.20, ..., +0.28]`).

## Promotion decision (T7 — DEFERRED)

**G1–G4 PASS modellessly.** Primitive ships **opt-in** (`latent_confounder_audit`
feature flag, default-off).

This is a **diagnostic primitive, not a capability**. Promotion to default-on
requires a concrete consumer (MAG/TILR/Steering/Blend) benchmarking a quality
gain (fewer misconfigured directions deployed). No consumer has adopted the
audit yet — so it stays opt-in by design.

Re-opens when a consumer adopts the audit and demonstrates a real-bug-caught gain.

## Consumer adoption (the T7 promotion gate)

Any of the following could adopt the audit as a pre-deployment gate, which would
re-open T7 for promotion to default-on:

| Consumer | What it audits | When |
|---|---|---|
| MAG (Plan 418) | Mined direction vectors | Before deploying a mined direction — reject if confounders detected |
| TILR (Plan 425) | Refined trajectory-invariant directions | After refinement pass |
| Latent Field Steering (Plan 309) | Steering direction vectors | Before injecting a steering vector |
| Committed Personality Blend (321) | Archetype direction vectors | Before committing a blend |
| HLA `evolve_hla` | Per-NPC affect direction vectors | CI test: verify hand-constructed directions are clean |
| `extract_functor` | Functor displacement vectors | CI test: verify functor has translation invariance |

## Deferred to riir-train

CD-LAM's training recipe (`L_emb` + `L_ctr` + `L_cal` + three-stage fine-tuning
pipeline) is genuinely gradient-descent and routes to riir-train if a video world
model or analogous training system is built. Documented in Research 460 §3.2;
§3.5 Path 0 confirmed all three objectives are genuinely training losses.

## Reproduction

```bash
# G1 unit tests:
cargo test -p katgpt-core --features latent_confounder_audit --lib latent_confounder_audit

# G2 perf bench:
cargo bench -p katgpt-core --bench bench_194_latent_confounder_audit_goat \
    --features latent_confounder_audit

# G4 alloc-free test (visible proof):
cargo test -p katgpt-core --features latent_confounder_audit --lib \
    latent_confounder_audit::tests::g4_audit_confounders_zero_alloc_steady_state \
    -- --nocapture

# G3 no-regression check:
cargo check -p katgpt-core --all-features
cargo test -p katgpt-core --lib  # default-feature test count unchanged (1814)
```

## Files shipped

- `crates/katgpt-core/src/latent_confounder_audit.rs` — the primitive: struct +
  audit fn + scratch + helpers + 12 unit tests + 1 doctest (~600 LOC).
- `crates/katgpt-core/Cargo.toml` — `latent_confounder_audit = []` feature
  declaration (opt-in) + `[[bench]]` entry for the GOAT bench.
- `crates/katgpt-core/src/lib.rs` — `#[cfg(feature = "latent_confounder_audit")]
  pub mod latent_confounder_audit;` + re-export of `AuditScratch`,
  `LatentConfounderAudit`, `audit_confounders`.
- `crates/katgpt-core/benches/bench_194_latent_confounder_audit_goat.rs` — G2
  perf bench (mirrors `bench_342_latent_trajectory_geometry_goat`).

Commit: `3c80389a` on `develop` (5 files, +1414 / -8).

## TL;DR

All four primitive-level GOAT gates PASS (G1 correctness 12 tests, G2 perf 292 ns
at belief d=8, G3 no-regression clean, G4 alloc-free sentinel-verified). Primitive
ships **opt-in** as a validated diagnostic; promotion to default-on is deferred
until a concrete consumer (MAG/TILR/Steering/Blend) benchmarks a real-bug-caught
gain. The CD-LAM training recipe routes to riir-train.
