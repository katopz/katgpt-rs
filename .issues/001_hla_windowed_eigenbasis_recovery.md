# Issue 001: HLA Windowed Eigenbasis Recovery

> **Type:** Optimization / Super-GOAT candidate
> **Status:** Open — needs GOAT/gain proof before promote
> **Owner:** unassigned
> **Created:** 2026-06-21
> **Cross-repo:** Would land as katgpt-rs primitive + riir-ai application guide
> **Origin:** SVD audit (conversation 2026-06-21) — gap #1 of 3

---

## TL;DR

HLA's per-NPC latent state ships 5 hand-tuned "emotional direction" axes
(valence, arousal, desperation, calm, fear + 3) defined by `evolve_hla`
(`katgpt-rs/crates/katgpt-core/src/sense/reconstruction.rs`) and the leaky
integrator in `katgpt-rs/crates/katgpt-core/src/leaky_core.rs`. Those axes
are **universal** — every NPC shares the same basis. Recovering a
**per-NPC eigenbasis** from a recent window of that NPC's own activations
would expose individualized affective geometry without any training.

This is a modelless, runtime, latent-to-latent operation: windowed
activations → small SVD → k orthogonal directions → sigmoid-bounded
projections. No backprop, no sync (stays local to NPC).

---

## Primitive

`fn recover_eigenbasis_from_window(
    window: &[f32],   // shape (T, D), row-major, T = ticks observed
    t_dim: usize,     // = D = HLA state dim (8 today)
    out_eigvecs: &mut [f32],  // shape (D, k), row-major
    out_eigvals: &mut [f32],  // length k, descending
    scratch: &mut [f32],
    k: usize,
    iters: u8,        // power-iteration count, default 5 (same as data_probe)
) -> Result<(), Error>`

Implementation choices (constrained by existing shipped patterns):

- **No LAPACK.** Follow `katgpt-rs/src/newton_schulz.rs` + `off_principal.rs`
  pattern: power iteration on the `D × D` Gram matrix `W^T W` with deflation
  (mirror `katgpt-rs/crates/katgpt-core/src/dec/hodge.rs::hodge_spectrum`).
- **Zero-alloc hot path.** Caller-owned scratch, same contract as
  `off_principal_project` and `stable_rank_update_into`.
- **`Uuid::now_v7()`** for any per-window ID (per AGENTS.md).
- **Deterministic across platforms** (same seed strategy as
  `stable_rank_update_into` — `1/sqrt(d)` init, no RNG).

---

## Gap (precise)

Already shipped, NOT what this issue asks for:

- `MultiLayerHlaCache` (`riir-ai/crates/riir-engine/src/hla/types.rs`)
  tracks per-Q-head moments (`cqv`, `mq`, `g`, `h`, `role`, `third_order`)
  and `ThirdOrderMoment { rank, dv }`. **That is a per-head compressed
  third-order moment, not a per-NPC direction eigenbasis.** Do not confuse
  the two.
- `latent_functor/k_selector.rs` (Plan 318 Phase 5) ships a UCB1 bandit
  choosing the rank `k` per `(npc, relation)`. **That is rank selection for
  an extraction operator, not eigenbasis recovery from raw activations.**
- `katgpt-rs/src/pruners/belief_rank_pruner.rs` uses participation ratio on
  the variance **diagonal** (deliberately avoiding full SVD). The gap here
  is the **opposite**: we *want* the off-diagonal eigenvectors.

Novel angle this issue proposes:

- Recover **k orthogonal directions** that capture the dominant variance of
  a single NPC's recent HLA activations.
- Use them as a **per-NPC rotation/projection matrix** for emotion routing,
  zone attention, or adapter selection.
- Per-NPC emergent individuality emerges from data, not from hand-tuned
  universal axes.

---

## Fusion angle

This is a Super-GOAT candidate (per skill §1.5, Q1–Q4) IF the gain is real.
Fusion ingredients already shipped:

- **A** = Research 032 (Functional Emotions HLA, riir-ai/.research/032) —
  hand-tuned emotional direction vectors.
- **B** = Plan 318 rank-k upgrade — operator-valued C matrix in
  `latent_functor/`. Already does rank-k on relations; this issue proposes
  rank-k on the **activations themselves**, not on relations.
- **C** = `katgpt-rs/src/off_principal.rs` — cached top-k SVD via NS on Gram
  matrix. Same primitive shape, different application.

**Super-GOAT claim:** "Our NPCs discover their own emotional axes from
experience, no two NPCs share the same affective basis, and the discovered
directions are versioned by freeze/thaw."

This is a private selling point → if it passes, an architectural guide must
land in `riir-ai/.research/NNN_HLA_Eigenbasis_Recovery_Guide.md` (NOT
public — per skill §1.5, "Super-GOAT = private moat; never skip the
riir-ai guide").

---

## GOAT / Gain proof — REQUIRED

This issue is **not done** until the following gate passes. No promotion to
default, no guide doc, until numbers are produced.

### Bench setup

- Synthetic windowed HLA activations: `T ∈ {128, 512, 2048}`, `D = 8`
  (current HLA dim).
- Baselines:
  1. **Identity** — current behavior (no per-NPC eigenbasis).
  2. **Hand-tuned axes** — Research 032's 5 emotional directions.
  3. **This primitive** — recovered eigenbasis at `k ∈ {1, 2, 4}`.
- NPC count for crowd test: `N ∈ {1, 1000}` (single-NPC and MMORPG scale).

### Gates (must hit all to be GOAT)

| Gate | Metric | Threshold | vs Baseline |
|------|--------|-----------|-------------|
| G1 — Latency | `recover_eigenbasis_from_window` single call, `T=512, D=8, k=4, iters=5` | ≤ **2 µs** on M-series SIMD | n/a (must fit plasma tier) |
| G2 — Determinism | bit-identical eigenvalues across `x86_64`, `aarch64`, `wasm32` | 0 diffs at `f32` round-trip through `LatCalFixed` (i64 × 10⁶) | mandatory — no SVD on the sync boundary |
| G3 — Quality | Reconstruction: `‖W − U_k Σ_k V_k^T‖_F / ‖W‖_F ≤ 0.10` for `k=4` on synthetic data with known rank-3 ground truth | < 10% relative error | vs identity (which gives 100% error) |
| G4 — Behavioral divergence | Two NPCs with same input distribution but different observed windows produce eigenbases with principal-direction cosine < 0.7 (i.e. genuinely different axes) | > 30° angular separation on ≥ 50% of NPC pairs in 1000-NPC sim | vs hand-tuned axes (always identical) |
| G5 — Memory | Per-NPC overhead (eigvecs + eigvals + scratch reuse) | ≤ **256 bytes** at `D=8, k=4` | mandatory — must fit per-NPC budget at MMORPG scale |

### Outcome matrix

- **All G1–G5 pass** → **GOAT**. Promote to default if it wins the head-to-head
  against hand-tuned axes on G3+G4. Demote the loser (likely identity).
- **G3 or G4 fail** → **Gain**. Keep behind feature flag
  `hla_eigenbasis_recovery`. Document the partial win (e.g. "good for
  diagnostic logging, not for routing").
- **G1 or G2 or G5 fail** → **Pass**. Do not implement. The primitive is not
  viable for plasma-tier per-NPC use; revisit only if HLA dim grows.

---

## Where to implement

| Layer | File | Notes |
|---|---|---|
| Primitive (public, MIT) | `katgpt-rs/src/hla_eigenbasis.rs` + `katgpt-rs/src/lib.rs` re-export | Generic: any `(T, D)` windowed matrix → top-k eigvecs/eigvals |
| Feature flag | `hla_eigenbasis_recovery` in `katgpt-rs/Cargo.toml` | Opt-in until GOAT passes |
| Bench | `katgpt-rs/benches/hla_eigenbasis_bench.rs` | G1, G3, G5 measurements |
| Determinism test | `katgpt-rs/tests/hla_eigenbasis_determinism.rs` | G2 cross-platform bit-identical check |
| NPC integration (private) | `riir-ai/crates/riir-engine/src/hla/eigenbasis.rs` (new file) | Consumes the katgpt-rs primitive, applies per-NPC |
| Game systems fusion | `riir-ai/crates/riir-games/src/npc/` and `latent_functor/zone_gating.rs` | Use recovered eigenbasis for zone attention + adapter routing |
| Architectural guide (private, IF GOAT) | `riir-ai/.research/NNN_HLA_Eigenbasis_Recovery_Guide.md` | Per skill §1.5 mandatory output |

---

## Raw vs latent boundary (critical)

Per AGENTS.md sync-boundary rule:

- **Stays latent / local to NPC:** the recovered eigenbasis, eigenvectors,
  eigenvalues, projections. **Never synced.**
- **Crosses sync boundary only as raw scalars:** if a downstream behavior
  (e.g. emotional taunt line selection) needs to commit, project to the 5
  raw emotion scalars (valence/arousal/desperation/calm/fear) via bridge
  function and commit those — NOT the 8-dim eigenbasis.
- **NEVER** pass eigenbasis through `LatCalFixed` commit. LatCal is for
  deterministic raw arithmetic (2×2 i64 matrices), not for f32 SVD outputs.
- **NEVER** use the recovered basis for anti-cheat validation — anti-cheat
  needs raw exact values (HP, position, wallet), not projections.

---

## Risks

1. **Determinism across platforms.** SVD on `f32` is famously non-portable
   (LAPACK vendor diffs, SIMD reduction order). Mitigation: power iteration
   on the Gram matrix with a fixed iteration count and a deterministic seed
   vector (same approach `stable_rank_update_into` uses — already proven
   cross-platform bit-identical).
2. **Overfitting to recent window.** If `T` is too small, the eigenbasis is
   noise. Mitigation: minimum `T ≥ 4·D` (32 ticks at D=8), EMA smoothing
   across windows, and a confidence gate `sigmoid(energy_ratio − τ)`.
3. **Cache invalidation under freeze/thaw.** When an NPC's frozen snapshot
   is reloaded, the cached eigenbasis must be BLAKE3-checked against the
   activation window that produced it. Re-derive if mismatch.
4. **False "novelty" risk.** `MultiLayerHlaCache::ThirdOrderMoment` already
   tracks compressed moments — must prove this primitive recovers
   *orthogonal directions*, not just diagonal variances. G3 + G4 disambiguate.

---

## Acceptance

- [ ] Primitive ships behind `hla_eigenbasis_recovery` feature flag.
- [ ] G1–G5 benchmark report committed to `katgpt-rs/.benchmarks/`.
- [ ] If GOAT: feature promoted to default, loser demoted, riir-ai guide
      created.
- [ ] If Gain: feature stays opt-in, partial-win documented in this issue.
- [ ] If Pass: primitive removed, this issue closed with verdict notes.

---

## TL;DR of the TL;DR

Add a per-NPC eigenbasis recovery primitive (windowed SVD via power iter on
Gram, no LAPACK, zero-alloc). Must hit 5 gates (latency ≤ 2 µs,
determinism cross-platform, reconstruction ≤ 10% error, behavioral
divergence > 30° on ≥ 50% NPC pairs, ≤ 256 bytes/NPC). If all pass → GOAT,
promote to default + write riir-ai private guide. If not → keep opt-in or
kill.
