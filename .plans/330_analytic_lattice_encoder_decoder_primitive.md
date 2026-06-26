# Plan 330: Analytic Lattice Encoder/Decoder + Chain Composer Primitive

**Date:** 2026-06-26 (revised 2026-06-26 — see revision note)
**Research:** [katgpt-rs/.research/311_Analytic_Lattice_Encoder_Decoder_Primitive.md](../.research/311_Analytic_Lattice_Encoder_Decoder_Primitive.md)
**Source paper:** Synthesis (R311 §2) — Functional Attention × PJ-RoPE × Gyrocalculus fusion
**Target:** `katgpt-rs/crates/katgpt-core/src/analytic_lattice/` (new module) + Cargo feature `analytic_lattice_encoder`
**Status:** Active — Phase 0 <scaffold>

> **Revision note (2026-06-26):** Original Phase 1 (`AnalyticLatticeEncoder`
> trait + 3 reference impls) is **DROPPED** — it is redundant with
> `riir-ai/crates/riir-engine/src/fourier/encoder.rs`, which already ships
> `FourierEncoder::encode_position_into` / `encode_offset_into` (closed-form
> analytic encoder). Phase 1 is replaced by **ASOC cascade**
> (`ComposerTick: GpuFuture`) as the new headline primitive, built on the
> already-shipped `riir-gpu-async` `GpuFuture` / `Join` / `block_on`
> substrate. A new Phase 2.5 (`batch_compose_chain`) is added for zone-batched
> prefix factoring. Phases 2/3/4/5 are otherwise preserved (Phase 3 stays
> `compose_chain`; Phase 2 stays `direction_vector_decode`; Phase 4 stays
> spectral audit; Phase 5 stays GOAT gate). See R311 revision note for the
> matching research-layer narrowing.

---

## Goal

Ship the primitives identified as genuinely novel in R311 (revised):

1. **`ASOC ComposerTick: GpuFuture`** (headline) — the autoplay bot's per-tick
   action selector. Always emits a synchronous plasma-tier draft; joins 3 hot-tier
   rederive futures (`C_boss`, `C_quest`, `C_player`) via `GpuFuture::Join`;
   returns the stale plasma draft on `Poll::Pending` so the bot loop never
   blocks.
2. **`compose_chain`** — cross-entity operator product
   `C[n-1] × ... × C[1] × C[0]` for an arbitrary-length chain of `f32`
   transport operators (k×k). The cross-entity analog of `funcattn_compose`
   (which is token-level).
3. **`batch_compose_chain`** — zone-batched prefix factoring: factor the
   shared prefix `C_qb = C_boss × C_quest` once per tick, then apply
   `C_player_i × C_qb` for N players in the same zone (O(N+k³) vs O(N·k³)).
4. **`direction_vector_decode`** — SIMD projection of a latent state onto a
   direction vector, producing an action-score scalar (generalization of
   riir-games `scalar_projection.rs`, lifted out of HLA-specific 5-scalar
   semantics into a generic single-direction primitive).

**Redundant (NOT shipped here):** `AnalyticLatticeEncoder` trait. The
encoder half is already shipped as `FourierEncoder::encode_*_into` in
`riir-engine/src/fourier/encoder.rs` — we reuse it instead of re-shipping.

All four primitives: zero-alloc, SIMD-first, ARM64/x86_64/wasm32-portable,
behind ONE feature flag. GOAT gate G1–G6 (per R311 §5) must pass before
promotion to `default`.

---

## Phase 0 — Module skeleton + types

**Target:** `katgpt-rs/crates/katgpt-core/src/analytic_lattice/mod.rs` (new)

### Tasks

- [ ] **T0.1** Add Cargo feature `analytic_lattice_encoder = []` to `katgpt-rs/crates/katgpt-core/Cargo.toml`. NOT default-on.
- [ ] **T0.2** Create `analytic_lattice/mod.rs` with module doc + sub-module declarations. **Note:** the originally-planned `encoder.rs` submodule is DROPPED (redundant with `fourier/encoder.rs`). New submodule set: `asoc.rs`, `chain.rs`, `batch_chain.rs`, `decoder.rs`, `audit.rs`.
- [ ] **T0.3** Define the typed-slot lattice vector and transport operator:

```rust
/// Typed per-slot lattice vector — 8 lanes matching Plan 335 eggshell.
/// Slot semantics are CALLER-defined (game IP); this primitive is slot-agnostic.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(transparent)]
pub struct LatticeVector<const N: usize>(pub [f32; N]);

/// A k×k transport operator (output of FuncAttn or extract_functor_rank_k).
#[derive(Clone, Debug)]
pub struct TransportOperator {
    pub k: usize,
    pub data: Vec<f32>, // row-major k×k
}
```

- [ ] **T0.4** Wire into `katgpt-core/src/lib.rs` behind the feature flag.

---

## Phase 1 — ASOC cascade core: `ComposerTick: GpuFuture` (HEADLINE)

**Target:** `katgpt-rs/crates/katgpt-core/src/analytic_lattice/asoc.rs` (new)

> This Phase is the **headline novel contribution** (per R311 §3, revised).
> It is the autoplay bot's per-tick action selector. It always emits a
> synchronous plasma-tier draft, then speculatively joins 3 hot-tier rederive
> futures, and returns the stale plasma draft on `Poll::Pending` so the bot
> loop never blocks on GPU.

### Tasks

- [ ] **T1.1** Define the plasma-tier draft trait (generic, no game IP):

```rust
/// Plasma-tier synchronous draft producer. Always completes in nanoseconds;
/// the ASOC cascade returns its stale output when the hot-tier join returns
/// `Poll::Pending` (GPU congestion).
///
/// The concrete implementation lives in riir-ai (e.g. wraps
/// `riir-games::quest_draft::QuestDraftModel`). katgpt-rs ships only the trait.
pub trait PlasmaDraft {
    type Action;
    /// Produce a synchronous draft action. Must not block, allocate, or fail.
    fn draft(&self, ctx: &ComposerCtx) -> Self::Action;
}
```

- [ ] **T1.2** Define the hot-tier rederive trait (generic `GpuFuture` producer):

```rust
/// Hot-tier transport-operator rederive. Produces a `GpuFuture` that resolves
/// to a `TransportOperator` when the GPU work completes. The ASOC cascade
/// joins 3 of these per tick (`C_boss`, `C_quest`, `C_player`).
pub trait RederiveOp {
    type Fut: GpuFuture<Output = TransportOperator>;
    fn rederive(&self, ctx: &ComposerCtx) -> Self::Fut;
}
```

- [ ] **T1.3** Implement `ComposerTick: GpuFuture` — the cascade core:

```rust
/// The ASOC cascade tick. On `poll`:
///   1. If hot-tier join (`C_boss` ⊗ `C_quest` ⊗ `C_player` via `GpuFuture::Join`)
///      returns `Ready`, compose the chain via `compose_chain` and return the
///      decoded action.
///   2. If hot-tier join returns `Pending`, return `Ready(stale_plasma_draft)`
///      instead of propagating `Pending`. **This is the key non-blocking
///      guarantee** — the bot loop never stalls on GPU congestion.
pub struct ComposerTick<P, Rb, Rq, Rp, D> {
    plasma: P,             // PlasmaDraft
    rederive_boss: Rb,     // RederiveOp
    rederive_quest: Rq,    // RederiveOp
    rederive_player: Rp,   // RederiveOp
    decoder: D,            // direction_vector_decode closure / trait obj
    stale_draft: Option<P::Action>,  // cached plasma draft
    join: Option<Join3<Rb::Fut, Rq::Fut, Rp::Fut>>,
}

impl<P, Rb, Rq, Rp, D> GpuFuture for ComposerTick<P, Rb, Rq, Rp, D>
where
    P: PlasmaDraft + Unpin,
    Rb: RederiveOp + Unpin, Rb::Fut: Unpin,
    Rq: RederiveOp + Unpin, Rq::Fut: Unpin,
    Rp: RederiveOp + Unpin, Rp::Fut: Unpin,
    D: FnMut(&TransportOperator) -> P::Action,
{
    type Output = P::Action;
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<P::Action> {
        // 1. Always refresh the stale plasma draft up-front (cheap, sync).
        // 2. Lazy-init the join on first poll.
        // 3. Poll the join:
        //    - Ready((Cb, Cq, Cp)) => compose_chain + decode => Ready(action)
        //    - Pending            => Ready(stale_draft.take().unwrap())  <-- KEY
        //    ^^^ Note: we return Ready with the stale draft, NOT Pending.
        //        This is what makes the bot loop non-blocking.
    }
}
```

- [ ] **T1.4** `Join3` helper — a 3-way `GpuFuture::Join` (the shipped `Join`
      is 2-way; nest `Join<Join<A, B>, C>` or add a small `Join3` combinator
      in this module — prefer nesting to avoid growing `riir-gpu-async`).

- [ ] **T1.5** G1 test (determinism): same `(ctx, plasma, rederive)` inputs →
      byte-identical action when the join is `Ready`. Stale-draft fallback
      path tested separately (see T1.6).

- [ ] **T1.6** G1b test (non-blocking contract): inject a `MockRederiveOp`
      that returns `Poll::Pending` indefinitely. Assert `ComposerTick::poll`
      returns `Ready(stale_draft)` (NOT `Pending`) — the bot loop contract.

- [ ] **T1.7** G4 test (latency): plasma-draft path (`Poll::Pending` injected)
      must complete in < 100ns. Hot-tier-join path must complete in < 1µs when
      the join resolves immediately.

---

## Phase 2 — `compose_chain` operator product

**Target:** `katgpt-rs/crates/katgpt-core/src/analytic_lattice/chain.rs`

### Tasks

- [ ] **T2.1** Implement the chain composer (consumed by Phase 1 ASOC):

```rust
/// Compose a chain of k×k transport operators: out = C[n-1] × ... × C[1] × C[0].
///
/// All operators MUST have the same k. Returns the composite operator.
/// This is the cross-entity analog of `funcattn_compose` (which is token-level).
pub fn compose_chain(ops: &[TransportOperator]) -> Result<TransportOperator, ChainError> {
    // Validate same-k, then row-major matmul reduction.
    // Reuse one scratch buffer; zero alloc after first call if caller reuses.
}

/// In-place variant for hot paths (used by ASOC `ComposerTick::poll`).
pub fn compose_chain_into(
    ops: &[TransportOperator],
    scratch: &mut Vec<f32>,
    out: &mut TransportOperator,
) -> Result<(), ChainError> { /* ... */ }
```

- [ ] **T2.2** G3 test: associativity `(A×B)×C ≈ A×(B×C)` within Frobenius ≤ 1e-5.
- [ ] **T2.3** G5 test: `TrackingAllocator` audit shows 0 allocs after warmup.

---

## Phase 2.5 — `batch_compose_chain` zone-batched prefix factoring

**Target:** `katgpt-rs/crates/katgpt-core/src/analytic_lattice/batch_chain.rs`

> **Why this Phase exists (perf).** In a zone with N players, all N players
> share the same `C_boss × C_quest` prefix (the boss and quest are zone-level
> facts). Only the `C_player_i` factor differs. Naive per-player
> `compose_chain(&[C_boss, C_quest, C_player_i])` is O(N·k³). Factoring the
> shared prefix `C_qb = C_boss × C_quest` once and applying
> `C_player_i × C_qb` for each player is O(N·k² + k³) — saves a factor of k
> per player. For k=8 (eggshell lanes) this is ~8× speedup.

### Tasks

- [ ] **T2.5.1** Implement the batched composer:

```rust
/// Zone-batched chain compose. Factors the shared prefix `ops[..prefix_len]`
/// once, then applies the per-player suffix for each of `suffixes`.
///
/// Caller identifies the prefix boundary (typically `prefix_len = 2` for
/// `[C_boss, C_quest]` and per-player suffix `[C_player_i]`).
///
/// Output: one composite operator per suffix.
pub fn batch_compose_chain(
    prefix: &[TransportOperator],
    suffixes: &[&[TransportOperator]], // one suffix per player
    out: &mut [TransportOperator],
    scratch: &mut Vec<f32>,
) -> Result<(), ChainError> {
    // 1. compose_chain_into(prefix, scratch, &mut prefix_composite)  // k³ once
    // 2. for each suffix_i: matmul(suffix_composite_i, prefix_composite)  // k²·N
}

/// Even-hotter variant: prefix + suffixes are both pre-laid-out in
/// row-major contiguous slices; output written into `out` in-place.
/// Zero alloc, SIMD-friendly. Used by ASOC `ComposerTick` when the zone
/// has multiple players (one ComposerTick per zone, not per player).
pub fn batch_compose_chain_into(
    prefix: &[f32],         // k×k row-major
    suffixes: &[f32],       // N×k×k row-major, contiguous
    out: &mut [f32],        // N×k×k row-major, contiguous
    k: usize,
    n: usize,
) { /* ... */ }
```

- [ ] **T2.5.2** G2 test (ranking preservation vs naive): for 100 random
      `(prefix, suffix_i)` sets, the batched output matches the per-player
      `compose_chain` output within Frobenius ≤ 1e-6.

- [ ] **T2.5.3** G4 benchmark: `batch_compose_chain` at N=64 players, k=8 must
      be ≥ 4× faster than 64× `compose_chain` (theoretical 8×; allow
      overhead). Write to `.benchmarks/330_batch_compose_chain.md`.

- [ ] **T2.5.4** Integration with ASOC: when the `ComposerCtx` carries a zone
      with N>1 players, `ComposerTick` switches from per-player
      `compose_chain` to per-zone `batch_compose_chain` and emits N actions
      per tick instead of 1. Document the API extension on `ComposerTick`.

---

## Phase 3 — `direction_vector_decode` SIMD primitive

**Target:** `katgpt-rs/crates/katgpt-core/src/analytic_lattice/decoder.rs`

### Tasks

- [ ] **T3.1** Implement the decoder as a zero-alloc SIMD dot-product + sigmoid
      (consumed by Phase 1 ASOC after the chain compose):

```rust
/// Project `state` onto `direction`, return scalar action score in (0,1).
///
/// This is the GENERALIZED version of riir-games `scalar_projection::project_to_scalars`,
/// lifted out of HLA-specific 5-scalar semantics into a generic single-direction primitive.
/// The 5-scalar HLA bridge in riir-games becomes a thin wrapper that calls this 5 times.
#[inline]
pub fn direction_vector_decode<const N: usize>(
    state: &LatticeVector<N>,
    direction: &LatticeVector<N>,
    temperature: f32,
) -> f32 {
    let z = dot(state.0.as_slice(), direction.0.as_slice()) / N as f32;
    sigmoid(z * temperature)
}
```

- [ ] **T3.2** Add `direction_vector_decode_into` variant for batch decode
      (multiple directions, single state) — used when ASOC decodes one
      composite operator against multiple action-type direction vectors.

- [ ] **T3.3** G2 test: 100 random states × fixed direction, verify ranking
      matches brute-force reference within cos ≥ 0.95.

- [ ] **T3.4** Audit: riir-games `scalar_projection.rs` SHOULD be refactored
      to call this (out of scope here — note as cleanup follow-up in
      `.issues/`).

---

## Phase 4 — Spectral audit verifier (G6)

**Target:** `katgpt-rs/crates/katgpt-core/src/analytic_lattice/audit.rs`

### Tasks

- [ ] **T4.1** Implement `spectral_audit(operator, fourier_modes) -> AuditReport` per arxiv 2606.02427:
  - Compute tangent operator (numerical Jacobian at identity).
  - Project onto Fourier modes (DCT-II for real symmetric, 8 modes default).
  - Return per-mode gain + spurious-coupling matrix.
- [ ] **T4.2** G6 test: known-good composite operator returns max spurious coupling ≤ 5%; known-bad (random operator) returns > 5%.
- [ ] **T4.3** Document: this is the GOAT-gate verifier for chain composition — fails loudly if the chain produces nonsense. Especially important for ASOC because the stale-draft fallback path skips the verifier by design (the bot accepted a possibly-wrong action to stay non-blocking); the spectral audit runs against the *completed* join path's composite operator in the warm-tier reflection cycle (`ReestimationScheduler`), not in the ASOC hot path.

---

## Phase 5 — GOAT gate + promotion/demotion

### Tasks

- [ ] **T5.1** Run all 6 gates in `katgpt-rs/crates/katgpt-core/tests/analytic_lattice_goat.rs`. Gates (per R311 §5):
  - G1 determinism (compose_chain + ASOC `ComposerTick` Ready path)
  - G1b non-blocking contract (ASOC `ComposerTick` returns stale draft on `Poll::Pending`)
  - G2 ranking preservation (decoder + batch_compose_chain vs naive)
  - G3 associativity (compose_chain)
  - G4 latency (plasma-draft path < 100ns, hot-join path < 1µs, batched N=64 ≥ 4× vs naive)
  - G5 zero-alloc steady state (`TrackingAllocator`)
  - G6 spectral audit (max spurious coupling ≤ 5%)
- [ ] **T5.2** Write benchmark to `katgpt-rs/.benchmarks/330_analytic_lattice_goat.md`.
- [ ] **T5.3** If all gates pass: promote `analytic_lattice_encoder` to `default` in katgpt-core Cargo.toml.
- [ ] **T5.4** If any gate fails: keep opt-in, document the failure in `.issues/`, decide modelless unblock path per workflow §3.5 (check freeze/thaw, raw/lora, latent correction before any riir-train deferral).

---

## Risks

| Risk | Mitigation |
|---|---|
| `compose_chain` numerically unstable for long chains | Normalize each operator before multiplication (operator norm ≤ 1); cap chain length at 16 in v1 |
| Spectral audit G6 too strict (false positives) | Calibrate threshold on known-good composites from Plan 335 eggshell lanes; document baseline |
| ASOC stale-draft fallback accepts a wrong action | Acceptable by design — the warm-tier `ReestimationScheduler` reflection cycle re-audits the completed composite operator via spectral audit and emits a correction if the action was wrong. Document this two-tier discipline. |
| ASOC `Join3` combinator duplicates `riir-gpu-async` API surface | Prefer nesting `Join<Join<A, B>, C>` over adding a new combinator to `riir-gpu-async`. Only add `Join3` locally if nesting hurts readability or perf. |
| Encoder determinism across targets (G1) fails on wasm32 | Use `floor` / `round` consistently; avoid `libm` calls that differ across targets |
| Decoder G2 ranking fails on adversarial direction vectors | Use temperature annealing during validation; document the failure envelope |
| `batch_compose_chain` G4 speedup < 4× at k=8 | Investigate: SIMD lane width mismatch, cache misses on N×k×k suffix block; document and either promote-at-lower-bound or keep opt-in |

## Out of scope

- Game-specific encoding schemas (quest/zone/boss/player) — those live in riir-ai (R162 guide, P339 demo).
- Bevy demo — lives in riir-ai/.plans/339.
- Chain length > 16 — defer until G3 holds at length 16.
- Cross-resolution transport (Plan 310) composition — separate primitive, may fuse later.
- Refactoring `riir-games::scalar_projection.rs` to call `direction_vector_decode` — noted as cleanup follow-up, not in this plan.
- ~~`AnalyticLatticeEncoder` trait~~ — DROPPED, redundant with `riir-engine/src/fourier/encoder.rs`.

## TL;DR

**Revised.** Four open primitives (ASOC `ComposerTick: GpuFuture` headline,
`compose_chain`, `batch_compose_chain`, generic `direction_vector_decode`) + 1
verifier (spectral audit) behind `analytic_lattice_encoder` feature flag.
7-gate GOAT (G1 determinism, G1b non-blocking contract, G2 ranking, G3
associativity, G4 latency + batch speedup, G5 zero-alloc, G6 spectral audit).
Promotes to default if all pass. The originally-planned `AnalyticLatticeEncoder`
trait is DROPPED — redundant with the already-shipped `fourier/encoder.rs`.
Game-side schemas live in riir-ai (Plan 339 demo, R162 guide). Math is generic —
no game IP leaks to katgpt-rs.
