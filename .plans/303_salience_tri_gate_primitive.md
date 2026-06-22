# Plan 303: Salience Tri-Gate Primitive — Per-Tick Speak / Silent / Delegate (Modelless)

**Date:** 2026-06-22
**Research:** [katgpt-rs/.research/281_Per_Tick_Salience_Tri_Gate_Speak_Silent_Delegate.md](../.research/281_Per_Tick_Salience_Tri_Gate_Speak_Silent_Delegate.md)
**Private guide:** [riir-ai/.research/148_Per_Tick_Emit_Salience_NPC_Guide.md](../../riir-ai/.research/148_Per_Tick_Emit_Salience_NPC_Guide.md)
**Runtime plan:** [riir-ai/.plans/330_proactive_npc_salience_gate_runtime.md](../../riir-ai/.plans/330_proactive_npc_salience_gate_runtime.md)
**Source paper:** [arxiv 2606.14777](https://arxiv.org/abs/2606.14777) — JoyAI-VL-Interaction (Yao et al., JD.com, Jun 2026)
**Target:** `katgpt-rs/src/salience/` (new module) + Cargo feature `salience_tri_gate`
**Status:** Active — Phase 1 (skeleton unblock)

---

## Goal

Ship the **open modelless primitive** that distills JoyAI-VL-Interaction's per-second emit decision into a generic 3-way gate. The primitive consumes any latent activation + two context scalars (zone-attention, curiosity) and produces one of three first-class decisions: `Speak`, `Silent`, `Delegate`. **Zero game semantics in this crate** — NPC wiring lives in riir-ai Plan 330.

The primitive must be:
- **Generic** over activation dimension `D` and delegate payload type `A`.
- **Zero-allocation** on the hot path (all state stack-allocated, fixed-size).
- **Two stacked sigmoids** (never softmax — per AGENTS.md).
- **Silent as a first-class variant**, not a threshold-suppression default.
- **Async-delegate-friendly**: `DelegateToken` is a typed handoff; the caller decides what to do with it. The primitive does not block.
- **Deterministic** given its inputs (replay-correct).

GOAT gate: G1 (determinism + monotonicity) and G2 (two-sigmoid ablation parity) must pass before merging Phase 2.

---

## Phase 1 — Unblocking Skeleton (CORE)

### Tasks

- [ ] **T1.1** Create `katgpt-rs/src/salience/mod.rs` with module-level doc referencing Plan 303 + Research 281.
- [ ] **T1.2** Add Cargo feature `salience_tri_gate` to `katgpt-rs/Cargo.toml` (opt-in, default off). Gate the entire module behind it.
- [ ] **T1.3** Wire `pub mod salience;` into `katgpt-rs/src/lib.rs` behind the feature flag.
- [ ] **T1.4** Define the core types in `katgpt-rs/src/salience/types.rs`:
  ```rust
  /// First-class output of the salience gate. Silent is a decision, not a default.
  #[derive(Clone, Copy, Debug, PartialEq)]
  pub enum SalienceDecision<A> {
      Silent,
      Speak,
      Delegate(A),
  }
  
  /// Newtype wrapper signaling "this NPC actively chose silence this tick".
  /// Flow through the same channels as Speak/Delegate so subscribers can observe it.
  #[derive(Clone, Copy, Debug, PartialEq, Eq)]
  pub struct SilenceToken {
      pub tick: u64,
  }
  
  impl SilenceToken {
      #[inline]
      pub fn new(tick: u64) -> Self { Self { tick } }
  }
  
  /// Typed handoff returned by the Delegate variant. Caller spawns async task.
  #[derive(Clone, Debug)]
  pub struct DelegateToken<A: Clone> {
      pub payload: A,
      pub issued_tick: u64,
      pub holding_reply_idx: u8,  // index into a caller-provided template table
      pub foldback_target: FoldbackTarget,
  }
  
  /// Where the async result lands. Open enum — generic over backend.
  #[derive(Clone, Copy, Debug, PartialEq, Eq)]
  #[repr(u8)]
  pub enum FoldbackTarget {
      ActivationState = 0,   // result becomes a new direction in the caller's latent state
      PatternMemory    = 1,  // result is a hash-addressed pattern (caller's memory system)
      ExternalJudge    = 2,  // result routes through an external gateway (caller's network)
      ColdTier         = 3,  // result is a frozen shard (caller's persistence layer)
  }
  ```
- [ ] **T1.5** Define the gate struct in `katgpt-rs/src/salience/gate.rs`:
  ```rust
  /// 3-way salience gate. Maps activation `a` + scalars `z`, `c` to one of
  /// {Speak, Silent, Delegate}. Uses two stacked sigmoids — never softmax.
  ///
  /// Generic over activation dimension `D` and delegate payload `A`.
  /// Zero-allocation on the hot path; all state is fixed-size.
  pub struct SalienceTriGate<A, const D: usize> {
      /// Direction vector for "what makes this agent want to speak".
      /// BLAKE3-committed at freeze/thaw by the caller (this crate is agnostic).
      d_speak: [f32; D],
      /// Direction vector for "what makes this agent want to delegate vs answer inline".
      d_delegate: [f32; D],
      /// Weights for zone-attention and curiosity scalar inputs.
      w_z: f32,
      w_c: f32,
      /// Sigmoid inverse temperatures (sharpness).
      beta_speak: f32,
      beta_delegate: f32,
      /// Decision thresholds.
      tau_speak: f32,
      tau_delegate: f32,
      /// Anti-babble floor — below this speak score, always Silent.
      floor_speak: f32,
      /// Delegate ceiling — above this delegate score, prefer Delegate over Speak.
      ceil_delegate: f32,
      _marker: PhantomData<A>,
  }
  ```
- [ ] **T1.6** Implement `SalienceTriGate::new(d_speak, d_delegate, w_z, w_c, beta_speak, beta_delegate, tau_speak, tau_delegate)` constructor. Validates that `D >= 1`, all direction vectors are finite, weights non-negative.
- [ ] **T1.7** Implement `SalienceTriGate::decide(&self, a: &[f32; D], z: f32, c: f32, delegate_payload: A, tick: u64) -> SalienceDecision<A>`:
  - Compute `salience = dot(a, d_speak) + w_z * z + w_c * c`.
  - Compute `score_speak = sigmoid(beta_speak * (salience - tau_speak))`.
  - Compute `delegate_dot = dot(a, d_delegate)`.
  - Compute `score_delegate = sigmoid(beta_delegate * (delegate_dot - tau_delegate))`.
  - Decision rule:
    ```
    if score_speak < floor_speak:        Silent
    elif score_delegate > ceil_delegate: Delegate(delegate_payload)
    else:                                Speak
    ```
  - All branches return a `SalienceDecision<A>` — Silent is first-class.
- [ ] **T1.8** Reuse `crate::simd::fast_sigmoid` for the sigmoid (already shipped, libm-exp-bounded). Add a doc note that we never use softmax.
- [ ] **T1.9** Use `mul_add` for the dot-product accumulation (matches the `ActionBridge` pattern in `bridge/mod.rs`). Add an inline SIMD note.
- [ ] **T1.10** Implement `SalienceTriGate::decide_batch(&self, activations: &[[f32; D]], z: &[f32], c: &[f32], payloads: &[A], tick: u64, out: &mut [SalienceDecision<A>])` — same logic, batched. Caller provides output buffer; no internal allocation.

### Phase 1 acceptance

- `cargo check --features salience_tri_gate` passes.
- `cargo check --no-default-features` still passes (no leakage).
- 3 unit tests: Silent path, Speak path, Delegate path — each constructs a gate with hand-tuned vectors, runs `decide()`, asserts the variant.

---

## Phase 2 — GOAT Gate Skeleton (G1 + G2)

### Tasks

- [ ] **T2.1** Implement property tests in `katgpt-rs/src/salience/gate.rs::tests`:
  - **G1 determinism**: same inputs → same decision (run `decide` twice, assert equal).
  - **G1 monotonicity in salience**: hold `a, z, c` such that `salience < tau_speak`; increase one component of `a` along `d_speak` direction; verify decision transitions Silent→Speak at exactly one threshold crossing.
  - **G1 monotonicity in delegate_dot**: hold others fixed; increase `a` along `d_delegate` direction; verify Speak→Delegate transition is monotone.
  - **G2 ablation parity**: a gate with `ceil_delegate = +∞` (delegate sigmoid never fires) produces bit-identical Silent/Speak sequence to a "speak/silent only" reference implementation over 1000 random inputs.
- [ ] **T2.2** Add a benchmark in `katgpt-rs/benches/salience_tri_gate_bench.rs`:
  - Single `decide()` call latency, D ∈ {8, 16, 32}. Target: < 50ns (cf. `evolve_hla` ~14ns for D=8).
  - Batched `decide_batch()` throughput at N ∈ {1000, 10000} — target ≥ 50M decisions/sec on the test machine.
- [ ] **T2.3** Document the G1/G2 gate criteria in the module doc with the actual numbers when the bench runs.

### Phase 2 acceptance

- G1 (determinism + monotonicity) passes for all D tested.
- G2 (ablation parity) passes — the delegate sigmoid is provably separable from the speak/silent decision.
- `decide()` latency < 50ns for D=8 (within 4× of `evolve_hla`'s 14ns; the gap is the second dot-product).
- `decide_batch()` throughput ≥ 50M/sec for D=8, N=1000.

---

## Phase 3 — Async Delegate Helpers (open, runtime-agnostic)

### Tasks

- [ ] **T3.1** Add `SalienceTriGate::build_delegate_token(&self, payload: A, tick: u64, holding_reply_idx: u8, foldback_target: FoldbackTarget) -> DelegateToken<A>` — convenience constructor. Validates `holding_reply_idx` is in range (caller's table size is caller's concern; we just store the index).
- [ ] **T3.2** Add a `PendingDelegateQueue<A: Clone, const CAP: usize = 2>` ring buffer in `katgpt-rs/src/salience/pending.rs`:
  ```rust
  pub struct PendingDelegateQueue<A: Clone, const CAP: usize = 2> {
      slots: [Option<DelegateToken<A>>; CAP],
      head: u8,
      len: u8,
  }
  ```
  Methods: `push(token) -> Result<(), DelegateToken<A>>` (Err if full — caller decides policy), `pop_completed()`, `is_empty()`, `len()`. Fixed-size, zero-alloc.
- [ ] **T3.3** Document the contract: this crate does **not** spawn async tasks. The caller (riir-ai runtime) owns the spawn. This crate only provides the typed handoff + queue.
- [ ] **T3.4** Add doc example showing the typical caller pattern (build token → push to queue → caller spawns async → on completion, caller removes from queue and applies foldback).

### Phase 3 acceptance

- Queue property tests: push/pop FIFO order; push when full returns Err with the token; CAP=2 holds exactly 2.
- No async runtime dependency in this crate.

---

## Phase 4 — Documentation + Examples

### Tasks

- [ ] **T4.1** Add `katgpt-rs/examples/salience_tri_gate_basic.rs` — minimal example: construct gate with hand-tuned direction vectors, run 100 random activations, print decision distribution. No game semantics.
- [ ] **T4.2** Add `katgpt-rs/examples/salience_tri_gate_batch.rs` — batched usage with N=10000, print throughput.
- [ ] **T4.3** Add module-level doc with the paper citation, the open/private split, and a pointer to `riir-ai/.research/148` (just the path, not the contents — private).
- [ ] **T4.4** Add `katgpt-rs/.docs/30_salience_tri_gate.md` (or next free number) documenting the API surface, design rationale (two sigmoids vs softmax, silence-as-variant), and the GOAT gate results once Phase 2 completes.

### Phase 4 acceptance

- Both examples compile and run with `--features salience_tri_gate`.
- Module doc renders cleanly via `cargo doc`.

---

## Phase 5 — GOAT Gate Run + Promotion Decision

### Tasks

- [ ] **T5.1** Run `cargo test --features salience_tri_gate` — all G1/G2 property tests pass.
- [ ] **T5.2** Run `cargo bench --features salience_tri_gate salience_tri_gate_bench` — capture latency + throughput numbers.
- [ ] **T5.3** Fill in actual numbers in the module doc.
- [ ] **T5.4** **GOAT promotion decision:**
  - If G1+G2 PASS and latency < 50ns → promote `salience_tri_gate` to **default feature** in `katgpt-rs/Cargo.toml`.
  - If G1+G2 PASS but latency ≥ 50ns → keep opt-in, file issue for SIMD optimization (cf. `bridge/mod.rs` i8→f32 lesson).
  - If G1 or G2 FAIL → do not promote; fix root cause before re-running.

### Phase 5 acceptance

- All gates pass with recorded numbers.
- Promotion decision is recorded in the plan with a date.
- If promoted: the feature appears in the default feature list in `Cargo.toml` and the README's "Always-On Hot Path" section (cf. README L122).

---

## Out of scope (explicitly)

- **NPC wiring** → riir-ai Plan 330. This crate is game-agnostic.
- **HLA binding** → riir-ai Plan 330. The activation `a` is generic in this crate.
- **R133 mind-reading `ca` scalar computation** → riir-ai Plan 311.
- **cgsp curiosity scalar computation** → riir-ai Plan 299 (curiosity runtime).
- **Async delegate backend implementations** (AnyRAG gateway, Engram, Cold-tier) → riir-neuron-db (gateway.rs) + riir-ai Plan 330 routing layer.
- **Training recipe** (GRPO + role-weighted SFT, the `w_first_silence=1.0`, `w_repeated_silence=0.4`, `w_response=1.5` role-token weights) → riir-train.
- **AdaCodec streaming visual codec** (paper §3.1) → orthogonal; separate paper (2606.02569); would be its own plan if pursued.
- **Long-horizon three-tier memory** (paper §4.3) → already covered by Plan 312 (Dual-Pool CGSP) + research 007 (Four-Tier Memory).

---

## TL;DR

Open primitive plan for the Super-GOAT declared in `katgpt-rs/.research/281`. Ships `SalienceTriGate<A, D>` in `katgpt-rs/src/salience/` behind feature `salience_tri_gate` — a 3-way per-tick emit gate (Speak / Silent / Delegate) with silence as a first-class variant, two stacked sigmoids (never softmax), BLAKE3-committed direction vectors (caller's responsibility), and a typed `DelegateToken` handoff with a fixed-size `PendingDelegateQueue`. **Phase 1 = skeleton + types + decide/decide_batch**; **Phase 2 = G1 (determinism + monotonicity) + G2 (two-sigmoid ablation parity) + latency bench (< 50ns for D=8)**; **Phase 3 = delegate token + pending queue helpers**; **Phase 4 = examples + docs**; **Phase 5 = GOAT gate run + promotion decision**. Game-side wiring is riir-ai Plan 330; training is riir-train. This crate stays math-only, MIT, no game IP.
