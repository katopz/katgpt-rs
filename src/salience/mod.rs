//! # Salience Tri-Gate Primitive — Per-Tick `Speak` / `Silent` / `Delegate` (Modelless)
//!
//! **Open primitive** for JoyAI-VL-Interaction per-tick emit (Plan 303,
//! Research 281). Game-side wiring lives in `riir-ai` Plan 330. This crate
//! stays **math-only, MIT, no game IP** — no NPC semantics, no async spawn,
//! no game-state coupling.
//!
//! ## Source paper
//! [arxiv 2606.14777](https://arxiv.org/abs/2606.14777) — JoyAI-VL-Interaction
//! (Yao et al., JD.com, Jun 2026). Distills the paper's per-second emit
//! decision into a generic 3-way gate.
//!
//! ## Design rules (enforced)
//! - **Two stacked sigmoids** — never softmax (AGENTS.md).
//! - **`Silent` is a first-class variant**, not a threshold-suppression default.
//! - **Generic** over activation dimension `D` (const generic) and delegate
//!   payload `A: Clone`.
//! - **Zero-allocation hot path** — no `Vec`, `Box`, or heap traffic in
//!   `decide` / `decide_batch`.
//! - **Deterministic** — bit-identical output for bit-identical input. No
//!   RNG, no thread-local state.
//! - **Modelless** — direction vectors are caller-supplied (BLAKE3-committed
//!   at freeze/thaw by the caller; this crate is agnostic).
//!
//! ## GOAT gates
//! - **G1 (determinism + monotonicity)** — same inputs ⇒ same output; one
//!   threshold crossing each on the salience sweep and the delegate-dot sweep.
//! - **G2 (two-sigmoid ablation parity)** — with `ceil_delegate = +∞`, the
//!   gate's Silent/Speak sequence is bit-identical to a speak/silent-only
//!   reference over 100 deterministic pseudo-random inputs.
//!
//! Both gates are implemented as `#[test]`s in [`gate::tests`].
//!
//! ## API surface
//! - [`SalienceTriGate`] — the gate itself.
//! - [`SalienceDecision`] — first-class decision enum (`Silent` / `Speak` /
//!   `Delegate(A)`).
//! - [`SilenceToken`] — emitted with `Silent` from Phase 3 onward.
//! - [`DelegateToken`] / [`FoldbackTarget`] — typed delegate handoff (caller
//!   spawns the async task in Phase 3+).
//! - [`PendingDelegateQueue`] — fixed-capacity ring buffer of pending
//!   `DelegateToken`s. Zero-allocation handoff between the per-tick decision
//!   and the caller-owned async spawn (Phase 3, Plan 303 T3.2). This crate
//!   does **not** spawn async tasks — see the contract note on
//!   [`PendingDelegateQueue`] (Plan 303 T3.3).
//! - [`SalienceTriGate::build_delegate_token`] — convenience constructor for
//!   `DelegateToken<A2>` (Phase 3, Plan 303 T3.1).
//!
//! ## Private boundary
//! - NPC wiring → `riir-ai` Plan 330.
//! - HLA / CGSP binding → `riir-ai` Plan 330 (activation `a` is generic here).
//! - Async delegate backends → `riir-neuron-db` + `riir-ai` routing.
//! - Training recipe (GRPO + role-weighted SFT) → `riir-train`.
//!
//! ## References
//! - Plan: `katgpt-rs/.plans/303_salience_tri_gate_primitive.md`
//! - Research: `katgpt-rs/.research/281_Per_Tick_Salience_Tri_Gate_Speak_Silent_Delegate.md`
//! - Private guide: `riir-ai/.research/148_Per_Tick_Emit_Salience_NPC_Guide.md`
//! - Runtime plan: `riir-ai/.plans/330_proactive_npc_salience_gate_runtime.md`

pub mod gate;
pub mod pending;
pub mod types;

pub use gate::SalienceTriGate;
pub use pending::PendingDelegateQueue;
pub use types::{DelegateToken, FoldbackTarget, SalienceDecision, SilenceToken};
