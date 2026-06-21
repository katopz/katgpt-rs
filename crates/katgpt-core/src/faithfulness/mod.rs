//! FaithfulnessProbe — Causal Intervention Diagnostic for Injected Memory (Modelless)
//!
//! Plan 278, Research 244. Compiled when either `faithfulness_probe` or
//! `triggered_injection` feature is on.
//!
//! - **`triggered_injection`** (default-ON after GOAT G3): enables [`gate`] —
//!   `TriggeredInjectionGate` + `EntropyThresholdGate` + `UncertaintySignal`.
//!   Hot-path inject/skip decision, <1ns/call.
//! - **`faithfulness_probe`** (opt-in, diagnostic): additionally enables [`probe`],
//!   [`perturb`], [`attribution`], and the full intervention suite. Runs at
//!   audit cadence (every N ticks), not per-tick.
//! - [`types`] (`Intervention`, `FaithfulnessProfile`, `ConsumerContext`,
//!   `MemorySlice`) is always available when the module is compiled.
//!
//! All modelless: zero training, zero backprop through base weights. Zero-allocation
//! on hot paths (`EntropyThresholdGate::should_inject` <10ns).
//!
//! Based on Zhao et al. 2026 (arxiv 2601.22436), "Large Language Model Agents Are
//! Not Always Faithful Self-Evolvers".

// `types` and `gate` are always available when the module is compiled
// (either feature on). They have no dependencies on the heavier submodules.
pub mod gate;
pub mod types;

// The diagnostic suite (probe + perturbation + attribution + GOAT gate tests)
// is opt-in via `faithfulness_probe`. It runs at audit cadence, not per-tick.
#[cfg(feature = "faithfulness_probe")]
pub mod attribution;
#[cfg(all(test, feature = "faithfulness_probe"))]
pub mod goat_gate;
#[cfg(feature = "faithfulness_probe")]
pub mod perturb;
#[cfg(feature = "faithfulness_probe")]
pub mod probe;

// SmearClassifier — ternary smear classification (Plan 298, Research 277).
// Extends the binary FaithfulnessProbe with a vocabulary for how latent
// mass is distributed (CoherentSingle / TokenSmear / SequenceSmear).
// Opt-in: depends on `faithfulness_probe` for the integration surface, but
// the classifier itself is standalone (zero deps on probe internals).
#[cfg(feature = "smear_classifier")]
pub mod smear;

// Convenience re-exports. Heavy items are only re-exported when their
// feature is on; `gate::*` and `types::*` are always available.
#[cfg(feature = "faithfulness_probe")]
pub use attribution::{AttributionProbe, FiniteDifferenceAttributionProbe};
pub use gate::{EntropyThresholdGate, TriggeredInjectionGate, UncertaintySignal};
#[cfg(feature = "faithfulness_probe")]
pub use probe::{DefaultFaithfulnessProbe, FaithfulnessProbe};
pub use types::{ConsumerContext, FaithfulnessProfile, Intervention, MemorySlice};

// SmearClassifier re-exports — Plan 298. The classifier itself is opt-in
// via `smear_classifier`; Phase 2 wires it into DefaultFaithfulnessProbe as
// an optional diagnostic that emits a SmearReport alongside the binary verdict.
// `SmearSource`, `InterventionOutcome`, and `FaithfulnessProfileFull` are the
// Phase 2 integration surface — only available when `smear_classifier` is on.
#[cfg(feature = "smear_classifier")]
pub use probe::{FaithfulnessProfileFull, InterventionOutcome, SmearSource};
#[cfg(feature = "smear_classifier")]
pub use smear::{CosineSmearClassifier, SmearClass, SmearClassifier, SmearReport};
