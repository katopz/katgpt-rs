//! SpecAsPruner — compile NL specs into symbolic ConstraintPruner rules.
//!
//! Research 229: ProgramAsWeights compiles specs into neural LoRA adapters (~22MB).
//! We compile specs into symbolic bitmap rules (~1KB), achieving:
//! - **4400× smaller** — bitmaps vs LoRA weight matrices
//! - **O(1) per token** — bitmap lookup vs neural forward pass
//! - **Zero training** — no GPU, no data, no gradient computation
//! - **Exact verification** — can prove all outputs satisfy spec constraints
//!
//! Supported spec types:
//! - **Classification**: "Classify sentiment as positive or negative" → allowlist
//! - **Extraction**: "Extract email addresses" → character-class allowlist
//! - **Format repair**: "Fix malformed JSON" → structural token constraints
//! - **Intent routing**: "Route to: search, create, delete" → label allowlist
//!
//! ## Feature Gates
//!
//! - `spec_pruner` — core SpecAsPruner: types, compiler, pruner, screening, proof
//! - `spec_compile` — full suite (includes `spec_pruner`): marginals, DFA, chain, router

pub mod compiler;
pub mod proof;
pub mod pruner;
pub mod screening;
pub mod types;

#[cfg(feature = "spec_compile")]
pub mod marginals;

#[cfg(feature = "spec_compile")]
pub mod dfa;

#[cfg(feature = "spec_compile")]
pub mod chain;

#[cfg(feature = "spec_compile")]
pub mod router;

pub use compiler::SpecCompiler;
pub use proof::{SpecCommitment, SpecProof};
pub use types::*;

#[cfg(feature = "spec_compile")]
pub use marginals::{SpecMarginals, TokenBias, spec_to_marginals};

#[cfg(feature = "spec_compile")]
pub use dfa::*;

#[cfg(feature = "spec_compile")]
pub use chain::*;

#[cfg(feature = "spec_compile")]
pub use router::*;
