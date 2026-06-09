//! Hierarchical GOAT Proof Certificates (Plan 145).
//!
//! Standalone, serializable proof certificates with dependency chains,
//! topological verification, and blake3 checksum integrity.

mod certificate;
mod chain;
mod macros;
mod serde_impls;
mod wasm_certificates;

#[cfg(feature = "wasm_proof_witness")]
mod wasm_proof_witness;

pub use certificate::{ProofCertificate, ProofEvidence, ProofProperty, ProofResult};
pub use chain::{ProofChainResult, verify_proof_chain};
pub use serde_impls::{load_certificates, save_certificates, verify_checksum};
pub use wasm_certificates::generate_wasm_validator_certificates;

#[cfg(feature = "wasm_proof_witness")]
pub use wasm_proof_witness::{WasmProofWitness, generate_wasm_witness_certificates};
