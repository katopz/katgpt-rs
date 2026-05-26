//! Hierarchical GOAT Proof Certificates (Plan 145).
//!
//! Standalone, serializable proof certificates with dependency chains,
//! topological verification, and blake3 checksum integrity.

mod certificate;
mod chain;
mod macros;
mod serde_impls;
mod wasm_certificates;

pub use certificate::{ProofCertificate, ProofEvidence, ProofProperty, ProofResult};
pub use chain::{verify_proof_chain, ProofChainResult};
pub use serde_impls::{load_certificates, save_certificates, verify_checksum};
pub use wasm_certificates::generate_wasm_validator_certificates;
