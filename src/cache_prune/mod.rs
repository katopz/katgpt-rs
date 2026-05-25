//! CachePrune — SAT + rolling hash + sensitivity masking for KV cache analysis.
//!
//! Extracts reusable algorithmic primitives from the CachePrune paper:
//! (1) Summed-Area Table for O(1) rectangular attention queries,
//! (2) Rolling hash for O(n) variable-length segment matching,
//! (3) Generic `SensitivityDetector` trait for selective KV sharing.
//!
//! All modelless — no training, no model changes. Pure algorithmic infrastructure.
//!
//! **Feature gate:** `cache_prune` (Plan 140, Research 101, opt-in)
//!
//! Reference: arXiv:2605.23640

pub mod rolling_hash;
pub mod sat;
pub mod sensitivity;

pub use rolling_hash::{CachedSegment, KvSegmentPool, MatchResult, RollingHash};
pub use sat::SummedAreaTable;
pub use sensitivity::{MaskedSegment, OpenDetector, SensitivityDetector, StrictDetector};
