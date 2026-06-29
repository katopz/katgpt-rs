//! Cross-arena tournament infrastructure — scheduling, scoring, leaderboards.
//!
//! Shared types for running RubricPlayer tournaments across Bomber and FFT domains.

pub mod scheduler;
pub mod types;

pub use scheduler::*;
pub use types::*;

#[cfg(feature = "tes_loop")]
pub use types::TrajectoryPruner;
