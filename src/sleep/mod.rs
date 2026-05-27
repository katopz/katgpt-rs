//! Sleep Consolidation — Offline Recursive Memory Consolidation at Eviction.
//!
//! Plan 154: When KV cache fills, perform N offline recurrent passes to consolidate
//! context into GDN2 fast weights, then evict. Preserves single-pass wake-time
//! latency for real-time game constraints (20Hz frame sampling).
//!
//! # Key Insight
//!
//! Sleep moves LT2's wake-time looping to eviction-time consolidation. This is
//! the model-based analog of AutoDreamer (Plan 107), applied to GDN2 fast weights.
//!
//! # Architecture
//!
//! ```text
//! Existing LT2 Pipeline:
//!   Input → [SDPA → GDN2 → SDPA → GDN2 → ...]×T (wake-time loops) → Output
//!
//! With Sleep:
//!   Input → Context fills → [SDPA → GDN2 → ...]×N (sleep-time consolidation) → Evict KV → Continue
//!          ↑ Single-pass at wake time (T=1)                    ↑ N-pass at eviction boundary
//! ```
//!
//! # Feature Gate
//!
//! `sleep_consolidation` — depends on `lt2_looped`, `gdn2_attention`
//!
//! # Usage
//!
//! ```ignore
//! use katgpt_rs::sleep::{sleep, SleepConfig, EvictionStrategy};
//!
//! let sleep_config = SleepConfig::default();
//!
//! // In your generation loop, when cache fills:
//! if sleep_config.should_sleep(pos) {
//!     sleep(&mut ctx, &weights, &mut kv_cache, &mut gdn2_cache, &sleep_config, &config);
//! }
//! ```
//!
//! Run tests: `cargo test --features sleep_consolidation`

pub mod consolidation;
pub mod eviction;
pub mod types;

pub use consolidation::{consolidation_pass, sleep};
pub use types::{EvictionStrategy, SleepConfig};
