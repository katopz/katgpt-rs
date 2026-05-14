//! FFT Tactics Arena — ATB battle engine with status effects and AI players.
//!
//! Final Fantasy Tactics-inspired headless battle arena.
//! Units act independently via Active Time Battle (ATB) when their CT gauge fills.
//! Supports 6 classes, 9 action types, 9 status effects.
//!
//! # Feature Gate
//!
//! All code behind `#[cfg(feature = "fft")]`.
//! Feature: `fft = ["bandit"]` in `Cargo.toml`.

pub mod battle;
pub mod players;
pub mod status;
pub mod types;

pub use battle::*;
pub use players::*;
pub use status::*;
pub use types::*;

#[cfg(feature = "g_zero")]
pub mod g_zero_player;

#[cfg(feature = "g_zero")]
pub use g_zero_player::GZeroFFTPlayer;

#[cfg(feature = "g_zero")]
pub mod tft_player;

#[cfg(feature = "g_zero")]
pub use tft_player::TftFFTPlayer;
