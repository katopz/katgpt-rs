//! Domain-specific constraint pruners — back-compat shim.
//!
//! Plan 005 (2026-06-29): the pruners module was extracted to the
//! `katgpt-pruners` workspace crate. Everything except `bomber` moved.
//! Bomber stays here because it depends on main-crate-only modules
//! (`crate::transformer`, `crate::inference_router`, `crate::trigger_gate`).
//!
//! Existing `use crate::pruners::*` import paths continue to work via the
//! blanket `pub use katgpt_pruners::*` re-export below — no caller-side
//! changes required. New code should prefer `katgpt_pruners::*` directly.

pub use katgpt_pruners::*;

#[cfg(feature = "bomber")]
pub mod bomber;

#[cfg(feature = "bomber")]
pub use bomber::{
    ArenaGrid, BomberAction, BomberPlayer, GridPos, ScoreBoard, TickCounter, run_tick,
    spawn_players,
};

// Re-export the bomber-specific game_state types that used to live behind
// `cfg(all(feature = "game_state", feature = "bomber"))` in this module.
// bomber_state.rs moved back into the bomber module (src/pruners/bomber/)
// because it is tightly coupled to the bomber types (ArenaGrid, Cell, ...).
#[cfg(all(feature = "game_state", feature = "bomber"))]
pub use bomber::bomber_state::{BombSnapshot, BomberHeuristic, BomberState, PlayerSnapshot};
#[cfg(all(feature = "game_state", feature = "bomber", feature = "bandit"))]
pub use bomber::bomber_state::BanditBomberHeuristic;
