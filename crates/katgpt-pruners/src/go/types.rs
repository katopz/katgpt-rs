//! Go domain types — actions, cells, and color utilities.
//!
//! Ported from `alpha_go/go.py` constants (`EMPTY=0, BLACK=1, WHITE=2`)
//! and `alpha_go/cpp/go/go_game.h` (`GoBoard::EMPTY/BLACK/WHITE`).

use std::fmt;

// ── GoAction ───────────────────────────────────────────────────

/// A move in Go: place a stone or pass.
///
/// Ported from `go.py:GoState` action representation (`(row, col)` tuple or `None`).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum GoAction {
    /// Place a stone at (row, col).
    Place(usize, usize),
    /// Pass turn. Two consecutive passes end the game.
    Pass,
}

impl fmt::Display for GoAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Place(row, col) => write!(f, "({row},{col})"),
            Self::Pass => write!(f, "Pass"),
        }
    }
}

// ── GoCell ─────────────────────────────────────────────────────

/// Board cell state. Values match both Python (`EMPTY=0, BLACK=1, WHITE=2`)
/// and C++ (`GoBoard::EMPTY/BLACK/WHITE`) representations for API compatibility.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(i8)]
pub enum GoCell {
    /// Empty intersection.
    #[default]
    Empty = 0,
    /// Black stone.
    Black = 1,
    /// White stone.
    White = 2,
}

impl GoCell {
    /// Opposite color. Panics on [`Empty`](GoCell::Empty).
    ///
    /// Ported from `go.py:opponent_of()`.
    #[inline]
    #[must_use]
    pub fn opponent(self) -> Self {
        match self {
            Self::Black => Self::White,
            Self::White => Self::Black,
            Self::Empty => panic!("GoCell::opponent() called on Empty"),
        }
    }

    /// Returns `true` for `Black` or `White` (not `Empty`).
    #[inline]
    #[must_use]
    pub fn is_stone(self) -> bool {
        matches!(self, Self::Black | Self::White)
    }

    /// Convert from `i8` value (0/1/2) to `GoCell`.
    ///
    /// Returns `None` for invalid values.
    #[inline]
    #[must_use]
    pub fn from_i8(v: i8) -> Option<Self> {
        match v {
            0 => Some(Self::Empty),
            1 => Some(Self::Black),
            2 => Some(Self::White),
            _ => None,
        }
    }

    /// Player index for MCTS: Black=0, White=1.
    ///
    /// Panics on `Empty`.
    #[inline]
    #[must_use]
    pub fn player_id(self) -> u8 {
        match self {
            Self::Black => 0,
            Self::White => 1,
            Self::Empty => panic!("GoCell::player_id() called on Empty"),
        }
    }

    /// Convert player_id (0/1) back to GoCell.
    #[inline]
    #[must_use]
    pub fn from_player_id(id: u8) -> Self {
        match id {
            0 => Self::Black,
            1 => Self::White,
            _ => panic!("invalid player_id: {id}"),
        }
    }
}

impl fmt::Display for GoCell {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => write!(f, "·"),
            Self::Black => write!(f, "X"),
            Self::White => write!(f, "O"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn go_action_display() {
        assert_eq!(format!("{}", GoAction::Place(3, 4)), "(3,4)");
        assert_eq!(format!("{}", GoAction::Pass), "Pass");
    }

    #[test]
    fn go_cell_opponent_roundtrip() {
        assert_eq!(GoCell::Black.opponent(), GoCell::White);
        assert_eq!(GoCell::White.opponent(), GoCell::Black);
        assert_eq!(GoCell::Black.opponent().opponent(), GoCell::Black);
    }

    #[test]
    #[should_panic(expected = "GoCell::opponent() called on Empty")]
    fn go_cell_opponent_empty_panics() {
        let _ = GoCell::Empty.opponent();
    }

    #[test]
    fn go_cell_from_i8() {
        assert_eq!(GoCell::from_i8(0), Some(GoCell::Empty));
        assert_eq!(GoCell::from_i8(1), Some(GoCell::Black));
        assert_eq!(GoCell::from_i8(2), Some(GoCell::White));
        assert_eq!(GoCell::from_i8(3), None);
        assert_eq!(GoCell::from_i8(-1), None);
    }

    #[test]
    fn go_cell_repr_values() {
        assert_eq!(GoCell::Empty as i8, 0);
        assert_eq!(GoCell::Black as i8, 1);
        assert_eq!(GoCell::White as i8, 2);
    }

    #[test]
    fn go_cell_is_stone() {
        assert!(!GoCell::Empty.is_stone());
        assert!(GoCell::Black.is_stone());
        assert!(GoCell::White.is_stone());
    }

    #[test]
    fn go_cell_player_id_roundtrip() {
        assert_eq!(GoCell::Black.player_id(), 0);
        assert_eq!(GoCell::White.player_id(), 1);
        assert_eq!(GoCell::from_player_id(0), GoCell::Black);
        assert_eq!(GoCell::from_player_id(1), GoCell::White);
    }

    #[test]
    fn go_cell_display() {
        let empty = GoCell::Empty;
        let black = GoCell::Black;
        let white = GoCell::White;
        assert_eq!(format!("{empty}"), "·");
        assert_eq!(format!("{black}"), "X");
        assert_eq!(format!("{white}"), "O");
    }

    #[test]
    fn go_cell_default_is_empty() {
        assert_eq!(GoCell::default(), GoCell::Empty);
    }
}

// ── Frozen Knowledge (Plan 092) ────────────────────────────────

/// Frozen Go bandit state for disk persistence (category-level learning).
///
/// Captures Q-values and visits for the 8 GoMoveCategory arms.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GoFrozenBandit {
    /// Magic bytes: b"GODT" (GO DaTa)
    pub magic: [u8; 4],
    /// Format version: 1
    pub version: u32,
    /// Q-value estimates per move category (8 categories)
    pub q_values: [f32; 8],
    /// Visit counts per category
    pub visits: [u32; 8],
    /// Total bandit pulls
    pub total_pulls: u32,
    /// Current exploration rate ε
    pub epsilon: f32,
    /// Reserved for future use
    pub reserved: [u8; 12],
}

impl GoFrozenBandit {
    /// Magic bytes for Go bandit format.
    pub const MAGIC: [u8; 4] = *b"GODT";
    /// Current format version.
    pub const VERSION: u32 = 1;

    /// Create a new empty frozen bandit.
    pub fn new_empty() -> Self {
        Self {
            magic: Self::MAGIC,
            version: Self::VERSION,
            q_values: [0.0; 8],
            visits: [0; 8],
            total_pulls: 0,
            epsilon: 0.15,
            reserved: [0; 12],
        }
    }

    /// Validate magic bytes and version.
    pub fn validate(&self) -> Result<(), String> {
        if self.magic != Self::MAGIC {
            return Err(format!(
                "Invalid magic: expected {:?}, got {:?}",
                Self::MAGIC,
                self.magic
            ));
        }
        if self.version != Self::VERSION {
            return Err(format!(
                "Unsupported version: expected {}, got {}",
                Self::VERSION,
                self.version
            ));
        }
        Ok(())
    }
}

/// Frozen Go template state for disk persistence (G-Zero template learning).
///
/// Captures Q-values and visits for the 4 GoTemplate arms.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GoFrozenTemplates {
    /// Magic bytes: b"GOTM" (GO TeMplates)
    pub magic: [u8; 4],
    /// Format version: 1
    pub version: u32,
    /// Q-value estimates per template (4 templates)
    pub q_values: [f32; 4],
    /// Visit counts per template
    pub visits: [u32; 4],
    /// Total bandit pulls
    pub total_pulls: u32,
    /// Reserved for future use
    pub reserved: [u8; 16],
}

impl GoFrozenTemplates {
    /// Magic bytes for Go template format.
    pub const MAGIC: [u8; 4] = *b"GOTM";
    /// Current format version.
    pub const VERSION: u32 = 1;

    /// Create a new empty frozen template.
    pub fn new_empty() -> Self {
        Self {
            magic: Self::MAGIC,
            version: Self::VERSION,
            q_values: [0.0; 4],
            visits: [0; 4],
            total_pulls: 0,
            reserved: [0; 16],
        }
    }

    /// Validate magic bytes and version.
    pub fn validate(&self) -> Result<(), String> {
        if self.magic != Self::MAGIC {
            return Err(format!(
                "Invalid magic: expected {:?}, got {:?}",
                Self::MAGIC,
                self.magic
            ));
        }
        if self.version != Self::VERSION {
            return Err(format!(
                "Unsupported version: expected {}, got {}",
                Self::VERSION,
                self.version
            ));
        }
        Ok(())
    }
}
