//! Go game replay — recording and playback for analysis and validation.
//!
//! Plan 065 (G6): Records every move in a game for:
//! - Post-game analysis (branching factor, move quality)
//! - Cross-validation against AutoGo API (same game → same legal moves)
//! - Deterministic replay from empty board to final position

use serde::{Deserialize, Serialize};

use super::state::GoState;
use super::types::{GoAction, GoCell};

/// Single move record for replay.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MoveRecord {
    /// The action taken.
    pub action: GoActionSer,
    /// Which player made this move.
    pub player: GoCellSer,
    /// Move number (1-based).
    pub move_number: u32,
    /// Branching factor at this point (legal move count).
    pub legal_move_count: usize,
}

/// Serializable wrapper for [`GoAction`].
///
/// Needed because Rust enums with tuple variants don't derive `Serialize` cleanly
/// when used across feature gates. Provides human-readable JSON.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum GoActionSer {
    Place { row: usize, col: usize },
    Pass,
}

impl From<&GoAction> for GoActionSer {
    fn from(a: &GoAction) -> Self {
        match a {
            GoAction::Place(r, c) => Self::Place { row: *r, col: *c },
            GoAction::Pass => Self::Pass,
        }
    }
}

impl From<GoActionSer> for GoAction {
    fn from(a: GoActionSer) -> Self {
        match a {
            GoActionSer::Place { row, col } => Self::Place(row, col),
            GoActionSer::Pass => Self::Pass,
        }
    }
}

/// Serializable wrapper for [`GoCell`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GoCellSer {
    Black,
    White,
}

impl From<GoCell> for GoCellSer {
    fn from(c: GoCell) -> Self {
        match c {
            GoCell::Black => Self::Black,
            GoCell::White => Self::White,
            GoCell::Empty => panic!("Cannot serialize GoCell::Empty as player"),
        }
    }
}

impl From<GoCellSer> for GoCell {
    fn from(c: GoCellSer) -> Self {
        match c {
            GoCellSer::Black => Self::Black,
            GoCellSer::White => Self::White,
        }
    }
}

/// Complete game replay.
///
/// Records all moves from an empty board to the final position.
/// Supports serialization to JSON for storage and cross-validation.
///
/// ## Example
///
/// ```ignore
/// use katgpt_rs::pruners::go::replay::GoReplay;
/// use katgpt_rs::pruners::go::state::GoState;
/// use katgpt_rs::pruners::go::types::{GoAction, GoCell};
///
/// let mut replay = GoReplay::new(9, 7.5);
/// let mut state = GoState::new(9);
///
/// let legal_count = state.legal_move_count();
/// state.play_move(4, 4);
/// replay.record(&GoAction::Place(4, 4), GoCell::Black, legal_count);
///
/// let legal_count = state.legal_move_count();
/// state.play_pass();
/// replay.record(&GoAction::Pass, GoCell::White, legal_count);
///
/// replay.finalize(Some(GoCell::Black), state.score());
/// let json = serde_json::to_string(&replay).unwrap();
/// ```
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GoReplay {
    /// Board size.
    pub size: usize,
    /// Komi value.
    pub komi: f32,
    /// All moves in order.
    pub moves: Vec<MoveRecord>,
    /// Winner (None for draw or incomplete).
    pub winner: Option<GoCellSer>,
    /// Final score (Black perspective).
    pub final_score: f32,
}

/// Error from replay validation.
#[derive(Debug)]
pub enum ReplayError {
    /// A move in the replay was illegal.
    IllegalMove {
        move_number: u32,
        action: GoActionSer,
        reason: String,
    },
    /// The replay has no winner/finalize but was replayed to completion.
    NotFinalized,
}

impl std::fmt::Display for ReplayError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IllegalMove {
                move_number,
                action,
                reason,
            } => write!(f, "Illegal move #{move_number} {action:?}: {reason}"),
            Self::NotFinalized => write!(f, "Replay not finalized (no winner/score)"),
        }
    }
}

impl std::error::Error for ReplayError {}

impl GoReplay {
    /// Create a new empty replay for a game with given size and komi.
    pub fn new(size: usize, komi: f32) -> Self {
        Self {
            size,
            komi,
            moves: Vec::new(),
            winner: None,
            final_score: 0.0,
        }
    }

    /// Record a move.
    ///
    /// Call this after the move has been applied to the board state.
    /// `legal_count` is the number of legal moves BEFORE the move was played.
    pub fn record(&mut self, action: &GoAction, player: GoCell, legal_count: usize) {
        let move_number = (self.moves.len() + 1) as u32;
        self.moves.push(MoveRecord {
            action: GoActionSer::from(action),
            player: GoCellSer::from(player),
            move_number,
            legal_move_count: legal_count,
        });
    }

    /// Finalize the replay with winner and score.
    pub fn finalize(&mut self, winner: Option<GoCell>, score: f32) {
        self.winner = winner.map(GoCellSer::from);
        self.final_score = score;
    }

    /// Total moves recorded.
    pub fn len(&self) -> usize {
        self.moves.len()
    }

    /// Is the replay empty?
    pub fn is_empty(&self) -> bool {
        self.moves.is_empty()
    }

    /// Average branching factor across all moves.
    pub fn avg_branching_factor(&self) -> f32 {
        if self.moves.is_empty() {
            return 0.0;
        }
        self.moves
            .iter()
            .map(|m| m.legal_move_count as f32)
            .sum::<f32>()
            / self.moves.len() as f32
    }

    /// Peak branching factor.
    pub fn peak_branching_factor(&self) -> usize {
        self.moves
            .iter()
            .map(|m| m.legal_move_count)
            .max()
            .unwrap_or(0)
    }

    /// Replay all moves from an empty board, validating every move is legal.
    ///
    /// Returns the final [`GoState`] if all moves are valid.
    /// Returns [`ReplayError`] if any move is illegal.
    pub fn replay(&self) -> Result<GoState, ReplayError> {
        let mut state = GoState::with_komi(self.size, self.komi);

        for record in &self.moves {
            let player: GoCell = record.player.into();
            let action: GoAction = record.action.clone().into();

            // Verify it's the correct player's turn
            if state.to_play != player {
                return Err(ReplayError::IllegalMove {
                    move_number: record.move_number,
                    action: record.action.clone(),
                    reason: format!(
                        "Expected {:?} to play, but it's {:?}'s turn",
                        player, state.to_play
                    ),
                });
            }

            match &action {
                GoAction::Place(row, col) => {
                    if !state.play_move(*row, *col) {
                        return Err(ReplayError::IllegalMove {
                            move_number: record.move_number,
                            action: record.action.clone(),
                            reason: format!("play_move({row},{col}) returned false"),
                        });
                    }
                }
                GoAction::Pass => {
                    state.play_pass();
                }
            }
        }

        Ok(state)
    }

    /// Serialize to JSON string.
    ///
    /// Deprecated: use [`GoReplay::to_bytes`] for zero-copy binary (postcard).
    /// JSON remains only because the riir-ai training pipeline still consumes JSONL;
    /// it will be removed once that pipeline migrates to binary (Issue 011).
    #[deprecated(
        note = "use to_bytes/from_bytes; JSON removed when riir-ai training pipeline migrates"
    )]
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    /// Serialize to pretty JSON string.
    ///
    /// Deprecated alongside [`GoReplay::to_json`]; see that method for rationale.
    #[deprecated(
        note = "use to_bytes/from_bytes; JSON removed when riir-ai training pipeline migrates"
    )]
    pub fn to_json_pretty(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Deserialize from JSON string.
    ///
    /// Deprecated: use [`GoReplay::from_bytes`] for binary (postcard).
    #[deprecated(
        note = "use to_bytes/from_bytes; JSON removed when riir-ai training pipeline migrates"
    )]
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Serialize to binary (postcard). Zero-copy friendly.
    ///
    /// Matches the pattern established in
    /// `src/pruners/bomber/replay.rs` (Issue 011). Prefer this over `to_json`.
    ///
    /// Routes through [`GoReplayBin`] because postcard cannot serialize the
    /// internally-tagged enums (`#[serde(tag = ...)]`) used for the JSON shape;
    /// the binary shadow uses externally-tagged enums (postcard's native form).
    pub fn to_bytes(&self) -> Vec<u8> {
        let bin = GoReplayBin::from(self);
        postcard::to_allocvec(&bin).unwrap_or_default()
    }

    /// Deserialize from binary (postcard).
    pub fn from_bytes(data: &[u8]) -> Result<Self, postcard::Error> {
        let bin: GoReplayBin = postcard::from_bytes(data)?;
        Ok(Self::from(bin))
    }
}

// ── Binary (postcard) shadow types ─────────────────────────────
//
// `GoReplay` and its enums use `#[serde(tag = "type", ...)]` /
// `#[serde(rename_all = "lowercase")]` for human-readable JSON consumed by the
// riir-ai training pipeline. Postcard returns `WontImplement` for
// internally-tagged enums, so the binary path routes through these
// externally-tagged (postcard-native) shadows. Conversions are exact and
// allocation-free beyond the returned `Vec`.

/// Postcard-native action variant (externally tagged — postcard's default).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
enum GoActionBin {
    Place { row: usize, col: usize },
    Pass,
}

impl From<&GoActionSer> for GoActionBin {
    fn from(a: &GoActionSer) -> Self {
        match a {
            GoActionSer::Place { row, col } => Self::Place { row: *row, col: *col },
            GoActionSer::Pass => Self::Pass,
        }
    }
}

impl From<GoActionBin> for GoActionSer {
    fn from(a: GoActionBin) -> Self {
        match a {
            GoActionBin::Place { row, col } => Self::Place { row, col },
            GoActionBin::Pass => Self::Pass,
        }
    }
}

/// Postcard-native player cell variant (externally tagged).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
enum GoCellBin {
    Black,
    White,
}

impl From<GoCellSer> for GoCellBin {
    fn from(c: GoCellSer) -> Self {
        match c {
            GoCellSer::Black => Self::Black,
            GoCellSer::White => Self::White,
        }
    }
}

impl From<GoCellBin> for GoCellSer {
    fn from(c: GoCellBin) -> Self {
        match c {
            GoCellBin::Black => Self::Black,
            GoCellBin::White => Self::White,
        }
    }
}

/// Postcard-native move record.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct MoveRecordBin {
    action: GoActionBin,
    player: GoCellBin,
    move_number: u32,
    legal_move_count: usize,
}

impl From<&MoveRecord> for MoveRecordBin {
    fn from(m: &MoveRecord) -> Self {
        Self {
            action: GoActionBin::from(&m.action),
            player: GoCellBin::from(m.player),
            move_number: m.move_number,
            legal_move_count: m.legal_move_count,
        }
    }
}

impl From<MoveRecordBin> for MoveRecord {
    fn from(m: MoveRecordBin) -> Self {
        Self {
            action: GoActionSer::from(m.action),
            player: GoCellSer::from(m.player),
            move_number: m.move_number,
            legal_move_count: m.legal_move_count,
        }
    }
}

/// Postcard-native replay envelope.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct GoReplayBin {
    size: usize,
    komi: f32,
    moves: Vec<MoveRecordBin>,
    winner: Option<GoCellBin>,
    final_score: f32,
}

impl From<&GoReplay> for GoReplayBin {
    fn from(r: &GoReplay) -> Self {
        Self {
            size: r.size,
            komi: r.komi,
            moves: r.moves.iter().map(MoveRecordBin::from).collect(),
            winner: r.winner.map(GoCellBin::from),
            final_score: r.final_score,
        }
    }
}

impl From<GoReplayBin> for GoReplay {
    fn from(r: GoReplayBin) -> Self {
        Self {
            size: r.size,
            komi: r.komi,
            moves: r.moves.into_iter().map(MoveRecord::from).collect(),
            winner: r.winner.map(GoCellSer::from),
            final_score: r.final_score,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_and_finalize() {
        let mut replay = GoReplay::new(9, 7.5);
        let mut state = GoState::new(9);

        let legal = state.legal_move_count();
        state.play_move(4, 4);
        replay.record(&GoAction::Place(4, 4), GoCell::Black, legal);

        let legal = state.legal_move_count();
        state.play_pass();
        replay.record(&GoAction::Pass, GoCell::White, legal);

        let legal = state.legal_move_count();
        state.play_pass();
        replay.record(&GoAction::Pass, GoCell::Black, legal);

        replay.finalize(Some(GoCell::Black), 7.5);

        assert_eq!(replay.len(), 3);
        assert_eq!(replay.winner, Some(GoCellSer::Black));
        assert!((replay.final_score - 7.5).abs() < 0.01);
    }

    #[test]
    fn replay_validates_moves() {
        let mut replay = GoReplay::new(9, 7.5);
        let mut state = GoState::new(9);

        let legal = state.legal_move_count();
        state.play_move(4, 4);
        replay.record(&GoAction::Place(4, 4), GoCell::Black, legal);

        let result = replay.replay();
        assert!(result.is_ok());
        let final_state = result.unwrap();
        assert_eq!(final_state.at(4, 4), GoCell::Black);
    }

    #[test]
    fn replay_detects_wrong_turn() {
        let mut replay = GoReplay::new(9, 7.5);
        // White plays first — should be Black
        replay.record(&GoAction::Place(4, 4), GoCell::White, 81);

        let result = replay.replay();
        assert!(result.is_err());
        match result {
            Err(ReplayError::IllegalMove { reason, .. }) => {
                assert!(
                    reason.contains("Black"),
                    "Error should mention Black: {reason}"
                );
            }
            _ => panic!("Expected IllegalMove error"),
        }
    }

    #[test]
    fn replay_detects_illegal_placement() {
        let mut replay = GoReplay::new(9, 7.5);
        // Play same position twice
        replay.record(&GoAction::Place(4, 4), GoCell::Black, 81);
        replay.record(&GoAction::Place(4, 4), GoCell::White, 80); // Already occupied

        let result = replay.replay();
        assert!(
            result.is_err(),
            "Should fail on second move at occupied position"
        );
    }

    #[test]
    #[allow(deprecated)] // JSON path kept until riir-ai migrates (Issue 011)
    fn json_roundtrip() {
        let mut replay = GoReplay::new(9, 7.5);
        let mut state = GoState::new(9);

        for _ in 0..5 {
            let legal = state.legal_move_count();
            let moves = state.legal_moves();
            if moves.is_empty() {
                state.play_pass();
                replay.record(&GoAction::Pass, state.to_play.opponent(), legal);
            } else {
                let (r, c) = moves[0];
                state.play_move(r, c);
                replay.record(&GoAction::Place(r, c), state.to_play.opponent(), legal);
            }
        }

        replay.finalize(None, 0.0);

        let json = replay.to_json().unwrap();
        let restored = GoReplay::from_json(&json).unwrap();

        assert_eq!(restored.size, 9);
        assert_eq!(restored.moves.len(), 5);
        assert_eq!(restored.komi, 7.5);
    }

    #[test]
    fn binary_roundtrip_matches_json_structure() {
        // Build a non-trivial replay so every field has a distinct value.
        let mut replay = GoReplay::new(9, 7.5);
        let mut state = GoState::new(9);

        for _ in 0..5 {
            let legal = state.legal_move_count();
            let moves = state.legal_moves();
            if moves.is_empty() {
                state.play_pass();
                replay.record(&GoAction::Pass, state.to_play.opponent(), legal);
            } else {
                let (r, c) = moves[0];
                state.play_move(r, c);
                replay.record(&GoAction::Place(r, c), state.to_play.opponent(), legal);
            }
        }
        replay.finalize(Some(GoCell::Black), 6.5);

        // Binary round-trip.
        let bytes = replay.to_bytes();
        let restored = GoReplay::from_bytes(&bytes).expect("postcard round-trip should succeed");

        assert_eq!(restored.size, replay.size);
        assert_eq!(restored.komi, replay.komi);
        assert_eq!(restored.final_score, replay.final_score);
        assert_eq!(restored.winner, replay.winner);
        assert_eq!(restored.moves.len(), replay.moves.len());
        for (a, b) in replay.moves.iter().zip(restored.moves.iter()) {
            assert_eq!(a.action, b.action);
            assert_eq!(a.player, b.player);
            assert_eq!(a.move_number, b.move_number);
            assert_eq!(a.legal_move_count, b.legal_move_count);
        }

        // Sanity: binary payload should be present and (typically) more compact
        // than the JSON equivalent — though compactness is not a hard guarantee.
        assert!(!bytes.is_empty(), "to_bytes must produce output");

        // Binary-restored replay must be re-playable to the same final state.
        let final_state = restored.replay().expect("restored replay must be legal");
        assert_eq!(final_state.size, replay.size);
    }

    #[test]
    fn branching_factor_stats() {
        let mut replay = GoReplay::new(9, 7.5);
        replay.moves.push(MoveRecord {
            action: GoActionSer::Place { row: 0, col: 0 },
            player: GoCellSer::Black,
            move_number: 1,
            legal_move_count: 81,
        });
        replay.moves.push(MoveRecord {
            action: GoActionSer::Pass,
            player: GoCellSer::White,
            move_number: 2,
            legal_move_count: 80,
        });

        assert_eq!(replay.peak_branching_factor(), 81);
        assert!((replay.avg_branching_factor() - 80.5).abs() < 0.01);
    }

    #[test]
    fn empty_replay() {
        let replay = GoReplay::new(9, 7.5);
        assert!(replay.is_empty());
        assert_eq!(replay.len(), 0);
        assert_eq!(replay.avg_branching_factor(), 0.0);
        assert_eq!(replay.peak_branching_factor(), 0);
    }
}
