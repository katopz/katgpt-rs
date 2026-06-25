//! Go replay writer — outputs per-move samples as JSONL for riir-ai LoRA training.
//!
//! Plan 271 T2.1 (Phase 2): Bridges Go self-play games to riir-ai training pipeline.
//!
//! ## Architecture
//!
//! ```text
//! GoState → per-move snapshot → JsonlGoSample → JSONL line → file
//!                                            ↕
//!                            quality = sigmoid margin (winner = 1.0, loser = 0.0)
//! ```
//!
//! ## JSONL Format
//!
//! Each line is a JSON object with fields matching `riir-gpu`'s `GoGameSample`:
//! - `board`: flat `size*size` array of u8 (0=empty, 1=black, 2=white)
//! - `board_size`: board dimension
//! - `action`: `{ "type": "Place", "row": r, "col": c }` or `{ "type": "Pass" }`
//! - `player`: 1=Black, 2=White
//! - `quality`: 0.0–1.0 (1.0 if mover's side won)
//! - `move_number`: 1-based move index
//! - `legal_moves`: flat indices of legal moves
//! - `checksum`: BLAKE3 hash of (board + action) for integrity

use std::io::{BufWriter, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};

use super::state::GoState;
use super::types::{GoAction, GoCell};

// ── JsonlGoSample ──────────────────────────────────────────────

/// Action type for JSONL Go samples, matching riir-gpu's `GoActionType`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum GoActionType {
    /// Place a stone at (row, col).
    Place { row: usize, col: usize },
    /// Pass turn.
    Pass,
}

impl GoActionType {
    /// Compute the flat action index for token encoding.
    ///
    /// For 9x9: vocab = 3 (cell types) + 81 (positions) + 1 (pass) = 85.
    /// Action offset = 3 (after cell-type tokens).
    /// Place(r,c) → 3 + r*size + c, Pass → 3 + size*size.
    pub fn to_flat_index(&self, board_size: usize) -> usize {
        const ACTION_OFFSET: usize = 3; // Matches riir-gpu GoTokenEncoder
        match self {
            Self::Place { row, col } => ACTION_OFFSET + row * board_size + col,
            Self::Pass => ACTION_OFFSET + board_size * board_size,
        }
    }
}

impl From<&GoAction> for GoActionType {
    fn from(a: &GoAction) -> Self {
        match a {
            GoAction::Place(r, c) => Self::Place { row: *r, col: *c },
            GoAction::Pass => Self::Pass,
        }
    }
}

/// Per-move Go sample serialized as one JSONL line.
///
/// Designed for riir-ai LoRA training via `GoJsonlLoader` (Plan 271 T1.3).
/// Compatible with `riir-gpu::GoGameSample` token encoding:
/// - vocab_size = 3 + board_size² + 1
/// - block_size = board_size² + 1 (board cells + action)
///
/// Binary (postcard) serialization is supported via [`JsonlGoSample::to_bytes`] /
/// [`JsonlGoSample::from_bytes`] and is the preferred wire format for new code
/// (Issue 011). The JSON shape remains for the existing riir-ai JSONL pipeline.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct JsonlGoSample {
    /// Board state: `board_size * board_size` cells (0=Empty, 1=Black, 2=White).
    pub board: Vec<u8>,
    /// Board dimension (typically 9).
    pub board_size: usize,
    /// Action taken in this state.
    pub action: GoActionType,
    /// Which player made this move: 1=Black, 2=White.
    pub player: u8,
    /// Outcome quality: 1.0 if mover's side won, 0.0 if lost.
    pub quality: f32,
    /// Move number within the game (1-based).
    pub move_number: u32,
    /// Flat indices of legal moves at this position.
    pub legal_moves: Vec<usize>,
    /// BLAKE3 hash of (board || action_flat_index) for integrity.
    ///
    /// Wire shape: hex string in JSONL (for riir-ai loader compatibility) and
    /// the same hex string under postcard. `serialize_with` / `deserialize_with`
    /// keep both formats round-trippable without changing the JSONL contract.
    #[serde(
        serialize_with = "serialize_blake3_hex",
        deserialize_with = "deserialize_blake3_hex"
    )]
    pub checksum: [u8; 32],
}

/// Serialize BLAKE3 hash as hex string for human-readable JSONL.
fn serialize_blake3_hex<S: serde::Serializer>(hash: &[u8; 32], s: S) -> Result<S::Ok, S::Error> {
    let hex: String = hash.iter().map(|b| format!("{b:02x}")).collect();
    s.serialize_str(&hex)
}

/// Deserialize BLAKE3 hash from the hex string written by [`serialize_blake3_hex`].
///
/// Used by both JSONL and postcard so the field round-trips identically across
/// both formats (Issue 011).
fn deserialize_blake3_hex<'de, D: serde::Deserializer<'de>>(d: D) -> Result<[u8; 32], D::Error> {
    use serde::de::Error;
    let hex = String::deserialize(d)?;
    if hex.len() != 64 {
        return Err(D::Error::custom(format!(
            "blake3 hex must be 64 chars, got {}",
            hex.len()
        )));
    }
    let mut out = [0u8; 32];
    let bytes = hex.as_bytes();
    for i in 0..32 {
        let hi = hex_nibble(bytes[i * 2]).map_err(D::Error::custom)?;
        let lo = hex_nibble(bytes[i * 2 + 1]).map_err(D::Error::custom)?;
        out[i] = (hi << 4) | lo;
    }
    Ok(out)
}

/// Decode a single hex ASCII byte to its nibble value.
fn hex_nibble(b: u8) -> Result<u8, &'static str> {
    match b {
        b'0'..=b'9' => Ok(b - b'0'),
        b'a'..=b'f' => Ok(b - b'a' + 10),
        b'A'..=b'F' => Ok(b - b'A' + 10),
        _ => Err("invalid hex character in blake3 checksum"),
    }
}

impl JsonlGoSample {
    /// Create a sample from the current game state and the action taken.
    ///
    /// Call this BEFORE applying the move to `state` (captures pre-move board).
    /// `winner` is determined post-game and assigned to all samples retroactively.
    pub fn from_state(state: &GoState, action: &GoAction, move_number: u32, quality: f32) -> Self {
        let board_size = state.size;
        let player = match state.to_play {
            GoCell::Black => 1u8,
            GoCell::White => 2u8,
            GoCell::Empty => 0u8,
        };

        // Board as flat u8 array
        let board: Vec<u8> = state.board.iter().map(|c| *c as u8).collect();

        // Legal moves as flat indices
        let legal_moves: Vec<usize> = state
            .legal_moves()
            .iter()
            .map(|(r, c)| r * board_size + c)
            .collect();

        let action_type = GoActionType::from(action);

        // BLAKE3 integrity hash
        let action_flat = action_type.to_flat_index(board_size);
        let mut hasher = blake3::Hasher::new();
        hasher.update(&board);
        hasher.update(&action_flat.to_le_bytes());
        let checksum = hasher.finalize().into();

        Self {
            board,
            board_size,
            action: action_type,
            player,
            quality,
            move_number,
            legal_moves,
            checksum,
        }
    }

    /// Expected token dimensions for riir-ai encoding.
    ///
    /// - `vocab_size = 3 + board_size² + 1` (3 cell types + positions + pass)
    /// - `block_size = board_size² + 1` (board + action)
    pub fn token_dims(board_size: usize) -> (usize, usize) {
        let vocab_size = 3 + board_size * board_size + 1;
        let block_size = board_size * board_size + 1;
        (vocab_size, block_size)
    }

    /// Serialize to binary (postcard). Zero-copy friendly.
    ///
    /// Matches the pattern from `src/pruners/bomber/replay.rs` (Issue 011).
    /// Prefer this over `serde_json::to_string` for new writers.
    ///
    /// Routes through [`JsonlGoSampleBin`] because (a) postcard cannot serialize
    /// the internally-tagged `GoActionType` (`#[serde(tag = "type")]`) used for
    /// the JSONL shape, and (b) the binary shadow stores `checksum` as raw
    /// `[u8; 32]` (32 bytes) instead of a hex string (64 bytes).
    pub fn to_bytes(&self) -> Vec<u8> {
        let bin = JsonlGoSampleBin::from(self);
        postcard::to_allocvec(&bin).unwrap_or_default()
    }

    /// Deserialize from binary (postcard).
    pub fn from_bytes(data: &[u8]) -> Result<Self, postcard::Error> {
        let bin: JsonlGoSampleBin = postcard::from_bytes(data)?;
        Ok(Self::from(bin))
    }
}

// ── Binary (postcard) shadow types ─────────────────────────────
//
// `JsonlGoSample` uses `#[serde(tag = "type")]` on `GoActionType` and a
// hex-string `serialize_with` / `deserialize_with` on `checksum` for the
// human-readable JSONL consumed by the riir-ai training pipeline. Postcard
// returns `WontImplement` for internally-tagged enums, so the binary path
// routes through these externally-tagged (postcard-native) shadows. The
// checksum is stored as raw bytes (compact, not hex).

/// Postcard-native action variant (externally tagged — postcard's default).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
enum GoActionTypeBin {
    Place { row: usize, col: usize },
    Pass,
}

impl From<&GoActionType> for GoActionTypeBin {
    fn from(a: &GoActionType) -> Self {
        match a {
            GoActionType::Place { row, col } => Self::Place { row: *row, col: *col },
            GoActionType::Pass => Self::Pass,
        }
    }
}

impl From<GoActionTypeBin> for GoActionType {
    fn from(a: GoActionTypeBin) -> Self {
        match a {
            GoActionTypeBin::Place { row, col } => Self::Place { row, col },
            GoActionTypeBin::Pass => Self::Pass,
        }
    }
}

/// Postcard-native Go sample envelope. `checksum` is raw `[u8; 32]` (no hex).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct JsonlGoSampleBin {
    board: Vec<u8>,
    board_size: usize,
    action: GoActionTypeBin,
    player: u8,
    quality: f32,
    move_number: u32,
    legal_moves: Vec<usize>,
    checksum: [u8; 32],
}

impl From<&JsonlGoSample> for JsonlGoSampleBin {
    fn from(s: &JsonlGoSample) -> Self {
        Self {
            board: s.board.clone(),
            board_size: s.board_size,
            action: GoActionTypeBin::from(&s.action),
            player: s.player,
            quality: s.quality,
            move_number: s.move_number,
            legal_moves: s.legal_moves.clone(),
            checksum: s.checksum,
        }
    }
}

impl From<JsonlGoSampleBin> for JsonlGoSample {
    fn from(s: JsonlGoSampleBin) -> Self {
        Self {
            board: s.board,
            board_size: s.board_size,
            action: GoActionType::from(s.action),
            player: s.player,
            quality: s.quality,
            move_number: s.move_number,
            legal_moves: s.legal_moves,
            checksum: s.checksum,
        }
    }
}

// ── GoReplayWriter ─────────────────────────────────────────────

/// Go replay writer — outputs per-move samples as JSONL for riir-ai training.
///
/// ## Usage
///
/// ```ignore
/// use katgpt_rs::pruners::go::replay_writer::GoReplayWriter;
/// use katgpt_rs::pruners::go::state::GoState;
/// use katgpt_rs::pruners::go::types::GoAction;
///
/// let mut writer = GoReplayWriter::create("output/replay.jsonl", 9).unwrap();
/// let mut state = GoState::new(9);
///
/// // For each move in the game:
/// let sample = JsonlGoSample::from_state(&state, &GoAction::Place(4, 4), 1, 1.0);
/// writer.write_sample(&sample).unwrap();
/// state.play_move(4, 4);
/// ```
///
/// ## Quality Assignment
///
/// Quality is binary: 1.0 if the mover's side won, 0.0 if lost.
/// Since winner is unknown until game end, collect samples during play,
/// then write them with the correct quality after the game finishes.
pub struct GoReplayWriter {
    writer: BufWriter<std::fs::File>,
    board_size: usize,
    sample_count: u64,
}

impl GoReplayWriter {
    /// Create a new JSONL writer at the given path.
    ///
    /// Creates parent directories if needed.
    pub fn create(path: &Path, board_size: usize) -> std::io::Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let file = std::fs::File::create(path)?;
        Ok(Self {
            writer: BufWriter::new(file),
            board_size,
            sample_count: 0,
        })
    }

    /// Write one sample as a JSON line.
    ///
    /// Each call appends one JSON object followed by newline.
    ///
    /// Deprecated: prefer [`GoReplayWriter::write_sample_binary`] for zero-copy
    /// binary records. The JSON path remains only because the riir-ai training
    /// pipeline still consumes JSONL; it will be removed once that pipeline
    /// migrates (Issue 011).
    #[deprecated(
        note = "use write_sample_binary; JSON removed when riir-ai training pipeline migrates"
    )]
    pub fn write_sample(&mut self, sample: &JsonlGoSample) -> std::io::Result<()> {
        // Validate board dimensions match writer config
        debug_assert_eq!(
            sample.board_size, self.board_size,
            "Sample board_size {} != writer board_size {}",
            sample.board_size, self.board_size
        );

        let json = serde_json::to_string(sample)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        self.writer.write_all(json.as_bytes())?;
        self.writer.write_all(b"\n")?;
        self.sample_count += 1;
        Ok(())
    }

    /// Write one sample as a length-prefixed binary record (no JSON).
    ///
    /// Layout per record: `len(4 LE) + postcard payload`. Matches the pattern
    /// from `src/pruners/bomber/replay.rs` (Issue 011).
    pub fn write_sample_binary(&mut self, sample: &JsonlGoSample) -> std::io::Result<()> {
        // Validate board dimensions match writer config
        debug_assert_eq!(
            sample.board_size, self.board_size,
            "Sample board_size {} != writer board_size {}",
            sample.board_size, self.board_size
        );

        let payload = sample.to_bytes();
        let len = payload.len() as u32;
        self.writer.write_all(&len.to_le_bytes())?;
        self.writer.write_all(&payload)?;
        self.sample_count += 1;
        Ok(())
    }

    /// Flush buffered writes to disk.
    pub fn flush(&mut self) -> std::io::Result<()> {
        self.writer.flush()
    }

    /// Board size this writer was configured for.
    pub fn board_size(&self) -> usize {
        self.board_size
    }

    /// Number of samples written so far.
    pub fn sample_count(&self) -> u64 {
        self.sample_count
    }
}

impl Drop for GoReplayWriter {
    fn drop(&mut self) {
        // Best-effort flush on drop
        let _ = self.flush();
    }
}

// ── Game Sample Collection ─────────────────────────────────────

/// Collects per-move samples during a game, then writes with winner quality.
///
/// Usage:
/// 1. `record_move()` for each move during play (quality = 0.5 placeholder)
/// 2. `finalize_and_write()` after game ends with the actual winner
pub struct GameSampleCollector {
    samples: Vec<(JsonlGoSample, u8)>, // (sample, player_id)
    #[allow(dead_code)] // Reserved for validation in future multi-size collector
    board_size: usize,
}

impl GameSampleCollector {
    /// Create a new collector for the given board size.
    pub fn new(board_size: usize) -> Self {
        Self {
            samples: Vec::new(),
            board_size,
        }
    }

    /// Record a move snapshot. Call BEFORE applying the move to `state`.
    pub fn record_move(&mut self, state: &GoState, action: &GoAction, move_number: u32) {
        let player = state.to_play.player_id();
        // Placeholder quality — finalized when winner is known
        let sample = JsonlGoSample::from_state(state, action, move_number, 0.5);
        self.samples.push((sample, player));
    }

    /// Finalize all samples with winner quality and write to JSONL.
    ///
    /// Quality = 1.0 for winner's moves, 0.0 for loser's moves.
    /// If `quality_threshold > 0.0`, only writes samples where quality > threshold
    /// (i.e., only winning player's moves).
    ///
    /// Returns the number of samples written.
    ///
    /// Deprecated: prefer [`GameSampleCollector::finalize_and_write_binary`] for
    /// zero-copy binary records. The JSONL path remains for the existing riir-ai
    /// training pipeline (Issue 011).
    #[deprecated(
        note = "use finalize_and_write_binary; JSON removed when riir-ai training pipeline migrates"
    )]
    #[allow(deprecated)] // intentionally routes through the deprecated JSON writer
    pub fn finalize_and_write(
        mut self,
        winner: Option<GoCell>,
        writer: &mut GoReplayWriter,
        quality_threshold: f32,
    ) -> std::io::Result<usize> {
        Self::finalize_inner(&mut self.samples, winner, quality_threshold, |w, s| {
            w.write_sample(s)
        }, writer)
    }

    /// Finalize all samples with winner quality and write as length-prefixed
    /// binary records via [`GoReplayWriter::write_sample_binary`].
    ///
    /// Same quality semantics as [`Self::finalize_and_write`]. Preferred for new
    /// code (Issue 011).
    pub fn finalize_and_write_binary(
        mut self,
        winner: Option<GoCell>,
        writer: &mut GoReplayWriter,
        quality_threshold: f32,
    ) -> std::io::Result<usize> {
        Self::finalize_inner(&mut self.samples, winner, quality_threshold, |w, s| {
            w.write_sample_binary(s)
        }, writer)
    }

    /// Shared finalize loop — applies quality assignment + threshold filter,
    /// then dispatches each sample to `emit`.
    ///
    /// Factored out so the JSON and binary paths can't drift in semantics.
    fn finalize_inner(
        samples: &mut Vec<(JsonlGoSample, u8)>,
        winner: Option<GoCell>,
        quality_threshold: f32,
        mut emit: impl FnMut(&mut GoReplayWriter, &JsonlGoSample) -> std::io::Result<()>,
        writer: &mut GoReplayWriter,
    ) -> std::io::Result<usize> {
        let winner_id = winner.map(|c| c.player_id());

        let mut written = 0usize;
        for (mut sample, player_id) in samples.drain(..) {
            // Assign quality: 1.0 if this player won, 0.0 if lost, 0.5 if draw
            sample.quality = match winner_id {
                Some(wid) if wid == player_id => 1.0,
                Some(_) => 0.0,
                None => 0.5, // Draw
            };

            // Quality filter: skip low-quality samples (typically loser moves)
            if sample.quality < quality_threshold {
                continue;
            }

            emit(writer, &sample)?;
            written += 1;
        }

        Ok(written)
    }

    /// Number of recorded moves (before filtering).
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Whether any moves have been recorded.
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
#[allow(deprecated)] // JSONL path is intentionally exercised until riir-ai migrates (Issue 011)
mod tests {
    use super::*;

    #[test]
    fn go_action_type_flat_index() {
        // 9x9 board
        assert_eq!(GoActionType::Place { row: 0, col: 0 }.to_flat_index(9), 3);
        assert_eq!(
            GoActionType::Place { row: 4, col: 4 }.to_flat_index(9),
            3 + 36 + 4
        );
        assert_eq!(GoActionType::Pass.to_flat_index(9), 3 + 81);
    }

    #[test]
    fn token_dims_9x9() {
        let (vocab, block) = JsonlGoSample::token_dims(9);
        assert_eq!(vocab, 85); // 3 + 81 + 1
        assert_eq!(block, 82); // 81 + 1
    }

    #[test]
    fn token_dims_19x19() {
        let (vocab, block) = JsonlGoSample::token_dims(19);
        assert_eq!(vocab, 3 + 361 + 1); // 365
        assert_eq!(block, 361 + 1); // 362
    }

    #[test]
    fn sample_from_state() {
        let state = GoState::new(9);
        let action = GoAction::Place(4, 4);
        let sample = JsonlGoSample::from_state(&state, &action, 1, 1.0);

        assert_eq!(sample.board_size, 9);
        assert_eq!(sample.board.len(), 81);
        assert!(sample.board.iter().all(|&c| c == 0)); // Empty board
        assert_eq!(sample.player, 1); // Black
        assert_eq!(sample.move_number, 1);
        assert!((sample.quality - 1.0).abs() < f32::EPSILON);
        assert_eq!(sample.action, GoActionType::Place { row: 4, col: 4 });
        assert_eq!(sample.legal_moves.len(), 81); // All positions legal on empty 9x9
    }

    #[test]
    fn writer_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.jsonl");

        let state = GoState::new(9);
        let action = GoAction::Place(4, 4);
        let sample = JsonlGoSample::from_state(&state, &action, 1, 1.0);

        {
            let mut writer = GoReplayWriter::create(&path, 9).unwrap();
            writer.write_sample(&sample).unwrap();
            writer.write_sample(&sample).unwrap();
            writer.flush().unwrap();
            assert_eq!(writer.sample_count(), 2);
        }

        // Read back and verify
        let content = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(lines.len(), 2);

        // Verify each line is valid JSON with expected fields
        for line in &lines {
            let parsed: serde_json::Value = serde_json::from_str(line).unwrap();
            assert_eq!(parsed["board_size"], 9);
            assert_eq!(parsed["player"], 1);
            assert_eq!(parsed["move_number"], 1);
            assert_eq!(parsed["quality"], 1.0);
            assert!(parsed["checksum"].is_string());
            assert!(parsed["legal_moves"].is_array());
            assert!(parsed["board"].is_array());
            // Action should have type "Place" with row/col data
            assert_eq!(parsed["action"]["type"], "Place");
        }
    }

    #[test]
    fn collector_finalize_winner_filter() {
        let state = GoState::new(9);

        let mut collector = GameSampleCollector::new(9);
        // Move 1: Black plays (4,4)
        collector.record_move(&state, &GoAction::Place(4, 4), 1);
        // Simulate Black played, White's turn
        let mut state2 = state.clone();
        state2.play_move(4, 4);
        // Move 2: White plays (0,0)
        collector.record_move(&state2, &GoAction::Place(0, 0), 2);

        assert_eq!(collector.len(), 2);

        // Write with Black winning, threshold 0.5 → only Black's move written
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("filtered.jsonl");
        let mut writer = GoReplayWriter::create(&path, 9).unwrap();

        let written = collector
            .finalize_and_write(Some(GoCell::Black), &mut writer, 0.5)
            .unwrap();
        writer.flush().unwrap();
        assert_eq!(written, 1); // Only Black's move

        // Verify the written sample has quality 1.0
        let content = std::fs::read_to_string(&path).unwrap();
        let parsed: serde_json::Value =
            serde_json::from_str(content.lines().next().unwrap()).unwrap();
        assert_eq!(parsed["quality"], 1.0);
        assert_eq!(parsed["player"], 1);
    }

    #[test]
    fn collector_finalize_all_moves() {
        let state = GoState::new(9);

        let mut collector = GameSampleCollector::new(9);
        collector.record_move(&state, &GoAction::Place(4, 4), 1);

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("all.jsonl");
        let mut writer = GoReplayWriter::create(&path, 9).unwrap();

        // threshold 0.0 → all moves written
        let written = collector
            .finalize_and_write(Some(GoCell::Black), &mut writer, 0.0)
            .unwrap();
        assert_eq!(written, 1);
    }

    #[test]
    fn collector_finalize_draw() {
        let state = GoState::new(9);

        let mut collector = GameSampleCollector::new(9);
        collector.record_move(&state, &GoAction::Place(4, 4), 1);

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("draw.jsonl");
        let mut writer = GoReplayWriter::create(&path, 9).unwrap();

        // Draw → quality 0.5, threshold 0.0 → written
        let written = collector
            .finalize_and_write(None, &mut writer, 0.0)
            .unwrap();
        writer.flush().unwrap();
        assert_eq!(written, 1);

        let content = std::fs::read_to_string(&path).unwrap();
        let parsed: serde_json::Value =
            serde_json::from_str(content.lines().next().unwrap()).unwrap();
        assert_eq!(parsed["quality"], 0.5);
    }

    /// Generate N random self-play games and verify JSONL round-trip (Plan 271 T2.3).
    #[test]
    fn self_play_jsonl_roundtrip() {
        let board_size = 9;
        let num_games = 10;
        let max_moves = board_size * board_size * 3;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("roundtrip.jsonl");

        let mut rng = fastrand::Rng::with_seed(12345);
        let mut writer = GoReplayWriter::create(&path, board_size).unwrap();

        let mut total_samples = 0usize;

        for _ in 0..num_games {
            let mut state = GoState::new(board_size);
            let mut collector = GameSampleCollector::new(board_size);
            let mut moves_played = 0usize;

            while !state.is_terminal() && moves_played < max_moves {
                let legal_moves = state.legal_moves();

                if legal_moves.is_empty() {
                    collector.record_move(&state, &GoAction::Pass, moves_played as u32 + 1);
                    state.play_pass();
                    moves_played += 1;
                    continue;
                }

                // Random move selection
                let (r, c) = legal_moves[rng.usize(..legal_moves.len())];
                let action = GoAction::Place(r, c);
                collector.record_move(&state, &action, moves_played as u32 + 1);
                state.play_move(r, c);
                moves_played += 1;
            }

            // Force end if not terminal
            if !state.is_terminal() {
                state.play_pass();
                state.play_pass();
            }

            let winner = state.get_winner();
            let written = collector
                .finalize_and_write(winner, &mut writer, 0.0)
                .unwrap();
            total_samples += written;
        }

        writer.flush().unwrap();

        // Verify sample count
        assert!(total_samples > 0, "Should have written some samples");

        // Read back and verify
        let content = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(
            lines.len(),
            total_samples,
            "Line count should match sample count"
        );

        // Verify each line is valid and dimensions are correct
        let (expected_vocab, _expected_block) = JsonlGoSample::token_dims(board_size);

        for line in &lines {
            let parsed: serde_json::Value = serde_json::from_str(line).unwrap();

            // Board dimensions
            let board = parsed["board"].as_array().unwrap();
            assert_eq!(
                board.len(),
                board_size * board_size,
                "Board should be {board_size}²"
            );

            // Legal moves are flat indices
            let legal = parsed["legal_moves"].as_array().unwrap();
            for m in legal {
                let idx = m.as_u64().unwrap() as usize;
                assert!(
                    idx < board_size * board_size,
                    "Legal move index {idx} out of range"
                );
            }

            // Action flat index within vocab
            let action_type = parsed["action"]["type"].as_str().unwrap();
            match action_type {
                "Place" => {
                    let row = parsed["action"]["row"].as_u64().unwrap() as usize;
                    let col = parsed["action"]["col"].as_u64().unwrap() as usize;
                    let flat = 3 + row * board_size + col;
                    assert!(
                        flat < expected_vocab,
                        "Action flat {flat} exceeds vocab {expected_vocab}"
                    );
                }
                "Pass" => {
                    let flat = 3 + board_size * board_size;
                    assert!(
                        flat < expected_vocab,
                        "Pass action {flat} exceeds vocab {expected_vocab}"
                    );
                }
                other => panic!("Unknown action type: {other}"),
            }

            // Quality is 0.0, 0.5, or 1.0
            let quality = parsed["quality"].as_f64().unwrap() as f32;
            assert!(
                matches!(quality, 0.0 | 0.5 | 1.0),
                "Quality should be binary or draw: {quality}"
            );
        }
    }

    #[test]
    fn blake3_checksum_deterministic() {
        let state = GoState::new(9);
        let action = GoAction::Place(4, 4);
        let s1 = JsonlGoSample::from_state(&state, &action, 1, 1.0);
        let s2 = JsonlGoSample::from_state(&state, &action, 1, 1.0);
        assert_eq!(
            s1.checksum, s2.checksum,
            "Same state+action must produce same checksum"
        );

        // Different action → different checksum
        let s3 = JsonlGoSample::from_state(&state, &GoAction::Place(0, 0), 1, 1.0);
        assert_ne!(
            s1.checksum, s3.checksum,
            "Different actions must produce different checksums"
        );
    }

    // ── Binary path (Issue 011) ──────────────────────────────────

    #[test]
    fn sample_binary_roundtrip_matches_original() {
        let state = GoState::new(9);
        let action = GoAction::Place(4, 4);
        let original = JsonlGoSample::from_state(&state, &action, 1, 1.0);

        let bytes = original.to_bytes();
        assert!(!bytes.is_empty(), "to_bytes must produce output");

        let restored = JsonlGoSample::from_bytes(&bytes)
            .expect("postcard round-trip should succeed");

        // Every field must round-trip exactly.
        assert_eq!(restored.board, original.board);
        assert_eq!(restored.board_size, original.board_size);
        assert_eq!(restored.action, original.action);
        assert_eq!(restored.player, original.player);
        assert_eq!(restored.quality, original.quality);
        assert_eq!(restored.move_number, original.move_number);
        assert_eq!(restored.legal_moves, original.legal_moves);
        assert_eq!(
            restored.checksum, original.checksum,
            "BLAKE3 checksum must round-trip through postcard hex-string codec"
        );
        assert_eq!(restored, original, "full struct equality after binary round-trip");
    }

    #[test]
    fn writer_binary_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.bin");

        let state = GoState::new(9);
        let action = GoAction::Place(4, 4);
        let sample = JsonlGoSample::from_state(&state, &action, 1, 1.0);

        {
            let mut writer = GoReplayWriter::create(&path, 9).unwrap();
            writer.write_sample_binary(&sample).unwrap();
            writer.write_sample_binary(&sample).unwrap();
            writer.flush().unwrap();
            assert_eq!(writer.sample_count(), 2);
        }

        // Read back length-prefixed binary records.
        let contents = std::fs::read(&path).unwrap();
        let mut offset = 0usize;
        let mut count = 0usize;
        while offset + 4 <= contents.len() {
            let len = u32::from_le_bytes(
                contents[offset..offset + 4].try_into().unwrap(),
            ) as usize;
            offset += 4;
            let restored = JsonlGoSample::from_bytes(&contents[offset..offset + len])
                .expect("each record must deserialize");
            assert_eq!(restored, sample, "binary record must round-trip identically");
            offset += len;
            count += 1;
        }
        assert_eq!(count, 2, "should read exactly two records");
    }

    #[test]
    fn collector_finalize_binary_winner_filter() {
        let state = GoState::new(9);

        let mut collector = GameSampleCollector::new(9);
        // Move 1: Black plays (4,4)
        collector.record_move(&state, &GoAction::Place(4, 4), 1);
        // White's turn, plays (0,0)
        let mut state2 = state.clone();
        state2.play_move(4, 4);
        collector.record_move(&state2, &GoAction::Place(0, 0), 2);

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("filtered.bin");
        let mut writer = GoReplayWriter::create(&path, 9).unwrap();

        // Black wins, threshold 0.5 → only Black's move written.
        let written = collector
            .finalize_and_write_binary(Some(GoCell::Black), &mut writer, 0.5)
            .unwrap();
        writer.flush().unwrap();
        assert_eq!(written, 1);

        // Read back the single binary record and verify quality=1.0, player=Black.
        let contents = std::fs::read(&path).unwrap();
        assert!(contents.len() >= 4, "file must contain at least one length prefix");
        let len = u32::from_le_bytes(contents[0..4].try_into().unwrap()) as usize;
        let restored = JsonlGoSample::from_bytes(&contents[4..4 + len]).unwrap();
        assert!((restored.quality - 1.0).abs() < f32::EPSILON);
        assert_eq!(restored.player, 1); // Black
    }

    /// Same shape as `self_play_jsonl_roundtrip` but writes binary records and
    /// reads them back, validating that every sample round-trips through postcard.
    #[test]
    fn self_play_binary_roundtrip() {
        let board_size = 9;
        let num_games = 10;
        let max_moves = board_size * board_size * 3;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("roundtrip.bin");

        let mut rng = fastrand::Rng::with_seed(12345);
        let mut writer = GoReplayWriter::create(&path, board_size).unwrap();

        let mut total_samples = 0usize;
        for _ in 0..num_games {
            let mut state = GoState::new(board_size);
            let mut collector = GameSampleCollector::new(board_size);
            let mut moves_played = 0usize;

            while !state.is_terminal() && moves_played < max_moves {
                let legal_moves = state.legal_moves();
                if legal_moves.is_empty() {
                    collector.record_move(&state, &GoAction::Pass, moves_played as u32 + 1);
                    state.play_pass();
                    moves_played += 1;
                    continue;
                }
                let (r, c) = legal_moves[rng.usize(..legal_moves.len())];
                let action = GoAction::Place(r, c);
                collector.record_move(&state, &action, moves_played as u32 + 1);
                state.play_move(r, c);
                moves_played += 1;
            }

            if !state.is_terminal() {
                state.play_pass();
                state.play_pass();
            }

            let winner = state.get_winner();
            let written = collector
                .finalize_and_write_binary(winner, &mut writer, 0.0)
                .unwrap();
            total_samples += written;
        }
        writer.flush().unwrap();

        assert!(total_samples > 0, "should have written samples");

        // Read every length-prefixed record back and validate shape.
        let contents = std::fs::read(&path).unwrap();
        let mut offset = 0usize;
        let mut read = 0usize;
        while offset + 4 <= contents.len() {
            let len = u32::from_le_bytes(
                contents[offset..offset + 4].try_into().unwrap(),
            ) as usize;
            offset += 4;
            let s = JsonlGoSample::from_bytes(&contents[offset..offset + len])
                .expect("each binary record must deserialize");
            offset += len;
            read += 1;

            assert_eq!(s.board.len(), board_size * board_size);
            assert!(
                matches!(s.quality, 0.0 | 0.5 | 1.0),
                "quality should be binary or draw: {}",
                s.quality
            );
        }
        assert_eq!(read, total_samples, "record count must match samples written");
    }
}
