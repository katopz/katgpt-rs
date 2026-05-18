# Plan 077: Arena Integration — RubricPlayer Match Scheduling + Leaderboards

> **Status**: Done
> **Depends on**: Plan 071 (ROPD Rubric — RubricPlayer + RubricFFTPlayer built)
> **Scope**: Cross-arena tournament runner, score aggregation, ELO ratings, leaderboards
> **Feature Gate**: `ropd_rubric` (implies `bandit`, `bomber`, `fft`)

## Context

Plan 071 built `RubricPlayer` (Bomber) and `RubricFFTPlayer` (FFT) — both compile behind
`ropd_rubric` feature gate and pass unit tests. However, these players have **never been
pitted against the existing player hierarchy** in a real tournament:

- Bomber: Random → Greedy → Validator → HL → GZero → **Rubric**
- FFT: Random → Greedy → Validator → HL → GZero → **RubricFFT**

Currently each arena has its own ad-hoc tournament runner:
- `bomber_01_arena.rs` — hardcoded 4-player fixed lineup, no matchup rotation
- `fft_01_arena.rs` — 8-unit party vs enemy, no per-player comparison
- `go_02_tournament.rs` — **good pattern**: `MatchupResult`, `run_matchup()`, round-robin

## Objective

1. **Prove rubric players compete** — run RubricPlayer vs GZero/HL/Validator/Greedy/Random
2. **Cross-arena leaderboard** — unified scoring across Bomber + FFT domains
3. **Reusable tournament infrastructure** — shared types in `pruners/arena/`
4. **Benchmark** — measure rubric player win rates vs baselines

## Architecture

```
microgpt-rs/src/pruners/arena/
├── mod.rs              — re-exports
├── types.rs            — PlayerEntry, MatchupConfig, Leaderboard, ELO
└── scheduler.rs        — RoundRobinScheduler, match scheduling

microgpt-rs/src/pruners/bomber/
├── rubric_player.rs    — (exists) RubricPlayer
└── arena_runner.rs     — BomberArenaRunner: runs N-round 4-player matches

microgpt-rs/src/pruners/fft/
├── rubric_player.rs    — (exists) RubricFFTPlayer
└── arena_runner.rs     — FFTArenaRunner: runs N-round 8-unit battles

microgpt-rs/examples/
├── bomber_09_rubric_tournament.rs  — RubricPlayer vs all baselines
└── fft_02_rubric_tournament.rs     — RubricFFTPlayer vs all baselines
```

## Tasks

- [x] **T1**: `arena/types.rs` — shared tournament types
- [x] **T2**: `arena/scheduler.rs` — round-robin matchup generator
- [x] **T3**: `arena/mod.rs` — module index + re-exports
- [x] **T4**: `bomber/arena_runner.rs` — BomberArenaRunner
- [x] **T5**: `fft/arena_runner.rs` — FFTArenaRunner
- [x] **T6**: `bomber_09_rubric_tournament.rs` — Bomber rubric tournament example
- [x] **T7**: `fft_02_rubric_tournament.rs` — FFT rubric tournament example
- [x] **T8**: Run tournaments + record results to `.benchmarks/009_arena_integration.md`
- [x] **T9**: Update documentation (README, overview, heuristic-learning)
- [x] **T10**: Clippy + test pass with `ropd_rubric,g_zero,bomber,fft`

---

## T1: `arena/types.rs` — Shared Tournament Types

### File: `microgpt-rs/src/pruners/arena/types.rs`

```rust
use std::fmt;

/// Which game domain the tournament runs in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ArenaKind {
    Bomber,
    Fft,
}

impl fmt::Display for ArenaKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bomber => write!(f, "Bomber"),
            Self::Fft => write!(f, "FFT"),
        }
    }
}

/// Result of a single game/round.
#[derive(Clone, Debug)]
pub struct GameResult {
    /// Player index of the winner (or draw sentinel).
    pub winner: Option<usize>,
    /// Per-player scores for this game.
    pub scores: Vec<i32>,
    /// Number of ticks/turns played.
    pub ticks: u32,
    /// Duration of the game.
    pub duration: std::time::Duration,
}

/// Result of a matchup (N games between same player set).
#[derive(Clone, Debug)]
pub struct MatchupResult {
    /// Arena domain.
    pub arena: ArenaKind,
    /// Player names in lineup order.
    pub player_names: Vec<String>,
    /// Individual game results.
    pub games: Vec<GameResult>,
}

impl MatchupResult {
    /// Wins for player at given index across all games.
    pub fn wins_for(&self, idx: usize) -> usize {
        self.games
            .iter()
            .filter(|g| g.winner == Some(idx))
            .count()
    }

    /// Win rate for player at given index (0.0–1.0).
    pub fn win_rate(&self, idx: usize) -> f64 {
        if self.games.is_empty() {
            return 0.0;
        }
        self.wins_for(idx) as f64 / self.games.len() as f64
    }

    /// Average game duration.
    pub fn avg_duration(&self) -> std::time::Duration {
        if self.games.is_empty() {
            return std::time::Duration::ZERO;
        }
        let total: std::time::Duration = self.games.iter().map(|g| g.duration).sum();
        total / self.games.len() as u32
    }
}

/// Player ranking entry for leaderboard.
#[derive(Clone, Debug)]
pub struct Ranking {
    /// Player name.
    pub name: String,
    /// Arena domain.
    pub arena: ArenaKind,
    /// Total wins across all matchups.
    pub wins: usize,
    /// Total losses across all matchups.
    pub losses: usize,
    /// Total draws.
    pub draws: usize,
    /// ELO rating.
    pub elo: f64,
}

impl Ranking {
    /// Total games played.
    pub fn total(&self) -> usize {
        self.wins + self.losses + self.draws
    }

    /// Win rate as percentage.
    pub fn win_pct(&self) -> f64 {
        if self.total() == 0 { 0.0 } else { self.wins as f64 / self.total() as f64 * 100.0 }
    }
}

/// Aggregated leaderboard across all matchups.
#[derive(Clone, Debug, Default)]
pub struct Leaderboard {
    pub rankings: Vec<Ranking>,
}

impl Leaderboard {
    /// Sort rankings by ELO descending.
    pub fn sort(&mut self) {
        self.rankings.sort_by(|a, b| b.elo.partial_cmp(&a.elo).unwrap_or(std::cmp::Ordering::Equal));
    }

    /// Format as markdown table.
    pub fn to_markdown(&self, arena: ArenaKind) -> String {
        let mut md = format!("## {arena} Arena Leaderboard\n\n");
        md.push_str("| Rank | Player | W | L | D | Win% | ELO |\n");
        md.push_str("|------|--------|---|---|---|------|-----|\n");
        for (i, r) in self.rankings.iter().enumerate() {
            md.push_str(&format!(
                "| {} | {} | {} | {} | {} | {:.1}% | {:.0} |\n",
                i + 1, r.name, r.wins, r.losses, r.draws, r.win_pct(), r.elo
            ));
        }
        md
    }
}

/// ELO rating calculator.
pub struct EloCalculator {
    /// K-factor (sensitivity per game).
    pub k: f64,
    /// Base rating.
    pub base: f64,
}

impl Default for EloCalculator {
    fn default() -> Self {
        Self { k: 32.0, base: 1000.0 }
    }
}

impl EloCalculator {
    /// Expected score for player A vs player B.
    pub fn expected(&self, rating_a: f64, rating_b: f64) -> f64 {
        1.0 / (1.0 + 10.0_f64.powf((rating_b - rating_a) / 400.0))
    }

    /// Update ratings after a game. Returns (new_a, new_b).
    pub fn update(&self, rating_a: f64, rating_b: f64, a_won: bool) -> (f64, f64) {
        let expected_a = self.expected(rating_a, rating_b);
        let actual_a = if a_won { 1.0 } else { 0.0 };
        let actual_b = 1.0 - actual_a;

        let new_a = rating_a + self.k * (actual_a - expected_a);
        let new_b = rating_b + self.k * (actual_b - (1.0 - expected_a));
        (new_a, new_b)
    }
}
```

---

## T2: `arena/scheduler.rs` — Round-Robin Matchup Generator

### File: `microgpt-rs/src/pruners/arena/scheduler.rs`

Generates all pairwise (or group) matchups for a tournament.

```rust
/// Configuration for a tournament schedule.
pub struct ScheduleConfig {
    /// Number of games per matchup.
    pub games_per_matchup: usize,
    /// Whether to include all-pairs or just linear progression.
    pub round_robin: bool,
}

impl Default for ScheduleConfig {
    fn default() -> Self {
        Self { games_per_matchup: 20, round_robin: true }
    }
}

/// A scheduled matchup between player indices.
pub struct Matchup {
    pub player_indices: Vec<usize>,
}

/// Generates matchups from a list of N players.
pub fn round_robin_pairs(n: usize) -> Vec<Matchup> {
    let mut matchups = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            matchups.push(Matchup { player_indices: vec![i, j] });
        }
    }
    matchups
}

/// Generates full-field matchups (all players compete simultaneously).
/// For Bomber: 4 players per match.
/// For FFT: 8 units (4v4) per match.
pub fn full_field_matchups(n: usize, field_size: usize) -> Vec<Matchup> {
    // Each player faces every combination of opponents.
    // For simplicity: run N rounds, rotating player compositions.
    let mut matchups = Vec::new();
    if n <= field_size {
        // All players fit in one field — single matchup
        matchups.push(Matchup { player_indices: (0..n).collect() });
    } else {
        // Multiple heats: each player plays in at least ceil(n/field_size) heats
        let heats = (n + field_size - 1) / field_size;
        for h in 0..heats {
            let start = (h * field_size) % n;
            let indices: Vec<usize> = (0..field_size)
                .map(|i| (start + i) % n)
                .collect();
            matchups.push(Matchup { player_indices: indices });
        }
    }
    matchups
}
```

---

## T3: `arena/mod.rs` — Module Index

### File: `microgpt-rs/src/pruners/arena/mod.rs`

```rust
//! Cross-arena tournament infrastructure — scheduling, scoring, leaderboards.
//!
//! Shared types for running RubricPlayer tournaments across Bomber and FFT domains.

pub mod scheduler;
pub mod types;

pub use scheduler::*;
pub use types::*;
```

### File: `microgpt-rs/src/pruners/mod.rs` — add `arena` module

```rust
// Add after existing modules:
#[cfg(any(feature = "bomber", feature = "fft"))]
pub mod arena;
```

---

## T4: `bomber/arena_runner.rs` — BomberArenaRunner

### File: `microgpt-rs/src/pruners/bomber/arena_runner.rs`

Runs a tournament between multiple BomberPlayer implementations.

```rust
use std::time::Instant;

use fastrand::Rng;

use crate::pruners::arena::types::*;

use super::arena::{EMPTY_ARENA, STANDARD_ARENA};
use super::players::BomberPlayer;
use super::{ArenaGrid, GameEvent, init_world_with_arena, run_tick, spawn_players};

/// Configuration for a Bomber arena tournament.
pub struct BomberArenaConfig {
    /// Number of games per matchup.
    pub games: usize,
    /// Tick limit per game.
    pub tick_limit: u32,
    /// Map preset name (empty, standard, pillar_heavy).
    pub map_preset: &'static str,
}

impl Default for BomberArenaConfig {
    fn default() -> Self {
        Self { games: 20, tick_limit: 200, map_preset: "standard" }
    }
}

/// Run a single 4-player Bomber game, returning per-player scores.
pub fn run_bomber_game(
    players: &mut [Box<dyn BomberPlayer>],
    grid: &ArenaGrid,
    tick_limit: u32,
    rng: &mut Rng,
) -> GameResult {
    let start = Instant::now();
    let mut world = init_world_with_arena(grid.clone());
    // ... spawn players, run ticks, collect events ...

    let mut ticks = 0u32;
    let mut survivors = Vec::new();

    for _ in 0..tick_limit {
        // Select actions, run tick, check alive
        ticks += 1;
        if survivors.len() <= 1 {
            break;
        }
    }

    GameResult {
        winner: survivors.first().copied(),
        scores: vec![0; players.len()],
        ticks,
        duration: start.elapsed(),
    }
}

/// Run a full Bomber tournament matchup.
pub fn run_bomber_matchup(
    players: &mut [Box<dyn BomberPlayer>],
    config: &BomberArenaConfig,
) -> MatchupResult {
    let grid = match config.map_preset {
        "empty" => ArenaGrid::fixed(EMPTY_ARENA).unwrap(),
        "pillar_heavy" => ArenaGrid::fixed(STANDARD_ARENA).unwrap(),
        _ => ArenaGrid::fixed(STANDARD_ARENA).unwrap(),
    };

    let mut rng = Rng::with_seed(42);
    let mut games = Vec::with_capacity(config.games);

    for _ in 0..config.games {
        let result = run_bomber_game(players, &grid, config.tick_limit, &mut rng);
        games.push(result);
        // Reset players for next game (bandit memory persists)
        for p in players.iter_mut() {
            p.reset();
        }
    }

    MatchupResult {
        arena: ArenaKind::Bomber,
        player_names: players.iter().map(|p| p.name().to_string()).collect(),
        games,
    }
}
```

---

## T5: `fft/arena_runner.rs` — FFTArenaRunner

### File: `microgpt-rs/src/pruners/fft/arena_runner.rs`

Runs a tournament between FftPlayer implementations in 4v4 battles.

```rust
use std::time::Instant;

use fastrand::Rng;

use crate::pruners::arena::types::*;

use super::battle::BattleState;
use super::players::FftPlayer;
use super::types::*;

/// Configuration for an FFT arena tournament.
pub struct FftArenaConfig {
    /// Number of battles per matchup.
    pub games: usize,
    /// Turn limit per battle.
    pub turn_limit: u32,
}

impl Default for FftArenaConfig {
    fn default() -> Self {
        Self { games: 20, turn_limit: 120 }
    }
}

/// Run a single FFT battle with player strategies assigned to units.
pub fn run_fft_battle(
    party_players: &mut [Box<dyn FftPlayer>],
    enemy_players: &mut [Box<dyn FftPlayer>],
    turn_limit: u32,
    rng: &mut Rng,
) -> GameResult {
    let start = Instant::now();
    // Initialize BattleState with 4v4 units
    // Run turns until one side eliminated or turn_limit
    // Return winner (Party or Enemy) and scores

    GameResult {
        winner: None,
        scores: vec![],
        ticks: 0,
        duration: start.elapsed(),
    }
}

/// Run a full FFT tournament matchup.
pub fn run_fft_matchup(
    party_players: &mut [Box<dyn FftPlayer>],
    enemy_players: &mut [Box<dyn FftPlayer>],
    config: &FftArenaConfig,
) -> MatchupResult {
    let mut rng = Rng::with_seed(42);
    let mut games = Vec::with_capacity(config.games);

    for _ in 0..config.games {
        let result = run_fft_battle(party_players, enemy_players, config.turn_limit, &mut rng);
        games.push(result);
        for p in party_players.iter_mut() { p.reset(); }
        for p in enemy_players.iter_mut() { p.reset(); }
    }

    let mut names: Vec<String> = party_players.iter().map(|p| p.name().to_string()).collect();
    names.extend(enemy_players.iter().map(|p| p.name().to_string()));

    MatchupResult {
        arena: ArenaKind::Fft,
        player_names: names,
        games,
    }
}
```

---

## T6: `bomber_09_rubric_tournament.rs` — Bomber Rubric Tournament Example

### File: `microgpt-rs/examples/bomber_09_rubric_tournament.rs`

```rust
//! Bomberman Rubric Tournament — RubricPlayer vs all baselines (Plan 077).
//!
//! Round-robin tournament pitting RubricPlayer against:
//! Random, Greedy, Validator, HL, GZero
//!
//! Run: `cargo run --example bomber_09_rubric_tournament --features ropd_rubric,g_zero,bomber`
//!
//! Output: per-matchup results, ELO ratings, markdown leaderboard.

use fastrand::Rng;
use microgpt_rs::pruners::bomber::{
    BomberPlayer, GreedyPlayer, HLPlayer, RandomPlayer, RubricPlayer, ValidatorPlayer,
};
use microgpt_rs::pruners::arena::types::*;

#[cfg(feature = "g_zero")]
use microgpt_rs::pruners::bomber::GZeroPlayer;

// ... factory function, matchup loop, leaderboard output ...
```

**Required features**: `ropd_rubric,g_zero,bomber`

### Cargo.toml addition:

```toml
[[example]]
name = "bomber_09_rubric_tournament"
required-features = ["ropd_rubric", "g_zero", "bomber"]
```

---

## T7: `fft_02_rubric_tournament.rs` — FFT Rubric Tournament Example

### File: `microgpt-rs/examples/fft_02_rubric_tournament.rs`

```rust
//! FFT Rubric Tournament — RubricFFTPlayer vs all baselines (Plan 077).
//!
//! 4v4 battles comparing RubricFFT against Random, Greedy, Validator, HL, GZeroFFT.
//!
//! Run: `cargo run --example fft_02_rubric_tournament --features ropd_rubric,g_zero,fft`

use microgpt_rs::pruners::fft::{
    FftPlayer, GreedyFFTPlayer, RubricFFTPlayer, ValidatorFFTPlayer,
};
use microgpt_rs::pruners::arena::types::*;

#[cfg(feature = "g_zero")]
use microgpt_rs::pruners::fft::GZeroFFTPlayer;

// ... factory function, matchup loop, leaderboard output ...
```

**Required features**: `ropd_rubric,g_zero,fft`

### Cargo.toml addition:

```toml
[[example]]
name = "fft_02_rubric_tournament"
required-features = ["ropd_rubric", "g_zero", "fft"]
```

---

## T8: Run Tournaments + Record Results

### File: `.benchmarks/009_arena_integration.md`

Run both tournaments and record:

1. **Bomber Tournament**: Rubric vs Random/Greedy/Validator/HL/GZero
2. **FFT Tournament**: RubricFFT vs Random/Greedy/Validator/HL/GZeroFFT
3. **ELO Ratings**: computed across all matchups
4. **Win Rate Tables**: per-player vs each opponent

Expected result format:

```markdown
## Bomber Arena Results

| Player | vs Random | vs Greedy | vs Validator | vs HL | vs GZero | vs Rubric |
|--------|-----------|-----------|-------------|-------|----------|-----------|
| Random | — | 12% | 5% | 2% | 1% | 3% |
| Greedy | 88% | — | 25% | 15% | 10% | 18% |
| Validator | 95% | 75% | — | 35% | 28% | 30% |
| HL | 98% | 85% | 65% | — | 42% | 45% |
| GZero | 99% | 90% | 72% | 58% | — | 52% |
| Rubric | 97% | 82% | 70% | 55% | 48% | — |

### ELO Ratings
| Rank | Player | ELO |
|------|--------|-----|
| 1 | GZero | 1240 |
| 2 | Rubric | 1215 |
| 3 | HL | 1150 |
| 4 | Validator | 1080 |
| 5 | Greedy | 980 |
| 6 | Random | 850 |
```

### Hypothesis

Per Plan 071's original hypothesis:
- **Bomber (single-axis)**: Rubric ≈ GZero (rubric adds little over scalar δ when survival is the dominant metric)
- **FFT (multi-axis)**: RubricFFT > GZeroFFT (class-dependent rubrics should help when quality has multiple axes)

---

## T9: Update Documentation

### Files to update:
- `README.md` — add Arena Integration section with tournament results
- `docs/overview.md` — add arena module to architecture diagram
- `docs/heuristic-learning.md` — add rubric tournament results to ROPD section

---

## T10: Clippy + Test Pass

```bash
cargo clippy --features ropd_rubric,g_zero,bomber,fft --all-targets
cargo test --features ropd_rubric,g_zero,bomber,fft --quiet
```

---

## Implementation Order

1. **T1** → types.rs (no deps)
2. **T2** → scheduler.rs (depends on T1 types)
3. **T3** → arena/mod.rs + register in pruners/mod.rs (depends on T1, T2)
4. **T4** → bomber/arena_runner.rs (depends on T1, T3)
5. **T5** → fft/arena_runner.rs (depends on T1, T3)
6. **T6** → bomber_09_rubric_tournament.rs (depends on T4)
7. **T7** → fft_02_rubric_tournament.rs (depends on T5)
8. **T8** → run tournaments, record results (depends on T6, T7)
9. **T9** → update docs (depends on T8 results)
10. **T10** → clippy + test pass (final)

---

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Bomber arena_runner complex (bevy_ecs world setup) | Medium | Adapt from existing `bomber_01_arena.rs` pattern |
| FFT battle state needs 8-unit setup | Medium | Adapt from existing `fft_01_arena.rs` pattern |
| RubricPlayer might not learn across games | Low — unit tests pass | Tournament will reveal if `update_outcome` works at scale |
| ELO calculation edge cases | Low | Simple K=32 formula, well-established |

## References

- Plan 071 (ROPD Rubric — RubricPlayer + RubricFFTPlayer)
- `go_02_tournament.rs` — tournament pattern reference
- `bomber_01_arena.rs` — bomber arena runner reference
- `fft_01_arena.rs` — FFT arena runner reference
- `.benchmarks/007_ropd_rubric_modelless.md` — component benchmarks