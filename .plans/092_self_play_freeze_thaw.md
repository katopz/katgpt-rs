# Plan 092: Self-Play Freeze/Thaw Knowledge Pipeline

> Self-play → extract knowledge → freeze to disk as `repr(C)` → reload → replay same rounds → measure improvement.

## Tasks

- [x] T1: Define `BomberFrozenBandit` `repr(C)` struct in `bomber/mod.rs`
- [x] T2: Define `GoFrozenBandit` + `GoFrozenTemplates` `repr(C)` structs in `go/types.rs`
- [x] T3: Add `freeze()` / `thaw()` methods to `HLPlayer` (bomber)
- [x] T4: Add `freeze()` / `thaw()` methods to `GZeroPlayer` (bomber)
- [x] T5: Add `freeze()` / `thaw()` methods to `GoHLPlayer`
- [x] T6: Add `freeze()` / `thaw()` methods to `GoGZeroPlayer`
- [x] T7: Add `save_frozen()` / `load_frozen()` helpers in shared `freeze.rs` module
- [x] T8: Create `bomber_12_self_play_freeze.rs` example (100 rounds × 2 phases)
- [x] T9: Create `go_08_self_play_freeze.rs` example (100 rounds × 2 phases)
- [x] T10: Add unit tests for freeze/thaw round-trip (bomber + go)
- [x] T11: Run clippy + existing tests, fix diagnostics
- [x] T12: Update README.md with freeze/thaw section

## Context

The bomber and Go arenas already support self-play with bandit-based learning players (`HLPlayer`, `GZeroPlayer`, `GoHLPlayer`, `GoGZeroPlayer`). These players accumulate knowledge in fixed-size arrays:

- **Bomber HLPlayer**: `q_values: [f32; 7]`, `visits: [u32; 7]`, `compressed: [bool; 7]`, `total_pulls: u32` — ~68 bytes
- **Bomber GZeroPlayer**: Same shape as HLPlayer + template bandit
- **Go GoHLPlayer**: `q_values: [f32; 8]`, `visits: [u32; 8]`, `total_pulls: u32` — via `BanditStats`
- **Go GoGZeroPlayer**: `q_values: [f32; 4]`, `visits: [u32; 4]`, `total_pulls: u32` — via `TemplateStats`

Currently this knowledge **only exists in-memory** and is lost when the process exits. There is no `repr(C)` struct, no serialization, no disk persistence for bandit state.

## Approach

### Phase 1: Define Frozen Knowledge Structs (`repr(C)`)

Small, fixed-size, C-compatible structs that capture only the learned bandit state (not transient game state like bomb positions):

```rust
// bomber/types.rs
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BomberFrozenBandit {
    pub magic: [u8; 4],           // b"BDTB" (Bomber DaTa Bandit)
    pub version: u32,             // 1
    pub q_values: [f32; 7],
    pub visits: [u32; 7],
    pub total_pulls: u32,
    pub compressed: [u8; 7],      // 0=false, 1=true (avoid bool padding issues)
    pub reserved: [u8; 16],       // future-proofing
}
// Total: 4 + 4 + 28 + 28 + 4 + 7 + 16 = 91 bytes, padded to 92

// go/types.rs
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GoFrozenBandit {
    pub magic: [u8; 4],           // b"GODT" (GO DaTa)
    pub version: u32,             // 1
    pub q_values: [f32; 8],
    pub visits: [u32; 8],
    pub total_pulls: u32,
    pub reserved: [u8; 16],
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GoFrozenTemplates {
    pub magic: [u8; 4],           // b"GOTM" (GO TeMplates)
    pub version: u32,             // 1
    pub q_values: [f32; 4],
    pub visits: [u32; 4],
    pub total_pulls: u32,
    pub reserved: [u8; 16],
}
```

### Phase 2: Add `freeze()` / `thaw()` to Players

Each learning player gets:
- `fn freeze(&self) -> FrozenStruct` — extract bandit state
- `fn thaw(frozen: &FrozenStruct) -> Result<Self, String>` — reconstruct player from frozen state (validates magic/version)

The `thaw()` creates a **fresh** player (no transient state) with pre-loaded bandit knowledge. Transient fields (bombs, powerups, round_actions) start empty.

### Phase 3: Disk I/O Helpers

A shared module `src/pruners/freeze.rs` with:
- `fn save_frozen<T>(path: &Path, data: &T) -> Result<(), String>` — raw bytes write
- `fn load_frozen<T>(path: &Path) -> Result<T, String>` — raw bytes read + magic/version check
- Uses `std::fs::write` / `std::fs::read` — zero dependencies, pure `repr(C)` binary.

### Phase 4: Examples

#### `bomber_12_self_play_freeze.rs`

```
Phase 1: LEARN (100 rounds)
  - 4 players: Random, Greedy, Validator, HL
  - Same seed per round (seed + round_index)
  - Track: wins, scores, deaths
  - After 100 rounds: freeze HL player → save to output/bomber_frozen_bandit.bin

Phase 2: REPLAY (same 100 rounds)
  - 4 players: Random, Greedy, Validator, HL(thawed)
  - Same seeds as Phase 1
  - Track: wins, scores, deaths
  - Compare Phase 1 vs Phase 2

Output: comparison table showing improvement
```

#### `go_07_self_play_freeze.rs`

```
Phase 1: LEARN (100 rounds)
  - GoHLPlayer vs GoRandomPlayer (alternating colors)
  - Track: wins, scores
  - After 100 rounds: freeze → save to output/go_frozen_bandit.bin

Phase 2: REPLAY (same 100 rounds)
  - GoHLPlayer(thawed) vs GoRandomPlayer
  - Same seeds
  - Compare
```

### Why 100 Rounds?

10 rounds is too noisy for bandit convergence:
- With 7 bomber actions and ε=0.15, most arms get <10 visits in 10 rounds
- Q-values haven't converged; compressed arms may not have triggered
- 100 rounds gives ~700 action observations, enough for:
  - Clear Q-value separation between good/bad actions
  - Several absorb-compress cycles (threshold=20 visits, 0.1 Q)
  - Statistical significance for win rate comparison

Expected improvement with 100 rounds:
- **Bomber HL**: compressed arms eliminate suicide/bad bomb placement → +15-25% win rate
- **Go GoHLPlayer**: category-level learning → +10-20% win rate vs random (already strong)

## Files to Create

| File | Purpose |
|------|---------|
| `src/pruners/freeze.rs` | Shared `repr(C)` disk I/O helpers |
| `examples/bomber_12_self_play_freeze.rs` | Bomber freeze/thaw demo |
| `examples/go_07_self_play_freeze.rs` | Go freeze/thaw demo |

## Files to Modify

| File | Change |
|------|--------|
| `src/pruners/bomber/types.rs` | Add `BomberFrozenBandit` struct |
| `src/pruners/bomber/mod.rs` | Re-export freeze module |
| `src/pruners/bomber/players.rs` | Add `freeze()`/`thaw()` to `HLPlayer` |
| `src/pruners/bomber/g_zero_player.rs` | Add `freeze()`/`thaw()` to `GZeroPlayer` |
| `src/pruners/go/types.rs` | Add `GoFrozenBandit` + `GoFrozenTemplates` |
| `src/pruners/go/mod.rs` | Re-export freeze module |
| `src/pruners/go/players.rs` | Add `freeze()`/`thaw()` to `GoHLPlayer`, `GoGZeroPlayer` |
| `src/pruners/mod.rs` | Add `freeze` module |
| `tests/test_freeze_thaw.rs` | Round-trip tests |
| `README.md` | Add freeze/thaw section |

## Risks

| Risk | Mitigation |
|------|-----------|
| Bandit state too small for meaningful transfer | 100 rounds should be enough; if not, we increase or add per-template weights |
| `repr(C)` padding differs across platforms | Use `u8` instead of `bool`; add `static_assert_size!` test |
| Same seeds don't guarantee identical games (RNG usage differs) | Players must use provided RNG only; log first 5 rounds to verify determinism |