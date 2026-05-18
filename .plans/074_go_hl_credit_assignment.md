# Plan 074: Go HL Credit Assignment — Modelless Reward Shaping

**Branch:** `develop/feature/074_go_hl_credit_assignment`
**Depends on:** Plan 065 (Go players), Plan 073 (territorial heuristic)
**Research:** Research 37 (REAP model-based/modelless duality)
**Model-based Twin:** Plans 072-074 (ROPD/SDAR/Interventional SFT for LoRA training)
**Goal:** Fix GoHLPlayer's credit assignment — currently only the last move's category gets win/loss reward (1 update per 302-move game = 0.3% signal ratio). Adapt Bomber HLPlayer's decay-based credit assignment pattern to distribute reward across ALL categories used, with recency weighting.

**Key Insight:** The Bomber HLPlayer already solved this problem (see `bomber/players.rs:1313-1350`). It distributes reward across ALL actions with exponential decay recency weighting. We adapt the same pattern for Go's 8 move categories.

**Why This Is Different from Plan 054:** Plan 054 found "NO GAIN" for DDTree reward shaping (16-step paths). Go's credit assignment is 100× worse — 302 moves, only last gets credit. This is a much lower bar to clear.

**REAP Duality Alignment:**
- Modelless (this plan): richer reward signals for bandit Q-values
- Model-based (Plans 072-074): gradient-based reward shaping via GRPO/SDAR/SFT
- Both paths needed per REAP spectrum (Research 37)

---

## Tasks

### Phase 0: Benchmark Baseline (DONE)

- [x] **T0: Run baseline tests** — `cargo test -p microgpt-rs --features go --lib` → 699 passed

### Phase 1: Category Trace + Recency-Weighted Credit

Adapt Bomber HLPlayer's decay-based credit assignment to Go.

- [x] **T1: Add `category_trace` field to `GoHLPlayer`**
  - Replace `last_category: Option<GoMoveCategory>` with `category_trace: Vec<GoMoveCategory>`
  - Push each move's category in `select_move()`
  - Clear in `reset()`

- [x] **T2: Rewrite `update_outcome()` with recency-weighted credit**
  - Distribute win/loss reward across ALL categories in trace
  - Exponential decay recency: `recency = 0.5^((total - 1 - i) / half_life)`
  - Use `half_life` tuned for Go's ~302 moves (half_life=50)
  - Aggregate per-category weighted rewards, then update bandit

- [x] **T3: Update `reset()` to clear trace**

- [x] **T4: Update existing tests for new behavior**
  - `hl_player_selects_and_tracks_category` → check trace not empty
  - `hl_player_update_outcome` → verify all categories in trace get updated
  - Add `hl_player_credit_assignment_distributes_across_trace` test

### Phase 2: Per-Move Reward Shaping

Intermediate rewards between moves (not just game-end binary win/loss).

- [ ] **T5: Add per-move reward computation in `select_move()`**
  - Compute heuristic score delta (before/after move) as intermediate reward
  - Store alongside category in trace: `Vec<(GoMoveCategory, f32)>`
  - Per-move reward = normalized heuristic delta (captures, territory change)

- [ ] **T6: Blend per-move and game-end rewards in `update_outcome()`**
  - Per-move reward: immediate signal (captures, territory)
  - Game-end reward: win/loss binary
  - Blend: `final_reward = α * per_move + (1-α) * game_end`
  - Start with α=0.3 (per-move is supplementary, game-end is primary)

- [ ] **T7: Add test for per-move reward shaping**

### Phase 3: Benchmark + Validation

- [x] **T8: Run all Go tests** — `cargo test -p microgpt-rs --features go --lib` → 700 passed
- [ ] **T9: Run TUI** — verify Q-values differentiate in scoreboard
- [ ] **T10: Run tournament** — verify win rate doesn't regress (100% vs Random)
- [x] **T11: Run clippy** — `cargo clippy --features go --quiet` → 0 warnings

---

## Success Criteria

1. All 699+ Go tests pass
2. Q-values differentiate within first few games (not stuck at 0.00)
3. Win rate vs Random ≥ 100% (no regression)
4. TUI shows meaningful learning curves with non-zero Q-values

## Failure Mode

If credit assignment degrades win rate:
1. Reduce α (per-move reward weight)
2. Increase half_life (spread credit more evenly)
3. Fallback: only enable for display, revert to last-category for actual Q-updates

## Hyperparameters

| Parameter | Default | Range | Effect |
|---|---|---|---|
| half_life | 50 | [20, 100] | Recency decay speed (moves) |
| α (per-move weight) | 0.3 | [0.0, 0.5] | How much per-move signal vs game-end |
| ε decay | 0.995 | [0.99, 0.999] | Exploration decay per game |

## Relationship to Other Plans

| Plan | Relationship |
|------|-------------|
| Plan 072 (ROPD Model-Based) | Model-based twin — rubric rewards for GRPO |
| Plan 073 (SDAR Model-Based) | Model-based twin — gated distillation loss |
| Plan 074 (Interventional SFT) | Model-based twin — token masking |
| Plan 054 (StepCodeReasoner) | Precedent — "NO GAIN" for DDTree, but different domain |
| Plan 065 (AutoGo) | Foundation — GoHLPlayer originally implemented here |
| Research 37 (REAP) | Justification — modelless/modelless duality |