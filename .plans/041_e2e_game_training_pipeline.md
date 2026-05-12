# Plan 041: E2E Game Training Pipeline — Real Training, Real Data, Real Validation

## Overview

Production-grade end-to-end training pipeline for Bomberman game AI.
No dummy gradients, no toy data, no shortcuts. Real forward pass, real loss,
real backpropagation, real optimizer steps.

### Problem

1. `train_bomber.rs` uses **dummy gradients** (all 0.01) and **placeholder loss**
2. Player type filter mismatch: `"bot"` vs `"Validator"`/`"HL"` → zero real samples loaded
3. No actual training happens → no real LoRA adapter is produced
4. Cannot validate that training improves game play

### Solution

Wire real game replay data through the existing Transformer LoRA training pipeline.
The infrastructure exists (`GpuForwardPass`, `GpuBackwardPass`, `GpuLoss`, `AdamWOptimizer`,
`Trainer::train()`) — it just isn't being used.

### Token Encoding

Board cells and actions share a 10-token vocabulary with no semantic overlap:

| Token | Meaning | Domain |
|-------|---------|--------|
| 0 | Floor | Board |
| 1 | FixedWall | Board |
| 2 | DestructibleWall | Board |
| 3 | PowerUpHidden | Board |
| 4 | Up | Action |
| 5 | Down | Action |
| 6 | Left | Action |
| 7 | Right | Action |
| 8 | Bomb | Action |
| 9 | Wait | Action |

Each game sample becomes a 170-token sequence:
```
tokens = [board[0], board[1], ..., board[168], action + BOARD_VOCAB]
```
- Input: `tokens[0..169]` = board cells
- Target: `tokens[1..170]` = shifted board cells + action token
- Position 168: model predicts action given full board context via causal attention

### Model: `Config::game()`

Tiny Transformer tuned for board-game action prediction:

| Param | Value | Why |
|-------|-------|-----|
| vocab_size | 10 | 4 board cells + 6 actions |
| block_size | 170 | 169 board positions + 1 action |
| n_embd | 32 | Small but sufficient for 13×13 board |
| n_head | 4 | Standard for tiny models |
| head_dim | 8 | 32 / 4 = 8 |
| mlp_hidden | 128 | 4× expansion |
| n_layer | 1 | Single layer — model is ~18K params |
| n_kv_head | 4 | Match n_head (no GQA) |

~18K base params, ~1.5K LoRA params (rank=4). Trains in seconds.

### Scope

- **In scope**: Real training pipeline, real data flow, real loss curves
- **In scope**: E2E run demonstrating real training
- **Deferred**: Loss masking (only compute loss at action position)
- **Deferred**: Quality-weighted loss
- **Deferred**: NNPlayer inference integration (separate plan)

## Architecture

```
┌─────────────────────┐     ┌──────────────────────┐     ┌──────────────────┐
│ bomber_04_replay_gen │────▶│ game::GameTrainer    │────▶│ game_lora.bin    │
│ (Plan 039, done)     │     │ ├─ encode_samples()  │     │ training_report  │
│                      │     │ ├─ Trainer::train()  │     │ .json            │
│ JSONL: board,action, │     │ └─ export + report   │     │ domain_latent    │
│ quality,player_type  │     │                      │     │ .dlat            │
└─────────────────────┘     │ Uses: Config::game() │     └──────────────────┘
                             │ Uses: real forward/  │
                             │ backward/loss/optim  │
                             └──────────────────────┘
```

Data flow:
1. `bomber_04_replay_gen` → JSONL files (real game traces from 1000 rounds)
2. `parse_jsonl_dir()` → `Vec<GameSample>` (filtered: quality > 0.5, Validator/HL only)
3. `encode_game_samples()` → `Vec<TrainingSample>` (board+action → token sequences)
4. `Trainer::train()` → real forward/backward/loss/optimizer on GPU
5. Export: `game_lora.bin` + `training_report.json` + `game_domain_latent.dlat`

## Tasks

- [x] **Task 1: `Config::game()` in microgpt-rs** ✅
  - Add `Config::game()` to `microgpt-rs/src/types.rs`
  - vocab=10, block=170, n_embd=32, n_head=4, head_dim=8, mlp_hidden=128, n_layer=1
  - Include LoRA defaults: rank=4, alpha=8.0, targets=[q,k,v,o,mlp1,mlp2]
  - Tests: verify param counts, verify block_size >= game_seq_len

- [x] **Task 2: Game data adapter in riir-gpu** ✅
  - Add `riir-gpu/src/game/trainer.rs` with:
    - `const BOARD_VOCAB: usize = 4` (board cell types)
    - `const ACTION_OFFSET: usize = 4` (action tokens start at 4)
    - `fn encode_game_sample(sample: &GameSample) -> TrainingSample`
    - `fn encode_game_samples(samples: &[GameSample]) -> Vec<TrainingSample>`
    - `fn decode_action_token(token: usize) -> Option<GameAction>`
  - Export from `game/mod.rs` and `lib.rs`
  - Tests: encode/decode roundtrip, correct token offsets, sequence length

- [x] **Task 3: Fix player_type filter** ✅
  - In `train_bomber.rs`: change `&["bot"]` → `&["Validator", "HL"]`
  - Verify real samples load from JSONL (no more toy data fallback)

- [x] **Task 4: Rewrite `train_bomber.rs` with real Trainer** ✅
  - Remove dummy gradient code entirely
  - Remove `generate_toy_samples()` — fail if no replay data
  - Use `Config::game()` instead of `Config::micro_lora()`
  - Use `encode_game_samples()` to convert replay data
  - Use `Trainer::train()` for real forward/backward/loss/optimizer
  - Keep: BetaConfig, ReviewMetrics, GameTrainingReport, domain latent training
  - Keep: JSON report export, LoRA export
  - Remove: manual GpuLoraBuffers/GpuDomainLatent/GpuPipelines creation (Trainer handles this)
  - CLI: `--beta` flag (keep), `--replay-dir` flag (new, override default path)
  - Error if no replay data found (don't silently use toy data)

- [ ] **Task 5: E2E run + validation**
  - Run `cargo run --example bomber_04_replay_gen --features bomber`
  - Verify JSONL output has samples with player_type "Validator" or "HL"
  - Run `cargo run --example train_bomber -- --beta 0.3`
  - Verify training report shows real loss (monotonically decreasing, not placeholder)
  - Verify `game_lora.bin` is non-trivial (not all zeros)
  - Document results in `.docs/11_e2e_training_results.md`

## File Change Summary

### New files

| File | Lines | Purpose | Target |
|------|-------|---------|--------|
| `riir-gpu/src/game/trainer.rs` | ~80 | Game data encoding + training constants | riir-ai |
| `riir-ai/.docs/11_e2e_training_results.md` | ~50 | E2E validation results | riir-ai |

### Modified files

| File | Change | Target |
|------|--------|--------|
| `microgpt-rs/src/types.rs` | Add `Config::game()` (~20 lines) | microgpt-rs |
| `riir-gpu/src/game/mod.rs` | Add `pub mod trainer;` | riir-ai |
| `riir-gpu/src/lib.rs` | Export game trainer types | riir-ai |
| `riir-gpu/examples/train_bomber.rs` | Rewrite: real Trainer, no dummy grads (~200→150 lines) | riir-ai |

### Removed code

| What | Why |
|------|-----|
| `generate_toy_samples()` | No toy data. Fail if no real data. |
| Dummy gradient writes | Replaced by real Trainer. |
| Manual GpuLoraBuffers creation | Trainer handles this. |
| Manual GpuPipelines/GpuDomainLatent | Trainer handles this. |
| Placeholder loss computation | Real loss from Trainer. |

## Design Decisions

### 1. Transformer for Board Game Classification

The existing GPU pipeline (`GpuForwardPass`, `GpuBackwardPass`, `GpuLoss`, `AdamWOptimizer`)
is Transformer-based and thoroughly tested (60 tests passing). Building new GPU kernels
for an MLP is a separate project. Using the Transformer with `Config::game()` leverages
proven infrastructure. The model is tiny (~18K params) — overhead is negligible.

### 2. Board-as-Token-Sequence Encoding

Board cells become tokens in a causal sequence. Position 168 attends to all 169 cells
via causal attention — this correctly captures the full board state for action prediction.

Positions 0-167 produce auxiliary next-cell-prediction loss. This is harmless (the model
quickly learns the deterministic board patterns) and may help as regularization.
Future improvement: mask loss to only position 168.

### 3. Separate Vocab Spaces (No Overlap)

Board tokens (0-3) and action tokens (4-9) occupy distinct ranges. This avoids semantic
ambiguity — the model can learn that tokens 0-3 represent spatial information and tokens
4-9 represent action choices. The offset makes the encoding self-documenting.

### 4. Fail Fast — No Silent Fallback

If no replay data exists, `train_bomber.rs` should error with a clear message:
"Run bomber_04_replay_gen first." Silent fallback to toy data hides pipeline failures.
Production pipelines fail loudly.

### 5. Quality Filtering over Quality Weighting

Samples with quality < 0.5 are filtered (already implemented in `parse_jsonl_filtered`).
Quality-weighted loss requires modifying GPU loss kernels — deferred to future plan.
Filtering is sufficient for initial training.

### 6. Trainer Handles Infrastructure

The existing `Trainer::train()` manages `GpuForwardPass`, `GpuBackwardPass`, `GpuLoss`,
`AdamWOptimizer`, and `GpuLoraBuffers` internally. The example should use `Trainer`,
not manually create these. This reduces code and ensures consistency.

## Priority Order

| Priority | Task | Why | Effort |
|----------|------|-----|--------|
| P0 | Task 3: Fix filter | Unblock data flow — 1 line change | Trivial |
| P0 | Task 1: Config::game() | Required for training | Small |
| P0 | Task 2: Data adapter | Required for training | Small |
| P1 | Task 4: Rewrite trainer | Core implementation | Medium |
| P2 | Task 5: E2E validation | Prove it works | Small |

## Expected Outcomes

1. Real trained LoRA adapter (~1.5K params) that learned from game replay data
2. Training report with real loss curves (monotonically decreasing)
3. Documented E2E pipeline: `bomber_04_replay_gen` → `train_bomber` → `game_lora.bin`
4. Clear path to NNPlayer inference integration (future plan)

## Connection to Existing Plans

| Plan | Relationship |
|------|-------------|
| Plan 039 | Replay data generation (prerequisite — done) |
| Plan 040 | Cross-training config (BetaConfig, ReviewMetrics — partially done) |
| Plan 034 | Original training scaffolding (Trainer, GpuForwardPass, etc. — done) |
| Plan 038 | Domain latent (used alongside LoRA — done) |
| Future | NNPlayer inference: load game LoRA, encode board, select action |

## Research Citations

This plan is grounded in existing proven infrastructure rather than new research.
The techniques used (LoRA fine-tuning, causal attention, cross-entropy loss) are
standard. The encoding scheme (board-as-token-sequence) follows the same pattern
as vision transformers that flatten image patches into token sequences.