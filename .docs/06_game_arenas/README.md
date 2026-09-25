# Game Arenas — Heuristic-Learning Proof-of-Concept Engines

> **What we sell here.** The HL (heuristic-learning) thesis, validated across
> five self-contained game arenas: a frozen model with no per-game weights
> learns to play competitively via modelless heuristic evolution. Each arena is
> a runnable, measurable proof.

## Fusion map — the arenas + their shared substrate

```
   heuristic_learning.md (the HL thesis + shared infrastructure)
        │
        ├── sudoku.md (constraint-pruning TUI arena)
        ├── bomber_arena.md (4-player Bomberman)
        │     └── bomber_lora_ab.md (Plan 045 tech-isolation A/B artifacts)
        ├── monopoly_fsm.md (4-player board-game FSM)
        ├── fft_arena.md (4v4 ATB tactics)
        ├── go_arena.md (AI vs AI auto-play)
        └── open_ended_evolution.md (Plan 191 problem-evolution arena)
              │
              ▼
        hl_arena_detail.md (cross-arena detail + the HL thesis verdict)
```

| Doc | Role |
|---|---|
| [`heuristic_learning.md`](heuristic_learning.md) | HL infrastructure, FFT benchmarks, the shared arena harness |
| [`sudoku.md`](sudoku.md) | Sudoku domain — solver, constraint pruning, TUI |
| [`bomber_arena.md`](bomber_arena.md) | Bomberman HL Arena — 4-player heuristic-learning proof |
| [`bomber_lora_ab.md`](bomber_lora_ab.md) | Plan 045 Bomber tech-isolation A/B — LoRA artifacts |
| [`monopoly_fsm.md`](monopoly_fsm.md) | Monopoly FSM Arena — 4-player board-game engine |
| [`fft_arena.md`](fft_arena.md) | FFT Arena — 4v4 ATB tactics battle engine |
| [`go_arena.md`](go_arena.md) | Go Arena — AI vs AI auto-play engine (Plan 065) |
| [`moka_head_to_head.md`](moka_head_to_head.md) | Moka v1 head-to-head — complete record (Plan 563 + 565, Bench 204 + 205) |
| [`open_ended_evolution.md`](open_ended_evolution.md) | Plan 191 — open-ended problem-evolution arena |
| [`tetris_sim_fidelity_boundary.md`](tetris_sim_fidelity_boundary.md) | Tetris sim boundary rule — a placement-enumeration oracle, not an engine; new fidelity = new lane (ex-Issue 878) |
| [`hl_arena_detail.md`](hl_arena_detail.md) | Cross-cutting HL & arena detail + thesis verdict |

## See also

- [`../07_validator/constraint_validator.md`](../07_validator/constraint_validator.md) — the constraint validation the Sudoku arena exercises
