# katgpt-rs — Documentation

Public MIT engine: a from-scratch Rust GPT-2 style transformer plus a growing
catalog of **modelless inference primitives** — speculative decoding, KV
compression, calibration probes, memory kernels, and heuristic-learning game
arenas. The open sibling of the private `riir-ai` runtime.

> **Read it like a book organized by primitive class.** Each folder below is a
> self-contained context unit — drag the folder into an AI chat and it gets the
> complete narrative for that capability. The folder's `README.md` opens with a
> fusion map showing what combines with what.

## Convention

- **Folders are numbered** (`01_orientation/`, `02_inference/`, … `10_audits/`)
  for sort order. This matches `riir-ai/.docs/`.
- **Files inside have NO number prefix** — add a new doc by dropping `slug.md`
  in the right group folder and adding one line to the relevant group README.
- **Numbers in `.plans/` and `.docs/` live in independent namespaces** — they
  must NEVER share the same number for different topics (the old flat
  `NN_slug.md` scheme collided here, e.g. `191_open_ended_*` doc vs Plan 191).

## Doc groups

### Orientation — "what is this project"

| Folder | What it covers |
|---|---|
| [`01_orientation/`](01_orientation/) | Workspace overview + capability list, full core-architecture reference, paper → feature comparison matrix |

### Inference — the speculative decoding + search engine

| Folder | What it covers |
|---|---|
| [`02_inference/`](02_inference/) | Speculative decoding (DDTree + DFlash + Leviathan verification), SpecHop continuous multi-hop, KV cache compression alternatives, MTP threshold guide, Progressive MCGS graph search |

### Memory — modelless memory primitives

| Folder | What it covers |
|---|---|
| [`03_memory/`](03_memory/) | Raven RSM O(1) routing slot memory, Product Key Memory O(√N) retrieval, Engram hash-addressed conditional memory, MicroRecurrentBeliefState attractor/leaky kernel, NPC Sense Composition, Sleep consolidation at eviction |

### Calibration — probes, gates, and confidence calibration

| Folder | What it covers |
|---|---|
| [`04_calibration/`](04_calibration/) | CCE moderator, CausalHeadImportance scale-normalized fusion, FaithfulnessProbe causal-intervention diagnostic, Salience Tri-Gate per-tick emit gate, sigmoid-not-softmax universality-class escape |

### Adaptation — modelless adaptation & distillation

| Folder | What it covers |
|---|---|
| [`05_adaptation/`](05_adaptation/) | Model adaptation technique survey (LoRA / merge / spectral-quant KV), Lucebox-hub advanced techniques, PEIRA modelless distillation |

### Game Arenas — heuristic-learning proof-of-concept engines

| Folder | What it covers |
|---|---|
| [`06_game_arenas/`](06_game_arenas/) | The HL thesis arenas: Sudoku, Bomberman, Monopoly FSM, FFT Tactics, Go; HL infrastructure + arena detail, open-ended problem-evolution arena, Bomber LoRA A/B artifacts |

### Validator — constraint validation + transformer-VM

| Folder | What it covers |
|---|---|
| [`07_validator/`](07_validator/) | Deterministic constraint validator (Sudoku/Rust-AST pruning), Percepta transformer-VM in Rust |

### Performance — perf engineering

| Folder | What it covers |
|---|---|
| [`08_performance/`](08_performance/) | Throughput tables, SIMD kernels, benchmarks |

### Feature Catalog — the full opt-in / negative-results ledger

| Folder | What it covers |
|---|---|
| [`09_feature_catalog/`](09_feature_catalog/) | Opt-in & gated features (full feature-flag reference), negative results & replaced features |

### Audits — one-off consolidation / rubric audits

| Folder | What it covers |
|---|---|
| [`10_audits/`](10_audits/) | Phase 0.5 loser-sweep audit (Proposal 003), claim-rubric audit vs `Claim` fixtures, cross-repo consolidation audit (riir-ai/riir-chain/riir-neuron-db), doc-drift auditor design record (Issue 702), the silent-zero-pass / alloc-counter / dev-vs-release gate records (Issues 713–716), the Q2 matvec bandwidth-lever record (Issue 937) |

## Sibling repos (private runtimes consume these primitives)

| Repo | Role | Docs location |
|---|---|---|
| [`riir-ai`](../../riir-ai/) | Private SaaS game-AI runtime | `riir-ai/.docs/` (the consolidated selling-point book) |
| [`riir-chain`](../../riir-chain/) | Neuro-symbolic chain lib + daemon | `riir-chain/README.md` (build surface only) |
| [`riir-neuron-db`](../../riir-neuron-db/) | NeuronShard leaf crate | (no `.docs/` today) |
| [`riir-train`](../../riir-train/) | GPU training methods | `.research/` only |

## What does NOT belong in `.docs/`

A `.docs/` file must do at least one of:
1. Document a **shipped primitive** (its API, feature gate, and usage), OR
2. Highlight a **bold outstanding result** (GOAT/gain proven), OR
3. Show **fusion targets** (what this primitive combines WITH to become bigger).

If none → it belongs in `.research/` (academic distillation), `.plans/`
(execution tracking), `.issues/` (open work), or `.benchmarks/` (gate results)
— not `.docs/`. One-off audits live in `10_audits/` rather than the repo root.

## Historical note (2026-07-09 reindex)

This `.docs/` was reindexed from a flat `NN_slug.md` scheme (39 files at the
repo root, with collisions like `191_open_ended_*` doc vs Plan 191) to the
current numbered-folder / unnumbered-file scheme mirroring `riir-ai/.docs/`.
All root `README.md` links, internal `.docs/` cross-references, and
`.benchmarks/` path-bearing links were updated in the same pass. See git
history (`docs:` commit) for the full migration.
