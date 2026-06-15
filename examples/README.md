# Examples

All examples run with `cargo run --example <name>`. Most require feature flags — the
exact flag is listed per example and in the catalog below. Examples that gate
themselves internally (`#![cfg(feature = ...)]`) print a hint and exit cleanly if the
flag is missing.

**178 examples** across 25 groups. Full feature definitions live in
[`Cargo.toml`](../Cargo.toml) and the [README Feature Flags](../README.md#feature-flags) section.

## Catalog

| Example | Group | Feature(s) | What it shows |
|---------|-------|-----------|---------------|
| `bandit_01_basic` | Bandit | `bandit` | UCB1, ε-greedy (decay), Thompson Sampling on a 5-arm Bernoulli bandit |
| `bandit_02_ddtree` | Bandit | `bandit` | Model-based vs modelless speculative decoding with bandit arm selection |
| `bandit_03_slot` | Bandit | `bandit` | Rules-based speculative decoding via DDTree + BanditPruner |
| `bandit_04_combat` | Bandit | `bandit` | "Smart-ass modelless" monster AI — bandit learns to adapt behavior |
| `bandit_05_rps` | Bandit | `bandit` | Rock-Paper-Scissors — bandits converge to Nash equilibrium |
| `bandit_06_resolver` | Bandit | `bandit` | Domain validator + bandit + DDTree endgame with action masking |
| `bandit_07_director` | Bandit | `bandit` | L4D-style AI director — meta-bandit drives encounter pacing |
| `bandit_08_safe_phased` | Bandit | `safe_bandit` | PrudentBanker delay-calibrated safe exploration (Plan 137) |
| `hl_01_trial_log` | HL | `bandit` | Trial log + absorb-compress with a Bernoulli env |
| `hl_02_hotswap` | HL | `bandit` | Runtime pruner hot-swap + regression suite, no restart |
| `bomber_01_arena` | Bomber | `bomber` | Headless N-round 4-player tournament, progressive HL tiers |
| `bomber_02_tui` | Bomber | `bomber` | Animated ratatui replay (Space/←/→/F/A/Q) |
| `bomber_03_hl_proof` | Bomber | `bomber` | 1000-round HL proof — win/survival rate + bandit Q-values |
| `bomber_04_nn` | Bomber | `bomber-wasm` | NNPlayer with WASM validator safety checks vs native fallback |
| `bomber_05_replay_gen` | Bomber | `bomber` | Replay generator — P3/P4 winning episodes → JSONL |
| `bomber_06_replay_gen_v2` | Bomber | `bomber` | Balanced replay gen (all players, enriched features) for LoRA training |
| `bomber_07_bomb_types` | Bomber | `bomber` | Timed / Piercing / Remote / Landmine bomb behavior demo |
| `bomber_08_agent_loop` | Bomber | `bomber-agent` | Agent validator loop — TemplateProposer → evaluate → iterate |
| `bomber_09_rubric_tournament` | Bomber | `ropd_rubric,g_zero,bomber` | RubricPlayer (ROPD) vs full hierarchy (Plan 077) |
| `bomber_10_sdar_tournament` | Bomber | `sdar_gate,ropd_rubric,g_zero,bomber` | SdarPlayer vs all baselines (Plan 072) |
| `bomber_11_bt_rank_tournament` | Bomber | `bt_rank,g_zero,bomber` | Bradley-Terry pairwise ranking of player strategies |
| `bomber_12_self_play_freeze` | Bomber | `bomber` | Freeze/thaw knowledge pipeline across phases (Plan 092) |
| `bomber_13_reflection_qa` | Bomber | `memo_reflections` | MeMo 5-step reflection QA on bomber game data (Plan 094) |
| `bomber_14_sr2am_tournament` | Bomber | `sr2am_configurator,bomber` | Sr2amPlayer vs all baselines |
| `bomber_15_vpd_tournament` | Bomber | `vpd_em_distill,g_zero,bomber` | VpdPlayer vs SDAR/GZero/Random (Plan 120) |
| `bomber_16_rmsd_tournament` | Bomber | `rmsd_distill,vpd_em_distill,g_zero,bomber` | RmsdPlayer vs VPD/SDAR/GZero/Random (Plan 125) |
| `bomber_17_feedback_goat` | Bomber | `sia_feedback,g_zero,bomber` | FeedbackBandit 1000-round regression proof (Plan 178) |
| `bomber_18_sdpg_tournament` | Bomber | `sdpg_bandit,sdar_gate,ropd_rubric,g_zero,bomber` | SdpgPlayer vs all baselines (Plan 180) |
| `monopoly_01_arena` | Monopoly | `monopoly` | Headless N-game 4-player tournament |
| `monopoly_02_tui` | Monopoly | `monopoly` | Animated ratatui replay with walk effect |
| `monopoly_03_hl_proof` | Monopoly | `monopoly` | 1000-game HL proof — 56.5% win, +41.3pp over Validator |
| `monopoly_04_bench` | Monopoly | `monopoly` | Throughput + per-turn latency (p50/p90/p99) |
| `fft_01_arena` | FFT | — | 4v4 ATB tactics arena, 4 AI tiers (Plan 047) |
| `fft_02_rubric_tournament` | FFT | `ropd_rubric,g_zero,fft` | RubricFFTPlayer vs all baselines (Plan 077) |
| `fft_03_sdar_tournament` | FFT | `sdar_gate,ropd_rubric,g_zero,fft` | SDAR vs all baselines incl. Rubric (Plan 072) |
| `fft_04_feedback_goat` | FFT | `g_zero,fft` | 1000-round baseline regression proof (Plan 178) |
| `game_state_01_bomber_mcts` | GameState | `game_state` | Generic MCTS on BomberState snapshot (Plan 056) |
| `game_state_02_bomber_gvg` | GameState | `game_state` | 2v2 GvG MCTS showcase (Plan 058) |
| `bear_01_demo` | Blue Bear | — | DDTree + ConstraintPruner tactical puzzle solver |
| `bear_02_tui` | Blue Bear | — | Animated step-through TUI solver |
| `core_01_validator` | Core | `validator` | BPE → DDTree → SynPruner compiler-in-the-loop pipeline |
| `core_02_raven` | Core | — | Raven RSM recall + O(1) scaling + memory footprint |
| `core_03_ppot` | Core | `ppot` | PPoT logit-parameterized CPU resampling |
| `core_04_prefill` | Core | — | Bidirectional prefill + modality LoRA switching (Plan 025) |
| `core_05_maxsim` | Core | `maxsim` | MaxSim late-interaction scoring integration points (Plan 080) |
| `core_06_peira` | Core | `peira_distill` | PEIRA inter-view modelless distillation |
| `dungeon_01_tui` | Dungeon | — | Multi-floor animated dungeon solver TUI |
| `dungeon_02_multifloor` | Dungeon | — | DDTree (strategic) + multi-floor A* (tactical) |
| `sudoku_01_9x9` | Sudoku | `sudoku` | Streaming "thinking" 9×9 solver, symbolic validator |
| `sudoku_02_speculative` | Sudoku | `sudoku` | DDTree + validator pruning, 3-level comparison |
| `sudoku_03_tui` | Sudoku | `sudoku` | Real-time TUI visualization, two tabs |
| `sudoku_04_percepta_vs` | Sudoku | — | Rust hull attention vs Python+C++ Percepta (unfair speed comp) |
| `tactical_01_ai` | Tactical | — | DDTree (strategic) + A* (tactical) hierarchical AI |
| `tactical_02_terrain` | Tactical | — | Terrain-weighted pathfinding |
| `tactical_03_procedural` | Tactical | — | Procedural map generation + solving |
| `tactical_04_parallel` | Tactical | — | Parallel batch solving with rayon |
| `tactical_05_bench` | Tactical | — | Strategic vs brute-force DDTree benchmark |
| `tactical_06_tui` | Tactical | — | Interactive 16×16 hierarchical-solver TUI |
| `tactical_07_strategic` | Tactical | — | Multi-layer constraint puzzle (boss/traps/keys/levers/bridge) |
| `tactical_08_headless` | Tactical | — | Headless three-round strategic puzzle benchmark |
| `tactical_09_fog` | Tactical | — | Fog-of-war puzzle TUI, three exploration strategies |
| `tactical_10_fog_bench` | Tactical | — | Headless fog-of-war exploration strategy benchmark |
| `review_01_metrics` | Review | `bandit` | Inference-time review metrics — fix vs break ratio (Plan 036) |
| `go_00_api_bridge` | Go | `go` | Play random games against AutoGo via REST bridge |
| `go_01_mcts` | Go | `go` | MCTS vs Random on 9×9 |
| `go_02_tournament` | Go | `go` | Player-type round-robin on 9×9 |
| `go_03_head_to_head` | Go | `go` | Head-to-head vs external AutoGo agents (REST) |
| `go_04_gzero` | Go | `go` | G-Zero self-play with per-round δ tracking |
| `go_05_autoresearch` | Go | `go` | AutoResearch loop — automated hyperparameter search |
| `go_06_bench` | Go | `go` | GoState/MCTS throughput + player scaling laws |
| `go_07_tui` | Go | `go` | AI-vs-AI auto-play replay TUI |
| `go_08_self_play_freeze` | Go | `go` | GoHLPlayer freeze/thaw knowledge pipeline (Plan 092) |
| `go_09_reflection_qa` | Go | `memo_reflections,go` | MeMo 5-step reflection QA on Go data (Plan 094) |
| `cna_01_discovery` | CNA | `cna_steering` | Discover contrastive neuron circuits from activations |
| `cna_02_steering` | CNA | `cna_steering` | Runtime activation modulation with discovered circuits |
| `cna_03_go_circuit` | CNA | `cna_steering,go` | End-to-end circuit discovery from Go games |
| `stepcode_01_shaped_bandit` | StepCode | `stepcode` | Intra-trajectory reward shaping (Plan 054) |

| `and_or_demo` | AND-OR | `and_or_dtree` | `AndOrNode` tree construction & metrics walkthrough |
| `and_or_sudoku` | AND-OR | `and_or_dtree` | Rows-as-AND-subgoals decomposition on a 4×4 Sudoku |
| `ega_01_quality` | EGA | `ega_attn` | Energy-gated attention val-loss ablation (Plan 139) |
| `ega_02_energy_profile` | EGA | `ega_attn` | Energy-gate distribution profile |
| `ega_03_eviction` | EGA | `ega_attn` | Energy-threshold KV cache eviction |
| `ega_04_combined` | EGA | `ega_attn` | EGA + DashAttn + SdpaOutputGate combined pipeline |
| `cache_prune_01_sat_bench` | CachePrune | `cache_prune` | SAT build/query vs naive-scan benchmark (Plan 140) |
| `cache_prune_02_segment_match` | CachePrune | `cache_prune` | Rolling-hash segment matching demo |
| `rt_turbo_01_calibration` | RT-Turbo | `rt_turbo` | RTTurbo retrieval-head calibration (Plan 126) |
| `rt_turbo_02_decode_bench` | RT-Turbo | `rt_turbo` | RTTurbo sparse-decode benchmark |
| `spechop_01_pipeline` | SpecHop | `spechop` | 4-hop continuous speculation pipeline (Plan 131) |
| `spechop_02_cost_model` | SpecHop | `spechop` | α/β/p → k* and RelLat cost-model prediction |
| `kvarn_goat_proof` | KVarN | `kvarn` | KVarN variance-normalized KV-quant GOAT proof (Research 159) |
| `kvarn_thinking_demo` | KVarN | `kvarn,thinking_cot` | KV-quant quality during extended thinking sequences |
| `octpq_kvarn_fusion` | KVarN | `kvarn,hybrid_oct_pq` | Six-pipeline quantization comparison on 128×128 tiles |
| `ruliology_demo` | Ruliology | `ruliology` | Wolfram-style full enumeration + ranking pipeline (Plan 188) |
| `skill_lifecycle_demo` | Skill Lifecycle | `skill_lifecycle` | MUSE learn → validate → register → evolve flow (Plan 192) |
| `partial_scoring_demo` | Problem Evolution | `partial_scoring` | Binary vs graduated reward learning curves (Plan 191) |
| `problem_evolution_demo` | Problem Evolution | `problem_mutator` | EvolutionArena with Bomber/Go config mutators (Plan 191) |
| `idea_divergence_demo` | Problem Evolution | `idea_divergence` | Collapse prevention via novelty filter (Plan 191) |
| `directional_credit_demo` | Misc | `directional_credit` | Entropy-bifurcated direction-adaptive screening (Plan 184) |
| `kv_share_demo` | Misc | `kv_share` | Q-K=V projection sharing — KV cache halving (Plan 185) |
| `stiff_anomaly_demo` | Misc | `stiff_anomaly` | Stiff/soft subspace eigenvalue anomaly gate (Plan 138) |
| `randopt_01_basic` | Misc | `randopt_weight` | Synthetic weight-perturbation ensembling demo (Plan 121) |
| `datrie_01_bench` | Misc | `datrie_vocab` | HashMap vs double-array trie vocab benchmark (Research 137) |
| `spec_reconciliation_demo` | Misc | `spec_reconciliation` | Verify offline trajectories vs plausibility manifolds (Plan 177) |
| `percepta_phase0` | Misc | `percepta_compile` | Transformer-VM-in-the-browser feasibility proof (Plan 064) |
| `thinking_cot_demo` | Misc | `thinking_cot` | Adaptive CoT thinking vs non-thinking quality (Plan 194) |
| `mux_latent_compress` | MUX-Latent | `mux_latent_context` | Compress 4k tokens at X4/X8/X16, show KV savings, TTFT, adaptive LOD (Plan 238) |
| `mux_latent_expand` | MUX-Latent | `mux_latent_context` | Compress then selectively expand segments, query-based retrieval (Plan 238) |
| `dec_terrain_bench` | DEC Terrain | `dec_terrain_ai` | Dynamic topology update perf — `remove_face` + `recompute_if_dirty` (Plan 261) |
| `dec_terrain_quality_bench` | DEC Terrain | `dec_terrain_ai` | Hodge-decomposed route quality vs A* on modified terrain (Plan 261 T46–47) |
| `cgsp_minimal` | CGSP | `cgsp` | Minimal end-to-end CGSP loop: 8-direction pool + 1 target + 100 cycles + snapshot/BLAKE3 (Plan 274 Phase 4 T4.3) |
| `cgsp_collapse_recovery` | CGSP | `cgsp` | Force one-hot priority table → measure cycles to recover (1 cycle vs 200+ baseline) — the G2 proof (Plan 274 Phase 4 T4.4) |

---

## 1. Bandit (RL / Game Theory)

Multi-armed bandit strategies for adaptive decision-making under uncertainty.

```bash
cargo run --example bandit_01_basic --features bandit
cargo run --example bandit_02_ddtree --features bandit
cargo run --example bandit_03_slot --features bandit
cargo run --example bandit_04_combat --features bandit
cargo run --example bandit_05_rps --features bandit
cargo run --example bandit_06_resolver --features bandit
cargo run --example bandit_07_director --features bandit
cargo run --example bandit_08_safe_phased --features safe_bandit
```

## 2. Heuristic Learning (HL)

Trial logging and hot-swapping for the HL infrastructure.

```bash
cargo run --example hl_01_trial_log --features bandit
cargo run --example hl_02_hotswap --features bandit
```

## 3. Bomberman HL Arena

4-player Bomberman with `bevy_ecs` standalone — tick-based priority FSM, 4 AI tiers,
WASM-validated NN player, replay generation, and a ladder of distillation tournaments.

```bash
cargo run --example bomber_01_arena --features bomber
cargo run --example bomber_02_tui --features bomber                 # Space/←/→/F/A/Q
cargo run --example bomber_03_hl_proof --features bomber
cargo run --example bomber_04_nn --features bomber-wasm -- /path/to/bomber_validator.wasm
cargo run --example bomber_05_replay_gen --features bomber
cargo run --example bomber_06_replay_gen_v2 --features bomber
cargo run --example bomber_07_bomb_types --features bomber
cargo run --example bomber_08_agent_loop --features bomber-agent
cargo run --example bomber_09_rubric_tournament --features "ropd_rubric,g_zero,bomber"
cargo run --example bomber_10_sdar_tournament --features "sdar_gate,ropd_rubric,g_zero,bomber"
cargo run --example bomber_11_bt_rank_tournament --features "bt_rank,g_zero,bomber"
cargo run --example bomber_12_self_play_freeze --features bomber
cargo run --example bomber_13_reflection_qa --features memo_reflections
cargo run --example bomber_14_sr2am_tournament --features "sr2am_configurator,bomber"
cargo run --example bomber_15_vpd_tournament --features "vpd_em_distill,g_zero,bomber"
cargo run --example bomber_16_rmsd_tournament --features "rmsd_distill,vpd_em_distill,g_zero,bomber"
cargo run --example bomber_17_feedback_goat --features "sia_feedback,g_zero,bomber" --release
cargo run --example bomber_18_sdpg_tournament --features "sdpg_bandit,sdar_gate,ropd_rubric,g_zero,bomber"
```

**`bomber_09` results:** Random wins most (12.0%, 18W) — 4-player FFA has ~80% draws;
Bomber is single-axis (survival), so rubric/GZero add little. **`bomber_17`:** ❌ regression
— 6-arm UCB1 (18.6%) dilutes PlanNew convergence vs 4-arm baseline (29.0%).

## 4. Monopoly FSM Arena

4-player Monopoly with `bevy_ecs` standalone — turn-based event-driven FSM, 40-square board.

```bash
cargo run --example monopoly_01_arena --features monopoly
cargo run --example monopoly_02_tui --features monopoly             # Space/←/→/F/A/Home/End/Q
cargo run --example monopoly_03_hl_proof --features monopoly
cargo run --example monopoly_04_bench --features monopoly
```

**`monopoly_03` results:** HL 56.5% win, 93.7% survival, +41.3pp over Validator —
✅ HL thesis proven. **`monopoly_04`:** ~84.8 games/sec, 41µs/turn.

## 5. FFT Tactics Arena

Final Fantasy Tactics-inspired 4v4 ATB battle — data-driven, speed-based turn queue,
4 classes (Knight/Archer/Black Mage/White Mage), 4 AI tiers.

```bash
cargo run --example fft_01_arena
cargo run --example fft_02_rubric_tournament --features "ropd_rubric,g_zero,fft"
cargo run --example fft_03_sdar_tournament --features "sdar_gate,ropd_rubric,g_zero,fft"
cargo run --example fft_04_feedback_goat --features "g_zero,fft" --release
```

**`fft_02` results:** GZero champion (ELO 1185, 60% win). Multi-axis quality (kills/
survival/healing) gives rubrics more signal than single-axis bomber. **`fft_04`:** ✅ pass
— Greedy 63.3% > HL 33.7% > GZero 13.7% > Validator 11.9%, no degenerate dominance.

## 6. GameState Forward Model (STRATEGA)

Generic forward-model trait + MCTS across game domains.

```bash
cargo run --example game_state_01_bomber_mcts --features game_state
cargo run --example game_state_02_bomber_gvg --features game_state
```

`game_state_01` confirms the STRATEGA finding — generic MCTS ≈ Random (25%) in high-variance
FFA without domain heuristics. `game_state_02` shows team-aware MCTS beats Random (62%) but
Greedy (OSLA) still dominates (100%).

## 7. Blue Bear

```bash
cargo run --example bear_01_demo                                    # solves 3×3 BXT/#MG in 7 steps
cargo run --example bear_02_tui                                     # ←/→/Home/End, A toggles auto-play
```

## 8. Core

Core library features — validation, inference, sampling, scoring.

```bash
cargo run --example core_01_validator --features validator
cargo run --example core_02_raven
cargo run --example core_03_ppot --features ppot
cargo run --example core_04_prefill
cargo run --example core_05_maxsim --features maxsim
cargo run --example core_06_peira --features peira_distill
```

## 9. Dungeon

```bash
cargo run --example dungeon_01_tui
cargo run --example dungeon_02_multifloor
```

## 10. Sudoku

Streaming "thinking" Sudoku solver with deterministic validation.

```bash
cargo run --example sudoku_01_9x9 --features sudoku
cargo run --example sudoku_02_speculative --features sudoku
cargo run --example sudoku_03_tui --features sudoku
cargo run --example sudoku_04_percepta_vs                            # no feature flag
```

`sudoku_04` is an unfair-but-informative comparison: Rust backtracking (~350K steps/sec) vs
Percepta's WASM transformer (~30K tok/s) — mostly an algorithmic advantage, not language.

## 11. Tactical AI

Grid-based tactical AI with terrain, procedural maps, fog of war, and parallel simulation.
None require feature flags.

```bash
cargo run --example tactical_01_ai
cargo run --example tactical_02_terrain
cargo run --example tactical_03_procedural
cargo run --example tactical_04_parallel
cargo run --example tactical_05_bench
cargo run --example tactical_06_tui
cargo run --example tactical_07_strategic
cargo run --example tactical_08_headless
cargo run --example tactical_09_fog
cargo run --example tactical_10_fog_bench
```

## 12. Review

Inference-time review metrics (Plan 036) — tracks how often the bandit reviewer *fixes* a
wrong pick (helpful) vs *breaks* a correct one (harmful), with benefit-to-risk ratio.

```bash
cargo run --example review_01_metrics --features bandit
```

## 13. Go (AutoGo)

Go AI with 6 player strategies (Random/Greedy/Validator/HL/GZero/MCTS), Tromp-Taylor scoring.
Full docs: [`.docs/14_go_arena.md`](../.docs/14_go_arena.md). API-bridge examples need a
running AutoGo server (`scripts/autogo_server.sh`).

```bash
cargo run --features go --example go_00_api_bridge      # needs server
cargo run --features go --example go_01_mcts
cargo run --features go --example go_02_tournament
GO_GAMES=2 cargo run --features go --example go_03_head_to_head   # needs server
cargo run --features go --example go_04_gzero --release
cargo run --features go --example go_05_autoresearch
cargo run --features go --example go_06_bench --release
cargo run --features go --example go_07_tui             # ←/→ step, Space auto, R new, Q quit
cargo run --features go --example go_08_self_play_freeze
cargo run --features "memo_reflections,go" --example go_09_reflection_qa
```

**Selected results:** `go_01` MCTS wins 65% vs Random; `go_04` Black wins 98.6% (first-move
advantage); `go_05` top config 100% win-rate, convergence stable; `go_06` 9×9 `advance()`
~182K ops/sec (opening).

## 14. CNA (Contrastive Neuron Attribution)

Sparse-MLP circuit discovery and runtime modulation.

```bash
cargo run --example cna_01_discovery --features cna_steering
cargo run --example cna_02_steering --features cna_steering
cargo run --example cna_03_go_circuit --features "cna_steering,go"
```

## 15. Stepwise Reward Shaping (StepCodeReasoner, Plan 054)

Intra-trajectory reward shaping — rewards arms proportionally to how many downstream arms
they enable. ⚠️ Plan 054 proved NO GAIN; infrastructure only, off by default.

```bash
cargo run --example stepcode_01_shaped_bandit --features stepcode
```

## 16. AND-OR DDTree (Plan 190)

Blueprint-driven subgoal decomposition with AND/OR nodes.

```bash
cargo run --example and_or_demo --features and_or_dtree
cargo run --example and_or_sudoku --features and_or_dtree
```

## 18. EGA — Energy-Gated Attention (Plan 139)

```bash
cargo run --example ega_01_quality --features ega_attn
cargo run --example ega_02_energy_profile --features ega_attn
cargo run --example ega_03_eviction --features ega_attn
cargo run --example ega_04_combined --features ega_attn
```

## 19. CachePrune (Plan 140)

SAT regions + rolling-hash segment matching + sensitivity masking.

```bash
cargo run --example cache_prune_01_sat_bench --features cache_prune
cargo run --example cache_prune_02_segment_match --features cache_prune
```

## 20. RT-Turbo (Plan 126)

Retrieval-head sparse decode via low-dimensional indexing.

```bash
cargo run --example rt_turbo_01_calibration --features rt_turbo
cargo run --example rt_turbo_02_decode_bench --features rt_turbo
```

## 21. SpecHop (Plan 131)

Continuous multi-hop speculation pipeline. Full docs: [`.docs/16_spechop_architecture.md`](../.docs/16_spechop_architecture.md).

```bash
cargo run --example spechop_01_pipeline --features spechop
cargo run --example spechop_02_cost_model --features spechop
```

## 22. KVarN (Research 159)

Variance-normalized KV-cache quantization.

```bash
cargo run --example kvarn_goat_proof --features kvarn
cargo run --example kvarn_thinking_demo --features "kvarn,thinking_cot"
cargo run --example octpq_kvarn_fusion --features "kvarn,hybrid_oct_pq"
```

## 23. Ruliology (Plan 188)

Wolfram-style enumeration of simple program strategies as bandit arms.

```bash
cargo run --example ruliology_demo --features ruliology
```

## 24. Skill Lifecycle & Problem Evolution (Plans 191–192)

Inference-time skill evolution and open-ended problem-evolution arena.

```bash
cargo run --example skill_lifecycle_demo --features skill_lifecycle
cargo run --example partial_scoring_demo --features partial_scoring
cargo run --example problem_evolution_demo --features problem_mutator
cargo run --example idea_divergence_demo --features idea_divergence
```

## Misc / standalone demos

```bash
cargo run --example directional_credit_demo --features directional_credit
cargo run --example kv_share_demo --features kv_share
cargo run --example stiff_anomaly_demo --features stiff_anomaly
cargo run --example randopt_01_basic --features randopt_weight
cargo run --example datrie_01_bench --features datrie_vocab
cargo run --example spec_reconciliation_demo --features spec_reconciliation
cargo run --example percepta_phase0 --features percepta_compile
cargo run --example thinking_cot_demo --features thinking_cot
cargo run --example mux_latent_compress --features mux_latent_context
cargo run --example mux_latent_expand --features mux_latent_context
```

---

## Feature Flags

The flags below gate the example groups above. The full set (292 flags, including
production-default architecture features) lives in [`Cargo.toml`](../Cargo.toml) `[features]`
and the [README Feature Flags](../README.md#feature-flags) section.

| Flag | Gates |
|------|-------|
| `bandit` | BanditPruner; bandit/HL/review examples (Plan 030) |
| `safe_bandit` | PrudentBanker safe-phased bandit (Plan 137) |
| `bomber` | Bomberman arena (Plan 033) — pulls `bevy_ecs`, `bandit` |
| `bomber-wasm` | NNPlayer WASM validator (Plan 034) |
| `bomber-agent` | Coding-agent validator loop (Issue 052) |
| `monopoly` | Monopoly FSM arena (Plan 034) |
| `fft` | FFT Tactics arena (Plan 047) — pulls `bandit`, `sr2am_configurator` |
| `game_state` | GameState forward model + MCTS (Plan 056) |
| `go` | AutoGo bridge + Go GameState (Plan 065) — pulls `bandit`, `reqwest` |
| `sudoku` | SudokuPruner + sudoku examples |
| `validator` | SynPruner syntax validation — pulls `syn`, `proc-macro2` |
| `ppot` | PPoT CPU resampling (Plan 026/027) |
| `maxsim` | MaxSim late-interaction scoring (Plan 080) |
| `peira_distill` | PEIRA modelless distillation (Plan 153) |
| `cna_steering` | CNA circuit discovery + modulation (Plan 087) |
| `ropd_rubric` / `g_zero` / `sdar_gate` | Distillation signals for tournaments |
| `bt_rank` / `sr2am_configurator` / `sia_feedback` | Ranking + configurator + feedback bandits |
| `vpd_em_distill` / `rmsd_distill` / `sdpg_bandit` | Co-evolutionary distillation tournaments |
| `memo_reflections` | MeMo reflection-QA pipeline (Plan 094) |
| `stepcode` | StepCode shaped rewards (Plan 054 — no gain, off by default) |
| `and_or_dtree` | AND-OR DDTree decomposition (Plan 190) |
| `ega_attn` | Energy-gated attention (Plan 139) |
| `cache_prune` | CachePrune SAT + rolling hash (Plan 140) |
| `rt_turbo` | RTTurbo retrieval-head sparse decode (Plan 126) |
| `spechop` | SpecHop multi-hop speculation (Plan 131) |
| `kvarn` | KVarN variance-normalized KV quant (Research 159) |
| `thinking_cot` | Adaptive CoT thinking (Plan 194) |
| `ruliology` | Ruliology bandit strategies (Plan 188) |
| `skill_lifecycle` | Inference-time skill evolution (Plan 192) |
| `partial_scoring` / `problem_mutator` / `idea_divergence` | Problem-evolution arena (Plan 191) |
| `directional_credit` / `kv_share` | Plan 184 / 185 inference candidates |
| `stiff_anomaly` / `randopt_weight` / `datrie_vocab` | Plan 138 / 121 / Research 137 demos |
| `spec_reconciliation` | Speculative reconciliation engine (Plan 177) |
| `percepta_compile` | Full Percepta transformer-VM stack (Plan 064) |
| `mux_latent_context` | MUX-Latent zero-training context compression (Plan 238, default-ON) |
| `full` | Everything default-on (see `Cargo.toml`) |

```bash
# Run with a specific feature
cargo run --example monopoly_01_arena --features monopoly

# Or enable everything
cargo run --example sudoku_01_9x9 --features full
```
