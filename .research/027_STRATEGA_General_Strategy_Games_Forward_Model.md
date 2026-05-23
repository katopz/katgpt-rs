# Research: STRATEGA — General Strategy Games Forward Model (27)

> Source: [STRATEGA: A General Strategy Games Framework](https://r.jina.ai/https://www.tnt.uni-hannover.de/papers/data/1606/2020__AIIDE_SGW__STRATEGA__A_General_Strategy_Games_Framework.pdf) — Dockhorn et al., 2020 (AIIDE)
> Date: 2020, distilled 2026-06
> **Verdict: HIGH VALUE — Forward Model trait unifies Bomber/FFT/Monopoly under one `GameState` + `advance()` abstraction. Enables generic MCTS/RHEA/Bandit agents across all arenas without code duplication. STRATEGA's config-driven YAML maps directly to our TOML config pattern. Their logging/profiling validates our `BenchmarkResult` approach. Key finding: rule-based agents (92% win) crush naive MCTS (39% win) in complex games — confirms our HL tiered approach is correct.**

## TL;DR

STRATEGA is a C++ framework for n-player turn-based strategy games where **games are defined via YAML** and agents access a **Forward Model (FM)** — a copyable, roll-forwardable game state. The FM lets any agent simulate "what if I do X?" without modifying the real game state.

Three key abstractions:
1. **Config-driven game definition** — units, actions, terrain, win conditions all in YAML files
2. **Forward Model API** — `state.copy()` + `state.advance(action)` → new state, during agent thinking time
3. **Common agent interface** — one trait for all agents, whether rule-based or search-based (MCTS, RHEA)

Their benchmark result is the punchline: **rule-based agents dominate naive search** (RBC: 92% win in Kings vs MCTS: 39%). The branching factor (~10^15 in Polytopia-scale games) is too large for unguided tree search. This validates our HL tiered player architecture exactly.

---

## Core Mechanisms (What We Need)

### 1. Forward Model (FM)

The single most valuable abstraction. Every game state implements:

```text
state.copy()         → deep clone of entire game state
state.advance(action) → apply one action, return new state
state.is_terminal()   → game over?
state.score(player)   → heuristic evaluation for a player
state.actions(player) → available actions for this player
```

Agents receive `(state_copy, fm)` and can simulate arbitrarily deep what-if scenarios. The real game state is never touched during agent thinking.

**Why this matters for us:** Our `DDTree` already does this for token sequences. A `GameState` trait would let MCTS/Bandit/RHEA run on *any* arena (Bomber, FFT, Monopoly) without knowing game specifics.

### 2. Config-Driven Game Definition (YAML)

STRATEGA defines everything in YAML:

```yaml
# Actions
Attack:
  Value: 10
  Range: 6
  AttackerReward: 2
  CanExecutedToFriends: true

# Units
LongRangeUnit:
  RangeVision: 6
  RangeMovement: 4
  AttackDamage: 70
  Health: 100
  Actions: [Move, Attack]

# Game Rules
Game Rules:
  TimeForEachTurn: 10
  NumberOfMaxRounds: 100
  Players: [MCTS Player, RHEA Player]
```

We already use `GameConfig` for Monopoly and `domains.toml` for inference budgets. The gap: Bomber and FFT hardcode their rules in Rust structs.

### 3. Agent API

```text
interface Agent:
    init(forward_model, player_id)
    act(game_state_copy) → Action
```

Our `BomberPlayer::select_action(grid, pos, events, rng) → BomberAction` is structurally identical. The key difference: STRATEGA agents get the *full* FM (can call `advance()`), ours only get an observation snapshot.

### 4. Statistical Forward Planning (SFP) Agents

STRATEGA ships three SFP agents, all using the FM:

| Agent | Algorithm | Our Equivalent |
|---|---|---|
| OSLA | 1-step lookahead, pick best | `GreedyPlayer` scores each action |
| MCTS | UCB1 tree search, random rollouts | `BanditPruner` + `DDTree` (token domain) |
| RHEA | Evolutionary algorithm on action sequences | `PPoT` variants (token domain) |

### 5. Heuristic Evaluation (SDH)

Strength Difference Heuristic normalizes unit attributes:

```text
strength(unit) = Σ (attribute / max_of_all_types(attribute))
state_value = Σ self.unit.strength * hp_pct - Σ enemy.unit.strength * hp_pct
```

Our `score_action()` and `hint_score_override` do similar things but per-arena. A normalized `StateHeuristic` trait would generalize this.

### 6. Logging and Profiling

Per-tick logging of:
- Action space size (available actions)
- Actions executed
- Decision time (μs)
- State size (bytes)

STRATEGA's Figure 4 shows action space size over time correlates with agent quality — RHEA maintains larger action spaces longer. This is directly relevant to our DDTree branching analysis.

---

## Key Experimental Findings

### Rule-Based > Naive Search in Complex Games

```text
Kings (win rate):
  RBC (rule-based):  0.92
  RHEA (evolution):  0.56
  MCTS (tree search): 0.39
  OSLA (1-step):     0.01

Healers (win rate):
  RBC:  0.82
  RHEA: 0.68
  MCTS: 0.45
  OSLA: 0.03

Pushers (win rate — different game mechanic):
  RBP (rule-based):  0.73
  MCTS:              0.61
  RHEA:              0.53
  OSLA:              0.00
```

**Takeaway:** Domain knowledge (rule-based) beats blind search when branching factor is high. Our tiered HL approach (Random < Greedy < Validator < HL < GZero) is exactly right.

### Action Space Complexity

- Kings: ~150 actions/move at start, decreasing as units die
- Polytopia: 50+ actions/move, branching factor ~10^15
- RHEA wins over MCTS when it maintains action diversity longer

### Budget Constraints

All SFP agents use a fixed budget of 2000 FM calls per action selection. This maps directly to our `tree_budget` in `domains.toml`.

---

## Mapping to Our Stack

### What Maps Well

| STRATEGA | Our System | Status |
|---|---|---|
| Forward Model API | `GameState` trait with `advance()` | **Need to build** |
| Config YAML | TOML config pattern | Partial — Monopoly has it, Bomber/FFT don't |
| Agent trait | `BomberPlayer` / `FftPlayer` traits | ✅ Already exists per-arena |
| MCTS agent | `BanditPruner` + `DDTree` | ✅ Token domain, need game domain |
| RHEA agent | `PPoT` variants | ✅ Token domain, need game domain |
| SDH heuristic | `score_action()` / `hint_score_override` | ✅ Per-arena, could generalize |
| Tournament runner | `g_zero_02_tournament.rs` | ✅ Round-robin with seed swap |
| Action space logging | `BenchmarkResult` | Partial — no per-tick action space tracking |

### What Doesn't Map

| STRATEGA | Reason |
|---|---|
| C++ implementation | We're Rust — keep perf + safety |
| GUI + isometric rendering | We have TUI + riir-gpu if needed |
| LUA scripting for mods | WASM validators are safer and faster |
| Real-time game support | Our focus is turn-based |
| Single action per turn | Bomber does multi-action per tick (one per player) |

---

## Modelless Distillations

### D1: `GameState` Trait — Forward Model Abstraction

The core value extraction. A trait that all arenas implement:

```rust
pub trait GameState: Clone {
    type Action: Clone;

    fn available_actions(&self, player_id: u8) -> Vec<Self::Action>;
    fn advance(&self, action: &Self::Action, player_id: u8) -> Self;
    fn is_terminal(&self) -> bool;
    fn reward(&self, player_id: u8) -> f32;
    fn action_space_size(&self, player_id: u8) -> usize;
    fn current_player(&self) -> u8;
    fn tick(&self) -> u32;
}
```

Bomber implements this by wrapping its `World` state. FFT implements this with `BattleState`. Monopoly implements this with its ECS world.

### D2: Generic MCTS — Runs on Any `GameState`

```rust
pub fn mcts_search<S: GameState>(
    state: &S,
    player_id: u8,
    budget: usize,        // max advance() calls
    rollout_depth: usize,
    heuristic: &dyn Fn(&S, u8) -> f32,
) -> S::Action
```

Same UCB1 logic as `BanditPruner`, but operating on `GameState::advance()` instead of token sequences.

### D3: Generic RHEA — Evolutionary Action Sequences

```rust
pub fn rhea_search<S: GameState>(
    state: &S,
    player_id: u8,
    budget: usize,
    horizon: usize,       // sequence length
    pop_size: usize,
    heuristic: &dyn Fn(&S, u8) -> f32,
) -> S::Action
```

### D4: Action Space Logging — Per-Tick Metrics

```rust
pub struct ActionSpaceLog {
    pub tick: u32,
    pub player_id: u8,
    pub available_actions: usize,
    pub actions_executed: usize,
    pub cumulative_actions: usize,
}
```

---

## Relationship to Existing Work

| Our Existing | STRATEGA Parallel | Gap |
|---|---|---|
| Plan 033: Bomber Arena | STRATEGA Kings game | Bomber has simpler state (no fog-of-war by default) |
| Plan 047: FFT Tactics | STRATEGA Healers (multi-unit combat) | FFT already has multi-unit turns |
| Plan 035: Monopoly FSM | STRATEGA game config | Monopoly already has `GameConfig` |
| Plan 049: G-Zero Self-Play | STRATEGA agent comparison | G-Zero adds learning across episodes (STRATEGA agents don't learn) |
| Plan 052: GFlowNet Distillation | STRATEGA SFP agents | GFlowNet adds flow-based action sampling |
| `BanditPruner` (UCB1) | STRATEGA MCTS (UCB1) | Same algorithm, different domain |
| `PPoT` (variants) | STRATEGA RHEA (evolution) | Same algorithm, different domain |

**Key difference:** Our agents *learn* across episodes (bandit Q-values, absorb-compress memory, G-Zero templates). STRATEGA's agents are stateless between games. We're already ahead here.

---

## What Won't Transfer

1. **Real-time game loop** — STRATEGA plans RTS support but hasn't shipped it. Our tick-based loop is already correct for turn-based.
2. **LUA scripting** — WASM validators are strictly superior for our use case (sandboxed, typed, fast).
3. **Isometric GUI** — Our TUI is sufficient for proof-of-concept. GPU rendering exists in `riir-gpu`.
4. **Game Description Language (GDL)** — STRATEGA references GGP's formal game language. We don't need this; Rust trait implementations are our "game description language."

---

## Key Insight for Modelless

The forward model abstraction is the **only** thing we need from STRATEGA. Everything else (MCTS, RHEA, heuristics, logging) we already have in some form, just coupled to specific domains.

The `GameState` trait decouples search algorithms from game rules. This means:
- One `mcts_search()` works on Bomber, FFT, Monopoly, and *future games*
- One `rhea_search()` works everywhere
- Tournament runners become generic over `GameState`
- Cross-game agent comparison becomes trivial (same agent, different game)

STRATEGA's paper proves this works — their MCTS/RHEA agents run unchanged across Kings, Healers, and Pushers.

---

## Honest Assessment

### What We Get

- **Cross-game agent reuse** — MCTS/Bandit/RHEA written once, tested on all arenas
- **Forward model for planning** — agents can simulate what-if scenarios
- **Action space metrics** — correlate branching factor with agent quality
- **Config-driven game variants** — new game modes without recompilation

### What We DON'T Get

- **Better agents** — STRATEGA's own results show generic search loses to domain-specific heuristics
- **Real-time support** — only turn-based games
- **Learning across episodes** — STRATEGA agents are stateless between games (our G-Zero already does this)

### Magnitude Expectation

- **Small** code change — one trait, one arena refactor, one generic algorithm
- **Medium** architectural value — proves the abstraction for future arenas
- **Low** risk — trait is additive, doesn't change existing behavior

### Risk

1. **ECS `World` is not `Clone`** — Bomber's state is a `bevy_ecs::World`, which doesn't implement `Clone`. The `GameState` impl needs to snapshot/restore world state. Options:
   - Serialize/deserialize (slow but correct)
   - Manual snapshot struct (fast but maintenance burden)
   - Extract just what MCTS needs into a lightweight struct (best — see D1)
2. **Multi-player MCTS** — STRATEGA skips opponent turns in tree policy (assumes worst case). We need the same simplification.
3. **Partial observability** — Bomber events are visible to all players (no fog-of-war in current impl). If we add it, `GameState` needs a `observe(player_id)` method that filters hidden information.

### STRATEGA's Own Lesson

> "Results show that the RBC agent is very proficient... While MCTS and RHEA agents were able to beat the OSLA agent, they were no match against the RBC agent."

This is the most important finding for us: **generic search alone won't beat domain-specific heuristics**. The value of the `GameState` trait is not "better agents" — it's **agent portability** and **cross-game evaluation**. Our G-Zero self-play (which *does* learn across episodes) is the real agent quality lever.

---

## References

- Dockhorn, A., Hurtado-Grueso, J., Jeurissen, D., & Perez-Liebana, D. (2020). STRATEGA: A General Strategy Games Framework. *AIIDE Workshop on Strategy Games*.
- [STRATEGA GitHub](https://github.com/GAIGResearch/Stratega)
- Related: Polytopia framework (Perez-Liebana et al., 2020b) — branching factor ~10^15
- Related: microRTS (Ontañón et al., 2018) — RTS framework with forward model
- Related: GGP / GVGAI — general game playing predecessors