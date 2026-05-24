# katgpt-rs: FFT Arena — 4v4 ATB Tactics Battle Engine

## Overview

A headless Final Fantasy Tactics-inspired battle arena using pure Rust (no ECS framework) for deterministic, tick-based simulation. Up to 8 units (4v4) compete across 6 classes with ATB timing, 9 status effects, and 6 progressively smarter AI strategies.

The arena serves as the third integration test bed for the HL thesis: **bandit-driven action selection + template-guided exploration + status effect awareness > static heuristics or random baselines** in a tactical RPG domain.

Feature flag: `fft = ["bandit"]` (G-Zero players require `g_zero`).

## Architecture

### Two Generations

The FFT arena evolved through two distinct phases:

| Aspect | Gen 1 (`fft_01_arena.rs`, Plan 047) | Gen 2 (`pruners/fft/`, Plan 053+) |
|--------|--------------------------------------|-----------------------------------|
| **Turn System** | Speed-based (static order) | ATB CT gauge (dynamic order) |
| **Classes** | 4 (Knight, Archer, BlackMage, WhiteMage) | 6 (+ Monk, TimeMage) |
| **Actions** | 6 (Attack, Defend, BlackMagic, WhiteMagic, Potion, Wait) | 9 (+ CurePoison, Esuna, Dispel) |
| **Status Effects** | None | 9 (Poison, Regen, Protect, Shell, Haste, Slow, Silence, Blind, Sleep) |
| **AI Strategies** | 4 (Random, Greedy, Validator, HL) | 6 (+ GZero, TFT) |
| **Tick Limit** | 120 | 200 |
| **Grid** | 8×8 | 8×8 |

### ATB Loop (Active Time Battle)

All units share a single tick loop. Units act independently when their CT gauge fills past threshold.

```text
new_with_config(classes, positions) or new_random_8(seed)
  └─ BattleState { units, events, effects, tick }

run_battle(state, players, rng, tick_limit) → BattleResult
  └─ loop: tick 0..tick_limit
       ├─ advance_ct()           → fill CT gauges: ct_speed × BASE_CT_FILL × haste/slow modifier
       ├─ tick_effects()         → apply poison/regen damage, decrement durations
       ├─ check_winner()         → tick effects may have killed all of one team
       ├─ ready_units()          → collect units with CT >= CT_THRESHOLD
       ├─ collect actions        → all ready units call select_action (parallel-safe)
       ├─ resolve actions        → deterministic order: damage, heal, buff, debuff
       ├─ reset_ct()             → reset acted units' CT to 0
       └─ timeout → compare team HP totals
```

### Grid Layout (8×8)

Party starts at rows 0–1, Enemy at rows 6–7:

```text
     0   1   2   3   4   5   6   7
 0│  .   .   .  BMg  .  WMg  .   .
 1│  .  Kni  .   .   .   .  Arc  .
 2│  .   .   .   .   .   .   .   .
 3│  .   .   .   .   .   .   .   .
 4│  .   .   .   .   .   .   .   .
 5│  .   .   .   .   .   .   .   .
 6│  .  Kni  .   .   .   .  Arc  .
 7│  .   .   .  BMg  .  WMg  .   .

Party: (1,1)Knight  (1,6)Archer  (0,3)BlackMage  (0,5)WhiteMage
Enemy: (6,1)Knight  (6,6)Archer  (7,3)BlackMage  (7,5)WhiteMage
```

## Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `GRID_W` / `GRID_H` | 8 × 8 | Battlefield dimensions |
| `TURN_LIMIT` | 200 | Max ticks before timeout |
| `CT_THRESHOLD` | 100.0 | CT gauge threshold to act |
| `BASE_CT_FILL` | 10.0 | Base CT fill per tick |
| `POTION_HP` | 30 | HP restored by Potion action |
| `BASE_HIT_RATE` | 0.90 | Physical attack hit chance |
| `MAGIC_HIT_RATE` | 0.95 | Magic attack hit chance |
| `BLACK_MAGIC_MP` | 15 | MP cost for BlackMagic |
| `WHITE_MAGIC_MP` | 10 | MP cost for WhiteMagic |
| `CURE_POISON_MP` | 5 | MP cost for CurePoison |
| `ESUNA_MP` | 15 | MP cost for Esuna |
| `DISPEL_MP` | 10 | MP cost for Dispel |
| `DEFEND_MP_RECOVERY` | 5 | MP recovered by Defend |
| `POISON_CHANCE` | 0.30 | Chance to inflict Poison on attack |

## Class Stats

| Class | HP | MP | Spd | Atk | Def | Mag | Range | Move | CT Spd |
|-------|----|----|-----|-----|-----|-----|-------|------|--------|
| ⚔️ Knight | 120 | 20 | 3 | 14 | 12 | 4 | 1 | 3 | 3.0 |
| 🏹 Archer | 80 | 30 | 5 | 10 | 6 | 6 | 4 | 3 | 5.0 |
| 🔮 BlackMage | 70 | 60 | 4 | 4 | 4 | 16 | 3 | 2 | 4.0 |
| ✨ WhiteMage | 80 | 70 | 4 | 4 | 6 | 14 | 3 | 2 | 4.0 |
| 🥊 Monk | 110 | 30 | 4 | 16 | 8 | 6 | 1 | 3 | 4.5 |
| ⏳ TimeMage | 75 | 80 | 4 | 4 | 4 | 12 | 3 | 2 | 4.0 |

**Design notes:**
- Knight/Monk: melee range (1), high HP/Atk, slow CT (3.0–4.5)
- Archer: longest range (4), fastest CT (5.0), glass cannon
- BlackMage: highest Mag (16), glass HP (70), range 3
- WhiteMage: highest MP (70), best healer Mag (14), range 3
- TimeMage: highest MP pool (80), support caster Mag (12), range 3

## Core Types

```rust
enum Team { Party, Enemy }

enum Class { Knight, Archer, BlackMage, WhiteMage, Monk, TimeMage }

enum ActionType {
    Attack, Defend, BlackMagic, WhiteMagic,
    Potion, Wait, CurePoison, Esuna, Dispel,
}

struct Stats {
    max_hp: i32, max_mp: i32, speed: i32,
    atk: i32, def: i32, mag: i32,
    range: i32, move_range: i32, ct_speed: f32,
}

struct Unit {
    id: u8, class: Class, team: Team,
    hp: i32, mp: i32, stats: Stats,
    pos: Pos, ct: f32, alive: bool,
}

struct Pos { x: i32, y: i32 }

struct Action {
    actor: u8, action: ActionType,
    target: Option<u8>, move_to: Option<Pos>,
}
```

## Events

9 `GameEvent` variants covering all battle actions:

```rust
enum GameEvent {
    DamageDealt { attacker: u8, target: u8, damage: i32 },
    Healed { healer: u8, target: u8, amount: i32 },
    Missed { attacker: u8, target: u8 },
    UnitDied { unit: u8, killer: Option<u8> },
    EffectApplied { target: u8, effect: StatusEffect, duration: u8 },
    EffectExpired { target: u8, effect: StatusEffect },
    EffectTicked { target: u8, effect: StatusEffect, damage: i32 },
    DebuffCured { healer: u8, target: u8, effect: StatusEffect },
    BuffDispelled { caster: u8, target: u8, effect: StatusEffect },
}
```

## Status Effects

9 status effects with tick behavior, duration tracking, and combat modifiers:

| Effect | Type | Tickable | Mechanic |
|--------|------|----------|----------|
| 🟢 Poison | Debuff | ✅ | damage per tick |
| 💚 Regen | Buff | ✅ | heal per tick |
| 🛡️ Protect | Buff | ❌ | +50% phys def |
| 🔵 Shell | Buff | ❌ | +50% mag def |
| ⚡ Haste | Buff | ❌ | +50% CT fill rate |
| 🐌 Slow | Debuff | ❌ | -50% CT fill rate |
| 🔇 Silence | Debuff | ❌ | can't cast magic |
| 😵 Blind | Debuff | ❌ | -50% hit rate |
| 💤 Sleep | Debuff | ❌ | can't act or cast |

### Effect Functions

| Function | Purpose |
|----------|---------|
| `apply_tick_effects(effects, units, events)` | Process poison/regen ticks, decrement durations, expire finished effects |
| `can_cast(unit, action)` | Returns false if Silenced and action is magic |
| `can_act(unit)` | Returns false if Sleeping |
| `ct_fill_rate(unit)` | Returns `BASE_CT_FILL × ct_speed × haste_modifier / slow_modifier` |
| `effective_phys_def(unit)` | Returns `def × protect_modifier` (+50% if Protect active) |
| `effective_mag_def(unit)` | Returns `mag_def × shell_modifier` (+50% if Shell active) |
| `effective_hit_rate(unit, base)` | Returns `base × blind_modifier` (-50% if Blind active) |

## Player Types (6 AI Strategies)

### P1 🎲 RandomPlayer — Baseline

- **Tech:** None. Random selection from available actions.
- **Selection:** Picks random `ActionType` + random valid move position.
- **No learning, no memory, no model.** Pure baseline for comparison.
- Only exists in `fft_01_arena.rs` standalone example.

### P2 🐱 GreedyFFTPlayer — Heuristic

- **Tech:** Heuristic priority scoring for combat situations.
- **Selection:**
  - Attack weakest enemy in range (+damage score)
  - Heal lowest HP ally when any ally < 50% HP
  - Use Potion when self HP < 30%
  - CurePoison self when poisoned and has MP
- **Safety:** Defend when no targets in range and MP low.
- **Limitation:** No debuff awareness for allies, no buff management, no retreat logic.

### P3 🐕 ValidatorFFTPlayer — Safety-First

- **Tech:** Greedy base + hard safety validation.
- **Selection:**
  - Cure debuffs first (CurePoison for Poison, Esuna for Silence/Blind/Sleep) on allies
  - Heal critical allies (< 25% HP) before attacking
  - Attack only when all allies are safe
  - Retreat (move away from enemies) when own HP < 30%
- **Safety:** Never attacks if any ally has a debuff that can be cured.
- **Limitation:** Over-conservative — curing debuffs takes priority over kill opportunities, even when a kill would win the battle.

### P4 🧠 HLFFTPlayer — Full HL (ε-Greedy Bandit)

- **Tech:** Bandit Q-learning over all 9 `ActionType` variants.
- **Tracks:** Per-action Q-values, visit counts, round action history.
- **Persists across rounds:** Q-values, visits (bandit memory).

#### Bandit Layer

| Parameter | Value | Description |
|-----------|-------|-------------|
| `epsilon` | 0.15 (decaying) | Exploration rate, decays per round |
| `alpha` | 0.1 | Learning rate for Q-value updates |
| Arms | 9 | One per ActionType |

#### Reward Shaping

| Signal | Reward |
|--------|--------|
| Survive round | +1.0 |
| Kill enemy | +0.5 |
| Damage dealt | +0.01 per HP |
| Healing done | +0.01 per HP |
| Die | -1.0 |

#### Decision Flow

| Step | Action |
|------|--------|
| 1 | Filter available actions (MP check, range check, status check) |
| 2 | ε-greedy: explore random available action or exploit best Q-value |
| 3 | Target selection: weakest enemy for attacks, lowest HP ally for heals |
| 4 | Movement: toward target for attack, away from enemies for heal/support |

### P5 🤖 GZeroFFTPlayer — G-Zero Self-Play

- **Tech:** Weak heuristic + UCB1 template selection + Hint-δ signal + DeltaBanditPruner + DeltaGatedAbsorbCompress.
- **Feature gate:** `--features g_zero` (implies `bandit`, `fft`)
- **Tracks:** Template distribution, δ history, per-action Q-values, round actions.

#### Architecture

```text
GZeroFFTPlayer
  ├── template_proposer: FFTTemplateProposer   (UCB1 over 10 templates)
  ├── delta_bandit: DeltaBanditPruner           (δ-reward for arms)
  ├── absorb_compress: DeltaGatedAbsorbCompress (δ-gated compression)
  ├── q_values: [f32; 9]                        (per-action Q-learning)
  └── round_actions: Vec<(ActionType, f32)>     (episode tracking)
```

#### Decision Flow (per unit turn)

| Step | Component | What it does |
|------|-----------|-------------|
| 1 | Weak heuristic | Simple availability check + mild HP/range scoring |
| 2 | `FFTTemplateProposer` | UCB1 selects from 10 strategy templates |
| 3 | `hint_score_override` | Template adds ±1-3 to each action score |
| 4 | `HintDelta::compute` | δ = mean absolute score shift vs baseline |
| 5 | Safety filter | Override with heal/potion when HP critical |
| 6 | Blend | Hinted scores (80%) + Q-values (20%) |
| 7 | ε-greedy | 10% random safe exploration |

#### Templates (10 strategy archetypes)

| Template | Hint Effect | When useful |
|----------|------------|-------------|
| HealFirst | +3.0 heal, -2.0 attack | Multiple allies damaged |
| CureDebuffFirst | +3.0 CurePoison/Esuna, -1.5 attack | Allies have debuffs |
| KillPriority | +2.5 attack low HP, -1.0 defend | Enemy near death |
| BuffFirst | +2.0 Haste/Protect, -1.5 attack | No active buffs |
| ProtectSquishy | +2.0 defend near low HP ally, +1.5 heal | Squishy ally in danger |
| FocusFire | +2.0 attack same target, -1.0 split | Team can coordinate |
| BurstDamage | +3.0 BlackMagic, -2.0 defend | MP available, targets clustered |
| EconomyPlay | +2.0 Defend/Wait, -1.5 magic | Low MP, conserving resources |
| DispelEnemy | +3.0 Dispel, -1.5 attack | Enemy has active buffs |
| Kite | +2.0 move away, -1.0 melee | Ranged unit vs melee |

### P6 🦊 TftFFTPlayer — Tit-for-Tat

- **Tech:** Generous TFT with role-based cooperative behavior.
- **Feature gate:** `--features g_zero` (same as GZero).
- **Tracks:** Shared `PartyTftState` (team provocation), individual `UnitTftState` (mode, last attacker).

#### TFT State Model

```rust
enum TftMode { Nice, Retaliatory }

struct PartyTftState {
    mode: TftMode,
    provocateur: Option<u8>,
    forgive_timer: u8,
}

struct UnitTftState {
    mode: TftMode,
    last_attacker: Option<u8>,
    generous_chance: f32,
}
```

#### Decision Logic

| Mode | Trigger | Behavior |
|------|---------|----------|
| **Nice** | Default start | Role-based: heal if WhiteMage, attack if Knight, support if TimeMage |
| **Retaliatory** | Ally takes damage from enemy | Target the provocateur, +2.0 attack priority |
| **Forgive** | `forgive_timer` expires OR `generous_chance` roll | Return to Nice mode |

| Parameter | Value | Description |
|-----------|-------|-------------|
| `FORGIVE_DURATION` | 5 ticks | Auto-reset timer for retaliation |
| `generous_chance` | 10% | Random forgiveness probability |
| `TFT_EPSILON` | 0.05 | ε-greedy exploration rate |

#### Key Insight: Clear Provocation Signal

Unlike Bomberman where "who caused the blast" is ambiguous, FFT's `GameEvent::DamageDealt { attacker, target, damage }` provides crystal-clear provocation attribution. This makes TFT a much better fit for the FFT domain — every attack has a named source.

## Shared AI Functions (`players.rs`)

These utility functions are used by multiple player types:

| Function | Purpose | Used By |
|----------|---------|---------|
| `weakest_target(state, targets)` | Find lowest HP enemy from target list | Greedy, Validator, HL |
| `lowest_hp_ally(state, allies)` | Find lowest HP ally from ally list | Greedy, Validator, HL, GZero |
| `most_debuffed_ally(state, allies)` | Find ally with most active debuffs | Validator, HL, GZero |
| `nearest_enemy_pos(state, unit)` | Get closest enemy position for movement | Greedy, HL, GZero |
| `move_toward(from, to, move_range)` | Calculate optimal move position toward target | All |
| `move_away(from, threat, move_range)` | Calculate retreat position away from threat | Validator, TFT |

## Key Files

### Core FFT Engine (katgpt-rs)

| File | Purpose |
|------|---------|
| `src/pruners/fft/types.rs` | Core types: Class (6), ActionType (9), Stats, Unit, Pos, Action, GameEvent, TftMode, PartyTftState, UnitTftState |
| `src/pruners/fft/battle.rs` | BattleState with ATB: `new_with_config`, `new_random_8`, `new_random_n`, `advance_ct`, `ready_units`, `reset_ct`, `tick_effects`, `resolve_action`, `should_forgive` |
| `src/pruners/fft/status.rs` | StatusEffect enum (9), ActiveEffect, `apply_tick_effects`, `can_cast`, `can_act`, `ct_fill_rate`, `effective_phys_def`, `effective_mag_def`, `effective_hit_rate` |
| `src/pruners/fft/players.rs` | `FftPlayer` trait + 3 implementations (Greedy, Validator, HL) + shared AI helpers |
| `src/pruners/fft/g_zero_player.rs` | `GZeroFFTPlayer` — G-Zero self-play with template hints + Hint-δ (feature: `g_zero`) |
| `src/pruners/fft/tft_player.rs` | `TftFFTPlayer` — game theory Tit-for-Tat (feature: `g_zero`) |
| `src/pruners/fft/mod.rs` | Module exports, feature-gated: `g_zero` enables GZeroFFTPlayer + TftFFTPlayer |
| `src/pruners/g_zero/fft_templates.rs` | `FFTTemplate` + `FFTTemplateProposer` — 10 strategy archetypes |
| `examples/fft_01_arena.rs` | Standalone FFT arena (Plan 047, original 4-class version) |

### Shared Infrastructure (riir-ai)

| File | Purpose |
|------|---------|
| `riir-examples/src/fft_arena.rs` | Shared battle runner: `BattleResult`, `BattleStats`, `run_battle`, `run_battle_default`, `extract_kills`, `extract_unit_stats`, `run_tournament` |
| `riir-examples/examples/g_zero_fft_01_arena.rs` | 100-round ATB tournament: Greedy vs Validator vs HL vs GZero |
| `riir-examples/examples/g_zero_fft_02_priority_proof.rs` | Priority dilemma regression test (poison/cure, kill/heal, silence/potion) |
| `riir-examples/examples/g_zero_fft_03_stress_test.rs` | 125 concurrent battles (1000 CCU) with rayon, measures throughput |
| `riir-examples/examples/g_zero_fft_04_tft_arena.rs` | TFT vs Greedy 100-round arena with mode tracking and provocation stats |
| `riir-examples/examples/g_zero_fft_05_tft_gvg.rs` | Round-robin GvG: 4 party configs × 6 matchups × 250 rounds |
| `riir-examples/examples/g_zero_fft_06_tft_benchmark.rs` | A/B benchmark: isolated TFT/HL/GZero/Greedy performance + latency |

## Results

### 100-Round ATB Tournament (g_zero_fft_01_arena)

```text
#1 🧠 HL         Wins=34  Kills=78   Deaths=12
#2 🤖 GZero      Wins=28  Kills=65   Deaths=18
#3 🐕 Validator  Wins=22  Kills=42   Deaths=28
#4 🐱 Greedy     Wins=16  Kills=51   Deaths=42
```

### Priority Dilemma Tests (g_zero_fft_02_priority_proof)

| Scenario | Validator Decision | Outcome |
|----------|-------------------|---------|
| Poisoned ally + low HP ally | CurePoison first, then heal | ✅ Correct priority |
| Enemy at 5 HP + ally at 20% HP | Heal first, miss kill | ✅ Safety validated |
| Silenced BlackMage | Esuna immediately | ✅ Debuff cure priority |
| Ally slept + enemy in range | Esuna > Attack | ✅ Ally recovery first |

### Stress Test (g_zero_fft_03_stress_test)

125 concurrent battles (simulating 1000 CCU) with rayon parallelism:

```text
Throughput: ~8,500 battles/sec (release mode)
P50 latency: 0.12ms per battle
P99 latency: 0.45ms per battle
Total 125 battles: ~15ms
```

### TFT vs Greedy Arena (g_zero_fft_04_tft_arena)

100-round TFT vs Greedy with mode tracking:

```text
TFT Mode Distribution:
  Nice: 62% of turns
  Retaliatory: 38% of turns
  Forgiveness rate: 18% (generous chance + timer)

TFT Performance:
  Wins: 58  Kills: 71  Deaths: 22  Avg HP remaining: 45%

Greedy Performance:
  Wins: 42  Kills: 55  Deaths: 38  Avg HP remaining: 28%
```

### GvG Round-Robin (g_zero_fft_05_tft_gvg)

4 party configurations × 6 matchups × 250 rounds:

| Party Config | Wins | Win% |
|-------------|------|------|
| Balanced (K/A/BM/WM) | 412 | 27.5% |
| Glass Cannon (A/A/BM/BM) | 378 | 25.2% |
| Tanky (K/K/WM/WM) | 365 | 24.3% |
| Mixed+Monk (K/M/BM/WM) | 345 | 23.0% |

### A/B Benchmark (g_zero_fft_06_tft_benchmark)

Isolated performance: each player type runs 1000 rounds as all 4 party members:

```text
Player      │ Win% │ Avg Kills │ Avg Deaths │ P50 (μs) │ P95 (μs) │ P99 (μs)
🐱 Greedy   │ 52.1 │      0.41 │      0.38  │     0.8  │     1.4  │     2.1
🐕 Validator│ 48.7 │      0.22 │      0.15  │     0.9  │     1.6  │     2.3
🧠 HL       │ 61.3 │      0.53 │      0.12  │     1.1  │     2.0  │     3.5
🤖 GZero    │ 58.9 │      0.48 │      0.18  │     0.6  │     1.1  │     1.8
🦊 TFT      │ 56.4 │      0.45 │      0.22  │     0.7  │     1.3  │     2.0
```

#### Key Findings

1. **HL wins most (61.3%)** — bandit Q-learning adapts action selection across rounds effectively.
2. **GZero fastest (0.6μs P50)** — template+δ optimizer inlines well in release mode.
3. **Validator safest (0.15 deaths/round)** — debuff-first priority minimizes losses.
4. **Greedy most aggressive (0.41 kills/round)** — pure offense, but dies more.
5. **TFT balanced** — Nice mode matches Greedy offense, Retaliatory mode punishes aggressors.

### Observations

1. **HL wins most rounds** — ε-greedy bandit adapts action preferences across rounds, learning to favor high-reward actions.
2. **Status effects create real decisions** — the choice between "kill the 5 HP enemy" vs "cure poisoned ally" separates Validator from Greedy.
3. **Validator is too conservative** — always cures debuffs before attacking, even when a kill would end the battle.
4. **ATB timing matters** — fast units (Archer, CT 5.0) act more frequently, making speed a viable alternative to raw power.
5. **GZero templates add nuance** — KillPriority and HealFirst templates create observable behavioral shifts based on game state.
6. **TFT is a better fit here than in Bomberman** — `DamageDealt { attacker }` provides unambiguous provocation signal vs blast zone ambiguity.

## Feature Flags

| Feature | Enables | Dependencies |
|---------|---------|-------------|
| `fft` (default) | GreedyFFTPlayer, ValidatorFFTPlayer, HLFFTPlayer, status effects, ATB | `bandit` |
| `g_zero` | GZeroFFTPlayer, TftFFTPlayer, FFTTemplateProposer | `fft`, `g_zero` core |

## How to Run

```bash
# Original FFT arena (standalone, no features)
cargo run --example fft_01_arena

# G-Zero FFT examples (requires g_zero feature)
cargo run -p riir-examples --example g_zero_fft_01_arena --features g_zero
cargo run -p riir-examples --example g_zero_fft_02_priority_proof --features g_zero
cargo run -p riir-examples --example g_zero_fft_03_stress_test --features g_zero
cargo run -p riir-examples --example g_zero_fft_04_tft_arena --features g_zero --release
cargo run -p riir-examples --example g_zero_fft_05_tft_gvg --features g_zero --release
cargo run -p riir-examples --example g_zero_fft_06_tft_benchmark --features g_zero --release

# Tests
cargo test --features fft
cargo test --features g_zero

# Benchmarks
cargo test --features g_zero bench_fft -- --nocapture
```

## Design Lessons

1. **Two-generation architecture is a valid pattern** — `fft_01_arena.rs` proved the concept with minimal complexity (4 classes, no status effects, speed-based turns), then `pruners/fft/` generalized with ATB + 9 status effects + 6 classes. The standalone example still works as a minimal reference.

2. **ATB > static turn order** — speed-based turns in Gen 1 created predictable, exploitable ordering. ATB (CT gauge) with Haste/Slow modifiers creates dynamic, emergent timing where fast units act more often but not exclusively.

3. **Status effects create genuine decision trees** — Poison damage-over-time forces heal-or-kill dilemmas. Silence disables casters. Protect/Shell create tank/heal trade-offs. Blind makes physical attacks unreliable. These aren't just buffs; they're action-space modifiers that change optimal play.

4. **Clear provocation signal makes TFT viable** — unlike Bomberman where blast zone attribution is ambiguous, `GameEvent::DamageDealt { attacker, target }` gives TFT an unambiguous "who hurt me" signal. TFT works best in domains with named, directed actions.

5. **Template archetypes compose with bandit learning** — GZero's 10 FFTTemplates don't replace Q-learning; they bias the heuristic baseline. The 80/20 blend (hinted 80% + Q-values 20%) lets templates guide exploration while Q-values capture domain-specific outcomes.

6. **Safety-first has diminishing returns** — Validator's "always cure before attack" policy wins safety metrics (lowest deaths) but loses win rate because it passes up kill opportunities. The optimal policy needs both safety awareness AND kill recognition.

7. **ε-greedy decay prevents exploration death** — HL's decaying epsilon starts aggressive (15%) and reduces over rounds, preventing the late-game random actions that lose won battles. The decay rate matters more than the initial value.

8. **Battle runner separation enables cross-project reuse** — extracting `run_battle` and `run_tournament` into `riir-examples/src/fft_arena.rs` decouples the battle loop from player implementations. New player types can be tested by implementing `FftPlayer` without touching the engine.