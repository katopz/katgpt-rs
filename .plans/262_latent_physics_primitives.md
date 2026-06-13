# Plan 262: Latent Physics Primitives — SectorProjection + ActionBridge

**Date:** 2026-06
**Status:** ✅ COMPLETE (all benchmarks pass, GOAT PASS — already default-ON)
**Blocks:** Plan 001 in riir-armageddon (armageddon consumes these primitives)
**Context:** Armageddon's latent-space AI needs generic projection + bridge patterns that any game can use

---

## Goal

Extract two generic AI patterns that currently exist as ad-hoc implementations into proper reusable primitives in `katgpt-rs-core`:

1. **`SectorProjection`** — multi-sector spatial projection using `SenseModule::project()` 
2. **`ActionBridge`** — generic latent→raw action bridge via `sigmoid(dot(...))`

These are NOT game-specific. Any game with NPC AI that thinks in latent space needs them.

---

## Tasks

### Phase 0: SectorProjection
- [x] Create `katgpt-rs-core/src/sense/sector.rs`
- [x] Define `SectorProjection` struct:
  ```rust
  /// Multi-sector spatial projection for NPC perception.
  /// Divides space around NPC into N sectors, projects each into a latent score
  /// using pre-computed ternary direction vectors.
  pub struct SectorProjection<const N: usize> {
      /// Pre-computed direction vectors per sector (ternary {-1, 0, +1})
      sector_directions: [[i8; D]; N],
      /// Last projection scores per sector (updated on project call)
      scores: [f32; N],
  }
  ```
- [x] Implement `project(&mut self, observation: &[f32; D]) -> &[f32; N]`
  - For each sector: `scores[i] = fast_sigmoid(dot(observation, sector_directions[i]))`
  - Uses existing `CpuTernaryBackend` dot-product path
  - Zero allocation, fixed-size
- [x] Implement `update_directions(&mut self, new_directions: [[i8; D]; N])` — hotswap without restart
- [x] Tests: project known observation → verify sigmoid output range [0, 1]
- [x] Bench: measure N=8 sector projection latency (target: < 100ns, since SenseModule is 45ns) — **37.4ns, 2.7x under target**

### Phase 1: ActionBridge
- [x] Create `katgpt-rs-core/src/bridge/mod.rs` (new module)
- [x] Define `ActionBridge` struct:
  ```rust
  /// Bridges latent Q-values to raw game actions via sigmoid-gated projection.
  /// Generic over action space size.
  pub struct ActionBridge<const A: usize> {
      /// Direction vectors per action (ternary {-1, 0, +1})
      action_directions: [[i8; D]; A],
      /// Confidence threshold (actions below this are suppressed)
      threshold: f32,
  }
  ```
- [x] Implement `select_action(&self, q_values: &[f32; D]) -> (usize, f32)`
  - For each action: `score[a] = sigmoid(dot(q_values, action_directions[a]))`
  - Returns (best_action_index, confidence_score)
  - Suppressed if confidence < threshold
- [x] Implement `select_top_k(&self, q_values: &[f32; D], k: usize, out: &mut [(usize, f32)]) -> usize`
  - Top-K actions sorted by confidence, for games with multi-action turns
- [x] Tests: known Q-values → verify action selection is deterministic
- [x] Bench: measure action selection latency for A=8 (target: < 200ns) — **38.9ns select_action (5.1x under), 49.3ns select_top_k(k=3) (10.1x under)**

### Phase 2: Feature Gates
- [x] Gate `SectorProjection` behind `sector_projection` feature (default on)
- [x] Gate `ActionBridge` behind `action_bridge` feature (default on)
- [x] Add to `katgpt-rs-core/Cargo.toml` feature map

---

## What Already Exists (just wrapping)

| Pattern | Existing Code | How We Wrap It |
|---------|--------------|----------------|
| Sector projection | `SenseModule::project()` = `confidence * fast_sigmoid(dot())` | `SectorProjection` calls project() N times per sector |
| Action bridge | `latent_to_raw_scalar()` in `curator_bridge.rs` = `sigmoid(dot())` | `ActionBridge` calls it A times per action |
| Direction vectors | Ternary bit-planes in `CpuTernaryBackend` | Same storage format, same dot-product path |
| Confidence decay | `SpatialBelief::decay_confidence()` = `sigmoid(-λΔt)` | Not wrapped here — stays in riir-games |

No new math. Just structured wrappers over existing primitives.

---

## What This Unlocks

| Game | Uses |
|------|------|
| Armageddon | `SectorProjection<8>` for terrain, `ActionBridge<6>` for abilities |
| Civ sim | `SectorProjection<4>` for zone awareness, `ActionBridge<4>` for NPC actions |
| Dungeon | `SectorProjection<8>` for room awareness, `ActionBridge<4>` for dungeon abilities |
| Racing game | `SectorProjection<12>` for track awareness, `ActionBridge<3>` for steer/brake/accel |

TL;DR: Two generic AI primitives for katgpt-rs-core. `SectorProjection` wraps SenseModule for multi-sector spatial queries. `ActionBridge` wraps sigmoid(dot()) for latent→raw action selection. Both zero-alloc, fixed-size, ternary-backed. Any game uses them.
