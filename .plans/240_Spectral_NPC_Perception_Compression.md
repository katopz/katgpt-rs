# Plan 240: Spectral NPC Perception Compression

**Status:** GOAT-Gated — Experimental (CPU 25.2%, <40% threshold)
**Feature Flag:** `sense_lod` (opt-in, requires `sense_composition` + `slod`)
**Routing:** katgpt-rs → crates/katgpt-core/src/sense/

## Why

`batch_project_all` projects every module for every NPC every tick. In dense zones (200+ NPCs), most NPCs are far from the player or in low-relevance clusters — yet all 7 sense modules run full dot-product + sigmoid for each. Research 212 (Fusion A) proposes reusing SLoD's `ScaleBoundary` detection to assign per-NPC LOD levels, skipping low-value modules. Target: >40% CPU reduction with <5% behavioral quality loss.

## Architecture

```
SlodOperator ──ScaleBoundary──▶ SenseLodRouter ──SenseLodLevel──▶ NpcBrain.active_lod
                                                                      │
batch_project_all ◀──module_mask ◀─────────────────────────────────────┘
       │
       ▼
  skip modules not in mask → project only active → fill defaults for skipped
```

### SenseLodLevel

```rust
#[repr(u8)]
enum SenseLodLevel {
    Full,       // All 7 modules — nearby player/combat
    Compressed, // Common + Spatial + Fighter only — mid-range
    Minimal,    // Spatial only — background/ambient
}
```

| LOD | Modules Active | Dot-products Saved |
|-----|---------------|-------------------|
| Full | Common, Fighter, GameTheory, Spatial, Social, Skill | 0/7 |
| Compressed | Common, Fighter, Spatial | 4/7 (57%) |
| Minimal | Spatial | 6/7 (86%) |

### SenseLodRouter

Reads `ScaleBoundary` from `SlodOperator` + NPC distance to player/centroid. Assigns LOD per NPC:
- Within σ₁ boundary → Full
- Between σ₁ and σ₂ → Compressed
- Beyond σ₂ → Minimal

## Tasks

- [x] Create `SenseLodLevel` enum with `module_mask() -> &[SenseKind]` in `crates/katgpt-core/src/sense/lod.rs`
- [x] Add `active_lod: SenseLodLevel` field to `NpcBrain` (default: `Full`)
- [x] Create `SenseLodRouter` struct — takes `&[ScaleBoundary]` + distance metric, produces `SenseLodLevel`
- [x] Modify `NpcBrain::project_all_into` to skip modules not in LOD mask, push `0.0` for skipped
- [x] Modify `batch_project_all` / `batch_project_all_par` to accept `SenseLodRouter` and assign LODs pre-batch
- [x] Add `#[cfg(feature = "sense_lod")]` gate on all new code; feature requires `sense_composition` + `slod` in `Cargo.toml`
- [x] Add unit tests: mask correctness, skip behavior, fallback when no boundaries
- [x] Create benchmark `crates/katgpt-core/benches/sense_lod.rs`: 200 NPCs, measure CPU reduction vs behavioral delta

## GOAT Gate

| Metric | Threshold | Pass |
|--------|-----------|------|
| CPU reduction (200 NPC batch) | >40% vs baseline | ☒ 25.2% |
| Behavioral quality loss | <5% (max projection delta across modules) | ✅ 0.0% |
| Zero alloc in hot path | No new allocations in `project_all_into` | ✅ |
| Graceful fallback | No boundaries → Full LOD (no behavior change) | ✅ |

**Verdict:** GOAT FAIL on CPU reduction. Quality is perfect. CPU bottleneck is `project_kind`'s O(n) GM override scan, not the dot-product itself. Stays experimental — not promoted to default.

## Expected Result

NPCs in dense zones automatically run fewer sense modules based on spectral cluster boundaries. Background NPCs (Minimal) run 1 module instead of 7. Combat-adjacent NPCs stay Full. No behavioral regression for active NPCs. Benchmark proves the trade-off is worth the complexity.
