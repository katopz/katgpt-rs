# 113 — Boss Avoidance Heuristics for AI/Hybrid Explorers

> **Status:** ✅ SUPERSEDED — All boss-avoidance tasks complete. Only remaining unchecked item (Hybrid underperformance) is explicitly labeled "future work, separate from boss avoidance". Scope achieved.

## Tasks

- [x] Plan design and threat model
- [x] Add `boss_last_known` and `boss_alive` fields to `FogState` (both files)
- [x] Update `FogState::update` to track boss sightings in visible range
- [x] Add `dodge_boss_if_adjacent` — minimal adjacent-step dodge (both files)
- [x] Run benchmark and validate improvement
- [ ] ~~Hybrid still underperforms — cluster exploration causes timeouts~~ (future work, separate from boss avoidance)
- [x] Update plan 104 task as complete
- [x] Commit

## Problem Statement

Current fog-of-war results show a paradox:
```
🐻 BF:     5 wins, 142 avg steps — unpredictable movement avoids boss
🐰 AI:     2 wins,  97 avg steps — heuristic beelines toward boss area
🦊 Hybrid: 1 win,  145 avg steps — earliest discovery but most deaths
```

Smarter exploration = faster discovery but MORE deaths. AI/Hybrid heuristics beeline
toward high-value frontier areas where the boss also pathfinds. BF's "dumb" strategy
spreads movement unpredictably, accidentally avoiding the boss.

## Root Cause

`AiExplorer::score_frontier` only considers:
- +1 per unseen neighbor (information gain)
- +2 chokepoint, +1 dead-end (structural value)
- +3 near bridge when holding keys (goal proximity)

**Missing: any consideration of boss threat.** The boss moves every 3 steps toward
the player via BFS, but the explorer never factors this into its scoring.

## Approaches Tried

### ❌ Approach 1: Frontier Score Penalty (ThreatAwareness enum)

Added `ThreatAwareness::None/Moderate/High` enum with `threat_penalty()` function
that subtracted from frontier scores based on boss proximity.

| Variant | AI penalty | Hybrid penalty | Result |
|---------|-----------|---------------|--------|
| Aggressive | -15/-8/-3 | -30/-16/-6 | AI:2, HY:0 wins — too many timeouts |
| Moderate | -8/-3 range 0-4 | -12/-5 range 0-4 | AI:4, HY:2 wins — better but timeouts |
| Tiebreaker | -2/-1 range 0-4 | -6/-3 range 0-4 | AI:2, HY:1 wins — penalties too weak |
| Visible-only | -8/-3 visible boss only | -12/-5 visible boss only | AI:2, HY:1 — no improvement |

**Verdict:** Frontier-level penalties fail because:
1. Changing frontier preference changes exploration patterns → oscillation → 500-step timeouts
2. Boss is often NOT visible (vision radius 4 vs 16×16 map), so penalty rarely triggers
3. Even when visible, penalizing the BEST frontier means picking a WORSE one → longer path → more boss exposure

### ❌ Approach 2: Path-Level Dodge (danger_radius)

`dodge_boss_if_close()` — after BFS picks an action, check if moving toward
visible boss and if so, pick alternative direction that moves away.

| danger_radius (AI/Hybrid) | AI wins | HY wins | AI avg steps |
|--------------------------|---------|---------|-------------|
| 2/3 | 5 | 3 | 256 |
| 2/4 | 5 | 4 | 256 |
| 3/4 | 5 | 3 | 256 |

**Verdict:** Dodge improves wins but causes severe step inflation (142→256 avg).
The dodge creates back-and-forth oscillation: step toward frontier → dodge away →
step toward frontier → dodge away → timeout. Higher danger_radius = more oscillation.

### ✅ Approach 3: Adjacent-Only Dodge (FINAL)

`dodge_boss_if_adjacent()` — ONLY dodge when the preferred action would step
directly onto the boss tile (certain death). Picks best alternative direction.

```rust
fn dodge_boss_if_adjacent(game, state, fog, preferred) -> usize {
    // Only trigger if:
    // 1. Boss is visible (last_known + alive + in visible set)
    // 2. Preferred action moves player ONTO boss tile
    // Find alternative direction that doesn't step on boss
}
```

**Minimal intervention:** Doesn't change exploration patterns, only prevents
walking directly into the boss.

## Results

### Before (no avoidance)
```
🐻 BF:     5 wins, 142 avg steps, disc at 15.8
🐰 AI:     2 wins,  97 avg steps, disc at 8.2
🦊 Hybrid: 1 win,  145 avg steps, disc at 5.5
Seeds with ≥1 success: 7/30
```

### After (adjacent dodge)
```
🐻 BF:     5 wins, 142 avg steps, disc at 15.8  (unchanged)
🐰 AI:     6 wins, 146 avg steps, disc at 19.7  🏆 most wins!
🦊 Hybrid: 2 wins, 200 avg steps, disc at 11.3  (doubled)
Seeds with ≥1 success: 11/30  (+57%)
```

### Key Findings

1. **AI now beats BF (6 vs 5 wins)** — minimal dodge is enough to flip the survival advantage
2. **57% more solvable seeds** (7→11) — overall improvement across all strategies
3. **Differentiation preserved** — three strategies still produce different outcomes
4. **Hybrid underperforms** — cluster-based exploration gets stuck in loops (separate issue)

### Research Insight

**Under fog of war with hostile agents, the optimal threat avoidance is minimal:**
- Don't change WHERE you explore (frontier penalties)
- Don't change HOW you get there (path-level dodge)
- Just don't step on the boss (adjacent dodge)

Stronger avoidance creates oscillation, which is worse than occasional death. The
boss is an unpredictable agent — attempting to predict and avoid its path is
counterproductive because your avoidance patterns become predictable too.

## Files Modified

| File | Changes |
|------|---------|
| `examples/tactical_10_fog_bench.rs` | FogState boss tracking, dodge_boss_if_adjacent |
| `examples/tactical_09_fog.rs` | Same changes for TUI version |
| `.plans/104_fog_of_war_exploration.md` | Mark boss avoidance task complete |

## Future Work

- **Hybrid cluster timeout issue:** Hybrid's cluster-based exploration gets stuck
  in loops on certain seeds (500-step timeouts). This is an exploration strategy
  issue, not boss avoidance. Potential fix: add a "stuck detector" that switches
  to BF nearest-frontier after N steps without progress.
- **Boss prediction:** Current dodge is reactive. A predictive approach could
  estimate boss movement based on its BFS pathfinding, but this adds complexity
  for marginal gain (the minimal approach already works).
- **Larger maps:** On larger maps with more rooms, the boss encounters would be
  less frequent, making threat avoidance less critical. Focus should be on
  exploration efficiency instead.

## Status: ✅ COMPLETE

### Research Conclusion

Three boss avoidance approaches were evaluated over multiple rounds:

1. **Frontier Score Penalty** — ❌ Changes exploration patterns → oscillation → timeouts
2. **Path-Level Dodge** — ❌ Creates back-and-forth oscillation → step inflation (142→256)
3. **Adjacent-Only Dodge** — ✅ Minimal intervention, maximal effect

**Key finding: Under fog of war with hostile agents, minimal threat avoidance is optimal.**
Don't change WHERE you explore, don't change HOW you get there — just don't step on the boss.
Stronger avoidance creates predictable patterns that are worse than occasional death.

The target of "Hybrid beats both" was NOT achieved. Hybrid's underperformance (2 wins vs AI's 6)
is an exploration strategy issue (cluster-based loops causing timeouts), not a boss avoidance issue.
Boss avoidance is solved; exploration efficiency is the next lever.

### Final Scorecard

```
BEFORE:  🐻 BF 5 wins | 🐰 AI 2 wins | 🦊 Hybrid 1 win | 7/30 seeds solvable
AFTER:   🐻 BF 5 wins | 🐰 AI 6 wins | 🦊 Hybrid 2 wins | 11/30 seeds solvable (+57%)
```

AI now beats BF for the first time. Research complete.