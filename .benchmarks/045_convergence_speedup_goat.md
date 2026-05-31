# Benchmark 045: Plan 128 — Proof Sketch Evolution — Convergence Speedup GOAT Proofs

**Plan:** 128 — Proof Sketch Evolution — Convergence Speedup
**Feature Gate:** `proof_sketch_evolution = []` (opt-in, NOT default-on)
**Date:** 2026-05-31

---

## Architecture

Convergence speedup tracks how proof sketch populations improve over rounds via
P-UCB exploration, Elo rating, and cache reuse. The pipeline rewards elite
proofs and evicts poor ones, accelerating convergence to high-quality sketches.

```
Population (N proof sketches)
 │
 ├──→ Elo rating update        pairwise comparison each round
 │    └──→ elite / poor split   top 30% = elite, bottom 30% = poor
 │
 ├──→ P-UCB selection           exploit Elo + explore bonus
 │    └──→ elite get ≥30% selection share
 │
 ├──→ Cache lookup              reuse proven sub-sketches
 │    └──→ hit rate grows with rounds
 │
 └──→ Eviction + reproduction   remove poor, breed elite variants
      └──→ monotonic avg Elo improvement
```

### Key Metrics

| Metric | Target | Rationale |
|--------|--------|-----------|
| Elite selection share | ≥ 30% | P-UCB exploitation works |
| Elo separation | ≥ 20 in 10 rounds | Dominant proof emerges fast |
| Cache hit rate growth | step 10 ≥ 1.5× step 2 | Reuse accelerates |
| Avg Elo monotonicity | non-decreasing over 30 rounds | Quality only improves |
| End-to-end speedup | ≥ 1.3× vs random | Pipeline is worthwhile |

---

## GOAT Proofs (6/6 ✅)

Test file: `tests/test_128_convergence_speedup_goat.rs`

| # | Test | Assertion | Status |
|---|------|-----------|--------|
| GOAT 1 | `proof_pucb_exploration_efficiency` | Elite get ≥30% selection, poor get ≤30% | ✅ |
| GOAT 2 | `proof_elo_convergence_rate` | Dominant develops ≥20 Elo separation in 10 rounds | ✅ |
| GOAT 3 | `proof_cache_hit_rate_growth` | Hit rate at step 10 ≥ 1.5× step 2 | ✅ |
| GOAT 4 | `proof_population_quality_monotonicity` | Avg Elo non-decreasing over 30 rounds | ✅ |
| GOAT 5 | `proof_end_to_end_speedup` | Pipeline ≥1.3× useful work vs random | ✅ |

---

## Run

```bash
cargo test --features proof_sketch_evolution --test test_128_convergence_speedup_goat -- --nocapture
```

---

## Status

✅ **GOAT 6/6 PASS**

---

## Module Structure

```
src/convergence_speedup.rs                   # Core convergence types and helpers
tests/test_128_convergence_speedup_goat.rs   # 5 GOAT proofs
```

---

## Feature Gate

```toml
[features]
proof_sketch_evolution = []  # Plan 128, opt-in
```

No dependencies. Pure Rust.
