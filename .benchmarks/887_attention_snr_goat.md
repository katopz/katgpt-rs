# Bench 887 — streaming attention-SNR accumulators GOAT (Issue 882 P1)

**Status:** COMPLETE — G1a/G1b/G1c PASS · G2 PASS (3/3 runs; median +0.86–1.10% at N=1024 D=64, +0.35–0.64% at N=512 D=128, bar ≤ 2%) · G3 PASS (bit-identical at N ∈ {128, 1024}) · G4 PASS (0 allocs). Feature `attention_snr` (katgpt-core) stays **OPT-IN**: no consumer yet, and the issue's quality gate (`m_Y` correlation) belongs to P4.

**Box:** M3 Max (16 cores, 64 GB), macOS 26.6.2, **AC power, battery 100% charged** (the gate prints `pmset -g batt` itself), loadavg 2.9–3.4 (1 min) with sibling agent sessions active (one Metal A/B in another worktree; no concurrent cargo in this target dir), memory 3.6 GB free + 32 GB inactive (reclaimable) per vm_stat, no paging pressure. Release profile. G2 via the shared interleaved `ab_median_ratio` (15 rounds × 3 iters/arm, 2 warm-up), black_box on inputs and on every stats read (an unread accumulator is exactly what fat LTO deletes).

Run: `cargo test -p katgpt-core --release --features attention_snr --test bench_887_attention_snr_goat -- --nocapture`

## What was measured

| gate | fixture | result | bar |
|---|---|---|---|
| G1a | random N=130 D=64 | max \|ΔH\| 5.75e-6 nats · rel ΔPR 4.44e-6 | ≤ 1e-3 |
| G1a | random N=1024 D=64 | 4.55e-6 · 4.23e-6 | ≤ 1e-3 |
| G1a | rising max on every K tile, N=1024 | 8.12e-7 · 2.71e-7 | ≤ 1e-3 |
| G1b | planted gold key, 24 strengths | Spearman(Hn, gold mass) −1.000; Hn 0.999 → 0.001 | ≤ −0.95 |
| G1b trap 2 | planted **decoy** at the same strength | Hn 0.001, gold mass 4.8e-7 | pinned negative |
| G1c | `fit_tau` → `SsmaxMode::Fixed` → kernel, targets {0.9, 0.7, 0.5, 0.3} | max \|Hn − target\| 4.8e-7 | ≤ 1e-3 |
| G2 | N=1024 D=64 (plain ≈ 8.84 ms/call) | median 1.0110 / 1.0086 / 1.0103 | ≤ 1.02 |
| G2 | N=512 D=128 (plain ≈ 3.75 ms/call) | median 1.0035 / 1.0064 / 1.0062 | ≤ 1.02 |
| G3 | N ∈ {128, 1024} | bit-identical (8192 / 65536 values) | exact |
| G3 | N=64 (plain → materialized fallback) | max \|Δ\| 1.79e-7 | ≤ 1e-5 |
| G4 | 10 calls, caller scratch | 0 allocations | 0 |

## Reading it honestly

- **G2's cost is the pre-exp copy plus two dot products per K-tile row.** The overhead scales as 1/D, which is why D=128 costs half of D=64: the extra work is per score, and the kernel's own work per score is ∝ D. The worst single round at D=64 was 1.0357 (run 2). The median is the gated quantity and stayed under 1.012 in all three runs. Shorter heads (D=32) would roughly double it and are **not measured**.
- **G1b is a sanity proxy, not the quality gate.** With a single planted key, gold mass and entropy are near-monotone in strength by construction, so Spearman −1.000 is expected. What it pins is that the kernel's statistic moves the right way over a real forward pass. The issue's G1 is `m_Y` correlation on our models, and that instrument is **P4**. It is not claimed here.
- **Trap 2 is the finding worth keeping.** A confident-but-wrong row is indistinguishable from a hit by entropy alone (Hn 0.001 in both). Any policy built on these statistics (P2 clamps, per-head τ, eviction) must be falsified against `m_Y`, never against entropy.
- **The issue's "measure occupancy" / register-pressure concern is a GPU framing.** This is the CPU tiled kernel, where the cost is measured directly as latency. A GPU port (CubeCL / Metal) would need its own occupancy measurement.
- **G3's "ppl Δ within noise" is satisfied trivially.** The gate-off path is the same monomorphized kernel with a no-op sink, and its output is bit-identical, so perplexity cannot move.
