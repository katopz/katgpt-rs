# Issue 1232: katgpt-hla cache γ novelty gate (Plan 639 Phase 3 T3.3)

**Status:** DONE-NEGATIVE — G1 ✓ · G2 ✗ (0/36 cells) · G3 ✓ (median Δ 7.87 ns ≤ 10.0, hd=8, bmo) · G4 ✓; `novelty_gamma` stays opt-in (measured-negative lane).
**Claimed:** p639-t33@amo, 1795077881, expires 2026-11-20T08:44:41Z

**Plan:** [639](../.plans/639_novelty_gated_retention.md) Phase 3 T3.3 · **Substrate:** `katgpt_sense::novelty_retention::NoveltyChannel` (landed `f907edece`) · **Bench precedent:** [987](../.benchmarks/987_novelty_retention_goat.md) (T2.2 matched-point verdict, T2.5 G3 re-price).

## Scope

A per-cache novelty gate on the HLA decay γ (`MultiLayerHlaCache::gamma` /
the `gamma` argument of `hla_state_update` / `hla_layer_update`), LOCAL
state only. The cache, kernels, serde layout and the 5-scalar sync crossing
(valence, arousal, desperation, calm, fear) + bridge law are UNTOUCHED.

- **Placement:** new opt-in feature `novelty_gamma` on **katgpt-hla only**
  (no new katgpt-core key — katgpt-core sits at its feature ratchet).
  Dep edge `katgpt-hla → katgpt-sense` (optional, `novelty_retention`):
  both are katgpt-types leaves, no cycle (katgpt-sense does not reach
  katgpt-hla), no new external crate (blake3/fastrand already in the
  workspace allowlist posture). The law is CONSUMED, never re-implemented.
- **Type:** `katgpt_hla::novelty_gamma::HlaNoveltyGamma` — a separate struct
  the caller owns next to its cache (the cache type is not changed, so the
  off build and the existing serde snapshot are byte-identical). Per-instance
  armed (`HlaNoveltyGamma::arm(cfg)` → `None` on an invalid config or
  `NOVELTY_RETENTION=0`). Unarmed = the caller passes its fixed γ (the
  incumbent code path, untouched).
- **Observation signal (one scalar per cache-tick, k = 1):** the query's
  surprise against the cache's own first moment —
  `x_t = (1/hd)·Σ_i (q_t[i] − mQ_{t−1}[i]/Z_{t−1})²`, where `mQ` is the
  cache's per-head query accumulator and `Z_t = γ_t·Z_{t−1} + 1` is the
  decayed mass the gate tracks (the same recursion `mQ` follows). A shift in
  the query regime raises `x` persistently (sustained ⇒ κ pulls γ DOWN,
  forget faster); a one-tick outlier raises it once (transient ⇒ μ pushes γ
  UP, hold). The fold is a FIXED-ORDER 8-accumulator scalar fold (pinned
  order, no platform SIMD reduction — Issue 940).

## Sync / determinism statement (Issue 940)

γ scales `SK, CQV, mQ, G, h` — and `h`/`mQ` feed the COMMITTED
`hla_moments` (BLAKE3 on freeze, Issue 940 HL-1/HL-3). So an armed γ DOES
reach a committed value; therefore the armed path must be bit-deterministic
across platforms: elementwise f32 ops only (sub/mul/add/div/abs/max/min —
IEEE-exact, Rust never contracts to FMA), the signal fold is a pinned-order
fixed tree, no HashMap order, no tie rule needed (no pick). The gate state
(two EMAs + seeded flag + mass) is NOT part of the cache serde snapshot: a
resumed-from-snapshot replay must re-arm at the snapshot tick from the same
state or it diverges — a consumer that commits `hla_moments` from an armed
cache must either replay from genesis or keep the gate unarmed (recorded
caveat; snapshotting the gate is a consumer-side follow-up, not this issue).
The 5 synced scalars never pass through this code.

## Pre-registered gates (written and pushed BEFORE the first measurement)

Harness: `crates/katgpt-hla/tests/novelty_gamma_goat.rs`
(`--release --features novelty_gamma -- --nocapture --test-threads=1`).

- **G1 identity:** (a) default build (feature off) — every existing
  katgpt-hla test green, no file of the incumbent kernel/types changed;
  (b) the unarmed arm = the caller's fixed γ fed to the real kernel —
  cache state bit-identical to the incumbent run over 600 ticks;
  (c) armed κ = μ = 0 ⇒ γ_t == clamp(base, g_min, g_max) every tick and the
  cache bit-identical to the fixed-clamped-base run.
- **G2 quality (HLA-cache fixture):** hd = 8, one head, the REAL
  `hla_state_update`; cached latent = `m̂_t = mQ_t / Z_t`; truth = the
  query-regime mean `m_t`; `q_t = m_t + noise`.
  Cases (seeded xorshift + Box–Muller, 600 ticks, seeds {1,2,3}):
  (a) Steady σ = 0.3; (b) Steps σ = 0.1, mean re-drawn (components
  U[−1,1]) at ticks {150, 300, 450}; (c) Spikes = Steady + a one-tick
  outlier (`m + 3·sign` per dim) every 40 ticks.
  **adaptation** = mean over the 3 steps of ticks until
  `‖m̂ − m_new‖ ≤ 0.1·‖m_new − m_old‖` (cap 100);
  **noise_rmse** = RMSE(m̂ − m) over ticks ≥ 50 of (a) ∪ (c).
  Fixed curve: pinned γ ∈ {0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.93, 0.95,
  0.97, 0.98, 0.99} per seed + dense γ 0.50..0.99 step 0.01 (seed mean).
  Gated grid: base γ = 0.9, κ ∈ {0.2, 0.4, 0.8}, μ ∈ {0, 0.2, 0.4},
  β ∈ {4, 8}, g_max ∈ {0.99, 0.995} (36 cells; g_min 0.05, EMA 0.3/0.03).
  A cell PASSES iff the Plan 639 T2.2 verdict holds verbatim: (i) ∃ ONE
  pinned γ with adaptation ≤ 0.7× AND noise_rmse ≤ 1.1× on all 3 seeds;
  (ii) no dense-curve point dominates the seed-mean cell; (iii) distance
  below the piecewise-linear dense curve > 0. **G2 PASS ⇔ ≥ 1 passing
  cell**; both full tables recorded, no cherry-picks. 0 cells ⇒
  DONE-NEGATIVE (honest negative, the feature stays opt-in).
- **G3 cost (Plan 639 T2.5 bar, k = 1):** one gate step per cache-tick
  (the cache's γ is one scalar for all layers/heads). Interleaved A/B
  (`tests/common/ab_timing.rs`, 51 rounds × 20 000 iters, warmup 4 000,
  `black_box` both ends): a = `hla_state_update` at fixed γ = 0.9, hd = 8;
  b = `HlaNoveltyGamma::step` + `hla_state_update` at the gated γ, hd = 8.
  Statistic per run = `b_ns − a_ns` per entity-tick. **bmo**, release
  binary only, `bench_preflight.sh` PASS at launch AND end (MAX_LOAD 11.2);
  3 runs; **PASS ⇔ median Δ ≤ 10.0 ns** (k·Δ with k = 1). hd = 8 is the
  deployed per-NPC belief width (riir `BELIEF_DIM = 8`, `hla_moments:
  [f32; 8]`); an hd = 64 Δ is REPORTED, not gated (the signal fold is
  O(hd)). A tainted run is re-run, never folded in; no extra runs after 3.
  Disclosure: an incumbent at γ = 1.0 skips the decay passes entirely —
  arming there also ADDS the O(hd²) decay (γ ≤ g_max < 1), a behaviour
  change priced separately, not by this bar.
- **G4 alloc-free:** armed step + kernel over 600 ticks, 0 allocations
  (binary-local counting allocator).

Promotion: NOT in scope — the feature stays opt-in whatever the verdict.

## Tasks

- [x] T1 `novelty_gamma` feature + `HlaNoveltyGamma` + surprise fold (katgpt-hla)
- [x] T2 G1/G2/G4 harness + run (amo)
- [x] T3 G3 × 3 on bmo (binaries only)
- [x] T4 record verdict, close (HISTORY.md row, lease folded)

## Results (2026-11-19)

- **G1 ✓** — default-feature katgpt-hla tests 18/18, kernel.rs/types.rs
  untouched (diff = Cargo.toml + lib.rs `cfg`-gated `mod`/`use` only);
  identity arm bit-exact vs the fixed-clamped-base incumbent (3 seeds × 3
  cases × 600 ticks, every cache buffer compared by bits); armed path
  replay-deterministic.
- **G2 ✗ — 0/36 cells.** Fixed curve (seed mean): γ 0.85 → adapt 14.44 /
  noise 0.1251; γ 0.90 → 21.00 / 0.1002; γ 0.93 → 31.44 / 0.0825. Gated
  cells land ON the curve: e.g. κ 0.8 μ 0 β 8 → 9.00 / 0.1806 (dominated by
  γ 0.72); best κ 0.8 μ 0.4 β 8 g_max 0.995 → 9.44 / 0.1530, undominated,
  +0.0061 below the curve, but (i) fails (no pinned γ gives ≤ 0.7×
  adaptation at ≤ 1.1× noise). 12/36 cells pass (ii), 6/36 pass (iii), 0
  pass (i). Mechanism read (not a claim): the surprise signal fluctuates on
  the σ = 0.3 steady stream and the one-tick spikes enter the fast EMA, so
  κ pulls γ down in steady state too — the gate behaves like a lower fixed
  γ (a move along the curve, not below it). Full tables:
  `cargo test --release -p katgpt-hla --features novelty_gamma --test
  novelty_gamma_goat g2 -- --nocapture`.
- **G3 ✓** — bmo, release binary only, preflight PASS at launch AND end
  each run: Δ (hd = 8) **7.87 / 6.61 / 9.11 ns** (incumbent 65.58 / 53.67
  / 52.14 ns) → **median 7.87 ns ≤ 10.0** (k = 1). hd = 64 reported:
  Δ 9.96 / 13.93 / 2.79 ns over a ~1.22 µs kernel. PROVENANCE (all six):
  `power=AC Power load=1.08–1.47 swap=18627.69M canary=skipped
  powermode=0(auto) ane=ok`. Post-measurement change: clippy's
  `chunks_exact_to_as_chunks` rewrite of the fold (`as_chunks::<8>`) — same
  element order and arithmetic; G1/G2 re-run bit-identical tables.
- **G4 ✓** — armed step + kernel over 599 ticks: 0 allocations.

