//! Plan 558 GOAT Gate — Variable-Rank Domain Expert Clusters.
//!
//! Runs the G1–G5 GOAT gate for the `variable_rank_domain_expert` feature.
//! Mirrors the convention of `tests/bench_416_region_subspace_goat.rs`.
//!
//! Run with:
//! ```sh
//! cargo test -p katgpt-core --features committed_field_blend \
//!   --test bench_558_variable_rank_domain_expert_goat -- --nocapture --test-threads=1
//! ```
//!
//! For accurate latency numbers (G2), run in **release** mode:
//! ```sh
//! cargo test -p katgpt-core --features committed_field_blend \
//!   --test bench_558_variable_rank_domain_expert_goat --release -- --nocapture --ignored
//! ```
//!
//! # Gates
//!
//! - **G1 (correctness)**: variable-rank router produces finite outputs across
//!   10K random inputs. Covered by lib unit tests; re-checked here for the
//!   bench fixture.
//! - **G2 (perf)**: variable-rank router mean ns/NPC at 1K + 10K NPCs, compared
//!   vs uniform `CommittedFieldBlend<3, 32>` baseline. Pass criterion:
//!   variable-rank ≤ 1.0× baseline in release (variable-rank should NOT be
//!   slower — the masked-out dims skip blend work).
//! - **G3 (entropy)**: variable-rank produces ≥ 1.5× archetype utilization
//!   entropy vs uniform baseline at iso K×D=96 compute (reproduces Research
//!   453 PoC's 1.63× result on the production API).
//! - **G4 (alloc-free)**: 0 allocations in the steady-state hot path. Covered
//!   by the separate `tests/variable_rank_domain_expert_alloc.rs`.
//! - **G5 (modelless purity)**: documented audit (no training deps, no unsafe,
//!   closed-form math). Covered by `.benchmarks/558_*.md`.

#![allow(clippy::float_cmp)]

use katgpt_core::committed_field_blend::{ArchetypeFieldSource, CommittedFieldBlend};
use katgpt_core::variable_rank_domain_expert::{
    ClusterHolder, RoutingVerdict, VariableRankRouter, pick_domain, project_guided, scatter_guided,
};
use katgpt_core::variable_rank_router_static;
use std::time::Instant;

// ─── Deterministic direction field (shared with bench_453 PoC shape) ─────────

struct DirectionField<const D: usize> {
    direction: [f32; D],
    blake3: [u8; 32],
}

impl<const D: usize> DirectionField<D> {
    fn new(seed: usize) -> Self {
        let mut direction = [0.0f32; D];
        for (i, slot) in direction.iter_mut().enumerate() {
            let x = (seed * 37 + i * 13) as f32;
            *slot = ((x * 0.1).sin() + (x * 0.07).cos()) * 0.5;
        }
        let norm: f32 = direction
            .iter()
            .map(|v| v * v)
            .sum::<f32>()
            .sqrt()
            .max(1e-8);
        for v in direction.iter_mut() {
            *v /= norm;
        }
        let mut blake3 = [0u8; 32];
        for (i, b) in blake3.iter_mut().enumerate() {
            *b = ((seed * 251 + i) & 0xFF) as u8;
        }
        Self { direction, blake3 }
    }
}

impl<const D: usize> ArchetypeFieldSource<D> for DirectionField<D> {
    fn evolve<'a>(&self, z: &[f32], dz_scratch: &'a mut [f32]) -> &'a mut [f32] {
        let dot: f32 = (0..D).map(|i| z[i] * self.direction[i]).sum();
        for (i, slot) in dz_scratch.iter_mut().enumerate().take(D) {
            *slot = self.direction[i] * dot;
        }
        &mut dz_scratch[..D]
    }
    fn commitment(&self) -> [u8; 32] {
        self.blake3
    }
    fn lipschitz_bound(&self) -> f32 {
        1.0
    }
}

fn boxed_field<const D: usize>(seed: usize) -> Box<DirectionField<D>> {
    Box::new(DirectionField::new(seed))
}

// ─── Helpers ────────────────────────────────────────────────────────────────

#[inline]
fn shannon_entropy(counts: &[usize]) -> f32 {
    let total = counts.iter().sum::<usize>() as f32;
    if total == 0.0 {
        return 0.0;
    }
    let mut h = 0.0;
    for &c in counts {
        if c > 0 {
            let p = c as f32 / total;
            h -= p * p.log2();
        }
    }
    h
}

fn prng(seed: u64) -> f32 {
    let mut x = seed.wrapping_mul(0x2545F4914F6CDD1D).wrapping_add(1);
    x ^= x >> 13;
    x ^= x << 7;
    x ^= x >> 17;
    ((x >> 11) as f32 / (1u64 << 53) as f32) * 2.0 - 1.0
}

fn npc_state(seed: u64) -> ([f32; 32], [f32; 3]) {
    let mut state = [0.0f32; 32];
    for (i, slot) in state.iter_mut().enumerate() {
        *slot = prng(seed.wrapping_mul(31).wrapping_add(i as u64));
    }
    let activity = [
        prng(seed.wrapping_mul(7).wrapping_add(1)),
        prng(seed.wrapping_mul(7).wrapping_add(2)),
        prng(seed.wrapping_mul(7).wrapping_add(3)),
    ];
    (state, activity)
}

// ─── Uniform baseline: CommittedFieldBlend<3, 32> ───────────────────────────

struct Baseline {
    blend: CommittedFieldBlend<3, 32>,
    fields: [Box<DirectionField<32>>; 3],
}

impl Baseline {
    fn new() -> Self {
        let mut blend = CommittedFieldBlend::<3, 32>::uncommitted();
        blend.pi = [0.5, -0.3, 0.8];
        blend.tau = 1.0;
        Self {
            blend,
            fields: [boxed_field(100), boxed_field(200), boxed_field(300)],
        }
    }

    fn tick(&mut self, z: &[f32; 32], pi_override: &[f32; 3]) -> usize {
        let f0: &dyn ArchetypeFieldSource<32> = self.fields[0].as_ref();
        let f1: &dyn ArchetypeFieldSource<32> = self.fields[1].as_ref();
        let f2: &dyn ArchetypeFieldSource<32> = self.fields[2].as_ref();
        let fields_ref: [&dyn ArchetypeFieldSource<32>; 3] = [f0, f1, f2];
        let mut scratch = [0.0f32; 32];
        let mut out = [0.0f32; 32];
        self.blend.pi = *pi_override;
        self.blend
            .apply_blended(&fields_ref, z, &mut scratch, &mut out);
        // Winning archetype = highest pi (sigmoid monotonicity)
        let mut winner = 0usize;
        let mut best = pi_override[0];
        for (k, &p) in pi_override.iter().enumerate().skip(1) {
            if p > best {
                best = p;
                winner = k;
            }
        }
        winner
    }
}

// ─── Variable-rank router: 3 domains ────────────────────────────────────────
//
// Move    <K=12, L=8>    (project to dims [0..8])
// Combat  <K=6,  L=16>   (project to dims [0..16])
// Quest   <K=3,  L=32>   (no projection)
//
// All at K×L = 96 (iso-compute with the baseline's 3×32=96).

fn make_router() -> VariableRankRouter<3, 32, 3> {
    // Move cluster: K=12, L=8
    let mut move_blend = CommittedFieldBlend::<12, 8>::uncommitted();
    move_blend.tau = 1.0;
    let move_fields: [Box<dyn ArchetypeFieldSource<8>>; 12] =
        std::array::from_fn(|i| boxed_field::<8>(1000 + i) as Box<dyn ArchetypeFieldSource<8>>);
    let move_cluster = Box::new(ClusterHolder::<12, 8>::new(move_blend, move_fields));

    // Combat cluster: K=6, L=16
    let mut combat_blend = CommittedFieldBlend::<6, 16>::uncommitted();
    combat_blend.tau = 1.0;
    let combat_fields: [Box<dyn ArchetypeFieldSource<16>>; 6] =
        std::array::from_fn(|i| boxed_field::<16>(2000 + i) as Box<dyn ArchetypeFieldSource<16>>);
    let combat_cluster = Box::new(ClusterHolder::<6, 16>::new(combat_blend, combat_fields));

    // Quest cluster: K=3, L=32
    let mut quest_blend = CommittedFieldBlend::<3, 32>::uncommitted();
    quest_blend.tau = 1.0;
    let quest_fields: [Box<dyn ArchetypeFieldSource<32>>; 3] = [
        boxed_field(3000) as Box<dyn ArchetypeFieldSource<32>>,
        boxed_field(3100) as Box<dyn ArchetypeFieldSource<32>>,
        boxed_field(3200) as Box<dyn ArchetypeFieldSource<32>>,
    ];
    let quest_cluster = Box::new(ClusterHolder::<3, 32>::new(quest_blend, quest_fields));

    let domain_directions: [[f32; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let projection_indices: [Vec<usize>; 3] =
        [(0..8).collect(), (0..16).collect(), (0..32).collect()];

    VariableRankRouter::<3, 32, 3>::new(
        [move_cluster, combat_cluster, quest_cluster],
        projection_indices,
        domain_directions,
    )
}

// ─── Monomorphized macro router: 3 domains (Issue 189 T2/T3) ────────────────
//
// Same 3-domain topology as the dynamic `make_router()` above, but generated
// by the `variable_rank_router_static!` macro — zero `Box<dyn>`, zero vtable
// dispatch. This is the monomorphization escape hatch that eliminates the 4
// virtual calls per tick (3× override_pi + 1× apply_blended).

variable_rank_router_static! {
    /// 3-domain monomorphized router: move (K=12, L=8) + combat (K=6, L=16) + quest (K=3, L=32).
    /// Same topology as the dynamic `VariableRankRouter<3, 32, 3>` but with zero-vtable dispatch.
    pub struct StaticRouter3<3, 32, 3>;

    0 => move_cluster:   ClusterHolder<12, 8>  => [0, 1, 2, 3, 4, 5, 6, 7];
    1 => combat_cluster: ClusterHolder<6, 16>  => [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
    2 => quest_cluster:  ClusterHolder<3, 32>  => [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                                   16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31];
}

/// Build a macro router with the same cluster topology + seeds as `make_router()`.
fn make_static_router_3domain() -> StaticRouter3 {
    let mut move_blend = CommittedFieldBlend::<12, 8>::uncommitted();
    move_blend.tau = 1.0;
    let move_fields: [Box<dyn ArchetypeFieldSource<8>>; 12] =
        std::array::from_fn(|i| boxed_field::<8>(1000 + i) as Box<dyn ArchetypeFieldSource<8>>);
    let move_cluster = ClusterHolder::<12, 8>::new(move_blend, move_fields);

    let mut combat_blend = CommittedFieldBlend::<6, 16>::uncommitted();
    combat_blend.tau = 1.0;
    let combat_fields: [Box<dyn ArchetypeFieldSource<16>>; 6] =
        std::array::from_fn(|i| boxed_field::<16>(2000 + i) as Box<dyn ArchetypeFieldSource<16>>);
    let combat_cluster = ClusterHolder::<6, 16>::new(combat_blend, combat_fields);

    let mut quest_blend = CommittedFieldBlend::<3, 32>::uncommitted();
    quest_blend.tau = 1.0;
    let quest_fields: [Box<dyn ArchetypeFieldSource<32>>; 3] = [
        boxed_field(3000) as Box<dyn ArchetypeFieldSource<32>>,
        boxed_field(3100) as Box<dyn ArchetypeFieldSource<32>>,
        boxed_field(3200) as Box<dyn ArchetypeFieldSource<32>>,
    ];
    let quest_cluster = ClusterHolder::<3, 32>::new(quest_blend, quest_fields);

    StaticRouter3::new(
        move_cluster,
        combat_cluster,
        quest_cluster,
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    )
}

/// Build a macro router with per-NPC pi pre-baked (production shape).
fn make_static_router_3domain_with_pi(
    pi_move: &[f32; 12],
    pi_combat: &[f32; 6],
    pi_quest: &[f32; 3],
) -> StaticRouter3 {
    let mut router = make_static_router_3domain();
    router.override_cluster_pi(0, pi_move);
    router.override_cluster_pi(1, pi_combat);
    router.override_cluster_pi(2, pi_quest);
    router
}

// ═══════════════════════════════════════════════════════════════════════════════
// G1 — Correctness (no NaN, no panic across 10K random inputs)
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn g1_correctness_no_nan_across_10k_inputs() {
    let router = make_router();
    let n: u64 = 10_000;
    for i in 0..n {
        let seed = i.wrapping_add(1).wrapping_mul(6364136223846793005);
        let (z, activity) = npc_state(seed);
        let mut scratch = [0.0f32; 96];
        let mut dz_out = [0.0f32; 32];
        let verdict = router.tick(&z, &activity, &mut scratch, &mut dz_out);
        assert!(verdict.domain < 3, "domain out of range");
        for v in dz_out.iter() {
            assert!(v.is_finite(), "NaN in dz_out at iter {i}: {dz_out:?}");
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// G2 — Perf: variable-rank vs uniform baseline (1K + 10K NPCs)
// ═══════════════════════════════════════════════════════════════════════════════
//
// Pass criterion: variable-rank ≤ 1.0× baseline in release. Variable-rank
// should NOT be slower because the masked-out dims skip blend work.
// Debug builds will be slower (no inlining); this test is meaningful in release.

const N_NPCS_1K: usize = 1_000;
const N_NPCS_10K: usize = 10_000;

/// NPC data tuple: (state, activity, pi_baseline, pi_move, pi_combat, pi_quest).
type NpcData = ([f32; 32], [f32; 3], [f32; 3], [f32; 12], [f32; 6], [f32; 3]);

/// Run `body` `passes` times and return the **minimum** elapsed time in ns.
/// The minimum filters out noise from system load / frequency scaling — it's
/// the best-case timing, which is the closest to the true instruction cost.
/// The first 2 iterations act as warm-up (cache + branch predictor priming).
fn time_min_passes(passes: usize, mut body: impl FnMut()) -> f64 {
    // Warm-up passes (2) — prime the cache + branch predictor.
    body();
    body();
    let mut best_ns = u128::MAX;
    for _ in 0..passes {
        let t0 = Instant::now();
        body();
        let elapsed = t0.elapsed().as_nanos();
        if elapsed < best_ns {
            best_ns = elapsed;
        }
    }
    best_ns as f64
}

/// Generate NPC states + per-NPC pi vectors (simulating per-entity committed
/// personalities — in production each NPC has its own CommittedFieldBlend
/// with a distinct pi; the bench simulates this by overriding the shared
/// router's cluster pi per-NPC before each tick).
fn generate_npcs(n_npcs: usize) -> Vec<NpcData> {
    (0..n_npcs)
        .map(|i| {
            let seed = (i as u64).wrapping_add(1).wrapping_mul(6364136223846793005);
            let (state, activity) = npc_state(seed);
            let pi_baseline = [prng(seed + 10), prng(seed + 20), prng(seed + 30)];
            let pi_move = std::array::from_fn(|k| prng(seed + 100 + k as u64));
            let pi_combat = std::array::from_fn(|k| prng(seed + 200 + k as u64));
            let pi_quest = [prng(seed + 300), prng(seed + 310), prng(seed + 320)];
            (state, activity, pi_baseline, pi_move, pi_combat, pi_quest)
        })
        .collect()
}

#[test]
#[ignore = "GOAT G2 perf bench — run with --ignored in release mode"]
fn g2_perf_variable_rank_vs_baseline_1k() {
    g2_perf_inner(N_NPCS_1K, "1K");
}

#[test]
#[ignore = "GOAT G2 perf bench — run with --ignored in release mode"]
fn g2_perf_variable_rank_vs_baseline_10k() {
    g2_perf_inner(N_NPCS_10K, "10K");
}

fn g2_perf_inner(n_npcs: usize, label: &str) {
    let npcs = generate_npcs(n_npcs);

    // ── Baseline run ─────────────────────────────────────────────────
    // Baseline has 3 archetypes → entropy max = log2(3) ≈ 1.585 bits.
    let mut baseline = Baseline::new();
    let t0 = Instant::now();
    let mut baseline_winners = [0usize; 3];
    for (state, _, pi_baseline, _, _, _) in &npcs {
        let winner = baseline.tick(state, pi_baseline);
        baseline_winners[winner] += 1;
    }
    let baseline_latency_ns = t0.elapsed().as_nanos() as f64 / n_npcs as f64;

    // ── Variable-rank run ────────────────────────────────────────────
    // Variable-rank router has 3 domains × per-domain archetypes (12 + 6 + 3 = 21
    // total winner slots). The entropy is measured over the (domain, winner) pair,
    // matching the Research 453 PoC's per-archetype accounting. Max entropy is
    // NOT log2(21) because the domain gate distributes NPCs across domains first;
    // the effective max is the weighted sum of per-domain log2(K_d).
    //
    // We use a flat 21-bin histogram indexed as domain*MAX_K + winner, then
    // compute Shannon entropy over the non-empty bins.
    const FLAT_BINS: usize = 36; // 3 domains × max(K_d)=12 archetypes
    let mut router = make_router();
    let t0 = Instant::now();
    let mut router_bins = [0usize; FLAT_BINS];
    for (state, activity, _, pi_move, pi_combat, pi_quest) in &npcs {
        // Per-NPC pi override (simulates each NPC's committed personality).
        router.cluster_mut(0).override_pi(pi_move);
        router.cluster_mut(1).override_pi(pi_combat);
        router.cluster_mut(2).override_pi(pi_quest);
        let mut scratch = [0.0f32; 96];
        let mut dz_out = [0.0f32; 32];
        let verdict: RoutingVerdict = router.tick(state, activity, &mut scratch, &mut dz_out);
        let flat = verdict.domain * 12 + verdict.winner; // 12 = max K_d
        debug_assert!(flat < FLAT_BINS);
        router_bins[flat] += 1;
    }
    let router_latency_ns = t0.elapsed().as_nanos() as f64 / n_npcs as f64;

    let ratio = router_latency_ns / baseline_latency_ns;
    let baseline_entropy = shannon_entropy(&baseline_winners);
    let router_entropy = shannon_entropy(&router_bins);
    let entropy_ratio = if baseline_entropy > 1e-9 {
        router_entropy / baseline_entropy
    } else {
        f32::INFINITY
    };

    println!("\n═══ Plan 558 G2 Perf Bench ({label} NPCs) ═══");
    println!(
        "  Baseline <3,32>:    {baseline_latency_ns:.1} ns/NPC, entropy = {baseline_entropy:.3} bits"
    );
    println!(
        "  Variable-rank:      {router_latency_ns:.1} ns/NPC, entropy = {router_entropy:.3} bits"
    );
    println!("  Latency ratio:      {ratio:.3}× (pass: ≤ 1.0× in release)");
    println!("  Entropy ratio:      {entropy_ratio:.3}× (G3 target: ≥ 1.5×)");
    println!("  Baseline winners:   {baseline_winners:?}");
    println!("  Router bins:        {router_bins:?}");
    println!("═══════════════════════════════════════════════\n");

    // G3 entropy assertion (always enforced — the quality claim).
    assert!(
        entropy_ratio >= 1.5,
        "G3 FAIL: entropy ratio {entropy_ratio:.3}× < 1.5× target. \
         Variable-rank did not produce sufficient archetype diversity gain."
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// G2 — Monomorphization re-gate (Issue 189 T3)
// ═══════════════════════════════════════════════════════════════════════════════
//
// Two bench shapes measure the macro router's vtable-elimination gain:
//
// 1. **Shared-router** (conservative bound): 1 shared `StaticRouter3` + per-NPC
//    `override_cluster_pi`. Same shape as the dynamic bench above, but with
//    zero-vtable dispatch. Recovers the 4 vtable calls (3× override_pi +
//    1× apply_blended).
//
// 2. **Production-shape** (realistic): per-NPC-owned `StaticRouter3` routers
//    with pre-baked pi. Hot loop calls `tick()` only — zero override_pi, zero
//    vtable. This is the production lower bound (domain gate + projection +
//    apply + scatter).
//
// Caveat on the production shape: each NPC owns its own boxed archetype fields,
// scattered across the heap → cache misses during iteration. In production the
// frozen fields would be shared; only pi is per-NPC. This bench overestimates
// production cost — the true floor is lower.

#[test]
#[ignore = "GOAT G2 perf bench — run with --ignored in release mode"]
fn g2_perf_macro_shared_router_1k() {
    g2_perf_macro_shared_inner(N_NPCS_1K, "1K");
}

#[test]
#[ignore = "GOAT G2 perf bench — run with --ignored in release mode"]
fn g2_perf_macro_shared_router_10k() {
    g2_perf_macro_shared_inner(N_NPCS_10K, "10K");
}

fn g2_perf_macro_shared_inner(n_npcs: usize, label: &str) {
    let npcs = generate_npcs(n_npcs);

    // Baseline run (same baseline as the dynamic bench).
    let mut baseline = Baseline::new();
    let baseline_latency_ns = time_min_passes(5, || {
        for (state, _, pi_baseline, _, _, _) in &npcs {
            baseline.tick(state, pi_baseline);
        }
    }) / n_npcs as f64;

    // Macro shared-router run: 3× override_cluster_pi + 1× tick per NPC.
    // This is the conservative bound — same work as the dynamic router but
    // with zero vtable dispatch (all monomorphized inherent method calls).
    let mut router = make_static_router_3domain();
    let macro_latency_ns = time_min_passes(5, || {
        for (state, activity, _, pi_move, pi_combat, pi_quest) in &npcs {
            router.override_cluster_pi(0, pi_move);
            router.override_cluster_pi(1, pi_combat);
            router.override_cluster_pi(2, pi_quest);
            let mut scratch = [0.0f32; 96];
            let mut dz_out = [0.0f32; 32];
            router.tick(state, activity, &mut scratch, &mut dz_out);
        }
    }) / n_npcs as f64;

    let ratio = macro_latency_ns / baseline_latency_ns;
    println!("\n═══ Issue 189 T3 — Macro Shared-Router ({label} NPCs) ═══");
    println!("  Baseline <3,32>:    {baseline_latency_ns:.1} ns/NPC");
    println!("  Macro shared:       {macro_latency_ns:.1} ns/NPC");
    println!("  Latency ratio:      {ratio:.3}× (target: ≤ 1.0×)");
    println!("═══════════════════════════════════════════════\n");
}

#[test]
#[ignore = "GOAT G2 perf bench — run with --ignored in release mode"]
fn g2_perf_macro_production_1k() {
    g2_perf_macro_production_inner(N_NPCS_1K, "1K");
}

#[test]
#[ignore = "GOAT G2 perf bench — run with --ignored in release mode"]
fn g2_perf_macro_production_10k() {
    g2_perf_macro_production_inner(N_NPCS_10K, "10K");
}

fn g2_perf_macro_production_inner(n_npcs: usize, label: &str) {
    let npcs = generate_npcs(n_npcs);

    // Pre-construct per-NPC routers with pre-baked pi (NOT timed).
    // Each NPC owns its own StaticRouter3 with its own committed pi.
    let routers: Vec<StaticRouter3> = npcs
        .iter()
        .map(|(_, _, _, pi_move, pi_combat, pi_quest)| {
            make_static_router_3domain_with_pi(pi_move, pi_combat, pi_quest)
        })
        .collect();

    // Baseline run (same baseline as the dynamic bench).
    let mut baseline = Baseline::new();
    let baseline_latency_ns = time_min_passes(5, || {
        for (state, _, pi_baseline, _, _, _) in &npcs {
            baseline.tick(state, pi_baseline);
        }
    }) / n_npcs as f64;

    // Macro production run: tick() only — no override_pi, no vtable.
    let macro_latency_ns = time_min_passes(5, || {
        for (i, (state, activity, _, _, _, _)) in npcs.iter().enumerate() {
            let mut scratch = [0.0f32; 96];
            let mut dz_out = [0.0f32; 32];
            routers[i].tick(state, activity, &mut scratch, &mut dz_out);
        }
    }) / n_npcs as f64;

    let ratio = macro_latency_ns / baseline_latency_ns;
    println!("\n═══ Issue 189 T3 — Macro Production-Shape ({label} NPCs) ═══");
    println!("  Baseline <3,32>:    {baseline_latency_ns:.1} ns/NPC");
    println!("  Macro production:   {macro_latency_ns:.1} ns/NPC");
    println!("  Latency ratio:      {ratio:.3}× (target: ≤ 1.0×)");
    println!("  (per-NPC-owned router, tick only — no override_pi)");
    println!("═══════════════════════════════════════════════\n");
}

// ═══════════════════════════════════════════════════════════════════════════════
// G3 — Entropy: variable-rank produces ≥ 1.5× archetype utilization entropy
// ═══════════════════════════════════════════════════════════════════════════════
//
// This is the headline Research 453 result: variable-rank at iso K×D=96 compute
// produces 1.63× higher entropy than uniform <3,32>. The entropy assertion is
// embedded in the G2 perf bench (above) so both metrics run together. The
// separate g3 test below is the always-on (non-ignored) variant at smaller N.

#[test]
fn g3_entropy_at_1k() {
    const N: usize = 1_000;
    let mut baseline = Baseline::new();
    let mut router = make_router();

    let mut baseline_winners = [0usize; 3];
    const FLAT_BINS: usize = 36; // 3 domains × max(K_d)=12
    let mut router_bins = [0usize; FLAT_BINS];
    for i in 0..N {
        let seed = (i as u64).wrapping_add(1).wrapping_mul(6364136223846793005);
        let (state, activity) = npc_state(seed);
        let pi_baseline = [prng(seed + 10), prng(seed + 20), prng(seed + 30)];
        let w = baseline.tick(&state, &pi_baseline);
        baseline_winners[w] += 1;

        // Per-NPC pi override (simulates per-entity committed personality).
        let pi_move: [f32; 12] = std::array::from_fn(|k| prng(seed + 100 + k as u64));
        let pi_combat: [f32; 6] = std::array::from_fn(|k| prng(seed + 200 + k as u64));
        let pi_quest: [f32; 3] = [prng(seed + 300), prng(seed + 310), prng(seed + 320)];
        router.cluster_mut(0).override_pi(&pi_move);
        router.cluster_mut(1).override_pi(&pi_combat);
        router.cluster_mut(2).override_pi(&pi_quest);

        let mut scratch = [0.0f32; 96];
        let mut dz_out = [0.0f32; 32];
        let verdict = router.tick(&state, &activity, &mut scratch, &mut dz_out);
        let flat = verdict.domain * 12 + verdict.winner;
        debug_assert!(flat < FLAT_BINS);
        router_bins[flat] += 1;
    }

    let baseline_entropy = shannon_entropy(&baseline_winners);
    let router_entropy = shannon_entropy(&router_bins);
    let ratio = router_entropy / baseline_entropy.max(1e-9);

    println!("\n═══ Plan 558 G3 Entropy (1K NPCs) ═══");
    println!("  Baseline entropy: {baseline_entropy:.3} bits (winners {baseline_winners:?})");
    println!("  Router entropy:   {router_entropy:.3} bits (bins {router_bins:?})");
    println!("  Ratio:            {ratio:.3}× (target ≥ 1.5×)");
    println!("═══════════════════════════════════════\n");

    assert!(
        ratio >= 1.5,
        "G3 FAIL: entropy ratio {ratio:.3}× < 1.5× target"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// G5 — Determinism: same inputs → same outputs across runs
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn g5_determinism_bit_identical_across_runs() {
    let router1 = make_router();
    let router2 = make_router();

    for i in 0..100 {
        let seed = (i as u64).wrapping_add(1).wrapping_mul(6364136223846793005);
        let (state, activity) = npc_state(seed);

        let mut scratch1 = [0.0f32; 96];
        let mut dz_out1 = [0.0f32; 32];
        let v1 = router1.tick(&state, &activity, &mut scratch1, &mut dz_out1);

        let mut scratch2 = [0.0f32; 96];
        let mut dz_out2 = [0.0f32; 32];
        let v2 = router2.tick(&state, &activity, &mut scratch2, &mut dz_out2);

        assert_eq!(v1, v2, "verdict mismatch at iter {i}");
        assert_eq!(
            dz_out1, dz_out2,
            "dz_out bit-mismatch at iter {i}: {dz_out1:?} vs {dz_out2:?}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Sanity: pick_domain + project_guided + scatter_guided basics
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn sanity_pick_project_scatter() {
    // pick_domain
    let activity = [0.1, 0.8, 0.1];
    let dirs: [[f32; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    assert_eq!(pick_domain::<3, 3>(&activity, &dirs), 1);

    // project_guided
    let z_full = [0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    let indices = [0usize, 2, 4, 6];
    let mut z_out = [0.0f32; 4];
    project_guided::<8, 4>(&z_full, &indices, &mut z_out);
    assert_eq!(z_out, [0.0, 2.0, 4.0, 6.0]);

    // scatter_guided round-trip
    let mut dz_full = [0.0f32; 8];
    scatter_guided::<8, 4>(&z_out, &indices, &mut dz_full);
    assert_eq!(dz_full[0], 0.0);
    assert_eq!(dz_full[2], 2.0);
    assert_eq!(dz_full[4], 4.0);
    assert_eq!(dz_full[6], 6.0);
}
