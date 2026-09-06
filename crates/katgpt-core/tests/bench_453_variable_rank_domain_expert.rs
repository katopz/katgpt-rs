//! PoC: Variable-Rank Domain Expert Clusters (Research 453)
//!
//! Tests whether domain-specific variable-rank expert routing produces
//! higher archetype utilization entropy than uniform-rank CommittedFieldBlend,
//! at the same or lower per-tick compute cost.
//!
//! Inspired by LatentMoE [arXiv:2601.18089] — but applied to per-NPC cognition:
//! movement (ℓ=8, K'=12), combat (ℓ=16, K'=6), quest (ℓ=32, K'=3).
//! The total K×D is kept constant at 96 per domain (LatentMoE α-scaling).
//!
//! Run: cargo test -p katgpt-core --features committed_field_blend \
//!      --test bench_453_variable_rank_domain_expert -- --nocapture --ignored

#![cfg(feature = "committed_field_blend")]

use katgpt_core::committed_field_blend::{ArchetypeFieldSource, CommittedFieldBlend};
use std::time::Instant;

// ─── Deterministic direction field ──────────────────────────────────────────

/// A frozen archetype field that produces `direction · dot(z, direction)`.
/// The output depends on how well the NPC state aligns with this archetype's
/// direction — different NPC states produce different outputs even for the
/// same archetype. This mirrors real game behavior (seek-food field produces
/// different velocities depending on NPC position relative to food).
struct DirectionField<const D: usize> {
    direction: [f32; D],
    blake3: [u8; 32],
}

impl<const D: usize> DirectionField<D> {
    fn new(seed: usize) -> Self {
        let mut direction = [0.0f32; D];
        for (i, d) in direction.iter_mut().enumerate() {
            let x = (seed * 37 + i * 13) as f32;
            *d = ((x * 0.1).sin() + (x * 0.07).cos()) * 0.5;
        }
        // Normalize to unit length
        let norm: f32 = direction.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-8);
        for v in direction.iter_mut() {
            *v /= norm;
        }
        // Deterministic fake BLAKE3 (PoC only — real code uses blake3::Hasher)
        let mut blake3 = [0u8; 32];
        for (i, b) in blake3.iter_mut().enumerate() {
            *b = ((seed * 251 + i) & 0xFF) as u8;
        }
        Self { direction, blake3 }
    }
}

impl<const D: usize> ArchetypeFieldSource<D> for DirectionField<D> {
    fn evolve<'a>(&self, z: &[f32], dz_scratch: &'a mut [f32]) -> &'a mut [f32] {
        // f_k(z) = direction_k · dot(z, direction_k)
        let dot: f32 = z.iter().zip(self.direction.iter()).map(|(zi, di)| zi * di).sum();
        for (dz, di) in dz_scratch[..D].iter_mut().zip(self.direction.iter()) {
            *dz = di * dot;
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

// ─── Helpers ────────────────────────────────────────────────────────────────

#[inline]
#[allow(dead_code)]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Shannon entropy (base-2) of a distribution.
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

/// Generate a deterministic pseudo-random f32 in [-1, 1) from a seed.
fn prng(seed: u64) -> f32 {
    // xorshift64
    let mut x = seed.wrapping_mul(0x2545F4914F6CDD1D).wrapping_add(1);
    x ^= x >> 13;
    x ^= x << 7;
    x ^= x >> 17;
    ((x >> 11) as f32 / (1u64 << 53) as f32) * 2.0 - 1.0
}

/// Deterministic NPC state generator (D=32).
fn npc_state(seed: u64) -> ([f32; 32], [f32; 3]) {
    let mut state = [0.0f32; 32];
    for (i, s) in state.iter_mut().enumerate() {
        *s = prng(seed.wrapping_mul(31).wrapping_add(i as u64));
    }
    // Activity vector: which domain is this NPC currently engaged in?
    let activity = [
        prng(seed.wrapping_mul(7).wrapping_add(1)), // move
        prng(seed.wrapping_mul(7).wrapping_add(2)), // combat
        prng(seed.wrapping_mul(7).wrapping_add(3)), // quest
    ];
    (state, activity)
}

/// Domain gate: pick the highest-activity domain.
/// Returns 0=move, 1=combat, 2=quest.
fn domain_gate(activity: &[f32; 3]) -> usize {
    activity
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| katgpt_core::float_order::cmp_for_max(**a, **b)).map_or(0, |(i, _)| i)
}

// ─── Baseline: uniform CommittedFieldBlend<3, 32> ───────────────────────────

struct Baseline {
    blend: CommittedFieldBlend<3, 32>,
    fields: [DirectionField<32>; 3],
}

impl Baseline {
    fn new() -> Self {
        let mut blend = CommittedFieldBlend::<3, 32>::uncommitted();
        // Non-uniform pi so archetypes are differentiated
        blend.pi = [0.5, -0.3, 0.8];
        blend.tau = 1.0;
        Self {
            blend,
            fields: [
                DirectionField::new(100),
                DirectionField::new(200),
                DirectionField::new(300),
            ],
        }
    }

    /// Returns (output[D], winning_archetype_index).
    fn tick(&mut self, z: &[f32; 32], pi_override: &[f32; 3]) -> ([f32; 32], usize) {
        let fields_ref: [&dyn ArchetypeFieldSource<32>; 3] =
            [&self.fields[0], &self.fields[1], &self.fields[2]];

        let mut scratch = [0.0f32; 32];
        let mut out = [0.0f32; 32];

        // Apply blend with overridden pi (per-NPC personality)
        self.blend.pi = *pi_override;
        self.blend.apply_blended(&fields_ref, z, &mut scratch, &mut out);

        // Winning archetype = highest gate weight = highest pi (sigmoid monotonic)
        let winner = (0..3)
            .map(|k| (k, pi_override[k]))
            .max_by(|(_, a), (_, b)| katgpt_core::float_order::cmp_for_max(*a, *b)).map_or(0, |(k, _)| k);

        (out, winner)
    }
}

// ─── Variable-rank: domain-specific clusters ────────────────────────────────

struct MoveCluster {
    blend: CommittedFieldBlend<12, 8>,
    fields: [DirectionField<8>; 12],
}

struct CombatCluster {
    blend: CommittedFieldBlend<6, 16>,
    fields: [DirectionField<16>; 6],
}

struct QuestCluster {
    blend: CommittedFieldBlend<3, 32>,
    fields: [DirectionField<32>; 3],
}

impl MoveCluster {
    fn new() -> Self {
        let blend = CommittedFieldBlend::<12, 8>::uncommitted();
        let fields = std::array::from_fn(|i| DirectionField::new(1000 + i));
        Self { blend, fields }
    }

    fn tick(&mut self, z: &[f32; 32], pi_override: &[f32; 12]) -> ([f32; 8], usize) {
        let fields_ref: [&dyn ArchetypeFieldSource<8>; 12] =
            std::array::from_fn(|i| &self.fields[i] as &dyn ArchetypeFieldSource<8>);

        // Guided projection: select first 8 dims (move-relevant)
        let z_proj: &[f32] = &z[..8];

        let mut scratch = [0.0f32; 8];
        let mut out = [0.0f32; 8];

        self.blend.pi = *pi_override;
        self.blend.apply_blended(&fields_ref, z_proj, &mut scratch, &mut out);

        let winner = (0..12)
            .map(|k| (k, pi_override[k]))
            .max_by(|(_, a), (_, b)| katgpt_core::float_order::cmp_for_max(*a, *b)).map_or(0, |(k, _)| k);

        (out, winner)
    }
}

impl CombatCluster {
    fn new() -> Self {
        let blend = CommittedFieldBlend::<6, 16>::uncommitted();
        let fields = std::array::from_fn(|i| DirectionField::new(2000 + i));
        Self { blend, fields }
    }

    fn tick(&mut self, z: &[f32; 32], pi_override: &[f32; 6]) -> ([f32; 16], usize) {
        let fields_ref: [&dyn ArchetypeFieldSource<16>; 6] =
            std::array::from_fn(|i| &self.fields[i] as &dyn ArchetypeFieldSource<16>);

        // Guided projection: select first 16 dims (combat-relevant)
        let z_proj: &[f32] = &z[..16];

        let mut scratch = [0.0f32; 16];
        let mut out = [0.0f32; 16];

        self.blend.pi = *pi_override;
        self.blend.apply_blended(&fields_ref, z_proj, &mut scratch, &mut out);

        let winner = (0..6)
            .map(|k| (k, pi_override[k]))
            .max_by(|(_, a), (_, b)| katgpt_core::float_order::cmp_for_max(*a, *b)).map_or(0, |(k, _)| k);

        (out, winner)
    }
}

impl QuestCluster {
    fn new() -> Self {
        let blend = CommittedFieldBlend::<3, 32>::uncommitted();
        let fields = [
            DirectionField::new(3000),
            DirectionField::new(3100),
            DirectionField::new(3200),
        ];
        Self { blend, fields }
    }

    fn tick(&mut self, z: &[f32; 32], pi_override: &[f32; 3]) -> ([f32; 32], usize) {
        let fields_ref: [&dyn ArchetypeFieldSource<32>; 3] =
            std::array::from_fn(|i| &self.fields[i] as &dyn ArchetypeFieldSource<32>);

        let mut scratch = [0.0f32; 32];
        let mut out = [0.0f32; 32];

        self.blend.pi = *pi_override;
        self.blend.apply_blended(&fields_ref, z, &mut scratch, &mut out);

        let winner = (0..3)
            .map(|k| (k, pi_override[k]))
            .max_by(|(_, a), (_, b)| katgpt_core::float_order::cmp_for_max(*a, *b)).map_or(0, |(k, _)| k);

        (out, winner)
    }
}

// ─── PoC tests ──────────────────────────────────────────────────────────────

const N_NPCS: usize = 1000;

/// (state, pi_baseline, pi_move, pi_combat, domain) per generated NPC.
type NpcRecord = ([f32; 32], [f32; 3], [f32; 12], [f32; 6], usize);

#[test]
#[ignore = "PoC bench — run with --ignored"]
fn poc_variable_rank_domain_expert() {
    let mut baseline = Baseline::new();
    let mut move_cluster = MoveCluster::new();
    let mut combat_cluster = CombatCluster::new();
    let mut quest_cluster = QuestCluster::new();

    // Generate NPC states + per-NPC pi vectors + pre-computed domain
    let npcs: Vec<NpcRecord> = (0..N_NPCS)
        .map(|i| {
            let seed = (i as u64).wrapping_add(1).wrapping_mul(6364136223846793005);
            let (state, activity) = npc_state(seed);
            let domain = domain_gate(&activity);
            // Per-NPC personality weights for each domain
            let pi_baseline = [
                prng(seed + 10),
                prng(seed + 20),
                prng(seed + 30),
            ];
            let pi_move = std::array::from_fn(|k| prng(seed + 100 + k as u64));
            let pi_combat = std::array::from_fn(|k| prng(seed + 200 + k as u64));
            (state, pi_baseline, pi_move, pi_combat, domain)
        })
        .collect();

    // ─── Baseline run ─────────────────────────────────────────────────
    let mut baseline_counts = [0usize; 3];
    let t0 = Instant::now();
    for (state, pi_baseline, _, _, _) in &npcs {
        let (_out, winner) = baseline.tick(state, pi_baseline);
        baseline_counts[winner] += 1;
    }
    let baseline_latency_ns = t0.elapsed().as_nanos() as f64 / N_NPCS as f64;
    let baseline_entropy = shannon_entropy(&baseline_counts);

    // ─── Variable-rank run ─────────────────────────────────────────
    let mut move_counts = [0usize; 12];
    let mut combat_counts = [0usize; 6];
    let mut quest_counts = [0usize; 3];
    let mut domain_counts = [0usize; 3]; // move, combat, quest
    let t1 = Instant::now();
    for (state, _, pi_move, pi_combat, domain) in &npcs {
        domain_counts[*domain] += 1;
        match domain {
            0 => {
                let (_out, winner) = move_cluster.tick(state, pi_move);
                move_counts[winner] += 1;
            }
            1 => {
                let (_out, winner) = combat_cluster.tick(state, pi_combat);
                combat_counts[winner] += 1;
            }
            2 => {
                let pi_quest = [pi_move[0], pi_move[1], pi_move[2]];
                let (_out, winner) = quest_cluster.tick(state, &pi_quest);
                quest_counts[winner] += 1;
            }
            _ => unreachable!(),
        }
    }
    let variable_latency_ns = t1.elapsed().as_nanos() as f64 / N_NPCS as f64;

    // Combined variable-rank entropy: weighted by domain distribution
    let total_domain = domain_counts.iter().sum::<usize>() as f32;
    let move_entropy = shannon_entropy(&move_counts);
    let combat_entropy = shannon_entropy(&combat_counts);
    let quest_entropy = shannon_entropy(&quest_counts);
    let variable_entropy = if total_domain > 0.0 {
        (domain_counts[0] as f32 / total_domain) * move_entropy
            + (domain_counts[1] as f32 / total_domain) * combat_entropy
            + (domain_counts[2] as f32 / total_domain) * quest_entropy
    } else {
        0.0
    };

    // ─── G1: correctness ────────────────────────────────────────────────
    let baseline_total: usize = baseline_counts.iter().sum();
    let variable_total: usize = domain_counts.iter().sum();
    let g1_pass = baseline_total == N_NPCS && variable_total == N_NPCS;

    // ─── G2: latency ────────────────────────────────────────────────────
    let latency_ratio = variable_latency_ns / baseline_latency_ns.max(1.0);
    let g2_pass = latency_ratio <= 2.0; // Allow 2× overhead for domain gate + projection

    // ─── G3: entropy ────────────────────────────────────────────────────
    let entropy_ratio = variable_entropy / baseline_entropy.max(0.01);
    let g3_pass = entropy_ratio >= 1.0; // Variable-rank should be ≥ baseline

    // ─── Report ─────────────────────────────────────────────────────────
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║  Research 453 PoC: Variable-Rank Domain Expert Clusters     ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  N_NPCS = {N_NPCS}                                              ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  BASELINE: CommittedFieldBlend<3, 32> (uniform D=32)       ║");
    println!("║    Archetype wins: {baseline_counts:?}");
    println!("║    Entropy:        {:.4} bits (max=log₂(3)={:.4})", baseline_entropy, (3.0f32).log2());
    println!("║    Latency:        {baseline_latency_ns:.1} ns/NPC");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  VARIABLE-RANK: domain gate → project → per-domain blend    ║");
    println!("║    Domain split:  move={} combat={} quest={}",
        domain_counts[0], domain_counts[1], domain_counts[2]);
    println!("║    Move wins:     {move_counts:?}");
    println!("║    Combat wins:   {combat_counts:?}");
    println!("║    Quest wins:    {quest_counts:?}");
    println!("║    Move entropy:  {:.4} bits (max=log₂(12)={:.4})", move_entropy, (12.0f32).log2());
    println!("║    Combat entropy:{:.4} bits (max=log₂(6)={:.4})", combat_entropy, (6.0f32).log2());
    println!("║    Quest entropy: {:.4} bits (max=log₂(3)={:.4})", quest_entropy, (3.0f32).log2());
    println!("║    Weighted avg:  {variable_entropy:.4} bits");
    println!("║    Latency:       {variable_latency_ns:.1} ns/NPC");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  COMPARISON                                                 ║");
    println!("║    Entropy ratio:  {entropy_ratio:.2}× (variable / baseline)");
    println!("║    Latency ratio:  {latency_ratio:.2}× (variable / baseline)");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  GATES                                                      ║");
    println!("║    G1 correctness: {} (all {} NPCs processed)", if g1_pass { "✅ PASS" } else { "❌ FAIL" }, N_NPCS);
    println!("║    G2 latency:     {} (≤2.0× baseline, got {:.2}×)", if g2_pass { "✅ PASS" } else { "❌ FAIL" }, latency_ratio);
    println!("║    G3 entropy:     {} (≥1.0× baseline, got {:.2}×)", if g3_pass { "✅ PASS" } else { "❌ FAIL" }, entropy_ratio);
    println!("╠══════════════════════════════════════════════════════════════╣");

    let all_pass = g1_pass && g2_pass && g3_pass;
    if all_pass {
        println!("║  VERDICT: ✅ PASS — variable-rank ≥ baseline on all gates   ║");
    } else {
        println!("║  VERDICT: ⚠️ MIXED — see individual gates above             ║");
    }
    println!("╚══════════════════════════════════════════════════════════════╝");

    assert!(g1_pass, "G1 correctness failed");
}

#[test]
fn g1_correctness_no_nan_no_collapse() {
    let mut baseline = Baseline::new();
    let mut move_cluster = MoveCluster::new();
    let mut combat_cluster = CombatCluster::new();

    for i in 0..100 {
        let seed = (i as u64 + 1) * 12345;
        let (state, _) = npc_state(seed);
        let pi3 = [prng(seed + 1), prng(seed + 2), prng(seed + 3)];

        // Baseline
        let (out, winner) = baseline.tick(&state, &pi3);
        assert!(out.iter().all(|v| v.is_finite()), "baseline NaN at NPC {i}");
        assert!(winner < 3, "baseline invalid winner");

        // Move cluster
        let pi12 = std::array::from_fn(|k| prng(seed + 100 + k as u64));
        let (out8, winner8) = move_cluster.tick(&state, &pi12);
        assert!(out8.iter().all(|v| v.is_finite()), "move NaN at NPC {i}");
        assert!(winner8 < 12, "move invalid winner");

        // Combat cluster
        let pi6 = std::array::from_fn(|k| prng(seed + 200 + k as u64));
        let (out16, winner16) = combat_cluster.tick(&state, &pi6);
        assert!(out16.iter().all(|v| v.is_finite()), "combat NaN at NPC {i}");
        assert!(winner16 < 6, "combat invalid winner");
    }
}

#[test]
fn g4_domain_gate_routes_correctly() {
    // When move activity is dominant, domain gate should pick move (0)
    assert_eq!(domain_gate(&[0.9, 0.1, 0.0]), 0);
    // When combat activity is dominant, pick combat (1)
    assert_eq!(domain_gate(&[0.1, 0.9, 0.0]), 1);
    // When quest activity is dominant, pick quest (2)
    assert_eq!(domain_gate(&[0.0, 0.1, 0.9]), 2);
}
