//! Plan 255 GOAT Arena — NPC Tick Simulation (Part 5).
//!
//! Simulates a "game" of TICKS ticks where each tick evaluates all NPC brains
//! through the `NpcBrainRouter` (auto-selects ANE or CPU). Compares:
//! - Config A: `NpcBrainRouter::cpu()` — always CPU SIMD
//! - Config B: `NpcBrainRouter::new(Some(path))` — auto-routes to ANE when ≥100 NPCs
//!
//! Verifies same game outcome (aggregate scalar), high per-tick cosine (≥0.99),
//! and reports throughput. The "game outcome" is the sum of all NPC projection
//! values on the final tick — a deterministic scalar representing aggregate
//! NPC emotional state.
//!
//! Usage:
//!   cargo run --example ane_npc_arena --features sense_composition --release
//!   cargo run --example ane_npc_arena --features ane_npc --release  # full ANE comparison

use katgpt_core::sense::backend::{NpcBrainBackend, NpcBrainInput, NpcBrainOutput};
use katgpt_core::sense::brain::NpcBrain;
use katgpt_core::sense::octree::{KgEmbedding, SenseOctreeBuilder};
use katgpt_core::types::SenseKind;
use katgpt_rs::npc_brain_router::NpcBrainRouter;

const NPC_COUNT: usize = 1000;
const TICKS: usize = 200;
const WARMUP_TICKS: usize = 5;

// Comparison-only constants — gated to match the ANE comparison block below.
#[cfg(all(feature = "ane_npc", target_os = "macos"))]
const ANE_MODEL_PATH: &str = "npc_brain.mlpackage";
#[cfg(all(feature = "ane_npc", target_os = "macos"))]
const OUTCOME_REL_DIFF_THRESHOLD: f32 = 0.01;
#[cfg(all(feature = "ane_npc", target_os = "macos"))]
const COSINE_THRESHOLD: f32 = 0.99;

// ── Deterministic PRNG (xorshift64*) ─────────────────────────────

struct SeedRng {
    state: u64,
}

impl SeedRng {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 {
                0xDEAD_BEEF_CAFE_BABE
            } else {
                seed
            },
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    fn next_f32(&mut self) -> f32 {
        let bits = ((self.next_u64() >> 41) as u32) | 0x3F80_0000;
        f32::from_bits(bits) - 1.0
    }

    fn next_range(&mut self, lo: usize, hi: usize) -> usize {
        if hi <= lo {
            return lo;
        }
        ((self.next_u64() as usize) % (hi - lo)) + lo
    }
}

// ── Brain generation ─────────────────────────────────────────────

const ALL_KINDS: [SenseKind; 6] = [
    SenseKind::CommonSense,
    SenseKind::FighterSense,
    SenseKind::GameTheorySense,
    SenseKind::SpatialSense,
    SenseKind::SocialSense,
    SenseKind::SkillSense,
];

fn make_diverse_brains(n: usize) -> Vec<NpcBrain> {
    let builder = SenseOctreeBuilder::new(3);
    let mut rng = SeedRng::new(0xC0FFEE);
    let mut brains = Vec::with_capacity(n);

    for npc_id in 0..n {
        let n_modules = rng.next_range(1, 7);
        let mut modules = Vec::with_capacity(n_modules);

        for m in 0..n_modules {
            let kind = ALL_KINDS[m % ALL_KINDS.len()];
            let n_embs = rng.next_range(1, 5);
            let mut embeddings = Vec::with_capacity(n_embs);

            for _ in 0..n_embs {
                let entity_hash = rng.next_u64();
                let relation_hash = rng.next_u64();
                let mut embedding = [0.0f32; 8];
                for e in &mut embedding {
                    *e = rng.next_f32() * 2.0 - 1.0;
                }
                let confidence = 0.1 + rng.next_f32() * 0.9;
                let sign = rng.next_u64() & 1 == 0;

                embeddings.push(KgEmbedding {
                    entity_hash,
                    relation_hash,
                    embedding,
                    sign,
                    confidence,
                });
            }

            let mut module = builder.build(kind, &embeddings);
            module.confidence = 0.1 + rng.next_f32() * 0.9;
            module.commit();
            modules.push(module);
        }

        let mut brain = NpcBrain::compose(modules);

        for v in &mut brain.hla_state {
            *v = rng.next_f32() * 2.0 - 1.0;
        }

        if npc_id % 10 == 0 {
            let pin_kind = ALL_KINDS[npc_id % ALL_KINDS.len()];
            brain.pin_sense(pin_kind, rng.next_f32());
        }

        if npc_id % 20 == 0 {
            brain.disable_autonomous(npc_id as u64);
        }

        brains.push(brain);
    }

    brains
}

// ── Cosine similarity ────────────────────────────────────────────

// Gated to match the ANE comparison block — only used when ANE is available.
#[cfg(all(feature = "ane_npc", target_os = "macos"))]
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < 1e-10 || nb < 1e-10 {
        return 0.0;
    }
    dot / (na * nb)
}

/// Flatten final-tick outputs into a single emotion-state vector for cosine.
#[cfg(all(feature = "ane_npc", target_os = "macos"))]
fn flatten_outputs(outputs: &[NpcBrainOutput]) -> Vec<f32> {
    let mut v = Vec::with_capacity(outputs.len() * 6);
    for o in outputs {
        for p in &o.projections {
            v.push(*p);
        }
    }
    v
}

/// Game outcome = sum of all NPC projection values (deterministic scalar).
#[cfg(all(feature = "ane_npc", target_os = "macos"))]
fn aggregate_outcome(outputs: &[NpcBrainOutput]) -> f32 {
    let mut sum = 0.0f32;
    for o in outputs {
        for p in &o.projections {
            sum += *p;
        }
    }
    sum
}

// ── Run a config for TICKS ticks ─────────────────────────────────

struct RunResult {
    per_tick_us: f64,
    #[cfg(all(feature = "ane_npc", target_os = "macos"))]
    final_outputs: Vec<NpcBrainOutput>,
    #[cfg(all(feature = "ane_npc", target_os = "macos"))]
    aggregate: f32,
}

fn run_config(
    mut router: NpcBrainRouter,
    inputs: &[NpcBrainInput],
    label: &str,
) -> RunResult {
    let backend_name = router.backend_name();
    let is_ane = router.is_ane();
    let n = inputs.len();

    let mut outputs = vec![NpcBrainOutput::default(); n];

    for _ in 0..WARMUP_TICKS {
        router.batch_evaluate(inputs, &mut outputs).unwrap();
    }

    let start = std::time::Instant::now();
    for _ in 0..TICKS {
        router.batch_evaluate(inputs, &mut outputs).unwrap();
    }
    let elapsed = start.elapsed();

    let total_us = elapsed.as_secs_f64() * 1e6;
    let per_tick_us = total_us / TICKS as f64;
    let ticks_per_sec = if total_us > 0.0 {
        1e6 / per_tick_us
    } else {
        f64::INFINITY
    };

    let ane_tag = match is_ane {
        true => " (ANE active)",
        false => "",
    };
    println!("── Config {label}: routed backend = {backend_name}{ane_tag} ──");
    println!("  Total time: {total_us:.1} ms ({per_tick_us:.1} µs/tick)");
    println!("  Throughput: {ticks_per_sec:.0} ticks/sec");
    println!();

    #[cfg(all(feature = "ane_npc", target_os = "macos"))]
    {
        let aggregate = aggregate_outcome(&outputs);
        println!("  Final aggregate outcome: {aggregate:.4}");
        println!();
        RunResult {
            per_tick_us,
            final_outputs: outputs,
            aggregate,
        }
    }

    #[cfg(not(all(feature = "ane_npc", target_os = "macos")))]
    RunResult { per_tick_us }
}

// ── Main ─────────────────────────────────────────────────────────

fn main() {
    let gameplay_seconds = TICKS as f32 / 20.0;
    println!("=== Plan 255 GOAT Arena — NPC Tick Simulation ===\n");
    println!(
        "NPCs: {NPC_COUNT}, Ticks: {TICKS} ({gameplay_seconds:.1}s of 20Hz gameplay)\n"
    );

    let brains = make_diverse_brains(NPC_COUNT);
    let inputs: Vec<NpcBrainInput> = brains.iter().map(NpcBrainInput::from_brain).collect();

    let config_a = run_config(NpcBrainRouter::cpu(), &inputs, "A");

    #[cfg(all(feature = "ane_npc", target_os = "macos"))]
    let config_b = run_config(
        NpcBrainRouter::new(Some(std::path::Path::new(ANE_MODEL_PATH))),
        &inputs,
        "B",
    );

    #[cfg(not(all(feature = "ane_npc", target_os = "macos")))]
    {
        println!("── Config B: ANE-routed ──");
        println!("  Skipped: ANE not available (build with --features ane_npc on macOS)");
        println!("  Only Config A (CPU) ran. Equivalence and throughput comparison require ANE.\n");
        println!("── Arena Verdict ──");
        println!("  [SKIP] Outcome equivalence: requires ANE build");
        println!("  [SKIP] Per-tick cosine: requires ANE build");
        println!(
            "  [INFO] CPU forced: {:.1} µs/tick",
            config_a.per_tick_us
        );
        println!();
        println!("ℹ️  GOAT Arena: CPU-only run complete. Rebuild with --features ane_npc for full comparison.");
    }

    #[cfg(all(feature = "ane_npc", target_os = "macos"))]
    {
        let config_b = config_b;

        println!("── Arena Verdict ──");

        let denom = config_a.aggregate.abs().max(config_b.aggregate.abs());
        let rel_diff = if denom > 1e-9 {
            (config_a.aggregate - config_b.aggregate).abs() / denom
        } else {
            0.0
        };
        let outcome_pass = rel_diff < OUTCOME_REL_DIFF_THRESHOLD;
        println!(
            "  [{}] Outcome equivalence: rel diff = {:.4}% (< {:.1}% required)",
            if outcome_pass { "PASS" } else { "FAIL" },
            rel_diff * 100.0,
            OUTCOME_REL_DIFF_THRESHOLD * 100.0
        );

        let flat_a = flatten_outputs(&config_a.final_outputs);
        let flat_b = flatten_outputs(&config_b.final_outputs);
        let cos = cosine_similarity(&flat_a, &flat_b);
        let cosine_pass = cos >= COSINE_THRESHOLD;
        println!(
            "  [{}] Per-tick cosine: {:.6} (≥ {:.2} required)",
            if cosine_pass { "PASS" } else { "FAIL" },
            cos,
            COSINE_THRESHOLD
        );

        println!(
            "  [INFO] CPU forced: {:.1} µs/tick, ANE-routed: {:.1} µs/tick",
            config_a.per_tick_us, config_b.per_tick_us
        );
        println!();

        if outcome_pass && cosine_pass {
            println!("🎉 GOAT Arena PASS — ANE-routed path produces equivalent game outcome");
        } else {
            println!("❌ GOAT Arena FAIL — keep ane_npc opt-in, investigate divergence");
        }
    }
}
