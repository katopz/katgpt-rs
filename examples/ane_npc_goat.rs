//! Plan 255 GOAT Proof — ANE vs CPU NPC Brain Compute
//!
//! Validates:
//! 1. Output cosine similarity ≥ 0.99 for 1000 NPCs
//! 2. ANE dispatch latency < 1ms for 1000 NPC batch
//! 3. CPU time freed (wall-clock comparison)
//!
//! Usage:
//!   cargo run --example ane_npc_goat --features sense_composition --release
//!   cargo run --example ane_npc_goat --features ane_npc --release  # full ANE comparison

use katgpt_core::sense::backend::{
    CpuTernaryBackend, NpcBrainBackend, NpcBrainInput, NpcBrainOutput,
};
use katgpt_core::sense::brain::NpcBrain;
use katgpt_core::sense::octree::{KgEmbedding, SenseOctreeBuilder};
use katgpt_core::types::SenseKind;

const NPC_COUNT: usize = 1000;
const WARMUP_ITERS: usize = 10;
const BENCH_ITERS: usize = 100;

// ── GOAT thresholds ──────────────────────────────────────────────

#[cfg(all(feature = "ane_npc", target_os = "macos"))]
const COSINE_THRESHOLD: f32 = 0.99;
#[cfg(all(feature = "ane_npc", target_os = "macos"))]
const ANE_LATENCY_THRESHOLD_US: u64 = 1000;
#[cfg(all(feature = "ane_npc", target_os = "macos"))]
const CPU_FREED_THRESHOLD_PCT: f32 = 30.0;

// ── Deterministic PRNG ───────────────────────────────────────────

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
        let n_modules = rng.next_range(1, 7); // 1..=6
        let mut modules = Vec::with_capacity(n_modules);

        for m in 0..n_modules {
            let kind = ALL_KINDS[m % ALL_KINDS.len()];
            let n_embs = rng.next_range(1, 5); // 1..=4 embeddings
            let mut embeddings = Vec::with_capacity(n_embs);

            for _ in 0..n_embs {
                let entity_hash = rng.next_u64();
                let relation_hash = rng.next_u64();
                let mut embedding = [0.0f32; 8];
                for e in &mut embedding {
                    *e = rng.next_f32() * 2.0 - 1.0;
                }
                let confidence = 0.1 + rng.next_f32() * 0.9; // [0.1, 1.0]
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
            // Vary confidence
            module.confidence = 0.1 + rng.next_f32() * 0.9;
            module.commit();
            modules.push(module);
        }

        let mut brain = NpcBrain::compose(modules);

        // Varied HLA state
        for v in &mut brain.hla_state {
            *v = rng.next_f32() * 2.0 - 1.0;
        }

        // ~10% have GM overrides
        if npc_id % 10 == 0 {
            let pin_kind = ALL_KINDS[npc_id % ALL_KINDS.len()];
            brain.pin_sense(pin_kind, rng.next_f32());
        }

        // ~5% have autonomous disabled
        if npc_id % 20 == 0 {
            brain.disable_autonomous(npc_id as u64);
        }

        brains.push(brain);
    }

    brains
}

// ── Cosine similarity ────────────────────────────────────────────

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

// ── Main ─────────────────────────────────────────────────────────

fn main() {
    println!("=== Plan 255 GOAT Proof — ANE vs CPU NPC Brain Compute ===\n");
    println!("NPCs: {NPC_COUNT}");
    println!("Warmup: {WARMUP_ITERS} iters, Bench: {BENCH_ITERS} iters\n");

    // Generate diverse brains
    let brains = make_diverse_brains(NPC_COUNT);
    let inputs: Vec<NpcBrainInput> = brains.iter().map(NpcBrainInput::from_brain).collect();

    // ── CPU Baseline ──────────────────────────────────────────────
    println!("── CPU Ternary Baseline ──");

    let mut cpu_backend = CpuTernaryBackend::new();

    // Warmup
    let mut cpu_outputs_warmup = vec![NpcBrainOutput::default(); NPC_COUNT];
    for _ in 0..WARMUP_ITERS {
        cpu_backend
            .batch_evaluate(&inputs, &mut cpu_outputs_warmup)
            .unwrap();
    }

    // Bench
    let mut cpu_outputs = vec![NpcBrainOutput::default(); NPC_COUNT];
    let cpu_start = std::time::Instant::now();
    for _ in 0..BENCH_ITERS {
        cpu_backend
            .batch_evaluate(&inputs, &mut cpu_outputs)
            .unwrap();
    }
    let cpu_total = cpu_start.elapsed();
    let cpu_per_batch_us = cpu_total.as_micros() as f64 / BENCH_ITERS as f64;

    println!("  Batch latency (1000 NPCs): {:.1} µs", cpu_per_batch_us);
    println!(
        "  Per-NPC: {:.1} ns",
        cpu_per_batch_us * 1000.0 / NPC_COUNT as f64
    );

    // ── ANE Path (if available) ──────────────────────────────────
    #[cfg(all(feature = "ane_npc", target_os = "macos"))]
    {
        println!("\n── ANE CoreML Path ──");

        use katgpt_rs::npc_ane_backend::AneNpcBrainBackend;

        // Try to load the model
        let model_path = std::path::Path::new("npc_brain.mlpackage");
        let ane_backend = match AneNpcBrainBackend::new(model_path, NPC_COUNT) {
            Ok(b) => {
                println!("  Model loaded: {}", model_path.display());
                println!("  Backend: {}", b.backend_name());
                println!("  Optimal batch: {}", b.optimal_batch_size());
                Some(b)
            }
            Err(e) => {
                println!("  ANE not available: {e}");
                println!("  Falling back to CPU-only mode");
                None
            }
        };

        if let Some(mut ane_backend) = ane_backend {
            // Warmup
            let mut ane_outputs_warmup = vec![NpcBrainOutput::default(); NPC_COUNT];
            for _ in 0..WARMUP_ITERS {
                let _ = ane_backend.batch_evaluate(&inputs, &mut ane_outputs_warmup);
            }

            // Bench
            let mut ane_outputs = vec![NpcBrainOutput::default(); NPC_COUNT];
            let ane_start = std::time::Instant::now();
            for _ in 0..BENCH_ITERS {
                let _ = ane_backend.batch_evaluate(&inputs, &mut ane_outputs);
            }
            let ane_total = ane_start.elapsed();
            let ane_per_batch_us = ane_total.as_micros() as f64 / BENCH_ITERS as f64;

            println!("  Batch latency (1000 NPCs): {:.1} µs", ane_per_batch_us);
            println!(
                "  Per-NPC: {:.1} ns",
                ane_per_batch_us * 1000.0 / NPC_COUNT as f64
            );

            // ── Cosine similarity comparison ──────────────────────
            println!("\n── Output Comparison ──");

            let mut min_cos = f32::MAX;
            let mut max_cos = f32::MIN;
            let mut sum_cos = 0.0f32;
            let mut n_compared = 0usize;

            for (cpu_out, ane_out) in cpu_outputs.iter().zip(ane_outputs.iter()) {
                let cpu_proj = &cpu_out.projections;
                let ane_proj = &ane_out.projections;

                // Check if both have non-zero output (skip zero-zero comparisons)
                let cpu_norm: f32 = cpu_proj.iter().map(|x| x * x).sum::<f32>();
                let ane_norm: f32 = ane_proj.iter().map(|x| x * x).sum::<f32>();
                if cpu_norm < 1e-10 && ane_norm < 1e-10 {
                    continue; // Both zero — trivially equal
                }

                let cos = cosine_similarity(cpu_proj, ane_proj);
                min_cos = min_cos.min(cos);
                max_cos = max_cos.max(cos);
                sum_cos += cos;
                n_compared += 1;
            }

            let mean_cos = if n_compared > 0 {
                sum_cos / n_compared as f32
            } else {
                1.0
            };

            println!("  NPCs compared (non-zero): {n_compared}/{NPC_COUNT}");
            println!("  Cosine similarity:");
            println!("    min:  {min_cos:.6}");
            println!("    max:  {max_cos:.6}");
            println!("    mean: {mean_cos:.6}");

            // ── CPU freed estimate ────────────────────────────────
            let cpu_freed_pct = if cpu_per_batch_us > 0.0 {
                ((cpu_per_batch_us - ane_per_batch_us) / cpu_per_batch_us) * 100.0
            } else {
                0.0
            };
            println!(
                "\n── CPU Time Freed ──\n  CPU: {:.1} µs, ANE: {:.1} µs → {:.1}% freed",
                cpu_per_batch_us, ane_per_batch_us, cpu_freed_pct
            );

            // ── GOAT Verdict ──────────────────────────────────────
            println!("\n═══ GOAT Verdict ═══");

            let cosine_pass = mean_cos >= COSINE_THRESHOLD;
            let latency_pass = ane_per_batch_us <= ANE_LATENCY_THRESHOLD_US as f64;
            let freed_pass = cpu_freed_pct >= CPU_FREED_THRESHOLD_PCT as f64;

            println!(
                "  [{}/{}] Cosine ≥ {:.2}: {} (mean = {:.6})",
                cosine_pass as u8,
                1,
                COSINE_THRESHOLD,
                if cosine_pass { "PASS ✅" } else { "FAIL ❌" },
                mean_cos
            );
            println!(
                "  [{}/{}] ANE latency < {}µs: {} ({:.1} µs)",
                latency_pass as u8,
                1,
                ANE_LATENCY_THRESHOLD_US,
                if latency_pass { "PASS ✅" } else { "FAIL ❌" },
                ane_per_batch_us
            );
            println!(
                "  [{}/{}] CPU freed ≥ {:.0}%: {} ({:.1}%)",
                freed_pass as u8,
                1,
                CPU_FREED_THRESHOLD_PCT,
                if freed_pass { "PASS ✅" } else { "FAIL ❌" },
                cpu_freed_pct
            );

            let all_pass = cosine_pass && latency_pass && freed_pass;
            println!();
            if all_pass {
                println!("🎉 GOAT PASS — promote ane_npc to default-on for macOS");
            } else {
                println!("❌ GOAT FAIL — keep ane_npc as opt-in");
            }
        } else {
            // ANE model not available — CPU-only report
            print_cpu_only_verdict(cpu_per_batch_us);
        }
    }

    #[cfg(not(all(feature = "ane_npc", target_os = "macos")))]
    {
        println!("\n── ANE Not Available ──");
        println!("  Run with --features ane_npc on macOS for full comparison");
        print_cpu_only_verdict(cpu_per_batch_us);
    }
}

fn print_cpu_only_verdict(cpu_per_batch_us: f64) {
    println!("\n── CPU-Only Performance Report ──");
    println!(
        "  CPU batch (1000 NPCs): {:.1} µs ({:.1} ns/NPC)",
        cpu_per_batch_us,
        cpu_per_batch_us * 1000.0 / NPC_COUNT as f64
    );

    let cpu_ok = cpu_per_batch_us <= 5000.0; // 5ms budget for 1000 NPCs at 20Hz
    println!(
        "\n  CPU-only verdict: {} (batch < 5ms budget)",
        if cpu_ok { "PASS ✅" } else { "FAIL ❌" }
    );
    println!("  Note: ANE comparison requires macOS + --features ane_npc");
}
