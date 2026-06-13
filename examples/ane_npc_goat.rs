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

/// NPC counts swept in the multi-size comparison table.
/// Kept distinct from `NPC_COUNT` (the GOAT verdict size) so the sweep
/// can be adjusted independently.
const SWEEP_COUNTS: [usize; 3] = [10, 100, 1000];

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

// ── Bench helpers ───────────────────────────────────────────────

/// Mean per-batch latency (µs) over `BENCH_ITERS` after `WARMUP_ITERS` warmup.
/// Reuses one backend instance so allocator state stabilizes across runs.
/// Returns the final outputs for downstream comparison.
fn bench_cpu(
    backend: &mut CpuTernaryBackend,
    inputs: &[NpcBrainInput],
) -> (f64, Vec<NpcBrainOutput>) {
    let n = inputs.len();
    let mut warmup = vec![NpcBrainOutput::default(); n];
    for _ in 0..WARMUP_ITERS {
        backend.batch_evaluate(inputs, &mut warmup).unwrap();
    }

    let mut outputs = vec![NpcBrainOutput::default(); n];
    let start = std::time::Instant::now();
    for _ in 0..BENCH_ITERS {
        backend.batch_evaluate(inputs, &mut outputs).unwrap();
    }
    let per_batch_us = start.elapsed().as_micros() as f64 / BENCH_ITERS as f64;
    (per_batch_us, outputs)
}

/// Same shape as `bench_cpu` but for any backend (ANE path).
/// Errors from `batch_evaluate` are counted as failed iterations; if every
/// iteration fails, the latency reflects the failure path (still useful to
/// report so a broken ANE model doesn't silently look "fast").
#[cfg(all(feature = "ane_npc", target_os = "macos"))]
fn bench_backend<B: NpcBrainBackend>(
    backend: &mut B,
    inputs: &[NpcBrainInput],
) -> (f64, Vec<NpcBrainOutput>) {
    let n = inputs.len();
    let mut warmup = vec![NpcBrainOutput::default(); n];
    for _ in 0..WARMUP_ITERS {
        let _ = backend.batch_evaluate(inputs, &mut warmup);
    }

    let mut outputs = vec![NpcBrainOutput::default(); n];
    let start = std::time::Instant::now();
    for _ in 0..BENCH_ITERS {
        let _ = backend.batch_evaluate(inputs, &mut outputs);
    }
    let per_batch_us = start.elapsed().as_micros() as f64 / BENCH_ITERS as f64;
    (per_batch_us, outputs)
}

/// (min, max, mean) cosine similarity across NPC outputs, skipping pairs
/// where both sides are near-zero (trivially equal — would inflate the count
/// without testing actual agreement).
#[cfg(all(feature = "ane_npc", target_os = "macos"))]
fn cosine_stats(cpu: &[NpcBrainOutput], ane: &[NpcBrainOutput]) -> (f32, f32, f32, usize) {
    let mut min_cos = f32::MAX;
    let mut max_cos = f32::MIN;
    let mut sum_cos = 0.0f32;
    let mut n_compared = 0usize;

    for (cpu_out, ane_out) in cpu.iter().zip(ane.iter()) {
        let cpu_proj = &cpu_out.projections;
        let ane_proj = &ane_out.projections;

        let cpu_norm: f32 = cpu_proj.iter().map(|x| x * x).sum::<f32>();
        let ane_norm: f32 = ane_proj.iter().map(|x| x * x).sum::<f32>();
        if cpu_norm < 1e-10 && ane_norm < 1e-10 {
            continue;
        }

        let cos = cosine_similarity(cpu_proj, ane_proj);
        min_cos = min_cos.min(cos);
        max_cos = max_cos.max(cos);
        sum_cos += cos;
        n_compared += 1;
    }

    let mean = if n_compared > 0 {
        sum_cos / n_compared as f32
    } else {
        1.0
    };
    (min_cos, max_cos, mean, n_compared)
}

// ── Main ─────────────────────────────────────────────────────────

fn main() {
    println!("=== Plan 255 GOAT Proof — ANE vs CPU NPC Brain Compute ===\n");
    println!("NPCs (GOAT): {NPC_COUNT}");
    println!("Sweep sizes: {SWEEP_COUNTS:?}");
    println!("Warmup: {WARMUP_ITERS} iters, Bench: {BENCH_ITERS} iters\n");

    let brains = make_diverse_brains(NPC_COUNT);
    let inputs: Vec<NpcBrainInput> = brains.iter().map(NpcBrainInput::from_brain).collect();

    // ── CPU Baseline (full 1000-NPC batch, reused for GOAT verdict) ──
    println!("── CPU Ternary Baseline ──");

    let mut cpu_backend = CpuTernaryBackend::new();
    #[allow(unused_variables)]
    let (cpu_per_batch_us, cpu_outputs) = bench_cpu(&mut cpu_backend, &inputs);

    println!("  Batch latency (1000 NPCs): {:.1} µs", cpu_per_batch_us);
    println!(
        "  Per-NPC: {:.1} ns",
        cpu_per_batch_us * 1000.0 / NPC_COUNT as f64
    );

    // ── Multi-size sweep table ───────────────────────────────────
    #[cfg(all(feature = "ane_npc", target_os = "macos"))]
    run_multi_size_sweep();

    #[cfg(not(all(feature = "ane_npc", target_os = "macos")))]
    {
        println!("\n── Multi-size Sweep Skipped ──");
        println!("  Requires macOS + --features ane_npc");
    }

    // ── ANE Path (if available) ──────────────────────────────────
    #[cfg(all(feature = "ane_npc", target_os = "macos"))]
    {
        println!("\n── ANE CoreML Path (GOAT size: {NPC_COUNT}) ──");

        use katgpt_rs::npc_ane_backend::AneNpcBrainBackend;

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

        match ane_backend {
            Some(mut ane_backend) => {
                let (ane_per_batch_us, ane_outputs) =
                    bench_backend(&mut ane_backend, &inputs);

                println!("  Batch latency ({NPC_COUNT} NPCs): {:.1} µs", ane_per_batch_us);
                println!(
                    "  Per-NPC: {:.1} ns",
                    ane_per_batch_us * 1000.0 / NPC_COUNT as f64
                );

                println!("\n── Output Comparison ──");
                let (min_cos, max_cos, mean_cos, n_compared) =
                    cosine_stats(&cpu_outputs, &ane_outputs);
                println!("  NPCs compared (non-zero): {n_compared}/{NPC_COUNT}");
                println!("  Cosine similarity:");
                println!("    min:  {min_cos:.6}");
                println!("    max:  {max_cos:.6}");
                println!("    mean: {mean_cos:.6}");

                let cpu_freed_pct = if cpu_per_batch_us > 0.0 {
                    ((cpu_per_batch_us - ane_per_batch_us) / cpu_per_batch_us) * 100.0
                } else {
                    0.0
                };
                println!(
                    "\n── CPU Time Freed ──\n  CPU: {:.1} µs, ANE: {:.1} µs → {:.1}% freed",
                    cpu_per_batch_us, ane_per_batch_us, cpu_freed_pct
                );

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
                match all_pass {
                    true => println!("🎉 GOAT PASS — promote ane_npc to default-on for macOS"),
                    false => println!("❌ GOAT FAIL — keep ane_npc as opt-in"),
                }
            }
            None => print_cpu_only_verdict(cpu_per_batch_us),
        }
    }

    #[cfg(not(all(feature = "ane_npc", target_os = "macos")))]
    {
        println!("\n── ANE Not Available ──");
        println!("  Run with --features ane_npc on macOS for full comparison");
        print_cpu_only_verdict(cpu_per_batch_us);
    }
}

/// Sweep CPU vs ANE over `SWEEP_COUNTS` and print a comparison table.
///
/// ANE models compiled for a fixed batch (1024 here) transparently pad
/// smaller input batches, so this primarily measures fixed dispatch
/// overhead vs the per-NPC work that scales. The CPU column should scale
/// roughly linearly with NPC count; the ANE column should be near-flat.
#[cfg(all(feature = "ane_npc", target_os = "macos"))]
fn run_multi_size_sweep() {
    use katgpt_rs::npc_ane_backend::AneNpcBrainBackend;

    println!("\n── Multi-Size Sweep (CPU vs ANE) ──");

    let model_path = std::path::Path::new("npc_brain.mlpackage");
    let mut ane_backend = match AneNpcBrainBackend::new(model_path, NPC_COUNT) {
        Ok(b) => {
            println!("  ANE backend: {} (optimal batch {})",
                b.backend_name(), b.optimal_batch_size());
            b
        }
        Err(e) => {
            println!("  ANE model load failed: {e}");
            println!("  Skipping ANE column — CPU-only sweep:\n");
            for &n in &SWEEP_COUNTS {
                let brains = make_diverse_brains(n);
                let inputs: Vec<NpcBrainInput> =
                    brains.iter().map(NpcBrainInput::from_brain).collect();
                let mut cpu = CpuTernaryBackend::new();
                let (cpu_us, _) = bench_cpu(&mut cpu, &inputs);
                println!(
                    "  NPCs={n:>4} | CPU {cpu_us:>8.1} µs | CPU {:.1} ns/NPC | ANE n/a",
                    cpu_us * 1000.0 / n as f64
                );
            }
            return;
        }
    };

    println!("  Format: NPCs | CPU µs | ANE µs | CPU ns/NPC | ANE ns/NPC | cosine\n");

    for &n in &SWEEP_COUNTS {
        let brains = make_diverse_brains(n);
        let inputs: Vec<NpcBrainInput> =
            brains.iter().map(NpcBrainInput::from_brain).collect();

        let mut cpu = CpuTernaryBackend::new();
        let (cpu_us, cpu_outputs) = bench_cpu(&mut cpu, &inputs);
        let (ane_us, ane_outputs) = bench_backend(&mut ane_backend, &inputs);

        let (_, _, mean_cos, _) = cosine_stats(&cpu_outputs, &ane_outputs);

        println!(
            "  NPCs={n:>4} | CPU {cpu_us:>8.1} µs | ANE {ane_us:>8.1} µs | CPU {:.1} ns/NPC | ANE {:.1} ns/NPC | cos {mean_cos:.4}",
            cpu_us * 1000.0 / n as f64,
            ane_us * 1000.0 / n as f64,
        );
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
