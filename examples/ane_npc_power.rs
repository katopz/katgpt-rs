//! Plan 255 Part 5 — GOAT Power: CPU Utilization Measurement
//!
//! Measures actual CPU time (user + system via `getrusage`) consumed by NPC
//! brain evaluation, comparing CPU-forced vs ANE-routed paths.
//!
//! Hypothesis: routing NPC brain compute to ANE should reduce host CPU time
//! because the projection work moves to the Neural Engine.
//!
//! Usage:
//!   cargo run --example ane_npc_power --features sense_composition --release
//!   cargo run --example ane_npc_power --features ane_npc --release  # full ANE comparison

use katgpt_core::sense::backend::{NpcBrainBackend, NpcBrainInput, NpcBrainOutput};
use katgpt_core::sense::brain::NpcBrain;
use katgpt_core::sense::octree::{KgEmbedding, SenseOctreeBuilder};
use katgpt_core::types::SenseKind;

use katgpt_rs::npc_brain_router::NpcBrainRouter;

const NPC_COUNT: usize = 1000;
const WARMUP_ITERS: usize = 10;
const ITERATIONS: usize = 1000;

// GOAT gate: ANE must reduce CPU utilization by ≥ 30% at 1000 NPC load.
const CPU_UTIL_REDUCTION_TARGET_PCT: f64 = 30.0;

// ── CPU time FFI (Unix only, no external crate) ───────────────────

#[cfg(unix)]
mod cpu_time {
    /// `RUSAGE_SELF` — measure current process only. Same value on macOS/BSD/Linux.
    const RUSAGE_SELF: i32 = 0;

    /// C `struct timeval` — matches macOS x86_64/arm64 ABI (8 bytes).
    #[repr(C)]
    struct Timeval {
        tv_sec: i64,
        tv_usec: i32,
    }

    /// Partial `struct rusage` — only the first two fields (ru_utime, ru_stime)
    /// are read. Trailing padding makes the struct at least as large as the
    /// real C struct (~148 bytes on macOS); extra space is harmless since
    /// `getrusage` only writes fields it knows about.
    #[repr(C)]
    struct Rusage {
        ru_utime: Timeval,
        ru_stime: Timeval,
        _padding: [u8; 256],
    }

    // SAFETY: `getrusage` is a standard libc function with a stable C ABI.
    // The `unsafe` keyword on the extern block is required by Rust 2024 edition
    // to acknowledge that declaring external symbols is inherently unsafe.
    unsafe extern "C" {
        /// libc `getrusage(2)`. Returns 0 on success, -1 on error.
        /// Writes a `struct rusage` into the caller-provided buffer.
        fn getrusage(who: i32, usage: *mut Rusage) -> i32;
    }

    /// Returns `(user_seconds, system_seconds)` of CPU time consumed by this process.
    /// Uses `RUSAGE_SELF` — includes all threads, excludes child processes.
    pub fn cpu_time_secs() -> (f64, f64) {
        let mut ru = Rusage {
            ru_utime: Timeval { tv_sec: 0, tv_usec: 0 },
            ru_stime: Timeval { tv_sec: 0, tv_usec: 0 },
            _padding: [0; 256],
        };
        // SAFETY: `getrusage` writes into our stack-allocated `Rusage`, which is
        // larger than the real C struct. Pointer is valid for the call duration.
        // `RUSAGE_SELF` (=0) is always a valid `who` value on Unix, so no -1 path.
        unsafe {
            getrusage(RUSAGE_SELF, &mut ru);
        }
        let user = ru.ru_utime.tv_sec as f64 + ru.ru_utime.tv_usec as f64 / 1e6;
        let sys = ru.ru_stime.tv_sec as f64 + ru.ru_stime.tv_usec as f64 / 1e6;
        (user, sys)
    }
}

#[cfg(not(unix))]
mod cpu_time {
    /// Non-Unix stub — `getrusage` is unavailable. Caller must guard usage.
    pub fn cpu_time_secs() -> (f64, f64) {
        (0.0, 0.0)
    }
}

// ── Deterministic PRNG (copied from ane_npc_goat.rs) ──────────────

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
        match hi <= lo {
            true => lo,
            false => ((self.next_u64() as usize) % (hi - lo)) + lo,
        }
    }
}

// ── Brain generation (copied from ane_npc_goat.rs) ────────────────

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

// ── Measurement ───────────────────────────────────────────────────

struct PhaseResult {
    backend_name: &'static str,
    wall_secs: f64,
    cpu_user_secs: f64,
    cpu_sys_secs: f64,
    iterations: usize,
}

impl PhaseResult {
    fn cpu_total_secs(&self) -> f64 {
        self.cpu_user_secs + self.cpu_sys_secs
    }

    fn cpu_util_pct(&self) -> f64 {
        match self.wall_secs > 0.0 {
            true => self.cpu_total_secs() / self.wall_secs * 100.0,
            false => 0.0,
        }
    }

    fn per_iter_us(&self) -> f64 {
        self.wall_secs / self.iterations as f64 * 1e6
    }
}

/// Run `iterations` batch evaluations, measuring both wall-clock and CPU time.
///
/// `getrusage` is cumulative across the process — we sample before and after
/// and take the delta, so prior phase CPU time does not contaminate results.
fn measure_phase<B: NpcBrainBackend>(
    backend: &mut B,
    inputs: &[NpcBrainInput],
    iterations: usize,
    warmup: usize,
) -> PhaseResult {
    let mut outputs = vec![NpcBrainOutput::default(); inputs.len()];

    for _ in 0..warmup {
        let _ = backend.batch_evaluate(inputs, &mut outputs);
    }

    let (u0, s0) = cpu_time::cpu_time_secs();
    let t0 = std::time::Instant::now();
    for _ in 0..iterations {
        let _ = backend.batch_evaluate(inputs, &mut outputs);
    }
    let wall = t0.elapsed().as_secs_f64();
    let (u1, s1) = cpu_time::cpu_time_secs();

    PhaseResult {
        backend_name: backend.backend_name(),
        wall_secs: wall,
        cpu_user_secs: u1 - u0,
        cpu_sys_secs: s1 - s0,
        iterations,
    }
}

fn print_phase(label: &str, r: &PhaseResult) {
    println!("── {label} ──");
    println!("  Backend: {}", r.backend_name);
    println!("  Wall-clock: {:.2} ms", r.wall_secs * 1e3);
    println!(
        "  CPU time (user+sys): {:.2} ms",
        r.cpu_total_secs() * 1e3
    );
    println!("  CPU utilization: {:.1}%", r.cpu_util_pct());
    println!("  Per-iter: {:.2} µs\n", r.per_iter_us());
}

// ── Main ──────────────────────────────────────────────────────────

fn main() {
    println!("=== Plan 255 GOAT Power — CPU Utilization Measurement ===\n");
    println!("NPCs: {NPC_COUNT}, Iterations: {ITERATIONS}\n");

    #[cfg(not(unix))]
    {
        println!("CPU time measurement requires Unix (macOS/Linux).");
        println!("Aborting — nothing to measure without getrusage.");
        return;
    }

    let brains = make_diverse_brains(NPC_COUNT);
    let inputs: Vec<NpcBrainInput> = brains.iter().map(NpcBrainInput::from_brain).collect();

    // ── Phase 1: CPU-forced path ─────────────────────────────────
    let mut cpu_router = NpcBrainRouter::cpu();
    let cpu_result = measure_phase(&mut cpu_router, &inputs, ITERATIONS, WARMUP_ITERS);
    print_phase("CPU-Forced Path", &cpu_result);

    // ── Phase 2: ANE-routed path (macOS + feature ane_npc) ───────
    #[cfg(all(feature = "ane_npc", target_os = "macos"))]
    {
        let model_path = std::path::Path::new("npc_brain.mlpackage");
        let mut ane_router = NpcBrainRouter::new(Some(model_path));

        match ane_router.is_ane() {
            true => {
                let ane_result = measure_phase(&mut ane_router, &inputs, ITERATIONS, WARMUP_ITERS);
                print_phase("ANE-Routed Path", &ane_result);
                print_verdict(&cpu_result, &ane_result);
            }
            false => {
                println!("── ANE-Routed Path ──");
                println!("  Backend: {} (ANE model not resident, fell back)", ane_router.backend_name());
                println!("  Cannot measure ANE CPU savings — model load failed.");
                println!("  Place npc_brain.mlpackage next to binary and re-run.\n");
                print_cpu_only(&cpu_result);
            }
        }
    }

    #[cfg(not(all(feature = "ane_npc", target_os = "macos")))]
    {
        println!("── ANE-Routed Path ──");
        println!("  ANE requires macOS + --features ane_npc");
        println!("  Only CPU path measured.\n");
        print_cpu_only(&cpu_result);
    }
}

fn print_verdict(cpu: &PhaseResult, ane: &PhaseResult) {
    println!("── GOAT Power Verdict ──");

    let cpu_util = cpu.cpu_util_pct();
    let ane_util = ane.cpu_util_pct();
    let reduction_pct = match cpu_util > 0.0 {
        true => (cpu_util - ane_util) / cpu_util * 100.0,
        false => 0.0,
    };
    let pass = reduction_pct >= CPU_UTIL_REDUCTION_TARGET_PCT;

    println!(
        "  [{}] CPU utilization reduced ≥ {:.0}%: {:.1}% → {:.1}% ({:.1}% reduction)",
        if pass { "PASS ✅" } else { "FAIL ❌" },
        CPU_UTIL_REDUCTION_TARGET_PCT,
        cpu_util,
        ane_util,
        reduction_pct,
    );
    println!(
        "  [INFO] Wall-clock: CPU {:.2} ms vs ANE {:.2} ms",
        cpu.wall_secs * 1e3,
        ane.wall_secs * 1e3,
    );
    println!(
        "  [INFO] Absolute CPU time saved: {:.2} ms",
        (cpu.cpu_total_secs() - ane.cpu_total_secs()) * 1e3,
    );

    match pass {
        true => println!("\n🎉 GOAT PASS — promote ane_npc to default-on for macOS."),
        false => println!("\n❌ GOAT FAIL — keep ane_npc as opt-in feature."),
    }
}

fn print_cpu_only(cpu: &PhaseResult) {
    println!("── CPU-Only Power Report ──");
    println!(
        "  CPU backend '{}': {:.2} ms wall, {:.2} ms CPU ({:.1}% utilization)",
        cpu.backend_name,
        cpu.wall_secs * 1e3,
        cpu.cpu_total_secs() * 1e3,
        cpu.cpu_util_pct(),
    );
    println!("  Per-iter: {:.2} µs", cpu.per_iter_us());
    println!("\n  Note: ANE comparison requires macOS + --features ane_npc.");
}
