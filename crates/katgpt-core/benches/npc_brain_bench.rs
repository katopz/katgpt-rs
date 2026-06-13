//! Benchmark: NPC Brain Backend Throughput
//!
//! Measures batch projection performance for the CPU ternary backend.
//! ANE benchmarks are in the root example (ane_npc_goat) since they require
//! macOS + CoreML model file.
//!
//! Run:
//!   cargo bench --features sense_composition -- npc_brain

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use katgpt_core::sense::backend::{
    CpuTernaryBackend, NpcBrainBackend, NpcBrainInput, NpcBrainOutput,
};
use katgpt_core::sense::brain::NpcBrain;
use katgpt_core::sense::octree::{KgEmbedding, SenseOctreeBuilder};
use katgpt_core::types::SenseKind;

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

        // ~10% GM overrides
        if npc_id % 10 == 0 {
            let pin_kind = ALL_KINDS[npc_id % ALL_KINDS.len()];
            brain.pin_sense(pin_kind, rng.next_f32());
        }

        // ~5% autonomous disabled
        if npc_id % 20 == 0 {
            brain.disable_autonomous(npc_id as u64);
        }

        brains.push(brain);
    }

    brains
}

// ── Benchmark ────────────────────────────────────────────────────

fn bench_cpu_ternary(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_ternary");

    for &size in &[10usize, 100, 1000, 5000] {
        let brains = make_diverse_brains(size);
        let inputs: Vec<NpcBrainInput> = brains.iter().map(NpcBrainInput::from_brain).collect();
        let mut outputs = vec![NpcBrainOutput::default(); size];

        group.bench_with_input(
            BenchmarkId::new("batch_evaluate", size),
            &inputs,
            |b, inputs| {
                let mut backend = CpuTernaryBackend::new();
                b.iter(|| {
                    backend.batch_evaluate(inputs, &mut outputs).unwrap();
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_cpu_ternary);
criterion_main!(benches);
