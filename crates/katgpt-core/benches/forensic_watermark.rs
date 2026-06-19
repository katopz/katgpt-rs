//! Forensic watermark recipe primitive — criterion benchmarks (Plan 293 T7.1).
//!
//! Covers the hot-path operations:
//! - `derive_recipe` — per-recipient recipe derivation.
//! - `apply_vertex_marks_simd` on a 10⁴-vertex mesh.
//! - `apply_dct_marks` on a 10³-block texture.
//! - `apply_topology_marks` on a 10³-triangle mesh.
//! - `recover_codeword` — end-to-end DCT channel recovery.
//!
//! # Out of scope (GOAT gate, separate session)
//!
//! T7.2–T7.5 (G1 single-leak attribution, G2 collusion resistance,
//! G3 visual quality, G4 recompression robustness) require real assets
//! and are deferred to the GOAT gate session per Plan 293.
//!
//! # Run
//!
//! ```bash
//! cargo bench --bench forensic_watermark --features forensic_watermark -- --quick
//! ```
//!
//! # Feature gate
//!
//! Requires `forensic_watermark`.

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use katgpt_core::forensic::recover::{LeakedContent, recover_codeword};
use katgpt_core::forensic::texture::{Dct8x8Block, apply_dct_marks};
use katgpt_core::forensic::topology::{TriangleMesh, apply_topology_marks};
use katgpt_core::forensic::vertex::{apply_vertex_marks_simd};
use katgpt_core::forensic::{RecipeConfig, derive_recipe};

/// Vertex count for the SIMD bench (matches Plan 293 T7.1: 10⁴ verts).
const BENCH_VERTS: usize = 10_000;
/// Block count for the DCT bench (Plan 293 T7.1: 10³ blocks).
const BENCH_BLOCKS: usize = 1_000;
/// Triangle count for the topology bench.
const BENCH_TRIS: usize = 1_000;

fn bench_derive_recipe(c: &mut Criterion) {
    let mut group = c.benchmark_group("forensic/derive_recipe");
    group.sample_size(50);
    let cfg = RecipeConfig::default();
    let pk = [7u8; 32];
    let ms = [9u8; 32];
    group.bench_function("default_config", |b| {
        b.iter(|| {
            black_box(derive_recipe(black_box(&cfg), black_box(&pk), black_box(&ms)));
        });
    });
    group.finish();
}

fn bench_apply_vertex_marks_simd(c: &mut Criterion) {
    let mut group = c.benchmark_group("forensic/apply_vertex_marks_simd");
    group.sample_size(100);
    let cfg = RecipeConfig::default();
    let pk = [3u8; 32];
    let ms = [4u8; 32];
    let recipe = derive_recipe(&cfg, &pk, &ms);
    group.bench_function("10k_verts", |b| {
        b.iter_batched(
            || synth_mesh(BENCH_VERTS, BENCH_VERTS / 2),
            |mut mesh| {
                apply_vertex_marks_simd(&mut mesh, &recipe, &cfg);
            },
            criterion::BatchSize::SmallInput,
        );
    });
    group.finish();
}

fn bench_apply_dct_marks(c: &mut Criterion) {
    let mut group = c.benchmark_group("forensic/apply_dct_marks");
    group.sample_size(100);
    let cfg = RecipeConfig::default();
    let pk = [5u8; 32];
    let ms = [6u8; 32];
    let recipe = derive_recipe(&cfg, &pk, &ms);
    group.bench_function("1k_blocks", |b| {
        b.iter_batched(
            || synth_texture(BENCH_BLOCKS),
            |mut tex| {
                apply_dct_marks(&mut tex, &recipe, &cfg);
            },
            criterion::BatchSize::SmallInput,
        );
    });
    group.finish();
}

fn bench_apply_topology_marks(c: &mut Criterion) {
    let mut group = c.benchmark_group("forensic/apply_topology_marks");
    group.sample_size(100);
    let cfg = RecipeConfig::default();
    let pk = [11u8; 32];
    let ms = [12u8; 32];
    let recipe = derive_recipe(&cfg, &pk, &ms);
    group.bench_function("1k_tris", |b| {
        b.iter_batched(
            || synth_mesh(BENCH_TRIS * 2, BENCH_TRIS),
            |mut mesh| {
                apply_topology_marks(&mut mesh, &recipe, &cfg);
            },
            criterion::BatchSize::SmallInput,
        );
    });
    group.finish();
}

fn bench_recover_codeword(c: &mut Criterion) {
    let mut group = c.benchmark_group("forensic/recover_codeword");
    group.sample_size(50);
    let cfg = RecipeConfig::default();
    let pk = [13u8; 32];
    let ms = [14u8; 32];
    let recipe = derive_recipe(&cfg, &pk, &ms);
    let n_blocks = BENCH_BLOCKS;
    let mut mesh = synth_mesh(BENCH_VERTS, BENCH_TRIS);
    let mesh_ref = mesh.clone();
    let mut tex = synth_texture(n_blocks);
    let tex_ref = tex.clone();
    apply_dct_marks(&mut tex, &recipe, &cfg);
    apply_topology_marks(&mut mesh, &recipe, &cfg);
    let leaked = LeakedContent {
        mesh,
        texture_blocks: tex,
    };
    let reference = LeakedContent {
        mesh: mesh_ref,
        texture_blocks: tex_ref,
    };
    group.bench_function("end_to_end", |b| {
        b.iter(|| {
            black_box(recover_codeword(
                black_box(&leaked),
                black_box(&reference),
                black_box(&recipe),
                1e-6,
            ));
        });
    });
    group.finish();
}

// ─── Synthetic fixtures ────────────────────────────────────────────────

fn synth_mesh(n_verts: usize, n_tris: usize) -> TriangleMesh {
    let side = (n_verts as f32).sqrt().ceil() as usize;
    let mut positions = Vec::with_capacity(n_verts);
    for y in 0..=side {
        for x in 0..=side {
            positions.push([x as f32, y as f32, 0.0]);
            if positions.len() >= n_verts {
                break;
            }
        }
        if positions.len() >= n_verts {
            break;
        }
    }
    let mut indices = Vec::with_capacity(n_tris);
    let row_stride = (side + 1) as u32;
    let mut made = 0usize;
    for y in 0..side {
        for x in 0..side {
            if made >= n_tris {
                break;
            }
            let i00 = (y as u32) * row_stride + x as u32;
            indices.push([i00, i00 + 1, i00 + row_stride]);
            made += 1;
        }
    }
    TriangleMesh { positions, indices }
}

fn synth_texture(n_blocks: usize) -> Vec<Dct8x8Block> {
    (0..n_blocks)
        .map(|i| {
            let mut d = [0.0f32; 64];
            for j in 0..64 {
                d[j] = ((i + j) as f32) * 0.5;
            }
            Dct8x8Block { data: d }
        })
        .collect()
}

criterion_group!(
    forensic_benches,
    bench_derive_recipe,
    bench_apply_vertex_marks_simd,
    bench_apply_dct_marks,
    bench_apply_topology_marks,
    bench_recover_codeword,
);
criterion_main!(forensic_benches);
