//! `kron_chunk_wy` tests — the chunk lane's equality contract is the pinned
//! TOLERANCE vs sequential (rounding differs by construction); the refusal
//! posture is all-or-nothing.

use super::*;
use crate::kron_memory::{newton_beta, tests::Lcg};

type Cell8 = KronCell<8, 8, 16>;

fn chunk_of(rng: &mut Lcg, l: usize) -> Vec<ChunkItem<8, 8>> {
    (0..l)
        .map(|_| ChunkItem {
            k: rng.unit_vec::<8>(),
            v: rng.unit_vec::<8>(),
            c: rng.next_unit() * 2.0 - 1.0,
            beta: 0.25,
        })
        .collect()
}

#[test]
fn chunk_matches_sequential_within_pinned_tolerance() {
    let mut rng = Lcg::new(970);
    for l in [1usize, 2, 8, 17] {
        let items = chunk_of(&mut rng, l);

        let mut seq = Cell8::new();
        for item in &items {
            seq.write(&item.k, &item.v, item.c, item.beta)
                .expect("sequential write refused");
        }

        let mut ch = Cell8::new();
        let mut scratch = vec![0.0f32; 2 * l * l + 2 * l];
        let residuals = write_chunk_wy(&mut ch, &items, &mut scratch).expect("chunk write refused");
        assert_eq!(residuals.len(), l);

        // Whole-state tolerance: the solved residuals reproduce the sequential
        // result — the chunked round-trip error is the coupling norm, pinned
        // at 1e-3 relative on O(1) states.
        for (row_s, row_c) in seq.state_rows().iter().zip(ch.state_rows().iter()) {
            for (x, y) in row_s.iter().zip(row_c.iter()) {
                let scale = x.abs().max(y.abs()).max(1.0);
                assert!(
                    (x - y).abs() <= 1e-3 * scale,
                    "L={l}: chunk vs sequential drift {x} vs {y}"
                );
            }
        }
        // And the per-address reads agree.
        for item in &items {
            let a = seq.read(&item.k, &item.v);
            let b = ch.read(&item.k, &item.v);
            assert!((a - b).abs() <= 1e-3, "L={l}: readback drift {a} vs {b}");
        }
        // Every chunk item registered (registry capacity 16).
        assert_eq!(ch.registry().len(), l.min(16));
    }
}

#[test]
fn chunk_is_all_or_nothing_on_refusal() {
    let mut rng = Lcg::new(971);
    let mut items = chunk_of(&mut rng, 4);
    // Poison the last item's governor: β at the divergence edge.
    let bad_k = items[3].k;
    let bad_v = items[3].v;
    items[3].beta = 2.0 / (newton_beta(&bad_k, &bad_v).recip());

    let mut cell = Cell8::new();
    // Warm state so a silent partial apply would show.
    let warm_k = rng.unit_vec::<8>();
    let warm_v = rng.unit_vec::<8>();
    cell.write_newton(&warm_k, &warm_v, 3.0);
    let before = cell.frobenius_sq();

    let mut scratch = vec![0.0f32; 2 * 4 * 4 + 2 * 4];
    let err =
        write_chunk_wy(&mut cell, &items, &mut scratch).expect_err("a diverging chunk must refuse");
    assert_eq!(err, ChunkRefusal::Governor(3));
    assert_eq!(cell.frobenius_sq(), before, "refused chunk moved the state");
}

#[test]
fn chunk_refuses_empty_and_zero_address() {
    let mut cell = Cell8::new();
    let mut scratch = [0.0f32; 64];
    assert_eq!(
        write_chunk_wy(&mut cell, &[], &mut scratch),
        Err(ChunkRefusal::Empty)
    );

    let mut rng = Lcg::new(972);
    let mut items = chunk_of(&mut rng, 2);
    items[1].k = [0.0f32; 8];
    assert_eq!(
        write_chunk_wy(&mut cell, &items, &mut scratch),
        Err(ChunkRefusal::ZeroAddress(1))
    );
}

#[test]
fn chunk_first_residual_is_the_opening_state_miss() {
    // r⁰[0] against the opening state must equal what the first sequential
    // write would have seen — the same pinned read, bit-for-bit.
    let mut rng = Lcg::new(973);
    let mut items = chunk_of(&mut rng, 3);

    let warm_k = rng.unit_vec::<8>();
    let warm_v = rng.unit_vec::<8>();
    let mut seq = Cell8::new();
    seq.write_newton(&warm_k, &warm_v, 1.5);
    let mut ch = Cell8::new();
    ch.write_newton(&warm_k, &warm_v, 1.5);

    let first_residual = seq.read(&items[0].k, &items[0].v) - items[0].c;
    let mut scratch = vec![0.0f32; 2 * 9 + 6];
    for item in &mut items {
        item.beta = newton_beta(&item.k, &item.v);
    }
    let residuals = write_chunk_wy(&mut ch, &items, &mut scratch).expect("chunk refused");
    assert!(
        (residuals[0] - first_residual).abs() <= 1e-6,
        "r⁰[0] must be the opening-state residual: {} vs {}",
        residuals[0],
        first_residual
    );
}
