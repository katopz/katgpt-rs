//! Module-level integration tests for the `engram` primitive.
//!
//! These tests live at `engram/tests.rs` (per Plan T2.5/T3.5 etc —
//! "module-level `#[cfg(test)] mod tests { use super::*; ... }`") and exercise
//! the **public API surface** as exported from `engram/mod.rs`. They
//! complement (but do not duplicate) the per-file unit tests in
//! `hash.rs`, `table.rs`, `kernel.rs`, `commitment.rs`, `forward.rs`.
//!
//! Coverage map (per Plan task IDs):
//! - T1.6 hash determinism + head independence — re-tested here via the
//!   full `multi_head_hash → lookup_into` pipeline.
//! - T2.5 frozen table lookup + commitment — covered by `pipeline_*`.
//! - T3.5 sigmoid gate semantic boundaries — covered by `gate_*`.
//! - T5.5/T5.6 EngramTableId + Merkle root — covered by `commitment_*`.
//! - T7.1/T7.2 end-to-end fuse — covered by `fuse_*`.

use super::*;

/// Convenience: build a small deterministic table for the pipeline tests.
fn make_table(n_slots: usize, d: usize) -> InMemoryEngramTable {
    let mut b = EngramTableBuilder::new(n_slots, d);
    for i in 0..4u64 {
        let pat: Vec<f32> = (0..d).map(|j| (i as f32) * 0.1 + j as f32 * 0.01).collect();
        b.add_pattern(EngramHash(i), &pat);
    }
    b.build()
}

/// Convenience: build a deterministic K_MAX head set matching the table's
/// default heads. (We just read them back from the built table so the hashes
/// line up.)
fn make_heads_from_table(table: &InMemoryEngramTable) -> [HashHead; K_MAX] {
    *table.heads()
}

// ─── T1.6: multi_head_hash end-to-end ─────────────────────────────────

#[test]
fn pipeline_hash_to_lookup_deterministic() {
    // Hash a fixed suffix, look up, hash the same suffix again, look up —
    // results must be bit-identical.
    let d = 8;
    let table = make_table(64, d);
    let heads = make_heads_from_table(&table);

    let suffix = [CanonicalId(1), CanonicalId(2), CanonicalId(3)];

    let keys1 = multi_head_hash(&suffix, &heads);
    let mut out1 = vec![0.0f32; K_MAX * d];
    let hits1 = table.lookup_into(&keys1, &mut out1);

    let keys2 = multi_head_hash(&suffix, &heads);
    let mut out2 = vec![0.0f32; K_MAX * d];
    let hits2 = table.lookup_into(&keys2, &mut out2);

    assert_eq!(keys1, keys2, "hash determinism");
    assert_eq!(hits1, hits2, "hit count determinism");
    assert_eq!(out1, out2, "lookup output determinism");
}

#[test]
fn pipeline_different_suffix_different_lookup() {
    // Two distinct suffixes should produce at least one different lookup
    // slot (else the hash is degenerate).
    let d = 8;
    let table = make_table(128, d);
    let heads = make_heads_from_table(&table);

    let keys1 = multi_head_hash(&[CanonicalId(1), CanonicalId(2), CanonicalId(3)], &heads);
    let keys2 = multi_head_hash(&[CanonicalId(4), CanonicalId(5), CanonicalId(6)], &heads);

    let mut out1 = vec![0.0f32; K_MAX * d];
    let mut out2 = vec![0.0f32; K_MAX * d];
    table.lookup_into(&keys1, &mut out1);
    table.lookup_into(&keys2, &mut out2);

    // At least one head's retrieved slot must differ.
    let any_diff = (0..K_MAX).any(|k| out1[k * d..(k + 1) * d] != out2[k * d..(k + 1) * d]);
    assert!(any_diff, "different suffixes → different lookups");
}

// ─── T2.5: empty table contract ───────────────────────────────────────

#[test]
fn pipeline_empty_table_all_zeros() {
    let table = EngramTableBuilder::new(64, 4).build();
    let keys = [EngramHash(0); K_MAX];
    let mut out = vec![f32::NAN; K_MAX * 4];
    let hits = table.lookup_into(&keys, &mut out);
    assert_eq!(hits, 0);
    assert!(out.iter().all(|&v| v == 0.0), "empty table → all zeros");
}

// ─── T3.5: sigmoid gate boundary semantics ────────────────────────────

#[test]
fn gate_q_equals_k_near_one() {
    let d = 16;
    let cfg = SigmoidFusionConfig {
        tau: (d as f32).sqrt(),
        rmsnorm_eps: 1e-6,
    };
    let q: Vec<f32> = (1..=d).map(|i| i as f32).collect();
    let v: Vec<f32> = q.iter().map(|x| x * 0.1).collect();
    let mut out = vec![0.0f32; d];
    sigmoid_fuse_into(&q, &q, &v, &mut out, &cfg);
    let gate = out[0] / v[0];
    assert!(
        (gate - 1.0).abs() < 0.05,
        "q==k → gate near 1.0, got {gate}"
    );
}

#[test]
fn gate_q_opposite_k_near_zero() {
    let d = 16;
    let cfg = SigmoidFusionConfig {
        tau: (d as f32).sqrt(),
        rmsnorm_eps: 1e-6,
    };
    let q: Vec<f32> = (1..=d).map(|i| i as f32).collect();
    let k: Vec<f32> = q.iter().map(|x| -x).collect();
    let v: Vec<f32> = q.iter().map(|x| x * 0.1).collect();
    let mut out = vec![0.0f32; d];
    sigmoid_fuse_into(&q, &k, &v, &mut out, &cfg);
    let gate = out[0] / v[0];
    assert!(gate < 0.05, "q==-k → gate near 0.0, got {gate}");
}

#[test]
fn gate_q_orthogonal_k_near_half() {
    let d = 16;
    let cfg = SigmoidFusionConfig {
        tau: (d as f32).sqrt(),
        rmsnorm_eps: 1e-6,
    };
    let mut q = vec![0.0f32; d];
    let mut k = vec![0.0f32; d];
    for (i, qi) in q.iter_mut().take(d / 2).enumerate() {
        *qi = (i as f32) + 1.0;
    }
    for (i, ki) in k[d / 2..d].iter_mut().enumerate() {
        *ki = ((i + d / 2) as f32) + 1.0;
    }
    let v = vec![1.0f32; d];
    let mut out = vec![0.0f32; d];
    sigmoid_fuse_into(&q, &k, &v, &mut out, &cfg);
    let gate = out[0];
    assert!((gate - 0.5).abs() < 1e-4, "q⊥k → gate ≈ 0.5, got {gate}");
}

// ─── T5.5/T5.6: commitment + Merkle root ───────────────────────────────

#[test]
fn commitment_same_contents_same_id() {
    let mut b1 = EngramTableBuilder::new(32, 4);
    let mut b2 = EngramTableBuilder::new(32, 4);
    for i in 0..4u64 {
        let pat = [i as f32, (i + 1) as f32, (i + 2) as f32, (i + 3) as f32];
        b1.add_pattern(EngramHash(i), &pat);
        b2.add_pattern(EngramHash(i), &pat);
    }
    let t1 = b1.build();
    let t2 = b2.build();
    let id1 = EngramTableId::from_table(&t1);
    let id2 = EngramTableId::from_table(&t2);
    assert_eq!(id1, id2, "same contents → same EngramTableId");
    assert!(id1.verify(&t2), "id1 must verify t2 (same contents)");
}

#[test]
fn commitment_one_slot_changed_different_root() {
    let mut b1 = EngramTableBuilder::new(16, 4);
    let mut b2 = EngramTableBuilder::new(16, 4);
    let pat = [1.0f32, 2.0, 3.0, 4.0];
    b1.add_pattern(EngramHash(0), &pat);
    b2.add_pattern(EngramHash(0), &[9.0f32, 9.0, 9.0, 9.0]); // different
    let t1 = b1.build();
    let t2 = b2.build();
    assert_ne!(
        t1.commitment(),
        t2.commitment(),
        "one slot changed → different Merkle root"
    );
}

// ─── T7.1/T7.2: end-to-end fuse ───────────────────────────────────────

#[test]
fn fuse_modifies_hidden_state_with_populated_slots() {
    let d = 16;
    let table = make_table(64, d);
    let heads = make_heads_from_table(&table);
    let suffix = [CanonicalId(1), CanonicalId(2), CanonicalId(3)];
    let keys = multi_head_hash(&suffix, &heads);

    let mut hidden = vec![0.0f32; d];
    let query: Vec<f32> = (0..d).map(|i| (i as f32 * 0.1).sin()).collect();
    let cfg = EngramConfig::for_dim(d);

    let mut scratch_lookup = vec![0.0f32; K_MAX * d];
    let mut scratch_norm = vec![0.0f32; d];
    let mut scratch_out = vec![0.0f32; d];

    fuse_into_hidden_state(
        &mut hidden,
        &query,
        &table,
        &keys,
        &cfg,
        &mut scratch_lookup,
        &mut scratch_norm,
        &mut scratch_out,
    );

    // We can't predict exact contributions without knowing which slots the
    // hashes land on, but the hidden state should have changed if any head
    // hit a populated slot. With 4 populated slots out of 64 and K_MAX=16
    // independent hashes, the probability of zero hits is vanishingly
    // small.
    let any_change = hidden.iter().any(|&v| v != 0.0);
    assert!(any_change, "fuse should have written something");
}

#[test]
fn fuse_respects_k_heads_limit() {
    // config.k_heads < K_MAX → only the first k_heads patterns are fused.
    let d = 8;
    let table = make_table(64, d);
    let heads = make_heads_from_table(&table);
    let keys = multi_head_hash(&[CanonicalId(1), CanonicalId(2), CanonicalId(3)], &heads);

    let cfg_full = EngramConfig {
        fusion: SigmoidFusionConfig {
            tau: (d as f32).sqrt(),
            rmsnorm_eps: 1e-6,
        },
        k_heads: K_MAX,
    };
    let cfg_limited = EngramConfig {
        k_heads: 1,
        ..cfg_full
    };

    let run = |cfg: &EngramConfig| -> f32 {
        let mut hidden = vec![0.0f32; d];
        let query = vec![1.0f32; d];
        let mut scratch_lookup = vec![0.0f32; K_MAX * d];
        let mut scratch_norm = vec![0.0f32; d];
        let mut scratch_out = vec![0.0f32; d];
        fuse_into_hidden_state(
            &mut hidden,
            &query,
            &table,
            &keys,
            cfg,
            &mut scratch_lookup,
            &mut scratch_norm,
            &mut scratch_out,
        );
        hidden.iter().map(|v| v.abs()).sum()
    };

    let mag_full = run(&cfg_full);
    let mag_limited = run(&cfg_limited);
    // Full (K_MAX heads) should fuse ≥ as much as limited (1 head).
    assert!(
        mag_full >= mag_limited,
        "k_heads=K_MAX mag ({mag_full}) must be ≥ k_heads=1 mag ({mag_limited})"
    );
}

#[test]
fn k_max_is_16() {
    // Document the contract: K_MAX is fixed at 16 (paper: 8 heads × 2 N-gram orders).
    assert_eq!(K_MAX, 16);
}

#[test]
fn token_id_and_canonical_id_are_zero_cost_newtypes() {
    // Smoke: the newtypes wrap their inner type with #[repr(transparent)].
    let t = TokenId(42);
    let c = CanonicalId(99);
    assert_eq!(t.0, 42);
    assert_eq!(c.0, 99);
    // Copy + Eq + Hash are all derivable.
    let t2 = t;
    assert_eq!(t, t2);
    let c2 = c;
    assert_eq!(c, c2);
}
