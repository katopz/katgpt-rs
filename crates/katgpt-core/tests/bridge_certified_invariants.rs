#![cfg(feature = "closure_instrument")]
//! `BridgeCertified` invariants — the audit half of Proposal 005 (Plan 604).
//!
//! Complements `examples/bridge_determinism_check.rs` (the G1 harness) and
//! `tests/bridge_certified_alloc_check.rs` (G4). What is pinned here:
//!
//! 1. the freeze hash is content-addressed (1-ULP flip ⇒ different hash);
//! 2. the replay byte path equals the direct kernel, bit-identically;
//! 3. the error taxonomy separates `FreezeSkew` (valid divergence — carries
//!    BOTH hashes) from BadInputs (malformed evidence);
//! 4. non-finite tables are refused (`NonFreezeable` / `None`) — sNaN
//!    quieting is architecture-defined, so a certified hash must not exist
//!    for such a table;
//! 5. the certified kernel is deliberately NOT the platform-SIMD kernel — a
//!    crafted cancellation input diverges them, keeping the distinction
//!    load-bearing (the `ordered_dot_differs_from_simd_dot` pin, lifted to
//!    the bridge level);
//! 6. the G1 corpus digest for THIS architecture matches the pinned aarch64
//!    value. On x86_64 this test is the G1 gate: if it reds here, the
//!    cross-arch determinism claim broke — rerun the harness on both arches
//!    (see the example's protocol comment) before touching the pin.

use katgpt_core::closure::{
    BridgeCertified, BridgeCertifyError, G1_SEED, MotifDirections, OperatorKind, PrimitiveKind,
    PrimitiveTransitionGraph, PtgRecorder, certified_ptg_to_motif_embedding,
    certified_ptg_to_motif_embedding_into, embedding_from_bytes, embedding_to_bytes, g1_corpus,
    ptg_to_motif_embedding, serialize_postcard,
};
use katgpt_core::simd::fast_sigmoid;

fn tiny_dirs(k: usize, n: usize, seed: u64) -> MotifDirections {
    let mut x = seed | 1;
    let mut next = move || {
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    };
    let directions: Vec<f32> = (0..k * n)
        .map(|_| ((next() >> 40) as f32 / 16_777_216.0 - 0.5) * 4.0)
        .collect();
    MotifDirections::from_flat(directions, k, n).expect("shape")
}

fn chain_ptg(task_family: u32, prims: &[u32]) -> PrimitiveTransitionGraph {
    let mut rec = PtgRecorder::new(task_family);
    let mut prev: Option<u32> = None;
    for (j, &p) in prims.iter().enumerate() {
        let node = rec.enter(PrimitiveKind::UserDefined(p), j as u32, None);
        if let Some(p_id) = prev {
            rec.exit(p_id, node, OperatorKind::Sequence);
        }
        prev = Some(node);
    }
    rec.finish()
}

/// 1. Content-addressed freeze hash.
#[test]
fn freeze_hash_is_content_addressed_and_stable() {
    let a = tiny_dirs(4, 8, 1);
    let b = a.clone();
    assert_eq!(a.freeze_version_hash(), b.freeze_version_hash());
    let mut flipped = a.clone();
    flipped.directions[5] = f32::from_bits(flipped.directions[5].to_bits() ^ 1);
    assert_ne!(a.freeze_version_hash(), flipped.freeze_version_hash());
}

/// 2+3. Replay round-trip + the two-sided error taxonomy.
#[test]
fn replay_matches_direct_kernel_and_classifies_errors() {
    let dirs = tiny_dirs(8, 16, 2);
    let ptg = chain_ptg(1, &[3, 1, 4, 1, 5]);
    let raw = serialize_postcard(&ptg).expect("postcard");
    let snap = dirs.to_freeze_snapshot();

    let out = dirs.replay(&raw, &snap).expect("same snapshot replays");
    let emb = embedding_from_bytes(&out).expect("canonical layout");
    assert_eq!(emb.len(), dirs.k);
    assert_eq!(emb, certified_ptg_to_motif_embedding(&ptg, &dirs));

    // Freeze skew: a different table's snapshot is rejected with BOTH hashes.
    // (Seeds 2 and 4: `seed | 1` must not collide — 2|1 = 3, 3|1 = 3.)
    let other = tiny_dirs(8, 16, 4);
    assert_ne!(dirs.freeze_version_hash(), other.freeze_version_hash());
    match dirs.replay(&raw, &other.to_freeze_snapshot()) {
        Err(BridgeCertifyError::FreezeSkew {
            client_freeze_hash,
            pillar_freeze_hash,
        }) => {
            assert_eq!(client_freeze_hash, other.freeze_version_hash());
            assert_eq!(pillar_freeze_hash, dirs.freeze_version_hash());
        }
        other => panic!("expected FreezeSkew, got {other:?}"),
    }

    // Bad inputs: garbage bytes must not panic and must not look like skew.
    match dirs.replay(&[0xFF, 0x00, 0x7F], &snap) {
        Err(BridgeCertifyError::BadInputs(_)) => {}
        other => panic!("expected BadInputs, got {other:?}"),
    }
}

/// 4. Non-finite tables are refused, not hashed.
#[test]
fn non_finite_table_is_never_certified() {
    let mut bad = tiny_dirs(2, 4, 4);
    bad.directions[0] = f32::NAN;
    assert_eq!(bad.replay(&[], &[]), Err(BridgeCertifyError::NonFreezeable));
    // The snapshot decoder refuses them too (round-trip cannot launder one in).
    assert!(MotifDirections::from_freeze_snapshot(&bad.to_freeze_snapshot()).is_none());
}

/// 5. The certified kernel is deliberately not the platform kernel.
#[test]
fn certified_differs_from_platform_on_cancellation_input() {
    // Mirror of `ordered_dot_differs_from_simd_dot`, lifted through the
    // bridge: direction row [1e8, 1.0, −1e8, 1.0] against a UNIFORM count
    // vector [1,1,1,1] (prims 4,5,6,7 map 1:1 onto indices 0..4 at n=4).
    // The sequential fold adds the trailing +1 AFTER the 1e8/−1e8
    // cancellation and lands on 1.0; the SIMD horizontal pairs the +1 into
    // the pre-cancellation partials and lands on 0.0. Certified ⇒
    // sigmoid(1.0), platform ⇒ sigmoid(0.0) = 0.5 — visibly different.
    let dirs = MotifDirections::from_flat(vec![1e8, 1.0, -1e8, 1.0], 1, 4).expect("shape");
    let ptg = chain_ptg(1, &[4, 5, 6, 7]);
    let certified = certified_ptg_to_motif_embedding(&ptg, &dirs);
    let platform = ptg_to_motif_embedding(&ptg, &dirs);
    assert_ne!(certified, platform, "certified must not collapse into simd");
    assert!(
        (platform[0] - 0.5).abs() < 1e-6,
        "platform dot must cancel to 0, got {}",
        platform[0]
    );
    assert!((certified[0] - fast_sigmoid(1.0)).abs() < 1e-6);
}

/// 6. The `_into` hot path equals the allocating path, bit-identically.
#[test]
fn into_variant_matches_allocating_variant() {
    let dirs = tiny_dirs(16, 32, 5);
    let ptg = chain_ptg(
        2,
        &(0..40).map(|i| (i * 7 % 256) as u32).collect::<Vec<_>>(),
    );
    let alloc = certified_ptg_to_motif_embedding(&ptg, &dirs);
    let mut feature = vec![9.9f32; dirs.n]; // garbage on entry: must be zeroed
    let mut out = vec![-1.0f32; dirs.k * 2]; // slack: only the first k written
    let k = certified_ptg_to_motif_embedding_into(&ptg, &dirs, &mut feature, &mut out);
    assert_eq!(k, dirs.k);
    assert_eq!(&out[..k], alloc.as_slice());
}

/// 7. G1 corpus digest pin (aarch64-apple-darwin, release, 2026-09-21).
///
/// The corpus (v1, seed `G1_SEED`) is generated in-library, so this pin and
/// the harness digest cover the same bytes on every arch. If this reds on
/// `x86_64`: the cross-arch determinism claim broke. Do NOT re-pin from the
/// failing arch — rerun the harness on BOTH arches first; only re-pin after
/// the divergence is understood and the fix is deliberate.
#[test]
fn g1_corpus_digest_matches_pinned_aarch64_value() {
    let corpus = g1_corpus(G1_SEED);
    let dirs = &corpus.dirs;
    let mut hasher = blake3::Hasher::new();
    let mut feature = vec![0.0f32; dirs.n];
    let mut out = vec![0.0f32; dirs.k];
    for ptg in &corpus.ptgs {
        certified_ptg_to_motif_embedding_into(ptg, dirs, &mut feature, &mut out);
        for v in &out {
            hasher.update(&v.to_le_bytes());
        }
    }
    let digest = format!("{}", hasher.finalize());
    assert_eq!(
        digest, G1_PINNED_DIGEST,
        "G1 corpus digest drifted from the pinned aarch64 value.\n\
         got:      {digest}\n\
         pinned:   {G1_PINNED_DIGEST}\n\
         Protocol: run examples/bridge_determinism_check.rs on aarch64 AND \
         x86_64 (Rosetta) and compare before re-pinning."
    );
}

/// The pinned G1 digest — recorded 2026-09-21 from BOTH the aarch64-apple-darwin
/// release run AND the x86_64-apple-darwin (Rosetta) release run (identical:
/// `.benchmarks/604_bridge_certified_g1.md`). The digest is also
/// profile-independent on aarch64 (debug == release), as expected for a fixed
/// IEEE op sequence.
const G1_PINNED_DIGEST: &str = "bae6e6228bcf1b12cd8b374eee8c6320a58fd788ae1af8e4539f5bd831d4be68";

/// 8. Wire-shape sanity: the snapshot is little-endian canonical (the hash
///    must not depend on the host's byte order).
#[test]
fn freeze_snapshot_bytes_are_little_endian_canonical() {
    let dirs = MotifDirections::from_flat(vec![1.0f32], 1, 1).expect("shape");
    let snap = dirs.to_freeze_snapshot();
    assert_eq!(&snap[..4], &1u32.to_le_bytes());
    assert_eq!(&snap[4..8], &1u32.to_le_bytes());
    assert_eq!(&snap[8..12], &1.0f32.to_le_bytes());
    assert_eq!(embedding_to_bytes(&[0.5]), 0.5f32.to_le_bytes().to_vec());
}
