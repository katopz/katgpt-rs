//! BridgeCertified G1 determinism harness (Proposal 005, Plan 604).
//!
//! Mirrors Plan 385's G3 cross-arch check at the bridge level: build the
//! deterministic corpus ([`katgpt_core::closure::g1_corpus`], v1), run the
//! certified kernel over all inputs, and digest the output bytes with BLAKE3.
//!
//! - **Record mode** (no args): prints `digest <hex>` plus the platform and
//!   box state. Run once per architecture and compare.
//! - **Verify mode** (`--expect <hex>`): exits non-zero if the digest differs.
//!
//! G1 protocol (the gate that promotes nothing on its own — the Phase-4
//! promotion decision needs the Phase-3 wiring too):
//!
//! ```sh
//! cargo run --release -p katgpt-core --features closure_instrument \
//!   --example bridge_determinism_check                 # aarch64 (native)
//! cargo run --release -p katgpt-core --features closure_instrument \
//!   --target x86_64-apple-darwin --example bridge_determinism_check  # Rosetta
//! ```
//!
//! Both digests must be equal. A divergence means a fixed-op-sequence claim
//! broke (FMA contraction, denormal flushing, libm leakage) — the certified
//! kernel MUST NOT be promoted until the offending op is replaced.
//!
//! The harness also reports how often the certified (ordered) digest chain
//! differs from the platform-SIMD path on the same corpus, on THIS arch —
//! the empirical size of the reassociation gap the certification closes.

use katgpt_core::closure::{
    BridgeCertified, G1_SEED, MotifDirections, PrimitiveTransitionGraph,
    certified_ptg_to_motif_embedding, certified_ptg_to_motif_embedding_into, g1_corpus,
    ptg_to_motif_embedding,
};

use std::hint::black_box;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let expect = args
        .iter()
        .position(|a| a == "--expect")
        .and_then(|i| args.get(i + 1).cloned());
    let bench_mode = args.iter().any(|a| a == "--bench");

    let t0 = std::time::Instant::now();
    let corpus = g1_corpus(G1_SEED);
    let gen_us = t0.elapsed().as_micros();

    let G1CorpusShim {
        ptgs,
        dirs,
        freeze_snapshot,
    } = G1CorpusShim::from(corpus);

    // ── The G1 digest: certified kernel over the whole corpus ────────────
    let mut hasher = blake3::Hasher::new();
    let mut feature = vec![0.0f32; dirs.n];
    let mut out = vec![0.0f32; dirs.k];
    let mut platform_gap = 0usize;
    let mut embedding_bytes_total = 0usize;
    for ptg in &ptgs {
        let k = certified_ptg_to_motif_embedding_into(ptg, &dirs, &mut feature, &mut out);
        assert_eq!(k, dirs.k);
        for v in &out {
            hasher.update(&v.to_le_bytes());
        }
        embedding_bytes_total += out.len() * 4;
        // Platform path on the same input — counted, not digested.
        let platform = ptg_to_motif_embedding(ptg, &dirs);
        if platform.as_slice() != out {
            platform_gap += 1;
        }
    }
    let digest = hasher.finalize();
    let hex = format!("{digest}");

    // ── Replay-path self-check: the trait's byte replay must agree ───────
    let raw = katgpt_core::closure::serialize_postcard(&ptgs[0]).expect("postcard");
    let replayed = dirs.replay(&raw, &freeze_snapshot).expect("replay");
    let direct = certified_ptg_to_motif_embedding(&ptgs[0], &dirs);
    assert_eq!(
        replayed,
        katgpt_core::closure::embedding_to_bytes(&direct),
        "replay path must equal the direct kernel"
    );

    // ── Freeze-skew arm: a 1-ULP-flipped snapshot must be rejected ───────
    let mut skew = freeze_snapshot.clone();
    let last = skew.len() - 4;
    let mut bits = u32::from_le_bytes([skew[last], skew[last + 1], skew[last + 2], skew[last + 3]]);
    bits ^= 1;
    skew[last..last + 4].copy_from_slice(&bits.to_le_bytes());
    assert!(
        dirs.replay(&raw, &skew).is_err(),
        "freeze-skew snapshot must be rejected"
    );

    println!("arch          {}", std::env::consts::ARCH);
    println!("os            {}", std::env::consts::OS);
    println!(
        "profile       {}",
        if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        }
    );
    println!("corpus        v1, {} PTGs, seed {G1_SEED:#x}", ptgs.len());
    println!("corpus_gen    {gen_us} us");
    println!("embedding     {embedding_bytes_total} bytes digested");
    println!(
        "platform_gap  {platform_gap}/{} (ordered vs platform-SIMD on THIS arch)",
        ptgs.len()
    );
    println!("digest        {hex}");
    match expect {
        Some(expected) if expected == hex => {
            println!("verdict       G1 PASS (digest matches --expect)");
        }
        Some(expected) => {
            eprintln!("verdict       G1 FAIL: expected {expected}, got {hex}");
            std::process::exit(1);
        }
        None => {
            println!("verdict       record mode \u{2014} rerun on the other arch and compare");
        }
    }

    // ── Optional microbench (--bench): certified vs platform dot cost ────
    // Record-only (the GOAT G2 latency gate is riir-wasm's, Phase 3). The
    // certified kernel is EXPECTED to lose — the sequential fold gives up
    // lane reassociation — the question the record answers is by how much
    // at the certified shapes (K=32, N=64, ≤64-node PTGs).
    if bench_mode {
        // Warmup both paths.
        for ptg in ptgs.iter().take(100) {
            certified_ptg_to_motif_embedding_into(ptg, &dirs, &mut feature, &mut out);
            black_box(ptg_to_motif_embedding(ptg, &dirs));
        }
        let t = std::time::Instant::now();
        for ptg in &ptgs {
            certified_ptg_to_motif_embedding_into(ptg, &dirs, &mut feature, &mut out);
        }
        black_box(&out);
        let certified_ns = t.elapsed().as_nanos() as f64 / ptgs.len() as f64;

        let t = std::time::Instant::now();
        for ptg in &ptgs {
            black_box(ptg_to_motif_embedding(ptg, &dirs));
        }
        let platform_ns = t.elapsed().as_nanos() as f64 / ptgs.len() as f64;
        println!(
            "bench         certified {certified_ns:.0} ns/PTG vs platform {platform_ns:.0} ns/PTG \
             (K=32 N=64, certified is {:>.1}x slower)",
            certified_ns / platform_ns
        );
    }
}

/// Local unpacking shim so the example can destructure the corpus without
/// importing every field type by path.
struct G1CorpusShim {
    ptgs: Vec<PrimitiveTransitionGraph>,
    dirs: MotifDirections,
    freeze_snapshot: Vec<u8>,
}

impl From<katgpt_core::closure::G1Corpus> for G1CorpusShim {
    fn from(c: katgpt_core::closure::G1Corpus) -> Self {
        Self {
            ptgs: c.ptgs,
            dirs: c.dirs,
            freeze_snapshot: c.freeze_snapshot,
        }
    }
}
