//! katgpt-rs Issue 907 — the KVarN V-row bit-ladder attribution bench on
//! REAL rows (Bench 903).
//!
//! Issue 907's measured anomaly (riir-infer Issue 013 T1, Bench 011): on
//! gemma-2 f16 natural-chat decode the V-row bit arms are NOT monotone in
//! bits — b2 1.04% < b4 2.63% < b3 5.72% top-1 flip rate, and p-b3 read
//! BELOW f16. The hypothesis named there: the arms are DIFFERENT
//! quantizers — `KVarNKVCache::with_config` derives the whole machinery
//! from `bits` (skip-varn + grouped-4 RTN at b ≤ 2 vs per-tile var-norm at
//! b ≥ 3) — but the anomaly was never attributed to machinery vs width.
//!
//! This bench runs the issue's step-2 sweep on REAL captured rows: the
//! plain ladder (b2/b3/b4 through `with_config`, the recorded posture) +
//! the CROSSED arms (b3 forced down the b2 machinery, b2/b4 forced up to
//! var-norm) via the measurement-only `set_quant_mode` (feature
//! `quant_mode_override`). If b3's excess error follows the var-norm
//! MACHINERY, the crossed b3 arm (skip-varn machinery) collapses toward
//! b2-class error; if it follows the WIDTH, it stays b3-class. The
//! consumer rule either way: never interpolate V-row quality across bit
//! arms — each arm is a distinct quantizer.
//!
//! Fixture: `.raw/vrow/gemma2_vrows.bin` (gitignored; riir-infer
//! `vrow_capture` emits it — header magic VROW001, corpus BLAKE3 pinned
//! below; the bench SKIPS LOUD without it, never a green zero).
//!
//! Run:
//!   cargo test -p katgpt-kv --release --features kvarn,quant_mode_override \
//!     --test bench_903_kvarn_vrow_real_ladder -- --nocapture

/// BLAKE3 of the fixture file (the capture's reproducibility pin; the
/// capture is deterministic — same model + corpus + token budget →
/// byte-identical).
const FIXTURE_BLAKE3: &str = "5ae9e74cdc49c79a9f62efc77cf129d427f2f6eb718acd2e2e4572bea4cb4bb3";

const TILE: usize = 128;

fn main() {
    eprintln!("[b903] stage 0: enter main");
    // ── the fixture (gitignored; loud skip — never a green zero) ──
    // CWD differs by invocation shape (cargo test = the package dir in some
    // environments, the workspace root in others; direct exe = wherever) —
    // resolve by walking up from CWD until `.raw/vrow/gemma2_vrows.bin` or
    // the filesystem root. The FILE path is joined per-component — pushing a
    // slash-separated relative string then popping walks components that
    // pop() does not strip on Windows (measured: the path oscillated forever,
    // 100% CPU, the wedge this bench shipped with).
    let mut path = None;
    let mut dir = Some(std::env::current_dir().expect("cwd"));
    while let Some(d) = dir {
        let candidate = d.join(".raw").join("vrow").join("gemma2_vrows.bin");
        if candidate.is_file() {
            path = Some(candidate);
            break;
        }
        dir = d.parent().map(|p| p.to_path_buf());
    }
    eprintln!("[b903] stage 1: walk-up done");
    let path = match path {
        Some(p) => p,
        None => {
            eprintln!(
                "SKIP: fixture gemma2_vrows.bin not found in any `.raw/vrow/` above {} — \
                 generate it with riir-infer's \
                 `vrow_capture` (cargo run --release -p riir-infer-core --features vk_calibration \
                 --bin vrow_capture -- --tokens 2048). The bench measured nothing.",
                std::env::current_dir().unwrap_or_default().display()
            );
            return;
        }
    };
    println!("# fixture: {}", path.display());
    eprintln!("[b903] stage 2: reading fixture...");
    let bytes = std::fs::read(&path).expect("read fixture");
    eprintln!("[b903] stage 3: {} bytes read, hashing...", bytes.len());
    let digest = blake3::hash(&bytes).to_hex().to_string();
    eprintln!("[b903] stage 4: hash done = {digest}");
    if FIXTURE_BLAKE3 != digest {
        if FIXTURE_BLAKE3.ends_with("PINNED_AT_FIRST_RUN") {
            eprintln!(
                "PIN-ON-FIRST-RUN: fixture BLAKE3 is {digest} — paste it into \
                 FIXTURE_BLAKE3 and re-run (the pin makes later fixture drift loud)."
            );
        } else {
            panic!(
                "fixture BLAKE3 mismatch: pinned {FIXTURE_BLAKE3}, found {digest} — \
                 the capture drifted; re-pin only after re-deriving every number"
            );
        }
    }

    // ── parse the header ──
    assert_eq!(&bytes[0..7], b"VROW001", "bad fixture magic");
    let n_layers = u32::from_le_bytes(bytes[7..11].try_into().unwrap()) as usize;
    let kv_dim = u32::from_le_bytes(bytes[11..15].try_into().unwrap()) as usize;
    let rows_per_layer = u32::from_le_bytes(bytes[15..19].try_into().unwrap()) as usize;
    let corpus_blake3 = String::from_utf8(bytes[19..83].to_vec()).expect("corpus blake3 utf8");
    let payload = &bytes[83..];
    let want = n_layers * rows_per_layer * kv_dim;
    assert_eq!(
        payload.len(),
        want * 4,
        "fixture payload {} != {}×{}×{}×4",
        payload.len(),
        n_layers,
        rows_per_layer,
        kv_dim
    );
    // The header is 83 bytes — not f32-aligned, so cast_slice refuses (the
    // bytemuck alignment law). One aligned copy of the payload (~0.2 s for
    // 218 MB) beats re-specifying the capture format.
    let mut aligned = Vec::with_capacity(want);
    aligned.extend(
        payload
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap())),
    );
    let rows_f32: &[f32] = &aligned;
    println!(
        "# fixture: {n_layers} layers × {rows_per_layer} rows × kv_dim {kv_dim} | corpus blake3 {corpus_blake3}"
    );

    // ── the arms: (label, bits, override) — override = Some((skip_varn, group_size))
    //    forces the machinery; None = the with_config derivation (the recorded ladder) ──
    type Override = Option<(bool, usize)>;
    let arms: Vec<(&str, u8, Override)> = vec![
        ("plain-b2 (skip-varn+g4, with_config)", 2, None),
        ("plain-b3 (varn, with_config)", 3, None),
        ("plain-b4 (varn, with_config)", 4, None),
        (
            "cross-b3-on-b2-machinery (skip-varn+g4)",
            3,
            Some((true, 4)),
        ),
        (
            "cross-b4-on-b2-machinery (skip-varn+g4)",
            4,
            Some((true, 4)),
        ),
        ("cross-b2-on-varn-machinery", 2, Some((false, 0))),
    ];

    // ── per-arm replay over every layer (rows are layer-major) ──
    // Metrics: mean relative MSE (MSE / mean row energy) + mean cosine, both
    // over (layer, row) cells. One cache per arm reused across layers via
    // reset (the tiles quantize at fill; reset clears between layers).
    let mut table: Vec<(String, f64, f64)> = Vec::new();
    for (label, bits, over) in &arms {
        let cfg = katgpt_kv::kvarn::kv_cache::KVarNConfig {
            n_layers: 1,
            kv_dim,
            max_seq_len: rows_per_layer,
            bits: *bits,
            tile_size: TILE,
            var_norm: Default::default(),
            hadamard: false,
            #[cfg(feature = "targeted_precision")]
            precision_budget: None,
        };
        let mut cache = katgpt_kv::kvarn::kv_cache::KVarNKVCache::with_config(&cfg);
        if let Some((skip_varn, group)) = over {
            cache.set_quant_mode(*skip_varn, *group);
        }
        let mut sum_rel_mse = 0.0f64;
        let mut sum_cos = 0.0f64;
        let mut cells = 0u64;
        let mut deq = vec![0.0f32; kv_dim];
        let layer_stride = rows_per_layer * kv_dim;
        for layer in 0..n_layers {
            cache.reset();
            let rows = &rows_f32[layer * layer_stride..(layer + 1) * layer_stride];
            for pos in 0..rows_per_layer {
                let row = &rows[pos * kv_dim..(pos + 1) * kv_dim];
                cache.store_value(0, pos, row);
            }
            for pos in 0..rows_per_layer {
                let row = &rows[pos * kv_dim..(pos + 1) * kv_dim];
                cache.dequantize_value_into(0, pos, &mut deq);
                let mut dot = 0.0f64;
                let mut se = 0.0f64;
                let mut en = 0.0f64;
                for i in 0..kv_dim {
                    let d = (row[i] - deq[i]) as f64;
                    dot += row[i] as f64 * deq[i] as f64;
                    se += d * d;
                    en += row[i] as f64 * row[i] as f64;
                }
                let na = en.sqrt();
                let nb_sq: f64 = deq.iter().map(|&v| (v as f64) * (v as f64)).sum();
                let cos = if na < 1e-20 || nb_sq < 1e-20 {
                    0.0
                } else {
                    dot / (na * nb_sq.sqrt())
                };
                sum_rel_mse += se / (en / kv_dim as f64).max(1e-20);
                sum_cos += cos as f64;
                cells += 1;
            }
        }
        let rel_mse = sum_rel_mse / cells as f64;
        let cos = sum_cos / cells as f64;
        println!("{label:<44} rel-MSE {rel_mse:.6}  cosine {cos:.6}");
        table.push((label.to_string(), rel_mse, cos));
    }

    // ── the attribution verdict ──
    println!("\n# ── attribution (machinery vs width) ──");
    let get = |name: &str| {
        table
            .iter()
            .find(|(l, _, _)| l.starts_with(name))
            .map(|(_, m, _)| *m)
            .expect("arm present")
    };
    let b2 = get("plain-b2");
    let b3 = get("plain-b3");
    let b4 = get("plain-b4");
    let b3x = get("cross-b3-on-b2-machinery");
    let b4x = get("cross-b4-on-b2-machinery");
    let b2x = get("cross-b2-on-varn-machinery");
    println!(
        "# plain ladder:      b2 {b2:.6}  b3 {b3:.6}  b4 {b4:.6}  (monotone in bits? {})",
        b2 <= b3 && b3 <= b4
    );
    println!(
        "# crossed:           b3→b2-machinery {b3x:.6}   b4→b2-machinery {b4x:.6}   b2→varn {b2x:.6}"
    );
    let toward_b2 = ((b3x - b2).abs() < (b3x - b3).abs()) as i32;
    let width_kept = ((b3x - b3).abs() < (b3x - b2).abs()) as i32;
    if toward_b2 == 1 && width_kept == 0 {
        println!(
            "# VERDICT: b3's excess error FOLLOWS THE MACHINERY — forced down the b2 path it collapses toward b2-class error. The var-norm path at 3 bits is the defect site; the fix is local to the var-norm machinery."
        );
    } else if width_kept == 1 && toward_b2 == 0 {
        println!(
            "# VERDICT: b3's excess error FOLLOWS THE WIDTH — forced onto the b2 machinery it stays b3-class. 3-bit RTN resolution itself is the cost; the machinery is exonerated and the inversion is a width artifact of the var-norm scale field."
        );
    } else {
        println!(
            "# VERDICT: MIXED — the crossed arm lands between both classes (b3x {b3x:.6} vs b2 {b2:.6} / b3 {b3:.6}); the error is a machinery×width INTERACTION. Read the full table before naming a defect site."
        );
    }
    println!(
        "\n# consumer rule (lands either way): never interpolate V-row quality across KVarN bit arms — each arm is a distinct quantizer (Issue 907)."
    );
}
