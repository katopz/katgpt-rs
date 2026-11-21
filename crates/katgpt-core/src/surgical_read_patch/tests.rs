//! The Issue-996 in-module gate battery (split from `surgical_read_patch.rs` per the ~2048-line soft limit; verdict round 1, condition 1). Every law in the parent module docs is pinned by a named test here.
use super::*;
use crate::types::lora_apply;

fn fixture_layers() -> Vec<LayerPatch> {
    vec![
        LayerPatch {
            layer_id: 1,
            neurons: vec![1, 3],
            a: [
                0.5, -1.25, 2.0, 0.75, -0.5, 1.0, 0.25, -2.0, 1.5, 0.125, -0.75, 3.0, -1.0, 0.5,
                2.5, -0.25,
            ]
            .to_vec(),
            gate: Some(vec![0.75, -0.5, 1.25, 2.0]),
            up: Some(vec![-1.0, 0.5, 0.25, -0.75]),
        },
        LayerPatch {
            layer_id: 3,
            neurons: vec![0, 2, 5],
            a: [
                1.0, 0.0, -0.5, 1.5, 2.25, -1.75, 0.5, 0.875, -2.5, 1.0, 0.375, -0.625, 1.75, 0.25,
                -1.25, 0.625,
            ]
            .to_vec(),
            gate: Some(vec![0.5, 1.0, -0.25, 1.75, -1.5, 0.375]),
            up: None,
        },
    ]
}

fn fixture_patch() -> SurgicalReadPatch {
    SurgicalReadPatch::from_parts(8, 1.5, fixture_layers()).expect("fixture validates")
}

fn gen_w() -> Vec<f32> {
    (0..6 * 8).map(|i| ((i % 13) as f32 - 6.0) * 0.25).collect()
}

#[test]
fn merge_is_bit_identical_across_runs() {
    let mut a_w = gen_w();
    let mut b_w = gen_w();
    fixture_patch()
        .merge_projection_into(1, Projection::Gate, &mut a_w, 6)
        .expect("merge a");
    fixture_patch()
        .merge_projection_into(1, Projection::Gate, &mut b_w, 6)
        .expect("merge b");
    for (x, y) in a_w.iter().zip(&b_w) {
        assert_eq!(x.to_bits(), y.to_bits());
    }
}

#[test]
fn merge_touches_only_selected_rows() {
    let mut w = gen_w();
    let original = w.clone();
    fixture_patch()
        .merge_projection_into(1, Projection::Gate, &mut w, 6)
        .expect("merge");
    for r in 0..6 {
        let row = &w[r * 8..(r + 1) * 8];
        let before = &original[r * 8..(r + 1) * 8];
        let changed = row
            .iter()
            .zip(before)
            .any(|(a, b)| a.to_bits() != b.to_bits());
        if r == 1 || r == 3 {
            assert!(changed, "selected row {r} did not move");
        } else {
            assert!(!changed, "unselected row {r} moved");
            for (a, b) in row.iter().zip(before) {
                assert_eq!(a.to_bits(), b.to_bits());
            }
        }
    }
}

#[test]
fn write_frozen_across_projections() {
    let mut gate_w = gen_w();
    let mut up_w = gen_w();
    let original = gen_w();
    fixture_patch()
        .merge_projection_into(1, Projection::Gate, &mut gate_w, 6)
        .expect("gate merge");
    fixture_patch()
        .merge_projection_into(1, Projection::Up, &mut up_w, 6)
        .expect("up merge");
    for r in 0..6 {
        let selected = r == 1 || r == 3;
        let row_g = &gate_w[r * 8..(r + 1) * 8];
        let row_u = &up_w[r * 8..(r + 1) * 8];
        let before = &original[r * 8..(r + 1) * 8];
        if selected {
            // Aggregate row change — the spec fixture carries one
            // exact-cancellation cell (up row 1 j=4), the row-level
            // law is what pins "patches alive".
            assert!(
                row_g
                    .iter()
                    .zip(before)
                    .any(|(a, b)| a.to_bits() != b.to_bits()),
                "gate row {r} dead"
            );
            assert!(
                row_u
                    .iter()
                    .zip(before)
                    .any(|(a, b)| a.to_bits() != b.to_bits()),
                "up row {r} dead"
            );
        } else {
            for j in 0..8 {
                let idx = r * 8 + j;
                assert_eq!(gate_w[idx].to_bits(), original[idx].to_bits());
                assert_eq!(up_w[idx].to_bits(), original[idx].to_bits());
            }
        }
    }
}

#[test]
fn canonical_round_trip_bit_identical() {
    let patch = fixture_patch();
    let bytes = patch.canonical_bytes();
    let decoded = SurgicalReadPatch::from_canonical(&bytes).expect("decode");
    assert_eq!(decoded, patch);
    assert_eq!(decoded.canonical_bytes(), bytes);
}

#[test]
fn golden_digest_pin() {
    // The encoding-drift tripwire: ANY change to the canonical form
    // (field order, widths, magic, version) moves this digest. The
    // pin was captured at first run of this exact fixture; a
    // legitimate format bump lands WITH a new pin and a version
    // bump, never by silently editing this row.
    const GOLDEN_CANONICAL_DIGEST_HEX: &str =
        "999984132f31ee15f6ca24d9e682d27861ac8d74b2a1f79129146b2896f6a982";
    let hex = blake3::hash(&fixture_patch().canonical_bytes())
        .to_hex()
        .to_string();
    assert_eq!(
        hex, GOLDEN_CANONICAL_DIGEST_HEX,
        "computed canonical digest: {hex}"
    );
}

#[test]
fn tampered_canonical_refuses() {
    let mut artifact = SurgicalPatchArtifact::mint(&fixture_patch(), 1);
    let last = artifact.canonical.len() - 1;
    artifact.canonical[last] ^= 0xFF;
    assert!(matches!(
        artifact.verify_digest(),
        Err(SurgicalPatchError::DigestMismatch { .. })
    ));
}

#[test]
fn bad_magic_version_truncated_trailing_refuse() {
    let canonical = fixture_patch().canonical_bytes();

    let mut bad_magic = canonical.clone();
    bad_magic[0] = b'X';
    assert!(matches!(
        SurgicalReadPatch::from_canonical(&bad_magic),
        Err(SurgicalPatchError::Encoding("bad canonical magic"))
    ));

    let mut bad_version = canonical.clone();
    bad_version[5] = 0xFF;
    assert!(matches!(
        SurgicalReadPatch::from_canonical(&bad_version),
        Err(SurgicalPatchError::Encoding(
            "unsupported surgical patch canonical version (this build reads 1)"
        ))
    ));

    assert!(matches!(
        SurgicalReadPatch::from_canonical(&canonical[..canonical.len() - 3]),
        Err(SurgicalPatchError::Encoding("truncated canonical patch"))
    ));

    let mut trailing = canonical.clone();
    trailing.extend_from_slice(&[0u8; 4]);
    assert!(matches!(
        SurgicalReadPatch::from_canonical(&trailing),
        Err(SurgicalPatchError::Encoding(
            "trailing bytes after canonical patch"
        ))
    ));
}

#[test]
fn non_canonical_order_refuses() {
    // Neurons written DESCENDING inside one layer.
    let mut enc = Enc::new();
    enc.raw(&MAGIC);
    enc.u32(SURGICAL_PATCH_FORMAT_VERSION);
    enc.u32(4);
    enc.u32(1);
    enc.f32(1.0);
    enc.u32(1); // one layer
    enc.u32(7); // layer_id
    enc.u32(2); // two neurons
    enc.u32(3);
    enc.u32(1); // DESCENDING
    enc.u8v(1); // gate present
    enc.u8v(0); // up absent
    for _ in 0..4 {
        enc.f32(0.5);
    }
    for _ in 0..2 {
        enc.f32(0.25);
    }
    let err = SurgicalReadPatch::from_canonical(&enc.0).unwrap_err();
    match err {
        SurgicalPatchError::Encoding(what) => {
            assert!(what.contains("neuron"), "got: {what}");
        }
        other => panic!("expected Encoding(neuron order), got {other:?}"),
    }

    // Two layers sharing one layer_id.
    let mut enc = Enc::new();
    enc.raw(&MAGIC);
    enc.u32(SURGICAL_PATCH_FORMAT_VERSION);
    enc.u32(4);
    enc.u32(1);
    enc.f32(1.0);
    enc.u32(2); // two layers, SAME id
    for _ in 0..2 {
        enc.u32(7); // layer_id 7, 7
        enc.u32(1); // one neuron
        enc.u32(0);
        enc.u8v(1);
        enc.u8v(0);
        for _ in 0..4 {
            enc.f32(0.5);
        }
        enc.f32(0.25);
    }
    let err = SurgicalReadPatch::from_canonical(&enc.0).unwrap_err();
    match err {
        SurgicalPatchError::Encoding(what) => {
            assert!(what.contains("layer"), "got: {what}");
        }
        other => panic!("expected Encoding(layer order), got {other:?}"),
    }
}

#[test]
fn non_finite_refuses() {
    let mut nan_dir = fixture_patch();
    nan_dir.layers[0].a[0] = f32::NAN;
    assert!(matches!(
        SurgicalReadPatch::from_parts(nan_dir.in_dim, nan_dir.scale, nan_dir.layers),
        Err(SurgicalPatchError::NonFinite("direction"))
    ));

    let mut inf_gate = fixture_patch();
    inf_gate.layers[1].gate.as_mut().unwrap()[0] = f32::INFINITY;
    assert!(matches!(
        SurgicalReadPatch::from_parts(inf_gate.in_dim, inf_gate.scale, inf_gate.layers),
        Err(SurgicalPatchError::NonFinite("gate"))
    ));

    assert!(matches!(
        SurgicalReadPatch::from_parts(8, f32::NAN, fixture_layers()),
        Err(SurgicalPatchError::NonFinite("scale"))
    ));
}

#[test]
fn from_parts_sorts_and_refuses_duplicates() {
    let reversed = {
        let mut layers = fixture_layers();
        layers.reverse();
        SurgicalReadPatch::from_parts(8, 1.5, layers).expect("reversed fixture validates")
    };
    assert_eq!(reversed.layers[0].layer_id, 1);

    let mut dup = fixture_layers();
    let first = dup[0].clone();
    dup.push(first);
    assert!(matches!(
        SurgicalReadPatch::from_parts(8, 1.5, dup),
        Err(SurgicalPatchError::Validation(
            "surgical patch: duplicate layer_id"
        ))
    ));
}

#[test]
fn rank_mismatch_across_layers_refuses() {
    let mut layers = fixture_layers();
    layers[1].a.truncate(8);
    layers[1].gate.as_mut().unwrap().truncate(3);
    assert!(matches!(
        SurgicalReadPatch::from_parts(8, 1.5, layers),
        Err(SurgicalPatchError::DimMismatch {
            expected: 16,
            got: 8
        })
    ));
}

#[test]
fn merge_unknown_layer_missing_projection_dim_row_refuse() {
    let mut w = gen_w();
    assert!(matches!(
        fixture_patch().merge_projection_into(9, Projection::Gate, &mut w, 6),
        Err(SurgicalPatchError::UnknownLayer { layer_id: 9 })
    ));
    let mut w = gen_w();
    assert!(matches!(
        fixture_patch().merge_projection_into(3, Projection::Up, &mut w, 6),
        Err(SurgicalPatchError::ProjectionMissing {
            layer_id: 3,
            which: Projection::Up
        })
    ));
    let mut short = vec![0.0f32; 6 * 4];
    assert!(matches!(
        fixture_patch().merge_projection_into(1, Projection::Gate, &mut short, 6),
        Err(SurgicalPatchError::DimMismatch {
            expected: 48,
            got: 24
        })
    ));

    let out_of_range = SurgicalReadPatch::from_parts(
        8,
        1.0,
        vec![LayerPatch {
            layer_id: 7,
            neurons: vec![100],
            a: vec![1.0; 8],
            gate: Some(vec![1.0]),
            up: None,
        }],
    )
    .expect("fixture validates");
    let mut w = vec![0.0f32; 6 * 8];
    assert!(matches!(
        out_of_range.merge_projection_into(7, Projection::Gate, &mut w, 6),
        Err(SurgicalPatchError::RowOutOfRange {
            layer_id: 7,
            row: 100,
            rows: 6
        })
    ));
    // ATOMIC: the refused merge wrote nothing.
    for v in &w {
        assert_eq!(v.to_bits(), 0.0f32.to_bits());
    }
}

#[test]
fn expand_to_lora_matches_row_merge() {
    let patch = fixture_patch();
    let x = [0.5f32, -1.0, 2.0, 0.25, -0.75, 1.5, -2.0, 1.0];
    let mut w = gen_w();
    let base_out: Vec<f32> = (0..6)
        .map(|r| dot_f32_ordered(&w[r * 8..(r + 1) * 8], &x))
        .collect();
    patch
        .merge_projection_into(1, Projection::Gate, &mut w, 6)
        .expect("merge");
    let adapter = patch
        .projection_to_lora(1, Projection::Gate, 6)
        .expect("expand");
    assert_eq!(adapter.rank, 2);
    assert_eq!(adapter.alpha, 3.0); // scale 1.5 × rank 2 — exact f32
    let mut delta = vec![0.0f32; 6];
    let mut lora_buf = vec![0.0f32; 2];
    lora_apply(&mut delta, &adapter, &x, &mut lora_buf);
    for r in 0..6 {
        let merged_out = dot_f32_ordered(&w[r * 8..(r + 1) * 8], &x);
        if r == 1 || r == 3 {
            assert_ne!(delta[r], 0.0, "row {r} delta dead");
            let approx = (merged_out - (base_out[r] + delta[r])).abs();
            assert!(approx < 1e-4, "row {r} lora mismatch: {approx}");
        } else {
            assert_eq!(delta[r].to_bits(), 0, "row {r} delta moved");
            assert_eq!(
                merged_out.to_bits(),
                base_out[r].to_bits(),
                "row {r} moved through the lora path"
            );
        }
    }
}

#[test]
fn expand_b_rows_off_neurons_are_positive_zero() {
    let patch = fixture_patch();
    let adapter = patch
        .projection_to_lora(1, Projection::Gate, 6)
        .expect("expand");
    for r in 0..6 {
        let row = &adapter.b[r * 2..(r + 1) * 2];
        match r {
            1 => assert_eq!(row, &[0.75f32, -0.5]),
            3 => assert_eq!(row, &[1.25f32, 2.0]),
            _ => {
                for v in row {
                    assert_eq!(v.to_bits(), 0.0f32.to_bits(), "row {r} not +0.0");
                }
            }
        }
    }
}

#[test]
fn ranking_preserved_on_patch_orthogonal_input() {
    let patch = SurgicalReadPatch::from_parts(
        8,
        1.0,
        vec![LayerPatch {
            layer_id: 0,
            neurons: vec![0, 2, 5],
            a: vec![2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            gate: Some(vec![1.0, -0.5, 0.25]),
            up: None,
        }],
    )
    .expect("rank-1 fixture validates");
    let x0 = [0.0f32, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0];
    let mut w = gen_w();
    let base_out: Vec<f32> = (0..6)
        .map(|r| dot_f32_ordered(&w[r * 8..(r + 1) * 8], &x0))
        .collect();
    patch
        .merge_projection_into(0, Projection::Gate, &mut w, 6)
        .expect("merge");
    for r in 0..6 {
        let merged = dot_f32_ordered(&w[r * 8..(r + 1) * 8], &x0);
        assert_eq!(
            merged.to_bits(),
            base_out[r].to_bits(),
            "orthogonal input ranking broke at row {r}"
        );
    }
    // Alive check: a non-orthogonal input must move the patched rows.
    let x1 = [1.0f32; 8];
    let unmerged = gen_w();
    let base1 = dot_f32_ordered(&unmerged[0..8], &x1);
    let mut fresh = unmerged.clone();
    patch
        .merge_projection_into(0, Projection::Gate, &mut fresh, 6)
        .expect("merge");
    let merged1 = dot_f32_ordered(&fresh[0..8], &x1);
    assert_ne!(merged1.to_bits(), base1.to_bits(), "alive check failed");
}

#[test]
fn patch_weight_epoch_stable_and_field_sensitive() {
    let e1 = fixture_patch().weight_epoch();
    let e2 = fixture_patch().weight_epoch();
    assert_eq!(e1, e2);

    let mut bumped = fixture_patch();
    bumped.layers[0].a[0] += 1.0;
    let bumped = SurgicalReadPatch::from_parts(bumped.in_dim, bumped.scale, bumped.layers)
        .expect("bumped fixture validates");
    assert_ne!(bumped.weight_epoch(), e1);

    let rescaled = SurgicalReadPatch::from_parts(8, 1.75, fixture_layers())
        .expect("rescaled fixture validates");
    assert_ne!(rescaled.weight_epoch(), e1);
}

#[test]
fn slot_monotonic_laws() {
    let base = fixture_patch();
    let rescaled = SurgicalReadPatch::from_parts(8, 1.75, fixture_layers())
        .expect("rescaled fixture validates");

    let mut slot = SurgicalPatchSlot::new();
    let a1 = SurgicalPatchArtifact::mint(&base, 5);
    assert!(matches!(
        slot.install(&a1),
        Ok(SurgicalPatchInstall::Advanced)
    ));
    assert!(matches!(
        slot.install(&a1),
        Ok(SurgicalPatchInstall::Idempotent)
    ));

    let a2 = SurgicalPatchArtifact::mint(&rescaled, 5);
    assert!(matches!(
        slot.install(&a2),
        Err(SurgicalPatchError::Fork {
            mounted_epoch: 5,
            ..
        })
    ));

    let a3 = SurgicalPatchArtifact::mint(&rescaled, 6);
    assert!(matches!(
        slot.install(&a3),
        Ok(SurgicalPatchInstall::Advanced)
    ));

    let a4 = SurgicalPatchArtifact::mint(&base, 4);
    assert!(matches!(
        slot.install(&a4),
        Err(SurgicalPatchError::Downgrade {
            mounted_epoch: 6,
            incoming_epoch: 4
        })
    ));

    assert_eq!(slot.epoch(), Some(6));
    assert_eq!(slot.digest(), Some(a3.patch_digest));
    assert_eq!(slot.patch().map(|p| p.scale), Some(1.75));
}

#[test]
fn slot_refuses_corrupt_and_stays_mounted() {
    let mut slot = SurgicalPatchSlot::new();
    let mounted = SurgicalPatchArtifact::mint(&fixture_patch(), 3);
    slot.install(&mounted).expect("initial mount");

    let mut corrupt = SurgicalPatchArtifact::mint(&fixture_patch(), 4);
    corrupt.canonical[40] ^= 0xFF;
    assert!(matches!(
        slot.install(&corrupt),
        Err(SurgicalPatchError::DigestMismatch { .. })
    ));
    assert_eq!(slot.epoch(), Some(3));
}

#[test]
fn artifact_bytes_round_trip_and_tamper_refuses() {
    let artifact = SurgicalPatchArtifact::mint(&fixture_patch(), 11);
    let bytes = artifact.to_bytes();
    let restored = SurgicalPatchArtifact::from_bytes(&bytes).expect("round trip");
    assert_eq!(restored, artifact);

    let mut tampered = bytes.clone();
    let last = tampered.len() - 1;
    tampered[last] ^= 0xFF;
    assert!(matches!(
        SurgicalPatchArtifact::from_bytes(&tampered),
        Err(SurgicalPatchError::DigestMismatch { .. })
    ));
}

fn kb_patch() -> SurgicalReadPatch {
    let d = 4096usize;
    let k = 4usize;
    let mut cursor = 0u32;
    let mut layers = Vec::new();
    for layer_i in 0..20u32 {
        let n = if layer_i < 9 { 5usize } else { 4usize };
        let neurons: Vec<u32> = (0..n).map(|j| cursor + (j as u32) * 97).collect();
        cursor += n as u32;
        layers.push(LayerPatch {
            layer_id: layer_i,
            neurons,
            a: (0..k * d)
                .map(|i| ((i % 17) as f32 - 8.0) * 0.125)
                .collect(),
            gate: Some(
                (0..n * k)
                    .map(|i| ((i % 23) as f32 - 11.0) * 0.25)
                    .collect(),
            ),
            up: Some((0..n * k).map(|i| ((i % 19) as f32 - 9.0) * 0.5).collect()),
        });
    }
    SurgicalReadPatch::from_parts(d, 1.0, layers).expect("kb fixture validates")
}

#[test]
fn kb_scale_artifact_bound() {
    let patch = kb_patch();
    let bytes = patch.canonical_bytes().len();
    // a-dominant: 20 × 4 × 4096 × 4 B = 1_310_720; headers < 0.1%.
    assert!(bytes <= 1_320_000, "canonical patch bytes: {bytes}");
    // The dense rank-4 gate+up LoRA twin at intermediate 12288.
    let dense = 20usize * 2 * (4 * 4096 + 12288 * 4) * 4;
    assert!(dense >= 7 * bytes, "dense {dense} vs sparse {bytes}");
}

#[test]
fn census_scores_exact_zero_on_null_and_planted_top() {
    let n = 32usize;
    let samples = 4usize;
    let mut base = vec![0.0f32; samples * n];
    for s in 0..samples {
        for i in 0..n {
            base[s * n + i] = ((s * n + i) % 13) as f32 * 0.5 - 3.0;
        }
    }
    let mut persona = base.clone();
    for s in 0..samples {
        persona[s * n + 3] += 10.0;
        persona[s * n + 17] += 10.0;
    }
    let mut scores = vec![0.0f32; n];
    census_scores_into(&persona, &base, samples, samples, &mut scores).expect("census");
    let mut work = vec![0.0f32; n];
    let mut top = [0u32; 2];
    let written = select_top_into(&scores, &mut work, 2, &mut top).expect("select");
    assert_eq!(written, 2);
    assert_eq!(&top, &[3, 17], "tie must keep the first index");

    // Null control: identical corpora → every score exactly zero.
    let mut null_scores = vec![1.0f32; n];
    census_scores_into(&base, &base, samples, samples, &mut null_scores).expect("null census");
    for v in &null_scores {
        assert_eq!(v.to_bits(), 0.0f32.to_bits());
    }
}

#[test]
fn select_top_first_index_tie_rule_and_nan_skip() {
    let mut work = vec![0.0f32; 4];
    let mut out = [0u32; 2];
    let written = select_top_into(&[5.0, 7.0, 7.0, 1.0], &mut work, 2, &mut out).expect("select");
    assert_eq!(written, 2);
    assert_eq!(&out, &[1, 2], "tie keeps the FIRST index (Issue-940)");

    let mut work = vec![0.0f32; 4];
    let mut out = [0u32; 2];
    let written =
        select_top_into(&[f32::NAN, 3.0, f32::NAN, 1.0], &mut work, 2, &mut out).expect("select");
    assert_eq!(written, 2);
    assert_eq!(&out, &[1, 3], "NaN is never selected");
}

#[test]
fn construct_from_census_matches_reference() {
    let d = 6usize;
    let rows = 4usize;
    let persona_mean = [1.0f32, 2.0, -1.0, 0.5, 3.0, -2.0];
    let base_mean = [0.5f32, 1.0, -0.5, 0.25, 1.0, -1.0];
    let w: Vec<f32> = (0..rows * d)
        .map(|i| ((i % 11) as f32 - 5.0) * 0.5)
        .collect();
    let neurons = [0u32, 2, 3];
    let mut a = vec![0.0f32; d];
    let mut b = vec![0.0f32; neurons.len()];
    construct_from_census_into(
        &persona_mean,
        &base_mean,
        &w,
        rows,
        &neurons,
        d,
        &mut a,
        &mut b,
    )
    .expect("construct");

    // Independent transcription of the documented reference.
    let mut ra: Vec<f32> = persona_mean
        .iter()
        .zip(&base_mean)
        .map(|(p, q)| p - q)
        .collect();
    let norm = dot_f32_ordered(&ra, &ra).sqrt();
    for v in &mut ra {
        *v /= norm;
    }
    for (i, &row) in neurons.iter().enumerate() {
        let expected = dot_f32_ordered(&w[row as usize * d..(row as usize + 1) * d], &ra);
        assert_eq!(b[i].to_bits(), expected.to_bits());
    }
    for j in 0..d {
        assert_eq!(a[j].to_bits(), ra[j].to_bits());
    }
}

#[test]
fn construct_refuses_null_corpus() {
    let means = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let base_rows = vec![0.0f32; 6];
    let mut a = [0.0f32; 6];
    let mut b = [0.0f32; 1];
    let err = construct_from_census_into(&means, &means, &base_rows, 1, &[0], 6, &mut a, &mut b)
        .unwrap_err();
    assert!(matches!(err, SurgicalPatchError::NoSeparation));
}

#[test]
fn construct_assemble_merge_bit_matches_reference() {
    let d = 6usize;
    let rows = 8usize;
    let persona_mean: Vec<f32> = (0..d).map(|i| i as f32 + 0.5).collect();
    let base_mean: Vec<f32> = (0..d).map(|i| i as f32 * 0.25).collect();
    let w: Vec<f32> = (0..rows * d)
        .map(|i| ((i % 11) as f32 - 5.0) * 0.5)
        .collect();
    let neurons = [1u32, 4, 6];
    let mut a = vec![0.0f32; d];
    let mut b = vec![0.0f32; neurons.len()];
    construct_from_census_into(
        &persona_mean,
        &base_mean,
        &w,
        rows,
        &neurons,
        d,
        &mut a,
        &mut b,
    )
    .expect("construct");

    let patch = SurgicalReadPatch::from_parts(
        d,
        2.0,
        vec![LayerPatch {
            layer_id: 2,
            neurons: neurons.to_vec(),
            a: a.clone(),
            gate: Some(b.clone()),
            up: None,
        }],
    )
    .expect("assembled patch validates");
    let mut merged = w.clone();
    patch
        .merge_projection_into(2, Projection::Gate, &mut merged, rows)
        .expect("merge");

    // Independent reference triple loop over the pinned fold order.
    let mut reference = w.clone();
    for (i, &row) in neurons.iter().enumerate() {
        let coef = 2.0f32 * b[i];
        for (j, &av) in a.iter().enumerate() {
            reference[row as usize * d + j] += coef * av;
        }
    }
    assert_eq!(merged.len(), reference.len());
    for (m, r) in merged.iter().zip(&reference) {
        assert_eq!(m.to_bits(), r.to_bits());
    }
}

#[test]
fn fact_swap_plan_deterministic_and_pinned() {
    const GOLDEN_FACT_SWAP_DIGEST_HEX: &str =
        "770230988ce3ae438697aa96ae7de2d2864962c5dcf891ba912b45371fd9fca1";
    let plan = plan_fact_swaps(7, 8, 3, 42).expect("plan builds");
    let again = plan_fact_swaps(7, 8, 3, 42).expect("plan rebuilds");
    assert_eq!(plan, again);
    assert_eq!(plan.swaps.len(), 3);
    let mut seen = std::collections::HashSet::new();
    for pair in &plan.swaps {
        assert!(pair.left < pair.right);
        assert!(pair.right < 8);
        assert!(seen.insert(*pair), "duplicate pair {pair:?}");
    }
    let hex = hex32(&plan.plan_digest);
    assert_eq!(
        hex, GOLDEN_FACT_SWAP_DIGEST_HEX,
        "computed fact-swap digest: {hex}"
    );
}

#[test]
fn fact_swap_plan_refusals() {
    assert!(matches!(
        plan_fact_swaps(7, 1, 1, 1),
        Err(SurgicalPatchError::Validation(_))
    ));
    assert!(matches!(
        plan_fact_swaps(7, 4, 7, 1),
        Err(SurgicalPatchError::Validation(_))
    ));
    let empty = plan_fact_swaps(7, 8, 0, 1).expect("zero swaps is the control plan");
    assert!(empty.swaps.is_empty());
}
