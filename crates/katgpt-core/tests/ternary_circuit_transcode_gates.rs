#![cfg(feature = "ternary_circuit")]
//! Issue 1225 T1 gates — the npz→`.cact` transcode lane.
//!
//! Arms:
//! 1. **Determinism + pin** — the transcode re-derived from the committed
//!    intermediate is BYTE-IDENTICAL to the committed `weights.cact`, and its
//!    BLAKE3 + `archive_id` match `pins.json` (the drift catcher).
//! 2. **Input pins** — the npz and all three intermediate members hash to
//!    their recorded pins (corruption catcher).
//! 3. **Archive + manifest laws** — the committed archive passes the reader's
//!    fail-closed ladder; layer count/dims match the manifest; trit layers
//!    carry neutral scales, vec layers are `cols = 0` with the declared flat
//!    count; scale roles follow the source's pairing rule.
//! 4. **Bitplane round-trip** — every trit of every layer, read back through
//!    the archive (`to_ternary_weights().get`), equals the unpacker decode of
//!    the intermediate at the mapped flat index (the two encodings meet).
//! 5. **Tamper sensitivity** — one flipped intermediate byte moves the
//!    archive bytes AND the `archive_id` (the vacuous-exactness catch at the
//!    container level; output-level falsifiers are Arm A's).

use std::path::Path;

use katgpt_core::kpt_archive::{KptArchive, build_archive};
use katgpt_core::ternary_circuit::{
    SOURCE_PIN, assemble_layers, parse_meta, scale_key_for, trit_at,
};

fn fixture_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/ternary_circuit")
}

fn blake3_hex(bytes: &[u8]) -> String {
    format!("{}", blake3::hash(bytes))
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// The pins file is flat string/number JSON; read exactly the fields the
/// gate asserts, without a serde round-trip of the whole document.
fn pins_of(text: &str) -> std::collections::BTreeMap<String, String> {
    let text = text.trim();
    assert!(
        text.starts_with('{') && text.ends_with('}'),
        "pins.json shape"
    );
    let inner = &text[1..text.len() - 1];
    let mut out = std::collections::BTreeMap::new();
    for part in inner.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let Some((k, v)) = part.split_once(':') else {
            panic!("pins.json pair without ':' — {part:?}");
        };
        let k = k.trim().trim_matches('"');
        let v = v.trim().trim_matches('"');
        assert!(
            !k.contains('"') && !v.contains('"'),
            "pins.json nested shape"
        );
        out.insert(k.to_string(), v.to_string());
    }
    out
}

struct Inputs {
    npz: Vec<u8>,
    meta_txt: String,
    trits: Vec<u8>,
    vecs_raw: Vec<u8>,
    vecs: Vec<f32>,
    cact_committed: Vec<u8>,
    pins: std::collections::BTreeMap<String, String>,
}

fn load_inputs() -> Inputs {
    let dir = fixture_dir();
    let npz = std::fs::read(dir.join("weights.npz")).expect("weights.npz");
    let meta_txt = std::fs::read_to_string(dir.join("intermediate/meta.txt")).expect("meta.txt");
    let trits = std::fs::read(dir.join("intermediate/trits.bin")).expect("trits.bin");
    let vecs_raw = std::fs::read(dir.join("intermediate/vecs.f32")).expect("vecs.f32");
    let cact_committed = std::fs::read(dir.join("compiled/weights.cact")).expect("weights.cact");
    let pins =
        pins_of(&std::fs::read_to_string(dir.join("compiled/pins.json")).expect("pins.json"));
    let vecs: Vec<f32> = vecs_raw
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();
    Inputs {
        npz,
        meta_txt,
        trits,
        vecs_raw,
        vecs,
        cact_committed,
        pins,
    }
}

fn retranscode(inp: &Inputs) -> Vec<u8> {
    let meta = parse_meta(&inp.meta_txt).expect("meta grammar");
    let out = assemble_layers(&inp.trits, &inp.vecs, &meta).expect("assembly laws");
    let layers: Vec<&katgpt_types::TernaryWeights> = out.layers.iter().collect();
    build_archive(&layers).expect("build")
}

#[test]
fn transcode_is_deterministic_and_byte_matches_committed_cact() {
    let inp = load_inputs();
    let cact = retranscode(&inp);
    assert_eq!(
        cact, inp.cact_committed,
        "re-derived .cact differs from the committed artifact — the transcode is not deterministic or the fixture drifted"
    );
    assert_eq!(
        blake3_hex(&cact),
        inp.pins["cact_blake3"],
        "cact_blake3 pin"
    );
    assert_eq!(
        inp.cact_committed.len().to_string(),
        inp.pins["cact_bytes"],
        "cact_bytes pin"
    );
    let verified = KptArchive::from_bytes(&cact).expect("fail-closed ladder");
    assert_eq!(hex(&verified.archive_id()), inp.pins["cact_archive_id"]);
}

#[test]
fn input_pins_hold() {
    let inp = load_inputs();
    assert_eq!(blake3_hex(&inp.npz), inp.pins["npz_blake3"], "npz pin");
    assert_eq!(
        blake3_hex(&inp.trits),
        inp.pins["trits_blake3"],
        "trits pin"
    );
    assert_eq!(
        blake3_hex(&inp.vecs_raw),
        inp.pins["vecs_blake3"],
        "vecs pin"
    );
    assert_eq!(
        blake3_hex(inp.meta_txt.as_bytes()),
        inp.pins["meta_blake3"],
        "meta pin"
    );
    assert_eq!(inp.pins["source_pin"], SOURCE_PIN);
}

#[test]
fn archive_and_manifest_laws_hold() {
    let inp = load_inputs();
    let verified = KptArchive::from_bytes(&inp.cact_committed).expect("fail-closed ladder");
    let manifest_text =
        std::fs::read_to_string(fixture_dir().join("compiled/manifest.json")).expect("manifest");
    let doc: serde_json::Value = serde_json::from_str(&manifest_text).expect("manifest parses");
    let rows = doc["layers"].as_array().expect("layers array");
    assert_eq!(verified.layers().len(), rows.len(), "layer count");
    assert_eq!(
        verified.layers().len(),
        inp.pins["layer_count"].parse::<usize>().unwrap(),
        "layer_count pin"
    );

    let meta = parse_meta(&inp.meta_txt).expect("meta grammar");
    let scale_names: std::collections::BTreeSet<String> =
        meta.trits.iter().map(|m| scale_key_for(&m.name)).collect();

    for (i, row) in rows.iter().enumerate() {
        let entry = &verified.layers()[i];
        assert_eq!(row["idx"].as_u64().unwrap() as usize, i, "row {i} idx");
        assert_eq!(
            row["rows"].as_u64().unwrap() as usize,
            entry.rows,
            "row {i} rows"
        );
        assert_eq!(
            row["cols"].as_u64().unwrap() as usize,
            entry.cols,
            "row {i} cols"
        );
        match row["kind"].as_str().unwrap() {
            "trit" => {
                let meta_e = &meta.trits[i];
                assert_eq!(row["name"].as_str().unwrap(), meta_e.name, "trit {i} name");
                assert!(entry.cols > 0, "trit layer {i} must be a matrix");
                // Neutral-scale law: the source folds its per-channel scales
                // along the LAST axis at forward-load time — a cols-axis
                // semantics the container's per-row row_scale must not
                // impersonate. The transcode emits neutral 1.0 everywhere.
                let tw = verified.to_ternary_weights(i);
                assert!(
                    tw.row_scale.iter().all(|&s| s == 1.0),
                    "trit layer {i} row_scale not neutral"
                );
            }
            "vec" => {
                let meta_e = &meta.vecs[i - meta.trits.len()];
                assert_eq!(row["name"].as_str().unwrap(), meta_e.name, "vec {i} name");
                assert_eq!(entry.cols, 0, "vec layer {i} must be degenerate");
                let flat: usize = meta_e.dims.iter().product();
                assert_eq!(entry.rows, flat, "vec layer {i} flat count");
                let expect_scale = scale_names.contains(&meta_e.name);
                let role = row["role"].as_str().unwrap();
                assert_eq!(
                    role,
                    if expect_scale { "scale" } else { "data" },
                    "vec layer {i} role"
                );
                if expect_scale {
                    let last_dim = *meta
                        .trits
                        .iter()
                        .find(|m| scale_key_for(&m.name) == meta_e.name)
                        .unwrap()
                        .dims
                        .last()
                        .unwrap();
                    assert_eq!(flat, last_dim, "scale {i} length law");
                }
                // The payload round-trips verbatim through the archive.
                let tw = verified.to_ternary_weights(i);
                let offset: usize = meta
                    .vecs
                    .iter()
                    .take(i - meta.trits.len())
                    .map(|e| e.count)
                    .sum();
                assert_eq!(tw.row_scale, &inp.vecs[offset..offset + flat]);
            }
            other => panic!("unknown layer kind {other:?} at row {i}"),
        }
    }
    assert_eq!(rows.len(), meta.trits.len() + meta.vecs.len());
}

#[test]
fn bitplane_roundtrip_matches_unpacker() {
    let inp = load_inputs();
    let verified = KptArchive::from_bytes(&inp.cact_committed).expect("fail-closed ladder");
    let meta = parse_meta(&inp.meta_txt).expect("meta grammar");
    let mut byte_offset = 0usize;
    for (i, e) in meta.trits.iter().enumerate() {
        let elems: usize = e.dims.iter().product();
        let tw = verified.to_ternary_weights(i);
        assert_eq!((tw.rows, tw.cols), flat_shape(&e.dims), "layer {i} dims");
        for flat in 0..elems {
            let r = flat / tw.cols;
            let c = flat % tw.cols;
            let name = &e.name;
            assert_eq!(
                tw.get(r, c),
                trit_at(&inp.trits, byte_offset * 5 + flat),
                "layer {i} ({name}) flat {flat}"
            );
        }
        byte_offset += e.packed_len;
    }
    assert_eq!(byte_offset, inp.trits.len());
}

/// The storage shape law: 2-D `(a, b)` → `(a, b)`; 3-D `(a, b, c)` →
/// `(a, b·c)` (the source's own flat C-order reshape).
fn flat_shape(dims: &[usize]) -> (usize, usize) {
    match dims {
        [a, b] => (*a, *b),
        [a, b, c] => (*a, b * c),
        other => panic!("unsupported rank {}", other.len()),
    }
}

#[test]
fn single_trit_flip_moves_the_archive() {
    let inp = load_inputs();
    let baseline = retranscode(&inp);
    let mut tampered = inp.trits.clone();
    // Flip the low bit of the first payload byte — digit 0 of byte 0 always
    // changes, so flat trit 0 always moves.
    tampered[0] ^= 0b0000_0001;
    let meta = parse_meta(&inp.meta_txt).expect("meta grammar");
    let out = assemble_layers(&tampered, &inp.vecs, &meta).expect("assembly laws");
    let layers: Vec<&katgpt_types::TernaryWeights> = out.layers.iter().collect();
    let cact = build_archive(&layers).expect("build");
    assert_ne!(cact, baseline, "a flipped trit must move the archive bytes");
    let a = KptArchive::from_bytes(&baseline).expect("baseline verifies");
    let b = KptArchive::from_bytes(&cact)
        .expect("tampered archive still verifies (integrity intact, content moved)");
    assert_ne!(a.archive_id(), b.archive_id(), "archive_id must move");
}
