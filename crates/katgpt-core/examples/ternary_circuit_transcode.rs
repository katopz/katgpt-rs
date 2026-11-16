#![cfg(feature = "ternary_circuit")]
//! Issue 1225 T1b — the npz→`.cact` transcode converter (fixture builder).
//!
//! Reads the committed fixture set under `tests/fixtures/ternary_circuit/`:
//!
//! - `weights.npz` — the verbatim source artifact (BLAKE3-pinned);
//! - `intermediate/{meta.txt, trits.bin, vecs.f32}` — the lossless
//!   stdlib-emitter intermediate (T1a).
//!
//! Assembles the KPT1 archive via [`katgpt_core::ternary_circuit`] +
//! `kpt_archive::build_archive`, verifies it through the reader's own
//! fail-closed ladder, and writes `compiled/`:
//!
//! - `weights.cact` — the archive (`.cact` in the lane's vocabulary; the
//!   container is the house `kpt_archive` KPT1 v1 format);
//! - `manifest.json` — the positional name→index map derived from `meta`
//!   (the archive is nameless; the forward port resolves through this);
//! - `pins.json` — the BLAKE3 pins of BOTH originals (npz + intermediate)
//!   and of the `.cact` (bytes hash + `archive_id`).
//!
//! Deterministic: a re-run is byte-identical (the gate test
//! `ternary_circuit_transcode_gates` re-derives and byte-compares).
//!
//! Run: `cargo run -p katgpt-core --features ternary_circuit --example ternary_circuit_transcode`

use std::path::Path;

use katgpt_core::kpt_archive::{KptArchive, build_archive};
use katgpt_core::ternary_circuit::{ManifestRow, SOURCE_PIN, assemble_layers, parse_meta};

fn blake3_hex(bytes: &[u8]) -> String {
    format!("{}", blake3::hash(bytes))
}

struct Pins {
    npz_blake3: String,
    trits_blake3: String,
    vecs_blake3: String,
    meta_blake3: String,
    cact_blake3: String,
    cact_archive_id: String,
    cact_bytes: usize,
    layer_count: usize,
    trit_layers: usize,
    vec_layers: usize,
    zero_frac: f64,
}

impl Pins {
    fn to_json(&self) -> String {
        // Field order here is the file's contract; serde_json would do, but
        // a hand render keeps the pin file byte-stable across serde versions.
        format!(
            "{{\n  \"source_pin\": \"{source}\",\n  \"npz_blake3\": \"{npz}\",\n  \
             \"trits_blake3\": \"{trits}\",\n  \"vecs_blake3\": \"{vecs}\",\n  \
             \"meta_blake3\": \"{meta}\",\n  \"cact_blake3\": \"{cact}\",\n  \
             \"cact_archive_id\": \"{aid}\",\n  \"cact_bytes\": {bytes},\n  \
             \"layer_count\": {layers},\n  \"trit_layers\": {tl},\n  \
             \"vec_layers\": {vl},\n  \"zero_frac\": {zf}\n}}\n",
            source = SOURCE_PIN,
            npz = self.npz_blake3,
            trits = self.trits_blake3,
            vecs = self.vecs_blake3,
            meta = self.meta_blake3,
            cact = self.cact_blake3,
            aid = self.cact_archive_id,
            bytes = self.cact_bytes,
            layers = self.layer_count,
            tl = self.trit_layers,
            vl = self.vec_layers,
            zf = self.zero_frac
        )
    }
}

#[derive(serde::Serialize)]
struct ManifestDoc<'a> {
    format: &'static str,
    source_pin: &'static str,
    layer_count: usize,
    layers: &'a [ManifestRow],
}

fn main() {
    let fixture_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/ternary_circuit");
    let compiled_dir = fixture_dir.join("compiled");
    let npz = std::fs::read(fixture_dir.join("weights.npz"))
        .expect("weights.npz missing — run emit_weights_intermediate.py's sibling checkout first");
    let meta_txt = std::fs::read_to_string(fixture_dir.join("intermediate/meta.txt"))
        .expect("intermediate/meta.txt missing — run emit_weights_intermediate.py first");
    let trits = std::fs::read(fixture_dir.join("intermediate/trits.bin"))
        .expect("intermediate/trits.bin missing — run emit_weights_intermediate.py first");
    let vecs_raw = std::fs::read(fixture_dir.join("intermediate/vecs.f32"))
        .expect("intermediate/vecs.f32 missing — run emit_weights_intermediate.py first");
    if !vecs_raw.len().is_multiple_of(4) {
        panic!(
            "vecs.f32 length {} is not 4-aligned (f32 LE law)",
            vecs_raw.len()
        );
    }
    let vecs: Vec<f32> = vecs_raw
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();

    let meta = parse_meta(&meta_txt).expect("meta grammar refused the committed intermediate");
    let out = assemble_layers(&trits, &vecs, &meta)
        .expect("assembly laws refused the committed intermediate");
    let cact = build_archive(&out.layers.iter().collect::<Vec<_>>())
        .expect("KPT1 build refused the assembled layers");
    // Verify-before-publish: our own output must survive the reader's
    // fail-closed ladder (the swap_in_place discipline, fixture-grade).
    let verified = KptArchive::from_bytes(&cact).expect("built archive failed verification");

    let trit_count: usize = meta
        .trits
        .iter()
        .map(|e| e.dims.iter().product::<usize>())
        .sum();
    let zeros = {
        let mut byte_offset = 0usize;
        let mut zeros = 0usize;
        for e in &meta.trits {
            let elems: usize = e.dims.iter().product();
            for i in 0..elems {
                if katgpt_core::ternary_circuit::trit_at(&trits, byte_offset * 5 + i) == 0 {
                    zeros += 1;
                }
            }
            byte_offset += e.packed_len;
        }
        zeros as f64 / trit_count as f64
    };

    let pins = Pins {
        npz_blake3: blake3_hex(&npz),
        trits_blake3: blake3_hex(&trits),
        vecs_blake3: blake3_hex(&vecs_raw),
        meta_blake3: blake3_hex(meta_txt.as_bytes()),
        cact_blake3: blake3_hex(&cact),
        cact_archive_id: hex(&verified.archive_id()),
        cact_bytes: cact.len(),
        layer_count: out.layers.len(),
        trit_layers: meta.trits.len(),
        vec_layers: meta.vecs.len(),
        zero_frac: zeros,
    };

    std::fs::create_dir_all(&compiled_dir).expect("create compiled/");
    std::fs::write(compiled_dir.join("weights.cact"), &cact).expect("write compiled/weights.cact");
    let manifest_doc = ManifestDoc {
        format: "kpt1",
        source_pin: SOURCE_PIN,
        layer_count: out.layers.len(),
        layers: &out.manifest,
    };
    std::fs::write(
        compiled_dir.join("manifest.json"),
        serde_json::to_string_pretty(&manifest_doc).expect("manifest renders") + "\n",
    )
    .expect("write compiled/manifest.json");
    std::fs::write(compiled_dir.join("pins.json"), pins.to_json())
        .expect("write compiled/pins.json");

    println!(
        "transcoded: {} layers ({} trit + {} vec) -> {} bytes | archive_id {}",
        out.layers.len(),
        meta.trits.len(),
        meta.vecs.len(),
        cact.len(),
        pins.cact_archive_id
    );
    println!("cact_blake3 {}", pins.cact_blake3);
    println!("npz_blake3   {}", pins.npz_blake3);
    println!("zero_frac {:.5} (report-only)", pins.zero_frac);
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
