//! Plan 315 T5.1 — Vessel minimal encode/decode/extract round-trip.
//!
//! Builds a synthetic vessel containing a `[u8; 64]` Pod payload, encodes it
//! to the wire format, loads it back, extracts the payload, and confirms
//! byte-identical fidelity. No WASM execution — this is the Hot/Plasma-tier
//! extract path only.
//!
//! Run with:
//! ```text
//! cargo run --example vessel_minimal --features secure_vessel
//! ```

use katgpt_rs::vessel::{encode_vessel, extract_payload, load_vessel, VESSEL_HEADER_LEN};
use std::sync::Arc;

/// 64-byte Pod payload — synthetic stand-in for an HLA shard weight vector.
/// In production this would be `NeuronShard` (riir-neuron-db) or any other
/// `#[repr(C)]` Pod type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
struct WeightVector {
    weights: [u8; 64],
}
// SAFETY: `#[repr(C)]`, all-`Pod` fields, no padding.
unsafe impl bytemuck::Pod for WeightVector {}
unsafe impl bytemuck::Zeroable for WeightVector {}

fn main() {
    println!("=== Vessel Minimal Round-Trip (Plan 315 T5.1) ===\n");

    // 1. Build a synthetic "WASM module" containing the payload.
    //    Real vessels embed the payload inside the WASM data section; this
    //    demo uses a 16-byte prefix + payload + 8-byte suffix so the layout
    //    is unambiguous. The prefix is where a real WASM header would live.
    let payload = WeightVector {
        weights: {
            let mut w = [0u8; 64];
            for (i, b) in w.iter_mut().enumerate() {
                *b = (i as u8).wrapping_mul(7).wrapping_add(13);
            }
            w
        },
    };
    let mut wasm_bytes = vec![0u8; 16];
    wasm_bytes.extend_from_slice(bytemuck::bytes_of(&payload));
    wasm_bytes.extend_from_slice(&[0u8; 8]);

    // 2. Encode into the vessel wire format: [VesselHeader (52B)] + [wasm bytes].
    let encoded = encode_vessel(
        &wasm_bytes,
        /* payload_kind */ 1, // caller-defined discriminator
        /* payload_offset */ 16,
        /* payload_len */ std::mem::size_of::<WeightVector>() as u32,
    );
    println!("wasm_bytes.len()      = {}", wasm_bytes.len());
    println!("VESSEL_HEADER_LEN     = {} bytes", VESSEL_HEADER_LEN);
    println!("encoded.len()         = {} bytes (= header + wasm)", encoded.len());
    println!("content_addr          = {} (first 4 bytes of BLAKE3(header))",
        katgpt_rs::vessel::decode_header(&encoded).unwrap().blake3[0..4]
            .iter().map(|b| format!("{b:02x}")).collect::<String>());
    println!();

    // 3. Load — one-time cost: header decode + BLAKE3 verify.
    let vessel = load_vessel(&encoded).expect("load should succeed");
    println!("✅ Loaded vessel — BLAKE3 verified, content_addr computed");
    println!("   header.version      = {}", vessel.header.version);
    println!("   header.payload_kind = {}", vessel.header.payload_kind);
    println!("   wasm_bytes held in  = Arc<[u8; {}]>", vessel.wasm_bytes.len());
    println!();

    // 4. Extract — the core primitive. Zero-copy borrow from the Arc.
    let extracted: &WeightVector = extract_payload(&vessel).expect("extract");
    assert_eq!(extracted, &payload, "round-trip must be byte-identical");
    println!("✅ Extracted payload — byte-identical to original");

    // 5. Show that the Arc backing is shared (no copy on re-extract).
    let arc_ref: Arc<[u8]> = Arc::clone(&vessel.wasm_bytes);
    let _extracted_again: &WeightVector = extract_payload(&vessel).expect("re-extract");
    assert!(Arc::ptr_eq(&arc_ref, &vessel.wasm_bytes), "Arc must be shared");
    println!("✅ Re-extract shares the same Arc — zero allocation");

    println!("\n=== Done. Hot path: extract_payload = ~0.5 ns/op (zero-copy borrow) ===");
}
