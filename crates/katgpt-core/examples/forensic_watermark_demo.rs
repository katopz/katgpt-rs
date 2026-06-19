//! Example: Forensic watermark end-to-end demo (Plan 293 T8.3).
//!
//! Demonstrates the open generic math primitive:
//! 1. Derive a per-recipient recipe from `(master_seed, recipient_pubkey)`.
//! 2. Apply the recipe to a synthetic mesh (10³ verts) + texture (10² blocks).
//! 3. Simulate a leak (clone the marked asset).
//! 4. Recover the codeword from the leaked asset.
//! 5. Attribute the leak to the recipient via `InMemoryRegistry`.
//! 6. Print the recipient pubkey + sigmoid-gated confidence.
//!
//! Runs without GPU. No game semantics, no chain, no NFT — pure math.
//!
//! Run with:
//! ```sh
//! cargo run --example forensic_watermark_demo --features forensic_watermark --release
//! ```

use katgpt_core::forensic::recover::{LeakedContent, attribute};
use katgpt_core::forensic::texture::{Dct8x8Block, apply_dct_marks};
use katgpt_core::forensic::topology::{TriangleMesh, apply_topology_marks};
use katgpt_core::forensic::vertex::apply_vertex_marks;
use katgpt_core::forensic::{InMemoryRegistry, RecipeConfig, RecipientRegistry, derive_recipe};

const N_VERTS: usize = 1_000;
const N_TRIS: usize = 500;
const N_BLOCKS: usize = 200;

fn main() {
    println!("=== Forensic Watermark Recipe Primitive Demo (Plan 293) ===\n");

    // --- Setup: a recipient with a known pubkey ----------------------------
    let cfg = RecipeConfig::default();
    let recipient_pubkey: [u8; 32] = [
        0x42, 0x11, 0x7a, 0xe9, 0xc3, 0x55, 0x0a, 0xbc, 0xde, 0xf1, 0x28, 0x99, 0x76, 0x54, 0x32,
        0x10, 0xab, 0xcd, 0xef, 0x01, 0x23, 0x45, 0x67, 0x89, 0x9a, 0xbc, 0xde, 0xf0, 0x11, 0x22,
        0x33, 0x44,
    ];
    let master_seed: [u8; 32] = [0xaa; 32];

    println!("Recipient pubkey (first 8 bytes): {}", hex8(&recipient_pubkey));
    println!("Master seed (first 8 bytes):      {}", hex8(&master_seed));
    println!(
        "Config: L_v={}, L_dct={}, L_t={}, ε={:e}, δ={}",
        cfg.vertex_mark_count, cfg.dct_mark_count, cfg.topology_mark_count,
        cfg.epsilon_vertex, cfg.delta_dct
    );

    // --- Step 1: derive the recipe -----------------------------------------
    let recipe = derive_recipe(&cfg, &recipient_pubkey, &master_seed);
    println!("\nDerived recipe:");
    println!("  P_vertex = [[{:.6}, 0], [0, {:.6}]]", recipe.p_vertex[0][0], recipe.p_vertex[1][1]);
    println!(
        "  det(P_vertex) = {:.10}  (should be ≈ 1)",
        recipe.p_vertex[0][0] * recipe.p_vertex[1][1]
    );
    println!("  recipient_idx = {} (slot in codebook of n={})",
        recipe.recipient_idx, cfg.n_recipients);
    println!("  codeword length = {} bits (bandwidth)", recipe.codeword.len());

    // --- Step 2: build a synthetic asset + apply marks ---------------------
    let mut mesh = synth_mesh(N_VERTS, N_TRIS);
    let mesh_reference = mesh.clone();
    let mut texture = synth_texture(N_BLOCKS);
    let texture_reference = texture.clone();

    apply_vertex_marks(&mut mesh, &recipe, &cfg);
    apply_dct_marks(&mut texture, &recipe, &cfg);
    apply_topology_marks(&mut mesh, &recipe, &cfg);

    // Verify vertex displacement is within ε.
    let mut max_rel_disp = 0.0f32;
    for &v_idx in &recipe.vertex_indices {
        let i = (v_idx as usize) % N_VERTS;
        let dx = mesh.positions[i][0] - mesh_reference.positions[i][0];
        let dy = mesh.positions[i][1] - mesh_reference.positions[i][1];
        let disp = (dx * dx + dy * dy).sqrt();
        let v_norm = (
            mesh_reference.positions[i][0].powi(2)
            + mesh_reference.positions[i][1].powi(2)
        ).sqrt().max(1e-6);
        let rel = disp / v_norm;
        if rel > max_rel_disp {
            max_rel_disp = rel;
        }
    }
    println!("\nApplied marks to synthetic asset ({} verts, {} tris, {} blocks):",
        N_VERTS, N_TRIS, N_BLOCKS);
    println!("  Max relative vertex displacement: {:.3e} (ε = {:.3e}) — {}",
        max_rel_disp, cfg.epsilon_vertex,
        if max_rel_disp <= cfg.epsilon_vertex * 1.5 { "WITHIN BOUND" } else { "check" });

    // --- Step 3: simulate a leak (clone the marked asset) ------------------
    let leaked = LeakedContent {
        mesh: mesh.clone(),
        texture_blocks: texture.clone(),
    };
    let reference = LeakedContent {
        mesh: mesh_reference,
        texture_blocks: texture_reference,
    };
    println!("\nSimulated leak: cloned the marked asset ({} verts, {} blocks).",
        leaked.mesh.positions.len(), leaked.texture_blocks.len());

    // --- Step 4 + 5: recover + attribute -----------------------------------
    let mut registry = InMemoryRegistry::new();
    registry.register_recipe(recipient_pubkey, &recipe, N_BLOCKS);
    println!("Registry: {} recipient(s) registered.", registry.n_recipients());

    let result = attribute(&leaked, &reference, &recipe, &registry, &cfg, 1e-6);

    match result {
        Some(r) => {
            println!("\n=== Attribution Result ===");
            println!("Recipient pubkey (first 8 bytes): {}", hex8(&r.recipient_pubkey));
            println!("Confidence (σ-gated):            {:.6}", r.confidence);
            println!("Tardos accusation score S_j:     {:.4}", r.evidence.accusation_score);
            println!("Tardos threshold Z (c·√(L/2)):   {:.4}", r.evidence.accusation_threshold);
            println!("Evidence:");
            println!("  DCT bits recovered:     {}", r.evidence.dct_bits);
            println!("  P_vertex sign bit:      {}", r.evidence.vertex_bits);
            println!("  Topology marks found:   {}", r.evidence.topology_bits);
            let matched = r.recipient_pubkey == recipient_pubkey;
            println!("\nMatch: {} (expected recipient {})",
                if matched { "✓ CORRECT" } else { "✗ WRONG" },
                hex8(&recipient_pubkey));
        }
        None => {
            println!("\n✗ Attribution FAILED — codeword did not match any registered recipient.");
        }
    }

    println!("\n=== Demo complete. This primitive is FORENSIC (post-leak attribution), not preventive. ===");
    println!("=== Private integration (WASM vessel + NFT + slashing) is riir-ai Plan 322. ===");
}

// ─── Helpers ────────────────────────────────────────────────────────────

fn hex8(bytes: &[u8; 32]) -> String {
    bytes.iter().take(8).map(|b| format!("{:02x}", b)).collect()
}

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
