//! Forensic watermark recipe primitive — open generic math.
//!
//! Plan 293 / Research 268. Per-recipient BLAKE3-derived perturbation
//! recipes for **forensic attribution** of leaked content. Generic and
//! domain-agnostic: no game semantics, no chain, no NFT, no WASM vessel.
//! Recipient identity is just `&[u8; 32]` (a pubkey hash or any 32-byte
//! identifier).
//!
//! ## What it does
//!
//! 1. **Recipe derivation** (`recipe::derive_recipe`): deterministic
//!    per-recipient seed → 2×2 LoopWM-stable vertex perturbation matrix
//!    + vertex indices + mid-frequency DCT positions + topology mask +
//!    Tardos anti-collusion codeword. All in a few KB.
//! 2. **Apply marks** to assets:
//!    - [`vertex::apply_vertex_marks`] / [`vertex::apply_vertex_marks_simd`]
//!      — sub-`ε` 2D displacement in the tangent plane.
//!    - [`texture::apply_dct_marks`] — ±δ DCT coefficient flips in the
//!      mid-frequency band.
//!    - [`topology::apply_topology_marks`] — zero-area leaf triangles.
//! 3. **Recover** the codeword from a leaked asset and attribute it to a
//!    recipient via [`recover::attribute`]. Confidence is **sigmoid-gated**
//!    (per AGENTS.md rule — never softmax).
//!
//! ## When to use
//!
//! - You need post-leak attribution of digital assets (meshes, textures,
//!   any 3D content).
//! - You're serving per-recipient content and want to trace leaks back to
//!   source without shipping N unique copies (recipes are ~64 bytes).
//! - You want the AACS / Widevine / PlayReady class of forensic
//!   watermarking without their proprietary crypto stack.
//!
//! ## Security model — forensic, not preventive
//!
//! This primitive **detects** leaks after the fact. It does not prevent
//! them. An attacker with the recipe application logic can produce
//! unmarked copies. The value is **deterrence**: each distributed copy
//! carries a per-recipient signature that survives recompression and
//! identifies the leaker.
//!
//! For the closed-loop commercial deployment (WASM vessel + NFT
//! attribution registry + chain slashing), see riir-ai Plan 322. This
//! open module is the "adoption hook" — engineers can wire their own
//! recipient registry via [`recover::RecipientRegistry`].
//!
//! ## References
//!
//! - **Research 268** — design rationale:
//!   `katgpt-rs/.research/268_Forensic_Asset_Fingerprinting_LatCal_Recipe.md`
//! - **Plan 293** — implementation plan:
//!   `katgpt-rs/.plans/293_forensic_watermark_recipe_primitive.md`
//! - **arxiv 2606.18208** — LoopWM `A = diag(-exp(a))` spectral
//!   stability (transferred to bound `P_vertex` cumulative displacement).
//! - **Tardos 2008** — anti-collusion codebook (J. ACM 55(2)).
//! - **riir-ai Plan 322** — private integration (WASM vessel + NFT +
//!   slashing). Not in this crate.
//!
//! ## Feature gate
//!
//! Entire module is behind `feature = "forensic_watermark"`, **default
//! OFF**. Promotion to default-on happens only after the G1–G4 GOAT
//! gate (Plan 293 Phase 7 T7.2–T7.5) passes on real assets — that gate
//! run is a separate session.

pub mod recipe;
pub mod recover;
pub mod tardos;
pub mod texture;
pub mod topology;
pub mod vertex;

// Convenience re-exports at module root.
pub use recipe::{Recipe, RecipeConfig, construct_perturbation_matrix, derive_recipe, derive_seed};
pub use recover::{
    InMemoryRegistry, LeakedContent, RecoveryEvidence, RecoveryResult, RecipientRegistry,
    attribute, recover_codeword, recover_p_vertex, sigmoid,
};
pub use tardos::{TardosCodebook, extract_codeword_from_seed, recipient_index};
pub use texture::{
    Dct8x8Block, TextureMarkable, apply_dct_marks, dct8x8_forward, dct8x8_inverse,
    recover_dct_marks,
};
pub use topology::{
    TriangleMesh, apply_topology_marks, count_surviving_marks, recover_topology_marks,
};
pub use vertex::{VertexMarkable, apply_vertex_marks, apply_vertex_marks_simd};
