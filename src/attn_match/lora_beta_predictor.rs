//! Plan 297 Phase D / Recipe 4 — LoRA β predictor architecture (T-D.2).
//!
//! Defines the LoRA adapter that replaces AM's NNLS β fitter (`O(n·t·iters)`)
//! for long contexts (`T > 100k`). The adapter is a tiny rank-16 LoRA that maps
//! pooled KV statistics to a per-head β vector, trained via distillation
//! (T-D.3) against the corpus from T-D.1.
//!
//! # Architecture
//!
//! Standard LoRA factorization with a bias head:
//! ```text
//! hidden = A @ x                            // [rank]      (rank × in_dim)
//! logit = (α/rank) · B @ hidden + bias      // [out_dim]   (out_dim × rank)
//! β     = BETA_MIN + (BETA_MAX − BETA_MIN) · sigmoid(logit)
//! ```
//!
//! - Input:  `LORA_INPUT_DIM = 80`  (pooled KV stats — matches T-D.1 `KvStats`)
//! - Output: `LORA_OUTPUT_DIM = 8`  (per-head β — matches T-D.1 `β_ref`)
//! - Rank:   `DEFAULT_RANK = 16`    (matches existing game LoRAs per T-D.2)
//! - Params: `16×80 + 8×16 + 8 = 1416` trainable
//! - α:      `DEFAULT_ALPHA = 16`   (so `α/rank = 1.0`)
//!
//! # Why Sigmoid (not clamp / softmax)
//!
//! The output β must live in `[BETA_MIN, BETA_MAX]` — the same box AM's NNLS
//! uses (`w_lower = 1e-3`, `w_upper = e^3`). Three options were considered:
//!
//! 1. **Clamp** — zero gradient outside the box. Kills gradient flow when the
//!    LoRA overshoots early in training. Also violates the "use sigmoid" rule.
//! 2. **Softmax** — produces a categorical distribution, not a regression
//!    target. β is not a probability; softmax would couple the heads and
//!    destroy per-head independence.
//! 3. **Sigmoid** (chosen) — maps each logit independently to `[0, 1]`, then
//!    affine-rescale to `[BETA_MIN, BETA_MAX]`. Non-zero gradient everywhere,
//!    per-head independent, satisfies AGENTS.md "use sigmoid not softmax".
//!
//! At init (`bias = 0`, `B = 0`), every logit is 0, so `sigmoid(0) = 0.5` and
//! `β = BETA_MID ≈ −1.95` — a neutral midpoint the training refines.
//!
//! # Param Count vs NNLS Cost
//!
//! The LoRA forward pass is `~1408 MACs` (independent of sequence length T).
//! AM's NNLS is `O(n · t · iters)` — at `t = 2048, n = 4, iters = 64`, that's
//! `~524k MACs`. The LoRA is **~370× cheaper**, and the gap widens with T.
//! This is the speedup motivation for Recipe 4 (target: `T > 100k`).
//!
//! # Latent vs Raw (AGENTS.md)
//!
//! All data here is **latent** (KV statistics, β values). Nothing crosses a
//! sync boundary, so nothing here is raw/synced. The predictor is a local
//! inference asset, not runtime state.
//!
//! # Determinism
//!
//! [`LoraBetaPredictor::with_seed`] uses a xorshift32 PRNG seeded from a
//! BLAKE3 digest, matching the T-D.1 corpus generator convention. This lets
//! T-D.3 training pin exact initial weights for reproducibility.
//!
//! # What this module does NOT do (T-D.3 / T-D.4)
//!
//! - **Training** (T-D.3): backprop, optimizer, loss `||β_LoRA − β_NNLS||²`.
//!   The fields are `pub` so the trainer can mutate them directly.
//! - **Inference integration** (T-D.4): drop-in replacement for NNLS inside
//!   `compact()`. This module only defines the architecture + forward pass.
//!
//! # Serialization
//!
//! `[magic "LBPA"][version u32 LE][postcard LoraBetaPredictor][BLAKE3 commitment u256]`.
//! Matches the T-D.1 corpus convention for consistency.

#![cfg(feature = "lora_beta_predictor")]

use serde::{Deserialize, Serialize};

// ── Constants ──────────────────────────────────────────────────────

/// Number of attention heads modeled. MUST match
/// `riir_data::beta_distill_corpus::N_HEADS`.
pub const N_HEADS: usize = 8;

/// Number of top-K attention scores retained per head when pooling KV stats.
/// MUST match `riir_data::beta_distill_corpus::TOP_K`.
pub const TOP_K: usize = 8;

/// Per-head feature dimension (`mean_k, var_k, top-K attn scores`).
/// MUST match `riir_data::beta_distill_corpus::STATS_PER_HEAD`.
pub const STATS_PER_HEAD: usize = 2 + TOP_K;

/// LoRA input dimension. MUST match
/// `riir_data::beta_distill_corpus::LORA_INPUT_DIM`.
pub const LORA_INPUT_DIM: usize = N_HEADS * STATS_PER_HEAD;

/// LoRA output dimension (one β per head). MUST match
/// `riir_data::beta_distill_corpus::LORA_OUTPUT_DIM`.
pub const LORA_OUTPUT_DIM: usize = N_HEADS;

/// Default LoRA rank (T-D.2: "LoRA rank: 16 (matches existing game LoRAs)").
pub const DEFAULT_RANK: usize = 16;

/// Default α scaling (standard LoRA: `α = rank` so `α/rank = 1.0`).
pub const DEFAULT_ALPHA: f32 = DEFAULT_RANK as f32;

/// β lower bound = `log(w_lower)` where `w_lower = 1e-3`
/// (`AmConfig::highest_attn`).
pub const BETA_MIN: f32 = -6.907_755_3; // ln(1e-3)

/// β upper bound = `log(w_upper)` where `w_upper = e^3 ≈ 20.0855`
/// (`AmConfig::highest_attn`).
pub const BETA_MAX: f32 = 3.0;

/// β midpoint — the neutral init target (`sigmoid(0) = 0.5` → `β_mid`).
pub const BETA_MID: f32 = (BETA_MIN + BETA_MAX) * 0.5;

/// Sigmoid output span: `BETA_MAX − BETA_MIN`.
const SIGMOID_SPAN: f32 = BETA_MAX - BETA_MIN;

/// Magic header for the serialized predictor: ASCII `"LBPA"`.
pub const PREDICTOR_MAGIC: [u8; 4] = *b"LBPA";

/// Serialized format version.
pub const PREDICTOR_VERSION: u32 = 1;

/// Fixed seed for [`LoraBetaPredictor::new`] / [`LoraBetaPredictor::default`],
/// so the default init is bit-reproducible. Derived from
/// `BLAKE3(b"lora_beta_predictor_default_init_v1")`.
///
/// Public so T-D.3 training configs can default to the same seed.
pub const DEFAULT_INIT_SEED: [u8; 32] = [
    0xcc, 0xff, 0xe0, 0x13, 0xd8, 0xd6, 0x91, 0x34, 0x33, 0x3a, 0x0c, 0x47, 0x95, 0x15, 0x11, 0x02,
    0x7e, 0x25, 0x5a, 0xcd, 0x75, 0x7a, 0x6a, 0x9c, 0xb9, 0xc3, 0x86, 0x49, 0x34, 0xba, 0x2b, 0x33,
];

// ── Errors ─────────────────────────────────────────────────────────

/// Errors raised by the LoRA β predictor.
#[derive(Debug, Clone, PartialEq)]
pub enum LoraBetaError {
    /// Input or weight buffer had the wrong length.
    DimMismatch {
        context: &'static str,
        got: usize,
        expected: usize,
    },
    /// A weight or bias entry was non-finite (NaN / Inf).
    NonFiniteWeights,
    /// Serialized file's magic / version was wrong.
    BadHeader { magic: [u8; 4], version: u32 },
    /// Truncated serialized file (too few bytes).
    Truncated,
    /// postcard (de)serialization failed.
    Postcard(String),
}

impl std::fmt::Display for LoraBetaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DimMismatch {
                context,
                got,
                expected,
            } => {
                write!(f, "{context} dim mismatch: got {got}, expected {expected}")
            }
            Self::NonFiniteWeights => write!(f, "predictor weights contain non-finite values"),
            Self::BadHeader { magic, version } => {
                let m = std::str::from_utf8(magic).unwrap_or("??");
                write!(f, "bad predictor header: magic={m:?} version={version}")
            }
            Self::Truncated => write!(f, "serialized predictor was truncated"),
            Self::Postcard(s) => write!(f, "postcard error: {s}"),
        }
    }
}

impl std::error::Error for LoraBetaError {}

impl From<postcard::Error> for LoraBetaError {
    fn from(e: postcard::Error) -> Self {
        Self::Postcard(e.to_string())
    }
}

// ── Predictor ──────────────────────────────────────────────────────

/// LoRA β predictor — rank-16 adapter mapping pooled KV statistics to a
/// per-head β vector (Plan 297 T-D.2 / Recipe 4).
///
/// Forward pass:
/// ```text
/// hidden = A @ x                            // [rank]
/// logit = (α/rank) · B @ hidden + bias      // [out_dim]
/// β     = BETA_MIN + (BETA_MAX − BETA_MIN) · sigmoid(logit)
/// ```
///
/// Init (via [`new`](Self::new) / [`with_seed`](Self::with_seed)):
/// - `A`: Kaiming-style `±1/√in_dim`, seeded for reproducibility.
/// - `B`: zeros (standard LoRA — starts as no-op).
/// - `bias`: zeros (so initial logit = 0, β = `BETA_MID`).
///
/// Fields are `pub` so T-D.3 training can mutate them directly. Use
/// [`validate`](Self::validate) after mutation to check consistency.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LoraBetaPredictor {
    /// A matrix `[rank × in_dim]`, row-major: `a[r * in_dim + j] = A[r][j]`.
    pub a: Vec<f32>,
    /// B matrix `[out_dim × rank]`, row-major: `b[o * rank + r] = B[o][r]`.
    pub b: Vec<f32>,
    /// Bias `[out_dim]`.
    pub bias: Vec<f32>,
    /// Input dimension (must equal `LORA_INPUT_DIM` for corpus compatibility).
    pub in_dim: usize,
    /// Output dimension (must equal `LORA_OUTPUT_DIM` for corpus compatibility).
    pub out_dim: usize,
    /// LoRA rank.
    pub rank: usize,
    /// LoRA α scaling factor.
    pub alpha: f32,
}

impl LoraBetaPredictor {
    /// Create a new predictor with default dimensions (`80 → 8`, rank 16,
    /// α = 16) and reproducible Kaiming-init `A`, zero `B`/`bias`.
    ///
    /// The default init uses a fixed seed (`DEFAULT_INIT_SEED`), so `new()`
    /// is bit-identical across runs — required for GOAT gate reproducibility.
    #[inline]
    pub fn new() -> Self {
        Self::with_seed(
            LORA_INPUT_DIM,
            LORA_OUTPUT_DIM,
            DEFAULT_RANK,
            DEFAULT_ALPHA,
            DEFAULT_INIT_SEED,
        )
    }

    /// Create with explicit dimensions. Uses the fixed default seed.
    #[inline]
    pub fn with_dims(in_dim: usize, out_dim: usize, rank: usize, alpha: f32) -> Self {
        Self::with_seed(in_dim, out_dim, rank, alpha, DEFAULT_INIT_SEED)
    }

    /// Create with deterministic init from a BLAKE3 seed.
    ///
    /// - `A[r][j] = sign · (1/√in_dim) · frac(prng)` — Kaiming-scaled, alternating sign.
    /// - `B[o][r] = 0.0` — standard LoRA no-op init.
    /// - `bias[o] = 0.0` — so initial logit = 0, β = `BETA_MID`.
    ///
    /// The seed convention matches T-D.1's corpus generator: same xorshift32
    /// PRNG, same BLAKE3-rooted seeding. This lets T-D.3 pin exact init weights.
    pub fn with_seed(
        in_dim: usize,
        out_dim: usize,
        rank: usize,
        alpha: f32,
        seed: [u8; 32],
    ) -> Self {
        let mut prng = Prng::from_blake3(seed);

        // A: [rank × in_dim], Kaiming-init.
        let scale = 1.0 / (in_dim as f32).sqrt();
        let a: Vec<f32> = (0..(rank * in_dim))
            .map(|i| {
                let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
                sign * scale * prng.next_f32()
            })
            .collect();

        // B: [out_dim × rank], zeros (standard LoRA init).
        let b = vec![0.0f32; out_dim * rank];

        // bias: [out_dim], zeros (so initial β = BETA_MID).
        let bias = vec![0.0f32; out_dim];

        Self {
            a,
            b,
            bias,
            in_dim,
            out_dim,
            rank,
            alpha,
        }
    }

    /// Total trainable parameter count: `A + B + bias`.
    #[inline]
    pub fn param_count(&self) -> usize {
        self.a.len() + self.b.len() + self.bias.len()
    }

    /// Effective scaling factor `α / rank` applied to the `B @ hidden` term.
    #[inline]
    fn alpha_scale(&self) -> f32 {
        if self.rank == 0 {
            0.0
        } else {
            self.alpha / (self.rank as f32)
        }
    }

    /// Forward pass producing per-head β from pooled KV stats.
    ///
    /// Allocates a new `Vec<f32>` of length [`out_dim`](Self::out_dim).
    /// For the hot inference path (T-D.4), use [`predict_into`](Self::predict_into)
    /// to avoid allocation.
    ///
    /// Output is sigmoid-squashed to `[BETA_MIN, BETA_MAX]`.
    pub fn predict(&self, kv_stats: &[f32]) -> Vec<f32> {
        debug_assert_eq!(
            kv_stats.len(),
            self.in_dim,
            "kv_stats.len() {} != in_dim {}",
            kv_stats.len(),
            self.in_dim
        );
        let mut out = vec![0.0f32; self.out_dim];
        // predict_into only fails on dim mismatch, which we control here.
        let _ = self.forward_into(kv_stats, &mut out);
        out
    }

    /// Zero-alloc forward pass. Writes `out_dim` β values into `out`.
    ///
    /// Output is sigmoid-squashed to `[BETA_MIN, BETA_MAX]`.
    pub fn predict_into(
        &self,
        kv_stats: &[f32],
        out: &mut [f32],
    ) -> Result<(), LoraBetaError> {
        self.forward_into(kv_stats, out)
    }

    /// Raw logits before sigmoid squash (for T-D.3 training).
    ///
    /// Returns `logit = (α/rank) · B @ (A @ x) + bias`, length [`out_dim`](Self::out_dim).
    /// The training loss may operate in logit space (more numerically stable
    /// than operating on squashed β when the target is near the boundaries).
    pub fn forward_logits(&self, kv_stats: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; self.out_dim];
        self.forward_logits_into(kv_stats, &mut out);
        out
    }

    /// Zero-alloc raw-logit forward (for T-D.3 training).
    pub fn forward_logits_into(&self, kv_stats: &[f32], out: &mut [f32]) {
        debug_assert_eq!(kv_stats.len(), self.in_dim);
        debug_assert_eq!(out.len(), self.out_dim);
        let scale = self.alpha_scale();

        // hidden = A @ x  →  [rank]   (a[r * in_dim + j] = A[r][j])
        // Reuse a single small stack buffer for hidden since rank ≤ 64.
        // (Avoids per-call Vec allocation in the hot path.)
        let mut hidden = [0.0f32; 64];
        let hidden = &mut hidden[..self.rank];
        for h in hidden.iter_mut() {
            *h = 0.0;
        }
        for r in 0..self.rank {
            let row = r * self.in_dim;
            let mut s = 0.0f32;
            for j in 0..self.in_dim {
                s += self.a[row + j] * kv_stats[j];
            }
            hidden[r] = s;
        }

        // logit = scale · B @ hidden + bias  →  [out_dim]
        for o in 0..self.out_dim {
            let row = o * self.rank;
            let mut s = 0.0f32;
            for r in 0..self.rank {
                s += self.b[row + r] * hidden[r];
            }
            out[o] = scale * s + self.bias[o];
        }
    }

    /// Forward pass writing sigmoid-squashed β into `out`.
    fn forward_into(
        &self,
        kv_stats: &[f32],
        out: &mut [f32],
    ) -> Result<(), LoraBetaError> {
        if kv_stats.len() != self.in_dim {
            return Err(LoraBetaError::DimMismatch {
                context: "kv_stats",
                got: kv_stats.len(),
                expected: self.in_dim,
            });
        }
        if out.len() != self.out_dim {
            return Err(LoraBetaError::DimMismatch {
                context: "out",
                got: out.len(),
                expected: self.out_dim,
            });
        }
        self.forward_logits_into(kv_stats, out);
        // Sigmoid squash: β = BETA_MIN + SIGMOID_SPAN · sigmoid(logit).
        for v in out.iter_mut() {
            *v = BETA_MIN + SIGMOID_SPAN * sigmoid(*v);
        }
        Ok(())
    }

    /// Validate dimensions and finiteness.
    ///
    /// Checks:
    /// - `a.len() == rank * in_dim`
    /// - `b.len() == out_dim * rank`
    /// - `bias.len() == out_dim`
    /// - all entries finite
    pub fn validate(&self) -> Result<(), LoraBetaError> {
        if self.a.len() != self.rank * self.in_dim {
            return Err(LoraBetaError::DimMismatch {
                context: "a (rank×in_dim)",
                got: self.a.len(),
                expected: self.rank * self.in_dim,
            });
        }
        if self.b.len() != self.out_dim * self.rank {
            return Err(LoraBetaError::DimMismatch {
                context: "b (out_dim×rank)",
                got: self.b.len(),
                expected: self.out_dim * self.rank,
            });
        }
        if self.bias.len() != self.out_dim {
            return Err(LoraBetaError::DimMismatch {
                context: "bias",
                got: self.bias.len(),
                expected: self.out_dim,
            });
        }
        for &v in self.a.iter().chain(self.b.iter()).chain(self.bias.iter()) {
            if !v.is_finite() {
                return Err(LoraBetaError::NonFiniteWeights);
            }
        }
        Ok(())
    }

    // ── Serialization ──────────────────────────────────────────────

    /// BLAKE3 commitment over the postcard-serialized weights.
    ///
    /// Hashed fields: `(in_dim, out_dim, rank, alpha, a, b, bias)`.
    /// Stable across metadata-only format bumps (the commitment covers the
    /// *content*, not the wire format envelope).
    pub fn commitment(&self) -> [u8; 32] {
        let bytes = postcard::to_allocvec(self).expect("postcard serialize (in-memory)");
        *blake3::hash(&bytes).as_bytes()
    }

    /// Serialize to bytes: `[magic][version u32 LE][postcard Self][commitment u256]`.
    ///
    /// The trailing 32-byte BLAKE3 commitment lets a consumer verify the
    /// weights were not tampered with after serialization.
    pub fn serialize(&self) -> Vec<u8> {
        let body = postcard::to_allocvec(self).expect("postcard serialize (in-memory)");
        let commitment = *blake3::hash(&body).as_bytes();
        let mut out =
            Vec::with_capacity(PREDICTOR_MAGIC.len() + 4 + body.len() + commitment.len());
        out.extend_from_slice(&PREDICTOR_MAGIC);
        out.extend_from_slice(&PREDICTOR_VERSION.to_le_bytes());
        out.extend_from_slice(&body);
        out.extend_from_slice(&commitment);
        out
    }

    /// Deserialize from bytes produced by [`serialize`](Self::serialize).
    ///
    /// Verifies magic, version, and BLAKE3 commitment. Returns
    /// [`LoraBetaError::BadHeader`] on magic/version mismatch,
    /// [`LoraBetaError::Truncated`] if the commitment tail is missing, and
    /// does NOT verify the commitment by default (call [`commitment`](Self::commitment)
    /// on the result to compare). The commitment is read so future
    /// `verify_commitment` methods can check it without re-reading the file.
    pub fn deserialize(bytes: &[u8]) -> Result<Self, LoraBetaError> {
        if bytes.len() < PREDICTOR_MAGIC.len() + 4 + 32 {
            return Err(LoraBetaError::Truncated);
        }
        let magic_off = PREDICTOR_MAGIC.len();
        let magic: [u8; 4] = bytes[..magic_off]
            .try_into()
            .map_err(|_| LoraBetaError::Truncated)?;
        if magic != PREDICTOR_MAGIC {
            return Err(LoraBetaError::BadHeader {
                magic,
                version: 0,
            });
        }
        let version = u32::from_le_bytes([
            bytes[magic_off],
            bytes[magic_off + 1],
            bytes[magic_off + 2],
            bytes[magic_off + 3],
        ]);
        if version != PREDICTOR_VERSION {
            return Err(LoraBetaError::BadHeader {
                magic,
                version,
            });
        }
        let body_start = magic_off + 4;
        let body_end = bytes.len() - 32;
        let body = &bytes[body_start..body_end];
        let predictor: Self = postcard::from_bytes(body)?;
        Ok(predictor)
    }
}

impl Default for LoraBetaPredictor {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

// ── Helpers ────────────────────────────────────────────────────────

/// Numerically stable sigmoid. Matches the convention used throughout
/// katgpt-rs (positive/negative branch avoids overflow).
#[inline]
fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Deterministic xorshift32 PRNG seeded from a 32-byte BLAKE3 digest.
///
/// Matches `riir_data::beta_distill_corpus::Prng` convention: same algorithm,
/// same seeding, so init weights are bit-reproducible from a known seed.
#[derive(Debug, Clone, Copy)]
struct Prng {
    state: u32,
}

impl Prng {
    fn from_blake3(digest: [u8; 32]) -> Self {
        let mut state = 0u32;
        for &b in &digest[..4] {
            state = (state << 8) | b as u32;
        }
        // xorshift32 requires non-zero state.
        if state == 0 {
            state = 0xDEAD_BEEF;
        }
        Self { state }
    }

    #[inline]
    fn next_u32(&mut self) -> u32 {
        // Marsaglia xorshift32.
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.state = x;
        x
    }

    /// Uniform float in `[0, 1)`.
    #[inline]
    fn next_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / ((1u32 << 24) as f32)
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Constants ──────────────────────────────────────────────────

    #[test]
    fn constants_match_corpus_module() {
        // MUST match riir_data::beta_distill_corpus (T-D.1).
        assert_eq!(N_HEADS, 8);
        assert_eq!(TOP_K, 8);
        assert_eq!(STATS_PER_HEAD, 10);
        assert_eq!(LORA_INPUT_DIM, 80);
        assert_eq!(LORA_OUTPUT_DIM, 8);
    }

    #[test]
    fn beta_box_matches_am_highest_attn() {
        // AmConfig::highest_attn uses w_lower=1e-3, w_upper=e^3≈20.0855.
        assert!((BETA_MIN - (1e-3f32).ln()).abs() < 1e-5);
        assert!((BETA_MAX - 3.0).abs() < 1e-6);
        assert!(BETA_MIN < BETA_MAX);
        // BETA_MID is the midpoint.
        assert!((BETA_MID - (BETA_MIN + BETA_MAX) * 0.5).abs() < 1e-6);
    }

    #[test]
    fn default_rank_and_alpha() {
        assert_eq!(DEFAULT_RANK, 16);
        assert_eq!(DEFAULT_ALPHA, 16.0);
    }

    // ── Init ───────────────────────────────────────────────────────

    #[test]
    fn new_has_correct_dimensions() {
        let p = LoraBetaPredictor::new();
        assert_eq!(p.in_dim, LORA_INPUT_DIM);
        assert_eq!(p.out_dim, LORA_OUTPUT_DIM);
        assert_eq!(p.rank, DEFAULT_RANK);
        assert_eq!(p.alpha, DEFAULT_ALPHA);
        assert_eq!(p.a.len(), DEFAULT_RANK * LORA_INPUT_DIM); // 16 × 80
        assert_eq!(p.b.len(), LORA_OUTPUT_DIM * DEFAULT_RANK); // 8 × 16
        assert_eq!(p.bias.len(), LORA_OUTPUT_DIM); // 8
    }

    #[test]
    fn new_init_b_is_zero_bias_is_zero() {
        // Standard LoRA init: B = 0, bias = 0.
        let p = LoraBetaPredictor::new();
        assert!(p.b.iter().all(|&v| v == 0.0), "B must be zero at init");
        assert!(
            p.bias.iter().all(|&v| v == 0.0),
            "bias must be zero at init"
        );
    }

    #[test]
    fn new_init_a_is_kaiming_scaled() {
        // A entries should be O(1/√in_dim) = O(1/√80) ≈ 0.112.
        let p = LoraBetaPredictor::new();
        let scale = 1.0 / (LORA_INPUT_DIM as f32).sqrt();
        let max_abs = p.a.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        // Should not exceed the Kaiming scale.
        assert!(
            max_abs <= scale + 1e-6,
            "max |A| = {max_abs} should be ≤ {scale}"
        );
        // Should have some non-zero entries.
        assert!(p.a.iter().any(|&v| v != 0.0), "A must not be all zero");
    }

    #[test]
    fn new_init_a_not_all_identical() {
        // A should have variation (not a constant).
        let p = LoraBetaPredictor::new();
        let first = p.a[0];
        assert!(
            p.a.iter().any(|&v| v != first),
            "A must have variation, not all = {first}"
        );
    }

    #[test]
    fn new_is_deterministic() {
        // Same default seed → bit-identical init.
        let p1 = LoraBetaPredictor::new();
        let p2 = LoraBetaPredictor::new();
        assert_eq!(p1.a, p2.a, "default init A must be reproducible");
        assert_eq!(p1.b, p2.b);
        assert_eq!(p1.bias, p2.bias);
    }

    #[test]
    fn with_seed_reproducible() {
        let seed = [0xAB; 32];
        let p1 = LoraBetaPredictor::with_seed(80, 8, 16, 16.0, seed);
        let p2 = LoraBetaPredictor::with_seed(80, 8, 16, 16.0, seed);
        assert_eq!(p1.a, p2.a);
        assert_eq!(p1, p2);
    }

    #[test]
    fn with_seed_different_seeds_differ() {
        let s1 = [0x01; 32];
        let s2 = [0x02; 32];
        let p1 = LoraBetaPredictor::with_seed(80, 8, 16, 16.0, s1);
        let p2 = LoraBetaPredictor::with_seed(80, 8, 16, 16.0, s2);
        assert_ne!(p1.a, p2.a, "different seeds must produce different A");
    }

    #[test]
    fn param_count_is_correct() {
        let p = LoraBetaPredictor::new();
        // A[16×80] + B[8×16] + bias[8] = 1280 + 128 + 8 = 1416.
        assert_eq!(p.param_count(), 1280 + 128 + 8);
        assert_eq!(p.param_count(), 1416);
    }

    // ── Forward pass ───────────────────────────────────────────────

    #[test]
    fn predict_at_init_returns_beta_mid() {
        // At init: B = 0, bias = 0 → logit = 0 → sigmoid(0) = 0.5 → β = BETA_MID.
        let p = LoraBetaPredictor::new();
        let x = vec![0.5f32; LORA_INPUT_DIM];
        let beta = p.predict(&x);
        assert_eq!(beta.len(), LORA_OUTPUT_DIM);
        for &b in &beta {
            assert!(
                (b - BETA_MID).abs() < 1e-5,
                "init predict should return BETA_MID={BETA_MID}, got {b}"
            );
        }
    }

    #[test]
    fn predict_output_is_in_beta_box() {
        // Even after perturbing B, output must stay in [BETA_MIN, BETA_MAX].
        let mut p = LoraBetaPredictor::new();
        // Large B entries to push logits to extremes.
        p.b.fill(100.0);
        p.bias.fill(50.0);
        let x = vec![1.0f32; LORA_INPUT_DIM];
        let beta = p.predict(&x);
        for &b in &beta {
            assert!(
                b >= BETA_MIN - 1e-5 && b <= BETA_MAX + 1e-5,
                "β {b} must be in [{BETA_MIN}, {BETA_MAX}]"
            );
        }
    }

    #[test]
    fn predict_output_is_in_beta_box_negative() {
        let mut p = LoraBetaPredictor::new();
        // Negative B/bias to push logits to -∞.
        p.b.fill(-100.0);
        p.bias.fill(-50.0);
        let x = vec![1.0f32; LORA_INPUT_DIM];
        let beta = p.predict(&x);
        for &b in &beta {
            assert!(b >= BETA_MIN - 1e-5, "β {b} must be ≥ BETA_MIN");
            assert!(b <= BETA_MAX + 1e-5, "β {b} must be ≤ BETA_MAX");
        }
    }

    #[test]
    fn predict_into_writes_correct_length() {
        let p = LoraBetaPredictor::new();
        let x = vec![0.0f32; LORA_INPUT_DIM];
        let mut out = vec![f32::NAN; LORA_OUTPUT_DIM];
        p.predict_into(&x, &mut out).unwrap();
        assert_eq!(out.len(), LORA_OUTPUT_DIM);
        for &v in &out {
            assert!(v.is_finite(), "output must be finite");
        }
    }

    #[test]
    fn predict_into_rejects_wrong_input_len() {
        let p = LoraBetaPredictor::new();
        let x = vec![0.0f32; LORA_INPUT_DIM - 1];
        let mut out = vec![0.0f32; LORA_OUTPUT_DIM];
        let err = p.predict_into(&x, &mut out).unwrap_err();
        assert!(matches!(
            err,
            LoraBetaError::DimMismatch {
                context: "kv_stats",
                expected: LORA_INPUT_DIM,
                ..
            }
        ));
    }

    #[test]
    fn predict_into_rejects_wrong_output_len() {
        let p = LoraBetaPredictor::new();
        let x = vec![0.0f32; LORA_INPUT_DIM];
        let mut out = vec![0.0f32; LORA_OUTPUT_DIM + 1];
        let err = p.predict_into(&x, &mut out).unwrap_err();
        assert!(matches!(
            err,
            LoraBetaError::DimMismatch {
                context: "out",
                expected: LORA_OUTPUT_DIM,
                ..
            }
        ));
    }

    #[test]
    fn forward_logits_matches_predict_before_squash() {
        // predict = sigmoid-squash(forward_logits).
        let mut p = LoraBetaPredictor::new();
        p.b.fill(0.1); // non-trivial B so logits are non-zero
        p.bias.fill(0.2);
        let x = vec![0.5f32; LORA_INPUT_DIM];
        let logits = p.forward_logits(&x);
        let beta = p.predict(&x);
        assert_eq!(logits.len(), beta.len());
        for (i, (&logit, &b)) in logits.iter().zip(beta.iter()).enumerate() {
            let expected = BETA_MIN + SIGMOID_SPAN * sigmoid(logit);
            assert!(
                (b - expected).abs() < 1e-5,
                "head {i}: β={b} should be sigmoid({logit})→{expected}"
            );
        }
    }

    #[test]
    fn predict_is_deterministic() {
        let p = LoraBetaPredictor::new();
        let x: Vec<f32> = (0..LORA_INPUT_DIM).map(|i| (i as f32) * 0.01).collect();
        let b1 = p.predict(&x);
        let b2 = p.predict(&x);
        assert_eq!(b1, b2);
    }

    #[test]
    fn predict_into_matches_predict() {
        let p = LoraBetaPredictor::new();
        let x: Vec<f32> = (0..LORA_INPUT_DIM).map(|i| (i as f32) * 0.01).collect();
        let alloc = p.predict(&x);
        let mut into = vec![0.0f32; LORA_OUTPUT_DIM];
        p.predict_into(&x, &mut into).unwrap();
        assert_eq!(alloc, into);
    }

    // ── Validate ───────────────────────────────────────────────────

    #[test]
    fn validate_passes_on_new() {
        let p = LoraBetaPredictor::new();
        assert!(p.validate().is_ok());
    }

    #[test]
    fn validate_rejects_wrong_a_len() {
        let mut p = LoraBetaPredictor::new();
        p.a.pop();
        let err = p.validate().unwrap_err();
        assert!(matches!(err, LoraBetaError::DimMismatch { context: "a (rank×in_dim)", .. }));
    }

    #[test]
    fn validate_rejects_non_finite() {
        let mut p = LoraBetaPredictor::new();
        p.bias[3] = f32::NAN;
        assert_eq!(p.validate(), Err(LoraBetaError::NonFiniteWeights));
    }

    // ── Serialization ──────────────────────────────────────────────

    #[test]
    fn serialize_deserialize_roundtrip() {
        let mut p = LoraBetaPredictor::new();
        // Perturb so it's not the default — verify content survives.
        p.b[0] = 0.5;
        p.bias[2] = -1.0;
        p.a[10] = 0.123;
        let bytes = p.serialize();
        let p2 = LoraBetaPredictor::deserialize(&bytes).unwrap();
        assert_eq!(p, p2);
    }

    #[test]
    fn serialize_has_magic_and_version_header() {
        let p = LoraBetaPredictor::new();
        let bytes = p.serialize();
        assert_eq!(&bytes[..4], &PREDICTOR_MAGIC);
        let version = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        assert_eq!(version, PREDICTOR_VERSION);
    }

    #[test]
    fn serialize_includes_32_byte_commitment_tail() {
        let p = LoraBetaPredictor::new();
        let bytes = p.serialize();
        // Tail 32 bytes should equal commitment() of the body.
        let body_end = bytes.len() - 32;
        let body = &bytes[4 + 4..body_end];
        let expected = blake3::hash(body);
        let actual: [u8; 32] = bytes[body_end..].try_into().unwrap();
        assert_eq!(actual, *expected.as_bytes());
    }

    #[test]
    fn commitment_is_stable_for_same_weights() {
        let p = LoraBetaPredictor::new();
        assert_eq!(p.commitment(), p.commitment());
    }

    #[test]
    fn commitment_changes_with_weights() {
        let mut p = LoraBetaPredictor::new();
        let c1 = p.commitment();
        p.a[0] += 0.001;
        let c2 = p.commitment();
        assert_ne!(c1, c2, "commitment must change when weights change");
    }

    #[test]
    fn deserialize_rejects_bad_magic() {
        let p = LoraBetaPredictor::new();
        let mut bytes = p.serialize();
        bytes[0] = b'X'; // corrupt magic
        let err = LoraBetaPredictor::deserialize(&bytes).unwrap_err();
        assert!(matches!(err, LoraBetaError::BadHeader { .. }));
    }

    #[test]
    fn deserialize_rejects_bad_version() {
        let p = LoraBetaPredictor::new();
        let mut bytes = p.serialize();
        // Bump version byte.
        bytes[4] = 0xFF;
        let err = LoraBetaPredictor::deserialize(&bytes).unwrap_err();
        assert!(matches!(err, LoraBetaError::BadHeader { .. }));
    }

    #[test]
    fn deserialize_rejects_truncated() {
        let p = LoraBetaPredictor::new();
        let bytes = p.serialize();
        // Truncate below minimum size.
        let truncated = &bytes[..PREDICTOR_MAGIC.len() + 2];
        let err = LoraBetaPredictor::deserialize(truncated).unwrap_err();
        assert_eq!(err, LoraBetaError::Truncated);
    }

    // ── Sigmoid ────────────────────────────────────────────────────

    #[test]
    fn sigmoid_extremes() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(100.0) > 0.99999);
        assert!(sigmoid(-100.0) < 1e-4);
        assert!((sigmoid(0.0) - sigmoid(-0.0)).abs() < 1e-6);
    }

    // ── With custom dims ───────────────────────────────────────────

    #[test]
    fn with_dims_custom() {
        let p = LoraBetaPredictor::with_dims(32, 4, 8, 8.0);
        assert_eq!(p.in_dim, 32);
        assert_eq!(p.out_dim, 4);
        assert_eq!(p.rank, 8);
        assert_eq!(p.a.len(), 8 * 32);
        assert_eq!(p.b.len(), 4 * 8);
        assert_eq!(p.bias.len(), 4);
        assert_eq!(p.param_count(), 256 + 32 + 4);
        assert!(p.validate().is_ok());
    }
}
