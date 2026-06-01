//! Gated DeltaNet-2 (GDN2) — O(1) decode with decoupled erase/write gates.
//!
//! Implements a CPU SIMD recurrent attention decoder that replaces the growing
//! KV cache with a fixed-size state matrix S ∈ R^{d_k × d_v} per KV head.
//! Per-token cost is O(d_k × d_v), independent of sequence length.
//!
//! # Key Mechanism
//!
//! The GDN2 recurrent step applies four operations per token:
//!
//! 1. **Decay**: S *= Diag(α) — row-wise exponential decay
//! 2. **Read**: r = Sᵀ(b ⊙ k) — gated matvec with erase gate b
//! 3. **Update**: S += k ⊗ (w⊙v − r) — outer product delta rule
//! 4. **Readout**: o = Sᵀ q — query the updated state
//!
//! # Gate Configurations
//!
//! | Variant | Erase gate b | Write gate w | Best for |
//! |---------|-------------|-------------|----------|
//! | **EraseOnly** | Channel-wise [dk] | Scalar | Default, ~90% of full gain |
//! | **Full** | Channel-wise [dk] | Channel-wise [dv] | Maximum quality |
//! | **KDA** | Scalar β (tied) | Scalar β (tied) | Baseline comparison |
//!
//! # State Size Comparison
//!
//! | Config | Flat KV (O(N)) | AHLA (O(1)) | GDN2 (O(1)) | GDN2 Savings vs Flat |
//! |--------|----------------|-------------|-------------|---------------------|
//! | micro (hd=4, block=16) | 2,048 B | 640 B | 448 B | 78.1% |
//! | game (hd=8, block=170) | 43,520 B | 2,304 B | 2,816 B | 93.5% |
//! | bpe (hd=8, block=256) | 65,536 B | 2,304 B | 2,816 B | 95.7% |
//!
//! GDN2 state per head: d_k × d_v floats (persistent) + 2×d_k + d_v (projected per token, temp).
//! For micro (d_k=d_v=4): 16 + 12 = 28 floats/head = 112 B.
//! For game/bpe (d_k=d_v=8): 64 + 24 = 88 floats/head = 352 B.
//!
//! Both GDN2 and AHLA are O(1) constant — the key advantage over flat KV's O(N) growth.
//!
//! # Usage
//!
//! ```ignore
//! use microgpt::gdn2::{forward_gdn2, MultiLayerGdn2Cache};
//!
//! let config = Config::micro();
//! let weights = TransformerWeights::random(&config);
//! let mut ctx = ForwardContext::new(&config);
//! let mut cache = MultiLayerGdn2Cache::new(&config);
//!
//! // Streaming inference — no context window limit
//! let logits = forward_gdn2(&mut ctx, &weights, &mut cache, token, pos, &config);
//! ```
//!
//! **Note:** GDN2 computes a different function than softmax attention.
//! Models must be trained with GDN2 from scratch for quality.
//!
//! Reference: Yang, Zhang, Kautz (2024). "Gated Delta Networks: Fast Recurrent
//! Language Models with Constant-State Attention."
//! See `.research/070_Gated_DeltaNet_2.md` for full derivation.

pub mod forward;
pub mod kernel;
pub mod types;

pub use forward::{forward_gdn2, generate_gdn2_into};
pub use kernel::{gdn2_recurrent_step, gdn2_state_readout, gdn2_state_update, l2_normalize, sigmoid};
pub use types::{Gdn2GateConfig, Gdn2HeadState, Gdn2LayerState, MultiLayerGdn2Cache};
