//! Core types for IsoQuant KV cache compression.

/// IsoQuant rotation mode.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IsoQuantMode {
    /// T(v) = q_L * v * conj(q_R) — full SO(4), 6 DOF per block.
    Full,
    /// T(v) = q_L * v — isoclinic SO(3) subgroup, 3 DOF per block.
    Fast,
}

/// Per-layer IsoQuant state.
#[derive(Debug, Clone)]
pub struct IsoQuantLayer {
    /// Key left quaternions: (w, x, y, z) per group — ceil(kv_dim/4) groups.
    pub key_q_left: Vec<[f32; 4]>,
    /// Key right quaternions: only for Full mode.
    pub key_q_right: Option<Vec<[f32; 4]>>,
    /// Value left quaternions.
    pub val_q_left: Vec<[f32; 4]>,
    /// Value right quaternions: only for Full mode.
    pub val_q_right: Option<Vec<[f32; 4]>>,
}

/// Configuration for IsoQuant KV cache.
#[derive(Debug, Clone)]
pub struct IsoQuantConfig {
    /// Number of transformer layers.
    pub n_layers: usize,
    /// KV dimension (head_dim × n_kv_heads). Padded to multiple of 4.
    pub kv_dim: usize,
    /// Maximum sequence length.
    pub max_seq_len: usize,
    /// Random seed for quaternion generation (deterministic).
    pub seed: u64,
    /// Rotation mode: Full (6 DOF) or Fast (3 DOF).
    pub mode: IsoQuantMode,
    /// Bits per key coordinate (2-4).
    pub key_bits: u8,
    /// Bits per value coordinate (2-4).
    pub val_bits: u8,
}
