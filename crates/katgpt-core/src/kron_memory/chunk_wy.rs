//! `kron_chunk_wy` — the chunked form of the scalar-target delta rule
//! (Issue 954 E10): for a chunk of `L` writes, the residual-coupling system
//! is `(I + N)·r = r⁰` with `N[l][m] = β_m·(G_kk ⊙ G_vv)[l][m]` — the
//! Hadamard product of the key and value Grams, COLUMN-scaled by the earlier
//! write's β — a unit-lower-triangular solve; the rank-1 features themselves
//! are never formed.
//!
//! Derivation (sequential causality): write `m` applies
//! `Δ_m = −β_m·r_m·v_m k_mᵀ`, so write `l > m` reads it back through
//! `v_lᵀΔ_m k_l = −β_m·r_m·(v_lᵀv_m)(k_lᵀk_m)` and
//!
//! ```text
//! r_l = r⁰_l − Σ_{m<l} β_m·r_m·G_kk[l][m]·G_vv[l][m]
//! ```
//!
//! with `r⁰_l` the residual against the chunk's OPENING state `S₀`. One
//! unit-LT forward substitution, then ONE fused rank-1 sum
//! `ΔS = −Σ_l β_l·r_l·v_l k_lᵀ`.
//!
//! Rounding differs from sequential (the residuals are computed against the
//! OPENING state and corrected through the solve, not through L intervening
//! state mutations) — the shipped sequential write path stays canonical for
//! anything synced or replayed; the chunk lane is a consolidation-replay /
//! batch-ingest throughput arm (G2) whose equality contract is the pinned
//! TOLERANCE, never bit identity.

use super::KronCell;
use crate::simd::dot_f32_ordered;

/// One chunk item: address pair + scalar target + per-write step size.
#[derive(Debug, Clone, Copy)]
pub struct ChunkItem<const DK: usize, const DV: usize> {
    pub k: [f32; DK],
    pub v: [f32; DV],
    pub c: f32,
    pub beta: f32,
}

/// Why a chunk was refused. The chunk is all-or-nothing: no state changes
/// unless the whole chunk applies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkRefusal {
    /// A zero-norm address in the chunk (position named).
    ZeroAddress(usize),
    /// A governor violation in the chunk (position named).
    Governor(usize),
    /// The chunk is empty — reported rather than silently doing nothing.
    Empty,
}

impl core::fmt::Display for ChunkRefusal {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ChunkRefusal::ZeroAddress(i) => write!(f, "chunk[{i}]: zero-norm address"),
            ChunkRefusal::Governor(i) => write!(f, "chunk[{i}]: contraction governor violation"),
            ChunkRefusal::Empty => write!(f, "empty chunk"),
        }
    }
}

/// The chunked apply: validates the whole chunk (governor + addresses),
/// computes `r⁰` against the OPENING state, builds the Gram triangles, solves
/// the unit-LT system, applies the fused rank-1 sum. Allocation-free for the
/// fixed `L` the caller's arrays carry (`scratch` must be `≥ 2·L² + 2·L`).
///
/// Returns the solved residuals (the audit trail: `|r_l|` is the per-write
/// miss the chunk as a whole drove to zero against the opening state).
///
/// # Errors
/// [`ChunkRefusal`] — the chunk is rejected whole; the cell is untouched.
pub fn write_chunk_wy<'s, const DK: usize, const DV: usize, const N: usize>(
    cell: &mut KronCell<DK, DV, N>,
    chunk: &[ChunkItem<DK, DV>],
    scratch: &'s mut [f32],
) -> Result<&'s [f32], ChunkRefusal> {
    let l = chunk.len();
    if l == 0 {
        return Err(ChunkRefusal::Empty);
    }
    let need = 2 * l * l + 2 * l;
    assert!(
        scratch.len() >= need,
        "write_chunk_wy scratch {} < required {need} (2·L² + 2·L)",
        scratch.len()
    );

    // Validate the WHOLE chunk before touching anything (all-or-nothing).
    for (i, item) in chunk.iter().enumerate() {
        let sk = dot_f32_ordered(&item.k, &item.k);
        let sv = dot_f32_ordered(&item.v, &item.v);
        let contraction = item.beta * sk * sv;
        if contraction <= 0.0 || contraction.is_nan() {
            return Err(ChunkRefusal::ZeroAddress(i));
        }
        if contraction >= 2.0 {
            return Err(ChunkRefusal::Governor(i));
        }
    }

    // Layout: [0..L²) g_kk lower triangle, [L²..2L²) g_vv lower triangle,
    // [2L²..2L²+L) r0, [2L²+L..2L²+2L) r. The solve reads only the lower
    // triangles (m < i), so only they are computed.
    let (g_kk, rest) = scratch.split_at_mut(l * l);
    let (g_vv, rest) = rest.split_at_mut(l * l);
    let (r0, rest) = rest.split_at_mut(l);
    let (r, _tail) = rest.split_at_mut(l);

    // r⁰ = v_lᵀ S₀ k_l − c_l  (against the OPENING state, the cell's pinned
    // read order).
    for (i, item) in chunk.iter().enumerate() {
        r0[i] = cell.read(&item.k, &item.v) - item.c;
    }

    for i in 0..l {
        for j in 0..=i {
            g_kk[i * l + j] = dot_f32_ordered(&chunk[i].k, &chunk[j].k);
            g_vv[i * l + j] = dot_f32_ordered(&chunk[i].v, &chunk[j].v);
        }
    }

    // Unit-LT solve (I + N)·r = r⁰, forward-substituted. The coupling on
    // r_m carries the EARLIER write's β (column scaling — see the module
    // derivation).
    for i in 0..l {
        let mut acc = r0[i];
        for m in 0..i {
            let n_im = chunk[m].beta * g_kk[i * l + m] * g_vv[i * l + m];
            acc -= n_im * r[m];
        }
        r[i] = acc;
    }

    // The fused application: ΔS = −Σ_l β_l·r_l·v_l k_lᵀ, walked row-wise over
    // the state (DK rows × DV cols × L rank-1s, nothing materialized).
    let rows = cell.state_rows_mut();
    for (i, row) in rows.iter_mut().enumerate() {
        for (item, &ri) in chunk.iter().zip(r.iter()) {
            let ki = item.k[i];
            if ki == 0.0 || ri == 0.0 {
                continue;
            }
            let step = -(item.beta * ri * ki);
            for (s, &vj) in row.iter_mut().zip(item.v.iter()) {
                *s += step * vj;
            }
        }
    }

    // Registry + bookkeeping: every chunk item is an APPLIED write (its
    // address enters the registry, its residual folds the EMA and the health
    // moments). The composite is NOT rank-1-invertible — undo_last returns
    // false until the next single write.
    for (item, &ri) in chunk.iter().zip(r.iter()) {
        cell.note_chunk_applied(&item.k, &item.v, ri, item.c);
    }

    Ok(r)
}

#[cfg(test)]
#[path = "chunk_tests.rs"]
mod chunk_tests;
