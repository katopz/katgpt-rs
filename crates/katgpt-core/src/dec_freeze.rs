//! Cochain Freeze Envelope — BLAKE3-committed freeze/thaw for [`CochainField`].
//!
//! Mirrors the [`LatentSteeringEnvelope`](crate::latent_steering::LatentSteeringEnvelope)
//! integrity pattern: serialize `dim` + `rank` + `data` into a flat byte
//! buffer, compute a BLAKE3 commitment over it, and verify before thawing.
//! This closes the cochain freeze-gap identified in Issue 455 (adaptive-sync
//! migration freeze gaps).
//!
//! ## Serialization format
//!
//! `dim` (u32 LE, 4 bytes) ‖ `rank` (u8, 1 byte) ‖ `data` (f32 LE × n, n·4 bytes)
//!
//! ## Tamper detection
//!
//! Any bit-flip in `payload` or `commitment` is detected on `verify()` / `thaw()`.
//! Malformed payloads (truncated, dim=0 with data, NaN in data) are rejected
//! on `thaw()`.

use crate::dec::CochainField;

/// BLAKE3-committed freeze/thaw envelope for a [`CochainField`].
///
/// The envelope is self-contained: it carries both the serialized cochain
/// payload and the BLAKE3 commitment over that payload. Thawing verifies the
/// commitment first, then deserializes — returning `None` on any tamper or
/// malformation.
#[derive(Clone, Debug)]
pub struct CochainFreezeEnvelope {
    /// BLAKE3 commitment over `payload`.
    commitment: [u8; 32],
    /// Serialized payload: `dim` (u32 LE) || `rank` (u8) || `data` (f32 LE × n).
    payload: Vec<u8>,
}

impl CochainFreezeEnvelope {
    /// Freeze a [`CochainField`] into a self-contained envelope.
    ///
    /// Serializes `dim` + `rank` + `data` into a byte buffer and computes a
    /// BLAKE3 commitment over it. The envelope commitment is an independent
    /// integrity layer over the serialized form.
    pub fn freeze(cf: &CochainField) -> Self {
        let dim = cf.dim as u32;
        let mut payload = Vec::with_capacity(4 + 1 + cf.data.len() * 4);
        payload.extend_from_slice(&dim.to_le_bytes());
        payload.push(cf.rank);
        for &f in &cf.data {
            payload.extend_from_slice(&f.to_le_bytes());
        }
        let commitment = *blake3::hash(&payload).as_bytes();
        Self {
            commitment,
            payload,
        }
    }

    /// Verify that the envelope's commitment matches its payload.
    ///
    /// Returns `false` if the payload has been tampered with after freezing.
    #[inline]
    pub fn verify(&self) -> bool {
        *blake3::hash(&self.payload).as_bytes() == self.commitment
    }

    /// Thaw the envelope back into a [`CochainField`].
    ///
    /// Verifies the commitment first; returns `None` if verification fails
    /// (tampered envelope) or if the payload is malformed (truncated,
    /// dim=0 with data, NaN values).
    pub fn thaw(&self) -> Option<CochainField> {
        if !self.verify() {
            return None;
        }
        Self::deserialize(&self.payload)
    }

    /// The BLAKE3 commitment over the serialized payload.
    #[inline]
    pub fn commitment(&self) -> [u8; 32] {
        self.commitment
    }

    /// Serialized payload bytes (for external persistence / network transport).
    #[inline]
    pub fn payload(&self) -> &[u8] {
        &self.payload
    }

    /// Deserialize a payload into a [`CochainField`].
    ///
    /// Returns `None` on any malformation: truncated buffer, data length not
    /// divisible by 4, dim=0 with non-empty data, data length not divisible
    /// by dim, or NaN in the data.
    fn deserialize(payload: &[u8]) -> Option<CochainField> {
        // Header: dim (u32 LE, 4 bytes) + rank (u8, 1 byte) = 5 bytes minimum.
        if payload.len() < 5 {
            return None;
        }
        let dim = u32::from_le_bytes(payload[..4].try_into().ok()?) as usize;
        let rank = payload[4];
        let data_bytes = payload.len() - 5;
        // Data must be a whole number of f32s.
        if !data_bytes.is_multiple_of(4) {
            return None;
        }
        let n_floats = data_bytes / 4;
        // dim=0 with non-empty data is a malformation; non-zero dim requires
        // data length divisible by dim for a well-formed cochain.
        if dim == 0 {
            if n_floats != 0 {
                return None;
            }
        } else if !n_floats.is_multiple_of(dim) {
            return None;
        }
        let mut data = Vec::with_capacity(n_floats);
        for chunk in payload[5..].as_chunks::<4>().0 {
            let f = f32::from_le_bytes(*chunk);
            if f.is_nan() {
                return None;
            }
            data.push(f);
        }
        Some(CochainField::from_vec(rank, dim, data))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Freeze a CochainField, thaw it, assert bit-identical reconstruction.
    #[test]
    fn freeze_thaw_roundtrip() {
        let cf = CochainField::from_vec(1, 3, vec![1.0, 2.5, -3.7, 0.0, 42.0, -0.001]);
        let env = CochainFreezeEnvelope::freeze(&cf);
        let thawed = env.thaw().expect("thaw should succeed on a valid envelope");

        assert_eq!(thawed.dim, cf.dim, "dim mismatch");
        assert_eq!(thawed.rank, cf.rank, "rank mismatch");
        assert_eq!(thawed.data.len(), cf.data.len(), "data length mismatch");
        for (a, b) in thawed.data.iter().zip(cf.data.iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "data bit mismatch");
        }
    }

    /// Tamper with the payload → verify fails, thaw returns None.
    #[test]
    fn tampered_payload_detected() {
        let cf = CochainField::from_vec(0, 1, vec![1.0, 2.0, 3.0]);
        let mut env = CochainFreezeEnvelope::freeze(&cf);
        // Flip a bit in the payload.
        env.payload[5] ^= 0x01;

        assert!(!env.verify(), "verify should fail on tampered payload");
        assert!(
            env.thaw().is_none(),
            "thaw should return None on tampered payload"
        );
    }

    /// Tamper with the commitment → verify fails.
    #[test]
    fn tampered_commitment_detected() {
        let cf = CochainField::from_vec(0, 1, vec![1.0, 2.0, 3.0]);
        let mut env = CochainFreezeEnvelope::freeze(&cf);
        // Flip a bit in the commitment.
        env.commitment[0] ^= 0x01;

        assert!(!env.verify(), "verify should fail on tampered commitment");
    }

    /// CochainField with empty data should freeze/thaw correctly.
    #[test]
    fn empty_data_freezes() {
        let cf = CochainField::from_vec(2, 1, vec![]);
        let env = CochainFreezeEnvelope::freeze(&cf);
        assert!(env.verify(), "verify should pass on empty-data envelope");

        let thawed = env.thaw().expect("thaw should succeed on empty data");
        assert_eq!(thawed.dim, 1);
        assert_eq!(thawed.rank, 2);
        assert!(thawed.data.is_empty(), "thawed data should be empty");
    }

    /// Payload containing NaN → thaw returns None.
    #[test]
    fn nan_in_data_rejected_on_thaw() {
        // Manually craft a payload with NaN in the data section.
        let dim = 1u32;
        let rank = 0u8;
        let nan_bytes = f32::NAN.to_le_bytes();
        let mut payload = Vec::new();
        payload.extend_from_slice(&dim.to_le_bytes());
        payload.push(rank);
        payload.extend_from_slice(&nan_bytes);
        let commitment = *blake3::hash(&payload).as_bytes();
        let env = CochainFreezeEnvelope {
            commitment,
            payload,
        };

        // Commitment is valid (verify passes), but NaN is rejected on deserialize.
        assert!(env.verify(), "commitment should match the crafted payload");
        assert!(env.thaw().is_none(), "thaw should reject NaN in data");
    }

    /// Freezing the same CochainField twice → identical commitments.
    #[test]
    fn deterministic_commitment() {
        let cf = CochainField::from_vec(1, 2, vec![0.1, 0.2, 0.3, 0.4]);
        let env_a = CochainFreezeEnvelope::freeze(&cf);
        let env_b = CochainFreezeEnvelope::freeze(&cf);

        assert_eq!(
            env_a.commitment(),
            env_b.commitment(),
            "commitments should be deterministic"
        );
        assert_eq!(
            env_a.payload(),
            env_b.payload(),
            "payloads should be identical"
        );
    }

    /// Same data, different rank → different commitment.
    #[test]
    fn different_rank_different_commitment() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let cf_a = CochainField::from_vec(0, 2, data.clone());
        let cf_b = CochainField::from_vec(1, 2, data);

        let env_a = CochainFreezeEnvelope::freeze(&cf_a);
        let env_b = CochainFreezeEnvelope::freeze(&cf_b);

        assert_ne!(
            env_a.commitment(),
            env_b.commitment(),
            "different ranks should produce different commitments"
        );
    }
}
