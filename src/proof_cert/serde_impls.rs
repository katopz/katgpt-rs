//! Binary persistence for proof certificates using `postcard`.
//!
//! No JSON. Binary format with magic bytes + version header.
//! Same pattern as `crate::pruners::freeze`.

use std::path::Path;

use super::certificate::ProofCertificate;
use super::chain::verify_proof_chain;

/// Magic bytes for certificate binary format.
const CERT_MAGIC: [u8; 4] = *b"CERT";

/// Format version. Bump if wire format changes.
const CERT_VERSION: u32 = 1;

/// Save proof certificates as a verifiable binary artifact with blake3 checksum.
pub fn save_certificates(
    certificates: &[ProofCertificate],
    path: &Path,
) -> Result<blake3::Hash, String> {
    let payload =
        postcard::to_allocvec(certificates).map_err(|e| format!("Serialization error: {e}"))?;

    // Header: magic(4) + version(4) + payload_len(4) = 12 bytes
    let mut buf = Vec::with_capacity(12 + payload.len());
    buf.extend_from_slice(&CERT_MAGIC);
    buf.extend_from_slice(&CERT_VERSION.to_le_bytes());
    buf.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    buf.extend_from_slice(&payload);

    let hash = blake3::hash(&buf);
    std::fs::write(path, &buf).map_err(|e| format!("Write error: {e}"))?;
    Ok(hash)
}

/// Load and verify proof certificates from binary format.
pub fn load_certificates(path: &Path) -> Result<Vec<ProofCertificate>, String> {
    let buf = std::fs::read(path).map_err(|e| format!("Read error: {e}"))?;

    if buf.len() < 12 {
        return Err("Certificate file too small".into());
    }
    if &buf[..4] != &CERT_MAGIC {
        return Err("Invalid certificate file (bad magic)".into());
    }
    let version = u32::from_le_bytes(
        buf[4..8]
            .try_into()
            .map_err(|e: std::array::TryFromSliceError| e.to_string())?,
    );
    if version != CERT_VERSION {
        return Err(format!("Unsupported certificate version {version}"));
    }
    let payload_len = u32::from_le_bytes(
        buf[8..12]
            .try_into()
            .map_err(|e: std::array::TryFromSliceError| e.to_string())?,
    ) as usize;
    if buf.len() < 12 + payload_len {
        return Err("Certificate file truncated".into());
    }

    let certs: Vec<ProofCertificate> = postcard::from_bytes(&buf[12..12 + payload_len])
        .map_err(|e| format!("Deserialization error: {e}"))?;
    let _result = verify_proof_chain(&certs);
    Ok(certs)
}

/// Verify blake3 checksum of a certificate file.
pub fn verify_checksum(path: &Path, expected: &blake3::Hash) -> bool {
    let Ok(bytes) = std::fs::read(path) else {
        return false;
    };
    &blake3::hash(&bytes) == expected
}
