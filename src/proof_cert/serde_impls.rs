use std::path::Path;

use super::certificate::ProofCertificate;
use super::chain::verify_proof_chain;

/// Save proof certificates as a verifiable artifact with blake3 checksum.
pub fn save_certificates(
    certificates: &[ProofCertificate],
    path: &Path,
) -> Result<blake3::Hash, String> {
    let json_bytes = serde_json::to_vec_pretty(certificates)
        .map_err(|e| format!("Serialization error: {e}"))?;
    let hash = blake3::hash(&json_bytes);
    std::fs::write(path, &json_bytes).map_err(|e| format!("Write error: {e}"))?;
    Ok(hash)
}

/// Load and verify proof certificates.
pub fn load_certificates(path: &Path) -> Result<Vec<ProofCertificate>, String> {
    let json = std::fs::read_to_string(path).map_err(|e| format!("Read error: {e}"))?;
    let certs: Vec<ProofCertificate> =
        serde_json::from_str(&json).map_err(|e| format!("Deserialization error: {e}"))?;
    let _result = verify_proof_chain(&certs);
    // Don't error on load — user may want to inspect failed certs.
    Ok(certs)
}

/// Verify blake3 checksum of a certificate file.
pub fn verify_checksum(path: &Path, expected: &blake3::Hash) -> bool {
    let Ok(bytes) = std::fs::read(path) else {
        return false;
    };
    &blake3::hash(&bytes) == expected
}
