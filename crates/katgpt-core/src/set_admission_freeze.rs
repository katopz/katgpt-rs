//! Frozen direction banks — the T4.4 self-adaptive freeze surface (Plan 599
//! track b): re-freeze improved direction banks from runtime evidence into a
//! BLAKE3-committed, tamper-evident, 1 KB artifact.
//!
//! [`crate::set_admission::DIRECTION_BANK`] is generated once from a fixed
//! seed — the maximally-separated-table design. Track b is the loop's other
//! half: when runtime evidence shows the const bank's fan-outs missing
//! grounding on an observed corpus distribution, the miss directions become
//! evidence for a NEW bank (evidence-biased sphere exclusion over the
//! augmented quasi-random pool), frozen into a
//! [`FrozenDirectionBank`] and consumed through the bank-parameterized fan
//! ([`crate::set_admission::fan_cap_ladder_into_bank`]).
//!
//! # The envelope pattern
//!
//! This matches the `MerkleFrozenEnvelope` pattern referenced across
//! katgpt-core (curator, latent_steering, swe_trajectory_freeze, …), but
//! lives here as a local lightweight BLAKE3 envelope — the facade constraint
//! forbids a cross-repo dep on riir-neuron-db's type, and this is a generic
//! open layer (the `TrajectoryFreezeEnvelope` precedent).
//!
//! ```text
//! magic        : [u8; 4]   // SDBF
//! version      : u32       // LE
//! bank_seed    : u64       // LE — provenance of the generator/bank
//! merkle_root  : [u8; 32]  // BLAKE3 over the 1 KB bank payload (f32 LE bits)
//! data_len     : u64       // LE — payload length (always 1024 at DIM=8)
//! commitment   : [u8; 32]  // BLAKE3 over the header above
//! bank         : 1024 B    // 32 × 8 × f32, little-endian bit pattern
//! ```
//!
//! All operations are deterministic: BLAKE3 is platform-independent, the
//! layout is fixed (no padding, no pointers, explicit LE), and the bank
//! payload is the raw `f32::to_bits` byte pattern — two freezers that agree
//! on the bank agree on every byte.
//!
//! # Evidence derivation (modelless)
//!
//! [`bank_from_evidence`] runs the SAME sphere-exclusion picker the const
//! bank was generated with ([`crate::set_admission::exclusion_centers_into`]
//! at [`crate::set_admission::BANK_THRESHOLD`]) over an augmented pool:
//! normalized evidence rows FIRST (biasing the maximally-separated centers
//! toward the observed distribution), then the standard quasi-random pool
//! (so coverage of the rest of the shell survives). Slots the picker leaves
//! unfilled backfill from the const table's tail rows — the envelope's
//! commitment pins the exact mix either way. No training, no gradient
//! descent: this is selection-time geometry, inside the modelless mandate.
//!
//! set_admission_freeze — the T4.4 self-adaptive freeze surface (Plan 599
//! track b): re-freeze improved direction banks from runtime evidence into a
//! BLAKE3-committed 1 KB artifact (the local MerkleFrozenEnvelope pattern),
//! consumed through `fan_cap_ladder_into_bank`. Opt-in with `set_admission`.

use crate::set_admission::{BANK_SIZE, BANK_THRESHOLD, DIM, DIRECTION_BANK};

/// Wire-format magic for [`FrozenDirectionBank`].
pub const SDBF_MAGIC: [u8; 4] = *b"SDBF";

/// Envelope version — bump to break wire compatibility.
pub const SDBF_VERSION: u32 = 1;

/// The bank payload's byte length: `BANK_SIZE × DIM × 4` (1024 B at d=8 —
/// the plan's stated artifact size).
pub const BANK_PAYLOAD_BYTES: usize = BANK_SIZE * DIM * 4;

/// Header bytes up to and including `data_len` — exactly what the
/// commitment hashes over.
const HASHED_HEADER_BYTES: usize = 4 + 4 + 8 + 32 + 8;

/// Full envelope header: the hashed part plus the commitment itself.
const HEADER_BYTES: usize = HASHED_HEADER_BYTES + 32;

/// A frozen, BLAKE3-committed direction bank: the 1 KB table plus its
/// tamper-evident envelope (see the module docs for the byte layout).
#[derive(Clone, Debug, PartialEq)]
pub struct FrozenDirectionBank {
    /// Format magic bytes — always [`SDBF_MAGIC`].
    pub magic: [u8; 4],
    /// Envelope version — currently [`SDBF_VERSION`].
    pub version: u32,
    /// Provenance: the generator/bank seed this table came from
    /// ([`crate::set_admission::BANK_SEED`] for the const table; the
    /// evidence seed for a re-frozen one).
    pub bank_seed: u64,
    /// BLAKE3 over the serialized bank payload.
    pub merkle_root: [u8; 32],
    /// Byte length of the payload (informational; [`BANK_PAYLOAD_BYTES`]).
    pub data_len: u64,
    /// BLAKE3 commitment over `(magic, version, bank_seed, merkle_root, data_len)`.
    pub commitment: [u8; 32],
    /// The bank rows themselves (unit directions; row-major).
    pub bank: [[f32; DIM]; BANK_SIZE],
}

/// Serialize a bank to its canonical little-endian bit pattern (row-major).
fn bank_bytes(bank: &[[f32; DIM]; BANK_SIZE]) -> [u8; BANK_PAYLOAD_BYTES] {
    let mut out = [0u8; BANK_PAYLOAD_BYTES];
    let mut o = 0usize;
    for row in bank {
        for v in row {
            out[o..o + 4].copy_from_slice(&v.to_bits().to_le_bytes());
            o += 4;
        }
    }
    out
}

/// Parse a bank from its canonical little-endian bit pattern.
fn bank_from_bytes(bytes: &[u8]) -> Option<[[f32; DIM]; BANK_SIZE]> {
    if bytes.len() != BANK_PAYLOAD_BYTES {
        return None;
    }
    let mut bank = [[0.0_f32; DIM]; BANK_SIZE];
    let mut o = 0usize;
    for row in bank.iter_mut() {
        for v in row.iter_mut() {
            let mut b = [0u8; 4];
            b.copy_from_slice(&bytes[o..o + 4]);
            *v = f32::from_bits(u32::from_le_bytes(b));
            o += 4;
        }
    }
    Some(bank)
}

impl FrozenDirectionBank {
    /// Freeze a bank with its provenance seed: `merkle_root =
    /// BLAKE3(payload)`, `commitment = BLAKE3(header)`.
    #[must_use]
    pub fn freeze(bank: [[f32; DIM]; BANK_SIZE], bank_seed: u64) -> Self {
        let payload = bank_bytes(&bank);
        let merkle_root = blake3::hash(&payload);
        let mut header = [0u8; HASHED_HEADER_BYTES];
        header[0..4].copy_from_slice(&SDBF_MAGIC);
        header[4..8].copy_from_slice(&SDBF_VERSION.to_le_bytes());
        header[8..16].copy_from_slice(&bank_seed.to_le_bytes());
        header[16..48].copy_from_slice(merkle_root.as_bytes());
        header[48..56].copy_from_slice(&(payload.len() as u64).to_le_bytes());
        let commitment = blake3::hash(&header);
        Self {
            magic: SDBF_MAGIC,
            version: SDBF_VERSION,
            bank_seed,
            merkle_root: *merkle_root.as_bytes(),
            data_len: payload.len() as u64,
            commitment: *commitment.as_bytes(),
            bank,
        }
    }

    /// The canonical artifact bytes — header fields followed by the payload.
    /// Storage format; [`Self::thaw`] is its inverse.
    #[must_use]
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(HEADER_BYTES + BANK_PAYLOAD_BYTES);
        out.extend_from_slice(&self.magic);
        out.extend_from_slice(&self.version.to_le_bytes());
        out.extend_from_slice(&self.bank_seed.to_le_bytes());
        out.extend_from_slice(&self.merkle_root);
        out.extend_from_slice(&self.data_len.to_le_bytes());
        out.extend_from_slice(&self.commitment);
        out.extend_from_slice(&bank_bytes(&self.bank));
        out
    }

    /// Verify the header: magic, version, and the commitment chain.
    #[must_use]
    pub fn verify_header(&self) -> bool {
        if self.magic != SDBF_MAGIC || self.version != SDBF_VERSION {
            return false;
        }
        if self.data_len != BANK_PAYLOAD_BYTES as u64 {
            return false;
        }
        let mut header = [0u8; HASHED_HEADER_BYTES];
        header[0..4].copy_from_slice(&self.magic);
        header[4..8].copy_from_slice(&self.version.to_le_bytes());
        header[8..16].copy_from_slice(&self.bank_seed.to_le_bytes());
        header[16..48].copy_from_slice(&self.merkle_root);
        header[48..56].copy_from_slice(&self.data_len.to_le_bytes());
        blake3::hash(&header).as_bytes() == &self.commitment
    }

    /// Verify the payload against the envelope: header chain AND
    /// `BLAKE3(bank payload) == merkle_root`.
    #[must_use]
    pub fn verify_payload(&self) -> bool {
        if !self.verify_header() {
            return false;
        }
        let payload = bank_bytes(&self.bank);
        blake3::hash(&payload).as_bytes() == &self.merkle_root
    }

    /// Parse + fully verify a canonical artifact. `None` on any shape,
    /// magic, version, or commitment mismatch — a tampered artifact never
    /// thaws into a bank.
    #[must_use]
    pub fn thaw(bytes: &[u8]) -> Option<Self> {
        if bytes.len() != HEADER_BYTES + BANK_PAYLOAD_BYTES {
            return None;
        }
        let mut magic = [0u8; 4];
        magic.copy_from_slice(&bytes[0..4]);
        let mut v4 = [0u8; 4];
        v4.copy_from_slice(&bytes[4..8]);
        let version = u32::from_le_bytes(v4);
        let mut v8 = [0u8; 8];
        v8.copy_from_slice(&bytes[8..16]);
        let bank_seed = u64::from_le_bytes(v8);
        let mut merkle_root = [0u8; 32];
        merkle_root.copy_from_slice(&bytes[16..48]);
        v8.copy_from_slice(&bytes[48..56]);
        let data_len = u64::from_le_bytes(v8);
        let mut commitment = [0u8; 32];
        commitment.copy_from_slice(&bytes[56..88]);
        let bank = bank_from_bytes(&bytes[HEADER_BYTES..])?;
        let candidate = Self {
            magic,
            version,
            bank_seed,
            merkle_root,
            data_len,
            commitment,
            bank,
        };
        if !candidate.verify_header() {
            return None;
        }
        if !candidate.verify_payload() {
            return None;
        }
        Some(candidate)
    }
}

/// Derive a bank from runtime evidence (the T4.4 track-b loop's producer).
///
/// `evidence` rows are the observed miss/corpus directions (normalized
/// internally; zero/non-finite rows skipped). The sphere-exclusion picker
/// runs over `normalized evidence ++ standard quasi-random pool` — evidence
/// first, so the maximally-separated centers land on the observed
/// distribution before the shell coverage fills in. Unfilled slots backfill
/// from the const table (rows `BANK_SIZE-1` downward, skipping any row
/// within the exclusion threshold of an already-picked center) so the bank
/// is always exactly [`BANK_SIZE`] unit rows.
///
/// Returns `(bank, picked)` where `picked` is the number of centers the
/// exclusion picker itself accepted (the evidence+pool-derived prefix).
///
/// Deterministic: same evidence order + same `evidence_seed` → identical
/// bytes. `evidence_seed` is provenance only (stamped into the envelope by
/// [`FrozenDirectionBank::freeze`]) — the derivation itself is
/// order-deterministic, seed-free.
#[must_use]
pub fn bank_from_evidence(
    evidence: &[[f32; DIM]],
    evidence_seed: u64,
) -> ([[f32; DIM]; BANK_SIZE], usize) {
    let _ = evidence_seed; // provenance stamp — see doc
    let mut pool: Vec<[f32; DIM]> = Vec::with_capacity(evidence.len() + 256);
    for row in evidence {
        if let Some(hat) = crate::set_admission::normalize(row) {
            pool.push(hat);
        }
    }
    let standard = crate::set_admission::bank_sample_pool::<DIM>();
    pool.extend_from_slice(&standard);

    // The picker needs a center slot per possible accept; the shared
    // certified_frontier cap bounds it exactly as the const generator does.
    let mut picked_idx = [0_usize; crate::certified_frontier::SPHERE_EXCLUSION_MAX_CENTERS];
    let accepted =
        crate::set_admission::exclusion_centers_into(&pool, BANK_THRESHOLD, &mut picked_idx);

    let mut bank = [[0.0_f32; DIM]; BANK_SIZE];
    for (row, &idx) in bank.iter_mut().zip(picked_idx.iter()) {
        *row = pool[idx];
    }
    let picked = accepted.min(BANK_SIZE);

    // Backfill from the const table's tail, skipping rows within the
    // picker's own exclusion radius (squared-chord, the SAME inclusive
    // comparison) of an already-picked center — separation holds bank-wide
    // wherever the picker itself left the slot unfilled.
    let t2 = BANK_THRESHOLD * BANK_THRESHOLD;
    let mut filled = picked;
    for t in DIRECTION_BANK.iter().rev() {
        if filled == BANK_SIZE {
            break;
        }
        let too_close = bank[..filled].iter().any(|c| {
            let mut d = 0.0_f32;
            for (a, b) in c.iter().zip(t.iter()) {
                let e = a - b;
                d += e * e;
            }
            d <= t2
        });
        if !too_close {
            bank[filled] = *t;
            filled += 1;
        }
    }
    // Last resort (picker collapse): cycle the const table verbatim.
    let mut t = 0usize;
    while filled < BANK_SIZE {
        bank[filled] = DIRECTION_BANK[t % BANK_SIZE];
        filled += 1;
        t += 1;
    }
    (bank, picked)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::set_admission::BANK_SEED;

    /// The T4.4 BLAKE3 round-trip bit-identity pin: freeze the const bank →
    /// canonical bytes → thaw → every bank bit identical + both verifications.
    #[test]
    fn freeze_thaw_round_trip_is_bit_identical() {
        let frozen = FrozenDirectionBank::freeze(DIRECTION_BANK, BANK_SEED);
        let bytes = frozen.canonical_bytes();
        assert_eq!(bytes.len(), HEADER_BYTES + BANK_PAYLOAD_BYTES);

        let thawed = FrozenDirectionBank::thaw(&bytes).expect("round trip must thaw");
        assert!(thawed.verify_header());
        assert!(thawed.verify_payload());
        assert_eq!(thawed.commitment, frozen.commitment);
        assert_eq!(thawed.bank_seed, BANK_SEED);
        for (a, b) in thawed.bank.iter().zip(DIRECTION_BANK.iter()) {
            for (x, y) in a.iter().zip(b.iter()) {
                assert_eq!(x.to_bits(), y.to_bits(), "bank bits must round-trip");
            }
        }
        // The 1 KB artifact size the plan states.
        assert_eq!(BANK_PAYLOAD_BYTES, 1024);
    }

    #[test]
    fn freeze_is_deterministic() {
        let a = FrozenDirectionBank::freeze(DIRECTION_BANK, BANK_SEED);
        let b = FrozenDirectionBank::freeze(DIRECTION_BANK, BANK_SEED);
        assert_eq!(a.commitment, b.commitment);
        assert_eq!(a.canonical_bytes(), b.canonical_bytes());
    }

    #[test]
    fn tampered_artifact_never_thaws() {
        let frozen = FrozenDirectionBank::freeze(DIRECTION_BANK, BANK_SEED);
        // Flip one bit inside the bank payload region.
        let mut bytes = frozen.canonical_bytes();
        let last = bytes.len() - 1;
        bytes[last] ^= 0x01;
        assert!(
            FrozenDirectionBank::thaw(&bytes).is_none(),
            "a flipped payload bit must fail the merkle_root check"
        );
        // Flip one bit inside the header region (the seed).
        let mut bytes = frozen.canonical_bytes();
        bytes[9] ^= 0x01;
        assert!(
            FrozenDirectionBank::thaw(&bytes).is_none(),
            "a flipped header bit must fail the commitment check"
        );
    }

    #[test]
    fn evidence_bank_is_deterministic_unit_and_separated() {
        let mut evidence = [[0.0_f32; DIM]; 24];
        for (i, row) in evidence.iter_mut().enumerate() {
            row[i % DIM] = 1.0;
            row[(i + 3) % DIM] = 0.5;
        }
        let (a, pa) = bank_from_evidence(&evidence, 0x50);
        let (b, pb) = bank_from_evidence(&evidence, 0x50);
        assert_eq!(a, b, "same evidence → identical bank");
        assert_eq!(pa, pb);

        assert_eq!(a.len(), BANK_SIZE);
        for row in &a {
            let n: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((n - 1.0).abs() < 1e-4, "rows must be unit (got {n})");
        }
        // Pairwise separation: every distinct pair sits beyond a loosened
        // threshold (loosened only because backfilled const rows are only
        // guaranteed separated against PICKED centers, and the picker's own
        // guarantee is strict at full precision — inclusive `<= t2` reject).
        for i in 0..BANK_SIZE {
            for j in (i + 1)..BANK_SIZE {
                let mut d = 0.0_f32;
                for (a, b) in a[i].iter().zip(a[j].iter()) {
                    let e = a - b;
                    d += e * e;
                }
                let chord = d.sqrt();
                assert!(
                    chord > BANK_THRESHOLD * 0.99,
                    "rows {i},{j} collapsed ({chord})"
                );
            }
        }
    }

    #[test]
    fn evidence_bank_aims_at_the_evidence() {
        // A cone the const bank covers poorly: pick the direction with the
        // WORST best-alignment across the const table (fine sampling over a
        // 7-dim tangent slice is overkill — a coarse golden-angle sweep
        // finds a well-misaligned aim), then check the evidence-derived
        // bank aligns with it strictly better than the const table does.
        let golden = 2.399_963_f32;
        let mut aim = [0.0_f32; DIM];
        let mut best_worst = -1.0_f32;
        for k in 0..64 {
            let mut v = [0.0_f32; DIM];
            for (d, slot) in v.iter_mut().enumerate() {
                *slot = ((k as f32 + 1.0) * golden * (d as f32 + 1.0)).sin();
            }
            let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if n == 0.0 {
                continue;
            }
            for x in v.iter_mut() {
                *x /= n;
            }
            let worst: f32 = DIRECTION_BANK
                .iter()
                .map(|t| t.iter().zip(v.iter()).map(|(a, b)| a * b).sum::<f32>())
                .fold(f32::INFINITY, f32::min);
            if worst > best_worst {
                best_worst = worst;
                aim = v;
            }
        }
        // Evidence: jittered copies of the aim direction.
        let mut evidence = [[0.0_f32; DIM]; 12];
        for (i, row) in evidence.iter_mut().enumerate() {
            for (d, slot) in row.iter_mut().enumerate() {
                *slot = aim[d] + 0.01 * ((i + d) as f32);
            }
        }
        let (bank, picked) = bank_from_evidence(&evidence, 0x51);
        assert!(picked >= 1, "evidence rows must seed at least one center");

        let align = |bank: &[[f32; DIM]; BANK_SIZE]| {
            bank.iter()
                .map(|t| t.iter().zip(aim.iter()).map(|(a, b)| a * b).sum::<f32>())
                .fold(f32::NEG_INFINITY, f32::max)
        };
        assert!(
            align(&bank) > align(&DIRECTION_BANK) + 0.1,
            "evidence bank must aim at the observed cone better than the const table \
             (evidence {}, const {})",
            align(&bank),
            align(&DIRECTION_BANK)
        );
    }

    #[test]
    fn bank_parameterized_fan_matches_const_bank() {
        // The DRY refactor's regression pin: the bank-parameterized fan with
        // the const table is BIT-IDENTICAL to the const fan.
        let corpus: Vec<[f32; DIM]> = (0..40)
            .map(|i| {
                let mut v = [0.0_f32; DIM];
                v[i % DIM] = 1.0;
                v[(i * 3 + 1) % DIM] = 0.4;
                let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                v.map(|x| x / n)
            })
            .collect();
        let query = [0.3, 0.1, 0.0, -0.2, 0.5, 0.0, 0.0, 0.8];
        let rungs = crate::set_admission::CAP_RUNGS;
        let tau = crate::set_admission::DEFAULT_GROUND_TAU;

        let mut idx_a = [0u16; 8];
        let mut pool_a = [[0.0_f32; DIM]; 8];
        let mut fan_a = crate::set_admission::FanScratch::new();
        let ra = crate::set_admission::fan_cap_ladder_into(
            &query,
            &corpus,
            &rungs,
            tau,
            &mut idx_a,
            &mut pool_a,
            &mut fan_a,
        );
        let mut idx_b = [0u16; 8];
        let mut pool_b = [[0.0_f32; DIM]; 8];
        let mut fan_b = crate::set_admission::FanScratch::new();
        let rb = crate::set_admission::fan_cap_ladder_into_bank(
            &query,
            &corpus,
            &rungs,
            tau,
            &mut idx_b,
            &mut pool_b,
            &mut fan_b,
            &DIRECTION_BANK,
        );
        assert_eq!(ra, rb);
        assert_eq!(idx_a, idx_b);
        assert_eq!(pool_a, pool_b);
    }

    #[test]
    fn evidence_bank_fan_improves_grounding() {
        // The improvement pin: on a corpus that IS the misaligned cone, the
        // evidence-derived bank's fan grounds strictly better than the const
        // bank's fan at the same rung.
        let golden = 2.399_963_f32;
        let mut aim = [0.0_f32; DIM];
        let mut best_worst = -1.0_f32;
        for k in 0..64 {
            let mut v = [0.0_f32; DIM];
            for (d, slot) in v.iter_mut().enumerate() {
                *slot = ((k as f32 + 1.0) * golden * (d as f32 + 1.0)).sin();
            }
            let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if n == 0.0 {
                continue;
            }
            for x in v.iter_mut() {
                *x /= n;
            }
            let worst: f32 = DIRECTION_BANK
                .iter()
                .map(|t| t.iter().zip(v.iter()).map(|(a, b)| a * b).sum::<f32>())
                .fold(f32::INFINITY, f32::min);
            if worst > best_worst {
                best_worst = worst;
                aim = v;
            }
        }
        let mut corpus = Vec::with_capacity(12);
        for i in 0..12 {
            let mut v = aim;
            for (d, slot) in v.iter_mut().enumerate() {
                *slot += 0.01 * (i as f32 + d as f32);
            }
            let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            corpus.push(v.map(|x| x / n));
        }
        let query = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let rungs = [0.95_f32]; // the widest rung — the fair fight
        let tau = crate::set_admission::DEFAULT_GROUND_TAU;

        // ONE slot: the improvement is per-slot aim quality (evidence row 0
        // IS the cone axis). A multi-slot min would compare the shared
        // const-like rows' grounding and read identical in both banks.
        let fan_at = |bank: &[[f32; DIM]; BANK_SIZE]| {
            let mut idx = [0u16; 1];
            let mut pool = [[0.0_f32; DIM]; 1];
            let mut fan = crate::set_admission::FanScratch::new();
            crate::set_admission::fan_cap_ladder_into_bank(
                &query, &corpus, &rungs, tau, &mut idx, &mut pool, &mut fan, bank,
            )
        };
        let r_const = fan_at(&DIRECTION_BANK);
        let (evidence_bank, _) = bank_from_evidence(&corpus, 0x52);
        let r_evidence = fan_at(&evidence_bank);
        assert!(
            r_evidence.min_ground > r_const.min_ground,
            "evidence-bank fan must ground better on the cone corpus \
             (evidence {}, const {})",
            r_evidence.min_ground,
            r_const.min_ground
        );
    }
}
