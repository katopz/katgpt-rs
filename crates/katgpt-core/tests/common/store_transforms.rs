//! Shared BLAKE3-seeded store-corruption transforms (Issue 984's noise and
//! shuffle arms). Split out of the store-falsification harness at Issue 985
//! so consumers that need only the transforms (the soft-read GOAT cell)
//! don't drag the AURC harness into their binary.
//!
//! Included via `#[path]` — never `mod` a file that another module in the
//! same crate also includes (double-include = duplicate symbols).

use blake3::Hasher;

/// Deterministic `SplitMix64` (the workspace idiom — no rand dep).
pub struct SplitMix64(u64);
impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    pub fn next_unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// Unit-normalize; a zero vector passes through (never NaN).
pub fn unit<const D: usize>(v: [f32; D]) -> [f32; D] {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 {
        let inv = 1.0 / n;
        v.map(|x| x * inv)
    } else {
        v
    }
}

/// BLAKE3 domain seed over a label + the store's bit pattern. Same store +
/// same label → same seed across runs and platforms; a different store (or
/// a single flipped bit) → an unrelated seed.
pub fn store_seed<const D: usize>(label: &str, rows: &[[f32; D]]) -> u64 {
    let mut h = Hasher::new();
    h.update(label.as_bytes());
    h.update(&(rows.len() as u64).to_le_bytes());
    for r in rows {
        for x in r {
            h.update(&x.to_bits().to_le_bytes());
        }
    }
    let out = h.finalize();
    u64::from_le_bytes(out.as_bytes()[..8].try_into().expect("8 bytes"))
}

/// Gaussian draw via Box-Muller from the seeded stream. `u1` is clamped
/// into (0, 1] so `ln` is finite (the stream's 24-bit grid guarantees a
/// bounded |z|; no NaN can enter the store).
fn next_gauss(rng: &mut SplitMix64) -> f32 {
    let min_u = 1.0 / (1u64 << 24) as f32;
    let u1 = (rng.next_unit() + min_u).min(1.0);
    let u2 = rng.next_unit();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}

/// The noise store: BLAKE3-seeded iid Gaussian at the store's GLOBAL entry
/// scale (the per-dim marginal variant leaks the between-cluster subspace
/// through a max-sim read — see the store_falsification module doc for the
/// measured lesson), each row re-normalized to unit length (matching the
/// real store's row norms). Deterministic: same store → byte-identical
/// output across runs (G1).
pub fn noise_store<const D: usize>(rows: &[[f32; D]]) -> Vec<[f32; D]> {
    let k = rows.len();
    if k == 0 {
        return Vec::new();
    }
    let mut rng = SplitMix64::new(store_seed("bench984:noise:v1", rows));
    let _ = rng.next_u64(); // decouple the noise stream from the seed fold
    let n_entries = (k * D) as f32;
    let mut mean = 0.0f32;
    let mut dev = 0.0f32;
    for r in rows {
        for x in r {
            mean += x;
        }
    }
    mean /= n_entries;
    for r in rows {
        for x in r {
            let d = x - mean;
            dev += d * d;
        }
    }
    dev = (dev / n_entries).sqrt();
    (0..k)
        .map(|_| {
            let mut row = [0.0f32; D];
            for x in row.iter_mut() {
                *x = mean + dev * next_gauss(&mut rng);
            }
            unit(row)
        })
        .collect()
}

/// The shuffled store: one fixed BLAKE3-seeded permutation of the
/// DIMENSION axis applied to every row (orthogonal → internal geometry
/// conserved exactly, the query↔store basis channel destroyed; see the
/// store_falsification module doc for why the permutation is over
/// coordinates, not rows, for set-max reads).
pub fn shuffled_store<const D: usize>(rows: &[[f32; D]]) -> Vec<[f32; D]> {
    let mut rng = SplitMix64::new(store_seed("bench984:shuffle:v1", rows));
    let mut perm: Vec<usize> = (0..D).collect();
    for i in (1..D).rev() {
        let j = (rng.next_u64() as usize) % (i + 1);
        perm.swap(i, j);
    }
    rows.iter()
        .map(|r| std::array::from_fn(|i| r[perm[i]]))
        .collect()
}
