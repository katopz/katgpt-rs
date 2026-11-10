//! MAG G4 — transfer beats raw cosine gate (Plan 418 Phase 2 T2.4).
//!
//! Verifies the paper's §4 headline: MAG class-conditional transfer prediction
//! beats raw centroid cosine (which is near-uninformative, ρ ≈ 0.03 in the
//! paper). The class-conditional metrics capture transfer structure that the
//! raw centroid cosine cannot see.
//!
//! Protocol:
//! 1. Construct 6 candidate datasets + 1 target in ℝ^64.
//!    Each dataset has 2 balanced classes (50 pos, 50 neg) drawn from
//!    N(+v_i, I) and N(−v_i, I) respectively.
//!    Candidate i's class direction v_i is rotated from the target's v_t
//!    by θ_i ∈ {0°, 18°, 36°, 54°, 72°, 90°}.
//!    Candidate 0 (θ=0°) is the best transfer; candidate 5 (θ=90°) is orthogonal.
//! 2. The overall centroid ≈ 0 for all datasets (balanced classes cancel),
//!    making raw centroid cosine near-uninformative (≈ random ranking).
//!    The class-conditional centroids ≈ ±v_i, making class-conditional cosine
//!    informative (cos(θ_i) ranking).
//! 3. Run 50 independent trials (fresh random data each).
//! 4. Gate: MAG class-conditional Top-1 ≥ 0.50 (3× random 1/6 ≈ 0.167).
//!    Raw cosine Top-1 ≈ random (< 0.40).

use katgpt_core::mag::{DataSet, TransferMetric, rank_candidates};

const D: usize = 64;
const N_CANDIDATES: usize = 6;
const N_PER_CLASS: usize = 50;
const N_TRIALS: usize = 50;

#[inline]
fn gaussian(rng: &mut fastrand::Rng) -> f32 {
    let u1 = rng.f32().max(1e-10);
    let u2 = rng.f32();
    let r = (-2.0 * u1.ln()).sqrt();
    let theta = 2.0 * std::f32::consts::PI * u2;
    r * theta.cos()
}

/// Generate a 2-class dataset: positive from N(+v, I), negative from N(-v, I).
/// Returns (activations, labels).
fn make_dataset(rng: &mut fastrand::Rng, v: &[f32; D]) -> (Vec<[f32; D]>, Vec<bool>) {
    let n = N_PER_CLASS * 2;
    let mut activations = vec![[0.0_f32; D]; n];
    let mut labels = vec![false; n];
    for i in 0..N_PER_CLASS {
        // Positive class
        for j in 0..D {
            activations[i][j] = v[j] + gaussian(rng);
        }
        labels[i] = true;
    }
    for i in 0..N_PER_CLASS {
        let idx = N_PER_CLASS + i;
        // Negative class
        for j in 0..D {
            activations[idx][j] = -v[j] + gaussian(rng);
        }
        labels[idx] = false;
    }
    (activations, labels)
}

/// Class direction rotated by `angle_deg` from [1, 0, ..., 0] in the (dim 0, dim 1) plane.
fn class_direction(angle_deg: f32) -> [f32; D] {
    let mut v = [0.0_f32; D];
    let rad = angle_deg * std::f32::consts::PI / 180.0;
    v[0] = rad.cos();
    v[1] = rad.sin();
    v
}

/// Run one trial. Returns (raw_cosine_top1_correct, mag_top1_correct).
/// "Correct" = Top-1 is candidate 0 (θ=0°, best transfer).
fn run_trial(seed: u64) -> (bool, bool) {
    let mut rng = fastrand::Rng::with_seed(seed);

    // Target class direction: [1, 0, ..., 0].
    let v_t = class_direction(0.0);

    // Candidate class directions: rotated by 0°, 18°, 36°, 54°, 72°, 90°.
    let angles: [f32; N_CANDIDATES] = [0.0, 18.0, 36.0, 54.0, 72.0, 90.0];
    let candidate_dirs: Vec<[f32; D]> = angles.iter().map(|&a| class_direction(a)).collect();

    // Generate datasets.
    let (tgt_acts, tgt_labels) = make_dataset(&mut rng, &v_t);
    let mut cand_data = Vec::with_capacity(N_CANDIDATES);
    for dir in &candidate_dirs {
        cand_data.push(make_dataset(&mut rng, dir));
    }

    // Build DataSet views.
    let target_ds = DataSet::new(&tgt_acts, &tgt_labels);
    let candidate_dss: Vec<DataSet<'_, [f32; D]>> = cand_data
        .iter()
        .map(|(acts, labels)| DataSet::new(acts, labels))
        .collect();

    // Raw cosine ranking.
    let raw_entries = rank_candidates(
        &candidate_dss,
        &target_ds,
        &[TransferMetric::CentroidCosine],
    )
    .expect("rank_candidates raw");
    let raw_top1 = raw_entries[0].candidate_idx == 0;

    // MAG class-conditional ranking (the paper's informative metrics).
    let mag_entries = rank_candidates(
        &candidate_dss,
        &target_ds,
        &[
            TransferMetric::ClassConditionalCosineBenign,
            TransferMetric::ClassConditionalCosineMalicious,
        ],
    )
    .expect("rank_candidates mag");
    let mag_top1 = mag_entries[0].candidate_idx == 0;

    (raw_top1, mag_top1)
}

#[test]
fn g4_mag_beats_raw_cosine() {
    let mut raw_hits = 0usize;
    let mut mag_hits = 0usize;

    for trial in 0..N_TRIALS {
        let seed = 0xA400_0000 + trial as u64;
        let (raw_correct, mag_correct) = run_trial(seed);
        if raw_correct {
            raw_hits += 1;
        }
        if mag_correct {
            mag_hits += 1;
        }
    }

    let raw_rate = raw_hits as f32 / N_TRIALS as f32;
    let mag_rate = mag_hits as f32 / N_TRIALS as f32;
    let random_floor = 1.0 / N_CANDIDATES as f32;

    println!("G4 transfer prediction ({N_TRIALS} trials):");
    println!("  Raw centroid cosine Top-1: {raw_rate:.3} (random ≈ {random_floor:.3})");
    println!("  MAG class-conditional Top-1: {mag_rate:.3} (gate ≥ 0.50)");

    // MAG must beat 3× random.
    assert!(
        mag_rate >= 0.50,
        "G4 FAIL: MAG Top-1 = {mag_rate:.3} < 0.50. Transfer prediction not working."
    );

    // Raw cosine should be near-random (well below 0.50).
    // This confirms the paper's ρ ≈ 0 finding on this synthetic structure.
    assert!(
        raw_rate < 0.40,
        "G4 WARNING: raw cosine Top-1 = {raw_rate:.3} — expected near-random (< 0.40). \
         The synthetic structure may not adequately confound raw cosine."
    );
}
