//! GOAT gate — the sink-audit policy switch (Issue 981 T3, extending the
//! Issue-841 `kv_sink_window` module's gate surface).
//!
//! ```bash
//! cargo bench --features kv_sink_window --bench bench_981_sink_audit_switch_goat
//! # G4 needs an allocator in the profile:
//! cargo bench --features kv_sink_window,alloc_tracking \
//!     --bench bench_981_sink_audit_switch_goat
//! ```
//!
//! # What is being gated
//!
//! The census → verdict → policy chain: [`accumulate_head_mass_into`] over a
//! synthetic (layers × heads × positions) attention surface, the chosen cut
//! points, and [`SinkWindowPolicy::from_audit`]'s posture mapping.
//!
//! **G2 discrimination** — a planted-sink surface (K[0] aligned with every
//! query) must classify `SinkHeavy`; a sink-free surface (no alignment) must
//! classify `SinkFree`. The paper's cluster ordering, reproduced by
//! construction. The REAL gated-vs-ungated checkpoint pair (the paper's 1.7B
//! baseline/gated pair + the qwen3.8 dense read) is the census lane's
//! fixture — see riir-infer's sink-audit record; this bench pins the
//! arithmetic.
//!
//! **G3 policy correctness** — the switch arm: verdict ⇒ correct posture
//! over a grid of profiles (floor honored, incumbent never grown, window
//! carried, heavy/undecided keep the incumbent).
//!
//! **The planted-sink canaries MUST trip** — two deliberately-mislabeled
//! checkers (a SinkFree-expecting checker fed the planted surface, a
//! SinkHeavy-expecting checker fed the free surface) must FAIL. A gate that
//! cannot fail is a green zero.
//!
//! **Population floor** — the census arms scan a stated minimum of
//! layers × heads × positions; a walk regression that shrinks the surface
//! must red.
//!
//! **G1 determinism** — two passes over the same surface land bit-identical
//! profiles and byte-identical formatted payloads.
//!
//! **G4 alloc-free census pass** — `alloc_tracking` profile: the accumulate
//! loop must hold steady-state allocations at zero once the workspace row is
//! warm.

#[cfg(any(debug_assertions, feature = "alloc_tracking"))]
#[path = "../tests/common/alloc_tracking.rs"]
mod alloc_tracking;

use katgpt_core::kv_sink_window::{
    AuditedSinkPolicy, SinkAuditProfile, SinkAuditVerdict, SinkWindowPolicy,
    accumulate_head_mass_into, finalize_profile,
};

/// Census surface: the profile shape a real hybrid carries at small scale —
/// 4 full-attention layers × 8 heads × 256 positions.
const N_LAYERS: usize = 4;
const N_HEADS: usize = 8;
const N_POSITIONS: usize = 256;
const HEAD_DIM: usize = 32;
/// 1/√32 — the engine's 1/√hd scale, constant-folded (f32::sqrt is not const).
const SCALE: f32 = 0.176_776_7;
/// Population floor: every layer × head × query observation counts; the
/// census arms must scan at least this many (a shrunken surface reds).
const MIN_OBSERVATIONS: u64 = (N_LAYERS * N_HEADS * N_POSITIONS) as u64;

/// Deterministic LCG — a fixed seed, so every arm sees identical inputs and
/// a re-run reproduces the number (the bench_841 fixture pattern).
struct Lcg(u64);
impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f32 / (1u64 << 31) as f32) - 1.0
    }
}

/// One synthetic full-attention layer: every head's K rows for all positions
/// (row-major `[position × head_dim]`), shared across the layer's queries so
/// the surface stays L×H×N rather than L×H×N×N.
struct SyntheticLayer {
    /// `[head][position × head_dim]`.
    keys: Vec<Vec<f32>>,
}

/// A sink-free surface: K rows carry no alignment with the queries — the
/// mass stays diffuse (no position dominates).
fn sink_free_surface() -> Vec<SyntheticLayer> {
    let mut rng = Lcg(0x5151_7777);
    let mut layers = Vec::with_capacity(N_LAYERS);
    for _ in 0..N_LAYERS {
        let mut keys = Vec::with_capacity(N_HEADS);
        for _ in 0..N_HEADS {
            let mut k = vec![0.0f32; N_POSITIONS * HEAD_DIM];
            for v in k.iter_mut() {
                *v = rng.next_f32();
            }
            keys.push(k);
        }
        layers.push(SyntheticLayer { keys });
    }
    layers
}

/// A planted-sink surface: position 0's K row (every head, every layer) is
/// the shared direction every query also carries, so every query's mass
/// collapses onto position 0 — the sunk-cluster signature.
fn planted_sink_surface() -> (Vec<SyntheticLayer>, Vec<Vec<f32>>) {
    let mut rng = Lcg(0x2525_9999);
    let mut q_by_layer = Vec::with_capacity(N_LAYERS);
    let mut layers = Vec::with_capacity(N_LAYERS);
    for _ in 0..N_LAYERS {
        let sink_dir: Vec<f32> = (0..HEAD_DIM).map(|_| rng.next_f32()).collect();
        let mut q_layer = Vec::with_capacity(N_POSITIONS * HEAD_DIM);
        for _ in 0..N_POSITIONS {
            for &s in sink_dir.iter() {
                // Every query carries the sink direction plus diffuse noise.
                q_layer.push(s * 2.0 + rng.next_f32() * 0.1);
            }
        }
        let mut keys = Vec::with_capacity(N_HEADS);
        for _ in 0..N_HEADS {
            let mut k = vec![0.0f32; N_POSITIONS * HEAD_DIM];
            for (d, slot) in k.iter_mut().take(HEAD_DIM).enumerate() {
                *slot = sink_dir[d] * 2.0; // position 0 = the sink direction
            }
            for v in k[HEAD_DIM..].iter_mut() {
                *v = rng.next_f32();
            }
            keys.push(k);
        }
        q_by_layer.push(q_layer);
        layers.push(SyntheticLayer { keys });
    }
    (layers, q_by_layer)
}

/// Run one census pass over a surface, accumulating through the shipped
/// core. Returns (accumulator, observations).
///
/// Iteration order: layers ascending, heads ascending, queries ascending —
/// the fixed order the G1 determinism contract names.
fn census_pass(
    layers: &[SyntheticLayer],
    q_by_layer: &[Vec<f32>],
    row: &mut Vec<f32>,
    acc: &mut Vec<f64>,
) -> u64 {
    acc.clear();
    acc.resize(N_POSITIONS, 0.0);
    let mut observations = 0u64;
    for (layer, q) in layers.iter().zip(q_by_layer) {
        for h in 0..N_HEADS {
            let keys = &layer.keys[h];
            for qi in 0..N_POSITIONS {
                let q_row = &q[qi * HEAD_DIM..(qi + 1) * HEAD_DIM];
                let n_keys = qi + 1;
                accumulate_head_mass_into(q_row, keys, n_keys, HEAD_DIM, SCALE, row, acc);
                observations += 1;
            }
        }
    }
    observations
}

fn profile_from(acc: &[f64]) -> SinkAuditProfile {
    let first_n_profile = finalize_profile(acc, N_LAYERS, N_HEADS, N_POSITIONS);
    SinkAuditProfile {
        f_attn_token0: first_n_profile[0],
        m_act: 0.0,
        layers_scanned: N_LAYERS,
        heads_scanned: N_HEADS,
        positions_scanned: N_POSITIONS,
        first_n_profile,
    }
}

/// Canonical payload bytes: fixed-precision profile lines, layer/head/position
/// header first — the byte string the G1 equality is asserted on.
fn payload_bytes(profile: &SinkAuditProfile) -> Vec<u8> {
    let mut s = String::new();
    s.push_str(&format!(
        "sink_audit v1 layers={} heads={} positions={} m_act={:.6}\n",
        profile.layers_scanned, profile.heads_scanned, profile.positions_scanned, profile.m_act
    ));
    for (k, mass) in profile.first_n_profile.iter().enumerate() {
        s.push_str(&format!("{k} {:.9}\n", mass));
    }
    s.into_bytes()
}

/// A discriminating checker: expects exactly one verdict. The canaries feed
/// these the WRONG surfaces and require the refusal.
fn expect_verdict(profile: &SinkAuditProfile, want: SinkAuditVerdict) -> Result<(), String> {
    let got = profile.verdict();
    if got == want {
        Ok(())
    } else {
        Err(format!(
            "verdict mismatch: want {want:?}, got {got:?} (f_attn_token0 {:.6})",
            profile.f_attn_token0
        ))
    }
}

fn main() {
    println!(
        "# bench_981_sink_audit_switch_goat — Issue 981 T3 (kv_sink_window {})",
        if cfg!(debug_assertions) {
            "dev profile"
        } else {
            "release profile"
        }
    );

    let free_layers = sink_free_surface();
    let mut rng = Lcg(0xA11C_E001);
    let free_q: Vec<Vec<f32>> = (0..N_LAYERS)
        .map(|_| {
            (0..N_POSITIONS * HEAD_DIM)
                .map(|_| rng.next_f32())
                .collect()
        })
        .collect();
    let (sink_layers, sink_q) = planted_sink_surface();

    // ── G2 discrimination ────────────────────────────────────────────────
    let mut row = Vec::new();
    let mut acc = Vec::new();
    let free_obs = census_pass(&free_layers, &free_q, &mut row, &mut acc);
    let free_profile = profile_from(&acc);
    let sink_obs = census_pass(&sink_layers, &sink_q, &mut row, &mut acc);
    let sink_profile = profile_from(&acc);

    assert!(
        free_obs >= MIN_OBSERVATIONS && sink_obs >= MIN_OBSERVATIONS,
        "population floor: {free_obs} / {sink_obs} observations < {MIN_OBSERVATIONS}"
    );
    println!("population: {free_obs} + {sink_obs} observations (floor {MIN_OBSERVATIONS}) OK");

    expect_verdict(&free_profile, SinkAuditVerdict::SinkFree)
        .unwrap_or_else(|e| panic!("G2 sink-free arm: {e}"));
    expect_verdict(&sink_profile, SinkAuditVerdict::SinkHeavy)
        .unwrap_or_else(|e| panic!("G2 planted-sink arm: {e}"));
    println!(
        "G2 discrimination: free f_attn_token0={:.6} → SinkFree | planted f_attn_token0={:.6} → SinkHeavy OK",
        free_profile.f_attn_token0, sink_profile.f_attn_token0
    );

    // ── the canaries MUST trip ───────────────────────────────────────────
    expect_verdict(&sink_profile, SinkAuditVerdict::SinkFree)
        .expect_err("canary 1 must trip: a planted sink must NOT read SinkFree");
    expect_verdict(&free_profile, SinkAuditVerdict::SinkHeavy)
        .expect_err("canary 2 must trip: a free surface must NOT read SinkHeavy");
    println!("canaries: both mislabeled checkers refused (the gate can fail) OK");

    // ── G3 the policy switch, over a profile grid ────────────────────────
    let incumbent = SinkWindowPolicy::new(4, 256);
    let grid: [(&[f64], SinkAuditVerdict); 5] = [
        (&[0.06, 0.01, 0.005, 0.004], SinkAuditVerdict::SinkFree),
        (&[0.06, 0.055, 0.01, 0.01], SinkAuditVerdict::SinkFree),
        (&[0.40, 0.02, 0.01, 0.01], SinkAuditVerdict::SinkHeavy),
        (&[0.18, 0.02, 0.01, 0.01], SinkAuditVerdict::Undecided),
        (&[0.01, 0.01, 0.01, 0.01], SinkAuditVerdict::SinkFree),
    ];
    for (masses, want) in grid {
        let p = SinkAuditProfile {
            f_attn_token0: masses[0],
            m_act: 0.0,
            layers_scanned: N_LAYERS,
            heads_scanned: N_HEADS,
            positions_scanned: masses.len(),
            first_n_profile: masses.to_vec(),
        };
        assert_eq!(p.verdict(), want, "grid profile {masses:?} misclassified");
        match SinkWindowPolicy::from_audit(want, &p, incumbent) {
            AuditedSinkPolicy::Audited(policy) => {
                assert_eq!(want, SinkAuditVerdict::SinkFree);
                assert!(
                    (1..=4).contains(&policy.n_sink),
                    "n_sink {} off the floor/incumbent band",
                    policy.n_sink
                );
                assert_eq!(policy.window, 256, "window carried");
                let leading = masses
                    .iter()
                    .take_while(|&&m| m >= katgpt_core::kv_sink_window::AUDIT_PROFILE_SINK_MIN)
                    .count()
                    .clamp(1, 4);
                assert_eq!(policy.n_sink, leading, "n_sink follows the leading mass");
            }
            verdict @ (AuditedSinkPolicy::KeepIncumbentSinkHeavy
            | AuditedSinkPolicy::KeepIncumbentUndecided) => {
                assert_ne!(want, SinkAuditVerdict::SinkFree, "free must audit");
                assert_eq!(
                    verdict,
                    if want == SinkAuditVerdict::SinkHeavy {
                        AuditedSinkPolicy::KeepIncumbentSinkHeavy
                    } else {
                        AuditedSinkPolicy::KeepIncumbentUndecided
                    }
                );
            }
        }
    }
    println!("G3 switch grid: 5 profiles → correct postures OK");

    // ── G1 determinism ───────────────────────────────────────────────────
    let mut acc_b = Vec::new();
    census_pass(&sink_layers, &sink_q, &mut row, &mut acc_b);
    assert_eq!(acc, acc_b, "two passes diverged");
    let profile_b = profile_from(&acc_b);
    assert_eq!(payload_bytes(&sink_profile), payload_bytes(&profile_b));
    println!("G1 determinism: bit-identical profiles + byte-identical payloads OK");

    // ── G4 alloc-free census pass ────────────────────────────────────────
    #[cfg(any(debug_assertions, feature = "alloc_tracking"))]
    {
        use katgpt_core::alloc::{get_alloc_stats, reset_alloc_stats};
        // Warm the workspace first: the G4 contract is a STEADY-STATE pass.
        let mut r = Vec::new();
        let mut a = Vec::new();
        census_pass(&sink_layers, &sink_q, &mut r, &mut a);
        reset_alloc_stats();
        let mut r2 = r.clone();
        let mut a2 = a.clone();
        census_pass(&sink_layers, &sink_q, &mut r2, &mut a2);
        let (count, bytes) = get_alloc_stats();
        // The clone() calls themselves allocate 2 (r2, a2); a warm census
        // pass must add ZERO beyond them — no per-observation allocation.
        if count > 2 {
            panic!(
                "G4 alloc-free census: {count} allocations ({bytes} B) over the bar (2 = the two workspace clones) — \
                 the census loop is allocating per observation"
            );
        }
        println!("G4 alloc-free census pass: {count} allocations over the clone bar OK");
    }
    #[cfg(not(any(debug_assertions, feature = "alloc_tracking")))]
    {
        println!(
            "G4 alloc-free census pass: SKIPPED (not measured at this profile) — \
             Re-run with `--features alloc_tracking`; a green run without it is not a G4 pass."
        );
    }

    println!("bench_981_sink_audit_switch_goat: ALL GATES PASS");
}
