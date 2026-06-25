//! Plan 319 — G8c: Formation Robustness Simulation (Super-GOAT gate).
//!
//! Validates the product selling point: **do complementarity-weighted NPC
//! parties survive longer than similarity-weighted parties?**
//!
//! # The hypothesis
//!
//! In a varied-threat environment, a party of NPCs with **diverse** role
//! competencies (each strong in a different role) survives longer than a party
//! of NPCs with **homogeneous** competencies (all strong in the same role),
//! because the diverse party can neutralize more threat types.
//!
//! The Clifford wedge `h_A ∧ h_B` measures structural divergence between two
//! NPCs' HLA directions. A **complementarity-weighted** formation selects NPCs
//! whose directions are orthogonal (high wedge) → diverse roles. A
//! **similarity-weighted** formation selects NPCs whose directions are aligned
//! (high dot product) → same role.
//!
//! # The model (minimal but honest)
//!
//! 1. **NPC roles**: Each NPC's 64-dim HLA direction is projected onto 4 role
//!    axes (Tank, Healer, DPS, Support). Role competency = projection magnitude.
//!    A direction aligned with one axis is strong in that role and weak in
//!    others.
//!
//! 2. **Threat model**: Each combat round, a random threat type appears (one of
//!    the 4 role types). If the party has a member with competency ≥ τ_role in
//!    that role, the threat is **neutralized** (no damage). Otherwise, the party
//!    takes damage proportional to the threat severity.
//!
//! 3. **Party formation**: From a pool of 100 NPCs, form a 4-NPC party two ways:
//!    - **Complementarity**: greedily select NPCs that maximize the minimum
//!      pairwise wedge L1 norm (max-min diversity).
//!    - **Similarity**: greedily select NPCs that maximize the minimum pairwise
//!      dot product (max-min similarity).
//!
//! 4. **Survival**: Run 200 combat rounds per trial. Party starts with HP=100.
//!    Each un-neutralized threat deals 10 damage. Party dies when HP ≤ 0.
//!    Survival = number of rounds until death (or 200 if never dies).
//!
//! 5. **Gate**: Average survival over 1000 trials. **PASS** if complementarity
//!    survival ≥ 1.15× (15% longer) similarity survival.
//!
//! # Why this is honest
//!
//! The combat model is deliberately simple, but the **mechanism** (role
//! coverage → threat neutralization) is the same mechanism that makes diverse
//! parties robust in real games. The sim doesn't artificially inflate the
//! advantage — if threats are uniform (all the same type), diversity provides
//! zero benefit and both parties survive equally. The ≥15% threshold is
//! meaningful: it's the minimum improvement that would justify implementing
//! complementarity-based formation in production.
//!
//! # What this does NOT prove
//!
//! This sim uses a **synthetic** role model (4 fixed axes) and a **synthetic**
//! threat model (random uniform types). Real game combat has emergent dynamics
//! (positioning, cooldowns, synergies) that this sim doesn't capture. A full
//! G8c validation requires the riir-games encounter simulation. This minimal
//! sim validates the **core hypothesis** (complementarity → role diversity →
//! threat coverage → survival) under controlled conditions.
//!
//! # Reference
//!
//! - Plan: `katgpt-rs/.plans/319_geometric_product_latent_interaction.md`
//! - Research: `katgpt-rs/.research/299_Clifford_Geometric_Product_Latent_Interaction.md`
//! - Bridge (private): `riir-ai/crates/riir-engine/src/cgsp_runtime/clifford_bridge.rs`
//! - Primitive: [`katgpt_core::linalg::geometric_product_wedge_into`]

use std::time::Instant;

use katgpt_core::linalg::geometric_product_wedge_into;

// ─── Constants ─────────────────────────────────────────────────────────────

/// HLA direction dimension (CGSP `DEFAULT_HLA_DIM`).
const DIM: usize = 64;

/// Cyclic shifts for D=64 (clifford_bridge `DEFAULT_SHIFTS`).
const SHIFTS: &[usize] = &[1, 2, 4, 8, 16, 32];

/// Number of combat roles (Tank, Healer, DPS, Support).
const N_ROLES: usize = 4;

/// Role competency threshold for threat neutralization.
const ROLE_TAU: f32 = 0.35;

/// NPC pool size.
const POOL_SIZE: usize = 100;

/// Party size.
const PARTY_SIZE: usize = 4;

/// Combat rounds per trial.
const MAX_ROUNDS: usize = 200;

/// Damage per un-neutralized threat.
const DAMAGE_PER_THREAT: f32 = 10.0;

/// Starting party HP.
const START_HP: f32 = 100.0;

/// Number of trials (for statistical stability).
const N_TRIALS: usize = 1000;

/// G8c PASS threshold: complementarity survival ≥ this multiple of similarity.
const SURVIVAL_RATIO_TARGET: f32 = 1.15;

// ─── Deterministic RNG ─────────────────────────────────────────────────────

struct Rng {
    state: u32,
}

impl Rng {
    #[inline]
    fn new(seed: u32) -> Self {
        Self {
            state: if seed == 0 { 0xDEAD_BEEF } else { seed },
        }
    }

    #[inline]
    fn next_u32(&mut self) -> u32 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.state = x;
        x
    }

    #[inline]
    fn uniform(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 * (1.0f32 / (1u32 << 24) as f32)
    }

    #[inline]
    fn range(&mut self, n: usize) -> usize {
        self.next_u32() as usize % n
    }
}

// ─── NPC model ─────────────────────────────────────────────────────────────

/// An NPC with a 64-dim HLA direction and derived role competencies.
struct Npc {
    /// Unit-normalized 64-dim HLA direction.
    hla: Vec<f32>,
    /// Role competency per role (projection onto role axis, clamped ≥ 0).
    /// Length N_ROLES. A direction aligned with role `r` has high `roles[r]`
    /// and low `roles[other]`.
    roles: [f32; N_ROLES],
}

/// Role axes: 4 orthogonal unit vectors in 64-dim, one per role.
/// Generated as canonical basis vectors e_0, e_16, e_32, e_48 (widely spaced
/// so that random directions project unevenly onto them).
fn role_axes() -> [[f32; DIM]; N_ROLES] {
    let mut axes = [[0.0f32; DIM]; N_ROLES];
    for (r, axis) in axes.iter_mut().enumerate() {
        axis[r * (DIM / N_ROLES)] = 1.0; // e_{r*16}
    }
    axes
}

/// Compute role competencies from an HLA direction.
fn role_competencies(hla: &[f32], axes: &[[f32; DIM]; N_ROLES]) -> [f32; N_ROLES] {
    let mut roles = [0.0f32; N_ROLES];
    for r in 0..N_ROLES {
        let mut dot = 0.0f32;
        for d in 0..DIM {
            dot += hla[d] * axes[r][d];
        }
        roles[r] = dot.abs(); // competency = alignment magnitude
    }
    roles
}

/// L2 normalize a slice in place.
#[inline]
fn normalize_in_place(v: &mut [f32]) {
    let mut sum_sq = 0.0f32;
    for &x in v.iter() {
        sum_sq += x * x;
    }
    let norm = sum_sq.sqrt().max(1e-10);
    let inv = 1.0 / norm;
    for x in v.iter_mut() {
        *x *= inv;
    }
}

/// Generate the NPC pool: 100 NPCs with random unit-norm HLA directions.
fn generate_pool(seed: u32) -> Vec<Npc> {
    let mut rng = Rng::new(seed);
    let axes = role_axes();
    let mut pool = Vec::with_capacity(POOL_SIZE);
    for _ in 0..POOL_SIZE {
        let mut hla = vec![0.0f32; DIM];
        // 80% of NPCs are "specialists" (clustered near one role axis),
        // 20% are "generalists" (random direction). This mirrors real games
        // where most NPCs specialize but some are flexible.
        let is_specialist = rng.uniform() < 0.8;
        if is_specialist {
            let role = rng.range(N_ROLES);
            // Start from the role axis + Gaussian noise.
            let noise_sigma = 0.3f32;
            for d in 0..DIM {
                // Box-Muller for Gaussian noise.
                let u1 = rng.uniform().max(1e-10);
                let u2 = rng.uniform();
                let g = (-2.0f32 * u1.ln()).sqrt() * (2.0f32 * std::f32::consts::PI * u2).cos();
                hla[d] = axes[role][d] + noise_sigma * g;
            }
        } else {
            for h in hla.iter_mut().take(DIM) {
                let u1 = rng.uniform().max(1e-10);
                let u2 = rng.uniform();
                let g = (-2.0f32 * u1.ln()).sqrt() * (2.0f32 * std::f32::consts::PI * u2).cos();
                *h = g;
            }
        }
        normalize_in_place(&mut hla);
        let roles = role_competencies(&hla, &axes);
        pool.push(Npc { hla, roles });
    }
    pool
}

// ─── Party formation ───────────────────────────────────────────────────────

/// Wedge L1 norm between two HLA directions (structural divergence).
/// Uses the same `geometric_product_wedge_into` primitive as clifford_bridge.
#[inline]
fn wedge_l1(
    a: &[f32],
    b: &[f32],
    scratch_w: &mut [f32],
    scratch_su: &mut [f32],
    scratch_sv: &mut [f32],
) -> f32 {
    geometric_product_wedge_into(a, b, DIM, SHIFTS, scratch_w, scratch_su, scratch_sv);
    scratch_w[..DIM].iter().map(|x| x.abs()).sum()
}

/// Dot product between two HLA directions (alignment/similarity).
#[inline]
fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    for d in 0..DIM {
        dot += a[d] * b[d];
    }
    dot
}

/// Form a party by greedily maximizing the minimum pairwise WEDGE (diversity).
///
/// Seed with the first NPC in the pool. Then greedily add the NPC that
/// maximizes the minimum wedge to all already-selected NPCs. This produces a
/// diverse (orthogonal) party.
fn form_complementarity_party(
    pool: &[Npc],
    scratch_w: &mut [f32],
    scratch_su: &mut [f32],
    scratch_sv: &mut [f32],
) -> Vec<usize> {
    let mut selected: Vec<usize> = vec![0]; // seed with NPC 0
    while selected.len() < PARTY_SIZE {
        let mut best_npc = 0usize;
        let mut best_score = f32::MIN;
        for candidate in 0..POOL_SIZE {
            if selected.contains(&candidate) {
                continue;
            }
            // Score = min wedge to all selected NPCs (max-min diversity).
            let mut min_wedge = f32::MAX;
            for &s in &selected {
                let w = wedge_l1(
                    &pool[candidate].hla,
                    &pool[s].hla,
                    scratch_w,
                    scratch_su,
                    scratch_sv,
                );
                if w < min_wedge {
                    min_wedge = w;
                }
            }
            if min_wedge > best_score {
                best_score = min_wedge;
                best_npc = candidate;
            }
        }
        selected.push(best_npc);
    }
    selected
}

/// Form a party by greedily maximizing the minimum pairwise DOT (similarity).
///
/// Seed with the first NPC in the pool. Then greedily add the NPC that
/// maximizes the minimum dot product to all already-selected NPCs. This
/// produces a homogeneous (aligned) party.
fn form_similarity_party(pool: &[Npc]) -> Vec<usize> {
    let mut selected: Vec<usize> = vec![0]; // seed with NPC 0
    while selected.len() < PARTY_SIZE {
        let mut best_npc = 0usize;
        let mut best_score = f32::MIN;
        for candidate in 0..POOL_SIZE {
            if selected.contains(&candidate) {
                continue;
            }
            // Score = min dot to all selected NPCs (max-min similarity).
            let mut min_dot = f32::MAX;
            for &s in &selected {
                let d = dot_product(&pool[candidate].hla, &pool[s].hla);
                if d < min_dot {
                    min_dot = d;
                }
            }
            if min_dot > best_score {
                best_score = min_dot;
                best_npc = candidate;
            }
        }
        selected.push(best_npc);
    }
    selected
}

// ─── Combat simulation ─────────────────────────────────────────────────────

/// Run one combat trial. Returns the number of rounds survived.
///
/// Each round, a random threat type (0..N_ROLES) appears. If the party has a
/// member with competency ≥ ROLE_TAU in that role, the threat is neutralized.
/// Otherwise, the party takes DAMAGE_PER_THREAT damage.
fn simulate_combat(party: &[usize], pool: &[Npc], rng: &mut Rng) -> usize {
    let mut hp = START_HP;
    for round in 0..MAX_ROUNDS {
        let threat_role = rng.range(N_ROLES);
        // Check if any party member can neutralize this threat type.
        let mut neutralized = false;
        for &npc_idx in party {
            if pool[npc_idx].roles[threat_role] >= ROLE_TAU {
                neutralized = true;
                break;
            }
        }
        if !neutralized {
            hp -= DAMAGE_PER_THREAT;
            if hp <= 0.0 {
                return round + 1;
            }
        }
    }
    MAX_ROUNDS
}

// ─── Main ──────────────────────────────────────────────────────────────────

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Plan 319 — G8c: Formation Robustness Simulation            ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!(
        "  Pool: {} NPCs (80% specialists, 20% generalists), D={}",
        POOL_SIZE, DIM
    );
    println!(
        "  Party: {} NPCs, {} roles (Tank/Healer/DPS/Support), role_τ={}",
        PARTY_SIZE, N_ROLES, ROLE_TAU
    );
    println!(
        "  Combat: {} rounds max, HP={}, damage={}/threat",
        MAX_ROUNDS, START_HP, DAMAGE_PER_THREAT
    );
    println!(
        "  Trials: {}, target: complementarity survival ≥ {:.0}% of similarity",
        N_TRIALS,
        SURVIVAL_RATIO_TARGET * 100.0
    );
    println!();

    // Build the NPC pool once (same pool for both formation strategies).
    let pool = generate_pool(0x68C_F00D); // deterministic seed

    // Scratch buffers for wedge computation.
    let mut scratch_w = vec![0.0f32; DIM];
    let mut scratch_su = vec![0.0f32; DIM];
    let mut scratch_sv = vec![0.0f32; DIM];

    // Form both parties from the same pool.
    let comp_party =
        form_complementarity_party(&pool, &mut scratch_w, &mut scratch_su, &mut scratch_sv);
    let sim_party = form_similarity_party(&pool);

    // Report party compositions.
    println!("── Party Composition ──");
    for (label, party) in [
        ("Complementarity (wedge)", &comp_party),
        ("Similarity (dot)", &sim_party),
    ] {
        let role_counts: Vec<usize> = (0..N_ROLES)
            .map(|r| {
                party
                    .iter()
                    .filter(|&&i| pool[i].roles[r] >= ROLE_TAU)
                    .count()
            })
            .collect();
        let total_coverage: usize = role_counts.iter().filter(|&&c| c > 0).count();
        println!(
            "  {}: roles covered = {}/{}  counts={:?}",
            label, total_coverage, N_ROLES, role_counts
        );
        for (i, &npc_idx) in party.iter().enumerate() {
            let dominant_role = pool[npc_idx]
                .roles
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(r, _)| r)
                .unwrap_or(0);
            let role_names = ["Tank", "Heal", "DPS", "Supp"];
            println!(
                "    NPC {}: dominant={}, competencies={:?}",
                i, role_names[dominant_role], pool[npc_idx].roles
            );
        }
    }
    println!();

    // Run combat trials.
    let start = Instant::now();
    let mut comp_survivals = Vec::with_capacity(N_TRIALS);
    let mut sim_survivals = Vec::with_capacity(N_TRIALS);
    let mut trial_rng = Rng::new(0x00C0_BA70); // "combat" mnemonic (valid hex)
    for _ in 0..N_TRIALS {
        comp_survivals.push(simulate_combat(&comp_party, &pool, &mut trial_rng));
        sim_survivals.push(simulate_combat(&sim_party, &pool, &mut trial_rng));
    }
    let elapsed = start.elapsed();

    // Statistics.
    let comp_mean = comp_survivals.iter().sum::<usize>() as f64 / N_TRIALS as f64;
    let sim_mean = sim_survivals.iter().sum::<usize>() as f64 / N_TRIALS as f64;
    let ratio = comp_mean / sim_mean.max(1.0);
    comp_survivals.sort_unstable();
    sim_survivals.sort_unstable();
    let comp_median = comp_survivals[N_TRIALS / 2];
    let sim_median = sim_survivals[N_TRIALS / 2];

    println!(
        "── Results ({} trials, {:.1} ms) ──",
        N_TRIALS,
        elapsed.as_secs_f64() * 1.0e3
    );
    println!(
        "  Complementarity (wedge): mean={:.1} rounds, median={}",
        comp_mean, comp_median
    );
    println!(
        "  Similarity (dot):        mean={:.1} rounds, median={}",
        sim_mean, sim_median
    );
    println!(
        "  Survival ratio:          {:.3}×  (target ≥ {:.2}×)  {}",
        ratio,
        SURVIVAL_RATIO_TARGET,
        if ratio >= SURVIVAL_RATIO_TARGET as f64 {
            "✓ PASS"
        } else {
            "✗ FAIL"
        }
    );
    println!();

    // Verdict.
    let pass = ratio >= SURVIVAL_RATIO_TARGET as f64;
    println!("════════════════════════════════════════════════════════════════");
    println!(
        "  G8c VERDICT:  {}  (ratio {:.3}× vs target {:.2}×)",
        if pass { "PASS" } else { "FAIL" },
        ratio,
        SURVIVAL_RATIO_TARGET
    );
    if pass {
        println!(
            "  → Complementarity-weighted parties survive {:.1}% longer than",
            (ratio - 1.0) * 100.0
        );
        println!("    similarity-weighted parties under varied threats.");
        println!("  → The Clifford wedge selects diverse-role parties that cover");
        println!("    more threat types → higher survival.");
    } else {
        println!(
            "  → FAIL: complementarity advantage ({:.1}%) below {:.0}% threshold.",
            (ratio - 1.0) * 100.0,
            (SURVIVAL_RATIO_TARGET - 1.0) * 100.0
        );
        println!("  → The minimal sim may not capture emergent dynamics. A full");
        println!("    validation requires the riir-games encounter simulation.");
    }
    println!("════════════════════════════════════════════════════════════════");

    if !pass {
        std::process::exit(1);
    }
}
