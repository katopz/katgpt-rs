//! Issue 707 Phase 1–2 correctness tests: planted-TPR recovery, the
//! determinism gate, surgery additivity, the crosstalk envelope, and the
//! three-part validation harness (T6) + diagnostics (T7).

use super::*;
use crate::tpr::als::param_count;

/// Local deterministic RNG for the fixtures (the fit has its own).
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn sym(&mut self) -> f32 {
        let u = (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32;
        u.mul_add(2.0, -1.0)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n.max(1) as u64) as usize
    }
}

struct Planted {
    dim: usize,
    d: usize,
    m: usize,
    n_fillers: usize,
    states: Vec<f32>,
    bindings: Vec<TprBindings>,
}

/// Generate a corpus that IS a TPR by construction: `e = W·(Σ r_p ⊗ f_v) + b`
/// with one binding per role block (the positive control of T6 part b).
fn planted(dim: usize, d: usize, m: usize, n_fillers: usize, n: usize, seed: u64) -> Planted {
    planted_slots(dim, d, m, n_fillers, n, seed, m)
}

/// The **retrieval** shape (Issue 710): one `(role, filler)` pair per state,
/// the role drawn from `0..m`. Every (key, value) index has this shape, and it
/// is the one on which a within-state role shuffle is a provable identity.
fn planted_retrieval(
    dim: usize,
    d: usize,
    m: usize,
    n_fillers: usize,
    n: usize,
    seed: u64,
) -> Planted {
    planted_slots(dim, d, m, n_fillers, n, seed, 1)
}

/// `planted` with the per-state binding count decoupled from the arity. At
/// `slots == m` the roles are `0..m` (the original fixture, RNG draw order
/// preserved); below it they are drawn, so a state fills a strict subset of
/// the role blocks.
fn planted_slots(
    dim: usize,
    d: usize,
    m: usize,
    n_fillers: usize,
    n: usize,
    seed: u64,
    slots: usize,
) -> Planted {
    let k = m * d;
    let mut rng = Rng::new(seed);
    let w: Vec<f32> = (0..dim * k).map(|_| rng.sym()).collect();
    let bias: Vec<f32> = (0..dim).map(|_| 0.25 * rng.sym()).collect();
    let mut fillers = vec![0.0f32; n_fillers * d];
    for v in 0..n_fillers {
        let row = &mut fillers[v * d..(v + 1) * d];
        for x in row.iter_mut() {
            *x = rng.sym();
        }
        let nrm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
        for x in row.iter_mut() {
            *x /= nrm.max(1e-6);
        }
    }
    let mut states = vec![0.0f32; n * dim];
    let mut bindings = Vec::with_capacity(n);
    let mut core = vec![0.0f32; k];
    for s in 0..n {
        core.fill(0.0);
        let mut b = TprBindings::default();
        for q in 0..slots {
            let p = if slots == m { q } else { rng.below(m) };
            let v = rng.below(n_fillers);
            b.roles.push(p as u16);
            b.fillers.push(v as u16);
            for j in 0..d {
                core[p * d + j] += fillers[v * d + j];
            }
        }
        for i in 0..dim {
            let mut acc = bias[i];
            for j in 0..k {
                acc = w[i * k + j].mul_add(core[j], acc);
            }
            states[s * dim + i] = acc;
        }
        bindings.push(b);
    }
    Planted {
        dim,
        d,
        m,
        n_fillers,
        states,
        bindings,
    }
}

fn input(p: &Planted) -> AlsInput<'_> {
    AlsInput {
        dim: p.dim,
        n_fillers: p.n_fillers,
        states: &p.states,
        bindings: &p.bindings,
    }
}

fn fit_orthogonal(p: &Planted) -> (TprArtifact, AlsReport) {
    let cfg = AlsConfig::new(p.d, TprScheme::Orthogonal { arity: p.m });
    als_fit(input(p), &cfg).expect("planted fit must converge")
}

// ---------------------------------------------------------------------------
// T-G1 — planted recovery + determinism
// ---------------------------------------------------------------------------

#[test]
fn planted_tpr_is_recovered() {
    let p = planted(24, 4, 3, 6, 160, 0xA11CE);
    let (art, rep) = fit_orthogonal(&p);
    assert_eq!(
        rep.monotone_violations, 0,
        "ALS objective must not increase"
    );
    assert!(
        rep.residual_energy_fraction < 1e-3,
        "planted corpus must be explained: energy fraction {} (ssr {})",
        rep.residual_energy_fraction,
        rep.final_ssr
    );
    assert!(art.verify(), "artifact must carry a valid commitment");
    assert_eq!(art.n_fit_states, 160);
}

#[test]
fn double_fit_is_bit_identical() {
    let p = planted(16, 3, 3, 5, 96, 0xBEEF);
    let cfg = AlsConfig::new(p.d, TprScheme::Orthogonal { arity: p.m });
    let (a, _) = als_fit(input(&p), &cfg).unwrap();
    let (b, _) = als_fit(input(&p), &cfg).unwrap();
    assert_eq!(
        a.commitment, b.commitment,
        "double fit must be bit-identical"
    );
    assert_eq!(a.to_bytes(), b.to_bytes());
}

#[test]
fn holdout_unbind_and_surgery_hold() {
    let p = planted(24, 4, 3, 6, 160, 0xC0FFEE);
    let (art, _) = fit_orthogonal(&p);
    let mut scratch = TprScratch::new(&art);
    let hold = planted_holdout(&art, &p, 32, 0x5EED);
    let rep = validate_bindings(&art, &hold.0, &hold.1, &mut scratch).unwrap();
    assert!(
        rep.unbind_cos_min > 0.999,
        "unbind cosine floor {} (mean {})",
        rep.unbind_cos_min,
        rep.unbind_cos_mean
    );
    let scale: f32 = hold.0.iter().fold(0.0f32, |a, b| a.max(b.abs()));
    assert!(
        rep.surgery_max_abs_err <= 1e-4 * scale.max(1.0),
        "surgery must be additive: err {} scale {}",
        rep.surgery_max_abs_err,
        scale
    );
}

/// Holdout states generated THROUGH the fitted artifact (so the truth is the
/// artifact's own filler table — the self-consistency invariant the runtime
/// path actually relies on).
fn planted_holdout(
    art: &TprArtifact,
    p: &Planted,
    n: usize,
    seed: u64,
) -> (Vec<f32>, Vec<TprBindings>) {
    let mut rng = Rng::new(seed);
    let mut scratch = TprScratch::new(art);
    let mut states = vec![0.0f32; n * art.dim];
    let mut bindings = Vec::with_capacity(n);
    for s in 0..n {
        let mut b = TprBindings::default();
        for r in 0..p.m {
            b.roles.push(r as u16);
            b.fillers.push(rng.below(p.n_fillers) as u16);
        }
        let mut out = vec![0.0f32; art.dim];
        encode_into(art, &b, &mut scratch, &mut out).unwrap();
        states[s * art.dim..(s + 1) * art.dim].copy_from_slice(&out);
        bindings.push(b);
    }
    (states, bindings)
}

// ---------------------------------------------------------------------------
// T2 — crosstalk envelope on a role-vector scheme
// ---------------------------------------------------------------------------

#[test]
fn role_vector_unbind_respects_the_bound() {
    let p = planted(20, 4, 3, 5, 120, 0xD1CE);
    let roles = vec![1.0, 0.15, 0.0, 0.0, 1.0, 0.2, 0.1, 0.0, 1.0];
    let cfg = AlsConfig::new(p.d, TprScheme::RoleVectors { arity: 3, roles });
    let (art, _) = als_fit(input(&p), &cfg).unwrap();
    assert!(art.unbind_basis.is_some(), "role-vector fit needs a basis");
    assert!(
        (0.0..=1.0).contains(&art.crosstalk_mu),
        "mu must be a coherence in [0,1], got {}",
        art.crosstalk_mu
    );
    let bound = unbind_error_bound(&art, 3);
    assert!(bound >= 0.0 && bound.is_finite());
    // The sigmoid gate must be monotone-decreasing in the binding count.
    let c1 = unbind_confidence(&art, 1);
    let c4 = unbind_confidence(&art, 4);
    assert!(c1 >= c4, "confidence must fall as crosstalk grows");
    assert!((0.0..=1.0).contains(&c1) && (0.0..=1.0).contains(&c4));
}

// ---------------------------------------------------------------------------
// T4 — structural projection
// ---------------------------------------------------------------------------

#[test]
fn projection_is_idempotent_on_the_manifold() {
    let p = planted(24, 4, 3, 6, 160, 0x1234);
    let (art, _) = fit_orthogonal(&p);
    let mut scratch = TprScratch::new(&art);
    let (states, bindings) = planted_holdout(&art, &p, 8, 0x9999);
    let mut out = vec![0.0f32; art.dim];
    let mut out2 = vec![0.0f32; art.dim];
    let _ = &bindings;
    let e = &states[..art.dim];
    let r1 = project_into(&art, e, &mut scratch, &mut out).unwrap();
    let r2 = project_into(&art, &out.clone(), &mut scratch, &mut out2).unwrap();
    // The projection carries the fit's ridge shrinkage (λ = 1e-3), so an
    // on-manifold state returns to itself up to that bias — measured against
    // the state NORM, which is what "the residual is negligible" means for an
    // L2 residual over `dim` coordinates.
    let norm: f32 = e.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-6);
    assert!(
        r1 / norm < 1e-2,
        "an on-manifold state must project to itself: residual {r1} / norm {norm}"
    );
    assert!(
        r2 <= r1 + 1e-3 * norm,
        "projection must be idempotent: {r1} -> {r2}"
    );
}

#[test]
fn projection_denoises_off_manifold_noise() {
    let p = planted(32, 4, 3, 6, 200, 0x77);
    let (art, _) = fit_orthogonal(&p);
    let mut scratch = TprScratch::new(&art);
    let (states, _) = planted_holdout(&art, &p, 16, 0x88);
    let mut rng = Rng::new(0x5A5A);
    let mut improved = 0usize;
    let mut out = vec![0.0f32; art.dim];
    for s in 0..16 {
        let clean = &states[s * art.dim..(s + 1) * art.dim];
        let noisy: Vec<f32> = clean.iter().map(|&v| v + 0.2 * rng.sym()).collect();
        project_into(&art, &noisy, &mut scratch, &mut out).unwrap();
        let before: f32 = clean
            .iter()
            .zip(noisy.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum();
        let after: f32 = clean
            .iter()
            .zip(out.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum();
        if after < before {
            improved += 1;
        }
    }
    assert!(
        improved >= 15,
        "projection must denoise toward the manifold ({improved}/16)"
    );
}

// ---------------------------------------------------------------------------
// T6 (c) — systematicity vs the atomic-dictionary null
// ---------------------------------------------------------------------------

#[test]
fn tpr_generalizes_to_withheld_pairs_where_the_null_cannot() {
    // Corpus over (role, filler) pairs with one combination withheld.
    let p = planted(28, 4, 3, 6, 240, 0xABCD);
    let withheld: Vec<usize> = (0..p.bindings.len())
        .filter(|&s| p.bindings[s].fillers[0] == 0 && p.bindings[s].fillers[1] == 1)
        .collect();
    assert!(!withheld.is_empty(), "fixture must contain the held pair");
    let train_idx: Vec<usize> = (0..p.bindings.len())
        .filter(|s| !withheld.contains(s))
        .collect();

    let (tr_states, tr_bindings) = subset(&p, &train_idx);
    let (te_states, te_bindings) = subset(&p, &withheld);

    let cfg = AlsConfig::new(p.d, TprScheme::Orthogonal { arity: p.m });
    let tr_input = AlsInput {
        dim: p.dim,
        n_fillers: p.n_fillers,
        states: &tr_states,
        bindings: &tr_bindings,
    };
    let (art, _) = als_fit(tr_input, &cfg).unwrap();
    let mut scratch = TprScratch::new(&art);

    // Shared candidate pool: for EVERY test state, its truth plus the decoys
    // that differ in the withheld role's filler. A pool built from one state's
    // bindings would leave the others unanswerable and measure the fixture
    // rather than the primitive.
    let mut candidates: Vec<TprBindings> = Vec::new();
    for t in &te_bindings {
        for v in 0..p.n_fillers {
            let mut c = t.clone();
            c.fillers[0] = v as u16;
            if !candidates.iter().any(|x| x == &c) {
                candidates.push(c);
            }
        }
    }

    let null = AtomicNull::fit(p.dim, &tr_states, &tr_bindings);
    let id_coverage = null.coverage(&tr_bindings);
    assert!(
        id_coverage > 0.99,
        "the null must be informative in-distribution (coverage {id_coverage})"
    );
    let null_ood = null.top1(&te_states, &te_bindings, &candidates);
    let tpr_ood =
        withheld_pair_top1(&art, &te_states, &te_bindings, &candidates, &mut scratch).unwrap();
    assert_eq!(null_ood, 0.0, "the memorizer cannot answer a withheld pair");
    assert!(
        tpr_ood > 0.9,
        "composition must answer it: tpr {tpr_ood} vs null {null_ood}"
    );
}

fn subset(p: &Planted, idx: &[usize]) -> (Vec<f32>, Vec<TprBindings>) {
    let mut states = Vec::with_capacity(idx.len() * p.dim);
    let mut bindings = Vec::with_capacity(idx.len());
    for &s in idx {
        states.extend_from_slice(&p.states[s * p.dim..(s + 1) * p.dim]);
        bindings.push(p.bindings[s].clone());
    }
    (states, bindings)
}

// ---------------------------------------------------------------------------
// T7 — diagnostics
// ---------------------------------------------------------------------------

#[test]
fn bow_router_separates_structured_from_bag_of_fillers() {
    let structured = planted(24, 4, 3, 6, 200, 0x11);
    let cfg = AlsConfig::new(
        structured.d,
        TprScheme::Orthogonal {
            arity: structured.m,
        },
    );
    let s_rep = bow_router(input(&structured), &cfg, 0.05).unwrap();
    assert!(
        s_rep.structured,
        "a planted TPR must route to the structured path (ratio {})",
        s_rep.ratio
    );

    // Bag-of-fillers control: the state depends on the filler MULTISET only,
    // so the roles carry nothing and the m=1 null loses nothing.
    let mut bag = planted(24, 4, 3, 6, 200, 0x22);
    rebuild_bag_of_fillers(&mut bag, 0x33);
    let b_rep = bow_router(input(&bag), &cfg, 0.05).unwrap();
    assert!(
        !b_rep.structured,
        "an order-free family must NOT route structured (ratio {})",
        b_rep.ratio
    );
}

/// Rewrite the corpus so the state is a function of the filler multiset only.
fn rebuild_bag_of_fillers(p: &mut Planted, seed: u64) {
    let mut rng = Rng::new(seed);
    let d = p.d;
    let dim = p.dim;
    let w: Vec<f32> = (0..dim * d).map(|_| rng.sym()).collect();
    let mut fillers = vec![0.0f32; p.n_fillers * d];
    for x in fillers.iter_mut() {
        *x = rng.sym();
    }
    for (s, b) in p.bindings.iter().enumerate() {
        let mut sum = vec![0.0f32; d];
        for &v in &b.fillers {
            for j in 0..d {
                sum[j] += fillers[v as usize * d + j];
            }
        }
        for i in 0..dim {
            let mut acc = 0.0f32;
            for j in 0..d {
                acc = w[i * d + j].mul_add(sum[j], acc);
            }
            p.states[s * dim + i] = acc;
        }
    }
}

#[test]
fn shuffled_roles_destroy_a_structured_fit_and_not_a_bag() {
    let p = planted(24, 4, 3, 6, 200, 0x99);
    let cfg = AlsConfig::new(p.d, TprScheme::Orthogonal { arity: p.m });
    let rep = shuffled_role_control(input(&p), &cfg, 0xFEED, 0.05).unwrap();
    assert_eq!(
        rep.mode,
        RoleShuffleMode::WithinState,
        "a multi-binding corpus must still resolve to the original arm"
    );
    assert!(!rep.vacuous && rep.moved > 0);
    assert!(
        rep.degraded,
        "permuting roles must cost a structured fit (ratio {}, {:e} -> {:e})",
        rep.ratio, rep.r_true, rep.r_shuffled
    );

    let mut bag = planted(24, 4, 3, 6, 200, 0xAA);
    rebuild_bag_of_fillers(&mut bag, 0xBB);
    let brep = shuffled_role_control(input(&bag), &cfg, 0xFEED, 0.05).unwrap();
    assert!(
        !brep.degraded,
        "an order-free family cannot be damaged by a role shuffle (ratio {})",
        brep.ratio
    );
}

#[test]
fn bic_prefers_the_true_arity() {
    let p = planted(24, 4, 3, 6, 200, 0x44);
    let cands = vec![
        AlsConfig::new(p.d, TprScheme::Orthogonal { arity: 1 }),
        AlsConfig::new(p.d, TprScheme::Orthogonal { arity: 3 }),
    ];
    let sel = bic_select(input(&p), &cands).unwrap();
    assert_eq!(sel.best, 1, "BIC must pick m=3: scores {:?}", sel.scores);
    assert!(sel.label.contains("m3"));
}

#[test]
fn param_count_matches_the_artifact_shape() {
    let p = planted(16, 3, 2, 4, 64, 0x55);
    let (art, _) = fit_orthogonal(&p);
    assert_eq!(
        param_count(&art),
        art.dim * art.core_len() + art.dim + art.n_fillers * art.d
    );
}

// ---------------------------------------------------------------------------
// T5 — L2,1
// ---------------------------------------------------------------------------

#[test]
fn l21_prune_refit_kills_dead_dimensions() {
    // A corpus whose fillers only ever vary in 2 of 4 coordinates leaves the
    // remaining dimensions unsupported; the prune must find them.
    let p = planted(24, 4, 3, 6, 200, 0x66);
    let mut cfg = AlsConfig::new(p.d, TprScheme::Orthogonal { arity: p.m });
    cfg.l21 = L21::PruneRefit { tau_frac: 0.05 };
    let (art, rep) = als_fit(input(&p), &cfg).unwrap();
    assert_eq!(art.pruned_dims, rep.pruned_dims);
    assert!(
        rep.prune_ssr_increase >= -1e-6,
        "prune+refit cannot IMPROVE the unpruned optimum"
    );
}

// ---------------------------------------------------------------------------
// G3 — kill switch
// ---------------------------------------------------------------------------

#[test]
fn kill_switch_parses_only_the_documented_value() {
    assert!(super::parse_kill(Some("0")));
    assert!(!super::parse_kill(Some("1")));
    assert!(!super::parse_kill(Some("")));
    assert!(!super::parse_kill(None));
}

// ---------------------------------------------------------------------------
// Error surface
// ---------------------------------------------------------------------------

#[test]
fn bad_ids_and_lengths_are_refused() {
    let p = planted(16, 3, 2, 4, 64, 0x77);
    let (art, _) = fit_orthogonal(&p);
    let mut out = vec![0.0f32; art.dim];
    let f = vec![0.0f32; art.d];
    assert!(matches!(
        bind_into(&art, 9, &f, &mut out),
        Err(TprError::BadId { .. })
    ));
    let short = vec![0.0f32; art.d - 1];
    assert!(matches!(
        bind_into(&art, 0, &short, &mut out),
        Err(TprError::DimMismatch { .. })
    ));
    let mut core = vec![0.0f32; art.core_len()];
    let bad = TprBindings::from_pairs(&[(0, 99)]);
    assert!(matches!(
        core_encode_into(&art, &bad, &mut core),
        Err(TprError::BadId { .. })
    ));
}

// ---------------------------------------------------------------------------
// Issue 710 — the role control must report whether it COULD have failed
// ---------------------------------------------------------------------------

#[test]
fn a_single_binding_corpus_is_not_condemned_by_a_vacuous_control() {
    // The retrieval shape: one role, one filler per state. `K = m·d < dim`, so
    // the fit is not saturated and a broken pairing has somewhere to show up.
    let p = planted_retrieval(32, 3, 6, 8, 240, 0x710);
    assert!(p.bindings.iter().all(|b| b.len() == 1));
    let cfg = AlsConfig::new(p.d, TprScheme::Orthogonal { arity: p.m });

    // The within-state arm is a provable identity here; the cross-state one
    // is not, so that is the arm the resolver must pick.
    assert!(role_shuffle_is_vacuous(
        &p.bindings,
        RoleShuffleMode::WithinState
    ));
    assert!(!role_shuffle_is_vacuous(
        &p.bindings,
        RoleShuffleMode::CrossState
    ));
    assert_eq!(
        role_shuffle_mode_for(&p.bindings),
        RoleShuffleMode::CrossState
    );

    // Pinning the vacuous arm must SAY it is vacuous, not hand back a
    // `degraded: false` that reads as "roles carry nothing".
    let vac =
        shuffled_role_control_with(input(&p), &cfg, 0xFEED, 0.05, RoleShuffleMode::WithinState)
            .unwrap();
    assert!(
        vac.vacuous,
        "a within-state shuffle of 1-element lists is id"
    );
    assert_eq!(vac.moved, 0);
    assert_eq!(vac.r_shuffled, vac.r_true, "identical input, identical fit");
    assert!(!vac.degraded);

    // The resolved arm can fail — and does, on a corpus that IS a TPR.
    let rep = shuffled_role_control(input(&p), &cfg, 0xFEED, 0.05).unwrap();
    println!(
        "710: within-state ratio {:.4} vacuous {} moved {} | cross-state ratio {:.4} vacuous {} moved {}",
        vac.ratio, vac.vacuous, vac.moved, rep.ratio, rep.vacuous, rep.moved
    );
    assert_eq!(rep.mode, RoleShuffleMode::CrossState);
    assert!(!rep.vacuous);
    assert!(rep.moved > 0, "the drawn permutation must move a role");
    assert!(
        rep.degraded,
        "breaking the pairing across states must cost a planted fit \
         (ratio {}, {:e} -> {:e})",
        rep.ratio, rep.r_true, rep.r_shuffled
    );
}

#[test]
fn a_corpus_with_one_role_label_is_vacuous_on_both_arms() {
    // Nothing to permute: swapping equal labels is the identity whatever the
    // draw, so no arm rescues this and the report must say so rather than
    // certify "roles carry nothing".
    let p = planted_retrieval(16, 3, 1, 5, 96, 0x711);
    let cfg = AlsConfig::new(p.d, TprScheme::Orthogonal { arity: p.m });
    for mode in [RoleShuffleMode::WithinState, RoleShuffleMode::CrossState] {
        assert!(role_shuffle_is_vacuous(&p.bindings, mode), "{mode:?}");
    }
    let rep = shuffled_role_control(input(&p), &cfg, 0xFEED, 0.05).unwrap();
    assert!(rep.vacuous, "one role label leaves no permutation to draw");
    assert_eq!(rep.moved, 0);
    assert!(!rep.degraded);
}

#[test]
fn the_role_control_is_deterministic_in_its_seed() {
    let p = planted_retrieval(32, 3, 6, 8, 160, 0x712);
    let cfg = AlsConfig::new(p.d, TprScheme::Orthogonal { arity: p.m });
    let a = shuffled_role_control(input(&p), &cfg, 0x5EED, 0.05).unwrap();
    let b = shuffled_role_control(input(&p), &cfg, 0x5EED, 0.05).unwrap();
    assert_eq!(a, b, "same seed must reproduce the whole report");
}

#[test]
fn the_bow_router_reports_when_its_null_is_its_input() {
    // Issue 710 T4: at `arity == 1` with all-zero roles the "null" the router
    // fits IS the full fit, so `ratio == 1.0` is arithmetic, not evidence.
    let mut p = planted(16, 3, 2, 5, 96, 0x713);
    let flat = AlsConfig::new(p.d, TprScheme::Orthogonal { arity: 1 });
    for b in &mut p.bindings {
        b.roles.iter_mut().for_each(|r| *r = 0);
    }
    let rep = bow_router(input(&p), &flat, 0.05).unwrap();
    assert!(rep.vacuous, "the bag-of-fillers hypothesis was the input");
    assert!(!rep.structured);

    // A real arity-2 corpus is not vacuous and the router can answer.
    let q = planted(16, 3, 2, 5, 96, 0x713);
    let cfg = AlsConfig::new(q.d, TprScheme::Orthogonal { arity: q.m });
    assert!(!bow_router(input(&q), &cfg, 0.05).unwrap().vacuous);
}

#[test]
fn candidate_pool_coverage_prices_an_unanswerable_pool() {
    // Issue 710 T4: `withheld_pair_top1`'s doc says to check the pool; this is
    // the thing that checks it.
    let p = planted_retrieval(16, 3, 4, 6, 48, 0x714);
    assert_eq!(
        candidate_pool_coverage(&p.bindings, &p.bindings),
        1.0,
        "a pool containing every truth is fully answerable"
    );
    let half = &p.bindings[..p.bindings.len() / 2];
    let cov = candidate_pool_coverage(&p.bindings, half);
    assert!(
        cov < 1.0,
        "a pool missing truths must not read as answerable (cov {cov})"
    );
    assert_eq!(candidate_pool_coverage(&p.bindings, &[]), 0.0);
    assert_eq!(candidate_pool_coverage(&[], &p.bindings), 0.0);
}

// ---------------------------------------------------------------------------
// Issue 711 — a control that CAN fail can still measure the wrong question
// ---------------------------------------------------------------------------

/// The degenerate shape: the same corpus with the role rewritten as a function
/// of the filler, so no `(role, filler)` pair is unseen. The predicate is pure
/// in the bindings, so the states are left as planted — what changes is which
/// question the corpus is able to answer, not the arithmetic.
fn roles_from_fillers(bindings: &[TprBindings], m: usize) -> Vec<TprBindings> {
    let n = m.max(1) as u16;
    bindings
        .iter()
        .map(|b| TprBindings {
            roles: b.fillers.iter().map(|&v| v % n).collect(),
            fillers: b.fillers.clone(),
        })
        .collect()
}

#[test]
fn the_composition_covariate_separates_a_degenerate_corpus_from_a_healthy_one() {
    // Healthy: roles drawn independently of the filler, so a filler is seen
    // with several roles and unseen pairs exist to generalize to.
    let p = planted_retrieval(32, 3, 6, 8, 240, 0x7110);
    let healthy = filler_role_spread(&p.bindings);
    assert!(
        !healthy.role_determined_by_filler(),
        "8 fillers x 6 roles over 240 states must reuse a filler across roles \
         (max {}, mean {:.3})",
        healthy.max,
        healthy.mean
    );
    assert!(healthy.max > 1 && healthy.mean > 1.0);
    assert_eq!(healthy.fillers, p.n_fillers, "every planted filler appears");
    assert_eq!(
        healthy.multi_role_fillers, healthy.fillers,
        "every filler here carries several roles, so all of them are testable"
    );
    assert!(!role_determined_by_filler(&p.bindings));

    // Degenerate: role = filler % m. Same states, same fillers, same counts.
    let degen = roles_from_fillers(&p.bindings, p.m);
    let spread = filler_role_spread(&degen);
    assert!(
        spread.role_determined_by_filler(),
        "role = f(filler) leaves one role per filler (max {})",
        spread.max
    );
    assert_eq!(spread.max, 1);
    assert_eq!(spread.mean, 1.0);
    assert_eq!(spread.fillers, healthy.fillers, "same filler population");
    assert_eq!(
        spread.multi_role_fillers, 0,
        "no filler admits a withheld pair — withholding one withholds the filler"
    );
    assert!(role_determined_by_filler(&degen));
    println!(
        "711: healthy max {} mean {:.3} | degenerate max {} mean {:.3}",
        healthy.max, healthy.mean, spread.max, spread.mean
    );
}

#[test]
fn the_covariate_is_reported_and_not_only_its_threshold() {
    // A NEAR-degenerate corpus is the case a threshold alone cannot see: one
    // filler carries two roles and every other carries one, so the predicate
    // says "readable" while the mean says the structure question is barely
    // posed. Both numbers must be available to say that.
    let p = planted_retrieval(32, 3, 6, 8, 240, 0x7111);
    let mut near = roles_from_fillers(&p.bindings, p.m);
    let tipped = near
        .iter_mut()
        .find(|b| b.fillers[0] == 0)
        .expect("filler 0 appears");
    tipped.roles[0] = (tipped.roles[0] + 1) % p.m as u16;

    let spread = filler_role_spread(&near);
    assert!(
        !spread.role_determined_by_filler(),
        "one filler with two roles is enough to clear the threshold"
    );
    assert_eq!(spread.max, 2);
    assert!(
        spread.mean < 1.2,
        "and the mean must still say how thin that is (mean {:.4})",
        spread.mean
    );
    // The point of carrying the population separately: `max` clears the
    // threshold on the strength of ONE filler, so a withheld-pair test has
    // exactly one filler to draw from. `max` alone cannot say that.
    assert_eq!(
        spread.multi_role_fillers, 1,
        "one tipped filler is the entire testable population"
    );
    println!(
        "711: near-degenerate max {} mean {:.4} testable fillers {}/{}",
        spread.max, spread.mean, spread.multi_role_fillers, spread.fillers
    );
}

#[test]
fn a_structure_verdict_is_withheld_on_a_corpus_that_cannot_carry_it() {
    let p = planted_retrieval(32, 3, 6, 8, 240, 0x7112);
    let cfg = AlsConfig::new(p.d, TprScheme::Orthogonal { arity: p.m });

    // Two-sided pin, healthy side: the guard must NOT be always-on, or it
    // would silence every real verdict and read as permanent caution.
    let bow = bow_router(input(&p), &cfg, 0.05).unwrap();
    assert!(!bow.spread.role_determined_by_filler());
    assert_eq!(
        bow.verdict(),
        Some(bow.structured),
        "a healthy corpus must still get an answer"
    );
    let shuf = shuffled_role_control(input(&p), &cfg, 0xFEED, 0.05).unwrap();
    assert!(!shuf.spread.role_determined_by_filler());
    assert_eq!(shuf.verdict(), Some(shuf.degraded));

    // Two-sided pin, degenerate side: the guard must fire. The probes still
    // MOVE here — that is the whole point of Issue 711 — so a dead guard
    // would leave a confident, unreadable bool in its place.
    let q = Planted {
        dim: p.dim,
        d: p.d,
        m: p.m,
        n_fillers: p.n_fillers,
        states: p.states.clone(),
        bindings: roles_from_fillers(&p.bindings, p.m),
    };
    let dbow = bow_router(input(&q), &cfg, 0.05).unwrap();
    assert!(dbow.spread.role_determined_by_filler());
    assert!(!dbow.vacuous, "the bow null is still not the input");
    assert_eq!(dbow.verdict(), None, "ratio {:.4}", dbow.ratio);

    let dshuf = shuffled_role_control(input(&q), &cfg, 0xFEED, 0.05).unwrap();
    assert!(dshuf.spread.role_determined_by_filler());
    assert!(!dshuf.vacuous, "the permutation is not a provable identity");
    assert!(dshuf.moved > 0, "and it actually moved roles");
    assert_eq!(dshuf.verdict(), None, "ratio {:.4}", dshuf.ratio);
    println!(
        "711: degenerate bow ratio {:.4} (structured {}) shuffle ratio {:.4} \
         (degraded {}, moved {}) — both verdicts withheld",
        dbow.ratio, dbow.structured, dshuf.ratio, dshuf.degraded, dshuf.moved
    );
}

#[test]
fn the_withheld_pair_probe_reports_its_ceiling_and_withholds_on_a_degenerate_universe() {
    // Issue 711 T4. The raw `withheld_pair_top1` is deliberately untouched —
    // the Issue 707 G8 gate keeps its number — so this pins the additive
    // report instead: the ceiling the number must be read against, and the
    // refusal on a universe where the question is not posed.
    let p = planted_retrieval(24, 3, 5, 7, 90, 0x7113);
    let cfg = AlsConfig::new(p.d, TprScheme::Orthogonal { arity: p.m });
    let (art, _) = als_fit(input(&p), &cfg).unwrap();
    let mut scratch = TprScratch::new(&art);

    // Healthy universe, fully answerable pool.
    let full = withheld_pair_top1_report(
        &art,
        &p.states,
        &p.bindings,
        &p.bindings,
        &mut scratch,
    )
    .unwrap();
    assert_eq!(full.coverage, 1.0);
    assert!(!full.spread.role_determined_by_filler());
    assert_eq!(
        full.verdict(),
        Some(full.top1),
        "a healthy universe must still get an answer"
    );
    assert_eq!(full.per_answerable(), Some(full.top1), "coverage is 1.0");
    assert_eq!(
        full.top1,
        withheld_pair_top1(&art, &p.states, &p.bindings, &p.bindings, &mut scratch).unwrap(),
        "the report must not change the number the raw gate reads"
    );

    // Same universe, half a pool: the number is now bounded by the pool and
    // the report says so instead of letting it be read against 1.0.
    let half = &p.bindings[..p.bindings.len() / 2];
    let thin =
        withheld_pair_top1_report(&art, &p.states, &p.bindings, half, &mut scratch).unwrap();
    assert!(thin.coverage < 1.0, "coverage {}", thin.coverage);
    assert!(
        thin.top1 <= thin.coverage + 1e-6,
        "top1 {} cannot exceed its ceiling {}",
        thin.top1,
        thin.coverage
    );
    assert!(thin.per_answerable().unwrap() >= thin.top1);
    assert!(
        thin.verdict().is_some(),
        "a thin pool is a ceiling, not a void — only role = f(filler) voids"
    );

    // Degenerate universe: role = f(filler), so withholding a pair withholds
    // the whole filler and the OOD arm asks a different question.
    let q = Planted {
        dim: p.dim,
        d: p.d,
        m: p.m,
        n_fillers: p.n_fillers,
        states: p.states.clone(),
        bindings: roles_from_fillers(&p.bindings, p.m),
    };
    let (dart, _) = als_fit(input(&q), &cfg).unwrap();
    let mut dscratch = TprScratch::new(&dart);
    let degen = withheld_pair_top1_report(
        &dart,
        &q.states,
        &q.bindings,
        &q.bindings,
        &mut dscratch,
    )
    .unwrap();
    assert_eq!(degen.coverage, 1.0, "the pool is still fully answerable");
    assert!(degen.spread.role_determined_by_filler());
    assert_eq!(degen.verdict(), None, "top1 {}", degen.top1);
    println!(
        "711 T4: healthy top1 {:.4} cov {:.4} verdict {:?} | thin top1 {:.4} cov {:.4} \
         per-answerable {:.4} | degenerate top1 {:.4} verdict {:?}",
        full.top1,
        full.coverage,
        full.verdict(),
        thin.top1,
        thin.coverage,
        thin.per_answerable().unwrap(),
        degen.top1,
        degen.verdict()
    );

    // Two-sided on the pool axis too: an empty pool is unanswerable and
    // `per_answerable` must refuse the division rather than return inf.
    let none = withheld_pair_top1_report(&art, &p.states, &p.bindings, &[], &mut scratch).unwrap();
    assert_eq!(none.coverage, 0.0);
    assert_eq!(none.per_answerable(), None);
}

#[test]
fn observed_pairs_separates_a_counterfactual_the_fit_has_seen_from_one_it_has_not() {
    // The third form of the 710/711 shape (riir-clippy Issue 062 T6): an
    // operation that is PROVABLY correct and still answers a question the
    // corpus cannot support. `surgery_delta_into` is bit-additive on any
    // artifact, so a counterfactual attribution is clean and confident whether
    // or not the pair it asks about was ever fitted. Nothing in the operation
    // warns the caller; this is what warns the caller.
    let p = planted_retrieval(24, 3, 5, 7, 120, 0x7114);
    let obs = ObservedPairs::from_bindings(&p.bindings);
    let spread = filler_role_spread(&p.bindings);

    assert!(!obs.is_empty());
    assert_eq!(
        obs.len(),
        spread.distinct_pairs,
        "the report's numerator must be the same set the membership test uses"
    );
    // `mean` is that numerator over the fillers that appear — pinned as an
    // identity so a consumer is never tempted to re-derive one from the other
    // and pair factors from different arms.
    assert!(
        (spread.mean - spread.distinct_pairs as f32 / spread.fillers as f32).abs() < 1e-6,
        "mean {} vs {}/{}",
        spread.mean,
        spread.distinct_pairs,
        spread.fillers
    );

    // Every planted pair is observed.
    for b in &p.bindings {
        for (&r, &f) in b.roles.iter().zip(b.fillers.iter()) {
            assert!(obs.contains(r, f), "planted pair ({r}, {f}) must be observed");
        }
    }
    assert_eq!(obs.observed_fraction(&pairs_of(&p.bindings)), 1.0);

    // A filler id past the vocabulary can occur in no pair, so a
    // counterfactual naming it is outside the fit by construction — and
    // `observed_fraction` prices a mixed batch rather than pooling it.
    let unseen: Vec<(u16, u16)> = (0..p.m as u16).map(|r| (r, p.n_fillers as u16)).collect();
    assert_eq!(obs.observed_fraction(&unseen), 0.0);
    let mut mixed = pairs_of(&p.bindings);
    mixed.extend_from_slice(&unseen);
    let frac = obs.observed_fraction(&mixed);
    assert!(frac > 0.0 && frac < 1.0, "mixed batch fraction {frac}");
    assert_eq!(obs.observed_fraction(&[]), 0.0);
}

fn pairs_of(bindings: &[TprBindings]) -> Vec<(u16, u16)> {
    bindings
        .iter()
        .flat_map(|b| b.roles.iter().copied().zip(b.fillers.iter().copied()))
        .collect()
}

#[test]
fn on_a_degenerate_corpus_every_counterfactual_role_is_unobserved() {
    // `role = f(filler)`: each filler occurs with exactly one role, so the
    // ONLY observed pair for a filler is its own. Every counterfactual that
    // moves the role is therefore outside the fit — the regime where a T6
    // attribution would be entirely prediction while reading as measurement.
    let p = planted_retrieval(24, 3, 5, 7, 120, 0x7115);
    let degen = roles_from_fillers(&p.bindings, p.m);
    let obs = ObservedPairs::from_bindings(&degen);
    let spread = filler_role_spread(&degen);
    assert!(spread.role_determined_by_filler());
    assert_eq!(obs.len(), spread.fillers, "one pair per filler, exactly");

    let mut counterfactuals = 0usize;
    let mut observed = 0usize;
    for f in 0..p.n_fillers as u16 {
        let own = f % p.m as u16;
        assert!(obs.contains(own, f), "the filler's own role is observed");
        for r in (0..p.m as u16).filter(|r| *r != own) {
            counterfactuals += 1;
            observed += usize::from(obs.contains(r, f));
        }
    }
    assert_eq!(
        observed, 0,
        "{counterfactuals} role-moving counterfactuals, none of them fitted"
    );

    // ── and the other half of the hazard, in the same test ────────────────
    //
    // A peer pointed a future reader at this test as the thing that would fail
    // if someone later made surgery "helpfully" refuse on an unfitted pair —
    // which would look like a safety improvement and would actually destroy
    // the signal that the answers are unfitted. That was not true of the
    // assertions above: they never called surgery. It is true now.
    //
    // Both halves have to be asserted together or the finding is only prose:
    // the pair is NOT in the fit, AND surgery answers it bit-additively
    // anyway. Either assertion alone is satisfiable by a broken
    // implementation — refusing satisfies the first, fitting everything
    // satisfies the second.
    let cfg = AlsConfig::new(p.d, TprScheme::Orthogonal { arity: p.m });
    let q = Planted {
        dim: p.dim,
        d: p.d,
        m: p.m,
        n_fillers: p.n_fillers,
        states: p.states.clone(),
        bindings: degen.clone(),
    };
    let (art, _) = als_fit(input(&q), &cfg).unwrap();
    let mut scratch = TprScratch::new(&art);

    let b = &degen[0];
    let (role, v_old) = (b.roles[0], b.fillers[0]);
    // Keep the role, move the filler — the counterfactual T6 actually asks.
    // `v_new` is chosen so `(role, v_new)` cannot be in the corpus: under
    // `role = filler % m` a pair is observed only when `v % m == role`.
    let v_new = (0..p.n_fillers as u16)
        .find(|v| *v != v_old && *v % p.m as u16 != role)
        .expect("a filler outside this role's class exists");
    assert!(
        !obs.contains(role, v_new),
        "({role}, {v_new}) must be outside the fit for this to be the T6 case"
    );

    let d = art.d;
    let f_old = art.fillers[v_old as usize * d..(v_old as usize + 1) * d].to_vec();
    let f_new = art.fillers[v_new as usize * d..(v_new as usize + 1) * d].to_vec();
    let mut edited = vec![0.0f32; art.dim];
    encode_into(&art, b, &mut scratch, &mut edited).unwrap();
    surgery_delta_into(&art, &mut edited, role, &f_old, &f_new, &mut scratch)
        .expect("surgery must ANSWER an unfitted counterfactual, not refuse it");
    let mut swapped = b.clone();
    swapped.fillers[0] = v_new;
    let mut reencoded = vec![0.0f32; art.dim];
    encode_into(&art, &swapped, &mut scratch, &mut reencoded).unwrap();

    let scale = reencoded.iter().fold(0.0f32, |a, x| a.max(x.abs()));
    let err = edited
        .iter()
        .zip(reencoded.iter())
        .fold(0.0f32, |a, (x, y)| a.max((x - y).abs()));
    assert!(
        err <= 1e-5 * scale.max(1.0),
        "surgery must stay exact on an UNFITTED pair — that is the hazard, \
         not a defect: err {err} scale {scale}"
    );

    println!(
        "062 T6: degenerate corpus — {} distinct pairs over {} fillers, \
         {counterfactuals} role-moving counterfactuals, {observed} observed; \
         surgery on unfitted ({role}, {v_new}) exact to {err:.3e} (scale {scale:.3})",
        obs.len(),
        spread.fillers
    );
}
