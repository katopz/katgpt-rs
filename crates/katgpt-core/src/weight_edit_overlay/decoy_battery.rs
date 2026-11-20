//! Two-sided decoy battery (Issue 1237 T3, Research 684 — the Fermion
//! 20-decoy discipline).
//!
//! The 980 gate battery carries ONE control arm: a norm-matched random
//! direction. That separates "the edit does something" from noise, but not
//! "the DIRECTION is good" from "the SITE is good". This battery is the
//! two-sided upgrade — per candidate [`WeightEditSpec`]:
//!
//! - **site-random arm** — the chosen direction (or fader) on random sites
//!   (same layer, same `|S|`, columns drawn disjoint from the chosen set);
//! - **direction-random arm** — norm-matched random directions on the chosen
//!   sites.
//!
//! Every arm is scored on ONE frozen held-out read: the battery takes the
//! read by shared reference ONCE and hands that same `&R` to every scorer
//! call, so a per-arm re-draw is structurally impossible from inside the
//! battery. Verdict = `chosen ≥ best_decoy + margin` with the margin
//! pre-registered in [`DecoyBatteryConfig`]; both arms' bests are reported
//! separately, never pooled (a FAIL names which arm beat the edit — that
//! IS the site-vs-direction diagnosis).
//!
//! Modelless: the battery constructs specs and calls the caller's scorer —
//! no gradients, no weight mutation of its own. Deterministic: every decoy
//! is BLAKE3-seeded from (`seed_domain`, arm, index).
//!
//! Consumer: riir-infer Issue 1241 T3 (census edit validation). The kill bar
//! is shared with that lane — if its census cell closes NEGATIVE this module
//! demotes with it.

use super::{EditForm, SplitMix64, WeightEditSpec, blake3_seed, norm_matched_random_direction};

/// Pre-registered battery parameters. Defaults pin Fermion's count (20
/// decoys, split 10 site-random + 10 direction-random).
#[derive(Clone, Debug, PartialEq)]
pub struct DecoyBatteryConfig {
    /// Site-random decoys.
    pub n_site_random: usize,
    /// Direction-random decoys.
    pub n_direction_random: usize,
    /// Pre-registered margin: PASS iff `chosen ≥ best_decoy + margin`.
    pub margin: f64,
    /// BLAKE3 seed domain for decoy draws.
    pub seed_domain: Vec<u8>,
}

/// Pinned default decoy count per arm (Fermion: 20 decoys per edit).
pub const DEFAULT_DECOYS_PER_ARM: usize = 10;

impl Default for DecoyBatteryConfig {
    fn default() -> Self {
        Self {
            n_site_random: DEFAULT_DECOYS_PER_ARM,
            n_direction_random: DEFAULT_DECOYS_PER_ARM,
            margin: 0.0,
            seed_domain: b"weight_edit_overlay/decoy_battery".to_vec(),
        }
    }
}

/// Which decoy arm.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum DecoyArm {
    /// The chosen direction/fader on random sites.
    SiteRandom = 0,
    /// Random directions on the chosen sites.
    DirectionRandom = 1,
}

/// Best decoy of one arm.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ArmBest {
    pub arm: DecoyArm,
    /// Decoys actually scored in this arm.
    pub n: usize,
    /// Best score (max) in this arm.
    pub best: f64,
    /// Index of the best decoy within its arm (first-index tie rule).
    pub best_index: usize,
}

/// Why the battery refused to produce a verdict.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecoyRefusal {
    /// The chosen spec failed validation.
    InvalidSpec,
    /// Bypass edits have no column/direction site — no decoy population.
    UnsupportedForm,
    /// A fader site-random decoy needs base columns at random sites and no
    /// reader was supplied (or it declined a column).
    NeedsBaseReader,
    /// `in_dim` cannot host `|S|` disjoint random columns.
    SiteSpaceTooSmall,
    /// A decoy arm was configured empty — a one-sided battery is the
    /// pre-1237 single control arm, not this battery.
    EmptyArm,
    /// The scorer returned a non-finite score.
    NonFiniteScore,
    /// Margin non-finite or negative.
    BadMargin,
}

/// Battery verdict.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DecoyVerdict {
    /// `chosen ≥ best_decoy + margin` against BOTH arms.
    Pass,
    /// Some decoy came within the margin; `beaten_by` names the arm whose
    /// best did (the higher one; site-random first on a tie).
    Fail { beaten_by: DecoyArm },
    /// No verdict.
    Refused(DecoyRefusal),
}

/// Full battery report — arms are reported separately, never pooled.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DecoyBatteryReport {
    pub chosen: f64,
    pub site_random: Option<ArmBest>,
    pub direction_random: Option<ArmBest>,
    pub margin: f64,
    pub verdict: DecoyVerdict,
}

impl DecoyBatteryReport {
    fn refused(r: DecoyRefusal, margin: f64) -> Self {
        Self {
            chosen: f64::NAN,
            site_random: None,
            direction_random: None,
            margin,
            verdict: DecoyVerdict::Refused(r),
        }
    }

    /// Chosen minus the best decoy across both arms (`NaN` when refused).
    pub fn separation(&self) -> f64 {
        match (self.site_random, self.direction_random) {
            (Some(s), Some(d)) => self.chosen - s.best.max(d.best),
            _ => f64::NAN,
        }
    }
}

/// Base-column reader: `(layer, column, out) -> bool` fills `out` with the
/// live base column; `false` declines.
pub type BaseColumnReader<'a> = &'a dyn Fn(u8, u32, &mut [f32]) -> bool;

/// Where decoys may be placed and how fader decoys read base columns.
pub struct DecoySites<'a> {
    /// Input width of the edited matrix — site-random columns are drawn
    /// from `[0, in_dim)` disjoint from the chosen set.
    pub in_dim: u32,
    /// Base-column reader for fader site-random decoys:
    /// `(layer, column, out) -> bool` fills `out` (`out_dim` entries) with
    /// the live base column; `false` declines. `None` is fine for
    /// direction-subtraction specs.
    pub base_column: Option<BaseColumnReader<'a>>,
}

/// Run the two-sided battery over `spec`.
///
/// `score(edits, read)` scores an edit set on the ONE frozen read; the
/// battery calls it once for the chosen spec and once per decoy, always
/// with the same `read`. Higher is better.
pub fn run_decoy_battery<R: ?Sized, F>(
    spec: &WeightEditSpec,
    sites: &DecoySites<'_>,
    cfg: &DecoyBatteryConfig,
    read: &R,
    mut score: F,
) -> DecoyBatteryReport
where
    F: FnMut(&[WeightEditSpec], &R) -> f64,
{
    let margin = cfg.margin;
    if !margin.is_finite() || margin < 0.0 {
        return DecoyBatteryReport::refused(DecoyRefusal::BadMargin, margin);
    }
    if cfg.n_site_random == 0 || cfg.n_direction_random == 0 {
        return DecoyBatteryReport::refused(DecoyRefusal::EmptyArm, margin);
    }
    if spec.validate().is_err() {
        return DecoyBatteryReport::refused(DecoyRefusal::InvalidSpec, margin);
    }
    let Some(columns) = spec.form.written_columns() else {
        return DecoyBatteryReport::refused(DecoyRefusal::UnsupportedForm, margin);
    };
    if (sites.in_dim as usize) < 2 * columns.len() {
        return DecoyBatteryReport::refused(DecoyRefusal::SiteSpaceTooSmall, margin);
    }

    let chosen = score(std::slice::from_ref(spec), read);
    if !chosen.is_finite() {
        return DecoyBatteryReport::refused(DecoyRefusal::NonFiniteScore, margin);
    }

    let mut decoy: Vec<WeightEditSpec> = Vec::with_capacity(columns.len());
    let mut arms = [
        (DecoyArm::SiteRandom, cfg.n_site_random),
        (DecoyArm::DirectionRandom, cfg.n_direction_random),
    ]
    .map(|(arm, n)| ArmBest {
        arm,
        n,
        best: f64::NEG_INFINITY,
        best_index: 0,
    });
    for arm_best in arms.iter_mut() {
        for idx in 0..arm_best.n {
            let seed = blake3_seed(&cfg.seed_domain, ((arm_best.arm as u64) << 32) | idx as u64);
            decoy.clear();
            let built = match arm_best.arm {
                DecoyArm::SiteRandom => site_random_decoy(spec, columns, sites, seed, &mut decoy),
                DecoyArm::DirectionRandom => direction_random_decoy(spec, seed, &mut decoy),
            };
            if let Err(r) = built {
                return DecoyBatteryReport::refused(r, margin);
            }
            let s = score(&decoy, read);
            if !s.is_finite() {
                return DecoyBatteryReport::refused(DecoyRefusal::NonFiniteScore, margin);
            }
            if s > arm_best.best {
                arm_best.best = s;
                arm_best.best_index = idx;
            }
        }
    }
    let [site, dir] = arms;
    let verdict = if chosen >= site.best + margin && chosen >= dir.best + margin {
        DecoyVerdict::Pass
    } else {
        DecoyVerdict::Fail {
            beaten_by: if site.best >= dir.best {
                DecoyArm::SiteRandom
            } else {
                DecoyArm::DirectionRandom
            },
        }
    };
    DecoyBatteryReport {
        chosen,
        site_random: Some(site),
        direction_random: Some(dir),
        margin,
        verdict,
    }
}

/// Draw `k` distinct columns from `[0, in_dim)` disjoint from `chosen`
/// (seeded rejection; `in_dim ≥ 2k` is checked by the caller so it
/// terminates quickly).
fn draw_disjoint_columns(chosen: &[u32], in_dim: u32, seed: u64, out: &mut Vec<u32>) {
    let mut rng = SplitMix64::new(seed);
    out.clear();
    while out.len() < chosen.len() {
        let c = (rng.next_u64() % in_dim as u64) as u32;
        if !chosen.contains(&c) && !out.contains(&c) {
            out.push(c);
        }
    }
}

fn site_random_decoy(
    spec: &WeightEditSpec,
    columns: &[u32],
    sites: &DecoySites<'_>,
    seed: u64,
    out: &mut Vec<WeightEditSpec>,
) -> Result<(), DecoyRefusal> {
    let mut cols = Vec::with_capacity(columns.len());
    draw_disjoint_columns(columns, sites.in_dim, seed, &mut cols);
    let form = match &spec.form {
        EditForm::DirectionSubtraction {
            direction,
            strength,
            ..
        } => EditForm::DirectionSubtraction {
            columns: cols,
            direction: direction.clone(),
            strength: *strength,
        },
        EditForm::ColumnScale {
            out_dim,
            scale,
            dose,
            ..
        } => {
            let reader = sites.base_column.ok_or(DecoyRefusal::NeedsBaseReader)?;
            let out_n = *out_dim as usize;
            let mut base_columns = vec![0.0f32; cols.len() * out_n];
            for (k, c) in cols.iter().enumerate() {
                if !reader(
                    spec.layer,
                    *c,
                    &mut base_columns[k * out_n..(k + 1) * out_n],
                ) {
                    return Err(DecoyRefusal::NeedsBaseReader);
                }
            }
            EditForm::ColumnScale {
                columns: cols,
                base_columns,
                out_dim: *out_dim,
                scale: *scale,
                dose: *dose,
            }
        }
        EditForm::BypassRows { .. } => return Err(DecoyRefusal::UnsupportedForm),
    };
    let d = WeightEditSpec {
        layer: spec.layer,
        form,
    };
    d.validate().map_err(|_| DecoyRefusal::NeedsBaseReader)?;
    out.push(d);
    Ok(())
}

/// Random directions on the chosen sites. For a direction edit: one
/// norm-matched random direction (`‖u‖ = ‖d‖`), same columns and strength.
/// For a fader: per chosen column `k`, a direction-subtraction spec whose
/// rank-1 column write `−strength·u` has the SAME norm as the fader's own
/// column write `(f−1)·base_k` — the random-direction arm at the fader's
/// dose magnitude.
fn direction_random_decoy(
    spec: &WeightEditSpec,
    seed: u64,
    out: &mut Vec<WeightEditSpec>,
) -> Result<(), DecoyRefusal> {
    match &spec.form {
        EditForm::DirectionSubtraction {
            columns,
            direction,
            strength,
        } => {
            let mut u = vec![0.0f32; direction.len()];
            norm_matched_random_direction(seed, direction.len(), &mut u);
            let norm = direction.iter().map(|v| v * v).sum::<f32>().sqrt();
            for v in u.iter_mut() {
                *v *= norm;
            }
            out.push(WeightEditSpec {
                layer: spec.layer,
                form: EditForm::DirectionSubtraction {
                    columns: columns.clone(),
                    direction: u,
                    strength: *strength,
                },
            });
            Ok(())
        }
        EditForm::ColumnScale {
            columns,
            base_columns,
            out_dim,
            scale,
            dose,
        } => {
            let out_n = *out_dim as usize;
            let g = EditForm::fader_factor(*scale, *dose) - 1.0;
            for (k, c) in columns.iter().enumerate() {
                let base = &base_columns[k * out_n..(k + 1) * out_n];
                let write_norm = g.abs() * base.iter().map(|v| v * v).sum::<f32>().sqrt();
                let mut u = vec![0.0f32; out_n];
                norm_matched_random_direction(
                    seed ^ (k as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
                    out_n,
                    &mut u,
                );
                out.push(WeightEditSpec {
                    layer: spec.layer,
                    form: EditForm::DirectionSubtraction {
                        columns: vec![*c],
                        direction: u,
                        strength: write_norm,
                    },
                });
            }
            Ok(())
        }
        EditForm::BypassRows { .. } => Err(DecoyRefusal::UnsupportedForm),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const OUT: usize = 8;
    const IN: usize = 32;

    fn unit(i: usize) -> Vec<f32> {
        let mut v = vec![0.0; OUT];
        v[i] = 1.0;
        v
    }

    /// Frozen read: target write direction `t` at target column `tc`. A
    /// spec set scores `Σ_specs Σ_{c∈S} [c == tc] · ⟨−strength·d, t⟩` —
    /// i.e. the write it lands on the target column, projected on `t`.
    struct Read {
        target_col: u32,
        target_dir: Vec<f32>,
        /// When true, the score ignores direction (site-only reward).
        site_only: bool,
    }

    fn score(edits: &[WeightEditSpec], r: &Read) -> f64 {
        let mut s = 0.0f64;
        for e in edits {
            if let EditForm::DirectionSubtraction {
                columns,
                direction,
                strength,
            } = &e.form
            {
                for c in columns {
                    if *c != r.target_col {
                        continue;
                    }
                    if r.site_only {
                        s += 1.0;
                    } else {
                        let dot: f32 = direction
                            .iter()
                            .zip(r.target_dir.iter())
                            .map(|(a, b)| -strength * a * b)
                            .sum();
                        s += dot as f64;
                    }
                }
            }
        }
        s
    }

    fn edit(col: u32, dir: Vec<f32>) -> WeightEditSpec {
        WeightEditSpec {
            layer: 3,
            form: EditForm::DirectionSubtraction {
                columns: vec![col],
                direction: dir,
                strength: -1.0,
            },
        }
    }

    fn sites() -> DecoySites<'static> {
        DecoySites {
            in_dim: IN as u32,
            base_column: None,
        }
    }

    #[test]
    fn param_pin_defaults() {
        let c = DecoyBatteryConfig::default();
        assert_eq!(c.n_site_random + c.n_direction_random, 20);
        assert_eq!(c.n_site_random, c.n_direction_random);
    }

    #[test]
    fn g1_true_edit_passes_both_arms() {
        let read = Read {
            target_col: 5,
            target_dir: unit(2),
            site_only: false,
        };
        let cfg = DecoyBatteryConfig {
            margin: 0.2,
            ..Default::default()
        };
        let rep = run_decoy_battery(&edit(5, unit(2)), &sites(), &cfg, &read, score);
        assert_eq!(rep.verdict, DecoyVerdict::Pass, "{rep:?}");
        assert!((rep.chosen - 1.0).abs() < 1e-6);
        assert_eq!(rep.site_random.unwrap().n, 10);
        assert_eq!(rep.direction_random.unwrap().n, 10);
        assert!(rep.separation() >= 0.2);
    }

    /// The canary: a DECOY planted as the edit must FAIL. Both shapes — a
    /// random direction on a random site, and a random direction on the
    /// good site.
    #[test]
    fn g1_canary_planted_decoy_fails() {
        let read = Read {
            target_col: 5,
            target_dir: unit(2),
            site_only: false,
        };
        let cfg = DecoyBatteryConfig {
            margin: 0.2,
            ..Default::default()
        };
        // Random direction on the chosen (good) site.
        let mut u = vec![0.0f32; OUT];
        norm_matched_random_direction(blake3_seed(b"planted", 0), OUT, &mut u);
        let rep = run_decoy_battery(&edit(5, u), &sites(), &cfg, &read, score);
        assert!(
            matches!(rep.verdict, DecoyVerdict::Fail { .. }),
            "planted random-direction decoy must fail: {rep:?}"
        );
        // The good direction on a wrong site — the site-random arm only
        // ties at best (0 vs 0) unless it hits col 5; margin refuses it.
        let rep = run_decoy_battery(&edit(9, unit(2)), &sites(), &cfg, &read, score);
        assert!(matches!(rep.verdict, DecoyVerdict::Fail { .. }), "{rep:?}");
    }

    /// Two-sidedness: when the read rewards the SITE only, the
    /// direction-random arm ties the chosen edit and the battery FAILS,
    /// naming the direction arm — "the site is good, not the direction".
    #[test]
    fn g1_site_only_reward_fails_on_direction_arm() {
        let read = Read {
            target_col: 5,
            target_dir: unit(2),
            site_only: true,
        };
        let cfg = DecoyBatteryConfig {
            margin: 0.2,
            ..Default::default()
        };
        let rep = run_decoy_battery(&edit(5, unit(2)), &sites(), &cfg, &read, score);
        assert_eq!(
            rep.verdict,
            DecoyVerdict::Fail {
                beaten_by: DecoyArm::DirectionRandom
            },
            "{rep:?}"
        );
    }

    #[test]
    fn deterministic_and_disjoint_sites() {
        let read = Read {
            target_col: 5,
            target_dir: unit(2),
            site_only: false,
        };
        let cfg = DecoyBatteryConfig::default();
        let a = run_decoy_battery(&edit(5, unit(2)), &sites(), &cfg, &read, score);
        let b = run_decoy_battery(&edit(5, unit(2)), &sites(), &cfg, &read, score);
        assert_eq!(a, b);
        // Site draws never land on the chosen column.
        let mut cols = Vec::new();
        for i in 0..200 {
            draw_disjoint_columns(&[5, 7], IN as u32, i, &mut cols);
            assert!(!cols.contains(&5) && !cols.contains(&7));
            assert_ne!(cols[0], cols[1]);
        }
    }

    #[test]
    fn refusals_are_typed() {
        let read = Read {
            target_col: 5,
            target_dir: unit(2),
            site_only: false,
        };
        let empty = DecoyBatteryConfig {
            n_site_random: 0,
            ..Default::default()
        };
        assert_eq!(
            run_decoy_battery(&edit(5, unit(2)), &sites(), &empty, &read, score).verdict,
            DecoyVerdict::Refused(DecoyRefusal::EmptyArm)
        );
        let nan = |_: &[WeightEditSpec], _: &Read| f64::NAN;
        assert_eq!(
            run_decoy_battery(&edit(5, unit(2)), &sites(), &Default::default(), &read, nan).verdict,
            DecoyVerdict::Refused(DecoyRefusal::NonFiniteScore)
        );
        // Fader without a base reader → NeedsBaseReader.
        let w: Vec<f32> = (0..OUT * IN).map(|i| i as f32 * 0.01).collect();
        let fader = WeightEditSpec::column_scale_from_weights(3, &w, OUT, IN, &[5], 0.0, 1.0)
            .expect("fader");
        assert_eq!(
            run_decoy_battery(&fader, &sites(), &Default::default(), &read, score).verdict,
            DecoyVerdict::Refused(DecoyRefusal::NeedsBaseReader)
        );
        // With a reader the fader battery runs to a verdict.
        let reader = |_l: u8, c: u32, out: &mut [f32]| {
            for (i, o) in out.iter_mut().enumerate() {
                *o = w[i * IN + c as usize];
            }
            true
        };
        let s = DecoySites {
            in_dim: IN as u32,
            base_column: Some(&reader),
        };
        let rep = run_decoy_battery(&fader, &s, &Default::default(), &read, score);
        assert!(!matches!(rep.verdict, DecoyVerdict::Refused(_)), "{rep:?}");
    }
}
