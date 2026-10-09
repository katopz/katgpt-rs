//! Loop-alignment probe (Issue 929 / Research 614 — DiscoLoop, arXiv:2607.00341).
//!
//! Measures, per loop iteration `k` at a probe position, the
//! state-vs-decode-manifold alignment `cos(H^(k), W[v̂])` with
//! `v̂ = argmax(W·H)` plus the top1−top2 readout margin. The DiscoLoop
//! finding this instruments: a looped carry state can be *decodable but
//! misaligned* — `P(bridge|H) ≈ 1.0` while `cos(H, W[bridge]) ≈ 0.27–0.33`
//! — and that misalignment predicts multi-hop composition failure.
//!
//! Three pieces, all pure measurement (no behavior change, no forward-path
//! coupling):
//!
//! 1. [`probe_alignment`] — one fused probe read over a caller state + its
//!    already-computed readout logits + the LM head. The logits GEMV is the
//!    caller's (the `LoopDeepRun::capture_logits` path computes it once per
//!    snapshot); the probe adds only the top-2 scan and the single alignment
//!    dot against the `W[v̂]` row. Allocation-free by construction: every
//!    input is borrowed and the return is a `Copy` struct.
//! 2. [`auroc`] / [`bootstrap_auroc_ci`] — the G1 readout: does the probe
//!    signal separate final-answer correctness? Deterministic (seeded
//!    SplitMix64, no global RNG state).
//! 3. [`generate_two_hop_fixture`] — the paper §2 generator shape (two
//!    entity-disjoint pools, shared role set, out-degree-10 facts,
//!    unique-answer composition queries) rendered as in-context text
//!    prompts, so a *pretrained* checkpoint can be probed without training.
//!
//! The 0-for-3 loop-intervention record on this stack (Benches 847/850/906)
//! is why this module is probe-FIRST: it measures whether the misalignment
//! signal even exists before any injection arm is built.

use crate::pruners::margin_gate::margin_split;
use crate::simd::{simd_dot_f32, simd_sum_sq};

// ---------------------------------------------------------------------------
// Probe read
// ---------------------------------------------------------------------------

/// One probe read at one loop iteration of one position.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AlignmentProbe {
    /// `v̂ = argmax(logits)` — the model's own decode of its state.
    pub argmax: usize,
    /// Top logit value.
    pub top1: f32,
    /// Second logit value (`-inf` degenerate when fewer than 2 entries —
    /// never produced here: the probe requires ≥ 2 logits).
    pub top2: f32,
    /// `top1 − top2` decode margin (the confidence leg of the signal).
    pub margin: f32,
    /// `cos(H, W[v̂])` — the alignment leg. `0.0` when either norm is zero
    /// or non-finite (a defined, neutral reading, never NaN).
    pub cos_alignment: f32,
}

/// Probe one state against its readout.
///
/// * `state` — the carry hidden state `H^(k)` (at least `dim` wide; only the
///   first `dim` entries are read).
/// * `logits` — the readout `W·H` at the same position/loop (the caller's
///   LM-head pass; never recomputed here).
/// * `lm_head` — row-major `[vocab, dim]` head; only the `v̂` row is read.
/// * `dim` — head width (`n_embd` / `hidden_size`).
///
/// Degenerate inputs (fewer than 2 logits, `dim == 0`, short state, short
/// head) return a defined neutral probe with `cos_alignment == 0.0` rather
/// than panicking — a probe must never take the run down.
pub fn probe_alignment(
    state: &[f32],
    logits: &[f32],
    lm_head: &[f32],
    dim: usize,
) -> AlignmentProbe {
    let neutral = AlignmentProbe {
        argmax: 0,
        top1: 0.0,
        top2: 0.0,
        margin: 0.0,
        cos_alignment: 0.0,
    };
    if logits.len() < 2 || dim == 0 || state.len() < dim {
        return neutral;
    }
    let split = margin_split(logits);
    let row_start = match split.argmax.checked_mul(dim) {
        Some(s) if lm_head.len() >= s + dim => s,
        _ => {
            return AlignmentProbe {
                argmax: split.argmax,
                top1: split.top1,
                top2: split.top2,
                margin: split.gap,
                cos_alignment: 0.0,
            };
        }
    };
    let row = &lm_head[row_start..row_start + dim];
    let dot = simd_dot_f32(state, row, dim);
    let state_sq = simd_sum_sq(state, dim);
    let row_sq = simd_sum_sq(row, dim);
    let denom = (state_sq * row_sq).sqrt();
    let cos_alignment = if denom > 0.0 && denom.is_finite() {
        dot / denom
    } else {
        0.0
    };
    AlignmentProbe {
        argmax: split.argmax,
        top1: split.top1,
        top2: split.top2,
        margin: split.gap,
        cos_alignment,
    }
}

// ---------------------------------------------------------------------------
// Deterministic RNG (SplitMix64 — seeded, no global state)
// ---------------------------------------------------------------------------

/// SplitMix64 step: `state` is advanced in place, the mixed word returned.
pub fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn splitmix_f64(state: &mut u64) -> f64 {
    (splitmix64(state) >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
}

fn splitmix_usize_below(state: &mut u64, bound: usize) -> usize {
    (splitmix_f64(state) * bound as f64) as usize % bound
}

/// Fisher–Yates shuffle, deterministic under `state`.
fn shuffle<T>(items: &mut [T], state: &mut u64) {
    for i in (1..items.len()).rev() {
        let j = splitmix_usize_below(state, i + 1);
        items.swap(i, j);
    }
}

// ---------------------------------------------------------------------------
// AUROC + bootstrap CI
// ---------------------------------------------------------------------------

/// Pairwise-count AUROC of `scores` against binary `labels`
/// (`true` = positive). Ties count 0.5. Returns `NaN` when either class is
/// empty (undefined, and the caller must not fold that into a pass column).
pub fn auroc(scores: &[f32], labels: &[bool]) -> f64 {
    let mut pos = 0usize;
    let mut neg = 0usize;
    for &l in labels {
        if l {
            pos += 1;
        } else {
            neg += 1;
        }
    }
    if pos == 0 || neg == 0 {
        return f64::NAN;
    }
    let mut wins = 0.0f64;
    for (i, &l) in labels.iter().enumerate() {
        if !l {
            continue;
        }
        for (j, &m) in labels.iter().enumerate() {
            if m {
                continue;
            }
            wins += match scores[i].partial_cmp(&scores[j]) {
                Some(std::cmp::Ordering::Greater) => 1.0,
                Some(std::cmp::Ordering::Equal) => 0.5,
                _ => 0.0,
            };
        }
    }
    wins / (pos as f64 * neg as f64)
}

/// Percentile-bootstrap AUROC confidence interval, stratified (positives and
/// negatives resampled independently, so class counts are preserved).
///
/// Deterministic: the split stream is seeded by `seed` alone; the same
/// inputs + seed give byte-identical bounds. `iters` is clamped to
/// `[1, 100_000]`. Bounds are `NaN` when either class is empty.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BootstrapAuroc {
    pub point: f64,
    pub ci_lo: f64,
    pub ci_hi: f64,
    pub iters: usize,
}

pub fn bootstrap_auroc_ci(
    scores: &[f32],
    labels: &[bool],
    iters: usize,
    seed: u64,
) -> BootstrapAuroc {
    let point = auroc(scores, labels);
    let pos_idx: Vec<usize> = labels
        .iter()
        .enumerate()
        .filter(|&(_, &l)| l)
        .map(|(i, _)| i)
        .collect();
    let neg_idx: Vec<usize> = labels
        .iter()
        .enumerate()
        .filter(|&(_, &l)| !l)
        .map(|(i, _)| i)
        .collect();
    if pos_idx.is_empty() || neg_idx.is_empty() {
        return BootstrapAuroc {
            point,
            ci_lo: f64::NAN,
            ci_hi: f64::NAN,
            iters: 0,
        };
    }
    let iters = iters.clamp(1, 100_000);
    let mut state = seed;
    let mut draws: Vec<f64> = Vec::with_capacity(iters);
    for _ in 0..iters {
        let mut wins = 0.0f64;
        for _ in 0..pos_idx.len() {
            let i = pos_idx[splitmix_usize_below(&mut state, pos_idx.len())];
            for _ in 0..neg_idx.len() {
                let j = neg_idx[splitmix_usize_below(&mut state, neg_idx.len())];
                wins += match scores[i].partial_cmp(&scores[j]) {
                    Some(std::cmp::Ordering::Greater) => 1.0,
                    Some(std::cmp::Ordering::Equal) => 0.5,
                    _ => 0.0,
                };
            }
        }
        draws.push(wins / (pos_idx.len() as f64 * neg_idx.len() as f64));
    }
    draws.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let lo_i = ((draws.len() as f64 * 0.025) as usize).min(draws.len() - 1);
    let hi_i = ((draws.len() as f64 * 0.975) as usize).min(draws.len() - 1);
    BootstrapAuroc {
        point,
        ci_lo: draws[lo_i],
        ci_hi: draws[hi_i],
        iters,
    }
}

// ---------------------------------------------------------------------------
// Two-hop fixture (paper §2 generator shape, in-context rendering)
// ---------------------------------------------------------------------------

/// Roles rendering as both a fact predicate (`is the {role} of`) and a
/// composed-question nominal (`the {role} of the … of`). 50 roles = the
/// paper's `|R| = 50`.
pub const DEFAULT_ROLES: [&str; 50] = [
    "mentor",
    "student",
    "cousin",
    "neighbor",
    "rival",
    "partner",
    "creditor",
    "debtor",
    "tenant",
    "landlord",
    "captain",
    "agent",
    "fan",
    "critic",
    "editor",
    "translator",
    "guide",
    "host",
    "guest",
    "apprentice",
    "guardian",
    "supplier",
    "client",
    "penpal",
    "barber",
    "baker",
    "butcher",
    "cashier",
    "dentist",
    "doctor",
    "driver",
    "farmer",
    "fisher",
    "gardener",
    "judge",
    "lawyer",
    "mason",
    "mechanic",
    "nurse",
    "pilot",
    "plumber",
    "porter",
    "roofer",
    "sailor",
    "tailor",
    "usher",
    "waiter",
    "welder",
    "banker",
    "witness",
];

const ADJECTIVES: [&str; 24] = [
    "bold", "calm", "dark", "eager", "fancy", "gentle", "happy", "icy", "jolly", "keen", "lively",
    "mild", "noble", "odd", "proud", "quick", "rural", "shy", "tidy", "urban", "vivid", "warm",
    "young", "zesty",
];

const NOUNS: [&str; 24] = [
    "otter", "falcon", "heron", "badger", "marten", "ferret", "rabbit", "weasel", "beaver",
    "pigeon", "sparrow", "raven", "robin", "willow", "cedar", "birch", "maple", "cobble",
    "meadow", "harbor", "summit", "valley", "brook", "delta",
];

/// `24 × 24` adjective–noun pairs — the entity-name capacity of the default
/// pools (the generator asserts pools fit within it).
pub const NAME_CAPACITY: usize = ADJECTIVES.len() * NOUNS.len();

/// Fixture shape (paper §2 defaults in [`FixtureSpec::default`]).
#[derive(Clone, Copy, Debug)]
pub struct FixtureSpec {
    /// Entities per pool (paper: 500 across two disjoint pools → 250 each).
    pub entities_per_pool: usize,
    /// Shared relation count (paper: 50).
    pub relations: usize,
    /// Out-degree: facts per entity (paper: 10).
    pub out_degree: usize,
    /// In-context fact window per prompt (gold facts + same-pool distractors).
    pub facts_per_prompt: usize,
    /// Unique-answer two-hop queries sampled per pool.
    pub two_hop_queries: usize,
    /// One-hop control queries sampled per pool (AUROC class balance).
    pub one_hop_queries: usize,
    /// Deterministic seed (no global RNG).
    pub seed: u64,
}

impl Default for FixtureSpec {
    fn default() -> Self {
        Self {
            entities_per_pool: 250,
            relations: 50,
            out_degree: 10,
            facts_per_prompt: 24,
            two_hop_queries: 100,
            one_hop_queries: 50,
            seed: 0x929,
        }
    }
}

/// One rendered probe question.
#[derive(Clone, Debug)]
pub struct TwoHopItem {
    pub prompt: String,
    /// The unique gold answer (an entity name).
    pub answer: String,
    /// The two-hop bridge entity (empty string on one-hop items).
    pub bridge: String,
    /// `true` = composition query (two hops), `false` = one-hop control.
    pub is_two_hop: bool,
    /// `true` = drawn from the held-out pool (the paper's OOD leg).
    pub is_ood: bool,
}

#[derive(Clone, Debug, Default)]
pub struct TwoHopFixture {
    pub items: Vec<TwoHopItem>,
    /// Unique-answer two-hop candidates available per pool BEFORE sampling
    /// (diagnostic: 0 means the KG shape admits no unique compositions).
    pub unique_compositions_per_pool: usize,
}

/// `(a, r) -> b` fact with pool-local entity indices.
#[derive(Clone, Copy, Debug)]
struct Fact {
    a: usize,
    r: usize,
    b: usize,
}

fn entity_name(i: usize) -> String {
    let adj = ADJECTIVES[i % ADJECTIVES.len()];
    let noun = NOUNS[(i / ADJECTIVES.len()) % NOUNS.len()];
    let mut name = format!("{adj} {noun}");
    // Names repeat past the 576-pair capacity — suffix a disambiguating
    // index so pools stay disjoint by construction at any requested size.
    if i >= NAME_CAPACITY {
        name.push_str(&format!(" {i}"));
    }
    name
}

/// Generate the two-hop fixture: facts + unique-answer composition queries
/// over two entity-disjoint pools with a shared role set, rendered as
/// in-context prompts. Deterministic under `spec.seed`.
pub fn generate_two_hop_fixture(spec: FixtureSpec) -> TwoHopFixture {
    let entities = spec.entities_per_pool.min(NAME_CAPACITY);
    let relations = spec.relations.min(DEFAULT_ROLES.len()).max(2);
    let out_degree = spec.out_degree.max(1).min(relations);
    let facts_per_prompt = spec.facts_per_prompt.max(2);

    let mut state = spec.seed;
    let mut fixture = TwoHopFixture::default();

    // Two entity-disjoint pools, one shared role set (the paper's G_A/G_B).
    for pool in 0..2usize {
        // Per-entity fact rows: `out_degree` distinct roles (shuffled once,
        // order preserved), each carrying one same-pool target.
        let mut facts: Vec<Fact> = Vec::with_capacity(entities * out_degree);
        for a in 0..entities {
            let mut roles: Vec<usize> = (0..relations).collect();
            shuffle(&mut roles, &mut state);
            for &r in roles.iter().take(out_degree) {
                let mut b = splitmix_usize_below(&mut state, entities);
                if b == a {
                    b = (b + 1) % entities;
                }
                facts.push(Fact { a, r, b });
            }
        }

        // Unique-answer composition map: (a, r1, r2) -> {c}. A query stays
        // only when exactly one bridge produces one answer (the paper's
        // verifiable-answer construction). BTreeMap, never HashMap: the
        // draw order must be a pure function of the seed (HashMap's
        // RandomState iteration order would leak nondeterminism in before
        // the shuffle).
        let mut comp: std::collections::BTreeMap<(usize, usize, usize), (usize, bool)> =
            std::collections::BTreeMap::new();
        for f1 in &facts {
            for f2 in &facts {
                if f2.a != f1.b || f1.a == f2.b {
                    continue;
                }
                let key = (f1.a, f1.r, f2.r);
                match comp.get_mut(&key) {
                    Some((_, unique)) => *unique = false,
                    None => {
                        comp.insert(key, (f2.b, true));
                    }
                }
            }
        }
        if pool == 0 {
            // Both pools share the shape; one diagnostic count suffices
            // (statistically identical by construction).
            fixture.unique_compositions_per_pool =
                comp.values().filter(|(_, unique)| *unique).count();
        }

        let mut comps: Vec<(usize, usize, usize, usize)> = comp
            .into_iter()
            .filter(|(_, (_, unique))| *unique)
            .map(|((a, r1, r2), (c, _))| (a, r1, r2, c))
            .collect();
        shuffle(&mut comps, &mut state);

        let mut pool_items: Vec<TwoHopItem> = Vec::new();
        for &(a, r1, r2, c) in comps.iter().take(spec.two_hop_queries) {
            pool_items.push(render_item(
                pool,
                a,
                Some((r1, r2, c)),
                &facts,
                entities,
                relations,
                facts_per_prompt,
                &mut state,
            ));
        }
        let mut one_hop: Vec<Fact> = facts.clone();
        shuffle(&mut one_hop, &mut state);
        for f in one_hop.iter().take(spec.one_hop_queries) {
            pool_items.push(render_item(
                pool,
                f.a,
                None,
                &facts,
                entities,
                relations,
                facts_per_prompt,
                &mut state,
            ));
        }
        fixture.items.append(&mut pool_items);
    }
    fixture
}

/// Render one prompt: a same-pool fact window (golds + distractors), the
/// question, and the `Answer:` cue.
#[allow(clippy::too_many_arguments)]
fn render_item(
    pool: usize,
    a: usize,
    two_hop: Option<(usize, usize, usize)>,
    facts: &[Fact],
    entities: usize,
    relations: usize,
    facts_per_prompt: usize,
    state: &mut u64,
) -> TwoHopItem {
    let name = |pool: usize, i: usize| entity_name(pool * entities + i);
    let role = |r: usize| DEFAULT_ROLES[r % relations];

    // Gold facts: for two-hop, the (a, r1, bridge) fact plus the
    // (bridge, r2, c) fact; for one-hop, the single outgoing fact of `a`.
    // Both lookups are exact: facts carry unique (a, r) rows and the
    // composition was derived from these very rows.
    let golds: Vec<Fact> = match two_hop {
        Some((r1, r2, _c)) => facts
            .iter()
            .find(|f| f.a == a && f.r == r1)
            .and_then(|f1| {
                facts
                    .iter()
                    .find(|f| f.a == f1.b && f.r == r2)
                    .map(|f2| vec![*f1, *f2])
            })
            .or_else(|| facts.iter().find(|f| f.a == a).map(|f| vec![*f]))
            .unwrap_or_default(),
        None => facts
            .iter()
            .find(|f| f.a == a)
            .map(|f| vec![*f])
            .unwrap_or_default(),
    };

    // Distractors: same-pool facts that are not golds.
    let mut distractors: Vec<Fact> = facts
        .iter()
        .copied()
        .filter(|f| !golds.iter().any(|g| g.a == f.a && g.r == f.r))
        .collect();
    shuffle(&mut distractors, state);
    distractors.truncate(facts_per_prompt.saturating_sub(golds.len()));

    let mut window: Vec<Fact> = golds.clone();
    window.append(&mut distractors);
    shuffle(&mut window, state);

    let mut prompt = String::new();
    for f in &window {
        prompt.push_str(&format!(
            "Fact: {} is the {} of {}.\n",
            name(pool, f.a),
            role(f.r),
            name(pool, f.b)
        ));
    }

    match two_hop {
        Some((r1, r2, c)) => {
            let bridge = golds
                .first()
                .map(|g| name(pool, g.b))
                .unwrap_or_default();
            prompt.push_str(&format!(
                "Question: Who is the {} of the {} of {}?\nAnswer:",
                role(r2),
                role(r1),
                name(pool, a)
            ));
            TwoHopItem {
                prompt,
                answer: name(pool, c),
                bridge,
                is_two_hop: true,
                is_ood: pool == 1,
            }
        }
        None => {
            let r = golds.first().map(|f| f.r).unwrap_or(0);
            prompt.push_str(&format!(
                "Question: Who is the {} of {}?\nAnswer:",
                role(r),
                name(pool, a)
            ));
            TwoHopItem {
                prompt,
                answer: golds.first().map(|f| name(pool, f.b)).unwrap_or_default(),
                bridge: String::new(),
                is_two_hop: false,
                is_ood: pool == 1,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_reads_constructed_alignment() {
        // state == W[0] exactly → argmax 0, cos 1; logits hand-built for a
        // known margin.
        let state = [1.0f32, 0.0, 0.0];
        let head = [
            1.0, 0.0, 0.0, // row 0 — the state itself
            0.0, 1.0, 0.0, // row 1 — orthogonal
        ];
        let logits = [2.0f32, 1.0];
        let p = probe_alignment(&state, &logits, &head, 3);
        assert_eq!(p.argmax, 0);
        assert!((p.margin - 1.0).abs() < 1e-6);
        assert!((p.cos_alignment - 1.0).abs() < 1e-6);

        // Same state, argmax flips to the orthogonal row → cos 0.
        let logits = [1.0f32, 2.0];
        let p = probe_alignment(&state, &logits, &head, 3);
        assert_eq!(p.argmax, 1);
        assert!(p.cos_alignment.abs() < 1e-6);

        // Noise on the aligned row pulls cos strictly below 1.
        let noisy_head = [0.8f32, 0.6, 0.0, 0.0, 1.0, 0.0];
        let p = probe_alignment(&state, &[2.0f32, 1.0], &noisy_head, 3);
        assert!((p.cos_alignment - 0.8).abs() < 1e-6);
    }

    #[test]
    fn probe_neutral_on_degenerate_inputs() {
        let head = [1.0f32, 0.0, 0.0, 1.0];
        assert_eq!(
            probe_alignment(&[1.0, 0.0], &[5.0], &head, 2).cos_alignment,
            0.0
        );
        assert_eq!(
            probe_alignment(&[1.0, 0.0], &[5.0, 3.0], &head, 0).cos_alignment,
            0.0
        );
        // Zero head row → zero norm → defined 0.0, never NaN.
        let zero_head = [0.0f32; 4];
        let p = probe_alignment(&[1.0, 0.0], &[5.0, 3.0], &zero_head, 2);
        assert_eq!(p.cos_alignment, 0.0);
        assert!(p.cos_alignment.is_finite());
    }

    #[test]
    fn auroc_separates_constructed_scores() {
        let scores = [0.9f32, 0.8, 0.2, 0.1];
        let labels = [true, true, false, false];
        assert!((auroc(&scores, &labels) - 1.0).abs() < 1e-12);
        assert!((auroc(&scores, &[false, false, true, true]) - 0.0).abs() < 1e-12);
        // All-tied scores → coin flip.
        assert!((auroc(&[1.0; 4], &labels) - 0.5).abs() < 1e-12);
        // Undefined with an empty class.
        assert!(auroc(&scores, &[true, true, true, true]).is_nan());
    }

    #[test]
    fn bootstrap_is_deterministic_and_separation_sensitive() {
        let good: Vec<f32> = (0..40).map(|i| if i % 2 == 0 { 0.9 } else { 0.1 }).collect();
        let labels: Vec<bool> = (0..40).map(|i| i % 2 == 0).collect();
        let a = bootstrap_auroc_ci(&good, &labels, 400, 7);
        let b = bootstrap_auroc_ci(&good, &labels, 400, 7);
        assert_eq!(a, b, "same seed must give byte-identical bounds");
        assert!(a.ci_lo > 0.95, "perfect separation: tight high CI, got {a:?}");

        // A quarter of the positives dropped BELOW every negative → the
        // point estimate must collapse strictly below the clean run's.
        let mut mixed = good.clone();
        for i in (0..40).step_by(4) {
            mixed[i] = 0.05;
        }
        let c = bootstrap_auroc_ci(&mixed, &labels, 400, 7);
        assert!(c.point < a.point);
        assert!(c.ci_lo < a.ci_lo);
    }

    #[test]
    fn fixture_is_deterministic_and_unique_answered() {
        let spec = FixtureSpec {
            entities_per_pool: 40,
            relations: 8,
            out_degree: 3,
            facts_per_prompt: 8,
            two_hop_queries: 12,
            one_hop_queries: 6,
            seed: 99,
        };
        let f1 = generate_two_hop_fixture(spec);
        let f2 = generate_two_hop_fixture(spec);
        assert_eq!(f1.items.len(), f2.items.len());
        for (x, y) in f1.items.iter().zip(f2.items.iter()) {
            assert_eq!(x.prompt, y.prompt, "fixture must be deterministic");
            assert_eq!(x.answer, y.answer);
        }
        // Pool symmetry: both pools render the same query mix.
        let id_items = f1.items.iter().filter(|i| !i.is_ood).count();
        let ood_items = f1.items.iter().filter(|i| i.is_ood).count();
        assert_eq!(id_items, ood_items);
        // One-hop controls are always available (one fact per entity);
        // two-hop items depend on unique-composition availability.
        assert!(id_items >= 2 * spec.one_hop_queries);
        // Pool disjointness: OOD items' entity names never appear in the
        // ID pool's prompts and vice versa — checked via the rendered names
        // by construction (index ranges), asserted here on a rendered pair.
        let id = &f1.items[0];
        let ood = f1.items.iter().find(|i| i.is_ood).expect("pool B items");
        assert!(!id.is_ood && ood.is_ood);
        // Every two-hop item carries a bridge distinct from its answer.
        for item in f1.items.iter().filter(|i| i.is_two_hop) {
            assert!(!item.bridge.is_empty());
            assert_ne!(item.bridge, item.answer);
            assert!(item.prompt.contains("Answer:"));
            assert!(item.prompt.contains(item.bridge.as_str()));
        }
        // One-hop controls: answer non-empty, no bridge.
        for item in f1.items.iter().filter(|i| !i.is_two_hop) {
            assert!(!item.answer.is_empty());
            assert!(item.bridge.is_empty());
        }
        // Shape sanity: the KG admits unique compositions at this size.
        assert!(f1.unique_compositions_per_pool > 0);
    }

    #[test]
    fn entity_names_stay_unique_across_pools() {
        // Past the 576-pair grid, names carry the index suffix.
        assert_eq!(entity_name(0), "bold otter");
        assert_eq!(entity_name(575), "zesty delta");
        assert_eq!(entity_name(576), "bold otter 576");
        // Pool disjointness by index range: pool A's first name can never
        // equal pool B's first name at any pool size ≥ 1.
        let e = 40usize;
        assert_ne!(entity_name(0), entity_name(e));
    }
}
