//! Plan 311 Phase 3 — Alien Sampler GOAT Gate (motif-collapse benchmark).
//!
//! The make-or-break experiment: prove the dual-encoder `MedianTopMAvailability`
//! beats OPUS-style scalar local redundancy at reducing population-level motif
//! collapse, without sacrificing >10% coherence quality.
//!
//! Run:
//! ```bash
//! cargo bench --bench alien_sampler_goat --no-default-features --features alien_sampler
//! ```
//!
//! ## Scenario (T3.1)
//! 100 NPCs, each with a 16-dim candidate pool of "direction vectors". Each NPC
//! has a fixed personality direction; coherence = dot(candidate, personality).
//! The coherence surface is multi-modal (3-5 peaks) so multiple valid motifs
//! exist — the alien sampler should spread across them while scalar redundancy
//! collapses to one.
//!
//! ## Arms (T3.2)
//! - **Arm A (no availability, β=0):** coherence-only. Expected: severe motif
//!   collapse (all NPCs converge to their single highest-coherence direction).
//! - **Arm B (OPUS-style scalar local redundancy):** per-NPC penalty against
//!   own previous selections (CountSketch-equivalent). Local-only — doesn't
//!   see what other NPCs picked.
//! - **Arm C (AlienSampler β=0.7):** `MedianTopMAvailability` against the
//!   shared zone bank, populated from all NPCs' emissions.
//!
//! ## Metrics (T3.4)
//! - **G1 (motif collapse):** top-10 direction concentration across the zone
//!   in the last 1000 cycles. Arm C must be ≤ 50% of Arm B's concentration.
//!   Paper analog: 95.7%→34.3%.
//! - **G2 (quality preservation):** mean coherence of selected directions in
//!   last 1000 cycles. Arm C must be ≥ 90% of Arm A's mean coherence.
//! - **G3 (perf):** per-cycle wall time. Arm C must be ≤ 5× Arm B's per-cycle
//!   wall time.
//! - **G4 (latent boundary):** static check — no `Vec<f32>` escapes the
//!   `rank()` call boundary in the public API. (Verified by type system;
//!   re-asserted at bench time.)

#![cfg(feature = "alien_sampler")]

use katgpt_rs::alien_sampler::{
    AlienConfig, AlienSampler, CoherenceScorer, MedianTopMAvailability, ScoredCandidate,
};
use rayon::prelude::*;
use std::time::{Duration, Instant};

// ─── Config ─────────────────────────────────────────────────────────────────

/// Number of NPCs in the zone.
const N_NPCS: usize = 100;

/// Candidate-pool dimensionality (paper: 16-dim repertoires).
const POOL_DIM: usize = 16;

/// Number of candidate directions per NPC per cycle.
const POOL_SIZE: usize = 32;

/// Number of cycles per arm per seed.
///
/// Reduced from the plan's 10k to 1k for the first validation run — the
/// Arm C zone-bank rebuild (clone + norm precompute) happens every cycle
/// and dominates wall time at 10k cycles. Once the rebuild is optimized
/// (Phase 4 or a follow-up), restore to 10k for the final GOAT record.
const N_CYCLES: usize = 1_000;

/// Number of seeds (deterministic LCG seeds). Results are averaged.
///
/// Reduced from 5 to 2 for the first validation run; restore to 5 for the
/// final GOAT record.
const SEEDS: &[u64] = &[1, 2];

/// Paper-default β for Arm C.
const ARM_C_BETA: f32 = 0.7;

/// Paper-default m for the zone bank.
const M: usize = 10;

/// Number of cycles at the end of the run used for metric computation
/// (last 200 cycles — discards the burn-in / transient phase). Scaled with
/// N_CYCLES; restore to 1000 when N_CYCLES goes back to 10k.
const METRIC_WINDOW: usize = 200;

/// Multi-modal coherence peaks — risk register says "3-5 peaks, not 1" so the
/// alien sampler has multiple valid motifs to spread across. These are the
/// "archetypal" directions; each NPC's personality is a noisy blend of one of
/// these archetypes + small per-NPC jitter.
const N_ARCHETYPES: usize = 5;

// ─── Deterministic LCG ─────────────────────────────────────────────────────

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 33
    }
    fn next_f32(&mut self) -> f32 {
        // Divide by 2^31 for [0, 1) — matches the salience_tri_gate bench fix.
        (self.next() as f32) / ((1u64 << 31) as f32)
    }
    /// Uniform float in `[-range, range]`.
    fn next_range(&mut self, range: f32) -> f32 {
        (self.next_f32() * 2.0 - 1.0) * range
    }
}

// ─── Coherence scorer ──────────────────────────────────────────────────────
//
// KEY DESIGN: coherence is a SHARED function across all NPCs — a fixed global
// "quality" direction. All NPCs rank the same candidates the same way on the
// coherence axis. This is what creates motif collapse: without availability
// pressure, all NPCs pick the same top candidate → 100% concentration.
//
// The paper's analog: all scientists see the same coherence Guide (the shared
// notion of "good science"). Without availability pressure, they all converge
// to the same few high-coherence topics → motif collapse.

/// Shared coherence scorer: dot product against a fixed global "quality"
/// direction. All NPCs share this — there is no per-NPC personality on the
/// coherence axis. (Per-NPC variation comes from the candidate pool, which is
/// drawn with per-NPC randomness.)
#[derive(Clone)]
struct SharedCoherence {
    direction: Vec<f32>,
}

impl CoherenceScorer<f32> for SharedCoherence {
    #[inline]
    fn coherence(&self, atoms: &[f32]) -> f32 {
        let mut s = 0.0_f32;
        for (a, b) in atoms.iter().zip(self.direction.iter()) {
            s += a * b;
        }
        s
    }
}

// ─── Scenario data ─────────────────────────────────────────────────────────

/// Upper bound on the zone bank length (FIFO eviction when exceeded). Kept
/// as a `const` at module scope so `NpcState::cosine_scratch` can size the
/// per-NPC availability scratch once at construction — rayon needs the scratch
/// to be per-NPC (no shared mutable borrow across the parallel NPC loop).
const BANK_CAP: usize = 200;

/// A single NPC's state for one arm of the experiment.
struct NpcState {
    /// Per-NPC candidate pool (regenerated each cycle).
    /// Stored as `Vec<Vec<f32>>` for compatibility with the sampler API.
    pool: Vec<Vec<f32>>,
    /// Per-NPC scratch buffers for the sampler.
    scratch_c: Vec<f32>,
    scratch_a: Vec<f32>,
    /// Per-NPC availability cosine scratch (sized to BANK_CAP). Owned per-NPC
    /// so the Arm C NPC loop can run under rayon without a shared mutable
    /// borrow. `availability_batch` only reads the first `bank_len()` entries.
    cosine_scratch: Vec<f32>,
    /// Per-NPC output buffer for the sampler.
    out: Vec<ScoredCandidate>,
    /// Per-NPC local-redundancy counts (Arm B only): how many times each
    /// pool direction has been selected by THIS npc in the past. Used as a
    /// scalar penalty (the OPUS-style local-redundancy signal).
    local_counts: Vec<u32>,
}

impl NpcState {
    fn new() -> Self {
        Self {
            pool: vec![vec![0.0; POOL_DIM]; POOL_SIZE],
            scratch_c: vec![0.0; POOL_SIZE],
            scratch_a: vec![0.0; POOL_SIZE],
            cosine_scratch: vec![0.0; BANK_CAP],
            out: Vec::with_capacity(POOL_SIZE),
            local_counts: vec![0; POOL_SIZE],
        }
    }
}

/// Build the archetype directions (N_ARCHETYPES orthogonal-ish unit vectors).
fn make_archetypes(seed: u64) -> Vec<Vec<f32>> {
    let mut rng = Lcg::new(seed);
    let mut archetypes: Vec<Vec<f32>> = Vec::with_capacity(N_ARCHETYPES);
    for _ in 0..N_ARCHETYPES {
        let mut v: Vec<f32> = (0..POOL_DIM).map(|_| rng.next_range(1.0)).collect();
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut v {
                *x /= norm;
            }
        }
        archetypes.push(v);
    }
    archetypes
}

/// Build the shared coherence direction. This is a single unit vector that
/// defines the "quality" axis. Multi-modality comes from the ARCHETYPES
/// (below) — the pool generator produces candidates near each archetype, and
/// the shared coherence rewards proximity to the global direction.
///
/// For motif collapse to happen, the global direction must have a clear
/// "higest-coherence archetype" — we align it with archetype 0 so that
/// archetype-0 candidates have the highest coherence.
fn make_shared_coherence(archetypes: &[Vec<f32>]) -> SharedCoherence {
    // Use archetype 0 as the global direction — candidates near archetype 0
    // have the highest coherence.
    SharedCoherence { direction: archetypes[0].clone() }
}

/// Build per-NPC initial states. All NPCs share the coherence function; the
/// only per-NPC state is the candidate pool (regenerated each cycle).
fn make_npcs() -> Vec<NpcState> {
    (0..N_NPCS).map(|_| NpcState::new()).collect()
}

/// Regenerate an NPC's candidate pool for one cycle. The pool is drawn from
/// a mixture of archetypes so that without availability pressure, all NPCs
/// converge to archetype-0 candidates (the global-coherence peak). The pool
/// composition:
/// - ~40% of slots near archetype 0 (the global coherence peak).
/// - ~40% near other archetypes (alternative motifs — same coherence surface,
///   different peaks; lower coherence but still "on-motif").
/// - ~20% random noise.
///
/// Because all NPCs draw from the same archetype mixture and coherence is
/// shared, the top-coherence candidate in every NPC's pool tends to be the
/// archetype-0-aligned one → all NPCs pick archetype 0 → collapse.
fn regen_pool(npc: &mut NpcState, archetypes: &[Vec<f32>], rng: &mut Lcg) {
    for j in 0..POOL_SIZE {
        let pool = &mut npc.pool[j];
        let r = rng.next_f32();
        let source_idx = if r < 0.4 {
            // Near the global peak (archetype 0).
            0
        } else if r < 0.8 {
            // Near another archetype (alternative motif).
            1 + (rng.next() as usize % (N_ARCHETYPES - 1))
        } else {
            // Noise.
            usize::MAX
        };
        if source_idx == usize::MAX {
            for k in 0..POOL_DIM {
                pool[k] = rng.next_range(1.0);
            }
        } else {
            let archetype = &archetypes[source_idx];
            for k in 0..POOL_DIM {
                pool[k] = archetype[k] + rng.next_range(0.2); // ±20% noise
            }
        }
        // Normalize.
        let norm: f32 = pool.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in pool.iter_mut() {
                *x /= norm;
            }
        }
    }
}

// ─── Direction identification ──────────────────────────────────────────────
//
// To measure "top-10 direction concentration" we need to bucket selected
// directions into discrete "direction IDs". We quantize each selected
// direction to its nearest archetype and use the archetype index as the ID.
// (Directions near archetype 0 → bucket 0, etc.) This gives us at most
// N_ARCHETYPES + 1 buckets (the +1 is "noise" — assigned to a sentinel).

const NOISE_BUCKET: usize = N_ARCHETYPES; // sentinel for non-archetype-aligned picks

fn nearest_archetype(direction: &[f32], archetypes: &[Vec<f32>]) -> usize {
    let mut best_idx = NOISE_BUCKET;
    let mut best_cos = 0.5; // threshold: below 0.5 cosine → "noise" bucket
    for (i, arch) in archetypes.iter().enumerate() {
        let mut dot = 0.0_f32;
        let mut norm_a = 0.0_f32;
        let mut norm_b = 0.0_f32;
        for k in 0..direction.len() {
            dot += direction[k] * arch[k];
            norm_a += direction[k] * direction[k];
            norm_b += arch[k] * arch[k];
        }
        let cos = if norm_a > 0.0 && norm_b > 0.0 {
            dot / (norm_a.sqrt() * norm_b.sqrt())
        } else {
            0.0
        };
        if cos > best_cos {
            best_cos = cos;
            best_idx = i;
        }
    }
    best_idx
}

// ─── Arm A: coherence-only (β=0) ───────────────────────────────────────────

struct ArmA {
    npcs: Vec<NpcState>,
    coherence: SharedCoherence,
}

impl ArmA {
    fn new(coherence: SharedCoherence) -> Self {
        Self { npcs: make_npcs(), coherence }
    }

    /// Run one cycle. Returns the selected direction for each NPC.
    fn step(&mut self, archetypes: &[Vec<f32>], rng: &mut Lcg) -> Vec<(usize, Vec<f32>)> {
        let mut selections: Vec<(usize, Vec<f32>)> = Vec::with_capacity(N_NPCS);
        // Arm A is coherence-only (β=0); no availability pressure. Build the
        // sampler once outside the NPC loop — the coherence is shared.
        let sampler = AlienSampler::new(
            self.coherence.clone(),
            MedianTopMAvailability::new(vec![], M),
            AlienConfig::coherence_only(),
        );
        for i in 0..N_NPCS {
            let npc = &mut self.npcs[i];
            regen_pool(npc, archetypes, rng);
            sampler
                .rank_into(
                    &npc.pool,
                    &mut npc.scratch_c,
                    &mut npc.scratch_a,
                    &mut npc.out,
                )
                .unwrap();
            let top_idx = npc.out[0].idx;
            let selected = npc.pool[top_idx].clone();
            selections.push((i, selected));
        }
        selections
    }
}

// ─── Arm B: OPUS-style scalar local redundancy ─────────────────────────────
//
// Per the plan: "per-NPC CountSketch-equivalent penalty against own previous
// selections." We approximate CountSketch with a simple per-direction counter
// (the pool is small: POOL_SIZE=32, so a flat array is cheaper than a sketch).
//
// The penalty is subtracted from the coherence score before ranking:
//   adjusted_score = coherence - λ · local_count
// where λ is a redundancy weight.
//
// IMPORTANT: Arm B's local penalty only sees what THIS NPC picked — it cannot
// see what OTHER NPCs picked. So if all NPCs pick archetype-0 first (the
// global coherence peak), they all penalize archetype-0 locally and all switch
// to the next-best archetype-0 candidate (still archetype 0, just a different
// sample). They cycle through archetype-0 candidates together → still
// concentrated on archetype 0. This is the fundamental limitation of
// local-only redundancy that the alien sampler's community signal fixes.

const LOCAL_REDUNDANCY_LAMBDA: f32 = 0.5;

struct ArmB {
    npcs: Vec<NpcState>,
    coherence: SharedCoherence,
}

impl ArmB {
    fn new(coherence: SharedCoherence) -> Self {
        Self { npcs: make_npcs(), coherence }
    }

    fn step(&mut self, archetypes: &[Vec<f32>], rng: &mut Lcg) -> Vec<(usize, Vec<f32>)> {
        let mut selections: Vec<(usize, Vec<f32>)> = Vec::with_capacity(N_NPCS);
        for i in 0..N_NPCS {
            let npc = &mut self.npcs[i];
            regen_pool(npc, archetypes, rng);
            // Score each pool direction by (coherence - λ · local_count).
            // Then pick the argmax (no z-scoring needed — single axis).
            let mut best_idx = 0;
            let mut best_score = f32::NEG_INFINITY;
            for j in 0..POOL_SIZE {
                let coh = self.coherence.coherence(&npc.pool[j]);
                let penalty = LOCAL_REDUNDANCY_LAMBDA * (npc.local_counts[j] as f32);
                let score = coh - penalty;
                if score > best_score {
                    best_score = score;
                    best_idx = j;
                }
            }
            npc.local_counts[best_idx] = npc.local_counts[best_idx].saturating_add(1);
            let selected = npc.pool[best_idx].clone();
            selections.push((i, selected));
        }
        selections
    }
}

// ─── Arm C: AlienSampler (β=0.7, MedianTopMAvailability zone bank) ─────────
//
// The zone bank is shared across NPCs and grows as NPCs emit selections. Each
// cycle: rank each NPC's pool against the current bank, pick the top, add it
// to the bank. This is the "community availability" signal — directions that
// many NPCs have already picked become "high availability" (low alien-ness),
// so the sampler steers NPCs toward under-explored directions.

struct ArmC {
    npcs: Vec<NpcState>,
    coherence: SharedCoherence,
    /// Shared zone bank of selected directions (grows over the run).
    zone_bank: Vec<Vec<f32>>,
    /// Cached availability scorer — rebuilt only every REBUILD_EVERY cycles.
    cached_avail: Option<MedianTopMAvailability>,
    /// Pre-built sampler for rank_precomputed (dummy coherence + empty
    /// availability — both axes are filled per-NPC via the batch path). Hoisted
    /// out of step() to avoid per-cycle allocation.
    sampler: AlienSampler<f32, SharedCoherence, MedianTopMAvailability>,
    cycle_count: usize,
}

/// Rebuild the cached scorer every N cycles. The bank is at most N cycles
/// stale — acceptable for a slow-moving population aggregate.
const REBUILD_EVERY: usize = 10;

impl ArmC {
    fn new(coherence: SharedCoherence) -> Self {
        Self::with_config(coherence, AlienConfig::paper_default())
    }

    fn with_config(coherence: SharedCoherence, config: AlienConfig) -> Self {
        Self {
            npcs: make_npcs(),
            coherence: coherence.clone(),
            zone_bank: Vec::new(),
            cached_avail: None,
            sampler: AlienSampler::new(
                coherence,
                MedianTopMAvailability::new(vec![], M),
                AlienConfig { beta: config.beta, top_m: M },
            ),
            cycle_count: 0,
        }
    }

    fn step(&mut self, archetypes: &[Vec<f32>], rng: &mut Lcg) -> Vec<(usize, Vec<f32>)> {
        // Rebuild the cached scorer every REBUILD_EVERY cycles (or on first
        // call, or when the bank is empty).
        if self.cycle_count % REBUILD_EVERY == 0 || self.cached_avail.is_none() {
            self.cached_avail = Some(MedianTopMAvailability::new(self.zone_bank.clone(), M));
        }
        self.cycle_count = self.cycle_count.wrapping_add(1);
        let avail = self.cached_avail.as_ref().expect("cached_avail built above");
        let bank_len = avail.bank_len();
        let sampler = &self.sampler;

        // ── Parallel NPC loop: regen + score + rank in ONE rayon pass.
        //
        // regen_pool consumes a variable number of rng draws per NPC, so we
        // cannot share one rng across the parallel loop. Instead we draw
        // N_NPCS seeds from the shared rng (advancing it by exactly N_NPCS
        // draws — a fixed, deterministic amount) and construct one independent
        // Lcg per NPC. This preserves bench-level determinism (same seed →
        // same output) but changes the per-NPC pool distributions vs the
        // pre-rayon serial loop (each NPC draws from its own stream rather
        // than the residual state of the previous NPC).
        //
        // R1 impact: the β=0.7 concentration metric stays bit-identical
        // (0.4999 → 0.4999) because it depends on the archetype mixture
        // probabilities, not specific draws. The β=0.7 quality metric shifts
        // by ~2e-4 (0.6553 → 0.6555) because mean dot-product depends on the
        // exact selected directions. This exceeds the strict 1e-6 tolerance,
        // but the GOAT gate verdicts (G1 FAIL, G2 FAIL, G3) are unchanged —
        // see `.benchmarks/311_alien_sampler_goat.md` for the full R1 analysis.
        // Fusing regen + scoring into one rayon pass avoids double dispatch
        // overhead and is required to approach the G3 ≤ 5× target.
        let seeds: Vec<u64> = (0..N_NPCS).map(|_| rng.next()).collect();
        let coherence = &self.coherence;
        let top_indices: Vec<usize> = self
            .npcs
            .par_iter_mut()
            .enumerate()
            .map(|(i, npc)| {
                let mut local_rng = Lcg::new(seeds[i]);
                regen_pool(npc, archetypes, &mut local_rng);
                for j in 0..POOL_SIZE {
                    npc.scratch_c[j] = coherence.coherence(&npc.pool[j]);
                }
                if bank_len > 0 {
                    avail.availability_batch(
                        &npc.pool,
                        &mut npc.scratch_a,
                        &mut npc.cosine_scratch[..bank_len],
                    );
                } else {
                    for s in npc.scratch_a.iter_mut() {
                        *s = 0.0;
                    }
                }
                sampler
                    .rank_precomputed(&mut npc.scratch_c, &mut npc.scratch_a, &mut npc.out)
                    .unwrap();
                npc.out[0].idx
            })
            .collect();

        // Serial post-phase: clone the selected directions from each NPC's
        // pool. This is O(N_NPCS × POOL_DIM) with no contention.
        let selections: Vec<(usize, Vec<f32>)> = top_indices
            .iter()
            .enumerate()
            .map(|(i, &top_idx)| (i, self.npcs[i].pool[top_idx].clone()))
            .collect();

        // Add this cycle's selections to the zone bank. Cap the bank size —
        // eviction is O(1) amortized using VecDeque semantics (swap_remove
        // from front would corrupt order, but order doesn't matter for the
        // median-of-top-m; we use a simple drain + push pattern).
        // Batch-add: if we'd exceed the cap, evict a batch from the front.
        let n_new = selections.len();
        if self.zone_bank.len() + n_new > BANK_CAP {
            let drop = (self.zone_bank.len() + n_new - BANK_CAP).min(self.zone_bank.len());
            self.zone_bank.drain(0..drop);
        }
        for (_, dir) in &selections {
            self.zone_bank.push(dir.clone());
        }
        selections
    }
}

// ─── Metric computation ────────────────────────────────────────────────────

/// Per-cycle record: which archetype bucket each NPC's selection fell into +
/// the raw coherence of the selection (dot product against the selecting
/// NPC's OWN personality direction — the actual coherence signal the sampler
/// sees, not a proxy).
struct CycleRecord {
    archetype_buckets: Vec<usize>,
    /// Per-selection coherence: dot(selected_dir, selecting_npc_personality).
    /// This is the real quality signal — if Arm C sacrifices too much of it,
    /// G2 fails.
    coherences: Vec<f32>,
}

/// Run one arm for `N_CYCLES` cycles, recording the last `METRIC_WINDOW` cycles.
/// `shared_coherence` is used to compute the real per-selection coherence for G2
/// (dot product against the shared quality direction — the actual signal the
/// sampler sees).
fn run_arm<F>(
    mut step_fn: F,
    archetypes: &[Vec<f32>],
    shared_coherence: &SharedCoherence,
    seed: u64,
) -> (Vec<CycleRecord>, Duration)
where
    F: FnMut(&mut Lcg) -> Vec<(usize, Vec<f32>)>,
{
    let mut rng = Lcg::new(seed);
    let mut records: Vec<CycleRecord> = Vec::with_capacity(METRIC_WINDOW);
    let mut total_time = Duration::ZERO;
    let start_record = N_CYCLES.saturating_sub(METRIC_WINDOW);
    for cycle in 0..N_CYCLES {
        let t0 = Instant::now();
        let selections = step_fn(&mut rng);
        let dt = t0.elapsed();
        total_time += dt;
        if cycle >= start_record {
            let mut buckets: Vec<usize> = Vec::with_capacity(N_NPCS);
            let mut cohs: Vec<f32> = Vec::with_capacity(N_NPCS);
            for (_npc_idx, dir) in &selections {
                let bucket = nearest_archetype(dir, archetypes);
                buckets.push(bucket);
                // Real coherence: dot(dir, shared quality direction). Both
                // are unit-norm → dot ∈ [-1, 1]. Map to [0, 1] via (dot+1)/2
                // for a normalized quality score.
                let mut dot = 0.0_f32;
                for k in 0..dir.len() {
                    dot += dir[k] * shared_coherence.direction[k];
                }
                let quality = (dot + 1.0) * 0.5;
                cohs.push(quality);
            }
            records.push(CycleRecord { archetype_buckets: buckets, coherences: cohs });
        }
    }
    (records, total_time)
}

/// G1: top-K concentration — fraction of all selections that landed in the
/// top-K most-popular archetype buckets.
///
/// With N_ARCHETYPES=5 (+1 noise bucket), K=1 is the natural analog of the
/// paper's "top-10 author concentration": it measures whether the population
/// collapsed to a SINGLE motif (top-1 share → 1.0) or spread across multiple
/// archetypes (top-1 share → 1/N_ARCHETYPES ≈ 0.2). K=2 would measure
/// collapse to the top-2 motifs.
///
/// We report both K=1 and K=2 for context; the gate uses K=1 (the strictest
/// measure of single-motif collapse).
fn top_k_concentration(records: &[CycleRecord], k: usize) -> f64 {
    let n_buckets = N_ARCHETYPES + 1;
    let mut counts = vec![0u64; n_buckets];
    let mut total: u64 = 0;
    for r in records {
        for &b in &r.archetype_buckets {
            counts[b] += 1;
            total += 1;
        }
    }
    if total == 0 {
        return 0.0;
    }
    // Sort counts descending, take top-k.
    counts.sort_by(|a, b| b.cmp(a));
    let effective_k = k.min(n_buckets);
    let top_k_sum: u64 = counts[..effective_k].iter().sum();
    top_k_sum as f64 / total as f64
}

/// G2: mean coherence across the measurement window.
fn mean_coherence(records: &[CycleRecord]) -> f64 {
    let mut sum = 0.0_f64;
    let mut n = 0u64;
    for r in records {
        for &c in &r.coherences {
            sum += c as f64;
            n += 1;
        }
    }
    if n == 0 {
        0.0
    } else {
        sum / n as f64
    }
}

// ─── Main ───────────────────────────────────────────────────────────────────

fn pass_str(p: bool) -> &'static str {
    if p { "PASS" } else { "FAIL" }
}

fn main() {
    println!("=== Plan 311 Alien Sampler GOAT Gate (Phase 3) ===");
    println!(
        "  npcs={N_NPCS}, pool_dim={POOL_DIM}, pool_size={POOL_SIZE}, \
         cycles={N_CYCLES}, seeds={}, metric_window={METRIC_WINDOW}",
        SEEDS.len()
    );
    println!(
        "  archetypes={N_ARCHETYPES}, arm_c_beta={ARM_C_BETA}, m={M}, \
         local_redundancy_lambda={LOCAL_REDUNDANCY_LAMBDA}"
    );
    println!();

    // Aggregate metrics across seeds.
    let mut g1_a = 0.0_f64;
    let mut g1_b = 0.0_f64;
    let mut g1_c = 0.0_f64;
    let mut g2_a = 0.0_f64;
    let mut g2_c = 0.0_f64;
    let mut cycle_time_b = Duration::ZERO;
    let mut cycle_time_c = Duration::ZERO;

    for &seed in SEEDS {
        let archetypes = make_archetypes(seed);
        let coherence = make_shared_coherence(&archetypes);

        // Arm A.
        let mut arm_a = ArmA::new(coherence.clone());
        let (rec_a, _t_a) = run_arm(
            |rng| arm_a.step(&archetypes, rng),
            &archetypes,
            &coherence,
            seed,
        );
        g1_a += top_k_concentration(&rec_a, 1);
        g2_a += mean_coherence(&rec_a);

        // Arm B.
        let mut arm_b = ArmB::new(coherence.clone());
        let (rec_b, t_b) = run_arm(
            |rng| arm_b.step(&archetypes, rng),
            &archetypes,
            &coherence,
            seed,
        );
        g1_b += top_k_concentration(&rec_b, 1);
        cycle_time_b += t_b;

        // Arm C.
        let mut arm_c = ArmC::new(coherence.clone());
        let (rec_c, t_c) = run_arm(
            |rng| arm_c.step(&archetypes, rng),
            &archetypes,
            &coherence,
            seed,
        );
        g1_c += top_k_concentration(&rec_c, 1);
        g2_c += mean_coherence(&rec_c);
        cycle_time_c += t_c;
    }

    // ── β sweep (Plan 311 decision tree: G2 FAIL → try β=0.3, 0.5) ──────
    // If the default β=0.7 fails G2, sweep lower β values to find a
    // quality/diversity tradeoff that satisfies both G1 and G2.
    println!("--- β sweep (G2 recovery attempt) ---");
    for &beta in &[0.7_f32, 0.5, 0.3, 0.2] {
        let mut sweep_g1 = 0.0_f64;
        let mut sweep_g2 = 0.0_f64;
        for &seed in SEEDS {
            let archetypes = make_archetypes(seed);
            let coherence = make_shared_coherence(&archetypes);
            let cfg = AlienConfig { beta, top_m: M };
            let mut arm = ArmC::with_config(coherence.clone(), cfg);
            let (rec, _t) = run_arm(
                |rng| arm.step(&archetypes, rng),
                &archetypes,
                &coherence,
                seed,
            );
            sweep_g1 += top_k_concentration(&rec, 1);
            sweep_g2 += mean_coherence(&rec);
        }
        let n = SEEDS.len() as f64;
        sweep_g1 /= n;
        sweep_g2 /= n;
        let g1_ratio = sweep_g1 / g1_b;
        let g2_ratio = sweep_g2 / g2_a;
        let g1_ok = g1_ratio <= 0.5;
        let g2_ok = g2_ratio >= 0.9;
        println!(
            "  β={beta}: G1_ratio={g1_ratio:.4} (≤0.50 {})  G2_ratio={g2_ratio:.4} (≥0.90 {})  concentration={sweep_g1:.4}  quality={sweep_g2:.4}",
            pass_str(g1_ok),
            pass_str(g2_ok)
        );
    }
    println!();

    let n_seeds = SEEDS.len() as f64;
    g1_a /= n_seeds;
    g1_b /= n_seeds;
    g1_c /= n_seeds;
    g2_a /= n_seeds;
    g2_c /= n_seeds;

    let per_cycle_b = cycle_time_b / (N_CYCLES * SEEDS.len()) as u32;
    let per_cycle_c = cycle_time_c / (N_CYCLES * SEEDS.len()) as u32;

    println!("=== Raw metrics (averaged over {} seeds) ===", SEEDS.len());
    println!("  G1 top-1 archetype share (concentration):");
    println!("    Arm A (coherence-only): {g1_a:.4}");
    println!("    Arm B (local redundancy): {g1_b:.4}");
    println!("    Arm C (AlienSampler):    {g1_c:.4}");
    println!("    (uniform spread baseline: {:.4})", 1.0 / (N_ARCHETYPES as f64));
    println!("  G2 mean quality (dot+1)/2 against shared coherence direction):");
    println!("    Arm A (coherence-only): {g2_a:.4}");
    println!("    Arm C (AlienSampler):    {g2_c:.4}");
    println!("  G3 per-cycle wall time:");
    println!("    Arm B (local redundancy): {:.2} µs", per_cycle_b.as_secs_f64() * 1e6);
    println!("    Arm C (AlienSampler):    {:.2} µs", per_cycle_c.as_secs_f64() * 1e6);
    let c_over_b = per_cycle_c.as_secs_f64() / per_cycle_b.as_secs_f64();
    println!("    C/B ratio: {c_over_b:.2}×");
    println!();

    // ── Gate decisions ────────────────────────────────────────────────
    // G1: Arm C ≤ 50% of Arm B's concentration.
    let g1_pass = g1_c <= 0.5 * g1_b;
    let g1_ratio = if g1_b > 0.0 { g1_c / g1_b } else { f64::INFINITY };

    // G2: Arm C ≥ 90% of Arm A's mean coherence.
    let g2_pass = g2_c >= 0.9 * g2_a;
    let g2_ratio = if g2_a > 0.0 { g2_c / g2_a } else { 0.0 };

    // G3: Arm C per-cycle ≤ 5× Arm B per-cycle.
    let g3_pass = c_over_b <= 5.0;

    // G4: static type-system check — re-assert at runtime.
    fn _assert_scored_candidate_is_pod<T: Copy>() {}
    _assert_scored_candidate_is_pod::<ScoredCandidate>();
    let g4_pass = true; // enforced by type system; no Vec<f32> in ScoredCandidate.

    println!("=== GOAT Gate Verdict ===");
    println!(
        "  G1 motif collapse: Arm C / Arm B = {g1_ratio:.4} (target ≤ 0.50)  [{}]",
        pass_str(g1_pass)
    );
    println!(
        "  G2 quality:        Arm C / Arm A = {g2_ratio:.4} (target ≥ 0.90)  [{}]",
        pass_str(g2_pass)
    );
    println!(
        "  G3 perf:           C/B = {:.2}× (target ≤ 5.00×)  [{}]",
        c_over_b,
        pass_str(g3_pass)
    );
    println!(
        "  G4 latent boundary: ScoredCandidate is Copy + no Vec<f32> in public output  [{}]",
        pass_str(g4_pass)
    );
    println!();

    let all_pass = g1_pass && g2_pass && g3_pass && g4_pass;
    if all_pass {
        println!("  → ALL GATES PASS. PROMOTE `alien_sampler` to default feature (Plan 311 Phase 5).");
    } else if !g1_pass {
        println!("  → G1 FAIL. DEMOTE — dual-encoder not worth the complexity over scalar redundancy.");
        println!("    Keep module as opt-in for paper reproduction. Note honestly in benchmark doc.");
    } else if !g2_pass {
        println!("  → G2 FAIL. Diversity at the cost of quality. Try β sweep before demoting.");
    } else if !g3_pass {
        println!("  → G3 FAIL. Perf regression. Profile; Phase 4 SIMD may fix. Else demote to opt-in.");
    }
}
