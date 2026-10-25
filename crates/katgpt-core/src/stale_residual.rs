//! Stale-residual speculative layer pipelining — the modelless primitives
//! (Issue 691 / Research 508, arXiv:2608.23841 §6.3 Approach A + B).
//!
//! For **standard** (non-delay-rewritten) transformer checkpoints, residual
//! dominance (`‖δℓ‖/‖x_in^ℓ‖ ≪ 1`) suggests layer ℓ+1 can begin on the
//! **stale** residual `x_in^ℓ` while layer ℓ is still computing —
//! accept-with-correction when the layer contribution lands small,
//! rollback-and-recompute when it doesn't. The paper proposes this and never
//! runs it; this module holds the pure analysis half so the first measured
//! verdict can be produced over real checkpoints (the K3-0.40B simulator
//! lives in the root crate's `kimi_k3` module; cross-repo trace producers in
//! riir-ai feed [`residual_dominance_from_trace`] for Bonsai/Gemma classes).
//!
//! Three primitives, all pure f32 math, zero deps, no allocs in the steady
//! state (callers own buffers):
//!
//! 1. **Residual-dominance table** ([`LayerRatioStats`],
//!    [`layer_ratio_stats_into`]) — per-layer `‖δℓ‖/‖x_in^ℓ‖` distribution
//!    over held activations + the paper's viability bar
//!    ([`PAPER_VIABILITY_BAR`]: >50% of layers with median ratio < 0.05).
//! 2. **Accept gate** ([`AcceptGate`]) — the threshold rule; accept iff
//!    `ratio < θ` (the rollback machinery is consumer-side).
//! 3. **Overlap latency model** ([`OverlapLatency`]) — the paper's
//!    `net = (C+IO)/max(C, IO_eff)` I/O-overlap predictor (their §1.2),
//!    parameterized by bits/weight so it projects at OUR ternary stream
//!    ratios (Research 508 §2.0) and not just the paper's Q4 numbers.
//!
//! Distinct from shipped cousins (do not conflate): `HydraSkipPlan` **drops**
//! layers on cumulative-DE thresholds; token-level speculative decode
//! (`LeviathanVerifier`, GDN tree-verify) speculates **tokens**, not layer
//! inputs. Nothing here executes anything — it scores what a simulator
//! measured.
//!
//! Stale-residual speculative layer pipelining — the modelless analysis half
//! (Issue 691 / Research 508, arXiv:2608.23841 §6.3 Approach A+B — the
//! paper's own UNTESTED hypothesis): residual-dominance ratios ‖δℓ‖/‖x_in‖
//! + the paper's viability bar (>50% of layers, median ratio < 0.05), the
//! accept/rollback threshold gate, the KL/top-1 SpecOutcome metrics + θ-sweep
//! reduction, and the stream-ratio-aware (C+IO)/max(C,IO) overlap latency
//! model (bits/weight-parameterized so ternary 1.58 b/w and Q4 4.6 b/w
//! project from one code). Pure analysis over captured activations — the
//! K3-0.40B simulator (which actually executes layers on stale residuals)
//! lives in the root crate's `kimi_k3` module; Bonsai/Gemma trace producers
//! in riir-ai feed `residual_dominance_from_trace`. Opt-in pending the Issue
//! 691 verdict.

#![allow(clippy::needless_range_loop)]

/// The paper's own viability bar: at least this fraction of layers must have
/// median residual ratio below [`PAPER_RATIO_THRESHOLD`] for Approach A to be
/// worth a wall-clock build (arXiv:2608.23841 §6.3 success criterion, quoted
/// verbatim as ">50% of layers with ratio < 0.05").
pub const PAPER_VIABILITY_BAR: f32 = 0.5;
/// The ratio threshold in the paper's viability criterion.
pub const PAPER_RATIO_THRESHOLD: f32 = 0.05;

/// L2 norm of a slice (f32 accumulate, f32 sqrt — matches the analyzer's
/// measurement precision).
fn l2_norm(v: &[f32]) -> f32 {
    let mut s = 0.0f32;
    for &x in v {
        s += x * x;
    }
    s.sqrt()
}

/// Squared-distance accumulation then one sqrt: `‖b − a‖` without allocating.
fn l2_diff_norm(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = 0.0f32;
    for i in 0..a.len() {
        let d = b[i] - a[i];
        s += d * d;
    }
    s.sqrt()
}

/// Per-layer residual-dominance statistics over a population of positions.
///
/// One row per layer ℓ: the distribution of
/// `r_ℓ,p = ‖x_out^ℓ,p − x_in^ℓ,p‖ / ‖x_in^ℓ,p‖` across measured positions p
/// (decode steps and/or prompts). Row fields are the analyzer's whole output;
/// the simulator supplies the vectors.
#[derive(Debug, Clone, PartialEq)]
pub struct LayerRatioStats {
    /// Layer index.
    pub layer: usize,
    /// Mean ratio across positions.
    pub mean: f32,
    /// Median ratio (population median — even N averages the two centers).
    pub median: f32,
    /// Minimum ratio observed.
    pub min: f32,
    /// Maximum ratio observed.
    pub max: f32,
    /// Number of positions sampled.
    pub n: usize,
}

impl LayerRatioStats {
    /// Does this layer pass the paper's per-layer ratio threshold?
    pub fn passes_paper_bar(&self) -> bool {
        self.median < PAPER_RATIO_THRESHOLD
    }
}

/// Fraction of measured layers whose median ratio is below
/// [`PAPER_RATIO_THRESHOLD`] — the paper's viability aggregate.
pub fn fraction_layers_under_paper_bar(rows: &[LayerRatioStats]) -> f32 {
    if rows.is_empty() {
        return 0.0;
    }
    let passing = rows.iter().filter(|r| r.passes_paper_bar()).count();
    passing as f32 / rows.len() as f32
}

/// Compute per-layer ratio statistics from captured per-layer residual
/// streams, writing rows into `out` (cleared first).
///
/// `streams[layer]` holds one concatenated run of per-position vectors for
/// that layer: for each position p, the layer's input `x_in^{ℓ,p}` followed
/// by its output `x_out^{ℓ,p}` (each `dim` long, interleaved `2·dim` stride).
/// This input/output adjacency is exactly what every capture substrate in the
/// stack already produces (K3 `prefix_sum_in` snapshots; the Bonsai/Gemma
/// per-layer hidden taps where `x_in^{ℓ+1} = x_out^ℓ` makes each capture row
/// both the previous layer's output and this layer's input).
///
/// Ratios with a degenerate denominator (`‖x_in‖ = 0`) are skipped, not
/// clamped — a zero-input position carries no dominance information.
pub fn layer_ratio_stats_into(streams: &[Vec<f32>], dim: usize, out: &mut Vec<LayerRatioStats>) {
    out.clear();
    for (layer, stream) in streams.iter().enumerate() {
        debug_assert!(
            stream.len() % (2 * dim) == 0,
            "layer {layer} stream must be pairs of {dim}-dim vectors"
        );
        let n_pos = stream.len() / (2 * dim);
        // Ratios collected in a caller-visible-free way: we need median, so
        // materialize them. n_pos is bounded by the capture population (≤ a
        // few thousand) — one Vec per layer here is the analyzer's cold path,
        // not a steady-state cost.
        let mut ratios = Vec::with_capacity(n_pos);
        for p in 0..n_pos {
            let x_in = &stream[p * 2 * dim..p * 2 * dim + dim];
            let x_out = &stream[p * 2 * dim + dim..p * 2 * dim + 2 * dim];
            let denom = l2_norm(x_in);
            if denom > 0.0 {
                ratios.push(l2_diff_norm(x_in, x_out) / denom);
            }
        }
        if ratios.is_empty() {
            out.push(LayerRatioStats {
                layer,
                mean: f32::NAN,
                median: f32::NAN,
                min: f32::NAN,
                max: f32::NAN,
                n: 0,
            });
            continue;
        }
        let n = ratios.len();
        let mean = ratios.iter().sum::<f32>() / n as f32;
        ratios.sort_by(|a, b| a.partial_cmp(b).expect("finite ratios (den>0 guard)"));
        let median = if n % 2 == 1 {
            ratios[n / 2]
        } else {
            0.5 * (ratios[n / 2 - 1] + ratios[n / 2])
        };
        out.push(LayerRatioStats {
            layer,
            mean,
            median,
            min: ratios[0],
            max: ratios[n - 1],
            n,
        });
    }
}

/// Convenience wrapper — same as [`layer_ratio_stats_into`] with an owned Vec.
pub fn layer_ratio_stats(streams: &[Vec<f32>], dim: usize) -> Vec<LayerRatioStats> {
    let mut out = Vec::new();
    layer_ratio_stats_into(streams, dim, &mut out);
    out
}

/// Analyze an externally produced per-layer trace (the cross-repo handoff:
/// riir-ai's Bonsai/Gemma dumpers write "SRTR" trace files; the consumer
/// decodes them and calls this).
///
/// `trace[layer]` = per-position vectors for that layer, `x_out^ℓ` after
/// layer ℓ (the capture substrates tap the residual stream after each
/// layer's residual add). `x_in^0` = the position's embedding row
/// (`embeddings[p]`, same dim). `x_in^{ℓ} = x_out^{ℓ-1}` for ℓ > 0 — the
/// residual stream IS the chain, so one capture table plus the embedding
/// rows reconstructs every ratio without a second capture tap.
///
/// Streams are assembled internally in the [`layer_ratio_stats_into`]
/// input format; for large traces prefer calling that fn directly with
/// reused buffers.
pub fn residual_dominance_from_trace(
    trace: &[Vec<f32>],
    embeddings: &[f32],
    dim: usize,
) -> Vec<LayerRatioStats> {
    debug_assert!(!trace.is_empty());
    let n_pos = trace[0].len() / dim;
    let n_layer = trace.len();
    let mut streams = vec![Vec::with_capacity(n_pos * 2 * dim); n_layer];
    for p in 0..n_pos {
        // Layer 0: x_in = embedding row.
        streams[0].extend_from_slice(&embeddings[p * dim..(p + 1) * dim]);
        for (layer, rows) in trace.iter().enumerate() {
            let x_out = &rows[p * dim..(p + 1) * dim];
            streams[layer].extend_from_slice(x_out);
            if layer + 1 < n_layer {
                streams[layer + 1].extend_from_slice(x_out);
            }
        }
    }
    layer_ratio_stats(&streams, dim)
}

/// The accept/rollback gate for speculative layer execution.
///
/// Accept iff the layer's relative contribution stays under the threshold:
/// the paper's Approach-A rule. Pure predicate — the correction application
/// (add δ̂ or δ) and the rollback recompute are consumer-side because they
/// are architecture-specific.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AcceptGate {
    /// Ratio threshold θ. Paper's analysis bar: 0.05.
    pub threshold: f32,
}

impl Default for AcceptGate {
    fn default() -> Self {
        Self {
            threshold: PAPER_RATIO_THRESHOLD,
        }
    }
}

impl AcceptGate {
    /// Gate verdict from the measured ratio `‖δℓ‖/‖x_in^ℓ‖`.
    pub fn accepts(&self, ratio: f32) -> bool {
        ratio < self.threshold
    }

    /// Gate verdict from the raw vectors (avoids recomputing norms when the
    /// simulator already has them — prefer [`AcceptGate::accepts`]).
    pub fn accepts_vectors(&self, x_in: &[f32], x_out: &[f32]) -> bool {
        let denom = l2_norm(x_in);
        if denom <= 0.0 {
            return false; // no information — treat as unsafe
        }
        self.accepts(l2_diff_norm(x_in, x_out) / denom)
    }
}

/// Stream-ratio-aware I/O-overlap latency model (paper §1.2, extraction #3).
///
/// The paper's staged-I/O overlap predictor: `speedup = (C + IO) /
/// max(C, IO_eff)` — serial time over the overlapped critical path, bounded
/// by 2×. **The win exists only when the I/O is hideable** (a slow stream
/// the compute can shadow: disk/NVMe-resident weights, or GPU H2D from a
/// Cold-tier thaw). When both layers' weight streams share ONE bus (our
/// RAM-resident CPU decode), speculation cannot create bandwidth —
/// [`OverlapLatency::pair_speedup`] with `shared_bus = true` models the ×2
/// per-stream IO inflation and correctly degenerates to ≈1×.
///
/// Parameterized by bits/weight so ternary (1.58 b/w) and Q4 (~4.6 b/w, the
/// paper's regime) project from the same code — Research 508 §2.0: ternary
/// sits ~2× below machine balance vs Q4's ~5×, so the regimes flip at
/// different bandwidths.
#[derive(Debug, Clone, Copy)]
pub struct OverlapLatency {
    /// Compute throughput of the overlapped span [FLOP/s] available to one
    /// stream (the speculative compute is assumed to have a second core —
    /// CPU decode with ≥2 threads).
    pub compute_rate: f64,
    /// Exclusive (single-stream) byte bandwidth of the weight-read path
    /// [B/s].
    pub bandwidth: f64,
    /// Fraction of speculative layer executions that are ACCEPTED (gate
    /// pass). A rejected speculation costs its rollback recompute.
    pub accept_rate: f64,
    /// Rollback cost as a fraction of the rejected layer's full span
    /// (1.0 = full recompute; the re-READ is usually a cache hit so the
    /// compute dominates — the model charges it against compute only).
    pub rollback_factor: f64,
}

impl OverlapLatency {
    /// Per-layer `(compute_s, io_s)` for a weight span of `n_weights`
    /// weights at `bits_per_weight` encoding with `flops_per_weight`
    /// arithmetic. Ternary matvec: 2 FLOP/w, 1.58 bits; Q4_K: ~2 FLOP/w,
    /// ~4.6 bits (block overhead included).
    pub fn layer_span(
        &self,
        n_weights: u64,
        bits_per_weight: f64,
        flops_per_weight: f64,
    ) -> (f64, f64) {
        let c = (n_weights as f64 * flops_per_weight) / self.compute_rate;
        let io = (n_weights as f64 * bits_per_weight / 8.0) / self.bandwidth;
        (c, io)
    }

    /// The paper's headline formula, verbatim: `(C + IO) / max(C, IO)` for a
    /// single overlapped span (IO_eff = IO when the overlapped stream has
    /// the bus to itself). Sanity anchor: at the paper's NVMe ratios this
    /// reproduces their ≤1.68×; bounded by 2× at C = IO.
    pub fn overlap_speedup(c: f64, io: f64) -> f64 {
        if c.max(io) <= 0.0 {
            return 1.0;
        }
        (c + io) / c.max(io)
    }

    /// Net speedup of stale-residual speculation for ONE layer pair (layer ℓ
    /// computing while layer ℓ+1 streams + speculatively computes),
    /// accept-rate adjusted.
    ///
    /// - Baseline: both spans fully serial.
    /// - `shared_bus = false` (hideable IO — disk/NVMe/H2D): layer ℓ+1's
    ///   entire span runs inside layer ℓ's span; rejected executions re-run
    ///   compute-only afterwards (`rollback_factor · C₁`).
    /// - `shared_bus = true` (one bus carries both streams — RAM-resident
    ///   CPU decode): per-stream IO time doubles during overlap; the
    ///   model degenerates toward 1× as IO comes to dominate, which is the
    ///   honest answer for that regime.
    pub fn pair_speedup(&self, span0: (f64, f64), span1: (f64, f64), shared_bus: bool) -> f64 {
        let (c0, io0) = span0;
        let (c1, io1) = span1;
        let serial = c0 + io0 + c1 + io1;
        let (host_io0, host_io1) = if shared_bus {
            (2.0 * io0, 2.0 * io1)
        } else {
            (io0, io1)
        };
        // Host span: layer 0's compute overlaps its own IO (streaming
        // matvec); layer 1's IO + (accepted) compute run inside it.
        let host = (c0 + host_io0).max(host_io1 + c1);
        // Rejected executions re-run after the host: compute-dominated
        // (weights are warm by then).
        let reject_tail = (1.0 - self.accept_rate) * self.rollback_factor * c1;
        let net = host + reject_tail;
        if net <= 0.0 {
            return 1.0;
        }
        serial / net
    }
}

/// Fused metrics for one (layer, position) speculative execution — the
/// simulator's measurement record, consumed by the T2 sweep.
#[derive(Debug, Clone, Copy, Default)]
pub struct SpecOutcome {
    /// Measured ratio `‖δℓ‖/‖x_in^ℓ‖`.
    pub ratio: f32,
    /// Final-logits top-1 token agreement (1.0 = same argmax as true run).
    pub top1_match: bool,
    /// KL(true ‖ spec) on the final logits (nats, softmax-normalized).
    pub kl_true_given_spec: f32,
    /// Symmetric-ish divergence KL(spec ‖ true) — reported separately since
    /// direction matters for over/under-confidence.
    pub kl_spec_given_true: f32,
    /// Top-1 logit margin |argmax − runner-up| of the TRUE run (calibration
    /// context: high-margin positions tolerate more divergence).
    pub true_top1_margin: f32,
}

/// Aggregate verdict over a (θ, layer) sweep cell.
#[derive(Debug, Clone, Copy, Default)]
pub struct SweepCell {
    /// Accept threshold used.
    pub theta: f32,
    /// Layer (or aggregated) index this cell summarizes.
    pub layer: usize,
    /// Accepted executions / total.
    pub accept_rate: f32,
    /// Top-1 preservation among ACCEPTED executions.
    pub top1_preserve_given_accept: f32,
    /// Mean KL(true‖spec) among accepted (nats).
    pub mean_kl_given_accept: f32,
    /// Executions in this cell.
    pub n: usize,
}

/// Reduce outcomes into a sweep cell: accept-rate at `theta`, plus the
/// conditional quality among accepted executions.
///
/// This is the T2 verdict shape: the paper's bar asks accept-rate AND top-1
/// preservation TOGETHER — a high accept rate that destroys the argmax is
/// not a pass.
pub fn sweep_cell(outcomes: &[SpecOutcome], theta: f32, layer: usize) -> SweepCell {
    let mut cell = SweepCell {
        theta,
        layer,
        ..Default::default()
    };
    let mut accepted = 0usize;
    let mut top1_ok = 0usize;
    let mut kl_sum = 0.0f32;
    for o in outcomes {
        cell.n += 1;
        if o.ratio < theta {
            accepted += 1;
            if o.top1_match {
                top1_ok += 1;
            }
            kl_sum += o.kl_true_given_spec;
        }
    }
    cell.accept_rate = if cell.n > 0 {
        accepted as f32 / cell.n as f32
    } else {
        0.0
    };
    if accepted > 0 {
        cell.top1_preserve_given_accept = top1_ok as f32 / accepted as f32;
        cell.mean_kl_given_accept = kl_sum / accepted as f32;
    }
    cell
}

/// Numerically stable KL(P‖Q) over two logit vectors (softmax both; P is the
/// `true` distribution in the primary metric).
///
/// KL(P‖Q) = Σᵢ pᵢ·(log pᵢ − log qᵢ) with logs computed via the max-shift
/// trick. Degenerate q mass (softmax underflow) is floored at ln(1e-30) ≈
/// −69 so a near-−∞ spec logit on a token the true run wants reads as a
/// large-but-finite divergence instead of +inf.
pub fn kl_logits(p_logits: &[f32], q_logits: &[f32]) -> f32 {
    debug_assert_eq!(p_logits.len(), q_logits.len());
    let mut p_max = f32::NEG_INFINITY;
    let mut q_max = f32::NEG_INFINITY;
    for i in 0..p_logits.len() {
        if p_logits[i] > p_max {
            p_max = p_logits[i];
        }
        if q_logits[i] > q_max {
            q_max = q_logits[i];
        }
    }
    let p_lse = logsumexp_shifted(p_logits, p_max) + p_max;
    let q_lse = logsumexp_shifted(q_logits, q_max) + q_max;
    const LOG_Q_FLOOR: f32 = -69.0; // ln(1e-30)
    let mut kl = 0.0f32;
    for i in 0..p_logits.len() {
        let lp = p_logits[i] - p_lse;
        let p = lp.exp();
        if p <= 0.0 {
            continue; // negligible mass in P contributes nothing
        }
        let lq = (q_logits[i] - q_lse).max(LOG_Q_FLOOR);
        kl += p * (lp - lq);
    }
    kl
}

/// log-sum-exp for a logit slice with the max pre-subtracted (stable form;
/// exposed for tests).
fn logsumexp_shifted(logits: &[f32], max: f32) -> f32 {
    let mut s = 0.0f32;
    for &l in logits {
        s += (l - max).exp();
    }
    s.ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol * b.abs().max(1.0)
    }

    #[test]
    fn ratio_stats_dominant_and_dormant_layers() {
        // dim=2, 2 positions. Layer 0: δ = 10% of x_in (dominant-ish).
        // Layer 1: δ = 1% (dominant).
        let dim = 2usize;
        let mut streams = vec![Vec::new(); 2];
        // Position 0: layer 0 x_in=[10,0], x_out=[11,0] → ratio 0.1.
        streams[0].extend_from_slice(&[10.0, 0.0, 11.0, 0.0]);
        // layer 1 x_in=[11,0], x_out=[11.11,0] → ratio 0.01.
        streams[1].extend_from_slice(&[11.0, 0.0, 11.11, 0.0]);
        // Position 1: layer 0 x_in=[0,20], x_out=[0,21] → ratio 0.05.
        streams[0].extend_from_slice(&[0.0, 20.0, 0.0, 21.0]);
        streams[1].extend_from_slice(&[0.0, 21.0, 0.0, 21.1]); // ratio ~0.00476

        let rows = layer_ratio_stats(&streams, dim);
        assert_eq!(rows.len(), 2);
        assert!(
            approx(rows[0].median, 0.075, 1e-3),
            "layer0 median {}",
            rows[0].median
        );
        assert!(rows[0].max - rows[0].min > 0.0);
        assert!(
            rows[1].median < 0.05,
            "layer1 should pass bar, median {}",
            rows[1].median
        );
        assert!(!rows[0].passes_paper_bar());
        assert!(rows[1].passes_paper_bar());
        assert_eq!(fraction_layers_under_paper_bar(&rows), 0.5);
    }

    #[test]
    fn zero_input_positions_are_skipped() {
        let dim = 2usize;
        let mut streams = vec![Vec::new()];
        // Zero x_in → skipped; nonzero pair counted.
        streams[0].extend_from_slice(&[0.0, 0.0, 1.0, 0.0]);
        streams[0].extend_from_slice(&[4.0, 0.0, 5.0, 0.0]); // ratio 0.25
        let rows = layer_ratio_stats(&streams, dim);
        assert_eq!(rows[0].n, 1);
        assert!(approx(rows[0].median, 0.25, 1e-3));
    }

    #[test]
    fn trace_reconstruction_matches_paired_streams() {
        // trace: per-layer x_out for 2 layers × 2 positions, dim 2.
        let dim = 2;
        let emb = vec![1.0, 0.0, 0.0, 2.0]; // 2 positions
        let l0 = vec![1.2, 0.0, 0.0, 2.4]; // x_out^0
        let l1 = vec![1.3, 0.0, 0.0, 2.5]; // x_out^1
        let trace = vec![l0, l1];
        let rows = residual_dominance_from_trace(&trace, &emb, dim);
        // Layer 0 pos 0: (1.2−1)/1 = 0.2; pos 1: (2.4−2)/2 = 0.2.
        assert!(approx(rows[0].mean, 0.2, 1e-3));
        // Layer 1 pos 0: (1.3−1.2)/1.2 ≈ 0.0833; pos 1: 0.1/2.4 ≈ 0.04167.
        assert!(approx(
            rows[1].mean,
            (0.083333336 + 0.041666668) / 2.0,
            1e-3
        ));
    }

    #[test]
    fn accept_gate_threshold_semantics() {
        let gate = AcceptGate::default();
        assert!(gate.accepts(0.049));
        assert!(!gate.accepts(0.05));
        assert!(!gate.accepts(0.5));
        // Zero-input is unsafe by construction.
        assert!(!gate.accepts_vectors(&[0.0; 4], &[0.0; 4]));
    }

    #[test]
    fn sweep_cell_conditional_quality() {
        let outcomes = vec![
            SpecOutcome {
                ratio: 0.01,
                top1_match: true,
                kl_true_given_spec: 0.1,
                ..Default::default()
            },
            SpecOutcome {
                ratio: 0.02,
                top1_match: false,
                kl_true_given_spec: 0.3,
                ..Default::default()
            },
            SpecOutcome {
                ratio: 0.9,
                top1_match: true,
                kl_true_given_spec: 9.0,
                ..Default::default()
            },
        ];
        let cell = sweep_cell(&outcomes, 0.05, 0);
        assert_eq!(cell.n, 3);
        assert!(approx(cell.accept_rate, 2.0 / 3.0, 1e-5));
        assert!(approx(cell.top1_preserve_given_accept, 0.5, 1e-5));
        assert!(approx(cell.mean_kl_given_accept, 0.2, 1e-5));
    }

    #[test]
    fn kl_logits_identical_is_zero_and_known_value() {
        let l = vec![1.0, 2.0, 3.0, -1.0];
        assert!(kl_logits(&l, &l).abs() < 1e-5);
        // KL(one-hot-ish ‖ shifted): P concentrated on idx 2, Q on idx 1.
        let p = vec![-10.0, -10.0, 10.0, -10.0];
        let q = vec![-10.0, 10.0, -10.0, -10.0];
        let kl = kl_logits(&p, &q);
        // P ≈ δ_2, Q ≈ δ_1 → KL ≈ log(P2/Q2) = log(e^20) = 20 nats.
        assert!((kl - 20.0).abs() < 0.5, "kl {kl}");
    }

    #[test]
    fn paper_formula_bounded_and_balanced_peak() {
        // (C+IO)/max(C,IO): bounded by 2×; peak at C=IO; →1 at extremes.
        assert!((OverlapLatency::overlap_speedup(1.0, 1.0) - 2.0).abs() < 1e-9);
        // Paper's NVMe regime sanity: IO ≈ 1.5×C → speedup ≈ 1.67 (their
        // measured 1.68×).
        let sp = OverlapLatency::overlap_speedup(1.0, 1.5);
        assert!((sp - 1.6667).abs() < 0.01, "{sp}");
        // Extreme I/O-bound: speedup → 1.
        assert!(OverlapLatency::overlap_speedup(1.0, 100.0) < 1.02);
        // Extreme compute-bound: speedup → 1.
        assert!(OverlapLatency::overlap_speedup(100.0, 1.0) < 1.02);
    }

    #[test]
    fn hideable_io_pair_wins_and_shared_bus_does_not() {
        // 1e9 weights, Q4 4.6 bits, 50 GB/s → IO = 115 ms; C = 2 ms at
        // 2 FLOP/w on 1 TFLOP/s. Hideable-IO (disk-resident): near-2× at
        // full accept.
        let m = OverlapLatency {
            compute_rate: 1e12,
            bandwidth: 50e9,
            accept_rate: 1.0,
            rollback_factor: 1.0,
        };
        let s0 = m.layer_span(1_000_000_000, 4.6, 2.0);
        let s1 = m.layer_span(1_000_000_000, 4.6, 2.0);
        let sp = m.pair_speedup(s0, s1, false);
        assert!(sp > 1.9, "hideable-io full-accept speedup {sp}");

        // Same spans, ONE shared bus (RAM-resident): per-stream IO doubles →
        // the bus is the binding resource; speedup collapses toward 1.
        let sp_shared = m.pair_speedup(s0, s1, true);
        assert!(sp_shared < 1.1, "shared-bus speedup {sp_shared}");

        // Compute-bound + hideable IO + full accept: C dominates, layer 1
        // hides inside layer 0's compute → ≈2×.
        let m2 = OverlapLatency {
            compute_rate: 1e10,
            bandwidth: 200e9,
            accept_rate: 1.0,
            rollback_factor: 1.0,
        };
        let t0 = m2.layer_span(1_000_000_000, 1.58, 2.0); // ternary bits
        let t1 = m2.layer_span(1_000_000_000, 1.58, 2.0);
        let sp2 = m2.pair_speedup(t0, t1, false);
        assert!(sp2 > 1.8, "compute-bound hideable speedup {sp2}");
    }

    #[test]
    fn overlap_latency_rejection_erodes_the_win() {
        // Full rejection in the COMPUTE-BOUND hideable-IO regime: the shadow
        // is compute (not IO) so a rejected speculation wastes it and the
        // recompute pays it back → speedup collapses to ~1.
        let m = OverlapLatency {
            compute_rate: 1e10,
            bandwidth: 200e9,
            accept_rate: 0.0,
            rollback_factor: 1.0,
        };
        let t0 = m.layer_span(1_000_000_000, 1.58, 2.0); // C=200ms, IO≈10ms
        let t1 = m.layer_span(1_000_000_000, 1.58, 2.0);
        let sp = m.pair_speedup(t0, t1, false);
        assert!(sp < 1.05, "compute-bound full-reject speedup {sp}");

        // Contrast — full rejection in the IO-BOUND disk regime: the shadow
        // is IO so the wasted speculative compute cost nothing; the rollback
        // is a cache-hit recompute. Speed stays high; the rejection cost
        // there is QUALITY-side (accepted-but-wrong), not latency-side.
        let m2 = OverlapLatency {
            compute_rate: 1e12,
            bandwidth: 50e9,
            accept_rate: 0.0,
            rollback_factor: 1.0,
        };
        let s0 = m2.layer_span(1_000_000_000, 4.6, 2.0); // C=2ms, IO=115ms
        let s1 = m2.layer_span(1_000_000_000, 4.6, 2.0);
        let sp2 = m2.pair_speedup(s0, s1, false);
        assert!(sp2 > 1.7, "io-bound full-reject speedup {sp2}");
    }

    #[test]
    fn logsumexp_helper() {
        let l = vec![0.0f32, 0.0];
        assert!((logsumexp_shifted(&l, 0.0) - 2f32.ln()).abs() < 1e-5);
    }
}
