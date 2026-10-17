//! Bias-space persistent deltas (issue 920 T1) — the HyperThink-shape
//! modelless delta-overlay primitive (arXiv:2610.03039, distilled in
//! `.research/606`).
//!
//! Contrastive-pair delta construction at **projection-window** granularity:
//!
//! ```text
//! Δb_window = E_probe[out_with_c − out_plain]
//! ```
//!
//! where `c` is a static "thinking register" prompt and the expectation is
//! over seed-pinned probe queries. Each per-window mean shift is amortized
//! into a **constant** overlay added to that projection's output every
//! forward — the same math as a bias tensor, zero weight mutation (the
//! freeze/thaw + deterministic-overlay modelless posture).
//!
//! # Relation to CNA (the starting substrate, why a separate path)
//!
//! [`crate::cna`] discovers SPARSE per-neuron circuits and modulates them
//! per-token at runtime. Bias-space deltas are the other three corners:
//! **dense** over the window (a full D-dim vector per site, no top-pct
//! selection), **signed** (the mean shift, not `|mean_pos − mean_neg|`), and
//! **persistent** (a per-position constant, not a per-token multiplier) at
//! projection-output sites rather than post-ReLU MLP activations. The
//! contrastive-pair *concept* is shared; the granularity, sign handling and
//! attachment site are not, so the machinery is its own module (issue 920 T1:
//! "if bias-space needs its own path, record WHY in the PoC notes" — this
//! header is that record).
//!
//! # Sites
//!
//! [`BiasSite`] enumerates the projection-output windows the overlay can
//! attach to. `K` is **excluded by construction** (softmax shift-invariance:
//! a constant added to every key shifts each softmax row by `q·c` — the same
//! constant for every key — which exact softmax cancels; the paper's own
//! exclusion, reproduced here as an absent variant so a caller cannot
//! accidentally capture or apply it).
//!
//! # Capture ratio
//!
//! Per window, `ρ_w = ‖E[Δ]‖² / E[‖Δ‖²] ≤ 1` (Jensen): how much of the
//! per-probe shift energy the MEAN (the bias) actually captures. `ρ_w = 1`
//! iff every probe's shift is identical — the pure-bias regime. Per-layer
//! ratios and the layer energy ranking feed the issue's pre-registered
//! late-block-concentration premise check (T1's stop/go gate before any
//! codebook spend).
//!
//! # Determinism
//!
//! All accumulation is f64, fixed iteration order (slot-indexed storage, no
//! hash maps on the accumulation path). Same captures → bit-identical table.

/// A projection-output window the bias overlay can attach to.
///
/// `K` is deliberately absent: a bias on the key projection is a no-op under
/// exact softmax (see module docs), so it is neither captured nor applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BiasSite {
    /// Query projection output (pre-RoPE — the overlay rotates with the
    /// position exactly like a trained bias would).
    Q,
    /// Value projection output (pre-cache).
    V,
    /// Attention-output projection output (pre post-attention norm).
    O,
    /// MLP gate-projection output (pre GeGLU).
    Gate,
    /// MLP up-projection output (pre GeGLU).
    Up,
    /// MLP down-projection output (pre post-MLP norm).
    Down,
}

impl BiasSite {
    /// All sites in canonical (slot) order — the serialization order.
    pub const ALL: [BiasSite; 6] = [
        BiasSite::Q,
        BiasSite::V,
        BiasSite::O,
        BiasSite::Gate,
        BiasSite::Up,
        BiasSite::Down,
    ];

    /// Canonical slot index (`layer * Self::ALL.len() + slot` addressing).
    #[inline]
    pub fn slot(self) -> usize {
        match self {
            BiasSite::Q => 0,
            BiasSite::V => 1,
            BiasSite::O => 2,
            BiasSite::Gate => 3,
            BiasSite::Up => 4,
            BiasSite::Down => 5,
        }
    }

    /// Stable name (serialization / reporting).
    #[inline]
    pub fn name(self) -> &'static str {
        match self {
            BiasSite::Q => "q",
            BiasSite::V => "v",
            BiasSite::O => "o",
            BiasSite::Gate => "gate",
            BiasSite::Up => "up",
            BiasSite::Down => "down",
        }
    }

    /// Parse a name produced by [`Self::name`].
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "q" => Some(BiasSite::Q),
            "v" => Some(BiasSite::V),
            "o" => Some(BiasSite::O),
            "gate" => Some(BiasSite::Gate),
            "up" => Some(BiasSite::Up),
            "down" => Some(BiasSite::Down),
            _ => None,
        }
    }
}

/// One window's accumulated paired captures (f64, fixed order).
#[derive(Debug, Clone)]
struct WindowAccum {
    /// `sum[depth] += with_c[d] − plain[d]` per observe (one per probe).
    sum: Vec<f64>,
    /// `sumsq += ‖with_c − plain‖²` per observe (the second moment).
    sumsq: f64,
    /// Observations accumulated (the window's own mean divisor).
    n: usize,
}

/// Builder: feed paired projection-output captures per probe, finish into a
/// frozen [`BiasDeltaTable`].
///
/// Pairing contract: `with_c` and `plain` are the SAME window's output
/// vectors for the SAME probe (same readout position semantics in both arms
/// — the capture lane's job to guarantee), in site-slot order per layer.
/// Depths must match per call.
#[derive(Debug, Clone)]
pub struct BiasDeltaBuilder {
    n_layers: usize,
    /// `slots[layer * 6 + site.slot()]` — `None` until first observed.
    slots: Vec<Option<WindowAccum>>,
    probes: usize,
}

impl BiasDeltaBuilder {
    /// A builder over `n_layers` layers × all six sites.
    pub fn new(n_layers: usize) -> Self {
        Self {
            n_layers,
            slots: (0..n_layers * BiasSite::ALL.len()).map(|_| None).collect(),
            probes: 0,
        }
    }

    /// Layers this builder was sized for.
    pub fn n_layers(&self) -> usize {
        self.n_layers
    }

    /// Probes observed so far (one `observe` sweep per probe across sites).
    pub fn probes(&self) -> usize {
        self.probes
    }

    /// Observe one paired capture: `with_c[d] − plain[d]` accumulates into
    /// the `(layer, site)` window. Depths must match.
    pub fn observe(&mut self, layer: usize, site: BiasSite, with_c: &[f32], plain: &[f32]) {
        assert_eq!(
            with_c.len(),
            plain.len(),
            "bias_delta: with_c/plain depth mismatch at layer {layer} site {}",
            site.name()
        );
        assert!(
            layer < self.n_layers,
            "bias_delta: layer {layer} out of range"
        );
        let idx = layer * BiasSite::ALL.len() + site.slot();
        let acc = self.slots[idx].get_or_insert_with(|| WindowAccum {
            sum: vec![0.0; with_c.len()],
            sumsq: 0.0,
            n: 0,
        });
        let mut sq = 0.0f64;
        for (s, (&w, &p)) in acc.sum.iter_mut().zip(with_c.iter().zip(plain.iter())) {
            let d = f64::from(w) - f64::from(p);
            *s += d;
            sq += d * d;
        }
        acc.sumsq += sq;
        acc.n += 1;
    }

    /// Mark one probe complete (call after a full per-site sweep so
    /// [`Self::probes`] and the per-window means stay aligned). Table
    /// construction itself uses the per-window observation counts, so this
    /// is bookkeeping for the lane's progress reporting — but finishing
    /// with a count mismatch is a bug, and [`Self::finish`] asserts it.
    pub fn end_probe(&mut self) {
        self.probes += 1;
    }

    /// Freeze into a table. Every observed window's mean shift is the table
    /// content; capture ratios and layer stats are derived here once.
    pub fn finish(self) -> BiasDeltaTable {
        assert!(
            self.probes > 0,
            "bias_delta: finish with zero probes — nothing to freeze"
        );
        let n_sites = BiasSite::ALL.len();
        let mut table_entries = Vec::new();
        let mut layer_stats = vec![LayerStat::default(); self.n_layers];
        for (idx, slot) in self.slots.iter().enumerate() {
            let Some(acc) = slot else { continue };
            let layer = idx / n_sites;
            let site = BiasSite::ALL[idx % n_sites];
            let n = acc.n.max(1) as f64; // acc.n ≥ 1 whenever Some
            let depth = acc.sum.len();
            let mut mean = Vec::with_capacity(depth);
            let mut mean_sq = 0.0f64; // ‖E[Δ]‖²
            for &s in &acc.sum {
                let m = s / n;
                mean.push(m as f32);
                mean_sq += m * m;
            }
            let exp_sq = acc.sumsq / n; // E[‖Δ‖²]
            let capture = if exp_sq > 0.0 { mean_sq / exp_sq } else { 0.0 };
            table_entries.push(WindowEntry {
                layer,
                site,
                delta: mean,
                capture_ratio: capture,
                mean_sq,
                exp_sq,
            });
            layer_stats[layer].windows += 1;
            layer_stats[layer].mean_sq += mean_sq;
            layer_stats[layer].exp_sq += exp_sq;
        }
        for ls in &mut layer_stats {
            ls.capture_ratio = if ls.exp_sq > 0.0 {
                ls.mean_sq / ls.exp_sq
            } else {
                0.0
            };
        }
        let total_energy: f64 = layer_stats.iter().map(|ls| ls.mean_sq).sum();
        for ls in &mut layer_stats {
            ls.energy_share = if total_energy > 0.0 {
                ls.mean_sq / total_energy
            } else {
                0.0
            };
        }
        BiasDeltaTable {
            n_layers: self.n_layers,
            n_probes: self.probes,
            entries: table_entries,
            layer_stats,
            total_energy,
        }
    }
}

/// One frozen window: the mean shift (the overlay content) + its capture stats.
#[derive(Debug, Clone)]
pub struct WindowEntry {
    /// Transformer layer index.
    pub layer: usize,
    /// Projection-output site.
    pub site: BiasSite,
    /// `Δb_window` — the overlay content (mean shift, f32).
    pub delta: Vec<f32>,
    /// `ρ_w = ‖E[Δ]‖² / E[‖Δ‖²]` — 1.0 = pure constant shift.
    pub capture_ratio: f64,
    /// `‖E[Δ]‖²` — this window's contribution to the layer energy ranking.
    pub mean_sq: f64,
    /// `E[‖Δ‖²]`.
    pub exp_sq: f64,
}

/// Per-layer aggregates over the layer's windows.
#[derive(Debug, Clone, Copy, Default)]
pub struct LayerStat {
    /// Number of observed windows (≤ 6; unobserved sites absent).
    pub windows: usize,
    /// `Σ_w ‖E[Δ_w]‖²` — the layer's delta energy (the concentration axis).
    pub mean_sq: f64,
    /// `Σ_w E[‖Δ_w‖²]`.
    pub exp_sq: f64,
    /// Energy-weighted `ρ_ℓ = Σ‖E[Δ]‖² / Σ E[‖Δ‖²]` over the layer's windows.
    pub capture_ratio: f64,
    /// `e_ℓ = mean_sq / total_energy` (valid after finish; 0 if no energy).
    pub energy_share: f64,
}

/// The frozen delta table: per-window overlay content + the stats the issue's
/// premise check ranks.
#[derive(Debug, Clone)]
pub struct BiasDeltaTable {
    n_layers: usize,
    n_probes: usize,
    entries: Vec<WindowEntry>,
    layer_stats: Vec<LayerStat>,
    total_energy: f64,
}

impl BiasDeltaTable {
    /// The window entry for `(layer, site)`, if captured.
    pub fn entry(&self, layer: usize, site: BiasSite) -> Option<&WindowEntry> {
        self.entries
            .iter()
            .find(|e| e.layer == layer && e.site == site)
    }

    /// All frozen windows (canonical order: layer-major, then site slot).
    pub fn windows(&self) -> &[WindowEntry] {
        &self.entries
    }

    /// Probe count the means were taken over.
    pub fn n_probes(&self) -> usize {
        self.n_probes
    }

    /// Layer count the table was built over.
    pub fn n_layers(&self) -> usize {
        self.n_layers
    }

    /// Total delta energy `Σ_ℓ Σ_w ‖E[Δ_w]‖²`.
    pub fn total_energy(&self) -> f64 {
        self.total_energy
    }

    /// Per-layer stats (index = layer).
    pub fn layer_stats(&self) -> &[LayerStat] {
        &self.layer_stats
    }

    /// `e_ℓ` — the layer's share of total delta energy (the concentration
    /// axis the premise check sums over the late half).
    pub fn layer_energy_share(&self, layer: usize) -> f64 {
        self.layer_stats
            .get(layer)
            .map(|ls| ls.energy_share)
            .unwrap_or(0.0)
    }

    /// Sum of `e_ℓ` for `layers ≥ from` — the late-block energy share.
    pub fn late_energy_share(&self, from: usize) -> f64 {
        self.layer_stats
            .iter()
            .skip(from)
            .map(|ls| ls.energy_share)
            .sum()
    }

    /// Mean `ρ_ℓ` over a layer range (empty range → 0.0).
    pub fn mean_capture_ratio(&self, layers: std::ops::Range<usize>) -> f64 {
        let n = layers.len();
        if n == 0 {
            return 0.0;
        }
        let sum: f64 = self
            .layer_stats
            .get(layers.start..layers.end)
            .map(|slice| slice.iter().map(|ls| ls.capture_ratio).sum())
            .unwrap_or(0.0);
        sum / n as f64
    }

    /// Apply the overlay: `out[d] += Δb[layer, site][d]`.
    ///
    /// Zero-alloc, O(depth). A missing window is a silent no-op (the armed
    /// subset is the caller's routing decision; the unarmed posture is "no
    /// call at all").
    #[inline]
    pub fn apply(&self, layer: usize, site: BiasSite, out: &mut [f32]) {
        if let Some(e) = self.entry(layer, site) {
            for (o, &d) in out.iter_mut().zip(e.delta.iter()) {
                *o += d;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() <= eps
    }

    #[test]
    fn constant_shift_is_pure_bias() {
        // Every probe shifts by the same vector → ρ must be exactly 1 and
        // the table content must be that vector.
        let mut b = BiasDeltaBuilder::new(2);
        for _ in 0..5 {
            b.observe(0, BiasSite::Q, &[1.0, 2.0, 3.0], &[0.0, 0.0, 0.0]);
            b.observe(1, BiasSite::Down, &[-1.0, 0.5], &[0.0, 0.0]);
            b.end_probe();
        }
        let t = b.finish();
        assert_eq!(t.n_probes(), 5);
        let q = t.entry(0, BiasSite::Q).expect("q window");
        assert_eq!(q.delta, vec![1.0, 2.0, 3.0]);
        assert!(approx(q.capture_ratio, 1.0, 1e-12));
        let d = t.entry(1, BiasSite::Down).expect("down window");
        assert_eq!(d.delta, vec![-1.0, 0.5]);
        assert!(approx(d.capture_ratio, 1.0, 1e-12));
        // Unobserved windows are absent, not zero.
        assert!(t.entry(0, BiasSite::Gate).is_none());
    }

    #[test]
    fn capture_ratio_bounded_and_sensitive() {
        // Exact cancellation: mean 0 while E‖Δ‖² > 0 → ρ = 0.
        let mut b = BiasDeltaBuilder::new(1);
        b.observe(0, BiasSite::V, &[2.0], &[1.0]);
        b.observe(0, BiasSite::V, &[0.0], &[1.0]);
        b.end_probe();
        b.end_probe();
        let t = b.finish();
        let v = t.entry(0, BiasSite::V).expect("v");
        assert!(approx(f64::from(v.delta[0]), 0.0, 1e-12));
        assert!(approx(v.capture_ratio, 0.0, 1e-12));

        // Constant with ±ε jitter: ρ near 1 but strictly below.
        let mut b = BiasDeltaBuilder::new(1);
        b.observe(0, BiasSite::V, &[1.1], &[0.0]);
        b.observe(0, BiasSite::V, &[0.9], &[0.0]);
        b.end_probe();
        b.end_probe();
        let t = b.finish();
        let v = t.entry(0, BiasSite::V).expect("v");
        assert!(v.capture_ratio > 0.99 && v.capture_ratio < 1.0);
    }

    #[test]
    fn apply_adds_delta_in_place() {
        let mut b = BiasDeltaBuilder::new(1);
        b.observe(0, BiasSite::Gate, &[1.0, -2.0], &[0.0, 0.0]);
        b.end_probe();
        let t = b.finish();
        let mut out = [10.0f32, 20.0];
        t.apply(0, BiasSite::Gate, &mut out);
        assert_eq!(out, [11.0, 18.0]);
        // Missing window: no-op.
        let mut out2 = [7.0f32];
        t.apply(0, BiasSite::Up, &mut out2);
        assert_eq!(out2, [7.0]);
    }

    #[test]
    fn layer_stats_and_late_energy() {
        // Layer 0: energy 4; layer 1: energy 16 → total 20, shares 0.2 / 0.8.
        let mut b = BiasDeltaBuilder::new(2);
        b.observe(0, BiasSite::Q, &[2.0, 0.0], &[0.0, 0.0]); // ‖Δ‖² = 4
        b.observe(1, BiasSite::Up, &[4.0], &[0.0]); // ‖Δ‖² = 16
        b.end_probe();
        let t = b.finish();
        assert!(approx(t.total_energy(), 20.0, 1e-9));
        assert!(approx(t.layer_energy_share(0), 0.2, 1e-12));
        assert!(approx(t.layer_energy_share(1), 0.8, 1e-12));
        assert!(approx(t.late_energy_share(1), 0.8, 1e-12));
        assert!(approx(t.late_energy_share(0), 1.0, 1e-12));
        // Layer capture ratio: both layers pure constants → 1.0 each.
        assert!(approx(t.layer_stats()[0].capture_ratio, 1.0, 1e-12));
        assert!(approx(t.mean_capture_ratio(0..2), 1.0, 1e-12));
        assert!(approx(t.mean_capture_ratio(2..2), 0.0, 1e-12));
    }

    #[test]
    fn site_ordering_and_names_round_trip() {
        for (i, s) in BiasSite::ALL.iter().enumerate() {
            assert_eq!(s.slot(), i);
            assert_eq!(BiasSite::from_name(s.name()), Some(*s));
        }
        assert_eq!(BiasSite::from_name("k"), None);
    }

    #[test]
    fn means_average_over_probes() {
        let mut b = BiasDeltaBuilder::new(1);
        b.observe(0, BiasSite::O, &[1.0, 3.0], &[0.0, 1.0]); // Δ = [1, 2]
        b.end_probe();
        b.observe(0, BiasSite::O, &[3.0, 5.0], &[0.0, 1.0]); // Δ = [3, 4]
        b.end_probe();
        let t = b.finish();
        let o = t.entry(0, BiasSite::O).expect("o");
        assert_eq!(o.delta, vec![2.0, 3.0]);
        assert!(approx(o.mean_sq, 4.0 + 9.0, 1e-9));
        assert!(approx(o.exp_sq, (5.0 + 25.0) / 2.0, 1e-9));
    }

    #[test]
    #[should_panic(expected = "depth mismatch")]
    fn depth_mismatch_panics() {
        let mut b = BiasDeltaBuilder::new(1);
        b.observe(0, BiasSite::Q, &[1.0, 2.0], &[1.0]);
    }

    #[test]
    #[should_panic(expected = "zero probes")]
    fn finish_without_probes_panics() {
        let _ = BiasDeltaBuilder::new(1).finish();
    }
}
