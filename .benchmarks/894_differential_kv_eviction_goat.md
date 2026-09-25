# Bench 894 — differential KV eviction GOAT (Issue 882 P3)

**Status:** COMPLETE at the primitive level. All gates pass on the 4 runs taken after the two first-run findings were fixed (G1c re-specified, G4 selector repaired). One pre-registered bar FAILED and was re-specified; both are recorded (G1c).

- **G1a:** needle retrieval at S = 2.5 is **0.945 at 25% cache / 0.977 at 50%**, against full cache 1.000 (bar ≥ full − 1/16). The λ = 0 max-recent baseline reads 0.680 / 0.758, the shipped usage-rate (H2O-class) score **0.000 / 0.000**, and the pinned-random null 0.289 / 0.531.
- **G1b:** the trap-4 negative is pinned. Under distractor spikes with a generic future, the differential policy is **1.35× / 1.47× worse** than the baseline on output error.
- **G1c:** sink exemption. Pinned, 0 sinks are lost at every λ. Unpinned, sinks are lost 32/32 at λ = 1.5. The pre-registered λ = 1 bar FAILED (0/32).
- **G1d:** λ = 0 ≡ max-recent. Specificity is `to_bits`-identical and the eviction order identical in 28/28 cells.
- **G2:** **0.95–1.24 ns/key/step** (best-of, bar ≤ 2.0). That is **−15.2% to −15.8%** against the shipped usage-rate `observe` loop (paired).
- **G3:** no eviction is a no-op, bit-identical.
- **G4:** 0 allocations. Selection included, after a fix to the shipped selector (see finding 5).

Feature `differential_kv_eviction` (katgpt-core, implies `kv_sink_window` → `usage_rate_eviction`) stays **OPT-IN**. The model-bound G1 (multi-needle @64K on Bonsai/Qwen at 25%/50% cache ≥ full − ε, no-eviction bit-identical) belongs to the consumer: riir-infer Issue 012.

**Box:** M3 Max (16 cores, 64 GB), macOS 26.6.2, **AC power, battery 100% charged**. The gate prints `pmset -g batt`, `vm.swapusage`, `vm_stat` free pages and loadavg itself.

- loadavg **14–17** (1 min) across the four runs: sibling agent sessions plus concurrent cargo builds in other target dirs. This is a LOADED box.
- free + speculative 4.9–7.6 GB.
- swap 1.08 of 2.0 GB used.
- Release profile.
- G2 uses the shared interleaved `ab_median_ratio` (15 rounds × 200 iters/arm) and `best_of_us` (200 samples), with `black_box` on the mass row (arguments) and one specificity read per call (result).

Run: `cargo test -p katgpt-core --release --features differential_kv_eviction --test bench_894_differential_kv_eviction_goat -- --nocapture`

## What the primitive is

`kv_eviction::differential` (a sibling of `dying.rs`, consuming the shipped substrate) implements

```text
d_j(t) = a_j(t) − λ·μ_j(t−1)
μ_j(t) = μ_j(t−1) + β(a_j(t) − μ_j(t−1))
specificity_j = max over recent queries of d_j
```

- **Reference:** the pre-query `μ`, so a spike cannot cancel itself.
- **Window:** "recent" is a two-bucket window. It holds the max over the current `W`-query bucket and over the previous one, so it covers `W` to `2W−1` queries at 3 `f32`/key. There is no ring buffer.
- **Selection:** the shipped `kv_eviction::select_evict_into`.
- **Sinks:** exempt via `kv_sink_window::sink_pin_mask_into`.
- **Admission:** a fresh key starts at `μ = 0`, a built-in recency prior.

## Fixture (modelless, synthetic, 8 seeds)

The fixture is one head with N = 1024 keys, all admitted at t = 0, followed by 192 observation queries (W = 64, so the score covers exactly the last 128 queries and the admission bucket has aged out). Each attention row is a softmax over per-key logits:

- 4 sinks at logit 6 (σ 0.05).
- ~55% hubs at logit 2 (σ 0.25), the stopword/formatting class.
- 16 needles and the filler at logit 0 (σ 0.5). Each needle is mentioned by 2 recent queries with a spike to logit S.

After eviction to budget B, 16 probes each spike their needle to logit 9, and retrieval counts a hit when the argmax over the retained keys is that needle. Pre-registered: λ = 0.8 (the λinit asymptote, trap 5), β = 0.1, W = 64, S = 2.5.

## What was measured

| gate | cell | result | bar |
|---|---|---|---|
| G1a | S=2.5, 25% | diff **0.945** · max-recent 0.680 · usage-rate 0.000 · random 0.289 · full 1.000 | ≥ full − 1/16, > baseline, beats random (`beats_random_prompt_pin`) |
| G1a | S=2.5, 50% | diff **0.977** · max-recent 0.758 · usage-rate 0.000 · random 0.531 · full 1.000 | same |
| G1a sweep | S=1.5 / 2.0 / 3.0 / 4.0 / 6.0 @25% | diff 0.125 / 0.625 / 0.984 / 1.000 / 1.000 vs max-recent 0.016 / 0.156 / 0.961 / 0.992 / 1.000 | (reported) |
| G1a λ grid | S=2.5, 25% | λ 0 → 0.680 · 0.25 → 0.750 · 0.5 → 0.867 · 0.8 → 0.945 · 1.0 → 0.977 · 1.5 → 0.992 | (reported) |
| G1b | distractor spikes, generic future, 25% | output rel err diff **1.581** vs max-recent 1.173 (**1.35×**), random 1.286 | diff > baseline (the negative must reproduce) |
| G1b | same, 50% | diff **0.546** vs max-recent 0.372 (**1.47×**), random 0.622 | same |
| G1b | needle fixture, generic future | 25%: 0.901 vs 0.880 · 50%: 0.223 vs 0.219 | (reported) |
| G1c | pinned (shipped sink mask), λ ∈ {0.8, 1.0, 1.5} | **0** sinks evicted / 24 runs | 0 |
| G1c | unpinned λ = 1.5 | **32/32** sinks evicted | ≥ 1 |
| G1c | unpinned λ = 1.0 (pre-registered) | **0/32**: bar FAILED | ≥ 1 |
| G1d | λ=0 vs an independent history reference, T ∈ {1, 37, 64, 65, 128, 150, 192} × 4 seeds | 28/28 cells `to_bits` + eviction order identical | exact |
| G2 | observe step N=4096, best-of | 3.875 / 3.875 / 5.083 / 4.208 µs → **0.946 / 0.946 / 1.241 / 1.027 ns/key** | ≤ 2.0 ns/key |
| G2 | paired vs shipped usage-rate `observe` loop | median **0.840 / 0.848 / 0.846 / 0.844** | (reported) |
| G2 | specificity + `select_evict_into` k=N/2, N=4096 | 117–131 µs best-of | (reported) |
| G3 | budget ∈ {N, N+1, 2N} | 0 rows selected; 512 output floats `to_bits`-identical | exact |
| G4 | 200 observe + specificity, N=4096 | 0 allocs | 0 |
| G4 | 200 observe + 10 sink-exempt selections, N=4096 | **10 → 0** allocs after the selector fix | 0 |
| G4 | unstable selector vs stable reference, 97-value ties, N=4096 | 3723 ranked rows identical | exact |

## Reading it honestly

1. **The win is regime-conditional, and the sweep shows the regime.** Differential beats max-recent only where the needle's recent evidence is weaker than a hub's max-over-window mass (S ≲ 3 here). By S ≥ 4 both saturate. At S ≤ 1.5 both fail and the random null (0.289) beats both. The λ = 0.8 claim is `0.945 vs 0.680` at the pre-registered cell, not a universal gain.
2. **The shipped usage-rate score (cumulative mass / age, H2O-class) retrieves ZERO needles at both budgets**, worse than random. With 55% hubs, cumulative mass ranks every hub above every needle. This is the "hubs crowd out needles" failure in its purest form, and it is a finding about the shipped `kv_eviction::score` on hub-heavy contexts. It does not change that score's own Bench 697 verdict, which was measured on a different axis (runaway).
3. **Trap 4 is real and measured.** When the recent window holds one-off spikes the future will not ask about, the differential policy keeps them and drops the hubs a generic future needs: **1.35–1.47× the baseline's output error**, and worse than the random null at 25%. The same needle fixture read through a generic future costs only ~2% (0.901 vs 0.880 at 25%, 0.223 vs 0.219 at 50%), so the damage is specific to distractor-heavy windows. It is silent, exactly as the trap predicts. The gate asserts the negative, so a change that makes it vanish reds.
4. **The sink trap does not fire where the issue expected it.** The pre-registered bar was "unpinned at λ = 1 loses a sink", and it measured 0/32. A sink's mass is ~55× a hub's, so even its small relative fluctuation above its own EMA out-scores the context. The trap fires at **λ > 1**: at λ = 1.5 the sink's specificity goes negative and all 32 are evicted first. **That is exactly where the λ grid peaks** (1.5 → 0.992), so the exemption is load-bearing in the regime a λ-tuner would choose. The gate was re-specified there and the failed bar is kept as a printed line.
5. **G4 found a latent defect in the SHIPPED selector.** `kv_eviction::select_evict_into` used the stable `sort_by`, which heap-allocates its merge scratch once the slice outgrows the on-stack buffer: 1 alloc per call at n = 4096. Its own G4 test ran at n = 128 and could not see it. The comparator is a total order over distinct indices (the index tie-break), so `sort_unstable_by` yields the identical sequence and allocates nothing. It was fixed in the same commit, together with the two same-class sorts in `kv_sink_window::select_evict_windowed`. The equivalence is gated at n = 4096 with heavy ties, and Bench 841 G4 (alloc_tracking) and the 52 kv_eviction/kv_sink_window lib tests are green after it.
6. **Bookkeeping is cheaper than the scorer it sits beside.** The SoA `observe_query` (EMA + subtract + max, 3 f32/key) vectorises. It measures **~16% faster** than the shipped per-row `observe` loop over `UsageScoreTable` (paired median 0.84, 4 runs), at ~1 ns/key on a loaded box. Selection (the n log n sort) dominates at ~120 µs per N=4096 call, so a consumer should select every few steps, not every step.
7. **λ = 0.8 is a prior (trap 5).** The grid is monotone up to λ = 1.5 on this fixture. A consumer picks λ on its own needles and must keep the sink pin whenever λ > 1.
