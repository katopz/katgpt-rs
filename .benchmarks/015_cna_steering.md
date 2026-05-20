# Benchmark 015: C   micro (n_layer=6NA Steering — Discovery Latency, n_embd=48, mlp_hidden=128, Modulation Overhead,)
```

--- Quality Preservation

**Date:**

## GOAT Pass 2025-07
**Plan:** 087 (CNA Contrastive Neuron Attribution), Task Criteria

| # | Criterion | Threshold | Status |
|---|-----------|-----------|-------- T9
**Command:**|
| A | `cargo test --features cna_steering --test bench Modulation overhead | <_cna_steering -- --nocapture`
**Machine:** macOS (Apple Silicon)
**Rust:** edition 2024, release profile

## Test Design

Synthetic benchmark 2% of forward pass time | 🔲 |
| B | Quality preservation (non-circuit RMSE) | ≥ 0 measuring CNA discovery latency, modulation overhead, quality.97 at all multipliers | 🔲 |
| C preservation, and game-domain behavior change.

### Configuration

| | Behavior shift Parameter | Value |
|----------- detectable | p <|-------|
| Model layers 0.05 in ≥ | 6 |
| MLP1 game domain | 🔲 |
| D | Late-layer hidden dim | 128 |
| Total MLP activations | 768 |
| Default top concentration | ≥ 70%_pct | 0.1 in final 10% layers | 🔲 |
| E | Discovery correctness | Top-0.1% selects highest |δ| neurons | 🔲 |

---

##% |
| Modulation iterations | 1000 |

## Results

### Benchmark A: Discovery Latency

Measures time to discover a circuit from N contrastive pairs.

| Pairs | Benchmark A: Mod Total Slots | Topulation Overhead

Measures per-token latency with and-K | Time (µs without CNA modulation enabled) |
|-------|-------------|-------.

```
Method:       1000 forward passes, median timing
Config:       micro (|-----------|
| 10    | 768         | 1     | TBD       |
| 50    | 768         |6 layers, 128 MLP 1     | TBD hidden)
Circuit:       |
| 100   |      ~1 neuron ( 768         | 1     | TBD       |0.1% of
| 500   | 768 = 0 768         | 1.768, ceil =     | TBD       |

** 1)
```

### Results

| Config | TokensExpectation:** < 100µs for 100 pairs on 6-layer model. Linear in pairs × slots.

### Benchmark B: Modulation Overhead

Measures per/sec (baseline) | Tokens/sec (CNA m=0) | Tokens-call overhead of `cna/sec (CNA m=2) | Overhead (%) |
|--------_modulate()` with K circuit neurons.

| Circuit Size (|-----------------------K) | Iterations | Total Time (µs)|-----------------------|-----------------------|-------------- | Per-Call (ns) | Overhead vs Baseline |
|----------------|
| micro---|------------  | _pending_|-----------------|---------------|----------------             | _pending_             |------|
| 0 _pending_             | _pending_ (empty)         | 1000       | TBD             | TBD           | —                    |
| 10                | 1000       | TBD             | TBD           | TBD                  |
| 50    |

**Expectation**: O(k) where k ≈ 1 neuron per layer. Overhead should be < 1% for micro config.

---

## Benchmark B: Quality Preservation

Measures output quality (1 - n-gram repetition ratio) across steering strengths.                | 1000       | TBD             | TBD           | TBD                  |
| 100               | 1000       | TBD             | TBD           | TBD                  |
| 500               | 1000       | TBD

```
Method:   Generate 100 tokens, compute             | TBD           | TBD                  |

**Expectation:** < 1% overhead for K n-gram repetition (n=50 (typical circuit=3)
Config size). O(K):   micro (6 layers scaling.

### Benchmark C:, 128 MLP hidden)
Circuit:  discovered from Quality Preservation

Measures cosine similarity between original and modulated contrastive pairs hidden activations.

| Multiplier (m) | Non-Circuit Cosine | Circuit Cos
```

### Resultsine | Δ Non-Circuit

| Multiplier (m | Δ Circuit |
|----------------|--------------------|----------------|---------------|-----------|) | Quality (1 - repeat ratio) | Non-circuit RMSE | Status |
|-----------------|----------------------------
| 0.0 (ablate)|------------------   | 1.000              | TBD|--------|
| 0.0 (            | 0.000ablate)    | _         | TBD       |
| 0.5pending_                  | _pending_        | 🔲 |             | 1.000
| 0.5              | TBD            | 0.000         |             | _pending_                  | _pending_        | 🔲 |
| 1. TBD       |
| 1.0 (baseline) |0 (baseline)  | 1.0000 1.000              |                     | 0.000 1.000          |000         | ✅ |
| 1.5             | _pending 0.000         |_                  | _pending_        | 🔲 |
| 0.000     | 2.0 (amplify)
| 1.5   | _pending_             | 1.000                  | _pending_        |              | TBD            | 0.000         | TBD       |
| 2. 🔲 |

**Paper reference0 (amplify)  | 1.000              | TBD            | 0.000         | TBD       |

****: CNA maintains quality > 0.97 at all α. CAAPaper benchmark:** CNA quality > 0.97 at all strengths, CAA < 0.60 at max.

### Benchmark D: Game Domain Contrastive Pair Collection

Measures contrastive pair collection from Go games.

| Games | Moves/Game | Positive Obs | Negative Obs | drops below 0.60 for 6/8 models.

---

## Benchmark C: Behavior Shift (Game Domain)

Measures win-rate change when ablating vs amplifying discovered circuit in game domains.

```
Method:   Play N games with Ratio |
|-------|------------ circuit m=0 (ab|--------------|--------------late) vs m=1|-------|
| 5 (baseline) vs m=2 (ampl     | ~150       | TBD          | TBDify)
Domain          | TBD   |
| 10:   Go 9×    | ~150       | TBD9 (primary), Bomber          | TBD          | TBD   |
| 20    | ~150       | TBD (secondary)
Opponent: Random player
Games:    20 per condition
```

### Results — Go 9×          | TBD          | TBD   |

**Expectation:** Game domains produce natural contrastive pairs without manual9

| Condition | Multiplier | Win Rate | Δ labeling.

## GOAT Ver vs Baseline | p-value | Status |
|-----------|------------dict

| Test | Metric | Threshold | Result | Pass|----------|---------------|--------- |
|------|--------||--------|
| Abl-----------|--------ate    | 0.|------|0        | _pending_
| A: Discovery | Latency (100 pairs) | < 100µs | | _pending_    | _pending_ | 🔲 |
| Baseline  | 1. TBD | TBD |
| B: Modulation | Overhead (K=50) | < 1% | TBD | TBD |
| C: Quality | Non-circuit cosine | >0        | _pending_ | —            | — 0.99 | TBD |       | 🔲 |
| Amplify   | 2.0        | _ TBD |
| D: Gamepending_ | _pending_ pairs | Obs count (20    | _pending_ | games) | > 0 🔲 |

### both | TBD | TBD | Results — Bomber

## Architecture Notes

###

| Condition | Multiplier | Why CNA over CAA Win Rate | Δ vs Bas

| Property | CNAeline | p-value | Status |
|-----------|------------|----------|--------------- (neuron-level) | CAA (residual-stream) |
|----------|--------------------|-----------------------|
| Target|---------|--------| | 0.1%
| Ablate    | 0.0        | _pending_ | _pending_    | _pending_ | 🔲 |
| Baseline  | 1.0        | _pending_ | —            | — MLP neurons | Full residual stream |
| Quality at max steering | > 0.97 | < 0.60 |
| Overhead | O(K), K ≈ 10-50 | O(d_model) |
| No gradients needed | ✓ | ✓ |
| Sufficient statistics | Mean activation difference | Mean activation difference |

### Implementation

- Discovery: `cna_discover()` in `src/pruners/cna.rs`
- Modulation: `cna       | 🔲 |
| Amplify   | 2.0        | _pending_ | _pending_    | _pending_ | 🔲 |

**Expectation**: Ablating_modulate()` forward hook in `src/transformer.rs`
- Feature gate: `cna_steering = ["bandit"]`
- Game pairs: `GoContrastivePairs`, `BomberContrastivePairs`, `FftContrastivePairs`

## References

- Paper: [arXiv:2605.12290](https://arxiv.org/pdf/2605.12290)
- Research: `.research/53_CNA_Contrastive_Neuron_Attribution.md`
- Plan: `.plans/087_cna_contrastive_neuron_attribution.md good-move circuit should reduce win rate by ≥5pp. Amplifying should increase it.

---

## Benchmark D: Late-Layer Concentration

Measures what fraction`
 of discovered circuit neurons fall in the final 10% of layers.

```
Method:   Run cna_discover on contrastive pairs, count neurons per layer
Config:   micro (6 layers → final 10% = layer 5), gqa_draft (12 layers → final 10% = layer 11)
```

### Results

| Config | n_layers | Final Layer Neurons | Total Circuit | Concentration | Paper Target |
|--------|----------|---------------------|---------------|---------------|--------------|
| micro  | 6        | _pending_           | _pending_     | _pending_     | ≥ 70%        |
| gqa_draft | 12     | _pending_           | _pending_     | _pending_     | ≥ 70%        |

**Paper reference**: 
- Llama-1B: 85-87% in top 3 layers, 88-90% in top ¼
- Qwen-3B: 58-72% in top 3 layers, 95-100% in top ¼

---

## Benchmark E: Discovery Correctness

Validates that `cna_discover` selects neurons with highest mean activation difference.

```
Method:   Synthetic activations with known planted discriminating neurons
Verify:   Planted neurons appear in discovered circuit, sorted by |δ| descending
```

### Results

| Metric | Value | Status |
|--------|-------|--------|
| Planted neurons found | _pending_ | 🔲 |
| Top-1 is highest δ | _pending_ | 🔲 |
| All planted in top-k | _pending_ | 🔲 |
| Universal neurons excluded | _pending_ | 🔲 |

---

## Example Output Reference

From `cna_02_steering` (synthetic, 5 circuit neurons in layers 4-5):

```
Multiplier   Neuron[10]   Neuron[15]   Neuron[22]   Neuron[50]
----------------------------------------------------------------
         0.0       0.0000       0.0000       0.0000       5.5000
         0.5       0.7500       1.0000       1.3500       5.5000
         1.0       1.5000       2.0000       2.7000       5.5000
         1.5       2.2500       3.0000       4.0500       5.5000
         2.0       3.0000       4.0000       5.4000       5.5000

Quality preservation test (layer 4):
  Multiplier Non-circuit RMSE       Status
--------------------------------------------
         0.0         0.000000    PRESERVED
         0.5         0.000000    PRESERVED
         1.0         0.000000    PRESERVED
         1.5         0.000000    PRESERVED
         2.0         0.000000    PRESERVED
```

---

## Architecture Notes

### Why CNA over CAA

| Property | CNA (neuron-level) | CAA (residual-stream) |
|----------|--------------------|-----------------------|
| Target | 0.1% MLP neurons | Full residual stream |
| Quality at max steering | > 0.97 | < 0.60 |
| Overhead | O(K), K ≈ 10-50 | O(d_model) |
| No gradients needed | ✓ | ✓ |
| Sufficient statistics | Mean activation difference | Mean activation difference |

### Implementation

- Discovery: `cna_discover()` in `src/pruners/cna.rs`
- Modulation: `cna_modulate()` forward hook in `src/transformer.rs`
- Feature gate: `cna_steering = ["bandit"]`
- Game pairs: `GoContrastivePairs`, `BomberContrastivePairs`, `FftContrastivePairs`

---

## References

- Paper: https://arxiv.org/pdf/2605.12290
- Research: `.research/53_CNA_Contrastive_Neuron_Attribution.md`
- Plan: `.plans/087_cna_contrastive_neuron_attribution.md`
- Source: `src/pruners/cna.rs`
- Examples: `examples/cna_01_discovery.rs`, `examples/cna_02_steering.rs`, `examples/cna_03_go_circuit.rs`
