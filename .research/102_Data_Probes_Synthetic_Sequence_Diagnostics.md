# Research 102: Data Probes — Synthetic Sequence Diagnostics for LLM Understanding

> **Paper:** [Let's Develop Data Probes to Fundamentally Understand How Data Affects LLM Performance](https://arxiv.org/pdf/2605.18801) — Wang, Woisetschläger, Jacobsen, Ji (Exeter/TUM/Toronto/UF), ICML 2026
> **Date:** 2026-05-25
> **Related Research:** 061 (Entropy Anomaly Detection), 037 (REAP Model-Based/Modelless Duality), 076 (SR²AM), 090 (Epiplexity), 103 (State Distribution View)
> **Related Plans:** 141 (Data Probe Diagnostics — katgpt-rs core)
> **Verdict: MEDIUM-HIGH VALUE — Strong conceptual alignment with our entropy anomaly + review metrics infrastructure. The typical-set regime classification (over-conservative / typical / uncertain) maps directly onto our existing `EntropyAnomalySummary` and `token_entropy()` primitives. The formal validation protocol (C1–C4, IV/EV) is a methodology upgrade for ALL our GOAT proofs. Not a Super GOAT — no game-specific knowledge, pure diagnostics tooling.**

---

## TL;DR

The paper advocates generating **synthetic sequences from known random processes** (data probes) to systematically study how data characteristics affect LLM behavior. Each data probe has a known probability distribution, enabling computation of ground-truth NLL, typical-set regime classification, and controlled intervention studies. The paper demonstrates with Markov chain probes on GPT-2 that typical-set analysis reveals over-conservative (greedy → repetitive), typical (sampling T=1–1.3 → meaningful), and uncertain (sampling T=1.5 → hallucinated) regimes.

**Key insight for us:** We already compute Shannon entropy (`token_entropy`), track entropy anomalies (`EntropyAnomalySummary`), and have underspecification scoring (`underspecification_score`). The missing piece is the **typical-set regime classifier** — mapping observed NLL against a known reference distribution to classify model output quality in real-time.

---

## Core Ideas

### 1. Data Probe Formal Definition (Eq. 1)

```
Π = (P, M, H, F)
```

| Component | Meaning | Our Analogue |
|-----------|---------|--------------|
| P | Known generative process + intervention controls | Our game FSM generators (Bomber, Go, Monopoly) |
| M | Measurable diagnostics | `EntropyAnomalySummary`, `ReviewSummary`, GOAT proofs |
| H | Testable claims | Our GOAT proof structure (threshold + metric) |
| F | Falsification rules | Our `assert!` thresholds in GOAT tests |

**Already captured:** Our GOAT proof system is already an informal version of C1–C4. The paper formalizes what we do ad-hoc.

### 2. Typical-Set Regime Classification (Figure 4)

For a known distribution with entropy rate H and tolerance ε:

| Regime | Condition | LLM Behavior | Our Equivalent |
|--------|-----------|--------------|----------------|
| Over-conservative | NLL < H - ε | Repetitive, mode-collapsed | High-confidence, low-entropy predictions |
| Typical | H - ε ≤ NLL ≤ H + ε | Meaningful, well-calibrated | Desired operating range |
| Uncertain | NLL > H + ε | Hallucinated, off-distribution | High-entropy anomaly, PPoT rescue zone |

**Already captured:** Our `identify_high_entropy_positions()` (Plan 027 PPoT) does binary classification. We lack the three-regime classification and the known-reference-distribution comparison.

### 3. Validation Protocol (C1–C4 + IV/EV)

Four criteria for valid data probes:
- **C1:** Known process fully specified and samplable
- **C2:** Controllable knobs with interpretable interventions
- **C3:** Diagnostic metrics computable
- **C4:** Pre-declared falsification conditions

Two-layer evaluation:
- **IV(h):** Internal validity — probe-side directional predictions hold under interventions
- **EV(h):** External validity — matched real-side directional effects hold

**Already captured (partially):** Our GOAT proofs check C1 (test fixtures), C3 (benchmark metrics), C4 (thresholds). We lack C2 (formal intervention knobs) and EV (transfer to real data).

### 4. Markov Chain Probe Generation (Appendix D)

Generate transition matrices via Dirichlet sampling, select by target entropy rate. The entropy rate of a Markov chain:

```
H(P) = -Σᵢ πᵢ Σⱼ Pᵢⱼ log Pᵢⱼ
```

**Not yet implemented.** This is the core algorithmic contribution we'd add.

---

## Distillation to Our Architecture

### What We Already Have (No New Code Needed)

| Existing Component | Paper Concept | Alignment |
|-------------------|---------------|-----------|
| `token_entropy()` (PPoT) | Shannon entropy of marginal distribution | Exact match |
| `EntropyAnomalySummary` | Entropy tracking across session | Subset — mean/max/count, no regime labels |
| `is_high_entropy_session()` | Detecting anomalous entropy regimes | Binary version of three-regime classification |
| `underspecification_score()` | Normalized entropy as quality metric | Same math, different name |
| `ReviewMetrics` (Plan 032) | Diagnostic recording infrastructure | C3-compatible |
| GOAT proof system | Falsification protocol | C4-compatible |
| `ConfiguratorContext.entropy_bin` | Coarse entropy discretization | Crude version of regime classification |

### What's Missing (New Code Needed)

| Gap | Description | Complexity |
|-----|-------------|------------|
| **Markov chain generator** | Dirichlet-sampled transition matrices with target entropy | ~200 lines |
| **Typical-set regime classifier** | Three-way (conservative/typical/uncertain) label from NLL vs H±ε | ~50 lines |
| **Reference distribution NLL** | Computing log-probability of sequences against known distribution | ~100 lines |
| **Formal claim cards** | Structured claim objects with IV/EV verdict tracking | ~150 lines |
| **Intervention framework** | Temperature/knob sweeps with controlled contrasts | ~100 lines |

### Where It Fits

```
katgpt-rs (MIT — generic diagnostics)
├── Markov chain probe generator
├── Typical-set regime classifier
├── NLL computation against known distribution
├── Claim card struct (C1–C4, IV/EV)
└── Intervention sweep framework

riir-ai (Private — game-specific probes)
└── Game FSM probe generators
    ├── Bomber FSM → known transition matrix
    ├── Go state → position-entropy reference
    └── Monopoly FSM → action frequency reference
```

---

## Modelless vs Model-Based Split

| Aspect | Mode | Why |
|--------|------|-----|
| Markov chain generation | **Modelless** | Pure random process, no neural net |
| NLL computation against reference | **Modelless** | Known distribution → exact log-prob |
| Typical-set regime labeling | **Modelless** | Comparison to known H, no model needed |
| Regime-based routing decisions | **Modelless** | Bandit arm selection by regime label |
| Training probe-LLM for behavior study | **Model-based** | Requires forward pass + training |

**The modelless parts are what we distill.** The model-based parts (training a separate probe-LLM) are research methodology, not production code.

---

## GOAT Pillar Assessment

Per `27_mmo_goat_pillars_decision_matrix.md`:

| Criterion | Score | Evidence |
|-----------|-------|----------|
| GOAT passed | ⏳ | No implementation yet |
| MMO-product | ⬜ | Indirect — improves training data quality → better LoRA (Secret A) |
| LoRA-independent | ✅ | All probe generation and analysis is modelless |
| Defensible | ❌ | Paper is public, algorithm is standard Markov chain + Shannon entropy |
| Secret coverage | None | Pure diagnostics, no private knowledge encoded |

**Verdict: NOT a GOAT Pillar.** This is infrastructure that improves ALL our GOAT proofs (methodology upgrade) but doesn't encode private game knowledge. It's the **microscope**, not the **specimen**.

### Is It a "Super GOAT" (Keep Secret)?

**No.** The paper is public ICML 2026. The algorithms (Dirichlet sampling, typical-set classification, C1–C4 protocol) are all published. Our implementation would be open-source (katgpt-rs is MIT). No selling point in secrecy.

The value-add is **applying** the microscope to our game domains (riir-ai), not the microscope itself.

---

## Comparison to Existing Research

| Research | Overlap | Delta |
|----------|---------|-------|
| R061 (Entropy Anomaly) | Same entropy primitives | Data probes add known-distribution reference + regime labels |
| R037 (REAP Modelless) | Same modelless/model-based taxonomy | Data probes add formal IV/EV validation protocol |
| R076 (SR²AM) | Same configurator context | Data probes add probe-based calibration of entropy thresholds |
| R090 (Epiplexity) | Same information-theoretic lens | Epiplexity measures complexity; data probes measure typicality |
| R103 (State Distribution) | Same state-visitation tracking | Data probes add controlled generation of visitation patterns |

---

## Potential Problems We Can Study With Data Probes

1. **Calibrate our entropy thresholds** — currently hardcoded (e.g., PPoT threshold=0.5, `is_high_entropy_session` threshold). Data probes can determine optimal thresholds for each game domain.

2. **Validate SpectralQuant regime behavior** — Do eigenbasis-compressed KV caches push model outputs toward over-conservative or uncertain regimes?

3. **PPoT rescue effectiveness** — Generate probe sequences where rescue should help (uncertain regime) vs. hurt (typical regime), measure rescue rate.

4. **LoRA adapter quality calibration** — Train probe-LLM with different LoRA configs, measure regime distribution shift. This gives us a **controlled calibration** for LoRA quality that doesn't require real game data.

5. **Bandit arm selection validation** — Use data probes as controlled stress tests for our BanditPruner, FlowPruner, DeltaBanditPruner under known distributional shifts.

---

## What We Would NOT Implement

- Training separate probe-LLMs (research methodology, not production)
- Large-scale probe experiments with real LLMs (requires GPU training)
- PCFG probes (Section 5 discussion — future work even for the paper authors)
- Full mechanistic interpretability pipeline using probes (Section 5 discussion)

---

## Formal Claim Cards for Our Distillation

### Claim 1: Typical-Set Regime Labels Are Calibrated Diagnostics

```
h₁ = (knob=entropy_rate, values={0.5, 1.0, 2.0}, diagnostic=regime_label_accuracy)
Direction: Higher entropy → more uncertain regime predictions
Falsification: Regime label accuracy < 80% on held-out probe sequences
Status: NOT YET TESTED
```

### Claim 2: Data-Probe Thresholds Transfer to Game Domains

```
h₂ = (knob=game_domain, values={bomber, go, monopoly}, diagnostic=bandit_win_rate)
Direction: Probe-calibrated entropy thresholds improve bandit win rate vs. hardcoded
Falsification: Win rate delta < 0 (probe-calibrated worse than hardcoded)
Status: NOT YET TESTED — requires riir-ai integration
```

---

## Bottom Line

Data probes are a **methodology upgrade**, not a feature. They give us:
1. Formal C1–C4 criteria for all GOAT proofs (better than ad-hoc thresholds)
2. Typical-set regime classification (three-way upgrade from our binary high/low entropy)
3. Known-distribution NLL computation (ground truth for probe experiments)
4. A calibration tool for entropy thresholds across game domains

**Priority: MEDIUM.** Useful for proving GOAT quality but doesn't create new capability. Ship as a `data_probe` feature gate in katgpt-rs, use in riir-ai for game-specific calibration.
