# Proposal 010 — Non-Hidden-State Canonical Construction (Source-Feature Directions)

Status: **draft** — ⚠ 2026-09-25: every MiniCPM5-1B hidden state behind this proposal's evidence (riir-train Benches 422–427, 605) came from a RoPE-broken forward (llama.cpp's interleaved Q/K under a rotate-half RoPE; fixed riir-infer `0b26b9a`, ppl 259.9 → 69.05 == HF fp32). The cross-arch negatives are UNVERIFIED, not refuted — re-run plan: riir-train Issue 571.
Branch: `develop` (per global rule — no feature branches)
Owner: unassigned
Fusion of: Proposal 009 (canonical intent space, PERMANENTLY DEMOTED cross-arch) + Research 459 (hidden-state path exhausted) + code2vec (Alon et al. 2018, AST path embeddings)
Related: [Research 459](../.research/459_canonical_intent_space_plug_and_play.md) (CLOSED), [Bench 427](../../riir-train/.benchmarks/427_canon_p4_recipe_d_length_matched.md) (Recipe D demotion), [Bench 562](../.benchmarks/562_katgpt_canon_goat.md) (substrate GOAT stamp)

## TL;DR

The cross-arch canonical intent claim (Proposal 009's headline) was permanently demoted after four hidden-state methods failed (Research 459: P3 centroid disagreement, P3b layer-0 vocabulary signal, P3c length-detrending reversal, Recipe D structural cross-arch disagreement). Research 459's status header explicitly states the claim "reopens only on a **non-hidden-state construction** (AST/clippy/ownership-graph features), NOT on any further hidden-state method."

This proposal sketches that non-hidden-state path: extract deterministic, architecture-independent features from Rust source code to construct canonical intent directions. Instead of fitting a Procrustes rotation to align two models' hidden states (which fails because hidden states are arch-specific), define canonical directions from source-code-level features that are identical regardless of which model processes them.

**This is a katgpt-rs invention.** No prior work uses source-code features to construct canonical steering directions for cross-architecture model plug-and-play. code2vec (Alon et al. 2018) embeds AST paths for code understanding (method naming, bug detection), NOT for steering model behavior. The novelty is the bridge: source features → fixed canonical space → per-model projection.

**Honest verdict: HIGHLY SPECULATIVE.** The hidden-state approach had a plausible mechanism (both models encode the same concept, just in different bases — Procrustes aligns the bases). The source-feature approach has no such guarantee — it bets that code structure correlates with model behavior directions strongly enough to define useful steering vectors. This bet may not pay off. The proposal exists because it's the ONLY remaining path, not because it's likely to succeed.

## The problem this solves

Proposal 009 shipped a canonical intent space substrate (`CanonicalIntent` + `ProcrustesAdapter` + `SubspaceAdapter` + `MaskAdapter` in `crates/katgpt-canon/`). The intra-arch path works (Bench 562 G1/G2/G4 PASS). The cross-arch path failed permanently (Bench 427):

- Canonical directions were constructed from **hidden states** (model activations on probe prompts)
- Different architectures encode the same concept in structurally different ways (not just different bases — different *locations* in activation space, per P3 centroid disagreement at −0.33)
- No hidden-state method (Procrustes, length detrending, joint SVD subspace, contrastive Recipe D) recovered cross-arch agreement above +0.01 (threshold ≥ 0.5)

The problem: **canonical directions that depend on hidden states are architecture-bound by construction.** If the direction "Rust idiom" is defined as "the activation difference between idiomatic and non-idiomatic code," then that direction lives in Gemma's activation space, not in a universal space. MiniCPM's "Rust idiom" direction points somewhere else entirely.

The fix: define canonical directions from features that are **external to all models** — the source code itself. A "Rust idiom" direction defined from AST structure + clippy lints is the same vector regardless of which model will be steered by it.

## The proposed design

### Three feature families (all deterministic, all architecture-independent)

```rust
/// A canonical direction constructed from source-code features, not hidden states.
///
/// The direction lives in a fixed-dimensional "source feature space" that is
/// identical across all models. A per-model `SourceFeatureAdapter` projects
/// from this space into the model's latent space for steering.
pub struct SourceFeatureDirection {
    /// BLAKE3 tag (same pattern as CanonicalIntent).
    tag: [u8; 32],
    /// The fixed-dim source-feature vector (normalized to unit length).
    direction: Vec<f32>,
    /// Which feature family this direction was constructed from.
    family: FeatureFamily,
}

pub enum FeatureFamily {
    /// AST node-type histogram (fixed vocab, deterministic).
    AstHistogram,
    /// Clippy lint-category fingerprint (style signature).
    ClippyFingerprint,
    /// Ownership/borrow graph topology (lifetime depth, borrow density).
    OwnershipGraph,
}
```

### Feature 1: AST node-type histogram

Parse Rust source with `syn`, count AST node types into a fixed vocabulary:

```text
ast_histogram(code) → [f32; N_AST]
  where N_AST = |{fn, trait, impl, struct, enum, match, if, loop, closure,
                   generic_param, lifetime, where_clause, async, const, ...}|
```

- **Deterministic**: same source → same histogram, bit-identical.
- **Architecture-independent**: the AST doesn't know which model will process the code.
- **Fixed dimension**: the node-type vocabulary is closed (Rust's syntax is finite).
- **Example direction**: "trait-heavy code" = histogram weighted toward `trait` + `impl` + `generic_param` nodes.

### Feature 2: Clippy lint-category fingerprint

Run clippy on the codebase, aggregate lint counts by category:

```text
clippy_fingerprint(crate) → [f32; N_CLIPPY]
  where N_CLIPPY = |{complexity, style, perf, correctness, pedantic, nursery, ...}|
```

- **Deterministic**: same code + same clippy version → same lint counts.
- **Architecture-independent**: clippy analyzes code structure, not model behavior.
- **Style signature**: a crate that triggers many `pedantic` lints has a different "style fingerprint" than one that's clippy-clean.
- **Example direction**: "defensive code" = histogram weighted toward `correctness` + `pedantic` categories.

### Feature 3: Ownership/borrow graph topology

Analyze the borrow checker's ownership graph (via `rustc`'s MIR or a custom visitor):

```text
ownership_graph(code) → [f32; N_OWN]
  where N_OWN = |{avg_lifetime_depth, borrow_density, move_count,
                  shared_borrow_count, mutable_borrow_count, ...}|
```

- **Deterministic**: same source → same ownership graph.
- **Architecture-independent**: ownership is a Rust language property, not a model property.
- **Example direction**: "zero-copy code" = graph with low clone count + high borrow density.

### The adapter: source-feature space → model latent space

The `ModelAdapter` trait (from Proposal 009) projects from canonical space to model latent space. For source features, the adapter is a **learned projection** (not Procrustes — source features and latent spaces have different dimensionalities + no orthogonality guarantee):

```rust
pub struct SourceFeatureAdapter {
    /// Linear projection: [N_FEATURES] → [D_MODEL].
    /// Fit via ridge regression on (source_features, model_activations) pairs.
    projection: Vec<f32>,  // row-major [D_MODEL × N_FEATURES]
    target_dim: usize,
    commitment: [u8; 32],
}
```

The projection is fit once (setup-time, like ProcrustesAdapter) by:
1. Collecting paired samples: `{(source_features(code_i), model_activations(code_i))}` for a probe corpus.
2. Ridge regression: `projection = argmin ||X·P - Y||² + λ||P||²` where X is the source-feature matrix, Y is the activation matrix.
3. The projection is **per-model** (each model gets its own adapter), but the **canonical directions are shared** (same source features for all models).

### Why this might work where hidden states failed

The hidden-state failure (Research 459) was structural: different architectures encode concepts at *different locations* in activation space, not just in different bases. Procrustes aligns bases, not locations.

Source features sidestep this entirely: the canonical direction is defined in a space that NO model inhabits. Each model learns its own projection FROM that neutral space INTO its latent space. There's no cross-arch alignment to fail — each adapter is fit independently.

The bet: source features capture enough semantic signal to define useful steering directions. If "trait-heavy code" (AST histogram) correlates with a coherent activation direction in Gemma AND in MiniCPM (independently fit), then the canonical direction is arch-neutral even though the projections are per-arch.

## Honest caveats — READ BEFORE IMPLEMENTING

1. **The source features may be too coarse.** AST histograms count *what* nodes appear, not *what they mean*. Two crates with the same node-type counts but completely different semantics would get the same direction. Hidden states (imperfectly) capture meaning; AST histograms capture structure. The correlation between structure and meaning may be too weak to define useful steering directions. This is the #1 risk.

2. **No orthogonality guarantee.** Procrustes preserves information because orthogonal rotations are bijective. Source-feature projections (ridge regression) are lossy — the source-feature space may not span the full latent space. Steering directions defined in source-feature space may have weaker effects than hidden-state steering vectors.

3. **The probe corpus problem.** Fitting the projection requires paired (source_features, model_activations) samples. The choice of probe corpus determines which source features map to which activation directions. A biased corpus produces biased projections. This is the same problem that killed Recipe D (length-matched contrastive), just at a different level.

4. **`syn` is a heavy dependency.** Adding `syn` to `katgpt-canon` pulls in a full Rust parser. This is acceptable for a setup-time feature (the projection is fit once, not per-token), but it violates the "prefer existing dependencies" rule. Alternative: use a lighter-weight parser (e.g., `ra_ap_syntax` from rust-analyzer, or a hand-rolled tokenizer for the most common node types).

5. **Clippy version sensitivity.** Clippy lint categories change between Rust versions. A fingerprint computed on Rust 1.85 may not match one on 1.95. The commitment must include the clippy version, and cross-version canonical directions may not be comparable. This limits the "architecture-independent" claim — it becomes "architecture-independent modulo toolchain version."

6. **This is the ONLY remaining path, not a LIKELY path.** Research 459 exhausted four hidden-state methods. This proposal exists because the alternative is "accept permanent demotion and move on." The expected outcome is that source features also fail to produce useful cross-arch steering — but the negative result is valuable (it closes the last theoretical escape hatch).

## Fusion lineage

1. **Proposal 009 (Canonical Intent Space)** — the substrate this builds on. `CanonicalIntent` + `ModelAdapter` trait + the three adapters. Source-feature directions are a 4th `CanonicalIntent` variant (same trait, different construction).
2. **Research 459 (hidden-state exhaustion)** — the negative result that motivates this proposal. The four failure modes (centroid, layer-0, length-detrending, Recipe D) define what the source-feature approach must NOT replicate.
3. **code2vec (Alon et al. 2018, arxiv 1803.09473)** — AST path embeddings for code representation. This proposal distills the *feature-extraction* idea (AST → fixed-length vector) but NOT the *learned-embedding* idea (code2vec trains a neural model; this proposal uses deterministic histograms). The deterministic choice is deliberate — modelless-first mandate.

## GOAT gate

This proposal does NOT request default-on promotion. It requests **research validation** behind an opt-in feature flag. The gates:

- **G1 (correctness):** source-feature extraction is deterministic — same input → bit-identical output. BLAKE3 commitment test (same pattern as Proposal 009).
- **G2 (perf):** extraction is setup-time (not per-token). AST parse + histogram on a 10K-line crate must complete in < 1s. The projection fit (ridge regression) must complete in < 10s.
- **G3 (no-regression):** opt-in feature flag; default features unaffected.
- **G4 (alloc-free):** the projection APPLY (not the fit) must be zero-alloc, same as ProcrustesAdapter.
- **G5 (cross-arch agreement — THE GATE):** the decisive test. Fit independent projections for Gemma-2-2B and MiniCPM5-1B on the same probe corpus. Measure whether steering by the same source-feature direction produces correlated behavior changes in both models. **Threshold: agreement > +0.3** (hidden-state Recipe D peaked at +0.009; the bar is low but non-trivial). If G5 fails, this proposal is PERMANENTLY DEMOTED alongside Proposal 009 and the cross-arch claim is closed forever.

## What ships now (katgpt-rs) vs deferred (riir-train)

### Ships now — katgpt-rs (if validated)
- `SourceFeatureDirection` + `FeatureFamily` types in `crates/katgpt-canon/`
- AST histogram extractor (behind `canon_source_features` feature, gates `syn`)
- Clippy fingerprint extractor (behind same feature)
- Ownership graph extractor (behind same feature, may require `rustc` internals — heaviest)
- `SourceFeatureAdapter` (ridge regression projection, setup-time fit + zero-alloc apply)
- G1/G2/G4 gates on the extraction + projection
- **G5 is the make-or-break** — runs in riir-train (needs both models loaded)

### Deferred — riir-train
- G5 cross-arch agreement measurement (requires loading Gemma + MiniCPM5 simultaneously)
- Probe corpus curation (the paired source-feature/activation samples)
- Ridge regression hyperparameter tuning (λ, feature weighting)

### Explicitly NOT shipped by this proposal
- **Default-on promotion** — this is research validation, not production. Even if G5 passes, promotion requires a separate proposal.
- **`syn` as a default dependency** — stays behind the `canon_source_features` feature flag. The canon crate's default build must remain lightweight.
- **Trained source-feature embeddings (code2vec-style)** — violates the modelless-first mandate. If deterministic histograms fail G5, the conclusion is "source features don't work," NOT "try learned embeddings."

## Phased rollout (sketch — a plan would expand this)

### Phase 1 — AST histogram extractor (lightest feature family)
- [ ] T1.1 Add `syn` dependency behind `canon_source_features` feature
- [ ] T1.2 Implement `ast_histogram(code: &str) -> Vec<f32>` with fixed node-type vocabulary
- [ ] T1.3 BLAKE3 commitment (deterministic verification)
- [ ] T1.4 G1 correctness tests (same code → same histogram)
- [ ] T1.5 G2 perf (10K-line crate < 1s)

### Phase 2 — SourceFeatureAdapter (the projection)
- [ ] T2.1 Implement ridge regression fit: `fit_source_adapter(X, Y, λ) -> SourceFeatureAdapter`
- [ ] T2.2 Zero-alloc apply path (`project_into`)
- [ ] T2.3 G4 alloc-free verification
- [ ] T2.4 Round-trip test: project → extract → verify information preservation

### Phase 3 — G5 cross-arch agreement (the decisive gate, runs in riir-train)
- [ ] T3.1 Curate probe corpus (Rust code samples with known style properties)
- [ ] T3.2 Collect paired (source_features, activations) for Gemma-2-2B
- [ ] T3.3 Collect paired (source_features, activations) for MiniCPM5-1B
- [ ] T3.4 Fit independent projections for both models
- [ ] T3.5 Measure cross-arch agreement on held-out steering directions
- [ ] T3.6 **HONEST NEGATIVE RESULT if G5 fails** — document, demote, close

### Phase 4 — Clippy fingerprint + ownership graph (only if Phase 3 is promising)
- [ ] T4.1 Clippy lint-category extractor
- [ ] T4.2 Ownership graph topology extractor
- [ ] T4.3 Re-run G5 with combined feature families

## Risks

1. **Coarse features risk (highest).** AST histograms may not capture enough semantic signal. Mitigation: Phase 1 is cheap (hours, not days); if Phase 3 G5 shows near-zero agreement, stop early.
2. **`syn` dependency weight.** Adding a full Rust parser to the canon crate's feature surface. Mitigation: keep behind opt-in feature; the default build doesn't pull it.
3. **Probe corpus bias.** The projection fit depends on the corpus. Mitigation: use the existing riir-train Rust corpus (the same one Recipe D used) for consistency.
4. **Ridge regression underfitting.** Source features (N~50) → latent space (D~2304) is a wide projection. Mitigation: the projection is per-model; it only needs to capture the *steering-relevant* subspace, not the full latent space.
5. **False positive on G5.** Source features might produce apparent cross-arch agreement that's actually just "both models prefer well-structured code" — a trivial signal, not a useful steering direction. Mitigation: G5 must test on *contrasted* directions (idiomatic vs non-idiomatic), not just absolute structure.

## Out of scope

- **Trained source-feature embeddings** (code2vec, CodeBERT). Violates modelless-first mandate. If deterministic features fail, the conclusion is "source features don't work," not "try learned features."
- **Cross-language canonical directions.** This proposal is Rust-only (the probe corpus + the AST vocabulary are Rust-specific). Cross-language (Rust ↔ Python) would require a shared AST vocabulary — a separate proposal.
- **Default-on promotion.** This is research validation. Promotion (if G5 passes) requires a separate proposal re-arguing the value proposition.
- **Production steering runtime.** The adapter fit + apply is the substrate; the runtime steering integration (riir-ai NPC cognition) is a separate plan.

## References

1. **code2vec: Learning Distributed Representations of Code** — Alon, Zilberstein, Levy, Yahav (ICLR 2019, 1903 citations). [arXiv:1803.09473](https://arxiv.org/abs/1803.09473). Distilled: AST path contexts → fixed-length code embedding via attention. We borrow the *AST → fixed vector* idea but NOT the learned embedding (deterministic histograms instead, per modelless-first mandate). Cited-only (not distilled into a research note — the transferable idea is one paragraph).
2. **cAST: Enhancing Code Retrieval-Augmented Generation with AST** — [arXiv:2506.15655](https://arxiv.org/abs/2506.15655). AST-based chunking for cross-language code retrieval. Relevant for the "architecture-independent" claim — AST structure generalizes across languages. Cited-only.
3. **Cross-Architecture Instruction Embedding** — Redmond et al. (NDSS BAR 2019, 90 citations). Cross-architecture binary code analysis via shared instruction embeddings. The closest prior art on "cross-architecture representation" — but for binary instructions, not source ASTs. Cited-only.
4. **Analyzing the Generalization and Reliability of Steering Vectors** — Tan, Chanin et al. Shows steering vectors have "substantial generalization" issues. Relevant caveat: even if source features produce a steering direction, its reliability across inputs is not guaranteed. Cited-only.
5. **Research 459** — in-house. The hidden-state exhaustion verdict that defines the bar this proposal must clear.
6. **Bench 427** — in-house. Recipe D length-matched contrastive: cross-arch agreement peaked at +0.009 (threshold ≥ 0.5). The source-feature approach must beat this to justify reopening the claim.

## TL;DR

**Verdict: draft for research validation, NOT production.** This proposal sketches the only remaining path to reopen the cross-arch canonical intent claim (per Research 459's explicit condition). The approach is novel (source-code features → canonical directions, no prior art on this specific bridge) but HIGHLY SPECULATIVE (source features may be too coarse; G5 is the make-or-break gate). Next action: if the user approves, open Plan NNN for Phase 1 (AST histogram extractor — the cheapest validation step). If G5 fails, the cross-arch claim closes permanently with this as the final negative result.
