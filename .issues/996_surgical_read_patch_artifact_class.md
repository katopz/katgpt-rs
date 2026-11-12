# Issue 996: Surgical read-patch artifact class + modelless extractions (from owl Sparse Rewiring)

**Status:** Open — dormant primitives, no wired consumer; each sub-item carries pre-registered falsification arms; pick up only when its trigger fires.
**Source:** riir-train/.research/492_Sparse_Rewiring_ReadOnly_Neuron_Patch.md (maxsloef owl-rewired-neurons, repo @ `a6dc314740acbf726dcac435b4e84eebbbde49d8`).
**Verdict context:** research Gain on both tracks; not GOAT (no consumer, no bench). This issue is the (b)-class terminal state for the modelless-extraction rows of Research 492's Path 0 inventory.

## Background

A Qwen3-8B persona was induced by training rank-4 additions to the gate/up READ rows of 89 L0-selected MLP neurons (write/down columns frozen); graded ≈ full SFT (93%/88% vs 94%/89%). The artifact merges as `W[idx] += b @ a` — a row-sparse, write-frozen special case of `LoraPair` (Plan 025). The stack's persona lane stays LATENT (style_weights / mb_value / CNA Plan 087) — nothing here changes that; these primitives serve weight-space artifacts only if a serving consumer ever lands.

## Sub-items

### (a) Modelless persona-neuron census scorer — activation separation, no GD

Score every MLP neuron by closed-form separation between a target corpus and a base corpus (e.g., mean activation difference, or a sigmoid-gated dot onto a per-layer direction; never softmax). The owl source selected neurons by trained L0 masks; the predecessor showed RANDOM neurons in layers 1–18 already work — so a cheap census score is plausibly sufficient for persona-style targets.

- [ ] Trigger: a lane needs neuron-level attribution on a dense model we serve (e.g., explaining what a specialist head/fine-tune moved).
- [ ] Falsification arms: census ranking vs the trained-L0 ranking on a public persona patch (owl artifacts are CC BY-NC — use for local comparison only, never ship); random-ranking control must NOT separate (the predecessor says random works — if random ≈ census, the scorer is noise and the issue closes negative); attribution-vs-causal gap check (detection ≠ prediction, per Research 267).
- [ ] Home if picked up: MLP-neuron sibling of `FaithfulnessProbe`/`CausalHeadImportance` (Research 362/Plan 358) — consume that substrate, never fork.

### (b) Surgical read-patch artifact class — row-sparse, write-frozen, deterministic merge

Container: per layer — neuron index list, shared read directions `a [k, d]`, per-neuron coefficients `b [n, k]`, for gate/up; merge = outer product into selected rows; write columns bit-untouched. Properties to prove: KB-scale artifact (89 neurons ≈ ≤1.25 MiB vs ~9–15 MiB dense rank-4 gate+up LoRA), deterministic bit-identical merge from the frozen bytes, BLAKE3-committed, monotonic hot-swap compatible (A2 law), write-path byte-identity (the defining property: neuron output columns unchanged).

- [ ] Trigger: a consumer needs a persona/style weight-space patch OR riir-train's adapter lane wants a second adapter shape. Until then this is a container spec, not code.
- [ ] **Fusion experiment (the one genuinely open question): closed-form construction vs trained values.** The source trained `a`/`b` by GD. Closed-form candidate: `a` = per-layer persona read direction from mean activation difference (persona corpus vs base); `b_i` = neuron i's projection onto it. If closed-form ≈ trained quality, the patch becomes deterministically CONSTRUCTIBLE at runtime from frozen stats — constraint-3-compliant without any training. Pre-registered bar: graded in-character within 5 points of the trained patch on a replicated eval, and persona-corpus val loss within +0.02 of trained.
- [ ] Falsification arms: merge determinism (same artifact twice → bit-identical weights); tamper (BLAKE3 mismatch refuses); write-path byte-identity (down columns hash-equal pre/post merge); ranking preservation on off-persona inputs (patch must not move the base's ranking on unrelated decisions); G4 alloc census on the merge path; ternary caveat — NEVER merge fp deltas into ternary rows (Bonsai purity); runtime side-patch post-load is the compatible shape.
- [ ] Home if picked up: the adapter-format lane (riir-train `lora.bin` checksum/order/`resolve_lora_pair` substrate) for storage; katgpt-core for the generic merge math. Consume `LoraPair` (Plan 025) vocabulary — the merge is just `W[idx] += b @ a`, so it MUST extend LoraPair's existing apply path rather than add a new kernel (substrate-first pre-emption of a duplicate).

### (c) Fact-swap verification recipe — "what does this frozen artifact actually carry"

Mutate the corpus's rule/fact-bearing content (train-data-level rewrites), retrain or re-derive nothing, and grade which facts the artifact follows (the owl grid: name→Bramble 97%, absurd values taken, strong-prior facts resist). Plus the template-position ablation (keep the patch only where the trigger fires; measure what survives).

- [ ] Trigger: next specialist-head/instinct artifact verification cycle that needs to prove decision content vs corpus style. Maps onto existing fixture discipline (riir-refine fixtures are standalone-compiled; instinct artifacts are BLAKE3+size-verified) — this adds the MUTATION axis (same artifact, mutated corpus, does the artifact follow the corpus or the rule?).
- [ ] Falsification arm: the control is the artifact re-verified against the UNmutated corpus (must stay identical); the fact-swap deltas are the measurement.

## Rules for any future pickup

- Consume existing substrate (Plan 025 `LoraPair`, riir-train adapter format, FaithfulnessProbe) — never fork (substrate-first).
- Any GOAT filing MUST run the falsification arms above FIRST and record numbers before claiming gain (Ungoat discipline).
- NC license: the owl weights/datasets are CC BY-NC 4.0 — usable for local comparison/reproduction; never ride a shipped artifact.
