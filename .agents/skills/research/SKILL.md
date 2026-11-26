---
name: research
description: Research workflow for distilling ML/AI papers into modelless inference primitives, freeze/thaw runtime patterns, latent-space operations, AND model-based training plans across the multi-repo stack. Use when reading arxiv papers, deciding which repo a paper belongs in, creating .research/ notes or .plans/ files, implementing modelless inference primitives, or routing training-vs-inference insights. Enforces the commercial strategy (public engine / private runtime / private chain / private neuron-db / private training / private SDK facade / private product-domain / public inference-substrate riir-infer / public decision-serving riir-reflex), three-track system (modelless inference + self-adaptive runtime + model-based trained weights — all three exist across the stack, not just riir-train), latent-to-latent preference, and freeze/thaw-over-fine-tuning rule. Every task ends with a mandatory Claude verdict ping-pong (§5) before any file is committed.
---

# Research Workflow — Modelless Inference, Freeze/Thaw, Latent-to-Latent

This repo (`katgpt-rs`) + `riir-ai` (freeze/thaw runtime + adaptive NPCs + game systems) + `riir-chain` (neuro-symbolic chain transport, LatCal) + `riir-neuron-db` (NeuronShard, BLAKE3/Merkle, freeze/thaw envelope) ship **runtime + latent-space operations**. The **inference substrate** (weights, quant formats, kernels, encoders, loaders) lives in `riir-infer`; **decision-engine serving + comparison** (calibration, abstention, arenas) lives in `riir-reflex` — both opened public 2026-09-23 as the first sanctioned Research-003 exceptions. Training-method research lives in `riir-train`. If a paper's value is its training loop → `riir-train` (see §3.5 Path 0.5 — applicable training papers get a Plan, not a lazy redirect). If its value is a latent-space insight, a routing trick, a freeze/thaw pattern, a chain-commitment bridge, a neuron-shard primitive, or a modelless inference primitive → distill here. Weight representation → `riir-infer`, modelless serving → `riir-reflex`, trained-encoder serving → `riir-rethink` (private forever) — full split in the routing rule of thumb below.

## When to use

Reading/fetching/summarizing ML/AI/systems papers · deciding which repo a paper belongs to · creating `.research/` notes or `.plans/` files · implementing modelless inference primitives · designing freeze/thaw cycles, adapter hot-swap, runtime adapter routing · designing latent-to-latent ops (dot-product projections, sigmoid gating, manifold geometry, spectral methods) · designing MMORPG-scale game AI (thousands of concurrent NPCs, 20Hz tick, fog-of-war, emergent social/economic behavior).

Do NOT activate for: pure refactor, bug fixes with no research angle, or ordinary feature work.

## Repos

**Canonical count + membership: `katgpt-rs/AGENTS.md` §"Repo count"** — the
product/distillation set is **7** (first block below). The workspace around
it is every sibling with a root `BOUNDARY.md` + `.git`. Derive it; never type
a count here (this file's typed counts have gone stale three times):

```bash
cd "$(git -C katgpt-rs rev-parse --show-toplevel 2>/dev/null || git rev-parse --show-toplevel)/.." && for d in */; do  # run from katgpt-rs or its parent
  case "$d" in *.wprobe/) continue;; esac   # probe artifacts carry BOUNDARY.md + .git but are not repos (a naive derive over-counts)
  [ -f "$d/BOUNDARY.md" ] && [ -d "$d/.git" ] && echo "${d%/}"
done
```

The product/distillation set (7):

- `katgpt-rs/` — public MIT engine. Generic modelless inference primitives. **No game/chain/shard IP.**
- `riir-ai/` — private game product. Freeze/thaw runtime, self-learn, game systems. Hosts the `.docs/` moat book.
- `riir-chain/` — private neuro-symbolic chain transport. LatCal, `riir-chaind`, economics, asset lifecycle, `catchup/` (Turso/libSQL, quorum). **Re-exports `riir-neuron-db` under `neuron_db` feature; canonical shard source is `riir-neuron-db/`.**
- `riir-neuron-db/` — private leaf. `NeuronShard` (Pod, zero-copy mmap), `ShardIndex` (lock-free papaya), generic `MerkleTree`/`MerkleProof`, `MerkleFrozenEnvelope`, MAPE-K, Raven/δ-Mem consolidation, AnyRAG gateway, vibe KG triples, spectral init, `ShardCompactor`, dendritic LoRA branch. **No chain dep — usable standalone.**
- `riir-train/` — private training vault. **As of 2026-08-06: actively pursued, not lazily redirected.** Applicable training papers get a Plan in `riir-train/.plans/` per §3.5 Path 0.5. `read_file riir-train/.docs/02_pipelines/training_data_pipeline.md` before any training-paper verdict.
- `riir-game-sdk/` — private facade + dev-tool workspace; downstream consumer, rarely a distillation target.
- `riir-dapps/` — private dApp layer (added 2026-08-20). Game outcome → generic chain settlement (`Settlement`, `MultiClaimEscrow` composition) + the KAT ledger/service SERVER plane (kat_ledger, epoch settle, payment rails, CF worker/DO shell). Route settlement-composition + service-economy papers here, NOT riir-chain.

**The wider workspace (route here only when the insight is
inseparable from that surface; the first three entries are first-class PUBLIC
distillation targets, not last resorts):

- `riir-refine/` — private multi-domain rulewright. **Fusion priority #2** (see the ladder below) — the workspace's densest latent-state consumer per LOC; rulewright-domain distills file their `.research/` notes HERE (renamed riir-refine -> riir-refine 2026-10-01) (MOAT row below).
- `riir-infer/` — **PUBLIC** LLM-inference substrate (carved from riir-ai 2026-09-22; opened public 2026-09-23). Workspace: `riir-infer-core` at the repo root (CPU substrate: weights/quant/architectures/loaders — `quant/` EXL3-trellis lane, `deltanet/` GDN lane, Bonsai2 rotation, `gguf_loader.rs`/`safetensors_loader.rs`/`tokenizer.rs`, `simd/`, the `src/bin/` measurement/ablation lanes) + `crates/riir-infer-gpu` (wgpu/CubeCL) + `crates/riir-infer-laya` (the pinned-checkpoint encoder lane: CPU + Metal + CUDA — reflex's comparison-lane substrate). **Zero riir-* deps — upstream of the engine.** Quant/weight-format/kernel/encoder/loader papers distill HERE (its `.research/` created on first use).
- `riir-reflex/` — **PUBLIC** decision-engine serving + comparison + contribution (opened public 2026-09-23). Serves `decision_wire` (`choice`/`score`/`noul`, abstention first-class) modellessly: hashed-feature embedding → `pick_domain` corpus routing → `Lz4FlexDrafter` corpus-is-the-model scoring → sigmoid normalization → `SigmoidGateCalibrator` → fused abstain. GOAT gates G1–G5 (G5 laya-parity gates every published laya number); honest-metrics harness (9 dataset suites, corpus-cap + cal-select levers) + comparison lanes (laya CPU/Metal/CUDA, CLM, GLiNER, AgentJev). Calibrated-decision-serving / abstention / selective-prediction / comparison-arena papers route here; the laya lane SUBSTRATE lives in riir-infer.
- `riir-instinct/` — **PUBLIC** open lane (carved from the moat 2026-10-03): the L1→L3 decision-serving stack crate — suites, the lane-backend seam (`riir_instinct::server::{LaneBackend, ExtBoots, install_ext_boots}` + the `Ext` seat), arena game heads, the **one-definition home of the escalation grammar** (`riir_instinct::arsenal::EscalateSpec`). Open-stack/lane-seating/escalation-grammar/arena-head papers route here; enforcement + product posture live rethink-side.
- `riir-rethink/` — **PRIVATE FOREVER** (the carve's moat): the trained-encoder decision product — L3 encoder serving rung (F16/Q8 adopted/Q4 FAIL), **ESC enforcement** (the manifest `escalate` table is the only ARMING surface; `RIIR_RETHINK_ESC=0` demote-only; rolling-rate guard), the HOSTED-ONLY vessel reader (A10 moat law), the decstat economy-capture plane, the ESC-depth research line. ESC/escalation/depth/population-scaling/serving-economy papers route here. `git@github.com:gist-rs/*` private; nothing leaks through a public-repo distill.
- `riir-auth/` — private. Latent-space session recording (`SessionFingerprint` → `NeuronShard`) + 3-pillar ban quorum + the `account_key` Ed25519 identity substrate. Behavioral-identity/fingerprinting/escalation papers route here.
- `riir-dao/` — private KAT tokenomics agent (signals → strategy → guards → advisory → commit; modelless Elo + Beta-LCB selection). Tokenomics/governance/mechanism-design papers route here.
- `riir-kat/` — private KAT network CLIENT + wire-protocol plane (spun out of riir-refine 2026-09-10, issue 088). Service-protocol papers only; rarely a target.
- `riir-deployer/` — private manifest-driven deploy orchestrator (genesis ceremony, ops ladder, freeze-lineage probe). Ops/infra papers; rarely a target.
- `riir-esp32/` — private POC/fun, **not prod**: the ESP32 Satellite device tier (esp-hal firmware, on-device crypto timing). Device-tier papers only.
- `riir-viewbridge/` — private Unity FFI seam, **PARKED 2026-09-03** (Bevy is the shipping render path). FFI raw/latent-wall papers only; expect no consumer until unfreeze.
- `katgpt-web/` — PUBLIC explainer site ("The Anatomy of KatGPT-RS"). Presentation surface only — **nothing private may ever appear here**; never a distillation target.
- `riir-shader/` — private shader surface (workspace member per `scripts/repo_set.txt`; no established research routing yet).
- Also on disk, never routed by name: `riir-mmorpg-examples`, `riir-llm`, `seal-game-editor`, `seal-remake`, `seal-render`, `seal-std`, `riir-refab`, `zed`, `riir-reflexer`, `reflex-site` — consumers/leaf/presentation surfaces, targets of last resort (`riir-mmorpg-examples` is the reference game consumer; `reflex-site` the public bench-publish front).

**Routing rule of thumb:** if it's about *how a shard is structured/committed/frozen/consolidated/retrieved/projected* → `riir-neuron-db`. If it's about *how a shard crosses quorum or bridges to LatCal fixed-point* → `riir-chain`. If it's about *code retrieval / fix trajectories / rule corpora / fixer selection* → `riir-refine`. If it's about *settlement composition or the service economy* → `riir-dapps` (economy governance/tuning → `riir-dao`; the client/wire protocol → `riir-kat`). If it's about *how trained weights are represented and run — quant formats, kernels, encoder forwards, loaders* → `riir-infer`. If it's about *how a MODELLESS decision is served — calibration, abstention, comparison arenas* → `riir-reflex`; a lane that SERVES product decisions from a trained encoder (encoder lane product, per-suite winners/vessels, serving-weights posture) is `riir-rethink`'s charter even when the proposal says "Reflex" — reflex gets at most an owner-gated public comparison lane. If it's about *the escalating decision ladder, suite/arena seating, or the open serving stack's grammar* → `riir-instinct` (open stack + `EscalateSpec` one-definition) / `riir-rethink` (private enforcement + depth economics; split by the carve law: grammar and stack in instinct, product posture and enforcement in rethink, lockstep edits in ONE family).

⛔ **Family ≠ repo (the 2026-10-07 encoder-lane instance, 6th of the challenge-rescue class; the record lives in the private tier):** the ladder reflex→instinct→rethink is a product family spanning THREE repos, and input documents routinely name "the Reflex family/stack" as host. Route by RUNG, never by the document's own name for its host: modelless floor + public arenas = reflex; open grammar/seam/arena heads = instinct; trained-encoder serving + enforcement + economy = rethink. The decisive question for any encoder/decision-lane proposal: *is this a public comparison lane or a private serving arm?*

## Commercial strategy (inline short version)

| Tier | Where | Role |
|---|---|---|
| **0 — Substrate** | `katgpt-core` (leaf, crates.io) | Pure inference mechanics (SIMD, transformer/weights, `hla`, `dd_tree`, `mcts`, `sampling`, `delta_mem`). Leaf-clean. |
| **1 — Engine** | `katgpt-rs` (public) | Re-exports substrate + BASIC cognitive primitives + toy games (each ships WITH its `.md`). |
| **2 — GOAT + IP** | `riir-ai` / `riir-chain` / `riir-neuron-db` / `riir-train` (private) | GOAT/Super-GOAT tuned versions, `*_runtime` composition layers, IP. |

**What = public. How = private.** Training how → `riir-train`. Runtime how → `riir-ai`. Chain how → `riir-chain`. Shard how → `riir-neuron-db`. When unsure → default private (safe to keep private; never safe to un-leak public). Toy 2D games (`bomber`, `monopoly`, `go`, `fft`) are NOT product IP — public. `*_runtime` suffix = private GOAT composition layer; bare-name = public primitive. FV moat: ~79 Lean 4 theorems across 4 `.proofs/` instances.

**Public exceptions (2026-09-23, owner-sanctioned Research-003 amendments): `riir-infer`** (the model-based inference substrate — weights/quant/kernels/loaders; the substrate tier of track c) **and `riir-reflex`** (decision-engine serving + comparison arenas). They are routing targets with their own MOAT rows (§1.6), not leaks — runtime/chain/shard/train IP stays private.

## Pre-flight — MANDATORY before any verdict/file

Do all seven before creating any file (the count drifts — read them, don't trust a typed number):

1. **`read_file` the product-set READMEs (the 7) + `riir-ai/.docs/README.md` + the AGENTS.md of `riir-refine` (rulewright — fusion priority #2), `riir-infer`, `riir-reflex`, `riir-instinct`, `riir-rethink` (the ladder split — these two separate a reflex public comparison lane from a rethink private serving arm) + (training papers) `riir-train/.docs/02_pipelines/training_data_pipeline.md`** — defines scope boundaries. Skipping = the #1 cause of false Super-GOAT claims + false PASS on training papers.
2. **`list_directory` every `.research/` folder in the workspace** — derive it (`ls -d /Users/katopz/git/*/.research`); never trust a typed count (15 today, derived — enumerations rot); create missing ones on first use.
3. **`list_directory` the 6 runtime/chain/db/substrate/serving src trees** — module names are vocab. Skipping = #2 cause of false Super-GOATs.
4. **`web_search` for published prior art** on the paper's headline technique (see §4). Skipping = #4 false-novelty failure mode. **Internal-first rule (GRAM/058 lesson, 2026-09-25): before any web search, grep every workspace `.research/` folder for the paper's arXiv ID AND its title verbatim** — a prior distillation is the controlling constraint, and a content grep that stops at page 1 of matches does not find it (Research 058 was missed exactly this way at note-590 round 1; caught only by the verdict reviewer).
5. **`grep` ALL repos for existing training/model-based/self-adaptive code** — the system is three-track, not modelless-only. Skipping = #5 false-PASS mode (canonical failure: arXiv:2511.18538 Code LLM survey was falsely PASSed because the agent only checked riir-train for training code, missing quest_grammar's LoRA training in riir-ai + TernaryDraftModel in riir-refine + self_evolve). Grep patterns: `*_training.rs|*_train*.rs|LoRA|SFT|GRPO|DPO|ternary|TernaryDraftModel|self_evolve|with_weights|\.bits` across `riir-ai/crates/**/src/`, `riir-refine/src/`, `riir-neuron-db/src/`. Training pipelines ship OUTSIDE riir-train — the three tracks are: (a) modelless inference, (b) self-adaptive runtime latent updates, (c) model-based trained weights.
6. **LIVE-STATE CHECK (2026-10-31, the Research-485→975 arc): before citing any issue/plan/consumer-list/wait-gate as open, blocked, or consumable, verify its LIVE state THIS session — the file on disk, `.highwater`, `git log --follow <file>` for close commits, `list_threads` for sibling lanes. A sibling session implemented + closed katgpt-rs Issue 975 between the Research-485 filing and its follow-up; the follow-up then cited the dead T5 consumer list (inventing rows (e)/(f)/(g)) and wrote three wait-gates on a passed gate — one full REVISE round + reword of three files. Corollary: never write "blocked on X" without a fresh read of X. **Citation sub-rule (2026-11-26, Plato's Key arc): numbered artifacts in OTHER repos (Bench N / Issue N / Plan N) are cited repo-QUALIFIED and existence-verified (one `ls` of the owning repo's folder) before entering a note — an unqualified "Bench 424–427" in a third repo's docs rebinds to the WRONG documents in the citing repo (katgpt-rs `.docs/05_adaptation` cites riir-train's canon benches unqualified; katgpt-rs's OWN 424–427 are unrelated benches — caught by verdict round 1).
7. **MODEL/ARTIFACT IDENTITY + DISK TRUTH (2026-11-26, Plato's Key arc): when a verdict pairs on-disk checkpoints or cites file inventories — (a) read a model's identity from its METADATA, never its filename: GGUF `general.architecture` via the owning repo's header probe (`riir-train scripts/plan402_gguf_probe.py`, ~0.2 s) — the session paired a 5-block `dflash` DRAFT HEAD as a factual-recall recipient because its name resembled the target LM's, and labeled a `qwen35` GDN hybrid "dense llama-family", both caught by verdict round 2; (b) file-on-disk claims are verified by `ls` of the data dir, never quoted from an AGENTS.md inventory row — riir-train's MiniCPM LOCAL-KEEP row was stale vs disk (only the tokenizer present).**

**`list_directory` the src trees** (vocab; skipping causes the vocabulary-miss class of false verdicts — mechanisms ship under non-obvious names):
- `riir-ai/crates/riir-engine/src/` (`latent_functor/`, `cgsp_runtime/`, `hla/`, ...), `riir-ai/crates/riir-games/src/`
- `riir-chain/src/` (`encoding/`, `consensus/`, `economics/`, `asset_lifecycle/`, `forensic/`, `catchup/`)
- `riir-neuron-db/src/` (`shard.rs`, `index.rs`, `merkle.rs`, `freeze.rs`, `mape_k.rs`, `consolidation.rs`, `gateway.rs`, `vibe.rs`, `spectral_flatness.rs`, `shard_compactor.rs`)
- `riir-train/crates/riir-train-{gpu,engine}/src/` (`distill_attention.rs`, `loss_grpo.rs`, `loss_dpo.rs`, `delta_filter.rs`)
- `riir-infer/src/` (the `riir-infer-core` root: `quant/`, `deltanet/`, `transformer/`, `simd/`, `gguf_loader.rs`, `safetensors_loader.rs`, `tokenizer.rs`, ...) + `riir-infer/crates/riir-infer-{gpu,laya}/src/`
- `riir-reflex/src/` (`engine.rs` + `engine/`, `embed.rs`, `label_heads.rs`, `readout.rs`, `harness/`, `lanes/`, `serve.rs`, `game_heads.rs`)

Do NOT create any file until all five done.

## Primary focus — Fusion-first

**The highest-value Super-GOATs come from fusing 2–3 papers/primitives into a novel combination, not from direct-mapping a single paper.** Always grep `.research/` + `.plans/` for the 2–3 closest cousins before verdict, and ask: *"what does paper × note A × note B produce that none of them alone can?"*

### Fusion priority ladder (owner re-set 2026-10-31; was 2026-09-02)

The repos are NOT equal-weight fusion surfaces. **Owner re-set: the product is the SELF-EVOLVE stack — `riir-refine` (rulewright + self-evolve) and the `Rethink (Reflex+Instinct)` decision-serving family — so research must AIM at improve/fusion/novel/adaptive for these lanes, tiered, with poc+proof to pass the GOAT gate or demote (ungoat).** File in THIS order (higher priority wins filing priority for the same primitive):

1. **Self-evolve product stack** — `riir-refine` (corpus, retrieval, fix trajectories, score-bench, the escalation chain `seat_rerank`/`hosted_expert`/`tier_guard`, `self_evolve`) + the decision family (ESC enforcement + rate guard + decstat → rethink; `EscalateSpec`/lane seam/arena → instinct; modelless engine + arenas → reflex).
2. **Game / MMORPG runtime** (`riir-ai` + `riir-game-sdk` + product repos).
3. **Inference-perf league** (kernels/quant/encoders → `riir-infer`; serving + measurement → `riir-reflex`).
4. **riir-chain** (consensus / LatCal / settlement).
5. **Training / fine-tuning / LoRA** (`riir-train` + in-repo pipelines; full pretraining out of scope).

The ladder orders EFFORT, not existence: lower-priority fusions still get recorded, but a higher-priority surface carrying the same mechanism gets the plan first. The standing per-verdict check is now TWO-sided: **"what does this do for the rulewright, and for the decision-serving self-evolve loop?"** — canonical failure #5 (below) is the never-asked case.

Patterns that ship here: latent-to-latent ops (dot-product/cosine retrieval, sigmoid-gated routing, manifold/spectral — never decode→re-encode) · freeze/thaw versioning + atomic hot-swap · runtime adapter routing (inference-time selection, zero training) · self-learn/adaptive CoT (curiosity, collapse detection, trajectory folding — latent state, never base weights) · modelless primitives (pruners, bandits, DDTree, spec decode) · MMORPG-scale crowd AI (batched latent ops; raw sync bit-identical).

### Super-GOAT factory modules — grep FIRST

The highest-value latent Super-GOATs cluster in the module trees below. `list_directory` these explicitly even if the paper looks pure-training — vocab mismatch is the #3 cause of false verdicts:

| Module | What ships | Super-GOAT angle |
|---|---|---|
| `katgpt-rs/crates/katgpt-sense/src/` (NOT katgpt-core/src/sense — that dir does not exist; only `sense_threat.rs` lives in katgpt-core) | belief-state kernels (`evolve_belief`, `reconstruction.rs`), ternary bit-plane projection | Per-NPC recurrent latent state — runtime substrate for any "hidden state"/"belief"/"activation" paper |
| `riir-ai/crates/riir-engine/src/latent_functor/` | `zone_gating.rs`, `reestimation.rs`, `arithmetic.rs`, `cross_game.rs`, `k_selector.rs`, `quality_gate.rs` | **Game-theory in latent space** — vector-op functors, coherence-driven re-estimation, zone-gated activation. Maps "stage"/"application"/"bypass"/"collapse" papers |
| `riir-ai/crates/riir-engine/src/hla/` | `kernel.rs`, `forward.rs`, `types.rs` — **Higher-order Linear Attention** (Transformer attention-layer replacement). Paper: Zhang et al. 2026. **NOT the per-NPC belief** (different layer, different repo). | Maps "attention layer"/"linear attention"/"recurrent state" papers to Transformer-scale ops |
| `riir-ai/crates/riir-engine/src/cgsp_runtime/` | Curiosity-guided self-play, latent prediction SSL, MCTS collapse bridge | Runtime curiosity/exploration — maps "self-learn"/"entropy-driven"/"collapse recovery" |
| `riir-neuron-db/src/` | `shard.rs` (`NeuronShard` Pod, `style_weights[64]`, dendritic branch), `freeze.rs` (`MerkleFrozenEnvelope`), `consolidation.rs` (Raven/δ-Mem), `gateway.rs` (AnyRAG), `vibe.rs` (KG triples), `merkle.rs`, `mape_k.rs`, `spectral_flatness.rs`, `shard_compactor.rs` | **Frozen latent-state storage + integrity + retrieval** — maps "memory"/"replay buffer"/"experience replay"/"spectral init"/"Merkle commitment"/"snapshot"/"KG triple" |
| `riir-chain/src/encoding/latcal*.rs` + `latcal_fixed.rs` | Lattice Calculus: 2×2 matrix arithmetic obfuscation, fixed-point bridge, spectral fixed-point, DeFi programs | **The sync-boundary bridge** — deterministic, committed, raw-numeric. Maps "fixed-point"/"deterministic commitment"/"raw↔latent bridge"/"arithmetic obfuscation" |
| `katgpt-rs/crates/katgpt-dec/src/` | `operators.rs` (`exterior_derivative` d, `codifferential` δ, `hodge_laplacian` Δ), `hodge.rs` (`hodge_decompose`, `betti_numbers`, `harmonic_projector`), `flow.rs` (`DecFlowField`), `stokes_calculus.rs` (`boundary_flux_mass`, `belief_mass_divergence`, `line_integral`) | **Generalized Stokes' Theorem substrate** — `d∘d=0` by construction. Maps "divergence"/"boundary flux"/"line integral"/"curl"/"Hodge"/"Fokker-Planck"/"mass conservation"/"manifold geometry"/"exterior calculus"/"Stokes". **Curse-of-dim caveat: boundary-vs-volume wins only for d≤3 (maps, belief regions, KG embeddings) — NOT high-dim shards.** |
| `riir-infer/src/` (the `riir-infer-core` root) + `crates/riir-infer-{gpu,laya}/src/` | quant (`quant/` EXL3-trellis lane, `deltanet/` GDN lane, Bonsai2 rotation), loaders (`gguf_loader.rs`, `safetensors_loader.rs`, `tokenizer.rs`), `transformer/` + `simd/` forwards, the laya encoder lane (flat `Vec<f32>` ops over CPU + Metal MSL + CUDA kernels); **`src/bin/` tool lanes** (`twt_skip_amputation` — the ShortGPT-skip/ablation arms on the league checkpoint, Bench 022-t54; the sink-audit/neuron-census/readout-screen bins; `si_zero_shot_ppl`) — the measurement + amputation surfaces live in BINS, not modules; grep bins too | **The substrate half of every serving claim** — maps "quantization / weight format / kernel / encoder port / loader / removal / skip" papers; the G5-parity discipline (bit-identical port vs frozen captures, p-drift ≤ 1e-3) is the template for any forward port |
| `riir-reflex/src/` (`engine.rs` + `engine/`, `embed.rs`, `label_heads.rs`, `readout.rs`) + `harness/` + `lanes/` | decision_wire engine (hashed-feature embedding → `pick_domain` corpus routing → `Lz4FlexDrafter` corpus-is-the-model scoring → sigmoid → `SigmoidGateCalibrator` → fused abstain via `CorpusDistanceGate`) + the dataset harness (9 suites, corpus-cap + cal-select levers, pair-head probes) + comparison lanes (laya CPU/Metal/CUDA, CLM, GLiNER, AgentJev) | **Labeled decision serving with first-class abstention** — maps "calibration / abstention / selective prediction / confidence / comparison arena" papers; published tables are CI-regenerated, never hand-typed, and G5 parity gates every laya number |
| `riir-refine/src/draft/` | span-level retrieval under `latent_retrieval`: `AstChunker` → `ModellessEmbedder` → fan-out `RuleIndex` (riir-rag: Clifford-wedge KNN + BM25) + `RerankMode::Structural` + oracle-labeled eval fixtures (`retrieval_eval.rs`, `eval_fixtures_composite.rs`, `clippy_oracle.rs`) | **A labeled (rule ↔ code-shape) binding corpus with measurable downstream accuracy** — maps any "structured index / retrieval / compositionality / binding" paper; the withheld-pair OOD generalization axis is unmeasured almost everywhere |
| `riir-refine/src/self_evolve.rs` + `src/traj_store.rs` + `src/elo.rs` | fix-trajectory memory (Warm-tier `.refine/` store, evidence tiers Certified/Heuristic/Withdrawn, Elo + BetaPosterior selection, `EvolveRecorder`) | Maps "experience replay / memory / trajectory store / bandit-evidence" papers — outcomes are labeled and persisted |
| `riir-refine/src/domains/` + `corpus_gen` + `rustc_errors` playbooks | 580+ rule corpora across six domains; error-code-keyed repair playbooks; per-domain centroids (`KernelExpertRouter::pick_domain`) | Domain/category keys ARE roles — maps "role-conditioned routing / expert routing / structured generation" papers |
| `riir-instinct/src/` (`arsenal.rs` — `EscalateSpec` grammar + manifest law; `arsenal_ops.rs` — `LaneSlot`/epoch swaps; `serve_edge.rs` — serve edge + R5 LRU) + `riir-rethink/src/` (`escalating_backend.rs` — ESC wrapper/`EscRateGuard`/ESC_POSTURES; `arsenal.rs`; `encoder_serve.rs`; decstat) | the decision-serving self-evolve family: open grammar/seam/arena + private ESC enforcement/rate-guard/economy | **Serving-surface consumer check** — maps "escalation / selective serving / lane seating / serving economy" papers; ⛔ the ESC gate is EXACT/stationary — dynamics signals need the mutated-state test (§1 step 4a) |

**Adapter routing, KV compression, speculative decode = GOAT-tier framings. Latent-to-latent ops on belief/functor/shard/LatCal state = Super-GOAT-tier framings — attempt Super-GOAT first; defaulting to adapter routing when a latent reframe is stronger is the primary failure mode.**

**Redirect to riir-train** — exhaust §3.5 first (a training-looking mechanism may decompose into shipped primitives — Flow Sampling is the canonical case). Training is actively pursued, not lazily redirected: applicable training papers (optimizer, loss, schedule, recipe, LoRA/OFT/IA3/QLoRA/DPO/GRPO/SFT/RL) get a **Plan in `riir-train/.plans/`** with recipe + GPU-hours estimate + GOAT gate vs modelless baseline; only genuinely out-of-scope training (image-specific DiT, medical imaging) gets a one-line redirect with justification. Runtime GRPO self-play stays in `riir-ai` (latent state, not weights); model-based training lives in BOTH riir-train AND in-repo pipelines; quant-aware **inference** stays here, quant-aware **training** → riir-train. Never assume a repo is modelless-only (pre-flight #5).

## Workflow

Every path through this workflow terminates at **§5 — the Claude verdict ping-pong** — before anything commits. §1.5 / §1.55 / §1.6 grade the verdict; §5 is what makes it final.

### 0. Read & classify

Fetch via `https://r.jina.ai/https://arxiv.org/pdf/{ID}`. Ask: *is the value in the training loop itself (optimizer/loss/schedule/RL), or in the math the training computes (closed-form drift, conditional score, Riemannian correction, steering formula)?* Optimizer/loss/schedule → riir-train. **Math → run §3.5 Path 0 first.**

**Label-anchoring hazard (canonical failure #2 — arXiv:2608.13335 NQF, 2026-08-18):** the paper's own vocabulary ("trained by gradient descent", "learning dynamics") is a routing hazard — training-dynamics papers routinely carry closed-form modelless content (ignition-time formulas, trajectory families, patience laws, leading-behavior predictors); NQF routed wholesale to riir-train although its theorems carried exactly that (surfaced by user challenge). The same hazard applies to NON-paper inputs: the document's name for its host is routing INPUT, never the routing verdict (2026-10-07: an encoder-lane proposal naming "the Reflex decision family" was first routed public although a trained-encoder serving lane is rethink's charter). Defense: the family≠repo rule + the MOAT rows + the §5 public/private tier check. Classification comes AFTER the §3.5 two-question decomposition + adversarial panel — never from the abstract's framing.

### 0.5 External-repo source access — clone into `.raw/` (ephemeral, `rm` when done)

Papers distill fine over the wire (`https://r.jina.ai/...`). **External CODE repos do not** — distill verdicts, mining batches, and prior-art verification need full-tree `grep`, byte-accurate quotes, and multi-file reads that URL fetches cannot provide. When a research task needs an external repo's source:

```bash
# Clone (shallow by default; full history ONLY if the task mines git log)
git clone --depth 1 https://github.com/<org>/<repo>.git .raw/<repo>
# Pin provenance BEFORE reading anything into a verdict:
git -C .raw/<repo> rev-parse HEAD   # record this sha + the license in the note
```

Rules (all mandatory):

1. **Location:** `.raw/` at the repo whose `.research/`/`.plans/` will hold the output (e.g. `katgpt-rs/.raw/ds4/`). Ephemeral scratch — gitignored everywhere, never committed, never a sibling repo.
2. **Pin before verdict:** record `@ <full-sha>` + license in the note BEFORE deleting the clone — the sha is the provenance; every later-cited quote must match it.
3. **Read-only.** Never modify files inside `.raw`; copy out before experimenting.
4. **No build-graph contamination:** never add a Cargo/path dep into `.raw`, and never include `.raw` in grep scopes — it is NOT shipped substrate; grepping it makes a mining candidate read as "already ships".
5. **Cleanup is part of the task:** `rm -rf .raw/<repo>` at the filing commit. A finished task with a live `.raw/` entry is unfinished; a stale `.raw/` on resume is UNTRUSTED — delete and re-clone at the pinned sha.
6. **Shallow unless history-mining:** `--depth 1` default; full clone only for `git log`/diff archaeology (say so in the note); `--filter=blob:none` for very large repos.

### 1. Distill fundamentally — fuse, don't direct-map

Find the transferable primitive (the geometric/spectral/information-theoretic insight that works without the paper's training setup). **Then look for fusion opportunities**: cross-pollinate with existing notes/plans/shipped primitives to synthesize something novel.

**Fusion protocol:**

1. **Grep ALL workspace repos, BOTH layers (notes AND code), in parallel via subagents — product set first, then the wider set (derive the set, never hand-type). Grep results are PAGINATED — run `offset` until exhausted, or scope per-repo so every repo's hits are visible.** Closest cousin is frequently in the OTHER repo or in CODE not notes (mechanisms ship without research notes — `evolve_belief` has no `.research/` framing). Grep:
   - Every `.research/` + `.plans/` folder in the workspace (derive both — typed counts of either rot within weeks; riir-train included: applicable training papers get Plans per §3.5 Path 0.5)
   - `riir-ai/.docs/` (the moat/selling-point book — grep alongside `.research/` so you don't claim novelty over a pillar that ships)
   - All shipped `src/`/`crates/` trees (notes describe intent; code describes what shipped)
   - The twelve Super-GOAT factory modules above

   **No exclusion filters on sweep results (2026-11-26, Plato's Key arc):** never pipe sweep output through a `grep -v` "false-positive" filter — classify hits by READING them, never by pattern exclusion. The session filtered out `cross_game` hits as presumed noise and with them the closest modelless cousin (`riir-ai/crates/riir-engine/src/latent_functor/cross_game.rs` — the paired-anchor cross-space transfer module, Plan 303 T5.5; surfaced by owner challenge → riir-ai Issue 1265). A filter removes candidates by NAME; only a read classifies by MECHANISM.

2. **Vocabulary translation BEFORE grepping.** List the paper's 3–5 key terms; for EACH brainstorm ≥2 codebase equivalents ("if we shipped this, what would we call it?") and grep BOTH sets. **Family/wire stems (2026-11-12):** also translate STRUCTURE families — the genus may ship under a stem the paper never uses (`gram`/`topogram`/`grammaturgy`, `vessel`, `book`+`walk`, `wire`); grep stems alongside mechanism terms; family maps (`riir-grams` README, `riir-ai/.docs/01_orientation/`) are first-class cousin surfaces. `read_file` `vocab.md` (sibling) for the standing vocabulary tables. **Glossary FIRST (canonical failure #6):** `read_file` the fleet glossary (`riir-ai/.docs/01_orientation/glossary.md`) BEFORE inventing variants — coined words (`drive`, `block λ`, latch) are standing vocabulary. **Coefficient-shape mechanisms** hide as hand-tuned constants (`W_EVO = 0.6`, `alpha: f32`), never under the word "blend" — vocab.md §6's grep set, run across ALL repos (canonical failure #3).

3. **Latent-space reframing BEFORE verdict.** Re-cast the core mechanism as a latent-to-latent op on the codebase's kernels: (a) per-NPC belief, (b) `latent_functor/` ops, (c) `cgsp_runtime/` curiosity, (d) LatCal fixed-point commitment, (e) `NeuronShard` style_weights/dendritic/`MerkleFrozenEnvelope`/Raven/AnyRAG, (f) DEC Stokes operators (d, δ, `hodge_decompose`, `DecFlowField`). If you only reach adapter routing/KV/spec-decode framing → likely GOAT, missed the Super-GOAT angle.

4. **Game-context AND consumer-context reframing BEFORE verdict** (especially when step 3 returns no hits). Game: *how does this manifest as a per-NPC behavior signal / crowd pattern / selling point?* (a guarantee → which per-entity scalar it bounds and what behavior it can drive; a structure → NPC routines / market cycles / quest scheduling; a number-theoretic property → fairness/diversity on a game signal; a geometric property → phase scheduling / spatial spread / fog coverage). **Consumer (priority #2, same evidence bar): does it manifest on a rulewright surface — (a) retrieval index / corpus structure, (b) fix-trajectory memory / selection, (c) score-bench / eval axis, (d) drafter-pruner selection, (e) rustc_errors playbooks?** One sentence each; a bare "no" without reading the surface is the canonical-failure-#5 shape. **Reframe sentences cite the grep actually run (pattern + scope) — an asserted absence before any surface grep is the same shape (2026-11-26: the Plato's Key note graded the game reframe "thin" before any game-surface grep; the owner challenge then surfaced `cross_game.rs` and regraded it moderate).** **If either reframe is missing, you are NOT ready to PASS.**

4a. **Stationary-surface test for DYNAMICS signals (2026-10-31, rethink Issue 047 round 1):** before wiring churn/slope/oscillation/drift onto a consumer, read its gate/loop and NAME the state variable it mutates in response to evidence. A pure function of inputs (exact gate, feed-forward scorer, frozen-pick replay) has no step size and a value slope stationary BY CONSTRUCTION — the seeded "improving" posture is unproducible and the monitor fails its own separation gate. Static surface ⇒ LEVEL readouts (counters/ratios: precision, value-per-use, band position); dynamics signals only where mutated state exists (bandits, belief integrators, training loops, LRU under load).

5. **Zero hits ≠ novelty.** Most likely you're using the wrong vocab — try a third semantic angle (grep *output behavior* "swap when X" instead of *mechanism name* "tightness monitor") before claiming "no prior art". **Zero hits from an UNPROVEN instrument are VOID (2026-11-26, Plato's Key arc): before trusting any zero-hit sweep, prove the instrument live in the same scope — grep one known-present string (e.g. `procrustes` in katgpt-rs crates) and require it to hit. The session read three zero-hit `rg` sweeps as "clean internal prior art"; `rg` resolved through a shell-function wrapper a non-interactive shell failed to resolve, and the workspace grep tool then surfaced `katgpt_spectral::procrustes` + `katgpt-canon` Bench 562 (the primitive ALREADY shipped). A zero from a tool that may not have run is not evidence.**

6. List the 2–3 closest cousins across all repos — **read, never title-judged**: every hit from the family sweep gets a TL;DR read (`head -30`) before the cousin list is final; a title that doesn't obviously match is not a dead lead (canonical failure #7's find path: `katgpt-rs/.research/594_TWT_Phase_Collapse_Decoder_LLMs.md` sat in the Model Carving sweep hits unread — it WAS the skip-removal-class cousin). Ask: *what novel combination of paper × A × B produces a capability none has alone?* Write that into the note's §Distillation as a **Fusion** subsection even if unplanned.

7. Verdict by tiers (§1.5). Create research `.md` at the right repo. Naming: `{NNN}_{Short_Title}.md` (next free number, monotonic, never reused). Run the allocation gate (`py ../katgpt-rs/scripts/dual_allocation_gate.py` from the owning repo, after `git fetch origin`) at allocation time AND re-run it after the final rebase, before push — a sibling's in-flight push can allocate your number inside the window (2026-11-26: riir-ai Issue 1264 allocated twice; citation weight kept origin's, the loser renumbered to 1265). Format: `read_file` `templates.md` (sibling) — canonical example: `katgpt-rs/.research/238_LoRA_Muon_Spectral_Low_Rank_Manifold.md`.

### 1.5. Novelty gate — is this Super-GOAT?

Score all four:

1. **No prior art?** Grep notes + plans + shipped code across all repos, using BOTH paper vocab AND codebase-vocab alternatives. **Read every grep hit's TL;DR before claiming novelty** — a filename match is a lead, not confirmation. Symmetric rule for the opposite direction: a cousin NAME match is a lead toward coverage, never coverage itself — "already ships" claims require the §3.6 signal-diff check. Mechanisms ship under non-obvious names (DiPOD's "interleave self-distillation when ELBO drifts" ships as "coherence-driven re-estimation scheduler when coherence < tau_reest"). When the candidate selling point touches per-NPC + memory + personality + swap, the `riir-ai/.research/` corpus is saturated — grep it broadly and READ every hit before claiming novelty.
2. **New behavior class?** Not better numbers — a capability no incumbent has. **Requires step 4 game-context reframe to detect.** If you reached Q2 without step 4, go back.
3. **Product selling point?** Can you finish: *"Our NPCs/systems do X that no competitor can"*? If not → Gain.
4. **Force multiplier?** Connects to ≥2 existing pillars/systems. Solo novelty = GOAT, not Super-GOAT.

**All 4 YES → Super-GOAT.** Mandatory outputs (in this same session):
- **Open primitive** → `katgpt-rs` (generic math, no game semantics).
- **Architectural GUIDE** → private selling-point doc. Repo by selling-point domain: `riir-ai/.research/` for game-runtime/belief/functor/self-learn; `riir-chain/.research/` for chain/LatCal/commitment/sync-bridge; `riir-neuron-db/.research/` for shard/freeze/consolidation/AnyRAG/vibe/Merkle. Cross-domain selling points → primary guide in the repo owning the boundary being crossed. Guide includes: TL;DR with commercial value, distilled modelless primitive, connection map, latent↔raw boundary, what's private vs open, validation protocol, P0–P3 priority.
- **Plan(s)** → appropriate `.plans/` folders.

> **No "candidate" escape hatch.** Writing "all 4 YES" / "passes the gate" / "Super-GOAT candidate" anywhere in a note triggers the mandatory outputs in THIS session. If not confident on all 4 → write "fusion idea, novelty TBD" + create `.issues/` entry. "Candidate" is not a deferred-commitment escape — it either triggers the guide now, or downgrades to an issue.

**One verdict per track, not one per paper (TTPO lesson, 2026-08-29).** When a paper yields BOTH a training recipe AND modelless extractions, score this gate SEPARATELY per track — a Q1 kill on one does NOT cascade to the other. "Recorded in the note" is NOT a tier: every Path-0 inventory row ends in exactly one of (a) filed plan (GOAT+), (b) `.issues/` entry (fusion idea / novelty TBD), or (c) an audited discard whose reason cites the mechanism-level §3.6 diff.

**Pin the claim BEFORE searching it (§4 precondition — TTPO lesson, 2026-08-29).** Before running ANY §4 search, write the claim in one sentence: "<mechanism> for <surface/consumer>, consuming <signal>, distinguished from <closest cousin> by <delta>." A novelty check against an unpinned claim is void — class-level prior art kills the CLASS, not the MECHANISM; the verdict regresses to vibes. If you cannot write the sentence, you have not distilled enough to claim novelty either way.

### 1.55. PASS vs Gain — no middle tier

PASS = no new research/plan files. Gain = files. Scan the paper for actionable improvements before verdict.

- Ships + actionable improvements → **Gain** (create `.issues/` per AGENTS.md "Create issue at .issues for poc, proof, optimization or refactor task, do not create plan", and/or plan behind feature flag).
- Ships + no actionable → **Pass** (verdict + one-line reason + closest shipped cousin, in conversation only; no files).
- Doesn't ship + modelless → **Gain** or higher.
- Doesn't ship + training-only → run §3.5 Path 0.5. Applicable → Plan in `riir-train/.plans/`. Out of scope → one-line redirect with explicit justification.

**PASS-Redirects line (mandatory):** PASS verdicts still update the 1–3 closest shipped cousin `.research/` notes with a one-line reference. Format:
```
> **PASS-Redirects (synthesis):** <Author> [arXiv:XXXX.XXXXX "<Full Title>"] — <one-line reason>.
```
Must include arxiv ID AND full title (so `grep arxiv:ID` AND `grep "Title"` hit). **Prevents paper-number invisibility** — without this, a future session greps the arxiv ID, finds nothing, re-distills from scratch.

### 1.57. Implementation-readiness audit for plans (T3 — Issue 977 / EGM plan-readiness)

**Before executing any `.plans/NNN` task set** (once per plan, FIRST): audit *does each task name WHAT to change, WHERE, and HOW — or is a design choice left to the implementer?* A NAMED decision point is a designed gate, well-formed; a silent deferral ("wire the lane" — which lane?) makes the implementer author design mid-flight, un-reviewed. Resolve findings INTO the plan before implementation, never mid-flight.
- **Self-test (the zero-finding trap):** the first execution seeds a deliberately vague task ("implement the extractor") that MUST be flagged — a zero-finding run on the fixture FAILS the audit rather than reading as clean plans.
- **Worked proof (2026-11-04, Issue 977):** the two most-recent plans at filing — `632_grid_reselect_lambda_g1` (CLOSED) + `626_swir_anchor_exit` (CLOSED-NEGATIVE): zero findings on both (632's T0 is the honest named-gate shape; 626's Phase-0 tasks each name fixture/model/bar/record). Human-confirmed real: both executed same/next-day to a recorded verdict with no mid-flight-design deviation notes — the audit's own success criterion.

**Actionable = Gain.** Actionable: paper data contradicts a current config default; exposes a failure mode with no existing mitigation; unblocks a deferred task. NOT actionable: "validates our design" / "theoretical lens" / "could inform a future config". Unsure → not actionable → Pass.

**Reverse-grep before PASS (mandatory).** Before any PASS, grep the codebase for documented gaps the paper could fill: `.docs/` for `Limitation|deferred|TODO|FIXME|gap|pending`; `.benchmarks/` for `Caveat|deferred|artifact`; `.rs` comments for `TODO|FIXME|deferred|limitation` near the paper's vocab. Any hit that maps → Gain, not Pass. Compact heuristic: "is there any documented limitation this paper could fix?" — if you can't answer "no, I checked" with evidence, don't PASS.

**Training papers third defense:** both defenses above can return clean and still produce a false PASS if "model-based track" is narrowed to one pipeline or one repo. Before PASS-ing any training paper: (1) `read_file riir-train/.docs/02_pipelines/training_data_pipeline.md` and ask: does a training pipeline for this domain already exist? (RL, distillation, linear attention, trajectory collection). (2) **`grep` ALL repos for existing training/model-based code** (pre-flight #5) — training pipelines ship OUTSIDE riir-train: `quest_grammar/grammar_training.rs` + `quest_training.rs` in riir-ai (LoRA training for quest grammar + quest corpus), `TernaryDraftModel` in riir-ai/riir-refine (ternary-weighted drafter loaded from `.bits` files), `self_evolve` in riir-refine (self-adaptive latent-first loop). If ANY existing training/model-based code maps to the paper's domain → Gain, not Pass. Canonical failure (arXiv:2511.18538): Code LLM survey falsely PASSed because agent checked only riir-train (game AI pipeline), missing quest_grammar's code-domain LoRA training + riir-refine's ternary drafter + Issue 010 (generalization gap that RLVR would fix).

**Ungoat discipline + file hygiene (owner directive 2026-10-31 — "poc+proof to pass goat gate or demote (ungoat), for less noise/filesize"):** every GOAT+ filing carries its PROOF PATH in the filing — the fixture, the falsification arm, the pre-registered gate (a number where one exists). No runnable proof path ⇒ `.issues/` entry at most, never a GOAT note/plan "for the record." A landed primitive that FAILS its gate is DEMOTED in the same commit (grade struck, tier → Withdrawn/demoted, verdict line updated) — stale grades are noise. Prefer the smallest artifact that carries the verdict: PASS → conversation + PASS-Redirects; fusion idea → one issue; a note only when the distillation itself IS the durable artifact.

### 1.6. MOAT gate per domain + promote/demote

Tier verdict measures *how strong*. MOAT gate measures *whether it strengthens THIS repo's moat*. Mismatch → reroute.

| Domain | MOAT bar | In scope | Out of scope |
|---|---|---|---|
| `katgpt-rs` | Fundamental/principle/base primitive via fusion; promote/demote tracked per stack | Transformer stack (layers/attn/KV/sampling/sparse/quant-aware **inference**/spec decode/DDTree/MCTS/bandits/pruners); 2D toy games; DEC/Stokes; belief kernel; sigmoid mechanics | Product game wiring → riir-ai; chain → riir-chain; shards → riir-neuron-db; weights → riir-train |
| `riir-ai` | Pillar-level or Super-GOAT (fusion connecting ≥2 pillars, or new pillar candidate) | Adaptive/self-learn NPCs, reasoning pack, MMORPG-scale (20Hz/fog/zone/crowd MCGS), 3D game wiring, freeze/thaw runtime, latent ops on belief/functor/cgsp | Generic transformer → katgpt-rs; chain → riir-chain; shards → riir-neuron-db; training → riir-train |
| `riir-chain` | Pillar-level or sync-boundary bridge novelty | LatCal, quorum/catchup, economics, asset lifecycle/forensic, DeFi, `riir-chaind`, raw↔latent bridge | Generic fixed-point without commitment → katgpt-rs; shards → riir-neuron-db |
| `riir-neuron-db` | Pillar-level or shard/freeze/consolidation novelty | `NeuronShard`, freeze envelope, Raven/δ-Mem, AnyRAG, vibe KG, Merkle, spectral init, compaction, dendritic | Chain commitment of shards → riir-chain; runtime swap → riir-ai |
| `riir-train` | **Active moat.** Training-method implementations + configs + trained weight assets | Adapter training, optimizers, losses, quant-aware **training**, DPO/GRPO/SFT, trained assets | Inference/runtime/latent ops → katgpt-rs or riir-ai |
| `riir-refine` | **Consumer-first moat (investment #2).** Measured rulewright-quality gains: retrieval floors, score-bench heal rate, OOD generalization, fixer coverage | Corpus/retrieval structure, trajectory memory, selection math, eval harnesses, rulewright-consumed primitives (`.research/` notes live here for rulewright-domain distills) | Generic math stays in katgpt-core (consume, never fork); game-runtime concerns → riir-ai |
| `riir-dapps` | Settlement-composition + service-economy novelty (game outcome → generic chain settlement; the KAT ledger/server plane) | `Settlement` vocabulary, escrow/wallet backends, kat_ledger + epoch settle, payment rails, DO/worker shell, mint-receipt pipeline | Chain value/authority primitives → riir-chain; game rules → riir-ai; economy governance → riir-dao |
| `riir-infer` | **Public substrate moat.** Inference-substrate novelty (quant formats, kernels, encoders, loaders) shipped public + upstream-clean (zero riir-* deps) | Weights/quant/architectures/kernels/loaders; EXL3-trellis, GDN, Bonsai2 lanes; the laya encoder lane (CPU/Metal/CUDA); GPU backend layers | Decision serving → `riir-reflex`; training loops/recipes → `riir-train`; engine/game wiring → `riir-ai`; generic modelless math → `katgpt-core` |
| `riir-reflex` | **Decision-serving moat.** Calibrated modelless decisions with first-class abstention + honest comparison arenas (G5 parity gates every published laya number) | `decision_wire` serving, calibration gates, fused abstention, dataset harness + corpus-cap levers, laya/CLM/GLiNER/AgentJev comparison lanes | The laya lane substrate → `riir-infer`; generic primitives → `katgpt-core`; game runtime → `riir-ai`; training → `riir-train`; TRAINED-ENCODER serving arms → `riir-rethink` |
| `riir-instinct` | **Public open-stack moat.** Lane/seating/escalation-grammar novelty shipped public | Suites, the `LaneBackend`/`ExtBoots` seam, `EscalateSpec` one-definition, arena game heads, the open L1→L3 stack | Product posture + ESC enforcement + depth economics → rethink (private); modelless engine + arenas → reflex; encoder forwards/kernels → riir-infer |
| `riir-rethink` | **Private-forever moat.** The trained-encoder decision product — nothing here may land in a public repo | The trained-encoder serving rung, its weight artifacts and serving posture, escalation-lane enforcement + rate guard, its economy planes | The escalation GRAMMAR → instinct (public); the encoder forward SUBSTRATE → riir-infer (public); re-fit recipes → riir-train; ANY public-repo file — this tier's product direction never lands in katgpt-rs/riir-infer/riir-reflex/riir-instinct/katgpt-web |
| `riir-dao` | Bounded-autonomy tokenomics governance novelty (modelless selection + guards + advisory, never unguarded execution) | Signals/strategy candidates, guard engine, simulator, advisory transport, decision artifacts | Ledger bound-enforcement stays riir-dapps (the agent only proposes); generic rating/Beta math stays katgpt-core |

9 sloppy-test pillars live in `riir-ai/.docs/03_pillars/README.md` — `read_file` `03_pillars/README.md` + `04_supergoat_candidates/README.md` before any "does this become a pillar?" verdict. Sloppy test: *if it doesn't exist, the system goes structurally sloppy — not slower, broken.*

**`katgpt-rs` per-stack ledger:** every primitive gets a feature flag + benchmark + GOAT gate; the verdict note records which stack slot (attention/KV/sampling/speculative/pruning) + promote-to-default or stay-opt-in; re-gate on feature touch; demote the loser when a newer primitive wins the same slot.

### 1.7. Pre-plan cherry-pick audit

If your plan will consume/wire/fuse with a katgpt-rs primitive into riir-* → run the `goat-audit` skill before opening the plan. Catches (a) stalls (default-on in katgpt-rs ≥7 days with zero riir-* consumer), (b) DRY violations (riir-* shipping a local duplicate of substrate — e.g. defining its own `KVCache` instead of consuming `katgpt-transformer`). When NOT to run: purely katgpt-rs-internal; bug fix no cross-repo angle; training-only genuinely out of scope.

### 2. If gain/GOAT, plan it

Plan `.md` in `katgpt-rs/.plans/` (modelless), `riir-ai/.plans/` (runtime/game), `riir-chain/.plans/` (chain/LatCal), `riir-neuron-db/.plans/` (shards), `riir-train/.plans/` (training-efficiency per §3.5 Path 0.5). `## Phase N` sections, `- [ ]` per task (`- [x]` done, `- [-]` deferred). Planning into riir-train IS allowed + actively encouraged for applicable training-efficiency papers. Super-GOAT plans come AFTER the guide — guide is strategy, plan is execution.

**Plan format + GOAT gate rule + UQ "Report the Floor" extension:** `read_file` `templates.md`. Compact: every new technique needs feature flag + benchmark proving gain before promoting to default; demote loser if newer wins same slot. UQ-bearing primitives (distributions/intervals/coverage) MUST benchmark against conformal-naive floor (`ConformalIntervalCalibrator<SeasonalNaiveForecaster>`) on CRPS/coverage/Winkler — if they can't beat it, the gate FAILS.

### 3. Implement to unblock

If a plan is blocked by a missing primitive, implement the minimal version. After GOAT check + proof of gain: promote to default if it wins, demote the loser.

### 3.5. Modelless unblock — MANDATORY before any riir-train deferral

Before deferring ANY gate/task/mechanism to riir-train, exhaust modelless paths first.

**Path 0 — training-target decomposition.** Is the value the **training-loop innovation** (new optimizer/loss/curriculum/RL) or the **math it computes** (closed-form drift, conditional score, Riemannian correction, steering formula, regression target)? If the math → decompose into components, grep for modelless analog of EACH. ALL have analogs → MODELLESS-VALIDABLE (no deferral). SOME missing → check paths 1–3 for those.

**Path 0 asks TWO questions per component, not one.** (1) **Coverage** — does an analog already ship? (feeds the §1.5 Q1 novelty gate). (2) **Extraction** — can the component be computed WITHOUT gradient descent: closed-form solution, derived schedule, predictor, bound, ordering law, limit behavior, offline-computable spectrum? A row with no coverage but yes-extraction is the STRONGEST finding class (open-primitive candidate → katgpt-rs) — marking it "no analog" and moving on is the canonical-failure-#2 miss.

**Path 0 outputs an INVENTORY, not just a verdict.** Complete the component table even when the funnel lands on Path 0.5 — every row marked "analog exists" or "partial" is a candidate open primitive. Before dismissing any row as "covered", run the **signal-diff check** (§3.6): read the shipped analog's core formula and state what signal it *consumes* vs the paper component's. Name-level cousin match ≠ coverage — the diff that surfaces a real gap sounds like "relevance vs utility" / "history vs current-query" / "aggregate vs counterfactual", answerable with ONE read of the cousin's code (canonical failure: the modelless δ — query-conditional counterfactual utility — was dismissed as "covered by" a relevance-only gate `σ(dot(q,k)/τ)` and a history-only tier; the gap was pinned as a feature, not a TODO, → katgpt-rs Issue 656; reverse-grep could NOT catch this class).

**Granularity rule (canonical failure #3 — arXiv:2608.16739 Le Critique, 2026-08-20): diff the CONTAINING expression, not only the named cousin.** The BetaPosterior↔TETHER signal-diff was run and correctly returned “different shapes” (per-candidate posterior vs global mixture weight) — but the formula that CONSUMES BetaPosterior, `W_EVO·evolution + W_RATE·reliability` in riir-refine `select_best_candidate`, is exactly TETHER's `b(ρ) = (1−ρ)p1 + ρp2` with ρ hand-pinned at 0.4 and never swept, while the realized-outcome feed (`EvolveRecorder::record_outcome`) already fires on every applied fix (→ riir-refine Issue 033; found by user challenge — third rescue of this class after LOPD and NQF; user challenge is a load-bearing defense layer, treat every challenge as a signal-diff granularity audit). Rule: after diffing cousin C, read ONE level up — the expression combining C with other signals. Hand-tuned constants composing C (`const W_X: f32 = 0.6`) are the paper's shape wearing a different name; the grep is `W_[A-Z]|const [A-Z_]+: f32` on C's caller (one cross-repo grep had sufficed).

**Three-track adversarial panel — MANDATORY whenever the first-pass classification touches training** (routes to riir-train, PASSes a training-adjacent paper, or the abstract carries optimizer/loss/RL/backprop framing). Spawn 2 `spawn_agent`s IN THE SAME parallel batch as the §4 prior-art searches (one spawn round total). Advocacy briefs — neutral merging is the coordinator's job; each agent argues FOR its track and does NOT inherit your classification (briefs above).

- **No-GD advocate** (tracks a+b: modelless inference + self-adaptive runtime): "Extract every closed-form/derived quantity, predictor, schedule, bound, ordering law, limit-behavior, offline-computable spectrum in this paper. For each: can it ship without training, in which repo/module, what GOAT gate? Argue FOR feasibility; do NOT judge whether it already ships — another pass owns coverage."
- **Model-based advocate** (track c: trained weights): "Extract every actionable training-recipe item (optimizer/loss/schedule/init/rank/width/capacity/data). Which existing pipeline does it improve (riir-train Plans, quest_grammar, TernaryDraftModel, edge_lora, riir-infer's EXL3/GDN quant lanes)? Recipe + GPU-hours + GOAT gate vs modelless baseline. Argue FOR applicability."

**Coordinator merges both into the Path 0 table.** Discarding an advocate finding requires a one-line auditable reason in the note; **a discard reason must survive mechanism-level scrutiny (TTPO lesson, 2026-08-29):** class-level prior art, pure cost notes, and action-level coverage ("the prune action ships in Plan 133") are NOT sufficient kills when the finding includes a calibration law, a product-form signal, or a wiring fusion those references do not carry — cite the exact uncovered delta, or do not discard. **Brief hygiene — do not leak classifications through the stack description:** describe repos by WHAT SHIPS (file + type names), never by your shape conclusion (the Le Critique miss propagated exactly this way). **Coordinator spot-check rule (2026-10-31, Research 485):** an advocate's CODE FACT (const, default, field name, file location) is cited into a filed artifact only after ONE coordinator grep — briefs are arguments, not evidence; unverified advocate facts ship as "advocate-reported, unverified" or not at all. Clearly inference-side papers (KV cache, sampling, spec-decode) skip the model-based advocate; clearly pure-hardware papers skip the panel because the §0 technique-stripping pass already ran.

**Path 0.5 — training-cost-weighted re-evaluation (DEFAULT for training-efficiency papers as of 2026-08-06).** Training efficiency is actively pursued. Applicable training paper → Plan in `riir-train/.plans/` with recipe + GPU-hours estimate + GOAT gate comparing trained-vs-modelless baseline. Only genuinely out-of-scope → one-line redirect with justification.

**Track priority — serving-envelope fit (TTPO lesson, 2026-08-29).** When BOTH tracks produce filed plans, rank by whether the mechanism's decision layer runs INSIDE the hot path: modelless shallow selection (branch rollouts, pruner/selector choices, rulewright spans, 20 Hz game ticks) vs outside it (gradient TTT / per-item training — structurally unaffordable where base competence ≈ 0 or per-fix budget ≫ serving budget; canonical datum: the L4 fixer's ~191 s/fix + 0/60 EM). The envelope-fit track is the PRIMARY plan; the other is SECONDARY and says so in its Status line. GPU-hours affordability (Path 0.5) is necessary but not sufficient — a cheap training run that never reaches a production consumer loses to a modelless gate on the live serving path.

**The model-based track = ALL training pipelines across ALL repos, not just riir-train** (GDN-blog lesson + arXiv:2511.18538 lesson). `read_file riir-train/.docs/02_pipelines/training_data_pipeline.md` before PASS-ing any training paper — RL/distillation/linear-attention/trajectory-collection pipelines all count. **BUT ALSO grep non-riir-train repos for training code** (pre-flight #5): `quest_grammar/grammar_training.rs` + `quest_training.rs` in riir-ai (LoRA training for quest grammar + quest corpus), `TernaryDraftModel` in riir-ai/riir-refine (ternary-weighted drafter loaded from `.bits` files), `self_evolve` in riir-refine (self-adaptive latent-first loop). Training is NEVER "deferred indefinitely" — it is actively pursued across the whole stack.

**Systematic backstop:** when ≥3 training-recipe gaps accumulate from PASS-Redirects across repos, batch them into a single `riir-train/.plans/NNN_training_recipe_gap_backlog.md` plan.

**Three modelless unblock paths (check ALL before deferring, AFTER Path 0 fails; run Path 0.5 AFTER all three fail):**

1. **Freeze/thaw snapshot correction** (`MerkleFrozenEnvelope`) — can a corrected snapshot, thawed at inference, fix a systematic bias (doubled signal, position offset, attention asymmetry)?
2. **Raw/lora reader-writer hot-swap** (`LoraPair { reader, writer }`; `LoRAHotSwap`, `dispatch_lora_merge` in riir-ai) — can a **deterministically constructed** (not trained) adapter fix it? Closed-form (scale-by-0.5, zero-out-positions, identity-minus-projection) rather than gradient descent?
3. **Latent-space correction** — project latent onto a correction direction, gate the output (dot-product + sigmoid per constraint #2); the modelless analog of a trained adapter.

**Decision protocol:**
```
Paper appears to need training
  → Path 0: value = MATH not training loop? YES → decompose + grep modelless analog of each
      ALL have analogs + a use case → MODELLESS-VALIDABLE. No deferral.
      SOME missing → paths 1–3 for those. ALL fail → Path 0.5.
      (either way: rows marked exists/partial → signal-diff EACH before "covered" — §3.6)
  → Systematic characterizable cause? YES → paths 1→2→3. ALL fail → Path 0.5.
  → Path 0.5: applicable → Plan in riir-train. Out of scope → redirect with WHY.
```
**Documentation requirement** for every riir-train Plan: Path 0 decomposition (analog / extracted / deferred per component) · paths 1–3 checked + why each failed · what specifically requires GD no deterministic construction can provide · Path 0.5 affordability (recipe + GPU-hours + GOAT gate) · dual-track contribution (modelless inference primitive + trained weight artifact).

### 3.6. Defend-wrong PoC — MANDATORY before any "already ships" / "parity" verdict

Before claiming a mechanism "already ships", achieves "parity", or "covers" the paper's loop, distinguish three claim types:

| Claim | Proof |
|---|---|
| **Architectural** ("runtime analog exists") | grep + read code (sufficient) |
| **Latency/resource** ("modelless, sub-µs, no GD") | criterion bench |
| **Quality** ("matches/beats paper's numbers") | **head-to-head PoC on controlled toy — architectural reasoning NOT sufficient** |

**Failure mode:** claiming all three with only architectural evidence. Grep proves mechanism exists; it does NOT prove it performs as well as the paper's version.

**PoC mandatory when:** quality-parity verdict ("matches", "competitive with", "covers at parity"); qualitative Super-GOAT/GOAT claim; **any PASS that downgrades on grounds "the runtime analog already ships"** (architectural-only PASS is the #1 false-PASS mode).

**Scope extension (2026-08-16, LOPD lesson): coverage-dismissals inside Gain verdicts need the same defense.** Claiming a paper's secondary mechanism is "covered by cousin X" — while filing the primary mechanism as a training plan — is a verdict, and it must rest on mechanism-level evidence, not a name match. The proportionate defense is the **signal-diff check**: read the cousin's core formula/gate and state what signal it *consumes* (relevance? history? aggregate?) vs the paper component's (utility? current-query? counterfactual?). Granularity: for a recurrence/gate, diff **drive / retention / readout** separately (canonical failure #6). Full PoC remains mandatory only for quality-parity claims; a dismissal needs one read of the cousin's formula. Two blind spots this closes: (a) every other defense (§3.6 triggers, reverse-grep) fires only on PASS — a Gain verdict's secondary dismissal had NO guard; (b) reverse-grep finds *documented* gaps (`TODO|FIXME|limitation`) — this miss class ships **undocumented**, with the blind spot pinned as correct behavior in the cousin's own test (engram zero-query gate=0.5 fusion; → katgpt-rs Issue 656).

**Negative-coverage claims are verdict-bearing too (canonical failure #7 — arXiv:2611.27962 Model Carving, 2026-11-26; owner challenge).** The defenses above guard POSITIVE claims ("already ships", "covered by X"); the INVERSE ("X does NOT ship", "no analog exists") had no guard — Model Carving's Path-0 row wrote "sublayer-skip does NOT ship" while `twt_skip_amputation.rs` (the zero-training skip/ablation arms on the league checkpoint, Bench 022-t54) sat one grep away (`amputate|ablate|skip_layer|block_influence` found it instantly). Rule: a negative coverage claim carries the SAME evidence bar — before writing it, run a 3+-vocabulary CODE grep (paper term + ≥2 codebase synonyms, INCLUDING the harsh action family codebases actually port to: amputate / ablate / excise / elide / drop / cut) across the surface-owning repos' `src/` AND `src/bin/` (ablation/measurement surfaces live in bins — factory table), citing the terms + hit disposition in the note. "I didn't find it" is absence of evidence; "grepped these terms, read these hits" is evidence of absence.

**PoC NOT required when:** pure architectural redirects (no quality claim); training-only genuinely out of scope; latency-only claims (single bench suffices); low-confidence verdicts that explicitly mark quality claim unproven + create `.issues/` follow-up.

**Where the PoC lives:** `riir-ai/crates/riir-poc/` (defend-wrong R&D crate). Three competitors minimum: paper's mechanism (or distilled modelless analog), frozen/no-adaptation baseline, shipped runtime analog. Head-to-head on controlled toy domain, no training, print verdict table. `CARGO_TARGET_DIR=/tmp/...` + clean up.

**PoC defends OR refutes.** If it refutes quality: do NOT silently revise. Record raw numbers in research note §"PoC Addendum". State which axes were confirmed (architectural, latency) vs refuted (quality). Verdict stands on confirmed axes; refuted axis becomes tracked follow-up. PoC stays as permanent regression check.

### 4. Published prior-art search — MANDATORY before any novelty verdict

**Hard gate. Not optional.** Before claiming ANY novelty (Q1), web-search for published prior art on the headline technique. The canonical failure mode: claiming "ternary MoE is novel" when one search for "mixture of ternary experts" would have found published prior art. The miss happens because the agent only read the paper's own references + grepped the codebase, never searching broader literature.

**Mandatory searches (run ALL before any novelty claim):**
1. Headline technique verbatim (`"mixture of ternary experts"`, `"BitNet distillation recipe"`).
2. 2–3 component techniques.
3. The selling-point framing you're about to claim.
4. Recent (2-year) surveys on the topic — they name the competitive landscape.

**If any published paper does what you claim is novel → downgrade Q1 BEFORE writing the verdict; cite explicitly. Do NOT write the verdict then discover prior art later.**

**Parallelize via subagents:** 2–3 `spawn_agent` in parallel (headline search, component search, codebase grep). Web catches published prior art; codebase grep catches shipped prior art. Both mandatory; neither substitutes.

**Re-run after corrections:** discovering prior art in a later pass is NOT complete until you've (a) updated novelty verdict, (b) re-checked surviving novelty claim, (c) committed the correction. Don't leave notes in overclaimed state.

### 4.5. Optional deeper search

If §4 surfaces a dense landscape, use web search to explore specific papers/authors/follow-ups. Not mandatory.

### 5. Final verdict — Claude ping-pong (MANDATORY before any commit)

**A verdict is not DONE until the Claude reviewer has seen it.** The gates above (§1.5 / §1.55 / §1.6) are your self-grade. Before anything commits, negotiate the verdict against the Claude reviewer sub-agent. This is the research-side merge gate and it operationalizes the standing owner rule — *ask Claude for verdict and make decision for any owner gated*. A self-graded verdict that never met the reviewer is the research analogue of an unreviewed merge.

**Availability premise:** the reviewer rides the harness tool `request_verdict` (claude-sub-agent-verdict; round cap `agent.verdict_max_rounds`, default 3), cited to a WORKED RUN (seal-game-editor Proposal 007 — two `#Verdict:` rounds to the cap on the claude backend). No `request_verdict` in the toolset does NOT skip the gate — run the same negotiation through a generic sub-agent reviewer with the same Summary + `#Verdict:` contract; an equal verdict, never provisional. The note records which path ran.

**Protocol** (the tool enforces the mechanics; the Summary is YOUR job):

0. **Paraphrase first (T1 — Issue 977 / EGM "Paraphrase First")** — before the Summary, paraphrase in your own words the pinned claim + every verdict-bearing claim of the note, then re-read the note against it. A contradicting paraphrase is a finding in EITHER direction: fix the note, or quote the passage showing the paraphrase misread it — never a silent accept. The paraphrase rides INTO the Summary, so BOTH verdict paths inherit the same contract; it arms the self-AGREE-drift check (the mismatch becomes CHECKABLE, not felt).
1. **Compose the `## Summary`** — complete, because the reviewer CANNOT see your context: paper ID + full title · the pinned one-sentence novelty claim (§1.5's precondition form) · per-track verdicts with tiers (never pooled) · the Path 0 inventory outcome (analog / extracted / deferred per component) · routing (repo + every file created or edited, exact `{NNN}_{Short_Title}.md` names) · the 2–3 closest cousins considered and why they don't kill the claim · and **the weakest point you know of, named by you** — the reviewer finds it anyway, and naming it first costs one round instead of two. First block: the T0 paraphrase.
2. **Round 1** — call `request_verdict` with the Summary + the note's file paths (so the reviewer can read the note itself) + the literal instruction: "Reply with a verdict that MUST start with `#Verdict: AGREE` or `#Verdict: REVISE` followed by bullet-point reasons."
3. **REVISE** → address EVERY bullet with EVIDENCE — a grep result, a verbatim quote at the pinned sha, a bench number, a file path — then call again with the SAME `session_id` (the negotiation stays in one thread). Rephrasing your position without new evidence does not address a reason.
4. **AGREE (and you agree)** → state your own agreement, restate the final agreed `## Summary`, pass `final_round: true` on that closing call, and only then stage the NAMED files + commit + push. **Cap-round AGREE (2026-11-26, Plato's Key arc): if the AGREE lands ON the cap round (3 of 3), the closing `final_round` call is mechanically refused — expected, not a failure: the AGREE round IS the verdict of record; record the cap refusal in the commit message.**
5. **Round cap (default 3)** → the tool refuses further calls. STOP. Present the remaining disagreement to the USER with both positions — the user is the owner gate, never a tiebreak you award yourself. A capped disagreement escalates; it does not self-resolve by picking your side.

**Scope — mandatory vs optional:**

- **Mandatory:** Super-GOAT and GOAT verdicts · every `riir-train` Plan filing (Path 0.5) · every verdict that creates or rewrites a `.research/` note, `.plans/` file, or architectural guide · every advocate-finding discard (§3.5) · every "already ships"/parity claim granted without the PoC (§3.6 exemptions) · every owner-gated call (feature promotion/demotion, default-on flips, §1.6 tier re-routing).
- **Optional (encouraged):** a plain PASS whose §4 sweep found zero redirect-worthy cousins — a checkable claim, since the searches are in the Summary and auditable next round; a routing call the owner already made in-conversation this session.

**What the reviewer checks — hand it the hooks:** novelty claim vs the §4 searches actually cited (never asserted) · §3.6 signal-diffs on every coverage dismissal · routing vs the MOAT table + fusion priority ladder · **public/private tier check** — a public-repo destination (katgpt-rs, riir-infer, riir-reflex, riir-instinct, katgpt-web) carrying rethink-class content (serving arms, vessel/hosted-only posture, ESC enforcement, economy planes) is a REVISE; default private on any doubt · per-track separation (no cascade from a training-track kill) · discard reasons surviving mechanism-level scrutiny (§3.5) · file hygiene (numbers from `.highwater`, PASS-Redirects lines present, no "candidate" escape-hatch wording) · live-state citations (every open/blocked claim verified this session) · stationary-surface discipline (dynamics signals name their mutated state) · proof-path presence (every GOAT+ filing carries fixture + falsification arm + pre-registered gate).

**Standing failure modes (how this gate dies quietly — pre-registered, not yet measured):**

- **Self-AGREE drift** — summarizing your way to AGREE by omitting the weak axis. The Summary must name the weakest point explicitly; an omission voids the verdict, and a re-opened verdict after files shipped costs a renumber + citation rewrite (the exact cost the numbering rules exist to price).
- **Cap-racing** — burning rounds re-arguing instead of producing evidence. Each REVISE round must ADD something checkable; a round that adds nothing is a round the user now has to arbitrate.
- **Post-AGREE drift** — editing a verdict-bearing section after AGREE without re-running the gate. A material post-AGREE edit re-opens the negotiation on the SAME `session_id` before the next commit; the commit after a `final_round` close is byte-frozen to what was agreed.
- **Reviewer-unavailable** — no `request_verdict` in the toolset? Run the SAME negotiation through a generic sub-agent reviewer (`spawn_agent` in this harness; `Agent`/`SendMessage` in Claude Code — the capability is a fresh reviewer that cannot see your context, whatever the local spelling) with the same Summary + the same `#Verdict:` reply contract. It is an equal verdict, not a weaker one — record which path ran, and never label a fallback verdict provisional (a permanently-provisional gate is an unexecutable mandatory, this file's own most-repeated shape).
- **Reviewer-failed ≠ reviewed (T2 — Issue 977 / EGM: "never count it as a reviewer with zero findings")** — a round returning NO parseable verdict (tool error, empty reply, no `#Verdict:` prefix) is retried ONCE on the same `session_id`; on a second failure the round is recorded **`not run`** with the failure mode named. A not-run round is never an AGREE and never a silent pass: the verdict stays PENDING on the note's Status line, and the file commits only as `wip:` — nothing verdict-bearing ships on a gate that did not run.
- **Reviewer-session expiry (2026-11-26, Model Carving round 3)** — a `verdict reviewer session expired` error mid-negotiation is a lifecycle event, NOT a T2 failed round: restart WITHOUT `session_id` (the tool's own instruction), carrying the complete Summary + every amendment, because the fresh reviewer cannot see the old thread. The restart takes a FRESH cap and the expired session's verdicts are NOT carried — the fresh reviewer re-adjudicates (a post-AGREE amendment MUST be re-adjudicated anyway per the drift rule, so nothing honest is lost). Record both session ids + the expiry in the note's provenance (bc54cea3 expired before its round 3; restarted as f26c955e).

## Constraints (non-negotiable)

1. **Three-track system (not modelless-only)** — (a) **modelless inference** (primary hot path — no weights, BLAKE3-deterministic, zero-alloc), (b) **self-adaptive** (runtime latent updates — `self_evolve`, EMA direction vectors, freeze/thaw — NO base weight mutation), (c) **model-based** (ternary `.bits` via `TernaryDraftModel`, LoRA/QLoRA adapters, Bonsai comparison, quest_grammar pipelines; the quant/kernel substrate ships public in `riir-infer`). Do NOT characterize any repo as "modelless only" without pre-flight #5. Closest to "training" for track (a): freeze/thaw, deterministically-constructed raw/lora hot-swap, latent direction-vector updates. **Before any riir-train deferral, exhaust §3.5 AND check for existing model-based code in the target repo.**
2. **Latent-to-latent preferred** — operate in latent space as long as possible. Decode/project only at boundary. **Sigmoid, never softmax**, for projections onto learned directions. Semantic (emotion/mood/curiosity/style) → latent. Physical (position/HP/wallet) → raw, deterministic, synced.
3. **Freeze/thaw over fine-tuning** — only runtime weight mutation is swapping a frozen snapshot (atomic, versioned, BLAKE3-checked) or applying a deterministically-constructed LoRA overlay (raw/lora hot-swap, no GD). Never mutate weights in-place during inference. Gradient updates (after §3.5) → riir-train.
4. **Self-learn / adaptive CoT welcome** — runtime curiosity, latent prediction, trajectory folding, collapse detection. Update latent state / direction vectors / routing tables, NOT base weights.
5. **7-repo discipline** (product/distillation set; count + membership live ONLY in `katgpt-rs/AGENTS.md` §"Repo count" + `scripts/repo_set.txt` — never copy the number here; read a prose count as a claim, derive the set) — katgpt-rs (public) → riir-ai → riir-chain → riir-neuron-db → riir-train (private) + riir-game-sdk (facade) + riir-dapps (settlement). `riir-refine`/`riir-auth`/`riir-dao`/`riir-viewbridge`/`riir-rethink` carry `.research/` too (rulewright notes → refine; rethink = ESC-depth; instinct's on first use). `riir-infer`, `riir-reflex`, `riir-instinct` are PUBLIC first-class routing targets (§1.6 MOAT rows): inference-substrate / decision-serving / open-stack-grammar respectively. The rest are last-resort surfaces. Training how never leaks to katgpt-rs; chain IP in riir-chain; shard IP in riir-neuron-db; SDK stays facade over `riir-games-shared`; riir-infer + riir-reflex are the sanctioned public substrate/serving tiers, not runtime/chain/shard/train IP.
6. **SOLID, DRY** — per `katgpt-rs/.contexts/optimization.md`. Zero-alloc hot paths. Pre-computed lookup tables. Fixed-size arrays for bounded domains.
7. **Tests/examples** — before/after showing the gain. Latent ops: projection preserves ranking. Freeze/thaw: readers never see torn snapshots.
8. **CPU/GPU/ANE auto-route** — threshold-adaptive. Plasma (µs SIMD) → Hot (sub-ms GPU) → Warm/Cold (ms+ GPU/ANE). L1-fitting latent ops stay SIMD; batched matmul goes GPU.
9. **Plasma → Hot → Warm → Cold → Freeze tiering** — perf on game side (plasma/hot budget), security on chain side (cold/freeze commitment, BLAKE3-hashed, tamper-evident). Latent state crossing sync boundary MUST be raw scalars (valence/arousal/desperation/calm/fear), never the full embedding.

## Latent vs raw space rules (critical for game AI)

Physical → raw, exact, synced; semantic → latent, dot-product + sigmoid; social → KG triples from latent proximity; two-brain spatial cognition. The full rules, including the sync boundary, bridge pattern and KG-triple emission, are in [`latent_raw.md`](latent_raw.md). **Read it before any verdict that touches game AI or sync.**

## Cross-references (read on demand)

Commercial strategy / moat map: the inline short version above suffices for routing decisions. For Super-GOAT novelty gates or "does this become a pillar?" questions, `read_file` `riir-ai/.docs/README.md` (+ `03_pillars/README.md`, `04_supergoat_candidates/README.md`). Exhaustive moat analysis at `riir-ai/.research/003_Commercial_Open_Source_Strategy_Verdict.md` (commercially sensitive — read only when inline short version insufficient).
