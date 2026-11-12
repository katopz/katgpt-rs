# Research 422: Statistically Undetectable Backdoors → Model Provenance Authentication

> **Source:** Andrej Bogdanov, Alon Rosen, Neekon Vafa. *Statistically Undetectable Backdoors in Deep Neural Networks.* arXiv:2607.09532v1 [cs.LG], 10 Jul 2026. <https://arxiv.org/abs/2607.09532>
> **Date:** 2026-07-14
> **Status:** Done (verdict resolved; fusion-PoC follow-up tracked in `.issues/138`)
> **Related Research:** 268 (Forensic Asset Fingerprinting), 230 (Shard Embedding JL projection)
> **Related Plans:** 293 (Forensic Watermark Recipe — assets, not models), 230 (Shard Embedding JL)
> **Classification:** Public

---

## TL;DR

The paper proves that a model trainer can plant a **statistically undetectable backdoor** in any feedforward DNN whose first layer is a frozen compressing Gaussian matrix, by sampling that matrix jointly with a secret `z ∈ {±1}^n` so that `‖Az‖∞` is cryptographically small. Without `z`, finding any near-collision is intractable under LWE-style lattice hardness; with `z`, the holder can compute `x′ = x + z` that collides with any `x` to within `δ₀`. The paper's positive flip (Theorem 1) is a **2-query black-box model-provenance verification**: `V(M, π=z)` accepts iff `‖M(z) − M(0)‖ ≤ δ₀`, and no efficient adversary can forge `π`.

**Distilled for katgpt-rs (modelless, inference-time):** The *verification* `V` is genuinely modelless (two forward passes + an `ℓ₂` compare, no training, no backprop). The *construction* of the backdoored matrix `A` is a **deterministic sampling procedure given `z`** — it is not gradient descent, but it is a **training-time / model-set-up decision** that requires three architectural constraints (frozen Gaussian first layer, bi-Lipschitz composition of remaining layers, discrete bounded inputs) that our shipped transformer + ShardEmbedding stack does not generally satisfy. The honest verdict is therefore **PASS for the modelless repos**: the paper's value is fundamentally training-side, the modelless verification is a corollary that is only useful once the construction has been done, and we have no existing backdoored-model workflow to verify against.

A **fusion idea** (novelty TBD, tracked in `.issues/138`) is called out in §3: the realisation that the Johnson–Lindenstrauss projection we already ship as `ShardEmbedding` (Plan 230) is *exactly* the "frozen compressing Gaussian" substrate the paper requires, so a backdoored JL projection could in principle give us **cryptographic shard-embedding provenance** verifiable in 2 modelless queries. That fusion is NOT committed as Super-GOAT here — it needs a PoC (per skill §3.6) to verify the bi-Lipschitz + discrete-input constraints can be met for the shard-embedding path, and it requires `riir-train` to own the actual first-layer construction.

---

## 1. Paper Core Findings

### 1.1 The construction (§2, §4)

`BackdoorMatrix(1ⁿ, 1ᵐ)`:

1. Sample `z ∼ U({±1}ⁿ)`.
2. For each row `i ∈ [m]`:
   - Sample `bᵢ ∼ N(0, n) |·|≤κ√n` (truncated scalar).
   - Sample `aᵢ ∼ N((bᵢ/n)·z, Iₙ − (1/n)·zzᵀ)` — a Gaussian conditioned on the affine constraint `aᵢᵀz = bᵢ`.
3. Output `(A, z)` with rows `a₁…aₘ`.

Properties (Theorem 2, `m = o(n)`):
- `‖Az‖∞ ≤ O(√n · 2^(−n/m))` — exponentially small in the compression ratio.
- `d_TV(A, N(0,1)^{m×n}) = O(√(mn log(m/n)) + e^(−Ω(m)))` — statistically indistinguishable from honest Gaussian.
- `z` is marginally uniform over `{±1}ⁿ`.

### 1.2 The DNN backdoor (§5)

Substitute the backdoored `A` as the frozen first layer; train the remaining layers normally with any training operator `T` whose output `g` is `(α, β)`-bi-Lipschitz. The composed model `F(x) = g(Ax)` then has:

- **Statistical undetectability** (white-box, full weights): `d_TV(F, bF) = o(1)`. No distinguisher (efficient or not) can tell the backdoored model from an honestly trained one.
- **Backdoor collision generation**: for every `x`, `x′ = x + z` satisfies `‖bF(x′) − bF(x)‖₂ ≤ δ₀ = O(β√m · 2^(−n/m))`.
- **Approximate collision resistance** (Assumption 2, LWE-hard): no p.p.t. adversary can produce any `x₁ ≠ x₂ ∈ [−B:B]ⁿ` with `‖bF(x₂) − bF(x₁)‖₂ ≤ δ₁ = Ω(α/(m^ε √n))`. Backdoor strength `δ₁/δ₀ = Ω(α·2^(n/m) / (β√n · m^(½+ε)))` — exponential in `n/m`.

### 1.3 The positive flip — model-provenance authentication (Theorem 1)

Re-cast the threat model: the backdoor is a **built-in authentication mechanism**. The trainer outputs `(M_B, π=z)`; anyone can verify via:

```
V(M, π):   y₀ ← M(0);  y_z ← M(π);  return 1 if ‖y_z − y₀‖ ≤ δ₀ else 0
```

- `Pr[V(M_B, π) = 1] = 1` — trainer's own proof always accepts.
- `Pr[V(M_B, π′) = 1] ≤ 1/n^ω(1)` — no efficient adversary can forge `π′`, even with full white-box access to `M_B`.
- `d_TV(M_A, M_B) = ÕO(√(m/n))` — the backdoored model is statistically close to the honest one; no input/output behaviour changes.

This is essentially a **digital signature for ML model provenance**: "this model was constructed by the entity holding `z`." The verification is **2 black-box queries**, no cryptographic library needed, post-quantum under LWE hardness.

### 1.4 Architectural constraints (the catch)

All guarantees require **all three** of:

| # | Constraint | Why |
|---|---|---|
| 1 | First layer is a **frozen compressing m×n Gaussian matrix** (m ≪ n) | The backdoor is planted in this matrix; it must stay frozen through training (random-feature learning paradigm, Rahimi–Recht 2007). |
| 2 | Composition of remaining layers is **bi-Lipschitz** with bounded distortion `β_upper` | Ensures any collision must originate in the first layer (so LWE hardness applies); obtained via LeakyReLU activations + bounded condition numbers on subsequent weight matrices. |
| 3 | Inputs are **discrete** integers in a bounded range | Required for the lattice-hardness argument (continuous relaxation breaks Assumption 1/2). |

Without all three, the verification `V` still runs (it is just 2 queries), but the **unforgeability guarantee degrades** — an adversary may be able to find collisions in the remaining layers without defeating LWE.

### 1.5 Proof-of-concept (§6)

Fashion-MNIST embedding model: 784 → 256 (frozen Gaussian) → 512 → 1024 → 2048, LeakyReLU(0.01), semi-orthogonal loss to enforce Constraint 2. Backdoored model retains ≈86.5% accuracy (vs 89% honest) after input rescaling. Embedding-space distances between colliding pairs are orders of magnitude smaller than same-class controls. LLL / Cool / KernelRound all fail to recover the planted `z` at `n = 100, m = 10…30`; backdoor strength ≈ `1/κ ≈ 10⁹`.

---

## 2. Distillation

### 2.1 What is genuinely modelless here

Only the **verification primitive `V`** is modelless in the strict sense (no training, no backprop, no weight mutation). It is two forward passes plus an `ℓ₂` norm compare — a handful of FLOPs, fully runtime.

The **construction** `BackdoorMatrix(z, n, m)` is *deterministic given `z`* (it is a sampling procedure, not gradient descent), but it is a **model-set-up decision**: it replaces the first-layer matrix of a model that is about to be trained. This is squarely a `riir-train` / training-pipeline concern, not a runtime inference operation. It does **not** fit any of the three modelless weight-mutation paths in the global `AGENTS.md`:

- **Freeze/thaw snapshot correction** — no; we are not correcting a biased snapshot, we are *planting a new cryptographic capability*.
- **Raw/lora reader-writer hot-swap** — no; the backdoor is a *replacement* of the first-layer matrix, not an additive overlay, and the goal is *planting a secret*, not *correcting a bias*. The hot-swap path is for closed-form corrections of systematic failures, not for installing cryptographic provenance.
- **Latent-space direction-vector update** — no; this modifies actual base-layer weights, not a latent routing table.

The §3.5 modelless-unblock protocol does not apply because we are not unblocking a gate — there is no failing gate to fix.

### 2.2 Prior-art audit (both-layer grep across all 5 repos)

| Existing shipped mechanism | Security property | Same as paper? |
|---|---|---|
| `forensic_fingerprint_commit.rs` (riir-chain, Plan 394) — commits the `dual_u_snapshot` collusion fingerprint | **Collusion detection** — "did these NPCs coordinate an attack?" | ❌ Different. Detects coordinated behaviour; does not authenticate *who trained the weights*. |
| `katgpt-rs/.plans/293_forensic_watermark_recipe_primitive.md` — Tardos codebook + vertex/texture/topology marks | **Asset watermarking** — "did we create this 3D mesh / texture?" | ❌ Different. Watermarks *external content* (meshes, textures); does not watermark *model weights*. |
| BLAKE3 + `Uuid::now_v7()` provenance (HLA eigenbasis recovery, regime transition `ProvenanceChain`) | **Tamper detection** — "has this blob been modified since commit?" | ❌ Different. Detects post-hoc modification; does not authenticate *who constructed the weights originally*. |
| Ed25519 / Dilithium signatures (`riir-chain` cold path, Research 144) | **Signer authentication** — "did entity X sign this message?" | ❌ Different. Signs a *message/blob*; the signature is *external* to the model. The paper's backdoor is *internal* to the model weights themselves. |
| `neuron_vessel_attestation.rs` (riir-chain, Plan 333 M3.4 + Plan 003 Phase 8) — LatCal committed projection + verifier replay | **Replay-attested computation** — "does the vessel produce this committed output?" | ❌ Different. Attests *deterministic computation*; does not authenticate *who trained the model that produced the output*. |

**No prior art for "cryptographic model-weight provenance via planted backdoor."** This is genuinely novel *as a capability class* for our codebase. The five mechanisms above cover five *different* security properties; none covers "the model weights themselves cryptographically prove who constructed them."

### 2.3 Vocabulary-translation check (per skill §1 step 2)

Paper vocabulary → codebase vocabulary, both grepped:

| Paper term | Codebase analog (grepped) | Hits? |
|---|---|---|
| backdoor | `backdoor` | **0 hits** across all repos |
| provenance / authentication | `provenance`, `authentication`, `attestation` | Many hits, but all are BLAKE3 / Ed25519 / replay-attestation — none model-internal |
| Johnson–Lindenstrauss / compressing Gaussian | `shard_embedding`, `JL projection`, `RandomRotation`, `QJL` | **Hits in katgpt-rs** — `ShardEmbedding` (Plan 230) IS a JL projection `[f32;64]→[f32;8]`. This is the closest shipped substrate. |
| invariance-based adversarial example / collision | `invariance`, `collision` | Many hits, but all in the BLAKE3-collision-resistance / hashmap-collision sense, not the model-output-collision sense |
| number balancing / symmetric perceptron / LWE | `number balancing`, `symmetric perceptron`, `LWE`, `lattice hardness` | **0 hits** — these cryptographic-hardness concepts are not present in the codebase |

The only substantive overlap is `ShardEmbedding` (JL projection). That is the hook for the §3 fusion idea.

### 2.4 Latent-space reframing (per skill §1 step 3, mandatory)

The paper has no useful latent-to-latent reframing for our seven Super-GOAT factory modules:

- **HLA / `latent_functor/` / `cgsp_runtime/`** — operate on per-NPC *runtime* latent state (affect, belief, curiosity). The backdoor operates on a *frozen first-layer weight matrix*; it is a static cryptographic property of the model, not a runtime latent operation. No reframe.
- **`NeuronShard` `style_weights[64]` / `MerkleFrozenEnvelope`** — closest nominal fit: the shard IS a frozen weight blob with a BLAKE3 commitment. But the backdoor requires the shard to BE a compressing Gaussian matrix with a planted `z`; `style_weights[64]` is a K-prior signature, not a Gaussian first layer. The shard's BLAKE3 commitment gives tamper-detection; it does not give trainer-authentication.
- **LatCal fixed-point bridge** — LatCal encodes latent→raw scalar bridges deterministically. The backdoor's `z` is a raw bit-string that could be LatCal-committed, but LatCal adds nothing to the backdoor mechanism itself — it is just a transport for the proof `π`.
- **DEC Stokes operators** — no connection. The backdoor is not a manifold-geometry / divergence / boundary-flux operation.

The honest conclusion: the paper's mechanism does **not** reframe into a latent-to-latent operation on our substrate. It is a weight-matrix cryptographic property, full stop. Per the skill's hard rule, this is a strong signal that the paper is **not a Super-GOAT for our codebase** — the latent reframing fails, and the natural framing (adapter routing / KV compression / speculative decode) does not apply either.

---

## 3. Verdict

**Tier: PASS for the modelless repos (katgpt-rs / riir-ai / riir-chain / riir-neuron-db).**

**One-line reasoning:** The paper's value is fundamentally training-side — the backdoor is planted by *constructing the first-layer matrix* during model set-up, which is a `riir-train` / training-pipeline concern. The modelless verification `V` is a 2-query corollary that is only useful once the construction has been done, and we have no existing backdoored-model workflow to verify against. The construction does not fit any of the three modelless weight-mutation paths (it plants a new cryptographic capability rather than correcting a bias), and §3.5 does not apply because there is no failing gate to unblock.

**MOAT gate per domain (§1.6):** N/A — PASS verdict, no primitive lands.

### 3.1 Fusion idea — novelty TBD, tracked in `.issues/138`

**Not committed as Super-GOAT.** I am not confident enough to write "all 4 YES" on the novelty gate (§1.5):

| Q | Confidence | Reason |
|---|---|---|
| Q1 No prior art? | **90% YES** | Grepped both layers, both vocabularies, all 5 repos. Zero `backdoor` hits. Closest cousins (Plan 293 asset watermark, `dual_u` collusion fingerprint, BLAKE3 tamper detection, Ed25519 signatures, vessel attestation) all cover *different* security properties. |
| Q2 New class of behaviour? | **80% YES** | "Cryptographic model-weight provenance" is a new capability *for our codebase*. Caveat: it is adjacent to existing Ed25519/BLAKE3 — the distinction is *internal-to-weights* vs *external-to-weights*, which is real but not a huge conceptual leap. |
| Q3 Product selling point? | **60% YES** | "On-chain AI assets carry cryptographic proof of training provenance, verifiable in 2 black-box queries, unforgeable under LWE." Real selling point — BUT it requires the architectural constraints (frozen Gaussian first layer, bi-Lipschitz rest, discrete inputs) which our shipped transformer does NOT satisfy. Applicability may be limited to the ShardEmbedding JL path. |
| Q4 Force multiplier? | **55% YES** | Connects katgpt-rs (ShardEmbedding JL, verification `V`) + riir-chain (commit `π` via LatCal) + riir-neuron-db (store backdoored shard). BUT the connection to `riir-train` (construct `A`, enforce constraints) is a *dependency*, not a force multiplier in the modelless repos. |

Average confidence ≈ 71%. Below the commit threshold. Per skill §1.5: write "fusion idea — novelty TBD, needs Q1–Q4 check before verdict" and create an issue.

**The fusion idea itself:** Our `ShardEmbedding` (Plan 230) IS a Johnson–Lindenstrauss random orthogonal projection `[f32;64]→[f32;8]` — the *exact* "frozen compressing Gaussian matrix" substrate the paper backdoors. If we replaced the honest JL projection with a backdoored one (constructed deterministically given a secret `z`), the shard-embedding retrieval would still work (statistically close to honest, `d_TV = o(1)`), but we would gain a **cryptographic shard-embedding provenance** primitive: anyone can verify "this shard was embedded by the entity holding `z`" in 2 modelless queries.

The open questions a PoC must settle (per skill §3.6 — architectural coverage ≠ quality parity):

1. **Constraint 1 (frozen Gaussian first layer):** Does our ShardEmbedding satisfy this? The honest matrix is Gram-Schmidt-orthogonalised Gaussian — yes, frozen, yes, compressing (64→8). The backdoored variant would need to preserve the JL guarantee *and* plant `z`. The paper's construction preserves `d_TV = o(1)`, so retrieval quality should survive — **needs measurement**.
2. **Constraint 2 (bi-Lipschitz downstream):** The shard-embedding downstream is a cosine-similarity retrieval (not a deep network). Cosine similarity is 1-Lipschitz. **Probably satisfied trivially** — but needs verification that no non-bi-Lipschitz op sits between the projection and the retrieval.
3. **Constraint 3 (discrete bounded inputs):** Shard inputs are `style_weights[64]` — *continuous* f32, not discrete integers. **This is the hardest constraint.** The paper's hardness argument (Assumption 1/2) is over `ℤⁿ`; continuous inputs may break the LWE reduction. **Needs a formal check** — possibly a rounding/quantisation pre-step on `style_weights` to restore discreteness, with a measurement of the retrieval-quality cost.
4. **Does verification `V` actually work on our stack?** Trivially yes (2 queries) — but the *unforgeability* guarantee depends on #1–#3.

If the PoC passes all four, the fusion becomes a **GOAT** (provable security gain over BLAKE3-only tamper detection, new capability for our codebase) with Super-GOAT potential if it also multiplies ≥2 pillars. If any of #1–#3 fail, the fusion is shelved.

**Where the PoC would live:** `riir-ai/crates/riir-poc/benches/jl_backdoor_provenance_modelless_goat.rs` — three competitors (backdoored JL with `z`, honest JL baseline, attempted forgery without `z`), head-to-head on a controlled toy shard-retrieval domain. Use `CARGO_TARGET_DIR=/tmp/jl_backdoor_poc` per AGENTS.md.

### 3.2 Why this is not riir-train territory *either* (yet)

The construction `BackdoorMatrix(z, n, m)` is deterministic given `z` — it is a sampling procedure, not gradient descent. In principle it could be implemented modellessly. But:

- It is a **model-set-up decision** (replaces the first-layer matrix before training the rest).
- The remaining layers DO need to be trained (riir-train), under the bi-Lipschitz constraint (semi-orthogonal loss, per §6.1 of the paper).
- The discrete-input constraint may require a quantisation pre-step that has its own training implications.

So the full value chain is: `riir-train` (construct `A`, train remaining layers under constraints) → `riir-neuron-db` (freeze backdoored shard) → `riir-chain` (commit `π=z` via LatCal) → `katgpt-rs` (verification `V` as a public primitive, IF the PoC passes). This workflow does not exist today and is not started by this research note — it is the subject of `.issues/138`.

---

## 4. Latent vs raw boundary (per AGENTS.md)

N/A — PASS verdict, no primitive lands. For reference if the §3.1 fusion PoC ever advances:

| Quantity | Domain | Sync? | Commitment |
|---|---|---|---|
| Secret `z ∈ {±1}ⁿ` (the proof `π`) | **Raw** (bit-string) | YES (commit via LatCal) | BLAKE3 of `z` on-chain; full `z` only held by trainer |
| Backdoored first-layer matrix `A` | **Raw** (weights) | YES (part of frozen shard) | `MerkleFrozenEnvelope` BLAKE3 |
| Verification query pair `(M(0), M(z))` | **Raw** (model outputs) | NO (runtime-only, 2 queries) | None — recomputed by verifier |
| Collision bound `δ₀` | **Raw** (scalar threshold) | YES (public parameter) | Part of the published scheme spec |

No latent-space quantity is involved. The backdoor is a raw-weight cryptographic property; the verification is a raw-output comparison. This is consistent with the sync-boundary rule (raw crosses sync, latent stays local).

---

## 5. Cross-references

- **`katgpt-rs/.plans/230_shard_embedding_projection.md`** — the JL projection we already ship (`[f32;64]→[f32;8]`, Gram-Schmidt orthogonal rows, SIMD dot-product, BLAKE3 commitment). The §3.1 fusion would replace the honest matrix here with a backdoored one.
- **`katgpt-rs/.research/268_Forensic_Asset_Fingerprinting_LatCal_Recipe.md`** + **`katgpt-rs/.plans/293_forensic_watermark_recipe_primitive.md`** — existing forensic watermarking for *assets* (meshes/textures). Different security property (asset attribution, not model-weight provenance), but same *pattern* (commit a fingerprint, verify later). The pattern transfers; the mechanism does not.
- **`riir-chain/src/forensic_fingerprint_commit.rs`** — commits the `dual_u` collusion fingerprint. Different property (collusion detection, not trainer authentication). Same receipt shape (BLAKE3 + Merkle) — if the §3.1 fusion advances, the commit bridge would mirror this module structurally.
- **`riir-ai/.research/144_PQC_Dilithium_Cold_Path_Optional.md`** — PQC signatures on the cold path. Different property (message signing, not model-internal authentication). The "deferred GOAT gate" pattern (G1–G5) is a good template for `.issues/138`'s PoC gate.
- **Boundary note (2026-11-12):** the reference-free *detection* question this note's Issue-152 follow-up left impractical has an answer for the sparse-edit class — Lazarus (Zeng et al., Microsoft, Nov 2026), distilled at `../riir-infer/.research/025_Lazarus_Dormant_Neuron_Weight_Backdoor_Detection.md`. It reads the layer's own population as the reference (no delta needed) and does NOT contradict R422: distribution-level undetectability stands; the detectable class is the footprint-leaving one, bounded below by Lazarus's Theorem 4 (norm-product lower bound on independent multi-token writes).

---

## TL;DR

**Verdict: PASS.** The paper proves a powerful training-time result (statistically undetectable backdoors in DNNs with frozen Gaussian first layers) with a modelless verification corollary (2-query provenance check). The construction is a training/set-up operation that does not fit any modelless weight-mutation path; the verification is only useful once the construction has been done; and we have no existing backdoored-model workflow to verify against. The latent-space reframing fails (the mechanism is a raw-weight cryptographic property, not a latent operation), which per skill rules is a strong signal against Super-GOAT. **One fusion idea** (backdoored ShardEmbedding JL projection → cryptographic shard-embedding provenance) is noted as novelty-TBD and tracked in `.issues/138` — it needs a §3.6 defend-wrong PoC to verify the three architectural constraints (frozen Gaussian, bi-Lipschitz downstream, discrete inputs) can be met for our shard-embedding path before any verdict upgrade.

---

## 6. PoC Addendum (2026-07-14) — Fusion SHELVED on G1; primitive sound

The defend-wrong PoC (`.issues/138` T1–T9) ran at `riir-ai/crates/riir-poc/benches/jl_backdoor_poc_gates.rs` (`CARGO_TARGET_DIR=/tmp/jl_backdoor_poc`). **Raw results:**

| Gate | m=8 | m=16 | m=32 | Threshold | Verdict |
|---|---|---|---|---|---|
| **G1 parity** (backdoor top-5 ≈ honest top-5, overlap ≥ 0.8) | **0.000** | **0.000** | **0.000** | ≥ 0.90 | ❌ **FAIL** |
| G1 honest absolute quality (top-5 vs ground truth) | 0.131 | 0.194 | 0.250 | (informational) | confirms Plan 230's 6% finding — honest JL is itself broken at these dims |
| **G2 verification correctness** (`V(A,z)=1`, `V(A,z′)=0`) | 1.000 / 1.000 | — | — | 1.000 | ✅ PASS |
| **G3 unforgeability** (budget=4096 matmuls, 100 trials × 3 adversaries) | random=0, greedy=0, pairwise=0 | — | — | ≤ 2⁻²⁰ | ✅ PASS (at this budget) |
| **G4 latency** (single `V` call) | 272 ns | — | — | ≤ 1 µs | ✅ PASS |

### 6.1 What the PoC defends

- **The provenance primitive itself is sound.** G2 + G3 + G4 all pass. The backdoor construction (`A′ = A − (Az)zᵀ/‖z‖² + σ·g⊗zᵀ/‖z‖²`) produces a matrix where `‖A′z‖ ≈ σ√m ≈ 2.3e-3` (verified), `V(A′, z)` accepts the true `z` and rejects decoys 100% of the time, and three gradient-free adversaries (random / greedy bit-flip / pairwise-flip lattice-style) found **zero forgeries** in 100 trials × 4096-matmul budget each.
- **C3 (discrete inputs) was a red herring for the provenance use case.** `V(A, z)` never takes `style_weights` — only the secret `z ∈ {±1}ⁿ` (always discrete by construction). The discrete-input constraint in the paper applies to the *colliding-input* threat model, not to forging the secret. The load-bearing gate was G3, and G3 holds.

### 6.2 What the PoC refutes (the defend-wrong payoff)

- **G1 retrieval parity fails catastrophically (0.000 at every tested dim).** The backdoored matrix and the honest matrix produce **completely disjoint** top-5 nearest-neighbour sets on 128 toy shards. This is *worse* than the honest baseline alone.
- **Root cause:** the backdoor construction projects out the `z` direction (`A₀ = A − (Az)zᵀ/‖z‖²`), removing one degree of freedom from an already aggressively-compressed space (64→8). The honest JL baseline itself barely preserves NN structure at these dims (~13–25% top-5 overlap with ground truth, confirming Plan 230's "64→8 too aggressive" verdict). Adding the projection-out step destroys the remainder.
- **The fusion as stated in §3.1 — "replace the honest JL projection with a backdoored one, retrieval still works" — is FALSE.** The paper's `d_TV = o(1)` guarantee holds for the matrix *distribution*, but retrieval quality depends on the *geometry* of the specific matrix instance, and projecting out a direction in a low-dim space is geometrically destructive.

### 6.3 Salvage path (NOT pursued; documented for completeness)

The provenance primitive (G2/G3/G4) is sound. The failure is specifically that **one matrix cannot serve both retrieval and provenance** at these dims. A *two-matrix* architecture — honest JL for retrieval, separate backdoored matrix for provenance — would preserve both properties. This is a different design than the §3.1 fusion ("replace") and would double the embedding storage. **Not pursued** because:

1. The honest baseline itself is broken at 64→8 (Plan 230's unresolved G1 failure). Fixing retrieval is a prerequisite; adding provenance on top is premature.
2. The paper's `d_TV = o(1)` construction is LWE-based and lives in `riir-train` per §3.2; our PoC used a simplified projection-based construction that achieves `‖Az‖` smallness but not the full statistical-closeness guarantee. A salvage would need the full LWE construction first.
3. The product-fit case (§3.1 Q3, 60% confidence) does not improve after this PoC — the architectural constraints remain.

### 6.4 Routing per Issue 138 outcome table

> G1 FAILS (retrieval parity) → **Fusion shelved** — backdoor destroys shard retrieval. No primitive; close this issue with PoC Addendum in Research 422.

**Verdict remains PASS.** No primitive lands in any of the 5 repos. Issue 138 is closed with this addendum as the negative result. The `jl_backdoor_poc` module stays in `riir-poc` as a permanent negative control — future re-exploration (e.g. after Plan 230's retrieval is fixed at higher dim, or if a two-matrix architecture becomes product-justified) can resurrect it.

### 6.5 Files

- `riir-ai/crates/riir-poc/src/jl_backdoor_poc.rs` — the PoC module (BackdoorMatrix, V, 3 adversaries, G1–G4 measurements, 12 unit tests).
- `riir-ai/crates/riir-poc/benches/jl_backdoor_poc_gates.rs` — the gate runner.
- Run: `CARGO_TARGET_DIR=/tmp/jl_backdoor_poc cargo bench -p riir-poc --bench jl_backdoor_poc_gates -- --nocapture`
