# Research 268: Forensic Asset Fingerprinting via Key-Derived LatCal Perturbation Recipes

> **Source:** Fusion of internal primitives (Plan 223 SIMD LatCal + Research 139 NFT + Plan 272 chunks + Plan 319 WASM vessel + Doc 57 egg/shell). Industry prior art: Tardos codes (Tardos 2008), Boneh–Shaw (1998), AACS / PlayReady / Widevine forensic watermarking.
> **Related paper (tangential):** [arxiv 2606.18208](https://arxiv.org/pdf/2606.18208) — LoopWM spectral stability `A = diag(−exp(a))` (bounds cumulative perturbation visual impact across chunks).
> **Date:** 2026-06-19
> **Status:** Active
> **Related Research:** 139 (riir-ai NFT guide), 262 (chunk store), 257 (FuncAttn — closed basis analog for latent operators), 123 (riir-ai latent functor), 196 (KG-Latent-Octree — asset graph)
> **Related Plans:** TBD (open primitive → katgpt-rs Plan 293; private integration → riir-ai Plan 322)
> **Cross-ref (riir-ai):** Research 140 (Asset Fingerprinting × LatCal × WASM × NFT — Private Runtime Guide)
> **Classification:** Public

---

## TL;DR

Forensic watermarking / traitor tracing is a 30-year-old industry technique (Hollywood AACS, Microsoft PlayReady, Google Widevine) that embeds per-recipient marks in content so leaks are attributable to source. **The transferable primitive:** a per-client perturbation *recipe* — derived from the client's pubkey via BLAKE3 — that is small (~64 bytes), applied client-side at decode time, recoverable from any leaked frame, and verifiable with the existing SIMD 4-wide determinant pipeline (`batch_validate_eggs`, Plan 223).

The novel fusion (not the math itself): the **anti-tamper egg/shell check `det(E₁ × E₂) = secret` IS already a fingerprint** when `E₂` is per-client-derived. Anti-tamper and attribution collapse into one mechanism. One ingredient, two questions answered. The recipe rides the existing WASM asset vessel (Plan 319), so we get forensic traceability as a near-free side effect of the existing security stack rather than as a parallel subsystem.

**Distilled for katgpt-rs (modelless, inference-time):**
- Recipe derivation: `pubkey → BLAKE3 → seed → 2×2 LatCal perturbation matrix P_client` (determinant-preserving for stability — see §4 LoopWM connection).
- Recipe application: vertex sub-mm displacement, mid-frequency DCT texture marks, topology micro-edges. All inference-time, no training, no gradient.
- Recipe recovery: extract from leaked frame → BLAKE3 inverse-lookup in NFT registry (Research 139).
- Recipe verification: SIMD `det(P_client · shell) = expected` reuses Plan 223 batch path verbatim.

**Verdict: GOAT.** Same logic as Research 257 (FuncAttn) — math is industry-known, fusion is novel, extends ≥5 existing pillars, clear selling point. Not Super-GOAT because post-leak attribution is an existing capability class (just novel for game assets + novel in unifying with anti-tamper). Plan + implement behind feature flag, with GOAT gate on attribution accuracy + collusion resistance.

---

## 1. Industry Prior Art (What's Known — Do Not Reinvent)

| System | Technique | What we adopt | What we don't |
|---|---|---|---|
| **Tardos codes (2008)** | Probabilistic fingerprinting code, c-collusion resistant with length `O(c² log(n/ε))` | Anti-collusion codebook structure (per-client binary codeword → recipe bits) | Their continuous distribution; we use BLAKE3-seeded discrete bits |
| **Boneh–Shaw (1998)** | Static fingerprinting, marking assumption | The "marking assumption" — attacker cannot change more than m positions per copy | Their asymptotic bounds (we have small user populations, ε loose) |
| **AACS / Blu-ray watermarking** | Per-disc video watermark in DCT coefficients | Mid-frequency DCT embedding for textures | Their dedicated detector ASIC; we run detection server-side on leaked frames |
| **PlayReady / Widevine L1** | Secure decoder enclave + forensic mark | Architecture: recipe applied inside trusted execution; output is fingerprinted raw bytes | Their proprietary crypto; we use BLAKE3 + Ed25519 (existing stack) |
| **Image steganography (DCT, LSB, F5)** | Embedding payload in image coefficients | Mid-frequency DCT coefficient sign flipping for texture payload | LSB embedding (destroyed by BC7/JPEG); spread-spectrum alternatives |

**What we are NOT inventing:** forensic watermarking itself, anti-collusion codes, steganography, DRM. These are 20–40 year old fields with deep literature. Our contribution is the **fusion** with our existing latent-state security stack (LatCal egg/shell, NFT registry, chunked Merkle, WASM vessel, chain slashing) — see §2.

---

## 2. The Fusion (What's Novel)

### 2.1 The key insight: anti-tamper + attribution are the SAME mechanism

Doc 57 Layer 1 ships the egg/shell anti-tamper:

```
det(E₁ × E₂) == derived_secret  →  valid transaction
det(E₁' × E₂) ≠ secret          →  tampered, rejected
```

Currently `E₂` is derived from a single server-held `combined_seed`. **The trivial upgrade:** derive `E₂` per-client from `BLAKE3(combined_seed ‖ client_pubkey_i)`. Then:

- Anti-tamper still works: each client can only produce eggs whose determinant matches *their* `E₂_i`.
- Attribution emerges for free: a leaked egg `E₁_leak` deterministically identifies which `E₂_i` produced it (determinant invariant + per-client seed).
- The same SIMD batch pipeline (`batch_validate_eggs`, Plan 223) verifies and attributes simultaneously.

One mechanism. Two questions. Zero extra hot-path cost.

### 2.2 From transaction eggs to asset recipes

The egg/shell pattern is currently applied to LatCalMatrix transactions (wallet, NPC emotion deltas). Extending to assets:

- The "asset" is a chunked blob (Plan 272) — chunks are BLAKE3-hashed, Merkle-rooted.
- Per-client, server derives a small **recipe** `(P_vertex, P_texture, P_topology)` of perturbation parameters from the client pubkey.
- Recipe bits map to a Tardos-style anti-collusion codebook: each client gets a unique codeword over the chunk graph.
- The WASM asset vessel (Plan 319) applies the recipe at decode time, producing a per-client-modified raw vertex/texture buffer that is written to `FfiRenderState` shared memory.
- The same recipe drives the egg/shell check that gates the asset serving.

### 2.3 Bandwidth economics

Naive forensic watermarking ships N unique content copies (one per client). Our recipe-based approach ships:

- ONE shared encrypted asset (chunked, dedup'd, Merkle-committed via Plan 272).
- N small recipes (~64 bytes each: 2×2 P_vertex + DCT coefficient indices + topology bit mask).
- Recipes are delivered inside the existing per-session key exchange (`α + β` split-key, Doc 57 Layer 3). No extra round trip.

For 10⁵ concurrent clients with a 50 MB LOD-0 asset: naive = 5 TB unique content; recipe-based = 50 MB shared + 6.4 MB of recipes. **78000× bandwidth reduction** vs naive per-client content.

---

## 3. The Math

### 3.1 Recipe derivation

```
seed_i = BLAKE3(combined_seed ‖ client_pubkey_i ‖ "asset_fingerprint_v1")
bits_i = seed_i[0..L]                          // L = codebook length
P_vertex_i = perturbation_matrix(seed_i, L_v)   // 2×2 det=1, eigenvalues ∈ (0,1)
dct_indices_i = codebook_select(bits_i)         // L_dct mid-frequency DCT positions
topology_mask_i = bits_i[L_v + L_dct ..]         // per-triangle bit
```

`P_vertex_i` is a **2×2 determinant-1 matrix** with eigenvalues in (0,1) — this is the LoopWM spectral stability constraint (§4) applied to the perturbation: it guarantees cumulative visual displacement across the chunk graph stays bounded regardless of how many vertices the recipe touches.

### 3.2 Recipe application (client-side, in WASM vessel)

```text
// Vertex perturbation (sub-millimeter, perceptually invisible)
for each marked vertex v_k:
    v_k' = (I + ε · P_vertex_i) · v_k          // ε ≈ 1e-4 m

// Texture DCT mark (mid-frequency, BC7-robust)
for each (block_idx, coef_idx) in dct_indices_i:
    sign = bits_i[block_idx × 8 + coef_idx]
    DCT[block_idx][coef_idx] += sign · δ        // δ ≈ 2 (just above BC7 noise floor)

// Topology watermark (degenerate triangles, invisible at render)
for each marked triangle t_j where topology_mask_i[j] == 1:
    insert zero-area leaf triangle adjacent to t_j
```

### 3.3 Recipe recovery (post-leak attribution)

```text
// From a leaked frame:
1. Extract mesh from frame (point cloud / vertex dump)
2. Recover P_vertex by least-squares fit against reference asset:
      P̂ = argmin ‖V_leak - (I + ε P) · V_ref‖_F
3. Read DCT marks from texture samples
4. Read topology mask from mesh analysis
5. Concatenate → codeword_i
6. BLAKE3-inverse-lookup codeword_i against NFT registry (Research 139)
7. Identified client_pubkey_i → NFT owner → chain slashing
```

### 3.4 Anti-collusion bound (Tardos)

For n users and c colludators, codeword length `L = O(c² · log(n/ε))` gives error probability < ε. For c=10 colludators, n=10⁵ users, ε=10⁻⁶: `L ≈ 1000` bits. The recipe encodes 1000 bits across vertex/DCT/topology channels — feasible (50 verts × 8 bits/vertex + 50 DCT coefs + 100 topology marks).

---

## 4. LoopWM Spectral Stability Connection (arxiv 2606.18208)

LoopWM (Looped World Models) parameterizes its state-retention matrix as `A = diag(−exp(a))`, then discretizes via zero-order hold `Ā = exp(Δ · A)`, guaranteeing all eigenvalues of `Ā` lie in `(0, 1)`. This bounds residual dynamics across arbitrary rollout lengths.

**Transferable to fingerprinting:** the per-client vertex perturbation matrix `P_vertex_i` should satisfy the same constraint — eigenvalues in `(0, 1)`. This ensures:

1. **Bounded cumulative displacement.** As the recipe is applied across the chunk graph (effectively a long "rollout" over chunks), the displacement cannot diverge.
2. **Perceptual invisibility.** Contraction toward zero means each marked vertex moves *less* than ε from its reference — well below the visual threshold.
3. **Numerical stability under recompression.** BC7/JPEG quantization preserves small-eigenvalue perturbations better than large ones.

LoopWM is primarily a training paper (latent dynamics world models), but this single inference-time construct (`A = diag(−exp(a))` → contractive `Ā`) is a clean transferable primitive. **No training required for our use case** — we use the parameterization at recipe derivation time to construct `P_vertex_i` with provably small spectral radius.

Other LoopWM insights (deferred decoding, sigmoid adaptive exit) are already shipped under different names in our stack (`evolve_hla` recurrent belief kernel, AGENTS.md sigmoid rule). They validate existing patterns; they are not new contributions here.

---

## 5. Verdict: GOAT

| Question | Answer | Notes |
|---|---|---|
| Q1 No prior art? | **Partial.** Industry prior art is significant (Tardos, Boneh–Shaw, AACS, Widevine). Our novelty is the **fusion** (LatCal egg/shell + NFT + WASM vessel + chain slashing + key-derived recipes), not the math. Same situation as Research 257 (FuncAttn). | Vocabulary translation: paper "traitor tracing" ↔ codebase "egg/shell per-client shell matrix"; paper "fingerprinting code" ↔ codebase "per-client recipe bits". |
| Q2 New class of behavior? | **For game industry: yes.** No shipped MMO does runtime asset fingerprinting with on-chain attribution + slashing. **For DRM/security industry: no.** Post-leak attribution is well-known. | The "new class" claim is domain-specific, not absolute. Same as Research 257 — the math pieces are known. |
| Q3 Product selling point? | **Yes.** "Leaked game assets are mathematically traceable to the source NFT owner within hours; attribution triggers on-chain slashing." No competitor has this. | Selling point for the riir-ai guide (140). |
| Q4 Force multiplier? | **Yes.** Connects Plan 223 (SIMD) + Plan 272 (chunks) + Plan 306 (WASM-FFI bridge) + Plan 319 (asset vessel) + Research 139 (NFT) + Doc 57 (egg/shell) + chain slashing. ≥5 pillars. | |

**Not Super-GOAT** because: (a) forensic watermarking is an existing capability class with 30 years of literature; (b) the egg/shell determinant is already shipped — we are extending it, not creating a new primitive; (c) the primary value is in the private integration (riir-ai 140), not in a new open math primitive. Same reasoning as Research 257 (FuncAttn).

**Why GOAT and not Gain:** the fusion has a concrete, measurable, in-domain gain (attribution accuracy > 0.99 against single-leak, collusion resistance up to c=10 colludators), backed by a clear GOAT gate (§7). The product selling point is real (no MMO competitor has this), even if the technique itself isn't novel in absolute terms.

**One-line verdict reasoning:** Forensic asset fingerprinting unified with the existing egg/shell anti-tamper mechanism — per-client BLAKE3-derived perturbation recipes applied client-side in the WASM vessel, attributed via NFT registry, slashed via chain consensus. Math is industry-known (Tardos / Widevine); fusion is novel (≥5 pillars, clear selling point); GOAT-tier behind feature flag with GOAT gate on attribution accuracy and collusion resistance.

### Routing

- **riir-ai/.research/140_Asset_Fingerprinting_LatCal_WASM_NFT_Guide.md** — private runtime guide. The selling-point doc. Contains the WASM vessel integration, NFT binding, chain slashing flow, validation protocol.
- **riir-ai/.plans/322_asset_fingerprinting_wasm_recipe.md** (sketch) — private implementation plan. Extends Plan 319 (executable asset vessel) with recipe application + Plan 223 (SIMD determinant) with per-client `E₂` derivation + Research 139 (NFT) with attribution registry + chain slashing hook.
- **katgpt-rs/.plans/293_forensic_watermark_recipe_primitive.md** (sketch, optional) — open primitive. Generic BLAKE3-seeded recipe derivation, Tardos codebook, DCT/topology embedding utilities. **NO game semantics, no chain, no NFT.** This is the "adoption hook" for engineers who want generic forensic watermarking in their modelless pipeline.
- **No Super-GOAT guide required** (verdict is GOAT, not Super-GOAT).

---

## 6. Closest Cousins (3)

1. **Plan 223 / Doc 43 — Latent Batch Matrix SimRing + Egg/Shell Read-Flow.** The closest *math* cousin. SIMD 4-wide determinant pipeline already ships `batch_validate_eggs`. Extending it to per-client `E₂` derivation is the smallest possible patch — same loop, per-client shell. The "Egg/Shell Read-Flow Architecture" (Doc 43 §T8–T12) already planned WASM read-flow for game state; this extends it to asset bytes.

2. **Research 139 (riir-ai) — Lore Distillation → Executable Asset Vessel × Quorum Gitflow.** The closest *runtime* cousin. The NFT binding (Research 139 §2.5: `MintAssetNft { asset_blob_id, owner }`) is exactly the attribution registry we need. `asset_blob_id` is the Merkle root; `owner` is the pubkey; per-client recipe is derived from this pair.

3. **Research 262 — Lore-Style Chunked Content Store.** The closest *granularity* cousin. Per-chunk watermarking needs the chunk graph as the embedding substrate — different chunks host different recipe bits, so a partial leak still recovers enough codeword for attribution. `ChunkFetcher` (Plan 272) is the lazy hydration path; recipe bits ride on chunk boundaries.

(Bonus: Research 257 FuncAttn — analog for *latent operator* perspective. Per-client `E₂` is to anti-tamper what per-domain `C` operator is to FuncAttn: a small structured transform applied to a low-dim space. Same architectural pattern.)

---

## 7. Latent vs Raw Boundary

Per AGENTS.md raw-vs-latent rules and Research 262 §5:

| Layer | Domain | Treatment |
|---|---|---|
| Recipe `(P_vertex, dct_indices, topology_mask)` | **Latent** (semantic) | Derived from client pubkey hash; small (~64 bytes); delivered via split-key α+β (Doc 57 Layer 3). Never synced directly — only its *effects* (perturbations) appear in raw asset. |
| Encrypted asset chunks (BLAKE3-Merkle) | **Raw** (encrypted bytes) | Bit-identical across all clients; chunk Merkle root is canonical identity; sync is bit-identical for replay compatibility. |
| Perturbed vertex/texture output | **Raw** (physical domain) | What gets written to `FfiRenderState` shared mem. Per-client distinct but visually indistinguishable. This is the only crossing point. |
| Attribution registry `(pubkey → recipe)` | **Latent** | Lives in chain Cold tier; queryable via NFT program (Research 139); not synced as raw. |
| Slashing event | **Raw** | Quorum-committed `TxDelta`; BLAKE3-hashed; replayable. |

**Bridge rule:** Recipe is latent (BLAKE3-seeded compact parameters). Perturbed asset is raw (physical domain, must be raw for GPU). The bridge is the WASM vessel's decode-time application: latent recipe → raw perturbation. The bridge is one-way (recipe → bytes), zero-allocation, gateable by feature flag. No raw→latent inversion in the hot path; raw→latent inversion happens only at forensic recovery time (offline, server-side).

---

## 8. Constraints Check

| Constraint | Status |
|---|---|
| Modelless / inference-time | ✅ Recipe derivation + application are pure functions of (pubkey, asset). No training, no backprop. Tardos codebook is pre-computed. |
| Latent-to-latent preferred | ⚠️ Partial. Recipe is latent-to-raw (perturbs raw vertices). Acceptable because the *output* is a physical-domain asset, by design. Recovery (raw→latent codeword) is offline forensic, not hot-path. |
| Use sigmoid not softmax | ✅ Codebook selection is BLAKE3-seeded deterministic; no softmax anywhere. Attribution confidence is a sigmoid-gated threshold (`σ(log P(attribution / null))`) to match AGENTS.md rule. |
| Freeze/thaw over fine-tuning | ✅ Recipes are derived, not trained. Per-session. Revoked on chain slashing (frozen NFT). |
| 3-repo discipline | ✅ Open math (this note) → katgpt-rs. Private runtime (WASM vessel + NFT + slashing) → riir-ai 140. No training know-how here. |
| Raw scalars at sync boundary | ✅ Slashing events cross sync as raw `TxDelta`. Recipe never crosses sync — only its byte-level effects do. |
| Zero-alloc hot path | ✅ Recipe application runs inside WASM vessel with pre-allocated vertex scratch buffer; one matmul per marked vertex; SIMD-batched where possible. |
| BLAKE3 not SHA | ✅ Recipe seed uses BLAKE3-XOF (matches Doc 57 Layer 1). |
| Ed25519 for keys | ✅ Reuses `gm_key_store` Ed25519 pubkeys (riir-chaind, already shipped). |

---

## 9. Open Questions / Risks

1. **Collusion attacks.** Two colludors compare their copies, find differing bits, erase them. **Mitigation:** Tardos codebook (c-collusion safe for c up to design bound). **Risk:** c beyond design bound — need to size codeword for the expected colluder population. Validation: G2 in riir-ai 140.

2. **Compression robustness.** BC7/JPEG texture compression, mesh simplification, re-meshing tools can erase watermarks. **Mitigation:** embed in mid-frequency DCT coefficients (more robust than LSB); embed topology marks in curvature invariants (survive simplification). **Risk:** aggressive re-encoding by sophisticated attacker. Validation: G3 in riir-ai 140.

3. **False positives.** A noisy extraction could mis-attribute, accusing innocent users. **Mitigation:** Reed-Solomon outer code + CRC + 99.99% confidence threshold; legal-grade bar for actual slashing. **Risk:** the deterrent value depends on the system being trusted to not false-accuse. Validation: G5 in riir-ai 140.

4. **GPU VRAM dump.** Once vertices are in VRAM, capture tools can dump them. The watermark survives the dump (it's in the geometry), so attribution still works — but the attacker can extract the clean reference asset if they have *both* a marked copy and the unmarked encrypted bytes. **Mitigation:** the unmarked bytes are AES-encrypted-at-rest; only the WASM vessel has the session key. If they compromise the vessel, they have the marked copy. **This is the inherent DRM limit — accepted.**

5. **WASM vessel extraction.** If attacker reverse-engineers the vessel and extracts the recipe application logic, they can produce unmarked copies. **Mitigation:** vessel is per-session keyed (Doc 57 Layer 3 split-key); session key is server-derived from α + β, never shipped whole. Vessel runs in anti-debug sandbox. **Accepted limit:** this is the same trust boundary Widevine L1 / FairPlay operate under.

6. **Detection latency.** Forensic watermarking is post-leak, not preventive. **Accepted** — the value is deterrence + attribution, not prevention. The "lure to jail" framing the user proposed is the standard DRM honeypot pattern.

7. **Spectral stability of `P_vertex` across long chunk graphs.** LoopWM's `A = diag(−exp(a))` guarantees contraction, but does contraction survive recompression? **Mitigation:** eigenvalue bound + ε small enough that perturbations are below BC7 noise floor. Validation: G4 in riir-ai 140.

8. **Recipe bandwidth.** ~64 bytes per client × 10⁵ clients = 6.4 MB. Negligible vs the 50 MB shared asset. ✅ No risk.

---

## 10. Connection Map (Force Multiplier)

```
                         ┌─ Plan 223 (SIMD LatCal batch det) ─┐
                         │                                    │
   Plan 319 (WASM        │  per-client E₂ derivation         │
   asset vessel) ────────┼─► applies recipe at decode time ───┤
   applies recipe in     │                                    │
   trusted shell         │  Plan 272 (chunked Merkle store)   │
                         │  recipe bits ride chunk boundaries │
                         └────────────────────────────────────┘
                                       │
                                       ▼
                       raw vertex/texture written to FfiRenderState
                                       │
                                       ▼
                       Unity renders — fingerprint INVISIBLE
                                       │
                                       ▼
                       (if leaked) forensic recovery:
                                       │
                                       ▼
          ┌─ Research 139 (NFT registry: pubkey ↔ asset_blob_id) ─┐
          │                                                        │
          │  chain slashing (existing penalty tracker, Plan 212)   │
          │  BLAKE3 attribution (Doc 57 Layer 1)                   │
          └────────────────────────────────────────────────────────┘
```

5+ pillars: Plan 223, Plan 272, Plan 306, Plan 319, Research 139, Doc 57, chain slashing, gm_key_store.

---

## TL;DR

Forensic asset fingerprinting is a 30-year-old DRM technique (Tardos, Widevine, PlayReady). Our novelty is the **fusion** with the existing latent-state security stack: per-client BLAKE3-derived perturbation recipes applied client-side inside the WASM asset vessel (Plan 319), reusing the SIMD egg/shell determinant pipeline (Plan 223), with attribution via NFT registry (Research 139) and slashing via existing chain penalty tracker. The **key insight** is that anti-tamper and attribution collapse into one mechanism when `E₂` is per-client-derived — same hot path, two questions answered. Recipe bandwidth is ~64 bytes/client (78000× reduction vs naive per-client content). LoopWM's `A = diag(−exp(a))` spectral stability primitive (arxiv 2606.18208) bounds cumulative perturbation across chunk graphs — a clean transferable inference-time construct. **GOAT verdict** (not Super-GOAT): math is industry-known, fusion is novel, ≥5 pillar multiplier, clear selling point ("leaked assets are traceable to source NFT owner within hours"). Same reasoning as Research 257 (FuncAttn). Plan + implement behind feature flag; GOAT gate on attribution accuracy, collusion resistance, visual quality preservation. Open primitive → katgpt-rs Plan 293; private runtime integration → riir-ai Research 140 + Plan 322.
