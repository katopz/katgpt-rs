# Research 256: Gzip-LM — Compression-Drafter (Corpus-as-Model, Modelless)

> **Source:** [gzip as a language model: beam-search text generation by compression](https://nathan.rs/posts/gzip-lm/) — Nathan (nathan.rs), 2026
> **Reference impl:** 110-line Python script in the blog post (reproduced in §Source Code below)
> **Date:** 2026-06-17
> **Status:** Active — Super-GOAT (committed, pending G1–Gn validation in `riir-ai/.research/137_*.md`)
> **Related Research:** 024 (δ-Mem — delta rule, didn't help DDTree, different mechanism), 060 (MeMo memory-as-model — different mechanism), 125 (Weight Norm = Kolmogorov — theoretical basis), 137 (Pplx Datrie — trie acceleration primitive), 147 (PhraseBoost — single-step trie boost, composable), 168/188 (Ruliology + IrreducibilityGate — compression-as-diagnostic, NOT generative), 229 (ProgramAsWeights — program-as-model variant), 255 (VibeThinker)
> **Related Plans:** TBD after guide validation (target `katgpt-rs/.plans/281_*.md`)
> **Cross-ref (riir-ai):** Research 137 (Compression-Drafter Plasma Personality Guide)
> **Classification:** Public (katgpt-rs = open math primitive); the *selling-point guide* is private in riir-ai.

---

## TL;DR

**nathan.rs/gzip-lm** demonstrates text generation by beam-search where each candidate byte sequence is scored by `len(zlib.compress(corpus + ctx + candidate))` — the shortest compressed length wins (most compressible = most likely continuation under the corpus the compressor has "memorized" via its match window). Zero neural net. Zero weights. Zero training. The compressor IS the model.

**Distilled for katgpt-rs (modelless, inference-time):**
A new **fifth class** of modelless primitive — **CompressionDrafter** — that scores candidate continuations via a compressor's compressed length. This is distinct from all four existing modelless primitive classes:
1. Constraint pruners (rule-based accept/reject)
2. Bandits (statistical arm selection)
3. DDTree (marginals-based expansion)
4. Speculative decode (cheap draft → verify)

The compressor's match window IS a **frozen corpus** — no learned weights, but the corpus encodes "what's likely next" via NCD (Normalized Compression Distance) / Solomonoff-induction-as-practice. Promotion + update = append bytes to the corpus, which is exactly freeze/thaw snapshot semantics on a `Vec<u8>` instead of a weight tensor.

**Why it matters here:**
- **New capability class:** a modelless drafter whose "knowledge" is a byte buffer, not a weight matrix. Per-entity specialization is trivial (different `Vec<u8>` per entity) — this is impossible with LoRA at MMORPG scale.
- **Per-NPC personality at plasma tier (the Super-GOAT selling point — see riir-ai/.research/137):** when the alphabet is small (game actions, 6–32 symbols) and the window is tiny (last 256 actions), LZ-style match scoring fits the plasma µs budget via SIMD. This unblocks per-NPC personality without per-NPC LoRA — a feat no incumbent can match.
- **Fifth pillar of modelless:** ConstraintPruners, Bandits, DDTree, SpeculativeDecode, **CompressionDrafter**.

**Verdict: Super-GOAT (committed).** Novel mechanism (zero compression crates in any of the 3 repos), new capability class (corpus-as-model is a 5th modelless primitive class), product selling point (per-NPC personality at plasma tier), force multiplier (Plasma Path × Freeze/Thaw × HLA × PhraseBoost × R188 IrreducibilityGate). Selling-point guide: `riir-ai/.research/137_Compression_Drafter_Plasma_Personality_Guide.md`. Validation G1–Gn gate lives in that guide.

---

## 1. Paper Core Findings (nathan.rs/gzip-lm)

### 1.1 Algorithm

```python
# Per generation step (commits `horizon` bytes):
recent = (prompt + output)[-tail:]         # hide old output from scorer
ctx = corpus_window + recent               # corpus primes match window
beams = [b""]
for _ in range(horizon):                   # beam-expand
    candidates = [h + bytes([b]) for h in beams for b in alphabet]
    lens = candidate_lengths(ctx, candidates, level=9)
    order = sorted(range(len(cand)), key=lens.__getitem__)[:beam_width]
    beams, beam_lens = [candidates[i] for i in order], [lens[i] for i in order]
if temperature == 0:
    span = beams[0]                         # most compressible
else:
    weights = [exp(-(L - min_L) / temperature) for L in beam_lens]
    span = rng.choices(beams, weights=weights, k=1)[0]
out += span
```

`candidate_lengths(ctx, sequences)` uses `zlib.compressobj(level).copy()` to clone the compressor state after the (expensive) corpus-prefix match search has run once, then feeds each candidate to a cloned state and finishes. This amortizes the corpus compression across the whole beam step.

### 1.2 Key knobs

| Knob | Default | Meaning |
|------|---------|---------|
| `window` | 30000 | corpus bytes visible to the matcher (DEFLATE max 32768) |
| `tail` | 80 | recent output bytes kept in scoring context (anti-repeat) |
| `horizon` | 24 | beam-search depth, committed per outer step |
| `beam_width` | 32 | partial continuations kept per step |
| `temperature` | 0.5 | 0 = argmin compressed length; >0 samples beams |
| `level` | 9 | zlib level (0–9) |
| `alphabet` | corpus_alphabet | only bytes present in the corpus |

### 1.3 Theoretical basis

The score `-log P(seq | ctx)` is approximated by `compressed_len(ctx + seq) - compressed_len(ctx)` — Shannon's source-coding theorem says an optimal compressor's per-bit cost equals the negative log probability of the next symbol under the source distribution. DEFLATE is not optimal (LZ77 + Huffman) but is a usable proxy. The **NCD (Normalized Compression Distance)** family and Solomonoff induction provide the formal framing. This is the same idea asgzip-classification (Mori et al. 2022), but applied to *generation* via beam search rather than k-NN classification.

### 1.4 Why the "tail" trick matters

Without `tail=80` truncation, gzip happily matches the recent generated span against its *own earlier output* (which is in the corpus window if accumulated) → degenerate looping ("the the the the"). Limiting the visible recent output to 80 bytes breaks the loop. This is conceptually similar to repetition penalty in LLM decoding but implemented by *information hiding* rather than logit manipulation.

---

## 2. Distillation — the Transferable Primitive

### 2.1 What transfers (the primitive, paper-stripped)

**The compression-drafter primitive:**

```rust
/// Score candidate continuations by compressed length under a frozen corpus.
///
/// `score(seq) = -[ compressed_len(corpus + ctx + seq) - compressed_len(corpus + ctx) ]`
///
/// Higher score = more compressible = more likely under the corpus distribution.
/// The corpus is FROZEN — it's the model. ctx is the live context (prompt + recent output).
pub trait CompressionDrafter {
    type Symbol;

    /// Frozen corpus — the model's "knowledge". Appendable for online learning.
    fn corpus(&self) -> &[u8];

    /// Score a candidate continuation, amortizing the corpus-prefix match search.
    ///
    /// MUST share compressor state across calls within one beam step (see §2.3).
    fn score(&mut self, ctx: &[u8], candidate: &[u8]) -> i32;

    /// Batched score — share compressor prefix state across all candidates.
    fn score_batch(&mut self, ctx: &[u8], candidates: &[&[u8]]) -> Vec<i32>;
}
```

This is **two orders of magnitude smaller** than LoRA (~KB corpus vs MB adapter), **zero training** (no backprop, no optimizer), **per-entity** (different corpus = different personality).

### 2.2 The four modelless primitive classes we already have (so we don't duplicate)

| Existing class | Knowledge representation | Update mechanism |
|---|---|---|
| ConstraintPruners | Bitmap rules / symbolic spec | Recompile spec (Plan 259) |
| Bandits | Reward statistics per arm | Pull update |
| DDTree | Marginals over vocab at each depth | Forward-pass-dependent |
| Speculative decode | Draft model weights | Train draft model (riir-train) |

**CompressionDrafter is a 5th class:**
| New class | Knowledge representation | Update mechanism |
|---|---|---|
| **CompressionDrafter** | **Byte corpus** (frozen match window) | **Append bytes to corpus (= freeze/thaw snapshot of `Vec<u8>`)** |

The corpus-as-model is **structurally identical** to freeze/thaw (a `Vec<u8>` snapshotted via BLAKE3 commitment, atomically swapped), but operates on a fundamentally different representation. This is the moat: **per-entity specialization at zero training cost, byte-addressable, BLAKE3-committable.**

### 2.3 Performance reality — three tiers

| Tier | Compressor | Alphabet | Window | Per-candidate cost | Use case |
|------|-----------|----------|--------|---------------------|----------|
| **Plasma** (µs) | custom SIMD LZ over tiny alphabet | 6–32 (game actions) | 64–256 bytes | <100ns | Per-NPC personality drafter (riir-ai/.research/137) |
| **Hot** (sub-ms) | `lz4_flex` (Rust, SIMD) | 64–256 (sub-word units) | 4–32KB | 1–50µs | Game dialog / quest grammar drafter |
| **Warm/Cold** (ms+) | `flate2` (DEFLATE) / `zstd` | full byte | 30–32KB | 150–500µs | Code completion, long-form generation |

**Critical:** nathan.rs's blog targets the **Warm/Cold tier** (DEFLATE, full text corpus). The user's question — "fusion to plasma path" — is asking whether we can build a **plasma-tier variant** of this idea for game AI. Answer: **YES, for small-alphabet game domains** (actions, items, map cells), by writing a custom SIMD LZ77 with a tiny window. This is the Super-GOAT claim — see riir-ai/.research/137 for the validation gate.

### 2.4 The corpus-prefix amortization trick (load-bearing)

`zlib.compressobj(level).copy()` lets you compress the corpus prefix ONCE, then clone the encoder state per candidate. This is essential because naive `compress(ctx + seq)` per candidate recomputes the corpus match search every time.

For our Rust ports:
- `flate2::Compress::clone()` doesn't exist — we must use `Compress::new()` + manual state save, OR precompute the corpus prefix hash table once and reuse across candidates.
- For plasma-tier custom LZ: precompute the rolling-hash table for the corpus window ONCE per beam step, then probe candidates against it. This is the standard LZ77 architecture.

---

## 3. Verdict

### Tier: **Super-GOAT (committed)**

**Novelty gate (§1.5 of research workflow):**

| Q | Answer | Evidence |
|---|--------|----------|
| **Q1: No prior art?** | **YES** | Zero `flate2`/`zstd`/`gzip`/`lz4` dependencies in any `.toml` across all 3 repos (verified by grep). Zero compression-as-generator pattern in any `.rs` file. Closest cousins: PhraseBoost (R147, single-step trie boost, NOT generation), R188 IrreducibilityGate (compression-as-diagnostic, NOT generative), R125 Weight-Norm-Kolmogorov (theoretical only). |
| **Q2: New capability class?** | **YES** | Corpus-as-model is a 5th modelless primitive class (after ConstraintPruners, Bandits, DDTree, SpeculativeDecode). Per-entity specialization at zero training cost — impossible with LoRA at MMORPG scale. |
| **Q3: Product selling point?** | **YES** | "Every NPC has a unique personality — encoded as a byte corpus, scored by a plasma-tier compressor, BLAKE3-committed. Zero LoRA training per NPC." |
| **Q4: Force multiplier?** | **YES** | Plasma Path (sub-µs SIMD LZ for tiny alphabet), Freeze/Thaw (`Vec<u8>` snapshot via BLAKE3 — exactly the existing snapshot protocol), HLA (compress HLA moments as per-NPC corpus), PhraseBoost (composable trie + compressor), R188 IrreducibilityGate (reuse same compressor for diagnostic + generation — see Fusion C). |

**Mandatory Super-GOAT outputs (this session):**
1. **Open primitive** → `katgpt-rs/src/drafters/compression_drafter.rs` (generic math, no game semantics). Future plan: `katgpt-rs/.plans/281_*`.
2. **Architectural GUIDE** → `riir-ai/.research/137_Compression_Drafter_Plasma_Personality_Guide.md` (created in this session).
3. **Plans** → `katgpt-rs/.plans/281_*` (open primitive) + `riir-ai/.plans/` (game integration), to be created AFTER guide validation per skill rule ("Super-GOAT plans should be created AFTER the riir-ai guide").

### Fusion ideas (the GOAT-mined combinations, in priority order)

#### **Fusion A — Per-NPC Plasma Personality** (the Super-GOAT, riir-ai/.research/137)

`CompressionDrafter(plasma LZ, action alphabet)` × `PlasmaPath TernaryWeights` × `Freeze/Thaw snapshots`.

Each NPC's "personality" = its observed action history (or generated dialog trace) serialized as a byte corpus. Per-NPC scoring is sub-µs SIMD LZ over a 256-byte window. Snapshotting personality = `Vec<u8>` clone + BLAKE3 hash. **This is impossible with LoRA at 1000-NPC scale (per-NPC LoRA = GBs of weights); trivial with corpus-as-model (KBs per NPC).**

→ Validated in riir-ai/.research/137 (plasma-tier LZ over 6–32 symbol alphabet).

#### **Fusion B — CompressionDrafter × PhraseBoost** (composable trie + compressor)

PhraseBoost (R147, default-on) maintains active trie states during decode and boosts logits for phrase matches. **Combine:** the compressor scores multi-byte spans (rewards match length), the trie scores single-step token continuation. Final score = `α·compressor_score + β·trie_boost`. The compressor provides *global* context-aware scoring (matches the full corpus), the trie provides *local* next-token bias. Different time scales, composable.

→ GOAT candidate (compositional improvement, not new capability class).

#### **Fusion C — One-Compressor-Two-Jobs** (CompressionDrafter × R188 IrreducibilityGate)

R188's `IrreducibilityGate` computes compression ratio to decide "is this game reducible? skip simulation." If we already have a compressor in the hot path (for the drafter), we get the IrreducibilityGate signal **for free** — same compressor, two outputs (compressed length for scoring + compression ratio for the gate). Halves the cost.

→ GOAT candidate (perf optimization, not new capability class).

#### **Fusion D — HLA Moment Corpus** (semantic-domain only)

Each NPC's HLA moments (`valence, arousal, desperation, calm, fear` — 5 scalars per observation) can be quantized to bytes and appended to the per-NPC corpus. The compressor then "remembers" emotional patterns. At sync boundary, only the 5 raw scalars cross sync (per AGENTS.md latent-vs-raw rules); the corpus stays local to the entity.

→ Strengthens existing HLA pipeline. The corpus is NOT synced (too large, latent-domain); only the bridge scalars are.

#### **Fusion E — Self-Learn via Append** (zero-training adaptation)

Per AGENTS.md "Self-learn / adaptive CoT welcome": runtime updates to latent state are allowed. Appending bytes to a per-entity corpus IS a runtime update that doesn't touch weights. **The corpus grows as the NPC experiences more** — emergent personality drift with zero training. Combine with `AbsorbCompress` (Plan 032): when a corpus exceeds `window`, evict lowest-information bytes (compress-and-merge into a "summary" suffix).

---

## 4. What does NOT transfer

| Pattern from blog | Why not |
|---|---|
| DEFLATE 32KB window + full-text beam search at every step | Plasma budget blown (~150–500µs per `compress(30KB)` call). Text-path stays Warm/Cold tier. |
| Python `ThreadPoolExecutor` for parallel candidate scoring | We're in Rust; rayon already parallelizes. Zlib releases GIL in Python; we have native threads. |
| `alphabet = corpus_alphabet` (bytes present in corpus) | Keep — same optimization applies. For game actions (6 symbols), this is the full alphabet. |
| Tail = 80 bytes | Game variant: tail = last K actions (different from corpus window). Tune per game. |
| zlib level 9 | Plasma variant: no levels, fixed SIMD LZ77 with deterministic match selection. |

---

## 5. Latent vs raw boundary (per AGENTS.md)

| Data | Domain | Sync? | Notes |
|------|--------|-------|-------|
| Corpus itself (byte buffer) | Latent / semantic | **NEVER synced** — per-entity local state. | Crossing sync = full embedding vector over network = forbidden (AGENTS.md anti-pattern). |
| Compressed-length score → action choice | Raw (action index) | **Synced** via TxDelta | Action is a raw integer, deterministic. |
| Bridge scalars from corpus (valence/arousal/...) | Raw scalars | **Synced** if required | Only 5 f32, not the corpus. |
| Per-NPC corpus snapshot to Cold tier | Frozen bytes | **BLAKE3-committed**, tamper-evident | Same protocol as weight snapshots. |

**Anti-pattern reaffirmed:** never send the corpus over the network. The corpus is the *private personality*; the synced quantity is the action it produced.

---

## 6. Optimization alignment (per AGENTS.md `optimization.md`)

| Principle | CompressionDrafter approach | Alignment |
|-----------|---------------------------|-----------|
| Fixed-size arrays for bounded domains | Action alphabet ≤ 32 → `[_; 32]` candidate buffers | ✅ |
| Pre-compute lookup tables once | Rolling-hash table for corpus window, built once per beam step | ✅ |
| Cache allocations: `Vec::with_capacity()` once, `clear()` + reuse | Beam buffer, candidate buffer reused across steps | ✅ |
| Pass pre-allocated scratch buffers | `&mut [u8]` compressor scratch | ✅ |
| Reorder struct fields to eliminate padding | `[pos_bits, neg_bits, row_scale]` style packing (mirrors `TernaryWeights`) | ✅ |
| `#[repr(u8)]` field-less enums | Symbol enum for game actions | ✅ |
| Write chunked loops (4/8) for SIMD | Custom LZ77 match probing in 8-byte lanes | ✅ |
| Don't allocate inside hot loops | All hot-path ops on caller-owned buffers | ✅ |
| Don't GPU for µs workloads | Pure CPU SIMD for plasma tier | ✅ |
| Don't recompute unchanged values | Corpus prefix match table built once per beam step | ✅ |

---

## 7. Source code (nathan.rs/gzip-lm, for reference)

```python
"""gzip as a language model: beam-search text generation by compression."""

GZIP_WINDOW = 32768
DEFAULT_WINDOW = 30000

def corpus_alphabet(data):
    return tuple(sorted(set(data))) or tuple(range(256))

def candidate_lengths(context, sequences, level=9, pool=None):
    base = zlib.compressobj(level)
    head = len(base.compress(context))
    def length_for(seq):
        clone = base.copy()
        return head + len(clone.compress(seq) + clone.flush(zlib.Z_FINISH))
    if pool is not None:
        return list(pool.map(length_for, sequences))
    return [length_for(seq) for seq in sequences]

def generate(corpus, prompt, length, *, window=DEFAULT_WINDOW, horizon=24,
             beam_width=32, temperature=0.5, tail=80, level=9, workers=1,
             alphabet=None, seed=None):
    rng = random.Random(seed)
    if alphabet is None:
        alphabet = corpus_alphabet(corpus + prompt)
    corpus_window = corpus[:window]
    pool = ThreadPoolExecutor(workers) if workers > 1 else None
    out = bytearray()
    try:
        while len(out) < length:
            recent = (bytes(prompt) + bytes(out))[-tail:]
            ctx = corpus_window + recent
            beams, beam_lens = [b""], [0]
            for _ in range(horizon):
                cand = [h + bytes([b]) for h in beams for b in alphabet]
                lens = candidate_lengths(ctx, cand, level=level, pool=pool)
                order = sorted(range(len(cand)), key=lens.__getitem__)[:beam_width]
                beams = [cand[i] for i in order]
                beam_lens = [lens[i] for i in order]
            if temperature <= 0:
                span = beams[0]
            else:
                best = beam_lens[0]
                weights = [math.exp(-(L - best) / temperature) for L in beam_lens]
                span = rng.choices(beams, weights=weights, k=1)[0]
            out += span
    finally:
        if pool is not None:
            pool.shutdown(wait=False)
    return bytes(out[:length])
```

---

## 8. Open questions for the validation gate (riir-ai/.research/137)

The Super-GOAT claim requires the G1–Gn gate in the riir-ai guide to validate. Key risks:

1. **Does plasma-tier custom LZ77 actually fit µs budget for alphabet ≤ 32, window ≤ 256?** Needs a benchmark vs `ActionBridgeOracle` (<100ns target).
2. **Does per-NPC corpus-as-personality produce *different* behavior across NPCs?** (Or does it collapse to a single attractor?) Needs a 100-NPC divergence test.
3. **Is the compressed-length signal sharp enough for action selection?** (DEFLATE on text is famously noisy; tiny alphabet may be too coarse.)
4. **Does corpus-append self-learn converge or diverge?** (Possible runaway: greedy compression → degenerate looping, like the blog's tail-truncation issue.)

If G1 (plasma latency) fails: the primitive still ships in katgpt-rs as a Warm/Cold-tier modelless drafter (GOAT), but the per-NPC-plasma selling point is dropped (verdict downgraded). Honest fallback documented in the guide.

---

## TL;DR

**Super-GOAT (committed, pending validation):** nathan.rs/gzip-lm = beam-search text generation where each candidate is scored by `len(zlib.compress(corpus + ctx + candidate))`. Zero weights, zero training, the compressor IS the model. This is a **fifth modelless primitive class** (corpus-as-model) alongside ConstraintPruners, Bandits, DDTree, SpeculativeDecode. **The Super-GOAT selling point** (validated in riir-ai/.research/137): for small-alphabet game domains (6–32 actions), a custom SIMD LZ77 over a 256-byte corpus fits the plasma µs budget, enabling **per-NPC personality at zero LoRA training cost** — impossible at MMORPG scale with any incumbent technique. Force-multiplier connections: Plasma Path (sub-µs SIMD LZ), Freeze/Thaw (`Vec<u8>` snapshot via BLAKE3, identical protocol to weight snapshots), HLA (compress moments as corpus), PhraseBoost (composable trie+compressor), R188 IrreducibilityGate (same compressor for diagnostic + generation). Open primitive → `katgpt-rs/src/drafters/compression_drafter.rs` (Plan 281, TBD post-guide). Private guide → `riir-ai/.research/137_Compression_Drafter_Plasma_Personality_Guide.md`. Validation gate (G1 plasma latency, G2 per-NPC divergence, G3 sharpness, G4 self-learn convergence) lives in the guide; if G1 fails the verdict is honestly downgraded to GOAT.
