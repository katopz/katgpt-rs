# Research 580: Bitstream (bit-parallel) splitting — the "bitcannon" technique class, distilled for katgpt

**Status:** RECORD — technique landed same day (Issue 872, Bench 872: the
whitespace-class SIMD splitter in `katgpt-tokenizer`'s `fast_bpe`).

## Source

- HF blog "tokenizers v1: encode, decode and scaling, measured"
  (2026-09-21) — `tokenizers` v1 RC: 3–30× faster encode, same token IDs,
  via six changes (workspace split, no-alloc model, **bitcannon** bitstream
  splitting, merge-loop rewrite, word cache, native parallelism). Credits
  gigatoken/tiktoken/kitoken/tokie/fastokens/wordchipper/ai-tokenizer as the
  ecosystem that proved each idea.
- Technique class: Parabix (bit-parallel text processing), simdjson (SIMD
  structural scanning). gigatoken is this crate's own vendored upstream
  (Issue 191 / Research 456) — v1's merge-loop + word-cache columns are the
  same wins Bench 191 already shipped; **bitcannon is the one column we had
  deferred** ("nightly portable_simd", Bench 191 §Phase 3).

## The technique (what transfers, generically)

A fixed byte-classification predicate over text becomes Boolean ops over
bitstreams: load 16/32/64 bytes, build a per-byte classification mask via
SIMD compares, reduce to an integer bitmask (`movemask`), and extract run
boundaries with `trailing_zeros`/`trailing_ones` — no per-char branch, no
regex engine, no UTF-8 decode on the fast path. Decides a whole register per
instruction group. Only sound when the pattern is a fixed compile-time
property (a model's pretokenizer grammar, a whitespace class).

Three reusable implementation laws (each cost us a caught bug — Bench 872):

1. **Unsigned SIMD compares on x86 need the xor-0x80 flip — and `>= 0x80` is
   better expressed as "the raw byte is negative as i8"** (`cmpgt(0, v)`),
   not `cmpgt(v^0x80, 0)`, which misses exactly `b == 0x80` (bx == 0).
2. **`1u32 << 32` overflows** — for a 32-byte (AVX2) chunk the all-ones mask
   must be computed in u64 (`((1u64 << w) - 1) as u32`). In release this
   wraps silently to a garbage `0` mask.
3. **Unicode predicates on the ASCII range must be checked against the
   Unicode tables, not Rust's ASCII conveniences** — `u8::is_ascii_whitespace`
   excludes vertical tab `0x0B`; `char::is_whitespace` (White_Space) includes
   it. Our SIMD predicate is the six-byte set.

Dispatch law (already house law, restated because this is the first
`fast_bpe` SIMD dispatch): never select a fast arm on a compile-time
`target_feature` — SSE2/NEON are arch baselines (always safe), AVX2 is
runtime-probed with a `#[target_feature(enable)]` kernel
(`shipped_target_feature_gate`).

## What we took vs what we left

**Took** (Issue 872, `fast_bpe/simd_split.rs`): the whitespace-class
bitstream splitter for `encode_into_pretok` — the one splitter class our own
trainer invariant (`split_whitespace`) defines. Words are maximal borrowed
byte runs (merges cross ASCII/multibyte boundaries inside a word);
identical-byte ws runs coalesce; multibyte chars classify via the exact
scalar `char::is_whitespace` predicate (bit-identity by construction).
Measured (loaded i7-13700K, median of 9 interleaved): avx2 1.60× vs the
scalar-mask level, 1.69× vs the old per-char loop shape.

**Left** (deliberate):

- **bitcannon's regex grammars** (GPT-2/cl100k/o200k/Tekken/DeepSeek): those
  belong to tokenizer *formats* this crate does not load. If a future
  consumer needs a GPT-2-class pretokenizer, the grammar→bitstream compiler
  is a new issue (and v1's source is the prior art to study, MIT/Apache).
- **HF `tokenizers` as a dependency**: katgpt-rs is public/leaf-clean; Bench
  191 already refused heavier tokenizer deps; our GGUF/SentencePiece/tiktoken
  loaders are ours. riir-train's optional `tokenizers` 0.21 stays put until
  1.0.0 leaves RC.
- **The word cache / merge-loop / batched-model-call columns**: already
  shipped here (Bench 191) or inapplicable (no per-model pipeline).

## The perf context (why this matters now, and when it doesn't)

The blog's own framing: the tokenizer is historically not the bottleneck;
it becomes one at corpus scale, high concurrency, or long repeated inputs.
For the riir-ai inference league (prefill pp2048/pp4096 vs llama.cpp) the
encode is once-per-prompt and noise vs the GPU pass — no action there. The
winning consumers are the corpus-scale ones Bench 191 already named
(riir-data ConvexTok pipeline, riir-train data gen) — still opt-in, still
waiting for the pull.
