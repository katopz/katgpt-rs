//! LIMIT — the adversarial retrieval fixture from
//! [arXiv:2508.21038](https://arxiv.org/abs/2508.21038) §5.2 (Weller, Boratko,
//! Naim & Lee, ICLR 2026), *Linguistically Simple, Geometrically Impossible Task*.
//!
//! Every retrieval quality number in this stack is measured on a **benign** qrel
//! matrix — schema-centroid item clustering, near-linear lineage chains, synthetic
//! chunk sets. LIMIT is the opposite: a *combinatorially dense* relevance
//! structure built so that every top-`k` subset of the relevant pool must be
//! returnable. The task is linguistically trivial (match one attribute) yet
//! geometrically hard, which is what makes it a capacity stress rather than a
//! semantics test.
//!
//! # Construction (faithful to §5.2)
//!
//! 1. Pick `n_relevant` documents so `C(n_relevant, k)` just exceeds the query
//!    count. The paper uses **46 docs**, since `C(46,2) = 1035` is the smallest
//!    binomial above 1000.
//! 2. Each query gets one random attribute; that attribute is written into each
//!    of its `k` relevant documents.
//! 3. Every document is padded with random non-query attributes to a uniform
//!    attribute count, so length and attribute density carry no signal.
//! 4. Optionally add `n_distractors` documents relevant to no query (the paper's
//!    "full" variant uses ~50k; the "small" variant uses 0).
//! 5. Optionally emit the **synonym** variant, where documents express each
//!    attribute with a token that shares no lexical overlap with the query's.
//!    This is what separates a lexical index from a semantic one — the paper
//!    measured BM25 dropping 97.8 → 10.6 recall@2 (−89%) across this switch while
//!    a dense model dropped only −38.9%.
//!
//! # Scope
//!
//! Generic, deterministic, allocation-tolerant (a cold-path eval fixture, not a
//! hot-path primitive), and dependency-free beyond `blake3`. The *legs* under test
//! — `ShardIndex`, `ItemEmbedIndex`, `Bm25Index` — live in `riir-neuron-db` and
//! consume this fixture from there; only the construction and the metric are
//! public.
//!
//! # Feature flag
//! `limit_fixture` — Issue 580
//!
//! Issue 580: the LIMIT adversarial retrieval fixture (arXiv:2508.21038 §5.2).
//! Opt-in — a cold-path eval fixture, never linked into production builds.

/// A document in the corpus.
#[derive(Debug, Clone, PartialEq)]
pub struct LimitDoc {
    /// Surface text: a name plus the attribute phrases.
    pub text: String,
    /// Attribute tokens this document carries, in emission order.
    pub attributes: Vec<String>,
    /// `true` when the document is relevant to no query (a distractor).
    pub is_distractor: bool,
}

/// A query and its ground-truth relevant document indices.
#[derive(Debug, Clone, PartialEq)]
pub struct LimitQuery {
    /// Surface text: the attribute phrase being asked for.
    pub text: String,
    /// The attribute token, without the surrounding phrasing.
    pub attribute: String,
    /// Indices into [`LimitFixture::docs`]. Length is `k`.
    pub relevant: Vec<usize>,
}

/// A generated LIMIT instance.
#[derive(Debug, Clone)]
pub struct LimitFixture {
    /// Relevant pool first (`0..n_relevant`), then distractors.
    pub docs: Vec<LimitDoc>,
    /// Queries, each with `k` relevant documents.
    pub queries: Vec<LimitQuery>,
    /// Size of the relevant pool — documents `0..n_relevant`.
    pub n_relevant: usize,
    /// `true` if built with the synonym mapping applied to document text.
    pub synonyms: bool,
}

/// Fixture parameters. [`LimitConfig::paper_small`] and
/// [`LimitConfig::paper_full`] reproduce the paper's two variants.
#[derive(Debug, Clone, PartialEq)]
pub struct LimitConfig {
    /// Documents in the relevant pool. Paper: 46.
    pub n_relevant: usize,
    /// Queries to emit, capped at `C(n_relevant, k)`. Paper: 1000.
    pub n_queries: usize,
    /// Documents relevant to no query. Paper: 0 (small) or 49_954 (full).
    pub n_distractors: usize,
    /// Relevant documents per query. Paper: 2.
    pub k: usize,
    /// Uniform attribute count per document (padding target).
    pub attrs_per_doc: usize,
    /// Deterministic seed.
    pub seed: u64,
    /// Emit the synonym variant (documents use non-overlapping tokens).
    pub synonyms: bool,
}

impl LimitConfig {
    /// The paper's **small** variant: 46 relevant docs, 1000 queries, no
    /// distractors. `C(46,2) = 1035`, the smallest binomial above 1000.
    pub fn paper_small(seed: u64) -> Self {
        Self {
            n_relevant: 46,
            n_queries: 1000,
            n_distractors: 0,
            k: 2,
            attrs_per_doc: 4,
            seed,
            synonyms: false,
        }
    }

    /// The paper's **full** variant: the same relevant pool plus ~50k
    /// distractors, so recall must survive a large irrelevant mass.
    pub fn paper_full(seed: u64) -> Self {
        Self {
            n_distractors: 49_954,
            ..Self::paper_small(seed)
        }
    }

    /// Flip to the synonym variant, which strips lexical overlap between query
    /// and document while preserving the relevance structure exactly.
    pub fn with_synonyms(mut self) -> Self {
        self.synonyms = true;
        self
    }
}

/// Upper bound on enumerated `k`-subsets, so a large `n_relevant` cannot make
/// the generator allocate without limit.
const MAX_SUBSETS: usize = 500_000;

/// All `k`-subsets of `0..n` in lexicographic order, capped at `limit`.
fn combinations(n: usize, k: usize, limit: usize) -> Vec<Vec<usize>> {
    let mut out: Vec<Vec<usize>> = Vec::new();
    if k == 0 || k > n || limit == 0 {
        return out;
    }
    let mut idx: Vec<usize> = (0..k).collect();
    loop {
        out.push(idx.clone());
        if out.len() >= limit {
            return out;
        }
        // Rightmost position that can still be advanced.
        let Some(i) = (0..k).rev().find(|&i| idx[i] < n - k + i) else {
            return out;
        };
        idx[i] += 1;
        for j in (i + 1)..k {
            idx[j] = idx[j - 1] + 1;
        }
    }
}

/// Deterministic xorshift64* — reproducible fixtures with no RNG dependency.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            0
        } else {
            (self.next_u64() % n as u64) as usize
        }
    }
}

/// Consonant/vowel syllable bank — generates distinct pronounceable tokens
/// without shipping a word list. Index `i` and index `j` yield tokens with no
/// shared prefix as long as `i != j`, which is what the synonym variant needs.
const ONSETS: [&str; 16] = [
    "br", "cl", "dr", "fl", "gr", "kr", "pl", "sk", "sn", "sp", "st", "tr", "vl", "wr", "zh", "th",
];
const NUCLEI: [&str; 8] = ["a", "e", "i", "o", "u", "ae", "ou", "ei"];
const CODAS: [&str; 8] = ["nd", "rk", "sh", "ft", "lm", "ng", "pt", "zz"];

/// Deterministic pseudo-word for slot `i`. Distinct `i` give distinct strings.
fn token(i: usize) -> String {
    let o = ONSETS[i % ONSETS.len()];
    let n = NUCLEI[(i / ONSETS.len()) % NUCLEI.len()];
    let c = CODAS[(i / (ONSETS.len() * NUCLEI.len())) % CODAS.len()];
    // The numeric suffix guarantees global uniqueness past the syllable space.
    format!("{o}{n}{c}{i}")
}

/// Attribute token pair: `.0` is what queries say, `.1` is the document-side
/// synonym used in the synonym variant. They share no substring.
fn attribute_pair(slot: usize) -> (String, String) {
    // Interleave from opposite ends of the token space so the two never collide
    // on the syllable bank.
    (token(slot * 2), token(slot * 2 + 1))
}

/// Build a LIMIT instance.
///
/// The number of queries is `min(cfg.n_queries, C(n_relevant, k))` — the paper's
/// sizing rule makes these nearly equal by construction. Queries enumerate
/// distinct `k`-subsets in a deterministic shuffled order, so no two queries
/// share a relevant set.
pub fn build_limit(cfg: &LimitConfig) -> LimitFixture {
    assert!(cfg.k >= 1, "k must be >= 1");
    assert!(cfg.n_relevant > cfg.k, "need more relevant docs than k");

    let mut rng = Rng::new(cfg.seed);

    // ── Enumerate distinct k-subsets of the relevant pool ──
    // Enumerate a generous superset, then shuffle and truncate. Truncating
    // *before* shuffling would bias every query toward lexicographically-early
    // documents, which would leak positional signal into the relevance structure.
    let cap = cfg
        .n_queries
        .saturating_mul(4)
        .max(cfg.n_queries)
        .min(MAX_SUBSETS);
    let mut subsets = combinations(cfg.n_relevant, cfg.k, cap);

    // Deterministic Fisher-Yates so query order carries no positional signal.
    for i in (1..subsets.len()).rev() {
        let j = rng.below(i + 1);
        subsets.swap(i, j);
    }
    subsets.truncate(cfg.n_queries);

    // ── Assign one attribute slot per query, write it into its relevant docs ──
    let n_docs = cfg.n_relevant + cfg.n_distractors;
    let mut doc_attrs: Vec<Vec<String>> = vec![Vec::new(); n_docs];
    let mut queries: Vec<LimitQuery> = Vec::with_capacity(subsets.len());

    for (slot, targets) in subsets.iter().enumerate() {
        let (q_tok, d_tok) = attribute_pair(slot);
        // The synonym variant changes ONLY the document surface form. The
        // relevance structure is byte-identical across variants, which is what
        // makes the two runs comparable.
        let doc_side = if cfg.synonyms { d_tok } else { q_tok.clone() };
        for &t in targets {
            doc_attrs[t].push(doc_side.clone());
        }
        queries.push(LimitQuery {
            text: format!("who likes {q_tok}"),
            attribute: q_tok,
            relevant: targets.clone(),
        });
    }

    // ── Pad every document to a uniform attribute count with filler ──
    // Filler tokens come from a slot range disjoint from the query attributes,
    // so padding can never accidentally satisfy a query.
    let filler_base = subsets.len() * 2 + 1;
    let target_attrs = cfg
        .attrs_per_doc
        .max(doc_attrs.iter().map(|a| a.len()).max().unwrap_or(0));
    for attrs in doc_attrs.iter_mut() {
        while attrs.len() < target_attrs {
            attrs.push(token(filler_base + rng.below(4096) * 2));
        }
    }

    // ── Surface text ──
    let docs: Vec<LimitDoc> = doc_attrs
        .into_iter()
        .enumerate()
        .map(|(i, attributes)| {
            let name = format!(
                "{} {}",
                token(filler_base + 8192 + i * 2),
                token(filler_base + 16384 + i * 2)
            );
            let body = attributes.join(" and likes ");
            LimitDoc {
                text: format!("{name} likes {body}"),
                attributes,
                is_distractor: i >= cfg.n_relevant,
            }
        })
        .collect();

    LimitFixture {
        docs,
        queries,
        n_relevant: cfg.n_relevant,
        synonyms: cfg.synonyms,
    }
}

// ── Metrics ──

/// Recall@k: fraction of a query's relevant documents appearing in the top `k`
/// of `ranked` (which must be ordered best-first).
///
/// Reported per leg and per `k` — never averaged across legs, per Issue 580 T2.
pub fn recall_at_k(ranked: &[usize], relevant: &[usize], k: usize) -> f32 {
    if relevant.is_empty() {
        return 1.0;
    }
    let cut = k.min(ranked.len());
    let hits = ranked[..cut]
        .iter()
        .filter(|d| relevant.contains(d))
        .count();
    hits as f32 / relevant.len() as f32
}

// ── Reference modelless embedder ──

/// Number of dimensions the reference embedder emits — matches `BELIEF_DIM`,
/// `ITEM_EMBED_DIM` and `LATENT_DIM` across the stack.
pub const LIMIT_EMBED_DIM: usize = 8;

/// Deterministic, weightless text → `[f32; 8]` embedding: BLAKE3 digest bytes
/// folded into 4 scalars, a DFT magnitude passed through a sigmoid, and 3 token
/// statistics, then L2-normalised.
///
/// **This mirrors the *shape* of riir-ai's `ModellessEmbedder` (BLAKE3 + DFT +
/// sigmoid → `[f32; 8]`, no weights); it is not a bit-exact copy.** The real one
/// is private and stays private — importing it would drag game IP into the public
/// engine. Numbers produced with this embedder therefore characterise *the
/// modelless 8-D regime*, not riir-ai's exact production embedder; re-running the
/// fixture against the real one is a riir-ai-side follow-up.
///
/// The honest caveat that carries over either way is the one riir-rag already
/// documents: a modelless embedding is **structural, not semantic** — two
/// structurally similar but semantically different strings score high cosine.
pub fn modelless_embed_8(text: &str) -> [f32; LIMIT_EMBED_DIM] {
    let digest = blake3::hash(text.as_bytes());
    let b = digest.as_bytes();

    let mut out = [0.0f32; LIMIT_EMBED_DIM];

    // 4 scalars from disjoint 8-byte windows of the digest, mapped to [-1, 1].
    for i in 0..4 {
        let mut acc = 0u64;
        for j in 0..8 {
            acc = (acc << 8) | b[i * 8 + j] as u64;
        }
        out[i] = ((acc >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0;
    }

    // One DFT magnitude over the digest bytes, squashed by a sigmoid. Uses the
    // repo's sigmoid-not-softmax rule: an independent bounded gate, not a
    // competitive normalisation.
    let n = b.len();
    let (mut re, mut im) = (0.0f32, 0.0f32);
    for (t, &byte) in b.iter().enumerate() {
        let ang = core::f32::consts::TAU * t as f32 / n as f32;
        let v = byte as f32 / 255.0;
        re += v * ang.cos();
        im -= v * ang.sin();
    }
    let mag = (re * re + im * im).sqrt();
    out[4] = 1.0 / (1.0 + (-mag).exp()) * 2.0 - 1.0;

    // 3 cheap surface statistics — length, token count, mean byte.
    let len = text.len() as f32;
    out[5] = (len / 128.0).tanh();
    out[6] = (text.split_whitespace().count() as f32 / 16.0).tanh();
    let mean = text.bytes().map(|c| c as f32).sum::<f32>() / len.max(1.0);
    out[7] = (mean / 128.0) - 1.0;

    let norm = out.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-12 {
        for x in out.iter_mut() {
            *x /= norm;
        }
    }
    out
}

/// Cosine of two equal-length slices, treating them as vectors (not assumed
/// unit-norm).
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < 1e-12 || nb < 1e-12 {
        return 0.0;
    }
    dot / (na * nb)
}

#[cfg(test)]
mod tests;
