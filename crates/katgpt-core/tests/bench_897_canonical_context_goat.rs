//! Issue 882 P4 rider (a) — canonical context assembly GOAT gate (Bench 897).
//!
//! Fixture (modelless, synthetic): a retrieved SET of N = 32 items, each a
//! byte string carrying a relevance byte, delivered in P = 256 random orders
//! (the nondeterministic retriever). The toy ORDER-SENSITIVE scorer reads the
//! assembled context bytes and weighs each item's relevance by a U-shaped
//! "lost-in-the-middle" position weight `w(p) = 1 − 0.6·sin(π p/(N−1))`
//! (the judge's per-item value is relevance + 0.25·length — it reads content,
//! so items the RETRIEVER ties are not tied for the judge).
//!
//! G1a permutation spread (max − min of the score over the P orders):
//!     - content-canonical assembly: spread == 0 exactly (every score
//!       `to_bits`-identical, every assembled buffer byte-identical);
//!     - retrieval-order assembly: spread > 0 (the defect exists);
//!     - relevance pipeline with score TIES (relevance quantized to 4
//!       levels): stable sort by score alone leaves spread > 0; the
//!       key-tie-broken `canonical_order_by_score_into` makes it 0 while
//!       keeping score-descending order;
//!     - idempotence: canonicalizing the canonical output is the identity
//!       permutation, for both orders;
//!     - the exact invariance condition: BLAKE3 keys report 0 ties; CALLER
//!       keys that collide on DISTINCT content report ties > 0 and the spread
//!       is allowed to be > 0 there — the tie count is the caller's alarm.
//! G2  sort cost: best-of µs for `canonical_order_into` at N ∈ {32, 256,
//!     1024} (pre-registered bar: N = 256 ≤ 20 µs); paired interleave of
//!     (plain assembly) vs (canonical order + assembly) at N = 256 reported;
//!     BLAKE3 ingest cost per item reported.
//! G3  the assembly path adds nothing: the identity order assembles
//!     byte-identical to plain concatenation.
//! G4  0 allocations over order + score-order + assemble (pre-sized buffer).
//!
//! Run:
//!   cargo test -p katgpt-core --release --features canonical_context \
//!     --test bench_897_canonical_context_goat -- --nocapture

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

#[path = "../../../tests/common/ab_timing.rs"]
mod ab_timing;

use ab_timing::{ab_median_ratio, best_of_us};
use katgpt_core::canonical_context::{
    ContentKey, assemble_into, assembled_len, canonical_order_by_score_into, canonical_order_into,
    content_key,
};

// ── G4: counting allocator ────────────────────────────────────────────────

static ALLOCS: AtomicUsize = AtomicUsize::new(0);

struct Counting;

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

fn allocs() -> usize {
    ALLOCS.load(Ordering::Relaxed)
}

// ── deterministic RNG ─────────────────────────────────────────────────────

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
    fn shuffle<T>(&mut self, xs: &mut [T]) {
        for i in (1..xs.len()).rev() {
            xs.swap(i, self.below(i + 1));
        }
    }
}

// ── fixture ───────────────────────────────────────────────────────────────

const N: usize = 32;
const P: usize = 256;
const SEP: &[u8] = b"\n\x1e\n";

/// Item i: `[relevance byte] ++ "doc-<i>-" ++ filler`. Relevance is the first
/// byte, so the scorer can read it back off the assembled context.
fn make_items(n: usize, levels: Option<u8>, seed: u64) -> Vec<Vec<u8>> {
    let mut rng = Rng::new(seed);
    (0..n)
        .map(|i| {
            let raw = (rng.next_u64() % 250) as u8 + 1;
            let rel = match levels {
                Some(l) => (raw % l) * (250 / l) + 1,
                None => raw,
            };
            let mut b = vec![rel];
            b.extend_from_slice(format!("doc-{i}-").as_bytes());
            let fill = 16 + rng.below(48);
            b.extend((0..fill).map(|k| b'a' + ((i * 31 + k * 7) % 26) as u8));
            b
        })
        .collect()
}

/// The toy order-sensitive scorer: reads the ASSEMBLED bytes, splits on SEP,
/// weighs each item's relevance byte by the U-shaped position weight.
fn score_context(ctx: &[u8], n: usize) -> f32 {
    let mut num = 0.0f32;
    let mut den = 0.0f32;
    let mut pos = 0usize;
    let mut start = 0usize;
    let mut i = 0usize;
    let denom = (n.max(2) - 1) as f32;
    loop {
        let end = if i + SEP.len() <= ctx.len() && &ctx[i..i + SEP.len()] == SEP {
            Some(i)
        } else if i >= ctx.len() {
            Some(ctx.len())
        } else {
            None
        };
        if let Some(e) = end {
            if e > start {
                let w = 1.0 - 0.6 * (std::f32::consts::PI * pos as f32 / denom).sin();
                // The judge reads CONTENT, not the retriever's score: its value
                // differs between retrieval-tied items (length term), which is
                // what makes tie order matter.
                num += w * (ctx[start] as f32 + 0.25 * (e - start) as f32);
                den += w;
                pos += 1;
            }
            if e >= ctx.len() {
                break;
            }
            i = e + SEP.len();
            start = i;
            continue;
        }
        i += 1;
    }
    num / den
}

fn relevance(items: &[Vec<u8>]) -> Vec<f32> {
    items.iter().map(|b| b[0] as f32).collect()
}

struct Spread {
    min: f32,
    max: f32,
    distinct_ctx: usize,
}

impl Spread {
    fn spread(&self) -> f32 {
        self.max - self.min
    }
}

/// Deliver the item set in P random orders, assemble each with `order_fn`,
/// score it. `order_fn(permuted_items, permuted_keys, order_out)`.
fn measure_spread(
    items: &[Vec<u8>],
    keys: &[ContentKey],
    seed: u64,
    mut order_fn: impl FnMut(&[&[u8]], &[ContentKey], &mut [u32]),
) -> Spread {
    let n = items.len();
    let mut rng = Rng::new(seed);
    let mut perm: Vec<usize> = (0..n).collect();
    let mut order = vec![0u32; n];
    let mut ctx = Vec::new();
    let mut first: Option<Vec<u8>> = None;
    let mut distinct = 0usize;
    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
    for _ in 0..P {
        rng.shuffle(&mut perm);
        let p_items: Vec<&[u8]> = perm.iter().map(|&i| items[i].as_slice()).collect();
        let p_keys: Vec<ContentKey> = perm.iter().map(|&i| keys[i]).collect();
        order_fn(&p_items, &p_keys, &mut order);
        assemble_into(&p_items, &order, SEP, &mut ctx);
        let s = score_context(&ctx, n);
        lo = lo.min(s);
        hi = hi.max(s);
        match &first {
            None => first = Some(ctx.clone()),
            Some(f) if *f != ctx => distinct += 1,
            _ => {}
        }
    }
    Spread {
        min: lo,
        max: hi,
        distinct_ctx: distinct,
    }
}

fn gate(name: &str, ok: bool, detail: String, fails: &mut usize) {
    println!("{} {name}: {detail}", if ok { "PASS" } else { "FAIL" });
    if !ok {
        *fails += 1;
    }
}

fn main() {
    let mut fails = 0usize;
    println!("# Bench 897 (a) — canonical context assembly (Issue 882 P4 rider a)\n");

    // ── G1a ───────────────────────────────────────────────────────────────
    let seeds = [11u64, 23, 37, 41, 59, 73, 89, 97];
    let mut worst_canon = 0.0f32;
    let mut min_raw = f32::INFINITY;
    let mut raw_sum = 0.0f32;
    let mut canon_distinct = 0usize;
    let mut worst_score_canon = 0.0f32;
    let mut min_score_stable = f32::INFINITY;
    let mut idem_ok = true;
    let mut ties_total = 0usize;
    for &seed in &seeds {
        let items = make_items(N, None, seed);
        let keys: Vec<ContentKey> = items.iter().map(|b| content_key(b)).collect();

        let raw = measure_spread(&items, &keys, seed ^ 0xA5, |_, _, o| {
            for (i, x) in o.iter_mut().enumerate() {
                *x = i as u32;
            }
        });
        let canon = measure_spread(&items, &keys, seed ^ 0xA5, |_, k, o| {
            ties_total += canonical_order_into(k, o);
        });
        worst_canon = worst_canon.max(canon.spread());
        canon_distinct += canon.distinct_ctx;
        min_raw = min_raw.min(raw.spread());
        raw_sum += raw.spread();

        // Relevance pipeline with ties.
        let t_items = make_items(N, Some(4), seed ^ 0x55);
        let t_keys: Vec<ContentKey> = t_items.iter().map(|b| content_key(b)).collect();
        let stable = measure_spread(&t_items, &t_keys, seed ^ 0x5A, |it, _, o| {
            for (i, x) in o.iter_mut().enumerate() {
                *x = i as u32;
            }
            // The ordinary pipeline: stable sort by score descending only.
            o.sort_by(|&a, &b| {
                katgpt_core::float_order::desc(it[a as usize][0] as f32, it[b as usize][0] as f32)
            });
        });
        let by_score = measure_spread(&t_items, &t_keys, seed ^ 0x5A, |it, k, o| {
            let sc: Vec<f32> = it.iter().map(|b| b[0] as f32).collect();
            canonical_order_by_score_into(&sc, k, o);
        });
        worst_score_canon = worst_score_canon.max(by_score.spread());
        min_score_stable = min_score_stable.min(stable.spread());

        // Idempotence: canonicalize the canonical output ⇒ identity.
        let mut o1 = vec![0u32; N];
        canonical_order_into(&keys, &mut o1);
        let k2: Vec<ContentKey> = o1.iter().map(|&i| keys[i as usize]).collect();
        let mut o2 = vec![0u32; N];
        canonical_order_into(&k2, &mut o2);
        idem_ok &= o2.iter().enumerate().all(|(i, &x)| x == i as u32);
        let sc = relevance(&t_items);
        canonical_order_by_score_into(&sc, &t_keys, &mut o1);
        let sc2: Vec<f32> = o1.iter().map(|&i| sc[i as usize]).collect();
        let tk2: Vec<ContentKey> = o1.iter().map(|&i| t_keys[i as usize]).collect();
        canonical_order_by_score_into(&sc2, &tk2, &mut o2);
        idem_ok &= o2.iter().enumerate().all(|(i, &x)| x == i as u32);
    }
    println!(
        "G1a retrieval-order spread: min {min_raw:.4} mean {:.4} over {} seeds × {P} orders",
        raw_sum / seeds.len() as f32,
        seeds.len()
    );
    gate(
        "G1a canonical spread == 0 (bitwise)",
        worst_canon == 0.0 && canon_distinct == 0,
        format!("worst spread {worst_canon:e}, non-identical contexts {canon_distinct}"),
        &mut fails,
    );
    gate(
        "G1a retrieval-order spread > 0 (defect exists)",
        min_raw > 0.0,
        format!("min over seeds {min_raw:.4}"),
        &mut fails,
    );
    gate(
        "G1a tied-relevance pipeline: stable-sort spread > 0, key-tie-broken == 0",
        min_score_stable > 0.0 && worst_score_canon == 0.0,
        format!("stable min {min_score_stable:.4}, canonical worst {worst_score_canon:e}"),
        &mut fails,
    );
    gate(
        "G1a idempotent (both orders)",
        idem_ok,
        "canon(canon(x)) = identity".into(),
        &mut fails,
    );
    gate(
        "G1a BLAKE3 keys report 0 ties",
        ties_total == 0,
        format!("{ties_total} ties over {} × {P} orderings", seeds.len()),
        &mut fails,
    );

    // Caller keys that collide on DISTINCT content: the tie count must fire.
    {
        let items = make_items(N, None, 5);
        // Key = relevance / 64 → 4 buckets, many collisions over distinct items.
        let ck: Vec<u8> = items.iter().map(|b| b[0] / 64).collect();
        let mut o = vec![0u32; N];
        let ties = canonical_order_into(&ck, &mut o);
        let fake_keys: Vec<ContentKey> = ck
            .iter()
            .map(|&k| {
                let mut a = [0u8; 32];
                a[0] = k;
                a
            })
            .collect();
        let s = measure_spread(&items, &fake_keys, 77, |_, k, o| {
            canonical_order_into(k, o);
        });
        gate(
            "G1a caller-key collision is FLAGGED (ties > 0)",
            ties > 0,
            format!(
                "ties {ties}, spread under colliding keys {:.4} (allowed > 0 — the tie count is the alarm)",
                s.spread()
            ),
            &mut fails,
        );
    }

    // ── G3 ────────────────────────────────────────────────────────────────
    {
        let items = make_items(N, None, 3);
        let refs: Vec<&[u8]> = items.iter().map(|b| b.as_slice()).collect();
        let id: Vec<u32> = (0..N as u32).collect();
        let mut out = Vec::new();
        assemble_into(&refs, &id, SEP, &mut out);
        let plain = refs.join(SEP);
        gate(
            "G3 identity order assembles byte-identical to plain concatenation",
            out == plain && out.len() == assembled_len(&refs, SEP),
            format!("{} bytes", out.len()),
            &mut fails,
        );
    }

    // ── G4 ────────────────────────────────────────────────────────────────
    {
        let items = make_items(256, Some(8), 9);
        let refs: Vec<&[u8]> = items.iter().map(|b| b.as_slice()).collect();
        let keys: Vec<ContentKey> = refs.iter().map(|b| content_key(b)).collect();
        let sc = relevance(&items);
        let mut order = vec![0u32; 256];
        let mut out = Vec::with_capacity(assembled_len(&refs, SEP));
        let before = allocs();
        for _ in 0..64 {
            canonical_order_into(black_box(&keys), &mut order);
            assemble_into(&refs, &order, SEP, &mut out);
            canonical_order_by_score_into(black_box(&sc), &keys, &mut order);
            assemble_into(&refs, &order, SEP, &mut out);
        }
        let n_alloc = allocs() - before;
        black_box(&out);
        gate(
            "G4 0 allocs (order + score-order + assemble, N=256, 64 rounds)",
            n_alloc == 0,
            format!("{n_alloc} allocs"),
            &mut fails,
        );
    }

    // ── G2 ────────────────────────────────────────────────────────────────
    for &n in &[32usize, 256, 1024] {
        let items = make_items(n, None, 13);
        let keys: Vec<ContentKey> = items.iter().map(|b| content_key(b)).collect();
        let mut shuffled = keys.clone();
        let mut rng = Rng::new(n as u64);
        let mut order = vec![0u32; n];
        let us = best_of_us(20, 200, || {
            rng.shuffle(&mut shuffled);
            let st = Instant::now();
            let t = canonical_order_into(black_box(&shuffled), black_box(&mut order));
            let e = st.elapsed();
            black_box((t, &order));
            e
        });
        let label = format!("G2 canonical_order_into N={n} best-of");
        if n == 256 {
            gate(
                &label,
                us <= 20.0,
                format!("{us:.2} µs (bar ≤ 20)"),
                &mut fails,
            );
        } else {
            println!("     {label}: {us:.2} µs");
        }
    }
    {
        let n = 256;
        let items = make_items(n, None, 17);
        let refs: Vec<&[u8]> = items.iter().map(|b| b.as_slice()).collect();
        let keys: Vec<ContentKey> = refs.iter().map(|b| content_key(b)).collect();
        let id: Vec<u32> = (0..n as u32).collect();
        let mut order = vec![0u32; n];
        let mut oa = Vec::with_capacity(assembled_len(&refs, SEP));
        let mut ob = Vec::with_capacity(assembled_len(&refs, SEP));
        let ab = ab_median_ratio(
            15,
            200,
            20,
            |_| {
                assemble_into(black_box(&refs), black_box(&id), SEP, &mut oa);
                black_box(&oa);
            },
            |_| {
                canonical_order_into(black_box(&keys), &mut order);
                assemble_into(black_box(&refs), &order, SEP, &mut ob);
                black_box(&ob);
            },
        );
        ab.report("G2 (report) N=256 plain assembly (a) vs canonical order + assembly (b)");
        let hash_us = best_of_us(5, 50, || {
            let st = Instant::now();
            for r in &refs {
                black_box(content_key(black_box(r)));
            }
            st.elapsed()
        });
        println!(
            "     G2 (report) BLAKE3 ingest: {:.3} µs/item (~{} B/item), done once at ingest",
            hash_us / n as f64,
            assembled_len(&refs, b"") / n
        );
    }

    if fails > 0 {
        println!("\n{fails} gate(s) FAILED");
        std::process::exit(1);
    }
    println!("\nall gates PASS");
}
