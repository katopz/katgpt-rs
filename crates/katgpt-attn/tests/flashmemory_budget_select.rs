//! Issue 962 — budget (top-rho) selection on `GqaFlashMemorySelector` (the
//! shared `select_scored_row` helper serves `FlashMemorySelector` too).
//!
//! Fixture: block_size 1, so each block centroid IS one key and scores are
//! `q · k · attn_scale`. The last block is the current one and is always kept.
//!
//! Separate test binary: the counting allocator must not see other tests.

#[path = "../../katgpt-core/tests/common/mod.rs"]
mod common;
counting_allocator!();

use katgpt_attn::dash_attn::flashmemory_sparse::{
    FlashMemoryConfig, GqaFlashMemoryBlockCache, GqaFlashMemorySelector,
};

const HEAD_DIM: usize = 2;

fn cfg() -> FlashMemoryConfig {
    FlashMemoryConfig {
        block_size: 1,
        refresh_period: 1,
        threshold: 0.5,
    }
}

/// One KV head; keys are `[k0, 1.0]` so a query `[1, c]` scores `k0 + c`.
fn cache(k0: &[f32]) -> GqaFlashMemoryBlockCache {
    let keys: Vec<f32> = k0.iter().flat_map(|&k| [k, 1.0]).collect();
    let mut c = GqaFlashMemoryBlockCache::new(1, HEAD_DIM, &cfg(), k0.len());
    c.rebuild_from_keys(&keys, k0.len());
    c
}

fn select(
    sel: &mut GqaFlashMemorySelector,
    c: &GqaFlashMemoryBlockCache,
    offset: f32,
) -> Vec<usize> {
    sel.force_refresh();
    sel.select(&[1.0, offset], c, 1.0, 0).blocks_per_head[0].clone()
}

// 10 blocks; block 9 is current. Ranking of 0..9: 3 (9.0) > 7 (8.0) > 0 (5.0) > 5 (4.0) > ...
const K0: [f32; 10] = [5.0, -1.0, 0.0, 9.0, 1.0, 4.0, -2.0, 8.0, 2.0, -9.0];

#[test]
fn budget_keeps_top_k_plus_current() {
    let c = cache(&K0);
    let s = |rho: f32| {
        let mut sel = GqaFlashMemorySelector::new(cfg(), 1, HEAD_DIM, K0.len()).with_budget(rho);
        select(&mut sel, &c, 0.0)
    };
    // k = round(rho * 10) - 1
    assert_eq!(s(0.1), vec![9], "k = 0: only the current block");
    assert_eq!(s(0.2), vec![3, 9]);
    assert_eq!(s(0.4), vec![0, 3, 7, 9], "ascending output");
    assert_eq!(
        s(1.0),
        (0..10).collect::<Vec<_>>(),
        "rho 1 keeps everything"
    );
    // the current block is kept even though it has the lowest score
    assert!(s(0.3).contains(&9));
}

#[test]
fn budget_is_offset_invariant_and_ties_are_deterministic() {
    let c = cache(&K0);
    let mut sel = GqaFlashMemorySelector::new(cfg(), 1, HEAD_DIM, K0.len()).with_budget(0.4);
    let base = select(&mut sel, &c, 0.0);
    for offset in [-30.0f32, 7.0, 50.0] {
        assert_eq!(select(&mut sel, &c, offset), base, "offset {offset}");
    }
    // all-equal scores: the lowest indices win
    let flat = cache(&[1.0; 6]);
    let mut sel = GqaFlashMemorySelector::new(cfg(), 1, HEAD_DIM, 6).with_budget(0.5);
    assert_eq!(select(&mut sel, &flat, 0.0), vec![0, 1, 5]);
}

#[test]
fn budget_single_block_row_keeps_it() {
    let c = cache(&[3.0]);
    let mut sel = GqaFlashMemorySelector::new(cfg(), 1, HEAD_DIM, 1).with_budget(0.05);
    assert_eq!(select(&mut sel, &c, 0.0), vec![0]);
}

#[test]
#[should_panic(expected = "budget must be in (0, 1]")]
fn zero_budget_is_refused() {
    let _ = GqaFlashMemorySelector::new(cfg(), 1, HEAD_DIM, 4).with_budget(0.0);
}

/// G4: steady-state budget refreshes allocate nothing — the rank scratch and
/// the per-head selection capacity are reserved at construction.
#[test]
fn budget_refresh_allocates_nothing() {
    assert_counter_is_live();
    let k0: Vec<f32> = (0..256)
        .map(|i| ((i * 37 % 256) as f32 - 128.0) / 16.0)
        .collect();
    let c = cache(&k0);
    let mut sel = GqaFlashMemorySelector::new(cfg(), 1, HEAD_DIM, k0.len()).with_budget(0.15);
    let q = [1.0f32, 0.0];
    sel.select(&q, &c, 1.0, 0); // warm-up refresh
    let (n_sel, bytes) = alloc_delta(|| {
        let mut total = 0usize;
        for step in 1..=64 {
            total += sel.select(&q, &c, 1.0, step).blocks_per_head[0].len();
        }
        total
    });
    assert_eq!(n_sel, 64 * 38, "round(0.15 * 256) = 38 blocks per refresh");
    assert_eq!(bytes, 0, "budget refresh path allocated");
}
