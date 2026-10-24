//! Issue 961 T1 — row-relative margin-to-max selection on
//! `GqaFlashMemorySelector` (shared helper with `FlashMemorySelector`).
//!
//! Fixture: block_size 1, so each block centroid IS one key; scores are
//! `q · k · attn_scale` and can be written down by hand.

use katgpt_attn::dash_attn::flashmemory_sparse::{
    FlashMemoryConfig, GqaFlashMemoryBlockCache, GqaFlashMemorySelector,
};

const HEAD_DIM: usize = 2;

fn cfg(threshold: f32) -> FlashMemoryConfig {
    FlashMemoryConfig {
        block_size: 1,
        refresh_period: 1,
        threshold,
    }
}

/// One KV head; keys are `[k0, 1.0]` so a query `[1, c]` scores `k0 + c`.
fn cache(k0: &[f32]) -> GqaFlashMemoryBlockCache {
    let keys: Vec<f32> = k0.iter().flat_map(|&k| [k, 1.0]).collect();
    let mut c = GqaFlashMemoryBlockCache::new(1, HEAD_DIM, &cfg(0.5), k0.len());
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

const K0: [f32; 4] = [2.0, 1.0, -1.0, 0.5];

#[test]
fn margin_selects_relative_to_row_max() {
    let c = cache(&K0);
    let s = |delta: f32| {
        let mut sel =
            GqaFlashMemorySelector::new(cfg(0.5), 1, HEAD_DIM, K0.len()).with_margin(delta);
        select(&mut sel, &c, 0.0)
    };
    assert_eq!(s(0.0), vec![0], "delta 0 keeps only the argmax");
    assert_eq!(s(1.0), vec![0, 1], "cut = 2 - 1 = 1 (inclusive)");
    assert_eq!(s(1.6), vec![0, 1, 3]);
    assert_eq!(s(1e6), vec![0, 1, 2, 3], "huge delta keeps every block");
}

#[test]
fn margin_rule_is_offset_invariant_and_absolute_is_not() {
    let c = cache(&K0);
    let mut m = GqaFlashMemorySelector::new(cfg(0.5), 1, HEAD_DIM, K0.len()).with_margin(1.0);
    let mut a = GqaFlashMemorySelector::new(cfg(0.9), 1, HEAD_DIM, K0.len());
    for offset in [-5.0f32, 0.0, 3.0, 40.0] {
        assert_eq!(select(&mut m, &c, offset), vec![0, 1], "offset {offset}");
    }
    // σ(s) ≥ 0.9 ⇔ s ≥ 2.197: nothing at offset 0, everything at offset 40.
    assert!(select(&mut a, &c, 0.0).is_empty());
    assert_eq!(select(&mut a, &c, 40.0), vec![0, 1, 2, 3]);
}

/// Without `with_margin` the selector is the absolute rule exactly — the
/// shared helper did not change the shipped path.
#[test]
fn threshold_path_matches_sigmoid_rule() {
    let k0: Vec<f32> = (0..64)
        .map(|i| ((i * 37 % 64) as f32 - 32.0) / 8.0)
        .collect();
    let c = cache(&k0);
    for thr in [0.1f32, 0.5, 0.73, 0.95] {
        let mut sel = GqaFlashMemorySelector::new(cfg(thr), 1, HEAD_DIM, k0.len());
        let got = select(&mut sel, &c, 0.25);
        let want: Vec<usize> = k0
            .iter()
            .enumerate()
            .filter(|&(_, &k)| katgpt_core::sigmoid(k + 0.25) >= thr)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(got, want, "thr {thr}");
    }
}

#[test]
#[should_panic(expected = "margin must be finite and >= 0")]
fn negative_margin_is_refused() {
    let _ = GqaFlashMemorySelector::new(cfg(0.5), 1, HEAD_DIM, 4).with_margin(-1.0);
}
