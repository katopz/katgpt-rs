use crate::transformer::{ForwardContext, MultiLayerKVCache, TransformerWeights, generate_into, tokens_to_string};
use crate::types::{Config, Rng};
use rayon::prelude::*;

pub fn generate_batch(count: usize, max_tokens: usize) {
    let config = Config::micro();
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);

    println!("\n📝 Generating {count} samples ({max_tokens} tokens each) in parallel...");

    let seeds: Vec<u64> = (0..count).map(|i| 42 + i as u64).collect();

    let mut samples: Vec<(usize, Vec<usize>)> = seeds
        .par_iter()
        .enumerate()
        .map_init(
            || {
                (
                    ForwardContext::new(&config),
                    MultiLayerKVCache::new(&config),
                )
            },
            |(ctx, cache), (idx, &seed)| {
                let mut sample_rng = Rng::new(seed);
                let mut tokens = Vec::with_capacity(max_tokens);
                generate_into(
                    ctx,
                    cache,
                    &weights,
                    &config,
                    &mut sample_rng,
                    max_tokens,
                    &mut tokens,
                );
                (idx, tokens)
            },
        )
        .collect();

    samples.sort_by_key(|(idx, _)| *idx);
    for (idx, tokens) in &samples {
        let text = tokens_to_string(tokens);
        println!("  Sample {}: \"{text}\"", idx + 1);
    }
}
