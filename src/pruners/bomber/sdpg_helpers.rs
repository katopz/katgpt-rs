//! Bomber-specific SDPG helpers — moved here from katgpt-pruners during Plan 005
//! extraction. These read bomber's `ReplaySample` format, which stayed in this
//! main crate (bomber module).

use std::path::Path;

use katgpt_pruners::bandit::BanditPruner;
use katgpt_pruners::sdpg::{
    AdvantageMode, BetaSchedule, KlAnchor, SdpgBanditPruner,
};
use katgpt_core::traits::ScreeningPruner;

use super::replay::ReplaySample;

/// Load teacher Q-values from a replay JSONL file.
///
/// Aggregates quality scores by action (template) index.
/// For each template: teacher_q[i] = mean(quality where action == i).
/// Templates with no samples get 0.0.
pub fn load_teacher_q_from_replay(
    path: &Path,
    num_arms: usize,
) -> std::io::Result<Vec<f32>> {
    let contents = std::fs::read(path)?;
    let mut offset = 0usize;

    let mut quality_sums = vec![0.0f32; num_arms];
    let mut counts = vec![0u32; num_arms];

    while offset + 4 <= contents.len() {
        let len = u32::from_le_bytes(contents[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;
        if offset + len > contents.len() {
            break;
        }
        if let Ok(sample) = ReplaySample::from_bytes(&contents[offset..offset + len]) {
            // Use template_id when available (0-7), fall back to action
            let idx = if sample.template_id < num_arms as u8 {
                sample.template_id as usize
            } else {
                sample.action as usize
            };
            if idx < num_arms {
                quality_sums[idx] += sample.quality;
                counts[idx] += 1;
            }
        }
        offset += len;
    }

    Ok(quality_sums
        .iter()
        .zip(counts.iter())
        .map(|(&sum, &count)| if count > 0 { sum / count as f32 } else { 0.0 })
        .collect())
}

/// Extension trait so `SdpgBanditPruner::from_replay` can live here (next to
/// its only consumer, the bomber module) instead of in katgpt-pruners.
pub trait SdpgBanditPrunerReplayExt<P: ScreeningPruner> {
    /// Create SDPG Bandit with teacher Q-values from replay data.
    ///
    /// Aggregates per-template quality from replay JSONL samples.
    /// Templates with no replay data get quality 0.0.
    fn from_replay(
        inner: BanditPruner<P>,
        replay_path: &Path,
        schedule: BetaSchedule,
        anchor: KlAnchor,
        temperature: f32,
        mode: AdvantageMode,
    ) -> std::io::Result<SdpgBanditPruner<P>>;
}

impl<P: ScreeningPruner> SdpgBanditPrunerReplayExt<P> for SdpgBanditPruner<P> {
    fn from_replay(
        inner: BanditPruner<P>,
        replay_path: &Path,
        schedule: BetaSchedule,
        anchor: KlAnchor,
        temperature: f32,
        mode: AdvantageMode,
    ) -> std::io::Result<SdpgBanditPruner<P>> {
        let teacher_q = load_teacher_q_from_replay(replay_path, inner.q_values().len())?;
        Ok(SdpgBanditPruner::new(
            inner,
            teacher_q,
            schedule,
            anchor,
            temperature,
            mode,
        ))
    }
}
