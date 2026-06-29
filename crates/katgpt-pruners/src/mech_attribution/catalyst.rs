//! Structural repetition detector — catalyst pattern matching (no ML needed).

use super::types::{CatalystPattern, InfluenceConfig, MechInfluenceScore};

/// Detect the dominant structural catalyst pattern in `text`.
///
/// Returns `(pattern, score)` where score ∈ [0, 1]. If all scores are below
/// the config threshold, returns `(CatalystPattern::None, 0.0)`.
pub fn detect_catalyst_pattern(text: &str) -> (CatalystPattern, f32) {
    detect_catalyst_pattern_with_threshold(text, 0.0)
}

/// Detect catalyst pattern with an explicit threshold override.
pub fn detect_catalyst_pattern_with_threshold(
    text: &str,
    threshold: f32,
) -> (CatalystPattern, f32) {
    let xml = xml_score(text);
    let code = code_score(text);
    let latex = latex_score(text);
    let db = database_score(text);
    let rep = pure_repetition_score(text);

    let candidates = [
        (CatalystPattern::XmlRepetition, xml),
        (CatalystPattern::CodeSignature, code),
        (CatalystPattern::LatexFormula, latex),
        (CatalystPattern::DatabaseRow, db),
        (CatalystPattern::PureRepetition, rep),
    ];

    let (best_pattern, best_score) = candidates
        .into_iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or((CatalystPattern::None, 0.0));

    if best_score <= 0.0 || best_score < threshold {
        (CatalystPattern::None, 0.0)
    } else {
        (best_pattern, best_score)
    }
}

/// Compute a full [`MechInfluenceScore`] for a single text sample.
pub fn catalyst_score(text: &str, config: &InfluenceConfig) -> MechInfluenceScore {
    let (pattern, overlap) =
        detect_catalyst_pattern_with_threshold(text, config.catalyst_threshold);
    MechInfluenceScore {
        catalyst_overlap: overlap,
        pattern,
        is_high_influence: false, // caller sets this after batch ranking
    }
}

// ── Individual detectors ──────────────────────────────────────────────

/// XML detection: count `<tag>...</tag>` pairs — score by tag-pair density.
fn xml_score(text: &str) -> f32 {
    if text.is_empty() {
        return 0.0;
    }

    let mut open_count: usize = 0;
    let mut close_count: usize = 0;
    let bytes = text.as_bytes();
    let len = bytes.len();

    let mut i = 0;
    while i < len {
        if bytes[i] == b'<' {
            if i + 1 < len && bytes[i + 1] == b'/' {
                // closing tag </...>
                close_count += 1;
            } else if i + 1 < len && bytes[i + 1] != b'!' && !bytes[i + 1].is_ascii_whitespace() {
                // opening tag <...> (not comment/declaration)
                open_count += 1;
            }
        }
        i += 1;
    }

    let pairs = open_count.min(close_count);
    if pairs == 0 {
        return 0.0;
    }

    // Density: pairs per 100 chars, clamped to [0, 1]
    let density = (pairs as f32) / (text.len() as f32) * 100.0;
    density.min(1.0)
}

/// Code detection: count structural delimiters `{`, `;`, `:` — score by delimiter density.
fn code_score(text: &str) -> f32 {
    if text.is_empty() {
        return 0.0;
    }

    let mut delimiter_count: usize = 0;
    let mut paren_pairs: usize = 0;
    let mut open_parens: usize = 0;

    for ch in text.chars() {
        match ch {
            '{' | '}' | ';' => delimiter_count += 1,
            '(' => open_parens += 1,
            ')' => {
                if open_parens > 0 {
                    open_parens -= 1;
                    paren_pairs += 1;
                }
            }
            _ => {}
        }
    }

    let total = delimiter_count + paren_pairs;
    if total == 0 {
        return 0.0;
    }

    let density = (total as f32) / (text.len() as f32) * 50.0;
    density.min(1.0)
}

/// LaTeX detection: count `\command{...}` patterns — score by backslash-command density.
fn latex_score(text: &str) -> f32 {
    if text.is_empty() {
        return 0.0;
    }

    let mut command_count: usize = 0;
    let chars: Vec<char> = text.chars().collect();
    let len = chars.len();

    let mut i = 0;
    while i < len {
        if chars[i] == '\\' && i + 1 < len && chars[i + 1].is_ascii_alphabetic() {
            command_count += 1;
        }
        i += 1;
    }

    if command_count == 0 {
        return 0.0;
    }

    let density = (command_count as f32) / (text.len() as f32) * 100.0;
    density.min(1.0)
}

/// Database row detection: count lines with consistent field count (split by `|` or `,`).
fn database_score(text: &str) -> f32 {
    if text.is_empty() {
        return 0.0;
    }

    let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    if lines.len() < 2 {
        return 0.0;
    }

    // Try pipe-delimited first, then comma-delimited
    for separator in &['|', ','] {
        let field_counts: Vec<usize> = lines.iter().map(|l| l.split(*separator).count()).collect();

        if field_counts.iter().all(|&c| c >= 2) {
            let first = field_counts[0];
            let consistent = field_counts.iter().filter(|&&c| c == first).count();
            let consistency = consistent as f32 / field_counts.len() as f32;
            if consistency >= 0.8 {
                return consistency * 0.9; // cap slightly below 1.0 for realism
            }
        }
    }

    0.0
}

/// Pure repetition detection: find repeated substrings of length ≥ min_len.
/// Score by repetition ratio (how much of the text is consumed by repeats).
fn pure_repetition_score(text: &str) -> f32 {
    let min_len = 3;
    let len = text.len();
    if len < min_len * 3 {
        return 0.0;
    }

    let bytes = text.as_bytes();
    let mut best_ratio: f32 = 0.0;

    // Check substrings of increasing length
    for sub_len in min_len..=(len / 3).max(min_len) {
        // Limit search space for performance
        if sub_len > 40 {
            break;
        }
        for start in 0..=(len - sub_len).min(30) {
            let candidate = &bytes[start..start + sub_len];
            let mut count: usize = 0;
            let mut pos = 0;
            while pos + sub_len <= len {
                if &bytes[pos..pos + sub_len] == candidate {
                    count += 1;
                    pos += sub_len;
                } else {
                    pos += 1;
                }
            }
            if count >= 3 {
                let ratio = (count * sub_len) as f32 / len as f32;
                best_ratio = best_ratio.max(ratio);
            }
        }
    }

    best_ratio.min(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xml_detection() {
        let text = r#"<root><item>hello</item><item>world</item></root>"#;
        let (pattern, score) = detect_catalyst_pattern(text);
        assert_eq!(pattern, CatalystPattern::XmlRepetition);
        assert!(score > 0.0, "XML score should be > 0, got {score}");
    }

    #[test]
    fn test_code_detection() {
        let text = "fn foo(x: i32) -> i32 { let y = x + 1; y }";
        let (pattern, score) = detect_catalyst_pattern(text);
        assert_eq!(pattern, CatalystPattern::CodeSignature);
        assert!(score > 0.0, "Code score should be > 0, got {score}");
    }

    #[test]
    fn test_latex_detection() {
        let text = r"\frac{a}{b} + \sqrt{c} = \sum_{i=0}^{n} x_i";
        let (pattern, score) = detect_catalyst_pattern(text);
        assert_eq!(pattern, CatalystPattern::LatexFormula);
        assert!(score > 0.0, "LaTeX score should be > 0, got {score}");
    }

    #[test]
    fn test_database_row_detection() {
        let text = "a|b|c\n1|2|3\n4|5|6\n7|8|9";
        let (pattern, score) = detect_catalyst_pattern(text);
        assert_eq!(pattern, CatalystPattern::DatabaseRow);
        assert!(score > 0.0, "Database score should be > 0, got {score}");
    }

    #[test]
    fn test_pure_repetition() {
        let text = "abc abc abc abc abc abc abc";
        let (pattern, score) = detect_catalyst_pattern(text);
        assert_eq!(pattern, CatalystPattern::PureRepetition);
        assert!(
            score > 0.3,
            "Pure repetition score should be > 0.3, got {score}"
        );
    }

    #[test]
    fn test_natural_language_none() {
        let text = "The quick brown fox jumps over the lazy dog. This is a normal sentence.";
        let (pattern, _score) = detect_catalyst_pattern_with_threshold(text, 0.5);
        assert_eq!(pattern, CatalystPattern::None);
    }

    #[test]
    fn test_empty_string() {
        let (pattern, score) = detect_catalyst_pattern("");
        assert_eq!(pattern, CatalystPattern::None);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_catalyst_score_fn() {
        let config = InfluenceConfig::default();
        let text = "<data><row>1</row><row>2</row></data>";
        let result = catalyst_score(text, &config);
        assert_eq!(result.pattern, CatalystPattern::XmlRepetition);
        assert!(!result.is_high_influence); // not set by catalyst_score alone
    }
}
