//! Template extraction → synthetic data generation pipeline.

use super::catalyst::detect_catalyst_pattern;
use super::types::{CatalystPattern, InfluenceConfig};

/// A template extracted from catalyst-pattern samples.
#[derive(Debug, Clone)]
pub struct CatalystTemplate {
    /// Which catalyst pattern this template captures.
    pub pattern: CatalystPattern,
    /// The template string with `{}` placeholders for variable fields.
    pub template: String,
    /// Parsed fields from the template.
    pub fields: Vec<TemplateField>,
}

/// A single field within a [`CatalystTemplate`].
#[derive(Debug, Clone)]
pub struct TemplateField {
    /// Byte position within the template string.
    pub position: usize,
    /// Whether this field is variable (placeholder) or fixed (anchor).
    pub is_variable: bool,
    /// Anchor token if the field is fixed.
    pub anchor_token: Option<String>,
}

/// Extract templates from a batch of samples that share the same catalyst pattern.
///
/// Groups samples by detected pattern, then builds one template per group by
/// aligning structural tokens and marking variable regions as placeholders.
pub fn extract_template(samples: &[&str], config: &InfluenceConfig) -> Vec<CatalystTemplate> {
    let mut groups: std::collections::HashMap<CatalystPattern, Vec<&str>> =
        std::collections::HashMap::new();

    for sample in samples {
        let (pattern, score) = detect_catalyst_pattern(sample);
        if score >= config.catalyst_threshold && pattern != CatalystPattern::None {
            groups.entry(pattern).or_default().push(*sample);
        }
    }

    let mut templates = Vec::new();

    for (pattern, group_samples) in groups {
        if group_samples.is_empty() {
            continue;
        }

        let template = match pattern {
            CatalystPattern::XmlRepetition => extract_xml_template(group_samples),
            CatalystPattern::CodeSignature => extract_code_template(group_samples),
            CatalystPattern::LatexFormula => extract_latex_template(group_samples),
            CatalystPattern::DatabaseRow => extract_db_template(group_samples),
            CatalystPattern::PureRepetition => extract_repetition_template(group_samples),
            CatalystPattern::None => continue,
        };

        templates.push(CatalystTemplate {
            pattern,
            template,
            fields: vec![], // simplified — fields populated by callers if needed
        });
    }

    templates
}

/// Generate `n` synthetic samples from a template.
///
/// For templates with `{}` placeholders, fills them with random alphanumeric tokens.
/// For structured templates without explicit placeholders, generates variations
/// based on the catalyst pattern type.
pub fn generate_synthetic(
    template: &CatalystTemplate,
    n: usize,
    rng: &mut fastrand::Rng,
) -> Vec<String> {
    let mut results = Vec::with_capacity(n);

    for _ in 0..n {
        let synthetic = match template.pattern {
            CatalystPattern::XmlRepetition => generate_xml_variation(&template.template, rng),
            CatalystPattern::CodeSignature => generate_code_variation(&template.template, rng),
            CatalystPattern::DatabaseRow => generate_db_variation(&template.template, rng),
            CatalystPattern::PureRepetition => {
                generate_repetition_variation(&template.template, rng)
            }
            _ => template.template.replace("{}", &random_token(rng, 4)),
        };
        results.push(synthetic);
    }

    results
}

// ── Template extraction helpers ───────────────────────────────────────

fn extract_xml_template(samples: Vec<&str>) -> String {
    // Find the first sample's tag structure as the template skeleton
    let sample = samples[0];
    let mut template = sample.to_string();

    // Replace text content between tags with placeholders
    // Simple approach: replace any >text< with >{}<
    {
        let bytes = template.as_bytes();
        let mut result = String::with_capacity(template.len());
        let mut i = 0;
        while i < bytes.len() {
            if bytes[i] == b'>' {
                result.push('>');
                i += 1;
                let start = i;
                while i < bytes.len() && bytes[i] != b'<' {
                    i += 1;
                }
                let content = &template[start..i];
                if !content.is_empty() && !content.trim().is_empty() {
                    result.push_str("{}");
                }
            } else {
                result.push(bytes[i] as char);
                i += 1;
            }
        }
        template = result;
    }
    template
}

fn extract_code_template(samples: Vec<&str>) -> String {
    let sample = samples[0];
    // Replace identifiers (word-chars between delimiters) with placeholders
    let mut template = String::with_capacity(sample.len());
    let chars: Vec<char> = sample.chars().collect();
    let mut i = 0;
    let mut ident_count = 0;

    while i < chars.len() {
        if chars[i].is_ascii_alphabetic() || chars[i] == '_' {
            let start = i;
            while i < chars.len() && (chars[i].is_ascii_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            // Replace every other identifier with a placeholder for variety
            if ident_count % 2 == 1 {
                template.push_str("{}");
            } else {
                template.push_str(&chars[start..i].iter().collect::<String>());
            }
            ident_count += 1;
        } else {
            template.push(chars[i]);
            i += 1;
        }
    }

    template
}

fn extract_latex_template(samples: Vec<&str>) -> String {
    // Keep LaTeX commands, replace arguments with placeholders
    let sample = samples[0];
    let chars: Vec<char> = sample.chars().collect();
    let len = chars.len();
    let mut result = String::with_capacity(len);
    let mut i = 0;

    while i < len {
        if chars[i] == '{' {
            // Find matching }
            let mut depth = 1;
            i += 1;
            while i < len && depth > 0 {
                if chars[i] == '{' {
                    depth += 1;
                } else if chars[i] == '}' {
                    depth -= 1;
                }
                i += 1;
            }
            // Replace content with placeholder
            result.push_str("{}");
        } else {
            result.push(chars[i]);
            i += 1;
        }
    }

    result
}

fn extract_db_template(samples: Vec<&str>) -> String {
    // Use first line's structure, replace fields with placeholders
    let first_line = samples[0].lines().next().unwrap_or("");
    let sep = if first_line.contains('|') { '|' } else { ',' };
    let fields: Vec<&str> = first_line.split(sep).collect();
    let placeholders: Vec<String> = fields.iter().map(|_| "{}".to_string()).collect();
    placeholders
        .iter()
        .enumerate()
        .fold(String::new(), |acc, (i, p)| {
            if i == 0 {
                p.clone()
            } else {
                acc + sep.to_string().as_str() + p
            }
        })
}

fn extract_repetition_template(samples: Vec<&str>) -> String {
    // Find the repeated unit and create a template with N repetitions
    let sample = samples[0];
    let words: Vec<&str> = sample.split_whitespace().collect();
    if words.len() < 3 {
        return sample.to_string();
    }

    // Find the most common word/trigram
    let unit = words[0];
    format!("{} {}", unit, "{} ".repeat(3).trim_end())
}

// ── Synthetic generation helpers ──────────────────────────────────────

fn generate_xml_variation(template: &str, rng: &mut fastrand::Rng) -> String {
    let mut result = template.to_string();
    while result.contains("{}") {
        result = result.replacen("{}", &random_token(rng, 5), 1);
    }
    result
}

fn generate_code_variation(template: &str, rng: &mut fastrand::Rng) -> String {
    let mut result = template.to_string();
    while result.contains("{}") {
        result = result.replacen("{}", &random_token(rng, 4), 1);
    }
    result
}

fn generate_db_variation(template: &str, rng: &mut fastrand::Rng) -> String {
    let mut result = template.to_string();
    while result.contains("{}") {
        result = result.replacen("{}", &random_token(rng, 3), 1);
    }
    result
}

fn generate_repetition_variation(_template: &str, rng: &mut fastrand::Rng) -> String {
    let count = rng.usize(2..=5);
    let token = random_token(rng, 3);
    (0..count)
        .map(|_| token.as_str())
        .collect::<Vec<_>>()
        .join(" ")
}

/// Generate a random alphanumeric token of given length.
fn random_token(rng: &mut fastrand::Rng, len: usize) -> String {
    const CHARSET: &[u8] = b"abcdefghijklmnopqrstuvwxyz0123456789";
    (0..len)
        .map(|_| CHARSET[rng.usize(..CHARSET.len())] as char)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_template_xml() {
        let config = InfluenceConfig {
            catalyst_threshold: 0.0,
            ..Default::default()
        };
        let samples = [
            "<data><item>hello</item><item>world</item></data>",
            "<data><item>foo</item><item>bar</item></data>",
        ];
        let templates = extract_template(&samples, &config);
        assert!(!templates.is_empty());
        let xml_template = templates
            .iter()
            .find(|t| t.pattern == CatalystPattern::XmlRepetition);
        assert!(xml_template.is_some(), "should find an XML template");
        let tmpl = xml_template.unwrap();
        assert!(
            tmpl.template.contains("{}"),
            "template should contain placeholders"
        );
    }

    #[test]
    fn test_generate_synthetic_xml() {
        let template = CatalystTemplate {
            pattern: CatalystPattern::XmlRepetition,
            template: "<root><item>{}</item></root>".to_string(),
            fields: vec![],
        };
        let mut rng = fastrand::Rng::with_seed(42);
        let synthetics = generate_synthetic(&template, 3, &mut rng);
        assert_eq!(synthetics.len(), 3);
        for s in &synthetics {
            assert!(s.starts_with("<root><item>"));
            assert!(s.ends_with("</item></root>"));
            assert!(!s.contains("{}"));
        }
    }

    #[test]
    fn test_generate_synthetic_repetition() {
        let template = CatalystTemplate {
            pattern: CatalystPattern::PureRepetition,
            template: "abc {} {} {}".to_string(),
            fields: vec![],
        };
        let mut rng = fastrand::Rng::with_seed(123);
        let synthetics = generate_synthetic(&template, 5, &mut rng);
        assert_eq!(synthetics.len(), 5);
        for s in &synthetics {
            assert!(!s.is_empty());
        }
    }

    #[test]
    fn test_extract_template_empty_samples() {
        let config = InfluenceConfig::default();
        let templates = extract_template(&[], &config);
        assert!(templates.is_empty());
    }

    #[test]
    fn test_random_token_deterministic() {
        let mut rng1 = fastrand::Rng::with_seed(42);
        let mut rng2 = fastrand::Rng::with_seed(42);
        let t1 = random_token(&mut rng1, 8);
        let t2 = random_token(&mut rng2, 8);
        assert_eq!(t1, t2, "same seed should produce same token");
        assert_eq!(t1.len(), 8);
    }
}
