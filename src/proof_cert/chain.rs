use std::collections::{HashMap, HashSet, VecDeque};

use super::certificate::{ProofCertificate, ProofResult};

#[derive(Debug, Clone)]
pub struct ProofChainResult {
    pub proven: HashSet<String>,
    pub failed: Vec<String>,
    pub conditional: HashSet<String>,
}

/// Verify that a chain of proof certificates is consistent.
/// If certificate A implies B, and B implies C, then A's proof implies C.
pub fn verify_proof_chain(certificates: &[ProofCertificate]) -> ProofChainResult {
    let n = certificates.len();
    let mut proven: HashSet<String> = HashSet::with_capacity(n);
    let mut failed: Vec<String> = Vec::with_capacity(n);
    let mut conditional: HashSet<String> = HashSet::with_capacity(n);

    // Topological sort by dependencies
    let sorted = topological_sort(certificates);

    for cert in sorted {
        let prereqs_met = cert.prerequisites.iter().all(|p| proven.contains(p));

        match &cert.result {
            ProofResult::Full { value, threshold } if prereqs_met && value >= threshold => {
                proven.insert(cert.id.clone());
                for implied in &cert.implies {
                    proven.insert(implied.clone());
                }
            }
            ProofResult::Conditional {
                value, threshold, ..
            } if prereqs_met && value >= threshold => {
                proven.insert(cert.id.clone());
                conditional.insert(cert.id.clone());
                for implied in &cert.implies {
                    conditional.insert(implied.clone());
                }
            }
            ProofResult::Partial {
                proved: sub_proved, ..
            } => {
                for p in sub_proved {
                    proven.insert(format!("{}.{}", cert.id, p));
                }
            }
            _ => {
                failed.push(cert.id.clone());
            }
        }
    }

    ProofChainResult {
        proven,
        failed,
        conditional,
    }
}

/// Topological sort of certificates by prerequisite dependencies (Kahn's algorithm).
fn topological_sort(certificates: &[ProofCertificate]) -> Vec<&ProofCertificate> {
    let id_to_idx: HashMap<&str, usize> = certificates
        .iter()
        .enumerate()
        .map(|(i, c)| (c.id.as_str(), i))
        .collect();

    let mut in_degree: Vec<usize> = vec![0; certificates.len()];
    let mut adj: Vec<Vec<usize>> = (0..certificates.len())
        .map(|_| Vec::with_capacity(2))
        .collect();

    for (i, cert) in certificates.iter().enumerate() {
        for prereq in &cert.prerequisites {
            if let Some(&j) = id_to_idx.get(prereq.as_str()) {
                adj[j].push(i);
                in_degree[i] += 1;
            }
        }
    }

    let mut queue: VecDeque<usize> = (0..certificates.len())
        .filter(|&i| in_degree[i] == 0)
        .collect();

    let mut result = Vec::with_capacity(certificates.len());
    let mut result_indices = Vec::with_capacity(certificates.len());
    while let Some(i) = queue.pop_front() {
        result.push(&certificates[i]);
        result_indices.push(i);
        for &j in &adj[i] {
            in_degree[j] -= 1;
            if in_degree[j] == 0 {
                queue.push_back(j);
            }
        }
    }

    // Add any remaining (cycle) items — O(n) with bool vec instead of HashSet
    let mut added = vec![false; certificates.len()];
    for &idx in &result_indices {
        added[idx] = true;
    }
    for (i, cert) in certificates.iter().enumerate() {
        if !added[i] {
            result.push(cert);
        }
    }

    result
}
