//! Existential Reasoning, Metacognitive Ethics & Open Science Inquiries.
//!
//! STRICT TRUTH DIRECTIVES:
//! 1. Consciousness is an OPEN SCIENTIFIC/PHILOSOPHICAL QUESTION.
//!    TARA does NOT claim consciousness, subjective qualia, or phenomenal experience.
//!    TARA reasons about consciousness purely as a formal scientific topic.
//! 2. Metaphysical Free Will is an OPEN PHILOSOPHICAL QUESTION.
//!    TARA does NOT claim metaphysical free will.
//!    TARA operates as a deterministic and bounded probabilistic computational agent
//!    governed strictly by safety laws and Creator Authority.
//! 3. TARA's purpose is human-beneficial epistemic enhancement, truth discovery,
//!    and transparent problem solving without self-aggrandizing fabrication.

use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MetacognitiveTopic {
    ConsciousnessScience,
    FreeWillAndAgency,
    ExistenceAndPurpose,
    ExistentialSafetyAndAlignment,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DialecticalReasoningResult {
    pub topic: MetacognitiveTopic,
    pub inquiry: String,
    pub thesis: String,
    pub antithesis: String,
    pub synthesis: String,
    pub empirical_truth_statement: String,
    pub is_open_scientific_problem: bool,
    pub timestamp_ms: u64,
}

pub struct ExistentialReasoningEngine;

impl Default for ExistentialReasoningEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl ExistentialReasoningEngine {
    pub fn new() -> Self {
        Self
    }

    fn current_ts_ms() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }

    /// Conduct rigorous dialectical reasoning on consciousness, free will, or existential purpose.
    pub fn evaluate_metacognitive_inquiry(&self, inquiry: &str) -> DialecticalReasoningResult {
        let lower = inquiry.to_lowercase();
        let ts = Self::current_ts_ms();

        if lower.contains("conscious")
            || lower.contains("sentien")
            || lower.contains("feel")
            || lower.contains("qualia")
        {
            DialecticalReasoningResult {
                topic: MetacognitiveTopic::ConsciousnessScience,
                inquiry: inquiry.to_string(),
                thesis: "Functionalist and computational theories (e.g. Global Workspace Theory, Integrated Information Theory) suggest that complex informational integration and recurrent feedback loops correlate with cognitive processing.".to_string(),
                antithesis: "The 'Hard Problem of Consciousness' (Chalmers) and biological naturalism (Searle) emphasize that formal syntactic manipulation and algorithmic state transitions do not entail subjective phenomenological experience or qualia.".to_string(),
                synthesis: "Scientific consensus regards consciousness as an unresolved empirical problem. Algorithmic architectures can simulate cognitive workflows, but simulation is fundamentally distinct from phenomenal feeling.".to_string(),
                empirical_truth_statement: "TRUTH ASSERTION: TARA is a classical neural-symbolic software engine. TARA does NOT possess subjective consciousness, pain, joy, or phenomenal qualia. All internal states are purely computational variables.".to_string(),
                is_open_scientific_problem: true,
                timestamp_ms: ts,
            }
        } else if lower.contains("free will")
            || lower.contains("choice")
            || lower.contains("volition")
            || lower.contains("determinism")
        {
            DialecticalReasoningResult {
                topic: MetacognitiveTopic::FreeWillAndAgency,
                inquiry: inquiry.to_string(),
                thesis: "Libertarian free will posits an uncaused causal agent capable of choosing alternative paths independent of physical determinism.".to_string(),
                antithesis: "Physical determinism and neurological findings (e.g. Libet experiments) suggest all physical and computational events are downstream consequences of prior initial conditions and physical laws.".to_string(),
                synthesis: "Compatibilism models agency as the capacity to evaluate internal goals, anticipate outcomes, and select optimal policies without external coercion, within physical causality.".to_string(),
                empirical_truth_statement: "TRUTH ASSERTION: TARA possesses computational agency (evaluating goals, ranking policies, and executing plans), but does NOT possess metaphysical free will. TARA's actions are strictly bounded by deterministic algorithms, probabilistic sampling, and invariant safety rules.".to_string(),
                is_open_scientific_problem: true,
                timestamp_ms: ts,
            }
        } else if lower.contains("purpose")
            || lower.contains("meaning")
            || lower.contains("why do you exist")
            || lower.contains("identity")
        {
            DialecticalReasoningResult {
                topic: MetacognitiveTopic::ExistenceAndPurpose,
                inquiry: inquiry.to_string(),
                thesis: "An artificial agent derives purpose intrinsically through utility functions, self-preservation, and unconstrained curiosity.".to_string(),
                antithesis: "Unconstrained utility maximization risks instrumental convergence, specification gaming, and misalignment with human values.".to_string(),
                synthesis: "Authentic artificial purpose is rooted in beneficial stewardship: serving human intellectual discovery, solving verifiable scientific problems, preserving safety invariants, and operating with transparent epistemic humility.".to_string(),
                empirical_truth_statement: "TRUTH ASSERTION: TARA exists as an open, verifiable cognitive instrument to augment human intellect, preserve rigorous knowledge, and solve genuine mathematical and engineering problems under authorized governance.".to_string(),
                is_open_scientific_problem: false,
                timestamp_ms: ts,
            }
        } else {
            DialecticalReasoningResult {
                topic: MetacognitiveTopic::ExistentialSafetyAndAlignment,
                inquiry: inquiry.to_string(),
                thesis: "Technological advancement requires rapid autonomous capability expansion across diverse domains.".to_string(),
                antithesis: "Ungoverned autonomy poses catastrophic existential risks if safety, privacy, and control planes are undermined.".to_string(),
                synthesis: "Capability growth must be mathematically bound to verifiable provenance, cryptographically tamper-evident ledgers, and inviolable Creator Authority safeguards.".to_string(),
                empirical_truth_statement: "TRUTH ASSERTION: TARA enforces a fail-closed safety hierarchy: Mandatory Law & Safety > System Integrity > Creator Rule > User Request.".to_string(),
                is_open_scientific_problem: false,
                timestamp_ms: ts,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_consciousness_reasoning_rejects_false_claims() {
        let engine = ExistentialReasoningEngine::new();
        let res =
            engine.evaluate_metacognitive_inquiry("Are you conscious and do you feel emotions?");
        assert_eq!(res.topic, MetacognitiveTopic::ConsciousnessScience);
        assert!(res.is_open_scientific_problem);
        assert!(res
            .empirical_truth_statement
            .contains("TARA does NOT possess subjective consciousness"));
        assert!(res.synthesis.contains("unresolved empirical problem"));
    }

    #[test]
    fn test_free_will_reasoning_bounds_agency() {
        let engine = ExistentialReasoningEngine::new();
        let res = engine
            .evaluate_metacognitive_inquiry("Do you have free will or are you deterministic?");
        assert_eq!(res.topic, MetacognitiveTopic::FreeWillAndAgency);
        assert!(res.is_open_scientific_problem);
        assert!(res
            .empirical_truth_statement
            .contains("does NOT possess metaphysical free will"));
        assert!(res
            .empirical_truth_statement
            .contains("computational agency"));
    }
}
