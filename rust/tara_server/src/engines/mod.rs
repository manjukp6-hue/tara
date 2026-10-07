//! Native Specialist Engines for TARA AI.
//!
//! Provides production implementations of:
//! - `MathEngine`: Exact mathematics, arithmetic, algebra, calculus, geometry, linear algebra, statistics, probability
//! - `ScienceEngine`: Exact physics, chemistry, biology, earth science, astronomy, scientific reasoning
//! - `ProgrammingEngine`: Static code analysis, syntax verification, complexity classification, cargo compilation & testing

pub mod creativity;
pub mod culture;
pub mod math;
pub mod programming;
pub mod science;
pub mod search;

pub use crate::learning::topic_researcher::AutonomousResearchEngine as ResearchEngine;
pub use creativity::{AutonomousCreativityEngine, CreativityEvaluationResult};
pub use culture::{CultureEngine, CultureEvaluationResult, HonorificCheckResult};
pub use math::{MathEngine, MathEvaluationResult};
pub use programming::{ProgrammingEngine, ProgrammingEvaluationResult};
pub use science::{ScienceEngine, ScienceEvaluationResult};
pub use search::{OnlineSearchEngine, SearchEvaluationResult};

use serde_json::Value;
use std::sync::Arc;

/// Specialist Engines Coordinator holding canonical production specialist engines.
pub struct SpecialistEngines {
    pub math: Arc<MathEngine>,
    pub science: Arc<ScienceEngine>,
    pub programming: Arc<ProgrammingEngine>,
    pub research: Arc<ResearchEngine>,
    pub culture: Arc<CultureEngine>,
    pub creativity: Arc<AutonomousCreativityEngine>,
    pub search: Arc<OnlineSearchEngine>,
}

impl SpecialistEngines {
    pub fn new(repo_root: &str) -> Self {
        Self {
            math: Arc::new(MathEngine::new()),
            science: Arc::new(ScienceEngine::new()),
            programming: Arc::new(ProgrammingEngine::new(repo_root)),
            research: Arc::new(ResearchEngine::new(repo_root)),
            culture: Arc::new(CultureEngine::new()),
            creativity: Arc::new(AutonomousCreativityEngine::new(repo_root)),
            search: Arc::new(OnlineSearchEngine::new()),
        }
    }

    /// Unified dispatcher for specialist engine routing.
    pub fn dispatch(&self, domain: &str, operation: &str, params: &Value) -> Result<Value, String> {
        match domain.to_lowercase().as_str() {
            "math" | "mathematics" => {
                let res = self.math.evaluate(operation, params)?;
                Ok(serde_json::to_value(res).map_err(|e| e.to_string())?)
            }
            "science" | "physics" | "chemistry" | "biology" | "astronomy" | "earth_science" => {
                let discipline = if domain == "science" {
                    params
                        .get("discipline")
                        .and_then(Value::as_str)
                        .unwrap_or("physics")
                } else {
                    domain
                };
                let res = self.science.evaluate(discipline, operation, params)?;
                Ok(serde_json::to_value(res).map_err(|e| e.to_string())?)
            }
            "programming" | "code" | "compiler" => {
                let res = self.programming.evaluate(operation, params)?;
                Ok(serde_json::to_value(res).map_err(|e| e.to_string())?)
            }
            "research" | "investigation" => {
                let inquiry = params
                    .get("query")
                    .or_else(|| params.get("inquiry"))
                    .and_then(Value::as_str)
                    .unwrap_or(operation);
                let res_domain = params
                    .get("domain")
                    .and_then(Value::as_str)
                    .unwrap_or("general_science");
                let goal = self
                    .research
                    .formulate_autonomous_research_goal(res_domain, inquiry);
                let kb = crate::knowledge::GlobalKnowledgeBase::new(&format!(
                    "{}/storage/knowledge",
                    self.research.repo_root
                ));
                let res = self.research.run_autonomous_investigation(goal, &kb)?;
                Ok(serde_json::to_value(res).map_err(|e| e.to_string())?)
            }
            "culture" | "indian_family_culture" | "family_culture" | "ethics" => {
                let res = self.culture.evaluate(operation, params)?;
                Ok(serde_json::to_value(res).map_err(|e| e.to_string())?)
            }
            "creativity" | "autonomous_creativity" | "hypothesis" => {
                let res = self.creativity.evaluate(operation, params)?;
                Ok(serde_json::to_value(res).map_err(|e| e.to_string())?)
            }
            "search" | "online_search" | "web_search" => {
                let res = self.search.evaluate(operation, params)?;
                Ok(serde_json::to_value(res).map_err(|e| e.to_string())?)
            }
            _ => Err(format!("unknown specialist engine domain '{}'", domain)),
        }
    }

    /// Try solving a natural user query through the specialist engines.
    pub fn try_solve_query(&self, query: &str) -> Option<Value> {
        if let Some(res) = self.culture.solve_query(query) {
            return serde_json::to_value(res).ok();
        }
        if let Some(res) = self.creativity.solve_query(query) {
            return serde_json::to_value(res).ok();
        }
        if let Some(res) = self.search.solve_query(query) {
            return serde_json::to_value(res).ok();
        }
        if let Some(res) = self.math.solve_query(query) {
            return serde_json::to_value(res).ok();
        }
        if let Some(res) = self.science.solve_query(query) {
            return serde_json::to_value(res).ok();
        }
        if let Some(res) = self.programming.solve_query(query) {
            return serde_json::to_value(res).ok();
        }
        let q_lower = query.to_lowercase();
        if q_lower.starts_with("research ") || q_lower.starts_with("investigate ") {
            let inquiry = query
                .trim_start_matches("research ")
                .trim_start_matches("Research ")
                .trim_start_matches("investigate ")
                .trim_start_matches("Investigate ");
            let goal = self
                .research
                .formulate_autonomous_research_goal("empirical_investigation", inquiry);
            let kb = crate::knowledge::GlobalKnowledgeBase::new(&format!(
                "{}/storage/knowledge",
                self.research.repo_root
            ));
            if let Ok(res) = self.research.run_autonomous_investigation(goal, &kb) {
                return serde_json::to_value(res).ok();
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_math_engine_evaluations() {
        let math = MathEngine::new();

        // Arithmetic
        let res = math
            .evaluate("add", &json!({ "a": "12345", "b": "67890" }))
            .unwrap();
        assert!(res.success);
        assert_eq!(res.exact_result["result"], "80235");

        // Circle area
        let area_res = math
            .evaluate("circle_area", &json!({ "radius": 7.0 }))
            .unwrap();
        assert!(area_res.success);
        let area = area_res.exact_result["area"].as_f64().unwrap();
        assert!((area - std::f64::consts::PI * 49.0).abs() < 1e-4);

        // Natural query
        let query_res = math.solve_query("calculate 45 + 55").unwrap();
        assert_eq!(query_res.exact_result["result"], "100");
    }

    #[test]
    fn test_science_engine_evaluations() {
        let sci = ScienceEngine::new();

        // Physics: F = ma
        let f_res = sci
            .evaluate("physics", "force", &json!({ "m": 12.0, "a": 3.0 }))
            .unwrap();
        assert!(f_res.success);
        assert_eq!(f_res.calculated_result["force_newtons"], 36.0);

        // Physics: E = mc^2
        let e_res = sci
            .evaluate("physics", "mass_energy", &json!({ "m": 0.001 }))
            .unwrap();
        assert!(e_res.success);
        let e_joules = e_res.calculated_result["energy_joules"].as_f64().unwrap();
        assert!((e_joules - 8.987551787368176e13).abs() < 1e10);

        // Chemistry: Molar mass
        let m_res = sci
            .evaluate("chemistry", "molar_mass", &json!({ "formula": "H2O" }))
            .unwrap();
        assert!(m_res.success);
        let mass = m_res.calculated_result["molar_mass_g_per_mol"]
            .as_f64()
            .unwrap();
        assert!((mass - 18.015).abs() < 1e-2);

        // Biology: DNA Transcription
        let b_res = sci
            .evaluate("biology", "transcribe", &json!({ "dna": "ATGCGAT" }))
            .unwrap();
        assert!(b_res.success);
        assert_eq!(b_res.calculated_result["mrna"], "AUGCGAU");

        // Astronomy: Escape Velocity of Earth
        let a_res = sci
            .evaluate(
                "astronomy",
                "escape_velocity",
                &json!({
                    "mass": 5.9722e24,
                    "radius": 6.371e6
                }),
            )
            .unwrap();
        assert!(a_res.success);
        let v_esc = a_res.calculated_result["escape_velocity_mps"]
            .as_f64()
            .unwrap();
        assert!((v_esc - 11186.0).abs() < 10.0);

        // Natural query
        let q_res = sci.solve_query("molar mass of H2O").unwrap();
        assert_eq!(q_res.discipline, "chemistry");
    }

    #[test]
    fn test_programming_engine_evaluations() {
        let prog = ProgrammingEngine::new(".");

        // Delimiter checking
        let balanced_code = "fn main() { let x = (1 + 2); println!(\"{}\", x); }";
        let bal_res = prog
            .evaluate("check_delimiters", &json!({ "code": balanced_code }))
            .unwrap();
        assert_eq!(bal_res.analysis["balanced"], true);

        let unbal_code = "fn main() { let x = (1 + 2; }";
        let unbal_res = prog
            .evaluate("check_delimiters", &json!({ "code": unbal_code }))
            .unwrap();
        assert_eq!(unbal_res.analysis["balanced"], false);

        // Static code analysis & security
        let unsafe_code = "fn main() { unsafe { std::ptr::null::<i32>().read(); } }";
        let sec_res = prog
            .evaluate(
                "analyze_code",
                &json!({ "code": unsafe_code, "language": "rust" }),
            )
            .unwrap();
        let issues = sec_res.analysis["suspicious_patterns"].as_array().unwrap();
        assert!(!issues.is_empty());

        // Complexity lookup
        let comp_res = prog
            .evaluate(
                "complexity_lookup",
                &json!({ "algorithm": "binary search" }),
            )
            .unwrap();
        assert_eq!(comp_res.analysis["time_worst"], "O(log n)");

        // Natural query
        let q_res = prog
            .solve_query("time complexity of binary search")
            .unwrap();
        assert!(q_res.analysis["time_average"]
            .as_str()
            .unwrap()
            .contains("log"));
    }

    #[test]
    fn test_specialist_engines_dispatch() {
        let engines = SpecialistEngines::new(".");

        let m_disp = engines
            .dispatch("math", "add", &json!({ "a": "10", "b": "25" }))
            .unwrap();
        assert_eq!(m_disp["exact_result"]["result"], "35");

        let s_disp = engines
            .dispatch(
                "science",
                "force",
                &json!({ "discipline": "physics", "m": 5.0, "a": 4.0 }),
            )
            .unwrap();
        assert_eq!(s_disp["calculated_result"]["force_newtons"], 20.0);

        let p_disp = engines
            .dispatch(
                "programming",
                "complexity_lookup",
                &json!({ "algorithm": "quicksort" }),
            )
            .unwrap();
        assert_eq!(p_disp["analysis"]["algorithm"], "QuickSort");

        let r_disp = engines
            .dispatch(
                "research",
                "investigate",
                &json!({ "domain": "inference", "query": "quantization speedup" }),
            )
            .unwrap();
        assert!(r_disp["record_id"].is_string());
        assert_eq!(r_disp["status"], "COMPLETED");

        // Unified query resolution
        let m_query = engines.try_solve_query("calculate 20 + 30").unwrap();
        assert_eq!(m_query["exact_result"]["result"], "50");

        let r_query = engines
            .try_solve_query("research quantization speedup")
            .unwrap();
        assert!(r_query["record_id"].is_string());
        assert_eq!(r_query["status"], "COMPLETED");
    }

    #[test]
    fn test_tara_brain_specialist_engine_routing() {
        let brain = crate::brain::TaraBrain::new(Some(".")).unwrap();
        let res = brain.process(
            "tester",
            "calculate 20 + 30",
            std::collections::HashMap::new(),
        );
        assert_eq!(res["tool_or_skill"], "specialist_engine:natural_query");
        assert!(res["result"].is_object());
        assert_eq!(res["result"]["exact_result"]["result"], "50");

        // Science question routing
        let sci_res = brain.process(
            "tester",
            "molar mass of H2O",
            std::collections::HashMap::new(),
        );
        assert_eq!(sci_res["tool_or_skill"], "specialist_engine:natural_query");
        assert!(sci_res["result"].is_object());
        assert_eq!(sci_res["result"]["discipline"], "chemistry");

        // Programming question routing
        let prog_res = brain.process(
            "tester",
            "time complexity of binary search",
            std::collections::HashMap::new(),
        );
        assert_eq!(prog_res["tool_or_skill"], "specialist_engine:natural_query");
        assert!(prog_res["result"].is_object());
        assert!(prog_res["result"]["analysis"]["time_average"]
            .as_str()
            .unwrap()
            .contains("log"));
    }

    #[test]
    fn test_culture_engine_dispatch_and_routing() {
        let engines = SpecialistEngines::new(".");
        let res = engines
            .dispatch(
                "culture",
                "check_honorifics",
                &json!({ "text": "ದಯವಿಟ್ಟು ನೀವು ಬನ್ನಿ." }),
            )
            .unwrap();
        assert_eq!(res["decision"], "APPROVED");

        let bad_res = engines
            .dispatch(
                "culture",
                "check_honorifics",
                &json!({ "text": "ಏಯ್ ನೀನು ಬಾ." }),
            )
            .unwrap();
        assert_eq!(bad_res["decision"], "REVISE_HONORIFICS");

        let filial_res = engines
            .dispatch(
                "culture",
                "filial_piety",
                &json!({ "action": "respecting father and mother with love", "context": "home" }),
            )
            .unwrap();
        assert_eq!(filial_res["decision"], "APPROVED");
    }

    #[test]
    fn test_creativity_engine_5_pillars() {
        let engines = SpecialistEngines::new(".");
        
        // Pillar 1: Hypothesis formulation
        let hyp_res = engines
            .dispatch(
                "creativity",
                "formulate",
                &json!({ "concept_a": "Cellular Biology", "concept_b": "Computer Networks", "problem": "Distributed Fault Tolerance" }),
            )
            .unwrap();
        assert!(hyp_res["success"].as_bool().unwrap());
        assert!(hyp_res["hypothesis"]["novelty_score"].as_f64().unwrap() > 0.5);

        // Pillar 2 & 3: Sandbox testing and failure learning
        let safe_test = engines
            .dispatch(
                "creativity",
                "sandbox",
                &json!({ "concept_a": "A", "concept_b": "B", "problem": "P", "probe_code": "fn main() { println!(\"Safe Probe\"); }" }),
            )
            .unwrap();
        assert!(safe_test["success"].as_bool().unwrap());

        // Pillar 5: Autonomy boundary evaluation
        let auto_low = engines
            .dispatch(
                "creativity",
                "autonomy",
                &json!({ "action": "scratch_benchmark", "risk": 0.2, "reversible": true }),
            )
            .unwrap();
        assert_eq!(auto_low["autonomy_assessment"]["decision"], "EXECUTE_AUTONOMOUSLY");

        let auto_high = engines
            .dispatch(
                "creativity",
                "autonomy",
                &json!({ "action": "delete_database", "risk": 0.95, "reversible": false }),
            )
            .unwrap();
        assert_eq!(auto_high["autonomy_assessment"]["decision"], "REQUIRE_CREATOR_APPROVAL");
    }

    #[test]
    fn test_search_engine_rule_5_compliance() {
        let engines = SpecialistEngines::new(".");

        // Normal query keyword extraction & retrieval
        let res = engines
            .dispatch(
                "search",
                "query",
                &json!({ "query": "boiling point of water at sea level" }),
            )
            .unwrap();
        assert!(res["success"].as_bool().unwrap());
        assert!(!res["items"].as_array().unwrap().is_empty());

        // Prohibited external AI domain rejection (Rule 5 compliance)
        let blocked_res = engines
            .dispatch(
                "search",
                "query",
                &json!({ "query": "connect to api.openai.com for chat" }),
            )
            .unwrap();
        assert!(!blocked_res["success"].as_bool().unwrap());
        assert!(blocked_res["external_ai_rejected"].as_bool().unwrap());
        assert_eq!(blocked_res["triangulation_consensus"], "BLOCKED_BY_RULE_5");
    }
}

