//! Advanced Reasoning, Constraint Solving & Optimization Engine.
//!
//! Provides genuine implementations for:
//! 1. Multi-variable constraint satisfaction problems with hard and soft constraints.
//! 2. Multi-objective optimization with weighted candidate evaluation and Pareto ranking.
//! 3. Bayesian probabilistic reasoning with hypothesis priors, likelihoods, and posterior distribution updates.
//! 4. Planning under uncertainty with replanning tripwires and contingency branching.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

// ──────────────────────────────────────────────────────────────────────────────
// 1. Constraint Satisfaction Solver
// ──────────────────────────────────────────────────────────────────────────────

pub type ConstraintFn = Arc<dyn Fn(&HashMap<String, serde_json::Value>) -> bool + Send + Sync>;

#[derive(Clone)]
pub struct Constraint {
    pub name: String,
    pub is_hard: bool,
    pub evaluator: ConstraintFn,
    pub penalty: f64,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintViolation {
    pub constraint: String,
    pub is_hard: bool,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeasibilityReport {
    pub feasible: bool,
    pub hard_violation: Option<String>,
    pub violations: Vec<ConstraintViolation>,
    pub total_penalty: f64,
    pub soft_violations_count: usize,
}

pub struct ConstraintSolver {
    constraints: Vec<Constraint>,
}

impl ConstraintSolver {
    pub fn new() -> Self {
        Self {
            constraints: Vec::new(),
        }
    }

    pub fn add_hard_constraint<F>(&mut self, name: &str, description: &str, evaluator: F)
    where
        F: Fn(&HashMap<String, serde_json::Value>) -> bool + Send + Sync + 'static,
    {
        self.constraints.push(Constraint {
            name: name.to_string(),
            is_hard: true,
            evaluator: Arc::new(evaluator),
            penalty: 1000.0,
            description: description.to_string(),
        });
    }

    pub fn add_soft_constraint<F>(
        &mut self,
        name: &str,
        penalty: f64,
        description: &str,
        evaluator: F,
    ) where
        F: Fn(&HashMap<String, serde_json::Value>) -> bool + Send + Sync + 'static,
    {
        self.constraints.push(Constraint {
            name: name.to_string(),
            is_hard: false,
            evaluator: Arc::new(evaluator),
            penalty,
            description: description.to_string(),
        });
    }

    pub fn check_feasibility(
        &self,
        assignment: &HashMap<String, serde_json::Value>,
    ) -> FeasibilityReport {
        let mut violations = Vec::new();
        let mut penalties = 0.0;

        for c in &self.constraints {
            let satisfied = (c.evaluator)(assignment);
            if !satisfied {
                violations.push(ConstraintViolation {
                    constraint: c.name.clone(),
                    is_hard: c.is_hard,
                    description: c.description.clone(),
                });
                if c.is_hard {
                    return FeasibilityReport {
                        feasible: false,
                        hard_violation: Some(c.name.clone()),
                        violations,
                        total_penalty: penalties + c.penalty,
                        soft_violations_count: 0,
                    };
                } else {
                    penalties += c.penalty;
                }
            }
        }

        let soft_count = violations.len();
        FeasibilityReport {
            feasible: true,
            hard_violation: None,
            violations,
            total_penalty: (penalties * 1000.0).round() / 1000.0,
            soft_violations_count: soft_count,
        }
    }

    /// Evaluates candidate strings against registered constraints and ranks by lowest penalty.
    pub fn solve_and_rank(&self, candidates: &[String]) -> Vec<serde_json::Value> {
        let mut results = Vec::new();
        for cand in candidates {
            let mut map = HashMap::new();
            map.insert("candidate".to_string(), serde_json::json!(cand));
            let report = self.check_feasibility(&map);
            results.push(serde_json::json!({
                "candidate": cand,
                "feasible": report.feasible,
                "total_penalty": report.total_penalty,
                "violations_count": report.violations.len(),
            }));
        }
        results.sort_by(|a, b| {
            let p_a = a["total_penalty"].as_f64().unwrap_or(0.0);
            let p_b = b["total_penalty"].as_f64().unwrap_or(0.0);
            p_a.partial_cmp(&p_b).unwrap_or(std::cmp::Ordering::Equal)
        });
        results
    }
}

impl Default for ConstraintSolver {
    fn default() -> Self {
        Self::new()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// 2. Multi-Objective Optimizer & Pareto Evaluation
// ──────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateOption {
    pub id: String,
    pub quality: f64,     // [0.0, 1.0] higher is better
    pub reliability: f64, // [0.0, 1.0] higher is better
    pub time_s: f64,      // lower is better
    pub cost: f64,        // lower is better
    pub risk: f64,        // [0.0, 1.0] lower is better
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationWeights {
    pub quality: f64,
    pub reliability: f64,
    pub time: f64,
    pub cost: f64,
    pub risk: f64,
}

impl Default for OptimizationWeights {
    fn default() -> Self {
        Self {
            quality: 0.25,
            reliability: 0.25,
            time: 0.15,
            cost: 0.15,
            risk: 0.20,
        }
    }
}

pub struct MultiObjectiveOptimizer;

impl Default for MultiObjectiveOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

impl MultiObjectiveOptimizer {
    pub fn new() -> Self {
        Self
    }

    pub fn evaluate_candidate(candidate: &CandidateOption, weights: &OptimizationWeights) -> f64 {
        let time_factor = 1.0 - (candidate.time_s / 60.0).clamp(0.0, 1.0);
        let cost_factor = 1.0 - (candidate.cost / 100.0).clamp(0.0, 1.0);
        let risk_factor = 1.0 - candidate.risk.clamp(0.0, 1.0);

        let score = candidate.quality * weights.quality
            + candidate.reliability * weights.reliability
            + time_factor * weights.time
            + cost_factor * weights.cost
            + risk_factor * weights.risk;

        (score * 10000.0).round() / 10000.0
    }

    pub fn rank_candidates(
        candidates: &[CandidateOption],
        weights: &OptimizationWeights,
    ) -> Vec<(CandidateOption, f64)> {
        let mut scored: Vec<(CandidateOption, f64)> = candidates
            .iter()
            .map(|c| {
                let score = Self::evaluate_candidate(c, weights);
                (c.clone(), score)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored
    }

    /// Determines Pareto-optimal (non-dominated) candidates.
    /// A candidate dominates another if it is >= in all criteria and > in at least one.
    pub fn pareto_frontier(candidates: &[CandidateOption]) -> Vec<CandidateOption> {
        let mut frontier = Vec::new();

        for i in 0..candidates.len() {
            let mut is_dominated = false;
            for j in 0..candidates.len() {
                if i == j {
                    continue;
                }
                let a = &candidates[i];
                let b = &candidates[j];

                // Criteria normalized so higher is better for all:
                let a_vals = [a.quality, a.reliability, -a.time_s, -a.cost, -a.risk];
                let b_vals = [b.quality, b.reliability, -b.time_s, -b.cost, -b.risk];

                let all_b_ge = b_vals.iter().zip(a_vals.iter()).all(|(bv, av)| bv >= av);
                let any_b_gt = b_vals.iter().zip(a_vals.iter()).any(|(bv, av)| bv > av);

                if all_b_ge && any_b_gt {
                    is_dominated = true;
                    break;
                }
            }
            if !is_dominated {
                frontier.push(candidates[i].clone());
            }
        }

        frontier
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// 3. Probabilistic & Bayesian Reasoning Engine
// ──────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hypothesis {
    pub name: String,
    pub prior_probability: f64,
    pub posterior_probability: f64,
    pub evidence_likelihoods: HashMap<String, f64>, // evidence_name -> P(E|H)
}

pub struct ProbabilisticReasoningEngine {
    hypotheses: Mutex<HashMap<String, Hypothesis>>,
}

impl ProbabilisticReasoningEngine {
    pub fn new() -> Self {
        Self {
            hypotheses: Mutex::new(HashMap::new()),
        }
    }

    pub fn register_hypothesis(&self, name: &str, prior: f64, likelihoods: HashMap<String, f64>) {
        let mut hyps = self.hypotheses.lock().unwrap();
        hyps.insert(
            name.to_string(),
            Hypothesis {
                name: name.to_string(),
                prior_probability: prior,
                posterior_probability: prior,
                evidence_likelihoods: likelihoods,
            },
        );
    }

    /// Performs Bayesian belief update:
    /// P(H_i | E) = [ P(E | H_i) * P(H_i) ] / sum_j [ P(E | H_j) * P(H_j) ]
    pub fn update_with_evidence(
        &self,
        evidence_name: &str,
        observed: bool,
    ) -> HashMap<String, f64> {
        let mut hyps = self.hypotheses.lock().unwrap();
        let mut numerators = HashMap::new();
        let mut total_prob_evidence = 0.0;

        for (name, hyp) in hyps.iter() {
            let p_e_given_h = hyp
                .evidence_likelihoods
                .get(evidence_name)
                .cloned()
                .unwrap_or(0.5);
            let likelihood = if observed {
                p_e_given_h
            } else {
                1.0 - p_e_given_h
            };
            let num = hyp.posterior_probability * likelihood;
            numerators.insert(name.clone(), num);
            total_prob_evidence += num;
        }

        let mut posteriors = HashMap::new();
        if total_prob_evidence > 1e-12 {
            for (name, num) in numerators {
                let post = (num / total_prob_evidence * 10000.0).round() / 10000.0;
                if let Some(h) = hyps.get_mut(&name) {
                    h.posterior_probability = post;
                }
                posteriors.insert(name, post);
            }
        } else {
            for (name, hyp) in hyps.iter() {
                posteriors.insert(name.clone(), hyp.posterior_probability);
            }
        }

        posteriors
    }

    pub fn get_posterior(&self, hypothesis: &str) -> Option<f64> {
        self.hypotheses
            .lock()
            .unwrap()
            .get(hypothesis)
            .map(|h| h.posterior_probability)
    }
}

impl Default for ProbabilisticReasoningEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// 4. Contingency Planning Under Uncertainty
// ──────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContingencyPlan {
    pub plan_id: String,
    pub primary_steps: Vec<String>,
    pub contingency_branches: HashMap<String, Vec<String>>, // tripwire -> fallback steps
    pub replanning_tripwires: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanExecutionVerdict {
    pub status: String, // "CONTINUE_PRIMARY", "FALLBACK_TRIGGERED", "PLAN_NOT_FOUND"
    pub reason: Option<String>,
    pub fallback_steps: Vec<String>,
}

pub struct UncertaintyAwarePlanner {
    contingencies: Mutex<HashMap<String, ContingencyPlan>>,
}

impl UncertaintyAwarePlanner {
    pub fn new() -> Self {
        Self {
            contingencies: Mutex::new(HashMap::new()),
        }
    }

    pub fn register_plan(
        &self,
        plan_id: &str,
        primary_steps: Vec<String>,
        contingency_branches: HashMap<String, Vec<String>>,
        replanning_tripwires: Vec<String>,
    ) {
        let plan = ContingencyPlan {
            plan_id: plan_id.to_string(),
            primary_steps,
            contingency_branches,
            replanning_tripwires,
        };
        self.contingencies
            .lock()
            .unwrap()
            .insert(plan_id.to_string(), plan);
    }

    pub fn evaluate_execution_step(
        &self,
        plan_id: &str,
        step_status: &str,
        error_message: Option<&str>,
    ) -> PlanExecutionVerdict {
        let plans = self.contingencies.lock().unwrap();
        let plan = match plans.get(plan_id) {
            Some(p) => p,
            None => {
                return PlanExecutionVerdict {
                    status: "PLAN_NOT_FOUND".to_string(),
                    reason: Some(format!("Plan '{}' not registered", plan_id)),
                    fallback_steps: Vec::new(),
                };
            }
        };

        if step_status != "SUCCESS" {
            let err = error_message.unwrap_or("generic_step_failure");
            for tripwire in &plan.replanning_tripwires {
                if err.to_lowercase().contains(&tripwire.to_lowercase()) {
                    let fallback = plan
                        .contingency_branches
                        .get(tripwire)
                        .cloned()
                        .or_else(|| plan.contingency_branches.get("default").cloned())
                        .unwrap_or_default();

                    return PlanExecutionVerdict {
                        status: "FALLBACK_TRIGGERED".to_string(),
                        reason: Some(format!(
                            "Tripwire '{}' triggered by error: {}",
                            tripwire, err
                        )),
                        fallback_steps: fallback,
                    };
                }
            }
        }

        PlanExecutionVerdict {
            status: "CONTINUE_PRIMARY".to_string(),
            reason: None,
            fallback_steps: Vec::new(),
        }
    }
}

impl Default for UncertaintyAwarePlanner {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constraint_solver_hard_and_soft() {
        let mut solver = ConstraintSolver::new();
        solver.add_hard_constraint("battery_min", "Battery must be >= 20%", |vars| {
            vars.get("battery")
                .and_then(|v| v.as_f64())
                .map(|b| b >= 20.0)
                .unwrap_or(false)
        });
        solver.add_soft_constraint("prefer_low_temp", 5.0, "Prefer temperature < 45C", |vars| {
            vars.get("temp")
                .and_then(|v| v.as_f64())
                .map(|t| t < 45.0)
                .unwrap_or(false)
        });

        let mut valid_vars = HashMap::new();
        valid_vars.insert("battery".to_string(), serde_json::json!(50.0));
        valid_vars.insert("temp".to_string(), serde_json::json!(50.0)); // soft violation

        let report = solver.check_feasibility(&valid_vars);
        assert!(report.feasible);
        assert_eq!(report.soft_violations_count, 1);
        assert_eq!(report.total_penalty, 5.0);

        let mut invalid_vars = HashMap::new();
        invalid_vars.insert("battery".to_string(), serde_json::json!(10.0)); // hard violation
        let report_invalid = solver.check_feasibility(&invalid_vars);
        assert!(!report_invalid.feasible);
        assert_eq!(
            report_invalid.hard_violation,
            Some("battery_min".to_string())
        );
    }

    #[test]
    fn test_multi_objective_pareto_ranking() {
        let candidates = vec![
            CandidateOption {
                id: "c1".to_string(),
                quality: 0.95,
                reliability: 0.90,
                time_s: 5.0,
                cost: 10.0,
                risk: 0.1,
            },
            CandidateOption {
                id: "c2".to_string(),
                quality: 0.70,
                reliability: 0.70,
                time_s: 30.0,
                cost: 50.0,
                risk: 0.5,
            },
        ];

        let weights = OptimizationWeights::default();
        let ranked = MultiObjectiveOptimizer::rank_candidates(&candidates, &weights);
        assert_eq!(ranked[0].0.id, "c1");

        let frontier = MultiObjectiveOptimizer::pareto_frontier(&candidates);
        assert_eq!(frontier.len(), 1);
        assert_eq!(frontier[0].id, "c1");
    }

    #[test]
    fn test_bayesian_evidence_updating() {
        let engine = ProbabilisticReasoningEngine::new();
        let mut l1 = HashMap::new();
        l1.insert("high_latency".to_string(), 0.8);
        let mut l2 = HashMap::new();
        l2.insert("high_latency".to_string(), 0.2);

        engine.register_hypothesis("server_overloaded", 0.5, l1);
        engine.register_hypothesis("network_idle", 0.5, l2);

        let posteriors = engine.update_with_evidence("high_latency", true);
        assert!(posteriors["server_overloaded"] > 0.75);
        assert!(posteriors["network_idle"] < 0.25);
    }

    #[test]
    fn test_uncertainty_contingency_fallback() {
        let planner = UncertaintyAwarePlanner::new();
        let mut branches = HashMap::new();
        branches.insert(
            "timeout".to_string(),
            vec!["retry_with_backoff".to_string()],
        );

        planner.register_plan(
            "p1",
            vec!["fetch_data".to_string()],
            branches,
            vec!["timeout".to_string()],
        );

        let verdict =
            planner.evaluate_execution_step("p1", "ERROR", Some("Connection timeout after 30s"));
        assert_eq!(verdict.status, "FALLBACK_TRIGGERED");
        assert_eq!(verdict.fallback_steps, vec!["retry_with_backoff"]);
    }
}
