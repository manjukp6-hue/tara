"""
python/tara_core/reasoning_engine_advanced.py

Advanced Reasoning, Constraint Solving & Optimization Engine for TARA Core.
Provides production-grade implementations for:
1. Constraint Solving (hard/soft constraints, variable domains, conflict detection)
2. Multi-Objective Optimization (Pareto optimality, trade-off scoring)
3. Probabilistic & Bayesian Reasoning (hypotheses, evidence updates, competing explanations)
4. Planning Under Uncertainty (contingency branching, fallback paths, replanning tripwires)
5. Value & Utility Modeling (multi-attribute decision analysis subject to safety policy)
"""

import math
import json
import logging
import threading
from typing import Dict, List, Any, Optional, Callable, Tuple, Set
from dataclasses import dataclass, field

logger = logging.getLogger("TARA.AdvancedReasoning")


# ----------------------------------------------------------------------------
# 11. Constraint Solving
# ----------------------------------------------------------------------------

@dataclass
class Constraint:
    name: str
    is_hard: bool
    evaluator: Callable[[Dict[str, Any]], bool]
    penalty: float = 1.0  # Used if soft constraint violated
    description: str = ""


class ConstraintSolver:
    """Solves multi-variable constraint satisfaction problems with hard and soft constraints."""

    def __init__(self):
        self._constraints: List[Constraint] = []
        self._lock = threading.RLock()

    def add_hard_constraint(self, name: str, evaluator: Callable[[Dict[str, Any]], bool], description: str = ""):
        with self._lock:
            self._constraints.append(Constraint(name=name, is_hard=True, evaluator=evaluator, description=description))

    def add_soft_constraint(self, name: str, evaluator: Callable[[Dict[str, Any]], bool], penalty: float = 1.0, description: str = ""):
        with self._lock:
            self._constraints.append(Constraint(name=name, is_hard=False, evaluator=evaluator, penalty=penalty, description=description))

    def check_feasibility(self, assignment: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            violations = []
            penalties = 0.0
            for c in self._constraints:
                try:
                    satisfied = c.evaluator(assignment)
                    if not satisfied:
                        violations.append({
                            "constraint": c.name,
                            "is_hard": c.is_hard,
                            "description": c.description
                        })
                        if c.is_hard:
                            return {
                                "feasible": False,
                                "hard_violation": c.name,
                                "violations": violations,
                                "total_penalty": penalties
                            }
                        else:
                            penalties += c.penalty
                except Exception as ex:
                    violations.append({"constraint": c.name, "error": str(ex), "is_hard": c.is_hard})
                    if c.is_hard:
                        return {"feasible": False, "hard_violation": c.name, "violations": violations}

            return {
                "feasible": True,
                "violations": violations,
                "total_penalty": round(penalties, 4),
                "soft_violations_count": len(violations)
            }


# ----------------------------------------------------------------------------
# 12. Optimization Engine
# ----------------------------------------------------------------------------

class MultiObjectiveOptimizer:
    """Multi-criteria optimization supporting Pareto optimality and weighted trade-offs."""

    @staticmethod
    def evaluate_candidate(
        candidate: Dict[str, Any],
        weights: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Computes composite utility score where higher is better.
        Metrics normalized to 0.0-1.0:
        quality (+), reliability (+), usefulness (+),
        time (-), cost (-), energy (-), risk (-)
        """
        w = weights or {
            "quality": 0.25,
            "reliability": 0.25,
            "time": 0.15,
            "cost": 0.15,
            "risk": 0.20
        }
        quality = candidate.get("quality", 0.8)
        reliability = candidate.get("reliability", 0.9)
        time_factor = 1.0 - min(1.0, candidate.get("time_s", 10.0) / 60.0)
        cost_factor = 1.0 - min(1.0, candidate.get("cost", 0.0) / 100.0)
        risk_factor = 1.0 - min(1.0, candidate.get("risk", 0.1))

        score = (
            quality * w.get("quality", 0.0) +
            reliability * w.get("reliability", 0.0) +
            time_factor * w.get("time", 0.0) +
            cost_factor * w.get("cost", 0.0) +
            risk_factor * w.get("risk", 0.0)
        )
        return round(score, 4)

    def rank_candidates(self, candidates: List[Dict[str, Any]], weights: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
        scored = []
        for c in candidates:
            score = self.evaluate_candidate(c, weights)
            item = dict(c)
            item["composite_score"] = score
            scored.append(item)
        scored.sort(key=lambda x: x["composite_score"], reverse=True)
        return scored


# ----------------------------------------------------------------------------
# 13. Probabilistic Reasoning
# ----------------------------------------------------------------------------

@dataclass
class Hypothesis:
    name: str
    prior_probability: float
    posterior_probability: float = 0.0
    evidence_likelihoods: Dict[str, float] = field(default_factory=dict) # evidence_name -> P(E|H)


class ProbabilisticReasoningEngine:
    """Manages competing hypotheses, Bayesian evidence updates, and posterior distributions."""

    def __init__(self):
        self._hypotheses: Dict[str, Hypothesis] = {}
        self._lock = threading.RLock()

    def register_hypothesis(self, name: str, prior: float, likelihoods: Optional[Dict[str, float]] = None):
        with self._lock:
            self._hypotheses[name] = Hypothesis(
                name=name,
                prior_probability=prior,
                posterior_probability=prior,
                evidence_likelihoods=likelihoods or {}
            )

    def update_with_evidence(self, evidence_name: str, observed: bool = True) -> Dict[str, float]:
        """Performs Bayesian belief update across all registered hypotheses."""
        with self._lock:
            numerators = {}
            total_prob_evidence = 0.0

            for name, hyp in self._hypotheses.items():
                p_e_given_h = hyp.evidence_likelihoods.get(evidence_name, 0.5)
                likelihood = p_e_given_h if observed else (1.0 - p_e_given_h)
                num = hyp.posterior_probability * likelihood
                numerators[name] = num
                total_prob_evidence += num

            posteriors = {}
            if total_prob_evidence > 0:
                for name, num in numerators.items():
                    post = num / total_prob_evidence
                    self._hypotheses[name].posterior_probability = round(post, 4)
                    posteriors[name] = round(post, 4)
            else:
                posteriors = {name: hyp.posterior_probability for name, hyp in self._hypotheses.items()}

            return posteriors


# ----------------------------------------------------------------------------
# 14. Planning Under Uncertainty
# ----------------------------------------------------------------------------

@dataclass
class ContingencyPlan:
    plan_id: str
    primary_steps: List[str]
    contingency_branches: Dict[str, List[str]] # condition/failure -> fallback steps
    replanning_tripwires: List[str]


class UncertaintyAwarePlanner:
    """Selects plans with contingency branches and monitors uncertainty thresholds."""

    def __init__(self):
        self._contingencies: Dict[str, ContingencyPlan] = {}

    def register_plan(self, plan_id: str, primary: List[str], branches: Dict[str, List[str]], tripwires: List[str]):
        self._contingencies[plan_id] = ContingencyPlan(
            plan_id=plan_id,
            primary_steps=primary,
            contingency_branches=branches,
            replanning_tripwires=tripwires
        )

    def evaluate_execution_step(self, plan_id: str, step_result: Dict[str, Any]) -> Dict[str, Any]:
        plan = self._contingencies.get(plan_id)
        if not plan:
            return {"status": "EXECUTE_PRIMARY", "plan_found": False}

        status = step_result.get("status", "SUCCESS")
        if status != "SUCCESS":
            err = step_result.get("error", "generic_failure")
            for tripwire in plan.replanning_tripwires:
                if tripwire.lower() in err.lower():
                    # Trigger fallback branch
                    fallback = plan.contingency_branches.get(tripwire) or plan.contingency_branches.get("default", [])
                    return {
                        "status": "FALLBACK_TRIGGERED",
                        "reason": f"Tripwire '{tripwire}' hit: {err}",
                        "fallback_steps": fallback
                    }

        return {"status": "CONTINUE_PRIMARY"}


# ----------------------------------------------------------------------------
# 15. Concept Abstraction & Inductive Formation (Capability 3)
# ----------------------------------------------------------------------------

@dataclass
class AbstractConcept:
    concept_id: str
    name: str
    abstraction_level: int
    common_attributes: Dict[str, Any]
    instance_ids: List[str]
    invariants: List[str]
    parent_concept_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "concept_id": self.concept_id,
            "name": self.name,
            "abstraction_level": self.abstraction_level,
            "common_attributes": self.common_attributes,
            "instance_ids": self.instance_ids,
            "invariants": self.invariants,
            "parent_concept_id": self.parent_concept_id
        }


class ConceptAbstractionEngine:
    """
    Synthesizes higher-order abstract schemas and inductive concept hierarchies
    from concrete entity instances and observations.
    """
    def __init__(self):
        self._concepts: Dict[str, AbstractConcept] = {}
        self._lock = threading.RLock()

    def form_concept_from_instances(
        self,
        concept_name: str,
        instances: List[Dict[str, Any]],
        invariance_threshold: float = 0.75,
        parent_concept_id: Optional[str] = None
    ) -> AbstractConcept:
        with self._lock:
            if not instances:
                raise ValueError("Cannot form concept from empty instances")

            attr_counts: Dict[str, Dict[str, int]] = {}
            for inst in instances:
                for k, v in inst.items():
                    if k == "id":
                        continue
                    if k not in attr_counts:
                        attr_counts[k] = {}
                    val_repr = json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else str(v)
                    attr_counts[k][val_repr] = attr_counts[k].get(val_repr, 0) + 1

            total_inst = len(instances)
            common_attrs = {}
            invariants = []

            for k, val_map in attr_counts.items():
                for v_str, count in val_map.items():
                    ratio = count / total_inst
                    if ratio >= invariance_threshold:
                        try:
                            parsed_val = json.loads(v_str)
                        except Exception:
                            parsed_val = v_str
                        common_attrs[k] = parsed_val
                        if ratio == 1.0:
                            invariants.append(f"{k} == {v_str}")

            cid = f"concept_{concept_name.lower().replace(' ', '_')}"
            level = 1 if not parent_concept_id else (self._concepts[parent_concept_id].abstraction_level + 1)
            concept = AbstractConcept(
                concept_id=cid,
                name=concept_name,
                abstraction_level=level,
                common_attributes=common_attrs,
                instance_ids=[str(inst.get("id", i)) for i, inst in enumerate(instances)],
                invariants=invariants,
                parent_concept_id=parent_concept_id
            )
            self._concepts[cid] = concept
            return concept

    def get_concept(self, concept_id: str) -> Optional[AbstractConcept]:
        with self._lock:
            return self._concepts.get(concept_id)

    def generalize_concept(self, child_concept_ids: List[str], general_name: str) -> AbstractConcept:
        with self._lock:
            child_concepts = [self._concepts[cid] for cid in child_concept_ids if cid in self._concepts]
            if not child_concepts:
                raise ValueError("No valid child concepts to generalize")

            shared = dict(child_concepts[0].common_attributes)
            for c in child_concepts[1:]:
                keys_to_del = [k for k, v in shared.items() if k not in c.common_attributes or c.common_attributes[k] != v]
                for k in keys_to_del:
                    del shared[k]

            cid = f"abstract_{general_name.lower().replace(' ', '_')}"
            max_lvl = max(c.abstraction_level for c in child_concepts)
            parent = AbstractConcept(
                concept_id=cid,
                name=general_name,
                abstraction_level=max_lvl + 1,
                common_attributes=shared,
                instance_ids=list({inst for c in child_concepts for inst in c.instance_ids}),
                invariants=[inv for c in child_concepts for inv in c.invariants if all(inv in other.invariants for other in child_concepts)]
            )
            self._concepts[cid] = parent
            for c in child_concepts:
                c.parent_concept_id = cid
            return parent


# ----------------------------------------------------------------------------
# 16. Structural Analogy & Cross-Domain Transfer (Capability 4)
# ----------------------------------------------------------------------------

@dataclass
class RelationalDomain:
    name: str
    entities: List[str]
    relations: List[Tuple[str, str, str]]


@dataclass
class AnalogyMapping:
    source_domain: str
    target_domain: str
    entity_map: Dict[str, str]
    relation_map: Dict[str, str]
    structural_similarity_score: float
    inferred_target_relations: List[Tuple[str, str, str]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_domain": self.source_domain,
            "target_domain": self.target_domain,
            "entity_map": self.entity_map,
            "relation_map": self.relation_map,
            "structural_similarity_score": self.structural_similarity_score,
            "inferred_target_relations": self.inferred_target_relations
        }


class AnalogyReasoningEngine:
    """
    Computes structural mappings and cross-domain relational isomorphisms (Structure-Mapping Engine).
    Transfers solution patterns from a familiar source domain to an unfamiliar target domain.
    """
    def __init__(self):
        self._domains: Dict[str, RelationalDomain] = {}
        self._lock = threading.RLock()

    def register_domain(self, name: str, entities: List[str], relations: List[Tuple[str, str, str]]) -> RelationalDomain:
        with self._lock:
            dom = RelationalDomain(name=name, entities=entities, relations=relations)
            self._domains[name] = dom
            return dom

    def map_analogy(self, source_domain_name: str, target_domain_name: str) -> AnalogyMapping:
        with self._lock:
            s_dom = self._domains.get(source_domain_name)
            t_dom = self._domains.get(target_domain_name)
            if not s_dom or not t_dom:
                raise ValueError(f"Domains '{source_domain_name}' and '{target_domain_name}' must be registered.")

            s_rel_types = {r[1] for r in s_dom.relations}
            t_rel_types = {r[1] for r in t_dom.relations}
            common_rel_types = s_rel_types.intersection(t_rel_types)

            entity_map: Dict[str, str] = {}
            inferred_target_rels: List[Tuple[str, str, str]] = []

            for s_src, s_rel, s_dst in s_dom.relations:
                if s_rel in common_rel_types:
                    for t_src, t_rel, t_dst in t_dom.relations:
                        if t_rel == s_rel:
                            if s_src not in entity_map:
                                entity_map[s_src] = t_src
                            if s_dst not in entity_map:
                                entity_map[s_dst] = t_dst

            for s_src, s_rel, s_dst in s_dom.relations:
                if s_rel not in t_rel_types:
                    mapped_src = entity_map.get(s_src, f"target_{s_src}")
                    mapped_dst = entity_map.get(s_dst, f"target_{s_dst}")
                    inferred_target_rels.append((mapped_src, s_rel, mapped_dst))

            sim_score = (len(common_rel_types) / max(1, len(s_rel_types.union(t_rel_types)))) * (len(entity_map) / max(1, len(s_dom.entities)))
            sim_score = round(min(1.0, max(0.0, sim_score)), 4)

            return AnalogyMapping(
                source_domain=source_domain_name,
                target_domain=target_domain_name,
                entity_map=entity_map,
                relation_map={r: r for r in common_rel_types},
                structural_similarity_score=sim_score,
                inferred_target_relations=inferred_target_rels
            )
