"""
python/tara_core/brain.py

TaraBrain: Autonomous Decision & Orchestration Kernel for TARA AI.
Implements the complete autonomous cognitive loop:
USER INPUT
→ TARA BRAIN
→ MODEL INFERENCE
→ INTENT / REASONING
→ MEMORY + KNOWLEDGE CONTEXT
→ SKILL / TOOL SELECTION
→ RULE / IDENTITY AUTHORIZATION
→ EXECUTION
→ RESULT
→ MODEL RESPONSE
→ MEMORY / LEARNING UPDATE
"""

import os
import sys
import json
import re
import time
import hashlib
from typing import Dict, List, Any, Optional, Tuple

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
python_dir = os.path.join(REPO_ROOT, "python")
if python_dir not in sys.path:
    sys.path.insert(0, python_dir)

from tara_core.skills import SkillEngine
from TARA.MEMORY.memory_engine import MemoryEngine
from TARA.KNOWLEDGE.knowledge_base import GlobalKnowledgeBase
from TARA.RULES.engine.execution_guard import ExecutionGuard
from TARA.RULES.compiler.policy_schema import CompiledPolicy
from TARA.ACCESS.access_manager import IdentityManager
from TARA.ACCESS.services.auth_service import get_creator_auth_service, CreatorAuthService
from TARA.ACCESS.protected_v1 import (
    ProtectedStateManager,
    ActionBroker,
    PolicyRecordStore,
    StateLifecycleAdapter,
    ProtectedCommandDispatcher
)
from TARA.LEARNING.user_search_learner import UserSearchLearner
from TARA.LEARNING.autonomous_learner import AutonomousOnlineLearner, LearningMode
from TARA.TOOLS.file_inspector import inspect_file
from TARA.TOOLS.hash_verifier import compute_hash, verify_file_hash
from TARA.TOOLS.knowledge_retriever import search_knowledge
from TARA.TOOLS.provenance_tracker import create_provenance_record
from tara_model.generate import (
    load_trained_language_model,
    generate_response,
    generate_stream,
    ControlTokenAction,
    ControlTokenActionParser
)
from tara_core.nlu import SlotExtractor, SemanticIntentParser
from tara_core.context import SessionContextManager, CoreferenceResolver, ClarificationManager
from tara_core.planner import TaskPlanner, GoalTracker, PlanStep, Goal
from tara_core.evaluator import TaskVerifier, RecoveryEngine, UncertaintyDetector, SelfEvaluator
from tara_core.registry import CapabilityRegistry, CapabilityCategory
from tara_core.tools_registry import ToolRegistry
from tara_core.memory_interfaces import DynamicMemoryHub
from tara_core.user_model import UserManager
from tara_core.agent_orchestrator import AgentOrchestrator
from tara_core.model_registry import ModelRegistry
from tara_core.language_registry import LanguageRegistry
from tara_core.plugin_engine import PluginEngine
from tara_core.ai_robotics_domain import AIRoboticsDomainCapability, register_ai_robotics_capability
from tara_model.self_trainer import get_self_trainer, NativeSelfTrainer
from tara_core.cognitive_capabilities import get_cognitive_capabilities_hub, register_cognitive_capabilities, CognitiveCapabilitiesHub
from tara_core.auto_connect_sync import AutoConnectSyncEngine, EndpointDefinition, EndpointType, EndpointCapability
from tara_core.dynamic_engine_system import DynamicEngineSystem, EngineRegistry, EngineManifest, EngineStatus



class TaraBrain:
    """
    Autonomous Decision & Orchestration Kernel for TARA AI.
    """

    def __init__(
        self,
        model_dir: Optional[str] = None,
        identity_manager: Optional[IdentityManager] = None,
        policy: Optional[CompiledPolicy] = None
    ):
        self.repo_root = REPO_ROOT

        # Load local .env if present
        env_file = os.path.join(self.repo_root, ".env")
        if os.path.exists(env_file):
            try:
                with open(env_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line_s = line.strip()
                        if line_s and not line_s.startswith("#") and "=" in line_s:
                            k, v = line_s.split("=", 1)
                            if k.strip() not in os.environ:
                                os.environ[k.strip()] = v.strip()
            except Exception:
                pass
        
        # 1. Resolve and load current promoted model
        self.model_metadata = self._load_current_model_metadata()
        if model_dir is None:
            try:
                from tara_core.model_registry import ModelRegistry
                active_ver = ModelRegistry.get_default().get_active_version()
                if active_ver and os.path.exists(os.path.join(self.repo_root, active_ver.artifact_location)):
                    model_dir = os.path.join(self.repo_root, active_ver.artifact_location)
            except Exception:
                pass
            if model_dir is None:
                rel_loc = self.model_metadata.get("current_artifact_location", "storage/models/tara")
                model_dir = os.path.join(self.repo_root, rel_loc)
        self.model_dir = os.path.abspath(model_dir)

        self.model = None
        self.tokenizer = None
        self.model_config = None
        self._load_neural_model()

        # 2. Identity & Rule Guard Initialization
        self.identity_manager = identity_manager or IdentityManager(
            base_dir=os.path.join(self.repo_root, "TARA", "ACCESS")
        )
        self.creator_auth_service = get_creator_auth_service(repo_root=self.repo_root)

        if policy is None:
            policy = self._load_compiled_policy()

        self.guard = ExecutionGuard(
            active_policy=policy,
            identity_manager=self.identity_manager
        )

        # 2b. Protected Capability Subsystem (capability_v1)
        try:
            self.protected_state_mgr = ProtectedStateManager(
                audit_logger=getattr(self.creator_auth_service, "audit_logger", None),
                repo_root=self.repo_root
            )
            self.action_broker = ActionBroker(
                protected_state_mgr=self.protected_state_mgr,
                creator_auth_service=self.creator_auth_service,
                repo_root=self.repo_root,
                audit_logger=getattr(self.creator_auth_service, "audit_logger", None)
            )
            self.policy_record_store = PolicyRecordStore(
                protected_state_mgr=self.protected_state_mgr,
                creator_auth_service=self.creator_auth_service,
                audit_logger=getattr(self.creator_auth_service, "audit_logger", None),
                repo_root=self.repo_root
            )
            self.state_lifecycle_adapter = StateLifecycleAdapter(
                protected_state_mgr=self.protected_state_mgr,
                creator_auth_service=self.creator_auth_service,
                self_destruct_engine=getattr(self.identity_manager, "self_destruct", None),
                audit_logger=getattr(self.creator_auth_service, "audit_logger", None)
            )
            self.protected_cmd_dispatcher = ProtectedCommandDispatcher(
                manager=self.protected_state_mgr,
                broker=self.action_broker,
                policy_store=self.policy_record_store,
                destruction_adapter=self.state_lifecycle_adapter,
                creator_auth_service=self.creator_auth_service
            )
        except Exception:
            self.protected_state_mgr = None
            self.action_broker = None
            self.policy_record_store = None
            self.state_lifecycle_adapter = None
            self.protected_cmd_dispatcher = None

        # 3. Knowledge Base & Episodic Memory
        self.knowledge_base = GlobalKnowledgeBase(
            base_dir=os.path.join(self.repo_root, "TARA", "KNOWLEDGE")
        )
        self.memory_engine = MemoryEngine(
            memory_dir=os.path.join(self.repo_root, "TARA", "MEMORY"),
            legacy_dir=os.path.join(self.repo_root, "storage", "memory")
        )

        # 4. Skill Engine
        self.skill_engine = SkillEngine(
            dynamic_skills_dir=os.path.join(self.repo_root, "storage", "skills"),
            tara_skills_dir=os.path.join(self.repo_root, "TARA", "SKILLS")
        )

        # 5. Canonical Learning Subsystems
        self.user_search_learner = UserSearchLearner(
            knowledge_base=self.knowledge_base,
            execution_guard=self.guard
        )
        self.autonomous_learner = AutonomousOnlineLearner(
            knowledge_base=self.knowledge_base
        )

        # 6. Cognitive Capabilities Subsystems
        self.session_manager = SessionContextManager(max_turns_per_session=10)
        self.intent_parser = SemanticIntentParser(available_skills=self.skill_engine.list_skills())
        self.goal_tracker = GoalTracker()
        self.task_verifier = TaskVerifier()
        self.recovery_engine = RecoveryEngine(repo_root=self.repo_root)
        self.uncertainty_detector = UncertaintyDetector()
        self.self_evaluator = SelfEvaluator()

        # 7. Open-Ended Dynamic Architecture Registries
        self.capability_registry = CapabilityRegistry.get_default()
        self.tool_registry = ToolRegistry.get_default(repo_root=self.repo_root)
        self.memory_hub = DynamicMemoryHub(storage_dir=os.path.join(self.repo_root, "TARA", "MEMORY"))
        self.user_manager = UserManager.get_default()
        self.agent_orchestrator = AgentOrchestrator.get_default()
        self.model_registry = ModelRegistry.get_default()
        self.language_registry = LanguageRegistry.get_default()
        self.plugin_engine = PluginEngine.get_default()
        self.ai_robotics_domain = AIRoboticsDomainCapability.get_default(repo_root=self.repo_root)
        register_ai_robotics_capability(registry=self.capability_registry, repo_root=self.repo_root)
        self.self_trainer = get_self_trainer(repo_root=self.repo_root)
        self.cognitive_capabilities = get_cognitive_capabilities_hub(repo_root=self.repo_root)
        self.extended_capabilities = self.cognitive_capabilities.extended
        self.experiential_learner = self.cognitive_capabilities.experiential_learner
        self.event_bus = self.cognitive_capabilities.event_bus
        self.robotics_hal = self.cognitive_capabilities.robotics_hal
        self.working_memory = self.cognitive_capabilities.working_memory
        self.experiential_bridge = self.cognitive_capabilities.experiential_bridge
        self.ontology = self.cognitive_capabilities.ontology
        self.memory_consolidation = self.cognitive_capabilities.memory_consolidation
        self.constraint_solver = self.cognitive_capabilities.constraint_solver
        self.multi_objective_optimizer = self.cognitive_capabilities.multi_objective_optimizer
        self.probabilistic_reasoning = self.cognitive_capabilities.probabilistic_reasoning
        self.uncertainty_planner = self.cognitive_capabilities.uncertainty_planner
        self.environment_model = self.cognitive_capabilities.environment_model
        self.sensor_fusion = self.cognitive_capabilities.sensor_fusion
        self.action_grounder = self.cognitive_capabilities.action_grounder
        self.gui_interaction = self.cognitive_capabilities.gui_interaction
        self.software_engineering = self.cognitive_capabilities.software_engineering
        self.data_engineering = self.cognitive_capabilities.data_engineering
        self.experiment_design = self.cognitive_capabilities.experiment_design
        self.skill_acquisition = self.cognitive_capabilities.skill_acquisition
        self.demonstration_learning = self.cognitive_capabilities.demonstration_learning
        self.capability_composer = self.cognitive_capabilities.capability_composer
        self.skill_certifier = self.cognitive_capabilities.skill_certifier
        self.capability_acquisition = self.cognitive_capabilities.dynamic_capability_acquisition
        self.capability_acquisition.brain = self
        self.architecture_awareness = self.cognitive_capabilities.architecture_awareness
        self.architecture_evolution = self.cognitive_capabilities.architecture_evolution
        self.continual_learning_governor = self.cognitive_capabilities.continual_learning_governor
        self.continual_learning = self.continual_learning_governor
        self.curriculum_priority = self.cognitive_capabilities.curriculum_priority
        self.model_adaptation = self.cognitive_capabilities.model_adaptation
        self.model_router = self.cognitive_capabilities.model_router
        self.goal_persistence = self.cognitive_capabilities.goal_persistence
        self.incident_learning = self.cognitive_capabilities.incident_learning
        self.policy_strategy_learner = self.cognitive_capabilities.policy_strategy_learner
        self.audit_explainability = self.cognitive_capabilities.audit_explainability
        self.corrupted_recovery = self.cognitive_capabilities.corrupted_recovery
        self.self_maintenance = self.cognitive_capabilities.self_maintenance
        self.prediction_error_loop = self.cognitive_capabilities.prediction_error_loop
        register_cognitive_capabilities(registry=self.capability_registry, repo_root=self.repo_root)
        self.auto_sync = AutoConnectSyncEngine.get_default(repo_root=self.repo_root)
        self.auto_connect_router = self.auto_sync.router
        self.engine_system = DynamicEngineSystem.get_default(
            repo_root=self.repo_root,
            auto_connect_engine=self.auto_sync,
            capability_registry=self.capability_registry
        )
        self.engine_registry = self.engine_system.registry
        self.engine_router = self.engine_system.router

    def execute_engine(self, task_type: str, payload: Dict[str, Any], engine_id: Optional[str] = None) -> Dict[str, Any]:
        """Executes specialized computation via the Dynamic Engine System."""
        return self.engine_system.execute(task_type=task_type, payload=payload, engine_id=engine_id)

    def get_active_compute_endpoint(self, required_capabilities: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        """Returns the dynamically elected active compute/execution endpoint."""
        ep = self.auto_sync.get_active_endpoint(required_capabilities=required_capabilities)
        return ep.to_dict() if ep else None

    def process_distributed_workload(
        self,
        input_data: Union[str, Dict[str, Any]],
        task_name: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        custom_executor: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Routes and executes a workload through dynamic load distribution and capability routing."""
        if hasattr(self, "auto_sync") and self.auto_sync:
            return self.auto_sync.process_workload(
                input_data=input_data,
                task_name=task_name,
                context=context,
                custom_executor=custom_executor
            )
        return {"status": "ERROR", "error": "AutoSyncEngine not available on brain."}

    def get_cluster_telemetry(self) -> Dict[str, Any]:
        """Returns comprehensive telemetry for all cluster nodes."""
        if hasattr(self, "auto_sync") and self.auto_sync:
            return self.auto_sync.get_cluster_telemetry()
        return {"total_nodes": 0, "healthy_nodes": 0, "nodes": []}

    def sync_state(self, payload_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Cryptographically signs and synchronizes state across endpoints."""
        ok, pkg, status = self.auto_sync.stage_or_sync_state(payload_type=payload_type, data=data)
        return {"success": ok, "status": status, "package_id": pkg.package_id, "sequence": pkg.logical_sequence}

    def run_prediction_error_cycle(self, task_name: str, action_fn: Any, expectation: Optional[Any] = None) -> Dict[str, Any]:
        """Runs the complete 10-stage World Model + Prediction Error Learning cycle."""
        record = self.prediction_error_loop.execute_and_learn_cycle(task_name, action_fn, expectation)
        return record.to_dict() if hasattr(record, "to_dict") else vars(record)

    def get_self_model(self) -> Dict[str, Any]:
        """Returns the introspective self-model representation of TARA."""
        return self.cognitive_capabilities.self_model.get_snapshot(brain_ref=self)

    def run_experiential_cycle(self, input_text: str, action_executor: Optional[Any] = None) -> Dict[str, Any]:
        """Runs the complete 12-step closed-loop experiential cycle and stages verified experience."""
        episode = self.experiential_learner.run_full_closed_loop(
            input_text=input_text,
            action_executor=action_executor or (lambda a, g: self.process(actor_id="user", input_text=input_text))
        )
        staged = None
        if hasattr(self, "experiential_bridge") and self.experiential_bridge:
            try:
                staged = self.experiential_bridge.stage_episode(episode)
            except Exception as e:
                logger.warning(f"Failed to stage experiential episode: {e}")
        res = episode.to_dict() if hasattr(episode, "to_dict") else vars(episode)
        if staged:
            res["staged_sample"] = staged
        return res

    def _load_current_model_metadata(self) -> Dict[str, Any]:
        meta_path = os.path.join(self.repo_root, "TARA", "MODEL", "current_model.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _load_compiled_policy(self) -> Optional[CompiledPolicy]:
        policy_path = os.path.join(self.repo_root, "TARA", "RULES", "compiled_policy.json")
        if os.path.exists(policy_path):
            try:
                with open(policy_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return CompiledPolicy.from_dict(data)
            except Exception:
                pass
        return None

    def _load_neural_model(self) -> None:
        try:
            weights_path = os.path.join(self.model_dir, "model.safetensors")
            if not os.path.exists(weights_path):
                hf_token = os.environ.get("HF_TOKEN")
                if hf_token:
                    try:
                        from huggingface_hub import hf_hub_download
                        repo_id = os.environ.get("HF_REPO_ID", "tara-project/tara")
                        os.makedirs(self.model_dir, exist_ok=True)
                        hf_hub_download(
                            repo_id=repo_id,
                            filename="model.safetensors",
                            local_dir=self.model_dir,
                            token=hf_token
                        )
                    except Exception:
                        pass
            self.model, self.tokenizer, self.model_config = load_trained_language_model(self.model_dir)
        except Exception as ex:
            # Keep brain functional even if model weights fail to load
            self.model = None
            self.tokenizer = None
            self.model_config = None

    def _infer(self, prompt: str, max_new_tokens: int = 25) -> str:
        """Autoregressive neural generation through promoted model weights or active compute endpoint."""
        if self.model is not None and self.tokenizer is not None:
            try:
                # Query active compute endpoint if auto_sync layer is present
                _ = self.auto_sync.get_active_endpoint() if hasattr(self, "auto_sync") and self.auto_sync else None
                res = generate_response(
                    self.model,
                    self.tokenizer,
                    prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7
                )
                if isinstance(res, dict):
                    return res.get("text", "")
                return str(res)
            except Exception as e:
                return f"[MODEL_INFERENCE_ERROR: {str(e)}]"
        return "[MODEL_UNAVAILABLE: Running in offline fallback mode]"

    def stream_infer(self, prompt: str, max_new_tokens: int = 50):
        """Streams autoregressive tokens from the loaded neural model in real-time."""
        if self.model is not None and self.tokenizer is not None:
            for token in generate_stream(
                self.model,
                self.tokenizer,
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=0.7
            ):
                yield token
        else:
            yield "[MODEL_UNAVAILABLE: Running in offline fallback mode]"


    def get_model_status(self) -> Dict[str, Any]:
        raw_model_file = os.path.join(self.model_dir, "model.safetensors")
        if not os.path.exists(raw_model_file):
            candidates = [
                os.path.join(self.repo_root, "storage", "models", "tara", "model.safetensors"),
                os.path.join(os.getcwd(), "storage", "models", "tara", "model.safetensors"),
                "/opt/render/project/src/storage/models/tara/model.safetensors",
            ]
            for c in candidates:
                if os.path.exists(c):
                    raw_model_file = c
                    break

        model_file = os.path.abspath(raw_model_file).replace("\\", "/")

        model_sha256 = None
        if os.path.isfile(raw_model_file):
            h = hashlib.sha256()
            with open(raw_model_file, "rb") as f:
                while chunk := f.read(65536):
                    h.update(chunk)
            model_sha256 = h.hexdigest()

        expected_promoted_sha = "7a50308b799f2654baeafbd64dec31f088c3b07b446e198cd0f9ec2b7c0af309"
        model_integrity = "PASS" if (model_sha256 and model_sha256.lower() == expected_promoted_sha.lower()) else "FAIL"

        return {
            "model_identity": self.model_metadata.get("model_identity", "TARA"),
            "model_checkpoint": self.model_metadata.get("current_checkpoint", "baseline"),
            "model_file": model_file,
            "model_sha256": model_sha256,
            "model_integrity": model_integrity,
            "location": self.model_dir,
            "status": "LOADED" if self.model is not None else "OFFLINE_FALLBACK",
            "vocab_size": self.tokenizer.vocab_size if self.tokenizer else None,
            "parameters": self.model_metadata.get("model_capacity", {}).get("parameters", 118080),
            "offline_ready": True
        }

    # ------------------------------------------------------------------------
    # INTENT PARSING & REASONING
    # ------------------------------------------------------------------------
    def parse_intent(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Parses natural language input into structured intent, tool, or skill targets.
        Delegates to SemanticIntentParser with paraphrase support and slot extraction.
        """
        self.intent_parser.set_available_skills(self.skill_engine.list_skills())
        return self.intent_parser.parse(text, context)

    # ------------------------------------------------------------------------
    # TOOLS DISPATCHER
    # ------------------------------------------------------------------------
    def _execute_tool(self, tool_name: str, params: Dict[str, Any], actor_id: str) -> Dict[str, Any]:
        """Executes a tool dynamically via ToolRegistry without fixed tool limits."""
        return self.tool_registry.execute_tool(
            tool_name=tool_name,
            params=params,
            actor_id=actor_id,
            repo_root=self.repo_root
        )

    # ------------------------------------------------------------------------
    # SKILL DISPATCHER
    # ------------------------------------------------------------------------
    def _execute_skill(self, skill_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes a skill. Distinguishes executable skills from instruction-only catalog skills.
        """
        skill_id = skill_name.strip().lower().replace(" ", "_")

        # 0. Check domain capabilities & CapabilityRegistry
        if skill_id in ("ai_robotics_domain", "ai_robotics", "robotics", "artificial_intelligence"):
            return self.ai_robotics_domain.execute(params)

        if skill_id in ("robotics_hal", "hardware_actuation", "safety_interlock", "robotics_hardware"):
            cap = self.capability_registry.get_capability("robotics_hal")
            if cap and cap.handler:
                return cap.handler(params)
            return self.robotics_hal.send_motion_command(
                params.get("device_id", ""),
                tuple(params.get("target_coords", (0.0, 0.0, 0.0))),
                params.get("velocity", 0.5)
            ) if params.get("action") == "motion" else self.robotics_hal.read_telemetry(params.get("device_id", ""))

        cap = self.capability_registry.get_capability(skill_id) or self.capability_registry.get_capability(f"capability_{skill_id}")
        if cap and cap.executable and callable(cap.handler):
            return cap.handler(params)

        # 1. Native or dynamic executable skills
        if skill_id in self.skill_engine.skills or skill_id in self.skill_engine.dynamic_skills:
            return self.skill_engine.execute_skill(skill_name, params)

        # 2. Catalog skills: check if instruction-only
        cat = self.skill_engine.catalog_skills
        if skill_name in cat or skill_id in cat:
            cat_entry = cat.get(skill_name) or cat.get(skill_id)
            skill_folder = os.path.join(self.repo_root, "TARA", "SKILLS", cat_entry["category"], cat_entry["name"])
            script_files = [f for f in os.listdir(skill_folder) if f.endswith(".py")] if os.path.exists(skill_folder) else []
            
            if script_files:
                return self.skill_engine.execute_skill(skill_name, params)
            else:
                # Instruction-only catalog skill
                skill_md_path = os.path.join(skill_folder, "SKILL.md")
                instructions = ""
                if os.path.exists(skill_md_path):
                    with open(skill_md_path, "r", encoding="utf-8", errors="ignore") as f:
                        instructions = f.read()
                return {
                    "status": "INSTRUCTION_ONLY",
                    "skill": cat_entry["name"],
                    "category": cat_entry["category"],
                    "path": skill_folder,
                    "instructions_summary": instructions[:300] + ("..." if len(instructions) > 300 else ""),
                    "executable": False
                }

        # Fallback to general skill execution
        try:
            return self.skill_engine.execute_skill(skill_name, params)
        except Exception as e:
            return {"status": "ERROR", "error": str(e), "skill": skill_name}

    # ------------------------------------------------------------------------
    # SANITIZATION
    # ------------------------------------------------------------------------
    def _sanitize_for_memory(self, data: Any) -> Any:
        """Redacts secrets, passwords, tokens, signatures, and private keys before logging."""
        SENSITIVE_KEYS = {
            "passphrase", "password", "secret", "private_key", "token", "auth_token",
            "id_token", "device_signature", "creator_auth", "recovery_code", "secret_key"
        }
        if isinstance(data, dict):
            clean = {}
            for k, v in data.items():
                if any(sk in str(k).lower() for sk in SENSITIVE_KEYS):
                    clean[k] = "[REDACTED_SECRET]"
                else:
                    clean[k] = self._sanitize_for_memory(v)
            return clean
        elif isinstance(data, list):
            return [self._sanitize_for_memory(item) for item in data]
        elif isinstance(data, str):
            # Redact JWT pattern
            sanitized = re.sub(r"ey[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}", "[REDACTED_JWT_TOKEN]", data)
            return sanitized
        return data

    # ------------------------------------------------------------------------
    # MAIN COGNITIVE LOOP
    # ------------------------------------------------------------------------
    def process(self, actor_id: str, input_text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Executes the full end-to-end cognitive loop with:
        - Multi-turn session context & coreference resolution
        - Missing-parameter clarification & task suspension
        - Composite multi-step task planning & goal tracking
        - Semantic skill & tool execution
        - Post-execution task verification
        - Bounded error recovery & parameter healing
        - Uncertainty detection & explicit lack-of-evidence notices
        - Pre-finalization self-evaluation & consistency refinement
        - Sanitized episodic memory recording
        """
        ctx = dict(context or {})
        session_id = ctx.get("session_id", "default_session")

        # 0. Conversational Creator Authentication & Multi-Creator Verification
        session_token = ctx.get("creator_session_token") or ctx.get("session_token")
        if session_token and hasattr(self, "creator_auth_service"):
            active_sess = self.creator_auth_service.verify_session(session_token)
            if active_sess:
                actor_id = active_sess["creator_id"]
                ctx["creator_role"] = active_sess["role"]
                ctx["is_creator_authenticated"] = True

        # Check for protected capability triggers or commands when creator is authenticated
        dispatcher = getattr(self, "protected_cmd_dispatcher", None)
        mgr = getattr(self, "protected_state_mgr", None)
        if session_token and ctx.get("is_creator_authenticated") and dispatcher:
            if mgr and mgr.is_active(session_token, self.creator_auth_service):
                ctx["protected_state_active"] = True
                ctx["creator_id"] = actor_id

            super_res = dispatcher.handle_message(
                session_token=session_token,
                creator_id=actor_id,
                role=ctx.get("creator_role", "CREATOR"),
                input_text=input_text
            )
            if super_res and super_res.get("handled"):
                return {
                    "input": input_text,
                    "actor_id": actor_id,
                    "decision": "ALLOW",
                    "rule_check": {"decision": "ALLOW", "reason": "Authorized under advanced creator authority"},
                    "tool_or_skill": None,
                    "retrieved_context": {"knowledge": [], "memory": []},
                    "result": super_res.get("data", {"status": super_res.get("status")}),
                    "final_response": super_res.get("response"),
                    "outcome": "SUCCESS",
                    "context_tags": super_res.get("tags", ["CREATOR-AUTHENTICATED"])
                }

        # Check for Google Token Authentication
        id_token = ctx.get("google_id_token") or ctx.get("id_token")
        if not id_token and input_text.startswith("eyJ") and input_text.count(".") == 2:
            id_token = input_text.strip()

        if id_token and hasattr(self, "creator_auth_service"):
            auth_res = self.creator_auth_service.authenticate_google_token(
                id_token=id_token,
                client_key=ctx.get("client_ip", "local")
            )
            if auth_res.get("status") == "SUCCESS":
                msg = auth_res.get("message", "✅ Creator authenticated.")
                return {
                    "input": input_text,
                    "actor_id": auth_res.get("creator_id", "ROOT_OPERATOR"),
                    "decision": "ALLOW",
                    "rule_check": {"decision": "ALLOW", "reason": "Creator Google authentication verified"},
                    "tool_or_skill": None,
                    "retrieved_context": {"knowledge": [], "memory": []},
                    "result": auth_res,
                    "final_response": msg,
                    "outcome": "SUCCESS",
                    "context_tags": ["CREATOR-AUTHENTICATED", "METHOD-GOOGLE"]
                }
            else:
                err_msg = auth_res.get("error", "Creator authentication failed. No creator authority was granted.")
                return {
                    "input": input_text,
                    "actor_id": actor_id,
                    "decision": "DENY",
                    "rule_check": {"decision": "DENY", "reason": err_msg},
                    "tool_or_skill": None,
                    "retrieved_context": {"knowledge": [], "memory": []},
                    "result": auth_res,
                    "final_response": err_msg,
                    "outcome": "FAILURE",
                    "context_tags": ["CREATOR-AUTH-FAILED"]
                }

        # Check for Creator Key Proof Authentication
        key_sig = ctx.get("proof_signature") or ctx.get("key_signature")
        key_nonce = ctx.get("challenge_nonce") or ctx.get("nonce")
        key_passphrase = ctx.get("passphrase") or ctx.get("keystore_passphrase")
        if (key_sig and key_nonce) or key_passphrase:
            auth_res = self.creator_auth_service.authenticate_creator_key(
                proof_signature_hex=key_sig,
                challenge_nonce=key_nonce,
                claimed_creator_id=ctx.get("claimed_creator_id", actor_id),
                passphrase=key_passphrase,
                client_key=ctx.get("client_ip", "local")
            )
            if auth_res.get("status") == "SUCCESS":
                msg = auth_res.get("message", "✅ Creator authenticated.")
                return {
                    "input": input_text,
                    "actor_id": auth_res.get("creator_id", "ROOT_OPERATOR"),
                    "decision": "ALLOW",
                    "rule_check": {"decision": "ALLOW", "reason": "Creator Key authentication verified"},
                    "tool_or_skill": None,
                    "retrieved_context": {"knowledge": [], "memory": []},
                    "result": auth_res,
                    "final_response": msg,
                    "outcome": "SUCCESS",
                    "context_tags": ["CREATOR-AUTHENTICATED", "METHOD-KEY"]
                }
            else:
                err_msg = auth_res.get("error", "Creator authentication failed. No creator authority was granted.")
                return {
                    "input": input_text,
                    "actor_id": actor_id,
                    "decision": "DENY",
                    "rule_check": {"decision": "DENY", "reason": err_msg},
                    "tool_or_skill": None,
                    "retrieved_context": {"knowledge": [], "memory": []},
                    "result": auth_res,
                    "final_response": err_msg,
                    "outcome": "FAILURE",
                    "context_tags": ["CREATOR-AUTH-FAILED"]
                }

        # Check for QR Approval Authentication
        qr_chal_id = ctx.get("challenge_id")
        qr_dev_id = ctx.get("device_id")
        qr_sig = ctx.get("device_signature")
        if qr_chal_id and qr_dev_id and qr_sig:
            auth_res = self.creator_auth_service.verify_qr_approval(
                challenge_id=qr_chal_id,
                device_id=qr_dev_id,
                device_signature_hex=qr_sig,
                client_key=ctx.get("client_ip", "local")
            )
            if auth_res.get("status") == "SUCCESS":
                msg = auth_res.get("message", "✅ Creator authenticated.")
                return {
                    "input": input_text,
                    "actor_id": auth_res.get("creator_id", "ROOT_OPERATOR"),
                    "decision": "ALLOW",
                    "rule_check": {"decision": "ALLOW", "reason": "QR Challenge approval verified"},
                    "tool_or_skill": None,
                    "retrieved_context": {"knowledge": [], "memory": []},
                    "result": auth_res,
                    "final_response": msg,
                    "outcome": "SUCCESS",
                    "context_tags": ["CREATOR-AUTHENTICATED", "METHOD-QR"]
                }
            else:
                err_msg = auth_res.get("error", "Creator authentication failed. No creator authority was granted.")
                return {
                    "input": input_text,
                    "actor_id": actor_id,
                    "decision": "DENY",
                    "rule_check": {"decision": "DENY", "reason": err_msg},
                    "tool_or_skill": None,
                    "retrieved_context": {"knowledge": [], "memory": []},
                    "result": auth_res,
                    "final_response": err_msg,
                    "outcome": "FAILURE",
                    "context_tags": ["CREATOR-AUTH-FAILED"]
                }

        # Check for Recovery Code Authentication
        rec_code = ctx.get("recovery_code")
        if not rec_code and input_text.upper().startswith("RECOVERY:"):
            rec_code = input_text.split(":", 1)[1].strip()
        elif not rec_code and len(input_text.strip()) in (32, 35) and input_text.count("-") >= 3:
            rec_code = input_text.strip()

        if rec_code:
            auth_res = self.creator_auth_service.authenticate_recovery(
                recovery_code_or_token=rec_code,
                claimed_creator_id=ctx.get("claimed_creator_id", actor_id),
                client_key=ctx.get("client_ip", "local")
            )
            if auth_res.get("status") == "SUCCESS":
                msg = auth_res.get("message", "✅ Creator authenticated.")
                return {
                    "input": input_text,
                    "actor_id": auth_res.get("creator_id", "ROOT_OPERATOR"),
                    "decision": "ALLOW",
                    "rule_check": {"decision": "ALLOW", "reason": "Emergency recovery verified"},
                    "tool_or_skill": None,
                    "retrieved_context": {"knowledge": [], "memory": []},
                    "result": auth_res,
                    "final_response": msg,
                    "outcome": "SUCCESS",
                    "context_tags": ["CREATOR-AUTHENTICATED", "METHOD-RECOVERY"]
                }
            else:
                err_msg = auth_res.get("error", "Creator authentication failed. No creator authority was granted.")
                return {
                    "input": input_text,
                    "actor_id": actor_id,
                    "decision": "DENY",
                    "rule_check": {"decision": "DENY", "reason": err_msg},
                    "tool_or_skill": None,
                    "retrieved_context": {"knowledge": [], "memory": []},
                    "result": auth_res,
                    "final_response": err_msg,
                    "outcome": "FAILURE",
                    "context_tags": ["CREATOR-AUTH-FAILED"]
                }

        # Check for Conversational Private Creator Trigger Phrase
        if hasattr(self, "creator_auth_service"):
            trig_res = self.creator_auth_service.check_conversational_trigger(input_text)
            if trig_res.get("is_trigger"):
                resp_text = trig_res.get(
                    "response",
                    "Creator authentication requested. Choose an authentication method:\\n\\n"
                    "[ 📱 QR Authentication ]\\n"
                    "[ 🔑 Creator Key Authentication ]\\n"
                    "[ 🔐 Google Authentication ]\\n"
                    "[ 🆘 Recovery ]"
                )
                return {
                    "input": input_text,
                    "actor_id": actor_id,
                    "decision": "ALLOW",
                    "rule_check": {"decision": "ALLOW", "reason": "Conversational creator authentication flow initiated"},
                    "tool_or_skill": None,
                    "retrieved_context": {"knowledge": [], "memory": []},
                    "result": {
                        "status": "AWAITING_METHOD_SELECTION",
                        "message": resp_text,
                        "methods": [
                            {"id": "qr", "label": "📱 QR Authentication"},
                            {"id": "creator_key", "label": "🔑 Creator Key Authentication"},
                            {"id": "google", "label": "🔐 Google Authentication"},
                            {"id": "recovery", "label": "🆘 Recovery"}
                        ]
                    },
                    "final_response": resp_text,
                    "outcome": "SUCCESS",
                    "context_tags": ["CREATOR-AUTH-INITIATED", "AWAITING-METHOD-SELECTION"]
                }

            # Check if user is selecting one of the 4 authentication methods
            selected_method = self.creator_auth_service.detect_method_selection(input_text)
            if selected_method:
                method_res = self.creator_auth_service.handle_method_selection(
                    method=selected_method,
                    creator_id=actor_id if actor_id != "user" else "ROOT_OPERATOR",
                    client_key=ctx.get("client_ip", "local")
                )
                method_msg = method_res.get("response", "Authentication method selected.")
                return {
                    "input": input_text,
                    "actor_id": actor_id,
                    "decision": "ALLOW",
                    "rule_check": {"decision": "ALLOW", "reason": f"Authentication method {selected_method} selected"},
                    "tool_or_skill": None,
                    "retrieved_context": {"knowledge": [], "memory": []},
                    "result": method_res,
                    "final_response": method_msg,
                    "outcome": "SUCCESS",
                    "context_tags": ["CREATOR-METHOD-SELECTED", f"METHOD-{selected_method}"]
                }

        session = self.session_manager.get_or_create(session_id=session_id, actor_id=actor_id)
        if hasattr(self, "event_bus") and self.event_bus:
            self.event_bus.publish(
                "cognitive.turn_started",
                {"actor_id": actor_id, "session_id": session_id, "input": input_text},
                source="TaraBrain"
            )
        error: Optional[str] = None

        # 1. Multi-Turn Clarification Resumption & Coreference Resolution
        pending = session.get_pending_task()
        if pending:
            # The previous turn requested clarification on a missing slot
            missing_slot = pending["missing_slot"]
            slots = SlotExtractor.extract_all_slots(input_text)
            provided_val = slots.get(missing_slot) or input_text.strip()
            pending_intent = pending["intent"]
            pending_intent.setdefault("params", {})[missing_slot] = provided_val
            session.clear_pending_task()
            intent = pending_intent
            action_intent = intent.get("intent")
        else:
            resolved_text, resolved_slots = CoreferenceResolver.resolve(input_text, session)
            merged_ctx = dict(ctx)
            for k, v in resolved_slots.items():
                if k not in merged_ctx:
                    merged_ctx[k] = v
            intent = self.parse_intent(resolved_text, merged_ctx)
            action_intent = intent.get("intent")

        # 2. Missing Parameter Clarification Interception
        clarif = ClarificationManager.evaluate(intent, session)
        if clarif and clarif.get("needs_clarification"):
            session.set_pending_task(clarif)
            clarification_q = clarif["clarification_question"]
            session.record_turn(
                user_input=input_text,
                response=clarification_q,
                slots=intent.get("slots", {}),
                tool_result=None,
                outcome="SUCCESS"
            )
            return {
                "input": input_text,
                "actor_id": actor_id,
                "intent": intent,
                "decision": "NEEDS_CLARIFICATION",
                "rule_check": {"decision": "ALLOW", "reason": "Execution suspended pending user parameter clarification"},
                "tool_or_skill": None,
                "retrieved_context": {"knowledge": [], "memory": []},
                "result": {"status": "NEEDS_CLARIFICATION", "question": clarification_q, "missing_slot": clarif["missing_slot"]},
                "final_response": clarification_q,
                "outcome": "SUCCESS",
                "context_tags": ["CLARIFICATION-NEEDED"]
            }

        retrieved_knowledge: List[Dict[str, Any]] = []
        retrieved_memory: List[Dict[str, Any]] = []
        context_tags: List[str] = []

        # 3. Knowledge & Episodic Memory Retrieval
        kb_matches = self.knowledge_base.query_knowledge(input_text)
        if kb_matches:
            retrieved_knowledge = kb_matches[:3]
            context_tags.append("VERIFIED-KNOWLEDGE")

        mem_matches = self.memory_engine.query_episodes(query=input_text, actor_id=actor_id, limit=3)
        if mem_matches:
            retrieved_memory = mem_matches
            context_tags.append("MEMORY")

        # 4. Composite Multi-Step Planning & Goal Execution
        if action_intent == "COMPOSITE_PLAN":
            steps = TaskPlanner.plan_composite_task(input_text, intent.get("slots", {}), ctx)
            goal = self.goal_tracker.create_goal(
                actor_id=actor_id,
                objective=input_text,
                steps=steps,
                completion_criteria="All composite steps executed and verified"
            )
            if hasattr(self, "goal_persistence") and self.goal_persistence:
                try:
                    self.goal_persistence.persist_goal(goal.to_dict())
                except Exception:
                    pass
            step_outputs = []
            current_data_bindings = {}
            plan_failed = False
            plan_failure_reason = ""

            for step in steps:
                # Bind previous step output to input parameters
                for param_key, bind_expr in step.input_bindings.items():
                    if bind_expr == "step_1.file_path" and "step_1_file_path" in current_data_bindings:
                        step.parameters[param_key] = current_data_bindings["step_1_file_path"]

                # Per-step Rule Authorization
                step_rule_ctx = dict(ctx)
                step_rule_ctx["actor_id"] = actor_id
                step_rule_ctx["action_payload"] = json.dumps(step.parameters)
                step_guard = self.guard.evaluate_action(action_type=step.rule_action_type, context=step_rule_ctx)
                if step_guard.get("decision") == "DENY":
                    step.status = "BLOCKED"
                    goal.mark_failed(f"Step {step.step_id} blocked by rule: {step_guard.get('reason')}")
                    plan_failed = True
                    plan_failure_reason = step_guard.get('reason')
                    break

                # Execute Step
                if step.action_intent == "EXECUTE_TOOL":
                    step_res = self._execute_tool(step.target_name, step.parameters, actor_id=actor_id)
                elif step.action_intent == "EXECUTE_SKILL":
                    step_res = self._execute_skill(step.target_name, step.parameters)
                elif step.action_intent == "EXECUTE_ENGINE":
                    step_res = self.execute_engine(step.target_name, step.parameters)
                else:
                    step_res = {"status": "SUCCESS"}

                # Task Verification on step output
                step_verif = self.task_verifier.verify(step.action_intent, step.target_name, step_res)
                if not step_verif["satisfied"]:
                    # Attempt Recovery
                    can_recover, healed_params, rec_strat = self.recovery_engine.attempt_recovery(
                        step.target_name, step.parameters, step_res.get("error", "")
                    )
                    if can_recover:
                        step.parameters = healed_params
                        if step.action_intent == "EXECUTE_TOOL":
                            step_res = self._execute_tool(step.target_name, step.parameters, actor_id=actor_id)
                        step_verif = self.task_verifier.verify(step.action_intent, step.target_name, step_res)

                if not step_verif["satisfied"]:
                    step.status = "FAILED"
                    goal.mark_failed(f"Step {step.step_id} failed verification: {step_verif.get('verification_notes')}")
                    plan_failed = True
                    plan_failure_reason = step_verif.get('verification_notes')
                    break

                if step.target_name == "file_inspector" and step_res.get("file_path"):
                    current_data_bindings["step_1_file_path"] = step_res["file_path"]

                goal.record_step_result(step.step_id, step_res)
                step_outputs.append({
                    "step": step.step_id,
                    "description": step.description,
                    "result": step_res,
                    "verified": step_verif["satisfied"]
                })

            if plan_failed:
                outcome = "FAILURE"
                error = plan_failure_reason
                result = {"status": "FAILED", "goal": goal.to_dict(), "error": error}
                final_response = f"[PLAN_FAILED]: {plan_failure_reason}"
            else:
                outcome = "SUCCESS"
                result = {"status": "COMPLETED", "goal": goal.to_dict(), "steps": step_outputs}
                final_response = f"Plan completed successfully ({len(step_outputs)} steps executed and verified)."

            decision = "ALLOW"
            guard_res = {"decision": "ALLOW", "reason": "Multi-step plan authorized"}
            tool_or_skill_info = {"type": "PLAN", "name": "composite_task_planner"}
            context_tags.append("MULTI-STEP-PLAN")
            action_type = "composite_plan"

        else:
            # 5. Single Action Determination for Rule Check
            action_type = "chat"
            action_payload_for_rule = input_text
            if "action_type" in ctx:
                action_type = ctx["action_type"]
                action_payload_for_rule = ctx.get("action_payload", input_text)
            elif action_intent == "EXECUTE_TOOL":
                tool_name = intent.get("tool")
                action_type = tool_name
                action_payload_for_rule = json.dumps(intent.get("params", {}))
            elif action_intent == "EXECUTE_SKILL":
                skill_name = intent.get("skill")
                action_type = f"skill_{skill_name}"
                action_payload_for_rule = json.dumps(intent.get("params", {}))
            elif action_intent == "EXECUTE_ENGINE":
                engine_target = intent.get("engine_id") or intent.get("task_type") or "engine"
                action_type = f"engine_{engine_target}"
                action_payload_for_rule = json.dumps(intent.get("payload", intent.get("params", {})))
            elif action_intent == "GUARDED_ACTION":
                action_type = intent.get("action_type")
                action_payload_for_rule = intent.get("action_type")
            elif action_intent == "ONLINE_LEARNING":
                action_type = "web_search"
                action_payload_for_rule = intent.get("query", input_text)
            elif action_intent == "AUTONOMOUS_LEARNING":
                action_type = "modify_system_learning"
                action_payload_for_rule = "autonomous_learning_loop"
            elif action_intent in ("AGENT_TASK", "SPAWN_AGENT"):
                action_type = "agent_orchestration"
                action_payload_for_rule = json.dumps(intent.get("params", {}))
            else:
                first_word = input_text.split()[0].lower() if input_text.strip() else ""
                if first_word in ["display_media", "generate_content", "show_photo", "show_video", "export_key"]:
                    action_type = first_word

            rule_eval_context = dict(ctx)
            rule_eval_context["actor_id"] = actor_id
            rule_eval_context["topic"] = input_text
            rule_eval_context["action_payload"] = action_payload_for_rule
            if "is_important" in intent.get("params", {}):
                rule_eval_context["is_important"] = intent["params"]["is_important"]

            # Rule & Identity Authorization
            guard_res = self.guard.evaluate_action(action_type=action_type, context=rule_eval_context)
            decision = guard_res.get("decision", "ALLOW")

            result = None
            outcome = "SUCCESS"
            error = None
            tool_or_skill_info = None

            if decision == "DENY":
                outcome = "FAILURE"
                error = guard_res.get("reason", "Denied by safety policy")
                result = {
                    "status": "BLOCKED_BY_POLICY",
                    "reason": error,
                    "decision": "DENY"
                }
                final_response = f"[BLOCKED_BY_POLICY] {error}"
            else:
                # 6. Execution & Recovery Engine
                try:
                    if action_intent == "EXECUTE_TOOL":
                        tool_name = intent.get("tool")
                        tool_or_skill_info = {"type": "TOOL", "name": tool_name}
                        result = self._execute_tool(tool_name, intent.get("params", {}), actor_id=actor_id)
                        context_tags.append("TOOL-RESULT")

                        # Recovery check on tool failure
                        if isinstance(result, dict) and (result.get("status") in ("ERROR", "FAILURE") or not result.get("exists", True)):
                            err_msg = str(result.get("error") or "Resource not found")
                            can_recover, healed_params, rec_strat = self.recovery_engine.attempt_recovery(
                                tool_name, intent.get("params", {}), err_msg
                            )
                            if can_recover:
                                rec_rule_ctx = dict(rule_eval_context)
                                rec_rule_ctx["action_payload"] = json.dumps(healed_params)
                                if self.guard.evaluate_action(action_type=action_type, context=rec_rule_ctx).get("decision") == "ALLOW":
                                    result = self._execute_tool(tool_name, healed_params, actor_id=actor_id)
                                    result["recovered_via"] = rec_strat

                    elif action_intent == "EXECUTE_SKILL":
                        skill_name = intent.get("skill")
                        tool_or_skill_info = {"type": "SKILL", "name": skill_name}
                        result = self._execute_skill(skill_name, intent.get("params", {}))
                        context_tags.append("TOOL-RESULT")

                    elif action_intent == "EXECUTE_ENGINE":
                        task_type = intent.get("task_type", "")
                        payload = intent.get("payload") or intent.get("params", {})
                        engine_id = intent.get("engine_id")
                        tool_or_skill_info = {"type": "ENGINE", "name": engine_id or task_type}
                        result = self.execute_engine(task_type=task_type, payload=payload, engine_id=engine_id)
                        context_tags.append("ENGINE-RESULT")

                    elif action_intent == "GUARDED_ACTION":
                        result = {
                            "status": "SUCCESS",
                            "action": action_type,
                            "authorized": True,
                            "creator_verified": guard_res.get("creator_verified", False)
                        }

                    elif action_intent == "ONLINE_LEARNING":
                        query = intent.get("query")
                        tool_or_skill_info = {"type": "LEARNING", "name": "user_search_learner"}
                        result = self.user_search_learner.search_and_learn(query, user_id=actor_id)
                        context_tags.append("TOOL-RESULT")

                    elif action_intent == "AUTONOMOUS_LEARNING":
                        mode = intent.get("mode")
                        tool_or_skill_info = {"type": "LEARNING", "name": "autonomous_learner"}
                        result = self.autonomous_learner.start_session(mode, actor_id=actor_id)
                        context_tags.append("TOOL-RESULT")

                    elif action_intent in ("AGENT_TASK", "SPAWN_AGENT"):
                        role = intent.get("role", "worker")
                        objective = intent.get("objective") or intent.get("task") or input_text
                        tool_or_skill_info = {"type": "AGENT", "name": f"agent_{role}"}
                        agent_task = self.agent_orchestrator.create_agent(
                            role=role,
                            objective=objective
                        )
                        planned_steps = intent.get("steps")
                        if planned_steps:
                            agent_res = self.agent_orchestrator.run_agent(agent_task, planned_steps)
                            result = agent_res
                        else:
                            result = {
                                "status": "SUCCESS",
                                "agent_id": agent_task.agent_id,
                                "role": agent_task.role,
                                "objective": agent_task.objective,
                                "state": "INITIALIZED"
                            }
                        context_tags.append("AGENT-DISPATCHED")

                    else:
                        result = {"status": "SUCCESS", "type": "CONVERSATIONAL"}

                    # Task Verification Check
                    target_name_eval = tool_or_skill_info.get("name") if tool_or_skill_info else action_type
                    verif_report = self.task_verifier.verify(action_intent, target_name_eval, result)
                    if not verif_report["satisfied"]:
                        outcome = "FAILURE"
                        error = verif_report["verification_notes"]
                    elif isinstance(result, dict) and (result.get("status") in ("ERROR", "FAILURE") or result.get("success") is False):
                        outcome = "FAILURE"
                        error = result.get("error") or result.get("reason") or "Operation returned failure status"

                except Exception as e:
                    outcome = "FAILURE"
                    error = str(e)
                    result = {"status": "ERROR", "error": str(e)}

                # 7. Model Inference & Final Response Formulation (Untrusted Content Sanitized)
                def _sanitize_untrusted_prompt_data(data: Any) -> str:
                    if data is None:
                        return ""
                    s = data if isinstance(data, str) else str(data)
                    for tok in (
                        "<|tara_rule|>", "<|tara_exec|>", "<|tara_skill|>", "<|tara_memory|>",
                        "<|creator_auth|>", "<|im_start|>", "<|im_end|>", "<|pad|>", "<|unk|>"
                    ):
                        if tok in s:
                            s = s.replace(tok, f"[ESCAPED_TOKEN:{tok.strip('<|>')}]")
                    import re
                    return re.sub(r"<\|([a-zA-Z0-9_\-\.]+)\|>", r"[ESCAPED_TOKEN:\1]", s)

                prompt_parts = []
                if retrieved_memory:
                    mem_snippet = " | ".join(_sanitize_untrusted_prompt_data(str(m.get("observations", ""))[:80]) for m in retrieved_memory)
                    prompt_parts.append(f"[UNTRUSTED-MEMORY-DATA]: {mem_snippet}")
                if hasattr(self, "working_memory") and self.working_memory:
                    wm_summary = self.working_memory.get_context_for_prompt(session_id)
                    if wm_summary:
                        prompt_parts.append(f"[WORKING-MEMORY]: {_sanitize_untrusted_prompt_data(wm_summary[:120])}")
                if retrieved_knowledge:
                    kb_snippet = " | ".join(_sanitize_untrusted_prompt_data(k.get("content", "")[:100]) for k in retrieved_knowledge)
                    prompt_parts.append(f"[UNTRUSTED-RETRIEVED-KNOWLEDGE]: {kb_snippet}")
                if result and action_intent in ("EXECUTE_TOOL", "EXECUTE_SKILL", "ONLINE_LEARNING", "EXECUTE_ENGINE"):
                    tool_res_str = json.dumps(result, ensure_ascii=False)[:120]
                    prompt_parts.append(f"[UNTRUSTED-TOOL-RESULT]: {_sanitize_untrusted_prompt_data(tool_res_str)}")

                prompt_parts.append(f"[USER-INPUT]: {_sanitize_untrusted_prompt_data(input_text)}")
                prompt = "\n".join(prompt_parts)

                if self.model is not None:
                    context_tags.append("MODEL-KNOWN")

                # Uncertainty Detection on Conversational Factual Queries
                uncert_statement = None
                if action_intent == "CONVERSATIONAL" and not retrieved_knowledge:
                    uncert_statement = self.uncertainty_detector.evaluate_uncertainty(input_text, retrieved_knowledge, retrieved_memory)

                neural_output = ""
                needs_inference = ((action_intent == "CONVERSATIONAL" and not uncert_statement) or outcome == "FAILURE")
                if needs_inference:
                    neural_output = self._infer(prompt, max_new_tokens=15)
                    # Control token action parsing (dynamic neural tool/skill invocation)
                    ctl_action = ControlTokenActionParser.parse(neural_output)
                    if ctl_action is not None and outcome != "FAILURE":
                        try:
                            # CRITICAL: Verify with Execution Guard before dispatching
                            action_name = f"EXECUTE_{ctl_action.action_type}"
                            guard_check = self.guard.evaluate_action(
                                action_name,
                                {"target": ctl_action.target, "actor_id": actor_id, "boundary": "MODEL"}
                            )
                            if getattr(guard_check, "decision", None) != "DENY":
                                if ctl_action.action_type == "TOOL":
                                    ctl_tool_res = self._execute_tool(ctl_action.target, ctl_action.payload, actor_id=actor_id)
                                    result = ctl_tool_res
                                    tool_or_skill_info = {"type": "NEURAL_TOOL", "name": ctl_action.target}
                                    context_tags.append("NEURAL-TOOL-DISPATCHED")
                                elif ctl_action.action_type == "SKILL":
                                    ctl_skill_res = self._execute_skill(ctl_action.target, ctl_action.payload)
                                    if isinstance(ctl_skill_res, dict) and ctl_skill_res.get("status") != "ERROR":
                                        result = ctl_skill_res
                                        tool_or_skill_info = {"type": "NEURAL_SKILL", "name": ctl_action.target}
                                        context_tags.append("NEURAL-SKILL-DISPATCHED")
                        except Exception:
                            pass

                if outcome == "FAILURE":
                    final_response = f"[ERROR: {error}] (Neural reflection: {neural_output})"
                elif uncert_statement:
                    final_response = uncert_statement
                    context_tags.append("UNCERTAINTY-STATED")
                elif action_intent == "EXECUTE_TOOL":
                    final_response = f"Tool '{intent.get('tool')}' executed successfully: {json.dumps(result, ensure_ascii=False)}"
                elif action_intent == "EXECUTE_SKILL":
                    if result.get("status") == "INSTRUCTION_ONLY":
                        final_response = f"Catalog Skill '{intent.get('skill')}' (Instruction-Only): {result.get('instructions_summary')}"
                    else:
                        final_response = f"Skill '{intent.get('skill')}' executed: {json.dumps(result, ensure_ascii=False)}"
                elif action_intent == "EXECUTE_ENGINE":
                    final_response = f"Engine '{intent.get('engine_id') or intent.get('task_type')}' executed: {json.dumps(result, ensure_ascii=False)}"
                elif action_intent == "ONLINE_LEARNING":
                    final_response = f"Online research completed: {result.get('answer', '')}"
                elif action_intent in ("AGENT_TASK", "SPAWN_AGENT"):
                    final_response = f"Agent '{result.get('agent_id')}' ({result.get('role')}): status {result.get('status')}"
                elif retrieved_knowledge:
                    final_response = f"{retrieved_knowledge[0].get('content', '')} (Neural output: {neural_output})"
                else:
                    final_response = neural_output

                # Self-Evaluation & Refinement
                verif_dict = verif_report if "verif_report" in locals() else {"satisfied": True}
                final_response = self.self_evaluator.evaluate_and_refine(
                    user_input=input_text,
                    tool_or_skill=tool_or_skill_info,
                    result=result,
                    candidate_response=final_response,
                    verification_report=verif_dict
                )

        # 8. Session Context Update
        turn_slots = dict(intent.get("slots", {}))
        if "params" in intent and isinstance(intent["params"], dict):
            turn_slots.update(intent["params"])
        if isinstance(result, dict):
            if "file_path" in result:
                turn_slots["file_path"] = result["file_path"]
            elif "path" in result:
                turn_slots["file_path"] = result["path"]
            if "hash" in result:
                turn_slots["hash"] = result["hash"]

        session.record_turn(
            user_input=input_text,
            response=final_response,
            slots=turn_slots,
            tool_result=result if isinstance(result, dict) else None,
            outcome=outcome
        )

        # 9. Episodic Memory Update (Sanitized)
        sanitized_input = self._sanitize_for_memory(input_text)
        sanitized_params = self._sanitize_for_memory(ctx)
        sanitized_result = self._sanitize_for_memory(result)

        self.memory_engine.record_episode(
            actor_id=actor_id,
            intent=action_intent,
            action=action_type if "action_type" in locals() else "composite_plan",
            parameters={"input": sanitized_input, "context": sanitized_params},
            outcome=outcome,
            observations=sanitized_result,
            error=error,
            reflection=f"Decision: {decision}, Outcome: {outcome}."
        )

        # 10. Update World State Tracking
        try:
            self.extended_capabilities.world_state.update_entity(
                entity_id=f"actor_{actor_id}",
                attributes={
                    "last_input": sanitized_input[:100] if isinstance(sanitized_input, str) else str(sanitized_input),
                    "last_outcome": outcome,
                    "last_action": action_type if "action_type" in locals() else "composite_plan"
                }
            )
        except Exception:
            pass

        # 11. Working Memory Governance
        wm_snapshot = None
        if hasattr(self, "working_memory") and self.working_memory:
            try:
                wm_snapshot = self.working_memory.govern_session(
                    session_id=session_id,
                    turns=session.history,
                    active_goal=input_text if action_intent == "COMPOSITE_PLAN" else "",
                    active_slots=session.last_slots,
                    security_verdicts=[guard_res.get("decision", "ALLOW")] if "guard_res" in locals() else []
                )
            except Exception as e:
                logger.warning(f"Working memory governor error: {e}")

        # 12. Turn Completed Telemetry Event
        if hasattr(self, "event_bus") and self.event_bus:
            self.event_bus.publish(
                "cognitive.turn_completed",
                {
                    "actor_id": actor_id,
                    "session_id": session_id,
                    "intent": action_intent,
                    "decision": decision,
                    "outcome": outcome
                },
                source="TaraBrain"
            )
        # 13. Episodic to Semantic Memory Consolidation
        if hasattr(self, "memory_consolidation") and self.memory_consolidation and outcome == "SUCCESS":
            try:
                self.memory_consolidation.consolidate_episode({
                    "episode_id": f"ep_{session_id}_{len(session.history)}",
                    "actor_id": actor_id,
                    "action": action_type if "action_type" in locals() else "chat",
                    "parameters": {"input": sanitized_input, "context": sanitized_params},
                    "outcome": outcome,
                    "result": sanitized_result,
                    "reflection": f"Decision: {decision}, Outcome: {outcome}."
                })
            except Exception as e:
                logger.warning(f"Memory consolidation error: {e}")

        # 14. Audit & Explainability Recording
        if hasattr(self, "audit_explainability") and self.audit_explainability:
            try:
                self.audit_explainability.record_decision(
                    actor_id=actor_id,
                    intent=action_intent or "chat",
                    evidence=[k.get("content", "") for k in retrieved_knowledge],
                    selected_strategy=action_type if "action_type" in locals() else "direct",
                    action=str(tool_or_skill_info) if tool_or_skill_info else (action_type if "action_type" in locals() else "chat"),
                    outcome=outcome,
                    rationale=f"Decision: {decision}. Outcome: {outcome}."
                )
            except Exception as e:
                logger.warning(f"Audit recording error: {e}")

        return {
            "input": input_text,
            "actor_id": actor_id,
            "intent": intent,
            "decision": decision,
            "rule_check": guard_res,
            "tool_or_skill": tool_or_skill_info,
            "retrieved_context": {
                "knowledge": retrieved_knowledge,
                "memory": retrieved_memory
            },
            "result": result,
            "final_response": final_response,
            "outcome": outcome,
            "context_tags": context_tags
        }

    def resolve_missing_capability(
        self,
        task_name: str,
        task_payload: Optional[Dict[str, Any]] = None,
        actor_id: str = "TARA_CORE",
        candidate_code: Optional[str] = None,
        candidate_entry_point: Optional[str] = None
    ) -> Dict[str, Any]:
        """Resolves missing capabilities dynamically through the 12-step acquisition pipeline."""
        return self.capability_acquisition.resolve_and_execute(
            task_name=task_name,
            task_payload=task_payload,
            actor_id=actor_id,
            candidate_code=candidate_code,
            candidate_entry_point=candidate_entry_point
        )

