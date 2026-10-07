# TARA Whole-System Architecture & Capability Map

This directory contains the authoritative, machine-readable architecture maps for TARA (The Autonomous Reasoning Architecture).

Every future agent, developer, and automated pipeline must consult this architecture map before making any structural changes to the codebase.

---

## 1. Directory Structure

```
ARCHITECTURE/
├── CAPABILITY_GRAPH.json     # Graph of all 16 core capabilities, nodes, and edges
├── SOURCE_INDEX.json         # Complete AST index: files, modules, structs, enums, functions, and cross-references
├── CONTROL_FLOW.json         # Step-by-step execution pipelines across the 4 major operational loops
├── DATA_FLOW.json            # 8-stage data lifecycle and persistent storage mount registry
└── CAPABILITIES/             # Dedicated specification for each individual capability
    ├── access_and_security.json
    ├── cognitive_loop.json
    ├── control_plane_and_workload.json
    ├── cryptographic_reward_and_goals.json
    ├── dataset_and_curriculum.json
    ├── evaluator_and_recovery.json
    ├── knowledge.json
    ├── language.json
    ├── memory.json
    ├── model_and_inference.json
    ├── robotics_and_hardware.json
    ├── rules_and_policy.json
    ├── skills.json
    ├── specialist_engines.json
    ├── training_and_self_evolution.json
    └── voice_and_interaction.json
```

---

## 2. The 16 Core Capabilities

1. **`skills`**: Dynamic skill discovery, parsing, certification, zero-trust authorization, and sandboxed execution.
2. **`memory`**: Multi-tier episodic memory recording, vector-based semantic retrieval, deduplication, and consolidation.
3. **`knowledge`**: Global knowledge base, domain partitioning, inverted index ranking, and axiomatic relationship graph.
4. **`cognitive_loop`**: TaraBrain 14-step cognitive loop orchestrating perception, context, guard, reasoning, deliberative tree search, and action.
5. **`specialist_engines`**: High-precision computational reasoning for Mathematics, Natural Science, Programming, Research, and Indian Cultural Ethics.
6. **`language`**: Native multilingual language engine, Indic/Kannada script detection, Kanglish parsing, and terminology translation.
7. **`rules_and_policy`**: Strict zero-tolerance policy enforcement, AST conflict resolution, policy rulebook compilation, and fail-closed execution gating.
8. **`access_and_security`**: Zero-trust identity, Creator authority, biometric security factors, QR device pairing, durable lockdown, and HMAC model weight integrity.
9. **`model_and_inference`**: 100% native Rust neural transformer inference, grouped query attention (GQA), SwiGLU activations, RoPE embeddings, RMSNorm, and byte-level BPE tokenization.
10. **`training_and_self_evolution`**: Native self-training, AdamW backpropagation, continual learning degradation control (EWC), candidate evaluation, and isolated model promotion.
11. **`dataset_and_curriculum`**: 35-point intelligence filtering, 15-tier educational curriculum ladder, streaming shard management, LSH deduplication, and benchmark decontamination.
12. **`voice_and_interaction`**: Multilingual wake-word spotting, acoustic energy tracking, barge-in cancellation interlock, and voice token streaming.
13. **`control_plane_and_workload`**: Distributed worker clustering, zero-cost compute routing, device capability negotiation, and multi-node state synchronization.
14. **`evaluator_and_recovery`**: Self-evaluation, task verification, uncertainty detection, and automated recovery fallbacks.
15. **`cryptographic_reward_and_goals`**: Creator roadmap goal tracking, cryptographic hash-chained reward ledger, milestone progression, and anti-tamper task completion verification.
16. **`robotics_and_hardware`**: Hardware Abstraction Layer (HAL) for robotic actuators, sensor fusion, motor kinematic limits, and physical destruction boundary enforcement.

---

## 3. Mandatory Operational Rules
### RULE — EXISTING CODE FIRST
Before creating any new file, function, module, or component:
1. Search the existing code first.
2. Reuse or extend existing code whenever possible.
3. Do not create duplicate implementations.
4. Create something new only when existing code cannot support the requirement.
5. Prove that need from the live source before creating it.
6. After changes, verify that no duplicate or broken links were introduced.

### The ADD Rule
Before adding any new feature, struct, function, or file:
1. **Search First**: Search `ARCHITECTURE/CAPABILITY_GRAPH.json` and `SOURCE_INDEX.json` to check if the capability or implementation already exists.
2. **Zero Duplicates**: Never create duplicate implementations or parallel shadow hierarchies.
3. **Reuse Existing Modules**: Extend or integrate with the existing capability module.
4. **Identify All Links**: Map required registration, discovery, execution, storage, memory, learning, safety, and test links.
5. **Never Claim Early Completion**: If any required link (e.g. registration, safety gate, or test) is missing, the feature is NOT complete.
6. **Update Architecture Map**: Regenerate and verify the architecture maps.

### The DELETE Rule
Before deleting any file, struct, function, or module:
1. **Identify Capability**: Check which capability the component belongs to in `CAPABILITY_GRAPH.json`.
2. **Trace All Callers**: Inspect `SOURCE_INDEX.json` and find all callers and dependent modules.
3. **Check System Links**: Verify if the component is registered in `TaraBrain`, `Server`, `ToolRegistry`, `ExecutionGuard`, or storage schemas.
4. **Block if Dependent**: If active dependencies exist, DO NOT delete.
5. **Complete Replacement First**: If replacing, implement and verify the replacement link before removing the old code.
6. **Verify Zero Orphans**: Run `cargo run --bin verify_architecture` after deletion to prove zero broken links.

---

## 4. Verification Command

Run the automated architecture and compiler verification suite at any time:
```powershell
cargo run --bin verify_architecture
```
This validates:
1. All capability implementation files exist on disk with zero broken links.
2. All 16 capability JSON specifications are coherent and intact.
3. `cargo check --workspace --all-targets` passes with **0 warnings and 0 errors**.
4. All indexed files in `SOURCE_INDEX.json` exist on disk (100% mutual consistency with zero ghost files).
