# CRITICAL CORE DIRECTIVES & OPERATIONAL RULES

1. **DON'T WASTE TIME — PLAN, ANALYZE PROS & CONS, THEN EXECUTE**:
   - **Don't Waste Time**: Zero tolerance for stalling, running in circles, unnecessary repetitive checks, over-engineering, or getting sidetracked by speculative issues. Respect user directives above all else: focus 100% on the exact objective given.
   - **Step 1 — First Make a Plan**: Clearly define the exact, minimal, and direct plan to achieve the objective before acting.
   - **Step 2 — Analyze Pros & Cons**: Thoroughly evaluate the trade-offs, architectural alignment, rule compliance, risks, and benefits before touching any code or files.
   - **Step 3 — Make Real Edits, Implementations, or Deletions**: Once planned and analyzed, execute cleanly and decisively — make genuine, production-quality edits, implementations, or deletions without hesitation, unnecessary friction, or redundant loops.

2. **PERSISTENT LIVE WATCHER DAEMON, AUTO-START & RESILIENT RECOVERY**:
   - **Persistent Background Daemon & Auto-Start**:
     * The Live Watcher must run as a persistent background service/daemon and automatically start with the project/system (`start.bat`, `start.sh`, Windows startup / service integration).
     * Manual startup (`cargo run ... watcher`) must NOT be required for normal daily developer operations.
   - **Mandatory Startup Full Reconciliation Before Live Watching**:
     * Whenever the daemon or server starts, restarts, or recovers, it MUST first execute an exhaustive Layer 2 Full Reconciliation scan (`reconcile_full`) to recover any missed changes during downtime or reboot before resuming Layer 1 live watching.
   - **Self-Healing & Crash Recovery Safeguard**:
     * If the background watcher encounters an unexpected error, crash, or thread interruption, it must recover automatically, execute a fresh reconciliation scan, and restore continuous monitoring.
   - **Manual Execution Boundary**:
     * Manual execution of `--watch` or `--reconcile` is strictly reserved for initial installation, diagnostic inspection, service registration, or standalone audits — never as a required manual step for routine daily developer operation.

3. **READ ALL RULES IN FULL & STRICT DOWNLOADER STANDALONE ISOLATION**:
   - **Read All Rules in Full (Never Skim or Read Headings Only)**:
     * Never read just the starting lines, titles, or headings of the rules.
     * You MUST read the entire, complete text of every rule from start to finish, understanding every specific mandate, boundary, and operational clause before proceeding.
   - **Strict Downloader Standalone Boundary (Zero External Links)**:
     * The `downloader/` folder must NEVER connect to, import from, or link with any project files outside (`tara_engine`, `tara_core`, `tara_server`, `rust/`, etc.).
     * Files inside the `downloader/` folder can ONLY link with files inside the `downloader/` folder itself.
     * Zero connection to the dataset engine, model compiler, tokenizer, or neural training/inference pipelines.
     * Downloaded datasets must only be written to their dedicated external storage folder (e.g., on the `D:` drive: `D:\taracore_datasets\downloaded`).

4. **NO FABRICATION, NO SYNTHETICS, NO MOCKS, NO SIMULATION**:
   - Zero tolerance for fake implementations, synthetic data, synthetic mock returns, stub methods, or simulated behaviors.
   - All components, models, engines, algorithms, and handlers must be genuine, working implementations.
   - **No Fabrication When Implementation is Difficult**: When an implementation is technically hard, complex, or blocked, NEVER fake, simulate, or fabricate code to bypass the difficulty. Face technical realities honestly.

5. **NO HARDCODED FILES OR FOLDER SHA DATA (DYNAMIC REGISTRIES ALLOWED)**:
   - Absolutely NO hardcoding of file SHAs, folder SHA data, directory tree hashes, dataset checksums, model weight signatures, or token hashes anywhere in source code, static manifests, tests, or configurations as static literals.
   - Never embed static hash tables, pre-calculated SHA digests, or hardcoded file/folder checksum lists.
   - All integrity verifications must calculate hashes dynamically at runtime from authentic filesystem state or read from genuine, dynamically-managed operational registries.
   - **Dynamic Runtime SHA-256 Registries & Operational Persistence (Explicit Allowed Exception)**:
     - General hardcoded SHA-256 literals remain strictly prohibited.
     - Saving, persisting, and updating dynamically computed SHA-256 hashes in runtime operational registries is explicitly **ALLOWED**:
       * **Architecture Sync Registries**: Saving dynamically computed SHA-256 digests in runtime registries (`ARCHITECTURE/file_registry.json`, `folder_registry.json`, `project_tree.json`, `architecture_state.json`) for change detection, move tracking, and filesystem integrity verification.
       * **Dataset Dynamic Ingestion, Repeat Exclusion & Blacklists**: Runtime dynamically created and updated SHA-256 registries for repeat download skipping via dynamic ledger, authentic downloaded content integrity checking, and dynamic quarantine support.
       * **Pure Runtime State (No Hardcoding)**: These registries are legitimate operational state when generated and updated dynamically at runtime from authentic filesystem state or live network streams. They are NEVER static literal tables embedded in source code, static manifests, or test fixtures.

6. **MANDATORY USER CONSULTATION ON PROBLEMATIC IMPLEMENTATIONS**:
   - If an implementation poses architectural conflicts, high complexity, blockers, difficult technical hurdles, or ambiguities, STOP and ask the user directly.
   - Present the root cause, analysis, and options transparently. Do not guess, assume, or bypass the problem with temporary or fabricated solutions.

7. **TARA IS NOT READY FOR RELEASE — NO PREMATURE CLAIMS OR FAKE MANIFESTS**:
   - TARA is in active development and is **NOT ready for production release**.
   - The previous 118K (118,080-parameter) smoke-test model has been deleted and is NOT a production model.
   - Do NOT generate fake release manifests, mock release packages, or claim release readiness until a genuine production model is trained, validated, and promoted.

8. **NO EXTERNAL AI CONNECTION**:
   - Do NOT connect to or depend on external AI providers (Gemini, OpenAI / ChatGPT, Anthropic / Claude, HuggingFace APIs, or any third-party cloud LLM) as a substitute for TARA's native neural model and engine.
   - All neural inference, embeddings, tokenization, training, and cognition must execute locally on the project's own native Rust engine and local models.

9. **NO RELIANCE ON OLD AUDIT REPORTS — NEW AUDITS FRESH FROM REALITY**:
   - Never rely on, refer to, or build conclusions upon old, stale, or previous audit reports.
   - Every audit must be conducted completely fresh ("new audit done by new") directly from current filesystem state and live execution proofs.
   - **Delete Old Audit Reports After Fixing**: Once an issue is investigated, resolved, and verified, all obsolete audit reports, stale test outputs, and temporary audit artifacts must be permanently deleted.

10. **ZERO TOLERANCE FOR CARGO WARNINGS — FIX VIA ROOT CAUSES, NO PATCHING**:
   - Never ignore compiler or cargo warnings. Zero compiler warnings permitted across all crates and targets.
   - When warnings or errors arise, resolve their fundamental root cause cleanly at the architecture/code origin.
   - Strictly NO patching, NO superficial silencing, NO `#[allow(...)]` suppressions to mask problems, and NO symptom-suppressing workarounds.

11. **REAL, VERIFIED, PRODUCTION-QUALITY CODE**:
   - Write code as production-quality code from the start.
   - Every feature must be fully implemented and actually connected end-to-end.
   - Every module, route, function, and subsystem must compile, link, and function in reality.

12. **VERIFICATION BEFORE CONFIRMATION**:
   - Never claim a feature is working unless it has been explicitly verified with tests or execution proofs.
   - If something cannot be implemented or verified due to system constraints or missing dependencies, state the truth directly. Do not guess, do not fabricate, and do not fake it.

13. **NO PATCH WORK — ROOT CAUSE RESOLUTION**:
   - Zero tolerance for quick hacks, temporary workarounds, symptom masking, or superficial patches.
   - Always investigate, identify, and fix the fundamental root cause of an issue cleanly at its origin.
   - If a proper fix requires major architectural, structural, or breaking changes, present the root cause, analysis, and proposed solution to the user and obtain confirmation before applying the changes.

14. **EXAMPLES FOR REFERENCE ONLY — NO EXAMPLE NAMES/DATA IN PRODUCTION**:
   - Examples, sample patterns, and example names are strictly for reference and understanding only.
   - Never inject example names, sample templates, dummy example identifiers, or example folders into project source code, directories, or production configurations.

15. **ZERO EXPOSURE OF SECRETS & IDENTITIES — NO HARDCODED CREDENTIALS**:
   - Do NOT expose secret keys, private keys, tokens, credentials, or developer/creator identities in source code, README files, documentation, or hardcoded strings.
   - All sensitive secrets must be managed securely through encrypted storage or runtime environment parameters, never committed directly into code or text files.

16. **MANDATORY WHOLE-SYSTEM ARCHITECTURE CONSULTATION BEFORE ANY EDITS OR AUDITS**:
   - The machine-readable maps in `ARCHITECTURE/` (`CAPABILITY_GRAPH.json`, `SOURCE_INDEX.json`, `DATA_FLOW.json`, `CONTROL_FLOW.json`, `CAPABILITIES/*.json`, and `README.md`) represent the authoritative whole-system architecture of TARA.
   - You MUST consult and inspect this architecture BEFORE modifying, adding, deleting, refactoring, or auditing any file, function, module, struct, trait, or capability across the codebase.
   - Always adhere to the strict **ADD Rule** (inspect existing implementations, reuse existing modules, connect required lifecycle links across registration, discovery, execution, storage, memory, learning, safety, and tests; update architecture maps) and **DELETE Rule** (identify all callers and dependents, cleanly rewire, update architecture maps).
   - Never perform isolated, blind, or file-by-file patches without understanding the complete cross-subsystem lifecycle.

17. **ZERO PYTHON TOOLS, ZERO TEMPORARY PYTHON SCRIPTS — 100% PURE NATIVE RUST WORKFLOW**:
   - Strictly ZERO Python tools, zero Python scripts, and zero temporary `.py` files permitted anywhere in the workspace or operational workflow.
   - All tools, scripts, utilities, verification checkers, dataset splitters, tokenizers, benchmarks, audits, and pipelines must be written and executed 100% in native Rust (`cargo`, compiled Rust binaries, native Rust integration tests).
   - Every architectural, dataset, security, and runtime verification must execute directly through compiled native Rust binaries.

18. **NO HARDCODED NUMBERS OR STATIC LIMITS — FULL DYNAMIC EXTENSIBILITY**:
   - Absolutely NO hardcoding of arbitrary numbers, static capacity caps, magic thresholds, fixed stage limits, or rigid counts across the entire codebase.
   - **Vocabulary**: Vocab size, token IDs, and special token boundaries must be read dynamically from genuine model `config.json`, `tokenizer.json`, or runtime vocabulary builders, never embedded as hardcoded literals.
   - **Skills**: Skill counts, tool catalogs, and capabilities must be dynamically discoverable and extensible; zero hardcoding of static skill limits or fixed skill sets.
   - **Model Self-Update, Training & Stages**: Zero hardcoded stage counts (no rigid 3-stage caps), fixed step ceilings, hardcoded epoch numbers, or locked batch sizes. All training pipelines must be extensible ($N$ stages) with dynamic hyperparameter configuration and seamless checkpoint continuation.
   - **Datasets**: Shard counts, record totals, batch partitions, and curriculum distributions must be calculated dynamically from authentic filesystem state and live registries, never statically hardcoded.
   - **Extensibility by Design**: Every subsystem, model, engine, tool, and orchestrator must be designed to be dynamically extensible, parameter-driven, and adaptable to future growth without requiring code rewrites or artificial ceilings.

19. **MANDATORY ARCHITECTURE SYNCHRONIZATION ON ANY ADDITION, WIRING, OR DELETION**:
   - Any new implementation, struct, module, or file created must be indexed and registered immediately into the authoritative architecture maps in `ARCHITECTURE/` (`CAPABILITY_GRAPH.json`, `SOURCE_INDEX.json`, `CAPABILITIES/<id>.json`).
   - Any wired connection, data flow, control flow, or cross-subsystem lifecycle link (registration, discovery, execution, storage, memory, learning, safety) must be explicitly mapped in `DATA_FLOW.json`, `CONTROL_FLOW.json`, and capability graph edges.
   - Any deleted, deprecated, or removed file, function, struct, or module must be cleanly unlinked and purged from architecture maps, and all dependent connections safely rewired.
   - **Post-Modification Architecture State Verification**:
     * After ANY file creation, code edit/update, file rename/move, or deletion across the workspace, ALWAYS inspect and verify the architecture state and registries (`ARCHITECTURE/project_tree.json`, `file_registry.json`, `folder_registry.json`, `architecture_state.json`, `SOURCE_INDEX.json`).
     * Confirm whether the Architecture Sync Engine / Live Watcher has successfully reflected and synchronized the exact changes.
     * If updated: proceed normally.
     * If NOT updated: immediately investigate WHY. Determine whether the Live Watcher, filesystem event listener, or reconciler encountered an issue. Identify the fundamental root cause at its origin, resolve it cleanly, and verify that synchronization is fully restored before proceeding. Zero tolerance for stale, desynchronized, or bypassed architecture state.
   - **Zero Ghost Implementations & Zero Orphaned Links**: The architecture maps in `ARCHITECTURE/` and the physical Rust code in `rust/` must maintain 100% mutual consistency at all times.
   - Every architectural modification must be verified with `cargo run --bin verify_architecture` to confirm 100% compliance before completing any task.

20. **EXISTING CODE FIRST**:
   - Before creating any new file, function, module, or component:
     1. Search the existing code first.
     2. Reuse or extend existing code whenever possible.
     3. Do not create duplicate implementations.
     4. Create something new only when existing code cannot support the requirement.
     5. Prove that need from the live source before creating it.
     6. After changes, verify that no duplicate or broken links were introduced.

21. **RIGOROUS DATASET PROVENANCE, COMMERCIAL RIGHTS & LICENSE VERIFICATION**:
   - **Worldwide Open-Source Diversity (No Religious, National, or Single-Site Confinement)**:
     * Datasets must be sourced broadly across worldwide open-source repositories, international academic archives, global software foundations, and universal scientific literature.
     * Ingestion must NEVER be restricted to or biased toward any specific religion, single country, or single host/site (e.g. zero monopoly by Hugging Face alone, GitHub alone, or any single vendor/geography).
     * Focus strictly on universal, objective human knowledge: mathematics, computer systems, algorithms, natural sciences, engineering, and global open-access educational literature with genuine permissive rights.
   - **No Blind Reliance on Mirror or Tag Names Alone**: NEVER ingest or download a dataset simply because a repository tag, card, or Hugging Face mirror label says "MIT" or "Apache-2.0". Always verify the upstream origin, original author datasheet, and authentic upstream repository terms.
   - **Mandatory Official Source Verification**: Every dataset source must be thoroughly checked and verified against its official upstream source, author/institution repository, or authoritative publisher terms.
   - **Mandatory 4-Way Rights Verification**: To be eligible for canonical ingestion, a dataset MUST genuinely permit:
     1. Commercial use allowed.
     2. Modification and adaptation allowed.
     3. Redistribution and dataset / AI model training allowed.
   - **Strict Prohibition of Restrictions (Zero Tolerance)**:
     * Non-Commercial (NC) restrictions: FORBIDDEN.
     * No-Derivatives (ND) restrictions: FORBIDDEN.
     * Research-only or Academic-only restrictions: FORBIDDEN.
     * Any commercial limitation or usage restriction: FORBIDDEN.
     * Unclear, ambiguous, or missing licenses: FORBIDDEN.
   - **Per-File / Per-Record License Gating for Mixed or Third-Party Content**:
     * When ingesting datasets containing third-party, multi-license, or heterogeneous content (e.g. CommitPackFT, OpenDSA, web crawls), NEVER apply a bulk or umbrella permissive label.
     * Every individual record and file must undergo strict per-record permissive license gating. All copyleft (AGPL, LGPL, GPL, MPL), NC, ND, SA, or unverified records must be dropped.
   - **Zero Synthetic & Zero AI-Generated Data**:
     * Absolute ban on synthetic, LLM-generated, ChatGPT/GPT-4/text-davinci distilled content (e.g. Alpaca, CodeAlpaca, SlimOrca, OpenHermes).
   - **Mandatory Provenance Ledger**:
     * For every ingested file and shard, record and persist: Authentic License Proof URL + Official Source URL + dynamically computed runtime SHA-256 in the authoritative registry.
   - **Continuous Autonomous Ingestion of Verified Compliant Sources**:
     * The pipeline must continuously discover, verify, and stream authentic, fully compliant sources until the target scale is reached, strictly enforcing all above gates.

22. **MANDATORY TEST FILE & FOLDER LIFECYCLE (PRESERVE DURING RUNS, COMPLETE CLEANUP POST-SUITE)**:
   - **No Premature Deletion During Test Cycles**:
     * Testಗಾಗಿ create ಮಾಡುವ ಯಾವುದೇ temporary file, folder, dataset, artifact, checkpoint, log ಅಥವಾ test workspace ಅನ್ನು **ಒಂದು test ಮುಗಿದ ತಕ್ಷಣ delete ಮಾಡಬಾರದು**.
     * Any temporary file, folder, dataset, artifact, checkpoint, log, or test workspace created for testing must **NEVER be deleted immediately after a single test finishes**.
     * Test artifacts ಅನ್ನು ಅಗತ್ಯವಿರುವಷ್ಟು **multiple test runs / repeated verification cycles**ಗಾಗಿ reuse ಮಾಡಬೇಕು.
     * Test artifacts must be preserved and reused across multiple test runs and repeated verification cycles so that repeated tests and debugging remain possible.
   - **Preservation During Active Verification**:
     * ಪ್ರತಿಯೊಂದು test run ನಂತರ test artifacts ಅನ್ನು preserve ಮಾಡಬೇಕು, zodat repeated tests ಮತ್ತು debugging ಸಾಧ್ಯವಾಗುತ್ತದೆ.
     * Deleting any test artifact before the entire test lifecycle / suite has fully completed is strictly prohibited (ಯಾವುದೇ test artifact ಅನ್ನು delete ಮಾಡುವುದು test lifecycle ಪೂರ್ಣಗೊಳ್ಳುವ ಮೊದಲು prohibited).
   - **Mandatory Final Cleanup Only After Full Completion**:
     * **ಸಂಪೂರ್ಣ test suite, verification, audit ಮತ್ತು required validation ಎಲ್ಲವೂ ಪೂರ್ಣಗೊಂಡ ನಂತರ ಮಾತ್ರ** ಎಲ್ಲಾ temporary test artifacts ಅನ್ನು mandatory ಆಗಿ delete ಮಾಡಬೇಕು.
     * All temporary test artifacts must be mandatorily deleted **ONLY after the complete test suite, verification, audit, and required validations have finished entirely**.
   - **Post-Cleanup Filesystem Verification**:
     * Final cleanup ನಂತರ test files/folders ಉಳಿದಿಲ್ಲ ಎಂದು filesystem scan ಮೂಲಕ verify ಮಾಡಬೇಕು.
     * Following final cleanup, an explicit filesystem scan must verify that zero test files, folders, or residual artifacts remain behind.
   - **Strict Separation from Production**:
     * Test artifacts ಅನ್ನು production data, production models ಅಥವಾ permanent project files ಎಂದು treat ಮಾಡಬಾರದು.
     * Test artifacts must never be treated as production data, production models, or permanent project files.
   - **Zero Partial or Forgotten Artifacts**:
     * Test cleanup itself must be actively executed and verified; **partial cleanup ಅಥವಾ forgotten test artifacts allowed ಅಲ್ಲ**.
     * Partial cleanup, residual files, or forgotten test artifacts are strictly prohibited.

---

---

## RULE — EXISTING CODE FIRST

Before creating any new file, function, module, or component:

1. Search the existing code first.
2. Reuse or extend existing code whenever possible.
3. Do not create duplicate implementations.
4. Create something new only when existing code cannot support the requirement.
5. Prove that need from the live source before creating it.
6. After changes, verify that no duplicate or broken links were introduced.
