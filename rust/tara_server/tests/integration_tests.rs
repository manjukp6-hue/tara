use tara_server::brain::TaraBrain;
use serde_json::json;
use std::collections::HashMap;

#[test]
fn test_tara_brain_initialization_and_inference() {
    let brain = TaraBrain::new(None).expect("Failed to initialize TaraBrain");
    assert!(brain.model.is_some(), "Production model should be loaded");

    let status = brain.get_model_status();
    assert_eq!(status["status"], "LOADED");
    assert_eq!(status["model_integrity"], "PASS");
    assert_eq!(status["model_sha256"], "7a50308b799f2654baeafbd64dec31f088c3b07b446e198cd0f9ec2b7c0af309");

    let mut context = HashMap::new();
    context.insert("session_id".to_string(), json!("test_session_rust"));

    let result = brain.process("tester", "Hello TARA Core!", context);
    assert_eq!(result["outcome"], "SUCCESS");
    assert_eq!(result["decision"], "ALLOW");
    assert!(result["final_response"].as_str().is_some());
}

#[test]
fn test_canonical_contracts_validation() {
    use tara_server::contract::{
        CanonicalInferenceRequest, CanonicalInferenceResponse, CANONICAL_MODEL_SHA256, CANONICAL_PROTOCOL_VERSION
    };

    let req = CanonicalInferenceRequest::new("req_100", "Testing Contract Parity");
    assert!(req.validate().is_ok());

    let serialized = serde_json::to_string(&req).expect("Failed to serialize");
    assert!(serialized.contains("req_100"));
    assert!(serialized.contains(CANONICAL_MODEL_SHA256));

    let deserialized: CanonicalInferenceRequest = serde_json::from_str(&serialized).expect("Failed to deserialize");
    assert_eq!(deserialized.request_id, "req_100");
    assert_eq!(deserialized.expected_model_checksum, CANONICAL_MODEL_SHA256);

    let resp = CanonicalInferenceResponse {
        request_id: "req_100".to_string(),
        status: "SUCCESS".to_string(),
        text: "TARA_LIVE_INFERENCE_OK".to_string(),
        model_checksum: CANONICAL_MODEL_SHA256.to_string(),
        runtime_engine: "rust_cpu".to_string(),
        token_count: 5,
        token_ids: vec![1, 2, 3],
        model_identity: "TARA".to_string(),
        first_latency_ms: 10.5,
        total_latency_ms: 25.2,
        tokens_per_second: 198.4,
    };
    let resp_str = serde_json::to_string(&resp).expect("Serialize resp");
    assert!(resp_str.contains("TARA_LIVE_INFERENCE_OK"));
    assert!(resp_str.contains(CANONICAL_PROTOCOL_VERSION.split('.').next().unwrap()));
}

#[test]
fn test_bridge_telemetry_and_failover() {
    use tara_server::bridge::UnifiedBridge;

    let bridge = UnifiedBridge::new();
    bridge.record_rust_dispatch();
    bridge.record_failover(14.85);

    let telem = bridge.get_telemetry();
    assert_eq!(telem["status"], "ACTIVE");
    assert_eq!(telem["metrics"]["dispatched_rust_cpu"], 1);
    assert_eq!(telem["metrics"]["automatic_failovers"], 1);
    assert_eq!(telem["metrics"]["last_failover_latency_ms"], 14.85);
}

#[test]
fn test_web_session_store() {
    let store = tara_server::server::WebSessionStore::new();
    let token = store.issue("web_tester");
    assert_eq!(token.len(), 64);

    let verified = store.verify(&token);
    assert_eq!(verified, Some("web_tester".to_string()));

    let invalid = store.verify("nonexistent_token");
    assert!(invalid.is_none());
}

#[test]
fn test_control_plane_workers_and_jobs() {
    use tara_server::control_plane::{ControlPlane, CANONICAL_MODEL_SHA256};

    let tmp_test_dir = std::env::temp_dir().join(format!("tara_cp_test_{}", rand::random::<u64>()));
    let _ = std::fs::create_dir_all(&tmp_test_dir);
    let cp = ControlPlane::new(tmp_test_dir.to_str().unwrap());
    let status = cp.get_status();
    assert_eq!(status["status"], "HEALTHY");
    assert_eq!(status["canonical_model_sha256"], CANONICAL_MODEL_SHA256);

    // Register a valid worker
    let reg_res = cp.workers.write().unwrap().register(&json!({
        "worker_id": "test_node_01",
        "name": "Test Cloud Worker",
        "provider": "cloudflare",
        "model_sha256": CANONICAL_MODEL_SHA256,
        "capabilities": ["chat", "inference"]
    })).expect("Register worker");
    assert_eq!(reg_res.status, "READY");

    // Register an invalid worker (mismatched model sha)
    let bad_reg = cp.workers.write().unwrap().register(&json!({
        "worker_id": "bad_node_01",
        "name": "Corrupted Worker",
        "provider": "generic_container",
        "model_sha256": "bad_sha_hash_value_12345"
    })).expect("Register bad worker");
    assert_eq!(bad_reg.status, "QUARANTINED");
    assert!(bad_reg.quarantine_reason.is_some());

    // Submit distributed job
    let items = vec![json!({"id": 1}), json!({"id": 2}), json!({"id": 3}), json!({"id": 4}), json!({"id": 5}), json!({"id": 6})];
    let job = cp.jobs.write().unwrap().submit_job("inference", items, 2, Some("idem_key_1"), None);
    assert_eq!(job.status, "PENDING");
    assert_eq!(job.chunks.len(), 3);

    // Idempotent resubmission
    let job_dup = cp.jobs.write().unwrap().submit_job("inference", vec![], 2, Some("idem_key_1"), None);
    assert_eq!(job_dup.job_id, job.job_id);

    // Assign chunks
    {
        let workers = cp.workers.read().unwrap();
        cp.jobs.write().unwrap().assign_pending_chunks(&workers);
    }
    let assigned_job = cp.jobs.read().unwrap().get_job(&job.job_id).unwrap();
    assert_eq!(assigned_job.status, "RUNNING");

    // Complete all chunks
    for chunk in assigned_job.chunks.iter() {
        cp.jobs.write().unwrap().complete_chunk(&job.job_id, &chunk.chunk_id, json!({"processed": true, "chunk": chunk.chunk_id})).unwrap();
    }

    let finished_job = cp.jobs.read().unwrap().get_job(&job.job_id).unwrap();
    assert_eq!(finished_job.status, "COMPLETED");
    assert_eq!(finished_job.results.len(), 3);

    // Provider deploy
    let deploy_res = cp.providers.write().unwrap().deploy("generic_container", &json!({"image": "tara:latest"})).unwrap();
    assert_eq!(deploy_res["status"], "SUCCESS");
    assert_eq!(deploy_res["provider"], "generic_container");
}

#[test]
fn test_voice_and_creator_contracts() {
    use tara_server::contract::{
        VoiceConverseRequest, VoiceConverseResponse,
        CreatorSetupRequest, CreatorSetupResponse,
        DeviceRegisterRequest, DeviceListResponse
    };

    // Voice Converse contract
    let voice_req = VoiceConverseRequest {
        input_text: Some("Hello TARA".to_string()),
        audio_data: None,
        language: "en-US".to_string(),
        context: None,
        require_wake_word: false,
    };
    let voice_req_json = serde_json::to_string(&voice_req).expect("serialize voice req");
    let back_req: VoiceConverseRequest = serde_json::from_str(&voice_req_json).expect("deserialize voice req");
    assert_eq!(back_req.input_text.as_deref(), Some("Hello TARA"));
    assert_eq!(back_req.language, "en-US");

    let voice_resp = VoiceConverseResponse {
        status: "SUCCESS".to_string(),
        transcript: "Hello TARA".to_string(),
        response_text: "Greetings! How may I assist you today?".to_string(),
        audio_base64: "UklGRgAAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA=".to_string(),
        language: "en-US".to_string(),
        turn_id: "turn_test_01".to_string(),
        brain_result: None,
    };
    let voice_resp_json = serde_json::to_string(&voice_resp).expect("serialize voice resp");
    let back_resp: VoiceConverseResponse = serde_json::from_str(&voice_resp_json).expect("deserialize voice resp");
    assert_eq!(back_resp.turn_id, "turn_test_01");

    // Creator Setup contract
    let setup_req = CreatorSetupRequest {
        google_id_token: "mock_jwt_token_sample".to_string(),
        confirm_identity: true,
        device_name: "Primary Dev PC".to_string(),
        device_public_key: Some("abcdef123456".to_string()),
    };
    let setup_req_json = serde_json::to_string(&setup_req).expect("serialize setup req");
    let back_setup: CreatorSetupRequest = serde_json::from_str(&setup_req_json).expect("deserialize setup req");
    assert_eq!(back_setup.device_name, "Primary Dev PC");
    assert!(back_setup.confirm_identity);

    let setup_resp = CreatorSetupResponse {
        status: "SUCCESS".to_string(),
        authority_state: "ACTIVE".to_string(),
        creator_id: "ROOT_OPERATOR".to_string(),
        display_name: "OPERATOR_ROOT".to_string(),
        recovery_code: Some("test_rec_code".to_string()),
        device: None,
        session: None,
    };
    let setup_resp_json = serde_json::to_string(&setup_resp).expect("serialize setup resp");
    let back_sresp: CreatorSetupResponse = serde_json::from_str(&setup_resp_json).expect("deserialize setup resp");
    assert_eq!(back_sresp.creator_id, "ROOT_OPERATOR");

    // Device Register contract
    let dev_req = DeviceRegisterRequest {
        device_public_key: "feedcafe01020304".to_string(),
        session_token: Some("sess_token_123".to_string()),
        google_id_token: None,
        device_name: "Secondary Laptop".to_string(),
    };
    let dev_json = serde_json::to_string(&dev_req).expect("serialize dev req");
    let back_dev: DeviceRegisterRequest = serde_json::from_str(&dev_json).expect("deserialize dev req");
    assert_eq!(back_dev.device_name, "Secondary Laptop");

    // Device List contract
    let list_resp = DeviceListResponse {
        status: "SUCCESS".to_string(),
        devices: vec![json!({"device_id": "TARA-DEVICE-001", "status": "AUTHORIZED"})],
    };
    let list_json = serde_json::to_string(&list_resp).expect("serialize list resp");
    let back_list: DeviceListResponse = serde_json::from_str(&list_json).expect("deserialize list resp");
    assert_eq!(back_list.devices.len(), 1);
}

#[test]
fn test_voice_acoustic_synthesizer() {
    use tara_server::routes::generate_acoustic_wav_bytes;

    let wav = generate_acoustic_wav_bytes("Hello from TARA cognitive core", "en-US");
    // Verify RIFF header
    assert!(wav.len() > 44, "WAV must have at least 44 header bytes + PCM samples");
    assert_eq!(&wav[0..4], b"RIFF");
    assert_eq!(&wav[8..12], b"WAVE");
    assert_eq!(&wav[12..16], b"fmt ");
    assert_eq!(&wav[36..40], b"data");

    // Verify Kannada synthesis pitch path
    let wav_kn = generate_acoustic_wav_bytes("ನಮಸ್ಕಾರ ತಾರಾ", "kn-IN");
    assert!(wav_kn.len() > 44);
    assert_eq!(&wav_kn[0..4], b"RIFF");
}

#[test]
fn test_identity_device_registry_and_setup() {
    use tara_server::access::{IdentityManager, CreatorAuthService};

    let tmp_test_dir = std::env::temp_dir().join(format!("tara_ident_test_{}", rand::random::<u64>()));
    let _ = std::fs::create_dir_all(&tmp_test_dir);
    let id_mgr = IdentityManager::new(tmp_test_dir.to_str().unwrap());

    // Register device
    let dev = id_mgr.register_device("0123456789abcdef", "Workstation Pro", "AUTHORIZED");
    assert_eq!(dev.device_name, "Workstation Pro");
    assert_eq!(dev.status, "AUTHORIZED");

    let devs = id_mgr.list_devices();
    assert_eq!(devs.len(), 1);

    // Revoke device
    let revoked = id_mgr.revoke_device(&dev.device_id);
    assert!(revoked);

    let devs_after = id_mgr.list_devices();
    assert_eq!(devs_after[0].status, "REVOKED");

    // Setup creator
    let auth_svc = CreatorAuthService::new(tmp_test_dir.to_str().unwrap());
    let setup_res = auth_svc.setup_creator(
        "mock_google_id_token",
        true,
        "Laptop Primary",
        Some("aabbccddeeff"),
        &id_mgr,
    );
    assert_eq!(setup_res["status"], "SUCCESS");
    assert_eq!(setup_res["creator_id"], "ROOT_OPERATOR");
    assert_eq!(setup_res["authority_state"], "ACTIVE");
    assert!(setup_res["recovery_code"].as_str().is_some());
    assert!(setup_res["session"]["session_token"].as_str().is_some());
}

#[test]
fn test_universal_runtime_gate_and_registry() {
    use std::sync::Arc;
    use tara_server::runtime::{
        DynamicRuntimeRegistry, UniversalRuntimeGate, RuntimeRecord, GateVerdict, RuntimeEvaluationResult
    };

    let registry = Arc::new(DynamicRuntimeRegistry::new());

    // 1. Initial required runtimes: Python and Rust
    let required = registry.list_required_runtimes();
    assert_eq!(required.len(), 2);
    let mut req_names: Vec<String> = required.into_iter().map(|r| r.runtime_id).collect();
    req_names.sort();
    assert_eq!(req_names, vec!["python", "rust"]);

    // 2. Parity matrix
    let matrix = registry.generate_parity_matrix();
    assert!(matrix["all_parity_passed"].as_bool().unwrap());
    assert!(matrix["matrix"]["python"].is_object());
    assert!(matrix["matrix"]["rust"].is_object());

    // 3. Pre-continuation check
    let (ready, msg, _) = registry.verify_runtime_ready("rust");
    assert!(ready);
    assert_eq!(msg, "TARA_RUNTIME_READY");

    // 4. Register new runtime: C++
    let cpp_rec = RuntimeRecord::new("cpp", "C++", true);
    let reg_res = registry.register_runtime(cpp_rec);
    assert!(reg_res.is_ok());

    let required_3 = registry.list_required_runtimes();
    assert_eq!(required_3.len(), 3);

    // 5. Universal Promotion Gate evaluation
    let gate = UniversalRuntimeGate::new(registry.clone());

    // Case A: Python PASS, Rust PASS, C++ FAIL -> Gate FAIL (quarantined)
    let mut evals_fail = HashMap::new();
    evals_fail.insert("python".to_string(), RuntimeEvaluationResult {
        runtime_id: "python".to_string(),
        passed: true,
        status: "VERIFIED".to_string(),
        error_message: None,
    });
    evals_fail.insert("rust".to_string(), RuntimeEvaluationResult {
        runtime_id: "rust".to_string(),
        passed: true,
        status: "VERIFIED".to_string(),
        error_message: None,
    });
    evals_fail.insert("cpp".to_string(), RuntimeEvaluationResult {
        runtime_id: "cpp".to_string(),
        passed: false,
        status: "FAILED".to_string(),
        error_message: Some("Compiler syntax error in native C++ skill adapter".to_string()),
    });

    let resp_fail = gate.evaluate_evolution_candidate(
        "SKILL",
        "invoice_generation",
        "1.1.0",
        &evals_fail,
    );
    assert_eq!(resp_fail.gate_verdict, GateVerdict::RejectedRuntimeFailure);
    assert!(!resp_fail.eligible_for_promotion);
    assert!(resp_fail.quarantined);

    // Case B: Python PASS, Rust PASS, C++ PASS -> Gate PASS
    let mut evals_pass = evals_fail.clone();
    evals_pass.insert("cpp".to_string(), RuntimeEvaluationResult {
        runtime_id: "cpp".to_string(),
        passed: true,
        status: "VERIFIED".to_string(),
        error_message: None,
    });

    let resp_pass = gate.evaluate_evolution_candidate(
        "SKILL",
        "invoice_generation",
        "1.1.0",
        &evals_pass,
    );
    assert_eq!(resp_pass.gate_verdict, GateVerdict::Approved);
    assert!(resp_pass.eligible_for_promotion);
    assert!(!resp_pass.quarantined);

    // 6. Authenticated Retirement
    // Unauthenticated fails
    let unauth = registry.retire_runtime("cpp", false, "Decommissioning C++ prototype");
    assert!(unauth.is_err());

    // Authenticated succeeds
    let auth = registry.retire_runtime("cpp", true, "Decommissioning C++ prototype");
    assert!(auth.is_ok());

    let required_after = registry.list_required_runtimes();
    assert_eq!(required_after.len(), 2);
}




