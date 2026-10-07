//! Live End-to-End Runtime Route Verification for TARA Architecture.
//!
//! Directly exercises the ApiRouter, TaraBrain, SpecialistEngines (Math, Science, Programming),
//! LanguageEngine, and RewardSystem across live HTTP route handler invocations.

use serde_json::json;
use std::collections::HashMap;
use std::sync::Arc;
use tara_server::brain::TaraBrain;
use tara_server::routes::register_all_routes;
use tara_server::server::{ApiRouter, RouteRequest};

fn create_test_request(method: &str, path: &str, body: serde_json::Value) -> RouteRequest {
    RouteRequest {
        method: method.to_uppercase(),
        path: path.to_string(),
        query_params: HashMap::new(),
        body,
        headers: HashMap::new(),
        client_ip: "127.0.0.1".into(),
        actor: Some("creator".into()),
    }
}

#[test]
fn test_live_runtime_route_verification() {
    let brain = Arc::new(TaraBrain::new(None).expect("TaraBrain failed to initialize"));
    let router = ApiRouter::new();
    register_all_routes(&router);

    // 1. Health check
    let req = create_test_request("GET", "/health", json!({}));
    let (handler, _, _) = router
        .match_route("GET", "/health")
        .expect("/health route missing");
    let res = handler.handle(&req, &brain);
    assert_eq!(res["status"].as_str(), Some("HEALTHY"));

    // 2. Math Engine Route
    let req = create_test_request(
        "POST",
        "/api/v1/engines/math",
        json!({
            "operation": "add",
            "a": "999999999999999999999999999",
            "b": "1"
        }),
    );
    let (handler, _, _) = router
        .match_route("POST", "/api/v1/engines/math")
        .expect("/api/v1/engines/math missing");
    let res = handler.handle(&req, &brain);
    assert_eq!(res["status"].as_str(), Some("SUCCESS"));
    assert_eq!(
        res["result"]["exact_result"]["result"].as_str(),
        Some("1000000000000000000000000000")
    );

    // 3. Science Engine Route
    let req = create_test_request(
        "POST",
        "/api/v1/engines/science",
        json!({
            "discipline": "physics",
            "operation": "force",
            "mass": 10.0,
            "acceleration": 9.8
        }),
    );
    let (handler, _, _) = router
        .match_route("POST", "/api/v1/engines/science")
        .expect("/api/v1/engines/science missing");
    let res = handler.handle(&req, &brain);
    assert_eq!(res["status"].as_str(), Some("SUCCESS"));
    assert_eq!(
        res["result"]["calculated_result"]["force_newtons"].as_f64(),
        Some(98.0)
    );

    // 4. Programming Engine Route
    let req = create_test_request(
        "POST",
        "/api/v1/engines/programming",
        json!({
            "operation": "check_delimiters",
            "code": "fn main() { let x = (1 + 2); }"
        }),
    );
    let (handler, _, _) = router
        .match_route("POST", "/api/v1/engines/programming")
        .expect("/api/v1/engines/programming missing");
    let res = handler.handle(&req, &brain);
    assert_eq!(res["status"].as_str(), Some("SUCCESS"));
    assert_eq!(res["result"]["analysis"]["balanced"].as_bool(), Some(true));

    // 5. Math Engine Numerical Integration (Monte Carlo Pi)
    let req = create_test_request(
        "POST",
        "/api/v1/engines/math",
        json!({
            "operation": "monte_carlo_pi",
            "samples": 50000,
            "seed": 12345
        }),
    );
    let (handler, _, _) = router
        .match_route("POST", "/api/v1/engines/math")
        .expect("/api/v1/engines/math missing");
    let res = handler.handle(&req, &brain);
    assert_eq!(res["status"].as_str(), Some("SUCCESS"));
    let pi_est = res["result"]["exact_result"]["estimated_pi"]
        .as_f64()
        .unwrap();
    assert!((pi_est - std::f64::consts::PI).abs() < 0.05);

    // 5b. Science Engine Kinematic Aerodynamic Drag Modeling
    let req = create_test_request(
        "POST",
        "/api/v1/engines/science",
        json!({
            "discipline": "physics",
            "operation": "projectile_drag",
            "v0": 50.0,
            "angle": 45.0,
            "mass": 1.0
        }),
    );
    let (handler, _, _) = router
        .match_route("POST", "/api/v1/engines/science")
        .expect("/api/v1/engines/science missing");
    let res = handler.handle(&req, &brain);
    assert_eq!(res["status"].as_str(), Some("SUCCESS"));
    assert!(
        res["result"]["calculated_result"]["range_meters"]
            .as_f64()
            .unwrap()
            > 0.0
    );

    // 6. Language Engine Route (Analyze)
    let req = create_test_request(
        "POST",
        "/api/v1/language/analyze",
        json!({
            "text": "ನಮಸ್ಕಾರ, kannada mathadu"
        }),
    );
    let (handler, _, _) = router
        .match_route("POST", "/api/v1/language/analyze")
        .expect("/api/v1/language/analyze missing");
    let res = handler.handle(&req, &brain);
    assert_eq!(res["status"].as_str(), Some("SUCCESS"));
    assert!(res["profile"]["is_mixed_or_code_switched"]
        .as_bool()
        .unwrap());

    // 7. Language Engine Route (Terminology Lookup)
    let req = create_test_request(
        "POST",
        "/api/v1/language/terminology",
        json!({
            "term": "gravity"
        }),
    );
    let (handler, _, _) = router
        .match_route("POST", "/api/v1/language/terminology")
        .expect("/api/v1/language/terminology missing");
    let res = handler.handle(&req, &brain);
    assert_eq!(res["status"].as_str(), Some("SUCCESS"));
    assert_eq!(res["translation"]["kannada"].as_str(), Some("ಗುರುತ್ವಾಕರ್ಷಣೆ"));

    // 8. Reward System Route (Event)
    let req = create_test_request(
        "POST",
        "/api/v1/reward/event",
        json!({
            "category": "task_completion",
            "score": 10.0,
            "reason": "Successfully verified live router dispatch"
        }),
    );
    let (handler, _, _) = router
        .match_route("POST", "/api/v1/reward/event")
        .expect("/api/v1/reward/event missing");
    let res = handler.handle(&req, &brain);
    assert_eq!(res["status"].as_str(), Some("SUCCESS"));

    // 9. Reward System Route (Summary & Integrity Verification)
    let req = create_test_request("POST", "/api/v1/reward/verify", json!({}));
    let (handler, _, _) = router
        .match_route("POST", "/api/v1/reward/verify")
        .expect("/api/v1/reward/verify missing");
    let res = handler.handle(&req, &brain);
    assert_eq!(res["status"].as_str(), Some("SUCCESS"));
    assert_eq!(res["ledger_valid"].as_bool(), Some(true));

    // 10. Brain Cognitive Loop turn execution with Specialist Engine resolution
    let brain_res = brain.process("creator", "calculate 25 * 4", HashMap::new());
    assert_eq!(brain_res["outcome"].as_str(), Some("SUCCESS"));
    assert!(brain_res["final_response"]
        .as_str()
        .unwrap()
        .contains("100"));
}
