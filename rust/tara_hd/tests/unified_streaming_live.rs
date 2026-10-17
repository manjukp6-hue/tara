//! Live Execution Verification for Universal Streaming, Token-Boundary Cancellation,
//! In-Flight Backpressure, and Multi-Tier Memory Placement in TARA-HD.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;
use tara_hd::event::{CancellationToken, StreamEvent};
use tara_hd::memory_governor::{
    MemoryExecutionMode, MemoryGovernor, ModelMemoryBudget, PlacementDecision,
};
use tara_hd::streaming::UniversalStreamEngine;
use tara_hd::telemetry::RuntimeTelemetrySnapshot;
use tara_hd::types::DeviceKind;
use tara_hd::ComputeChannel;

#[test]
fn test_live_universal_streaming_and_backpressure() {
    let engine = UniversalStreamEngine::new(4096); // 4KB in-flight memory ceiling
    let (sender, receiver) = engine.open_stream(4);

    let produced_count = Arc::new(AtomicUsize::new(0));
    let prod_clone = produced_count.clone();

    // Spawn producer thread
    let handle = thread::spawn(move || {
        for step in 0..10 {
            let event = StreamEvent::TokenChunk {
                sequence_id: 1,
                step,
                token_id: (100 + step) as u32,
                text: format!("token_{} ", step),
                is_final: step == 9,
            };
            sender.send(event).expect("Send must succeed");
            prod_clone.fetch_add(1, Ordering::SeqCst);
        }
        sender
            .send(StreamEvent::EndOfStream {
                sequence_id: 1,
                total_tokens: 10,
                elapsed_duration: Duration::from_millis(50),
            })
            .expect("EOS must succeed");
    });

    let mut received_tokens = Vec::new();
    loop {
        match receiver.recv().expect("Recv must succeed") {
            StreamEvent::TokenChunk { text, is_final, .. } => {
                received_tokens.push(text);
                if is_final {
                    // Consumer delay to verify backpressure flow
                    thread::sleep(Duration::from_millis(5));
                }
            }
            StreamEvent::EndOfStream { total_tokens, .. } => {
                assert_eq!(total_tokens, 10);
                break;
            }
            _ => {}
        }
    }

    handle.join().unwrap();
    assert_eq!(received_tokens.len(), 10);
    assert_eq!(produced_count.load(Ordering::SeqCst), 10);
    // Verified: in-flight bytes successfully returned to governor after reception
    assert_eq!(engine.governor().current_in_flight_bytes(), 0);
}

#[test]
fn test_live_token_boundary_cancellation_coordination() {
    let cancel_token = CancellationToken::new();
    let engine = UniversalStreamEngine::new(1024 * 1024);
    let (sender, receiver) = engine.open_stream(10);

    // Cancel from receiver side
    receiver.cancel();
    assert!(receiver.is_cancelled());
    assert!(cancel_token.is_cancelled() == false); // Independent tokens

    // Sender must immediately observe cancellation without wasting compute
    let event = StreamEvent::TokenChunk {
        sequence_id: 42,
        step: 0,
        token_id: 999,
        text: "should_fail".to_string(),
        is_final: false,
    };
    let send_res = sender.send(event);
    assert!(send_res.is_err());
    assert!(send_res.unwrap_err().contains("cancelled"));
}

#[test]
fn test_live_memory_budget_equation_and_placement() {
    // 100M parameter model test
    let budget = ModelMemoryBudget::estimate(100_000_000, 2, 2048, 1024, 16);
    let total_req = budget.total_required_bytes();
    assert!(total_req > 200_000_000); // Must account for weights + KV + activations + workspace

    // Simulate 16GB free VRAM
    let high_vram_telemetry = RuntimeTelemetrySnapshot {
        total_host_ram_bytes: 32 * 1024 * 1024 * 1024,
        free_host_ram_bytes: 16 * 1024 * 1024 * 1024,
        total_device_vram_bytes: Some(16 * 1024 * 1024 * 1024),
        free_device_vram_bytes: Some(14 * 1024 * 1024 * 1024),
        device_kind: DeviceKind::Gpu,
    };

    let decision = MemoryGovernor::evaluate_placement(
        &budget,
        &high_vram_telemetry,
        MemoryExecutionMode::HighPerformance,
        16,
    );
    assert!(matches!(decision, PlacementDecision::FullResident { .. }));

    // Simulate constrained 256MB VRAM
    let constrained_telemetry = RuntimeTelemetrySnapshot {
        total_host_ram_bytes: 16 * 1024 * 1024 * 1024,
        free_host_ram_bytes: 8 * 1024 * 1024 * 1024,
        total_device_vram_bytes: Some(2 * 1024 * 1024 * 1024),
        free_device_vram_bytes: Some(256 * 1024 * 1024),
        device_kind: DeviceKind::Gpu,
    };

    let partial_decision = MemoryGovernor::evaluate_placement(
        &budget,
        &constrained_telemetry,
        MemoryExecutionMode::ConstrainedMemorySaving,
        16,
    );
    assert!(matches!(
        partial_decision,
        PlacementDecision::PartialOffload { .. }
    ));
}

#[test]
fn test_live_compute_channel_integration() {
    let channel = ComputeChannel::auto().expect("Host compute channel must initialize");
    let telemetry = channel.live_telemetry();
    assert!(telemetry.total_host_ram_bytes > 0);
    assert!(telemetry.free_host_ram_bytes > 0);

    let stream = channel.universal_stream(1024 * 1024);
    assert_eq!(stream.governor().max_in_flight_bytes(), 1024 * 1024);
}
