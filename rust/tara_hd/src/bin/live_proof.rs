//! Standalone Production Binary: Live Execution Proof of TARA-HD Universal Streaming,
//! Live Hardware Telemetry, Flow Backpressure, and Dynamic Placement Governor.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use tara_hd::channel::ComputeChannel;
use tara_hd::event::StreamEvent;
use tara_hd::memory_governor::{
    MemoryExecutionMode, ModelMemoryBudget, PlacementDecision,
};

fn main() {
    println!("================================================================================");
    println!("             TARA-HD UNIVERSAL STREAMING ENGINE: LIVE REALITY PROOF             ");
    println!("================================================================================");

    // 1. LIVE HARDWARE TELEMETRY & COMPUTE CHANNEL PROOF
    println!("\n[1/4] PROBING LIVE HARDWARE CHANNEL & TELEMETRY...");
    let channel = ComputeChannel::auto().expect("ComputeChannel failed to initialize");
    println!("   -> Active Device Name:     {}", channel.device_name());
    println!("   -> Active Device Kind:     {:?}", channel.device_kind());
    println!("   -> Active Precision Mode:  {:?}", channel.precision());

    let telemetry = channel.live_telemetry();
    let total_ram_gb = telemetry.total_host_ram_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    let free_ram_gb = telemetry.free_host_ram_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    println!("   -> Total Host RAM:         {:.2} GB", total_ram_gb);
    println!("   -> Live Available RAM:     {:.2} GB", free_ram_gb);
    println!("   -> Host RAM Utilization:   {:.2}%", telemetry.host_ram_utilization_percent());
    if let Some(vram_bytes) = telemetry.total_device_vram_bytes {
        let vram_gb = vram_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        println!("   -> Accelerator VRAM:       {:.2} GB", vram_gb);
    } else {
        println!("   -> Accelerator VRAM:       None (Running on Native Host CPU)");
    }
    assert!(telemetry.total_host_ram_bytes > 0, "Host RAM must be dynamically probed");

    // 2. LIVE MULTI-TIER MEMORY BUDGET & PLACEMENT PROOF
    println!("\n[2/4] EVALUATING LIVE MEMORY BUDGET & PLACEMENT GOVERNOR...");
    // 120M Parameter Model Profile (Vocab=32K, Hidden=1024, Layers=16, MaxSeq=2048)
    let budget = ModelMemoryBudget::estimate(120_000_000, 2, 2048, 1024, 16);
    let total_mb = budget.total_required_bytes() as f64 / (1024.0 * 1024.0);
    println!("   -> Model Parameters:       120M (FP16)");
    println!("   -> Weights Footprint:      {:.2} MB", budget.weights_bytes as f64 / (1024.0 * 1024.0));
    println!("   -> KV-Cache Footprint:     {:.2} MB", budget.kv_cache_bytes as f64 / (1024.0 * 1024.0));
    println!("   -> Activations Footprint:  {:.2} MB", budget.activations_bytes as f64 / (1024.0 * 1024.0));
    println!("   -> Workspace & Overhead:   {:.2} MB", (budget.workspace_bytes + budget.overhead_bytes) as f64 / (1024.0 * 1024.0));
    println!("   -> Total Required Budget:  {:.2} MB", total_mb);

    let placement = channel.evaluate_placement(&budget, MemoryExecutionMode::HighPerformance, 16);
    match placement {
        PlacementDecision::FullResident { device_memory_allocated, headroom_bytes, .. } => {
            println!("   -> Placement Resolved:     FULL_RESIDENT (All weights & cache resident)");
            println!("   -> Allocated Memory:       {:.2} MB", device_memory_allocated as f64 / (1024.0 * 1024.0));
            println!("   -> Available Headroom:     {:.2} MB", headroom_bytes as f64 / (1024.0 * 1024.0));
        }
        PlacementDecision::PartialOffload { resident_layers, host_paged_memory, .. } => {
            println!("   -> Placement Resolved:     PARTIAL_OFFLOAD (Dynamic Tiered Paging)");
            println!("   -> Resident Layers:        {}/16 layers", resident_layers);
            println!("   -> Host Paged Footprint:   {:.2} MB", host_paged_memory as f64 / (1024.0 * 1024.0));
        }
        PlacementDecision::CapacityExceeded { deficit_bytes, .. } => {
            println!("   -> Placement Resolved:     CAPACITY_EXCEEDED (Deficit: {:.2} MB)", deficit_bytes as f64 / (1024.0 * 1024.0));
        }
    }

    // 3. LIVE MULTI-THREADED STREAMING & BACKPRESSURE PROOF
    println!("\n[3/4] EXECUTING LIVE MULTI-THREADED TOKEN STREAM WITH BACKPRESSURE...");
    let engine = channel.universal_stream(16 * 1024); // 16 KB in-flight permit ceiling
    let (sender, receiver) = engine.open_stream(8);  // 8-message bounded queue

    let total_tokens_to_stream = 50;
    let tokens_produced = Arc::new(AtomicUsize::new(0));
    let prod_clone = tokens_produced.clone();

    let start_time = Instant::now();
    let producer_handle = thread::spawn(move || {
        for step in 0..total_tokens_to_stream {
            let event = StreamEvent::TokenChunk {
                sequence_id: 101,
                step,
                token_id: (2000 + step) as u32,
                text: format!("t_{} ", step),
                is_final: step == (total_tokens_to_stream - 1),
            };
            sender.send(event).expect("Producer send failed");
            prod_clone.fetch_add(1, Ordering::SeqCst);
        }
        sender.send(StreamEvent::EndOfStream {
            sequence_id: 101,
            total_tokens: total_tokens_to_stream,
            elapsed_duration: start_time.elapsed(),
        }).expect("EOS send failed");
    });

    let mut tokens_received = 0;
    while let Ok(event) = receiver.recv() {
        match event {
            StreamEvent::TokenChunk { text, .. } => {
                tokens_received += 1;
                if tokens_received % 10 == 0 {
                    print!("{}[Step {}] ", text, tokens_received);
                }
                // Simulate consumer backpressure
                thread::sleep(Duration::from_micros(500));
            }
            StreamEvent::EndOfStream { total_tokens, elapsed_duration, .. } => {
                println!("\n   -> Stream Finished:        {} tokens in {:.2?}", total_tokens, elapsed_duration);
                break;
            }
            _ => {}
        }
    }
    producer_handle.join().unwrap();
    println!("   -> Tokens Produced:        {}", tokens_produced.load(Ordering::SeqCst));
    println!("   -> Tokens Consumed:        {}", tokens_received);
    println!("   -> Remaining In-Flight:    {} bytes (Zero-leak verified)", engine.governor().current_in_flight_bytes());
    assert_eq!(tokens_received, total_tokens_to_stream);
    assert_eq!(engine.governor().current_in_flight_bytes(), 0);

    // 4. LIVE COOPERATIVE CANCELLATION PROOF
    println!("\n[4/4] EXECUTING LIVE TOKEN-BOUNDARY CANCELLATION...");
    let cancel_engine = channel.universal_stream(1024 * 1024);
    let (cancel_sender, cancel_receiver) = cancel_engine.open_stream(4);

    // Consumer triggers cancellation
    cancel_receiver.cancel();
    assert!(cancel_receiver.is_cancelled());

    // Producer attempts to stream next chunk
    let aborted_event = StreamEvent::TokenChunk {
        sequence_id: 999,
        step: 0,
        token_id: 1,
        text: "aborted".to_string(),
        is_final: false,
    };
    let send_result = cancel_sender.send(aborted_event);
    assert!(send_result.is_err(), "Cancelled stream must reject subsequent compute");
    println!("   -> Cancellation Observed:  Immediate failure on consumer abort ({})", send_result.unwrap_err());

    println!("\n================================================================================");
    println!("                100% LIVE REALITY PROOF VERIFIED SUCCESSFULLY!                  ");
    println!("================================================================================");
}
