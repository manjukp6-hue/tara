//! Comprehensive verification tests for TARA-HD dynamic hardware orchestration.

use tara_hd::device::{ComputeDevice, CustomDevice};
use tara_hd::discovery::DiscoveredInventory;
use tara_hd::streaming::{CpuStreamHelper, StreamConfig};
use tara_hd::types::{DeviceCapability, DeviceKind, PrecisionMode, SelectionPolicy};
use tara_hd::ComputeChannel;

#[test]
fn test_hardware_discovery_and_cpu_baseline() {
    let inventory = DiscoveredInventory::probe_host();
    assert_eq!(inventory.cpu.kind, DeviceKind::Cpu);
    assert!(inventory.cpu.total_memory_bytes > 0);
    assert!(inventory.cpu.supports_matrix_multiply);
}

#[test]
fn test_priority_hierarchy_resolution() {
    let mock_gpu = DeviceCapability {
        kind: DeviceKind::Gpu,
        device_id: 0,
        name: "Test GPU".to_string(),
        backend: "cuda".to_string(),
        total_memory_bytes: 16 * 1024 * 1024 * 1024,
        available_memory_bytes: Some(15 * 1024 * 1024 * 1024),
        supported_precisions: vec![PrecisionMode::Fp16, PrecisionMode::Fp32],
        supports_matrix_multiply: true,
        supports_fused_attention: true,
        supports_in_place_optimizer: true,
    };

    let mock_tpu = DeviceCapability {
        kind: DeviceKind::Tpu,
        device_id: 0,
        name: "Test TPU".to_string(),
        backend: "tpu_xla".to_string(),
        total_memory_bytes: 32 * 1024 * 1024 * 1024,
        available_memory_bytes: None,
        supported_precisions: vec![PrecisionMode::Bf16, PrecisionMode::Fp32],
        supports_matrix_multiply: true,
        supports_fused_attention: true,
        supports_in_place_optimizer: true,
    };

    let mock_cpu = DeviceCapability {
        kind: DeviceKind::Cpu,
        device_id: 0,
        name: "Test CPU".to_string(),
        backend: "cpu".to_string(),
        total_memory_bytes: 32 * 1024 * 1024 * 1024,
        available_memory_bytes: None,
        supported_precisions: vec![PrecisionMode::Fp32],
        supports_matrix_multiply: true,
        supports_fused_attention: false,
        supports_in_place_optimizer: true,
    };

    // When both GPU and TPU and CPU are present: GPU must win
    let inv_all = DiscoveredInventory {
        gpus: vec![mock_gpu.clone()],
        tpus: vec![mock_tpu.clone()],
        cpu: mock_cpu.clone(),
        custom: Vec::new(),
    };
    let resolved_gpu = inv_all.resolve_device(&SelectionPolicy::Auto).expect("resolve gpu");
    assert_eq!(resolved_gpu.kind, DeviceKind::Gpu);

    // When GPU is absent but TPU and CPU are present: TPU must win
    let inv_no_gpu = DiscoveredInventory {
        gpus: Vec::new(),
        tpus: vec![mock_tpu.clone()],
        cpu: mock_cpu.clone(),
        custom: Vec::new(),
    };
    let resolved_tpu = inv_no_gpu.resolve_device(&SelectionPolicy::Auto).expect("resolve tpu");
    assert_eq!(resolved_tpu.kind, DeviceKind::Tpu);

    // When GPU and TPU are absent: CPU is the final fallback
    let inv_cpu_only = DiscoveredInventory {
        gpus: Vec::new(),
        tpus: Vec::new(),
        cpu: mock_cpu.clone(),
        custom: Vec::new(),
    };
    let resolved_cpu = inv_cpu_only.resolve_device(&SelectionPolicy::Auto).expect("resolve cpu");
    assert_eq!(resolved_cpu.kind, DeviceKind::Cpu);
}

#[test]
fn test_explicit_selection_with_and_without_fallback() {
    let mock_cpu = DeviceCapability {
        kind: DeviceKind::Cpu,
        device_id: 0,
        name: "Test CPU".to_string(),
        backend: "cpu".to_string(),
        total_memory_bytes: 16 * 1024 * 1024 * 1024,
        available_memory_bytes: None,
        supported_precisions: vec![PrecisionMode::Fp32],
        supports_matrix_multiply: true,
        supports_fused_attention: false,
        supports_in_place_optimizer: true,
    };

    let empty_inv = DiscoveredInventory {
        gpus: Vec::new(),
        tpus: Vec::new(),
        cpu: mock_cpu.clone(),
        custom: Vec::new(),
    };

    // Explicit GPU without fallback on missing hardware must return an error (never silent fake)
    let explicit_err = empty_inv.resolve_device(&SelectionPolicy::Explicit(DeviceKind::Gpu, None));
    assert!(explicit_err.is_err());

    // Explicit GPU with fallback on missing hardware must safely fall back to CPU
    let fallback_ok = empty_inv.resolve_device(&SelectionPolicy::ExplicitWithFallback(DeviceKind::Gpu, None));
    assert!(fallback_ok.is_ok());
    assert_eq!(fallback_ok.unwrap().kind, DeviceKind::Cpu);
}

#[test]
fn test_on_demand_chunked_streaming() {
    let stream_helper = CpuStreamHelper::with_config(StreamConfig {
        chunk_size_tokens: 64,
        max_resident_layers: 2,
        memory_budget_mb: 2048,
    });

    let full_sequence: Vec<u32> = (0..200).collect();
    let chunks: Vec<_> = stream_helper.stream_sequence_chunks(&full_sequence).collect();

    // 200 tokens with chunk size 64 should yield 4 chunks: 64, 64, 64, 8
    assert_eq!(chunks.len(), 4);
    assert_eq!(chunks[0].tokens.len(), 64);
    assert_eq!(chunks[0].start_idx, 0);
    assert_eq!(chunks[0].end_idx, 64);
    assert!(!chunks[0].is_last_chunk);

    assert_eq!(chunks[3].tokens.len(), 8);
    assert_eq!(chunks[3].start_idx, 192);
    assert_eq!(chunks[3].end_idx, 200);
    assert!(chunks[3].is_last_chunk);
}

#[test]
fn test_stream_buffer_pool_recycling() {
    let stream_helper = CpuStreamHelper::new();
    let mut pool = stream_helper.create_buffer_pool(128);

    let buf1 = pool.acquire();
    assert_eq!(buf1.len(), 128);
    pool.release(buf1);

    let buf2 = pool.acquire();
    assert_eq!(buf2.len(), 128);
}

#[test]
fn test_extensible_custom_device() {
    let custom_cap = DeviceCapability {
        kind: DeviceKind::Custom("AppleMetal".to_string()),
        device_id: 0,
        name: "Apple M3 Max GPU".to_string(),
        backend: "metal".to_string(),
        total_memory_bytes: 36 * 1024 * 1024 * 1024,
        available_memory_bytes: None,
        supported_precisions: vec![PrecisionMode::Fp16, PrecisionMode::Fp32],
        supports_matrix_multiply: true,
        supports_fused_attention: true,
        supports_in_place_optimizer: true,
    };

    let custom_dev = CustomDevice::new(custom_cap, vec!["matmul".to_string(), "attention".to_string()]);
    assert_eq!(custom_dev.name(), "Apple M3 Max GPU");
    assert!(custom_dev.supports_operation("matmul"));
    assert!(custom_dev.supports_operation("attention"));
    assert!(!custom_dev.supports_operation("unknown_kernel"));
}

#[test]
fn test_compute_channel_live_auto_initialization() {
    let channel = ComputeChannel::auto().expect("channel auto-init");
    assert!(!channel.device_name().is_empty());
    let summary = channel.summary_report();
    assert!(summary.contains("Device:"));
    assert!(summary.contains("Backend:"));
}
