//! Universal compute channel and dynamic execution dispatcher for TARA-HD.
//!
//! Connects and routes compute operations to the dynamically discovered and verified hardware device.

use std::sync::Arc;

use crate::device::{ComputeDevice, CpuDevice, CudaDevice, CustomDevice, TpuDevice};
use crate::discovery::DiscoveredInventory;
use crate::streaming::CpuStreamHelper;
use crate::types::{DeviceKind, PrecisionMode, SelectionPolicy};

/// Central hardware compute channel coordinating device execution across the system.
#[derive(Debug, Clone)]
pub struct ComputeChannel {
    active_device: Arc<dyn ComputeDevice>,
    precision: PrecisionMode,
    inventory: DiscoveredInventory,
}

impl ComputeChannel {
    /// Automatically discover host hardware and channel to the optimal accelerator.
    /// Priority order: GPU > TPU > Custom > CPU.
    pub fn auto() -> Result<Self, String> {
        Self::with_options(&SelectionPolicy::Auto, PrecisionMode::Auto)
    }

    /// Channel to a device matching the explicit policy and precision mode by probing the host.
    pub fn with_options(
        policy: &SelectionPolicy,
        precision_req: PrecisionMode,
    ) -> Result<Self, String> {
        let inventory = DiscoveredInventory::probe_host();
        Self::with_inventory(inventory, policy, precision_req)
    }

    /// Channel to a device using a pre-discovered or custom-populated hardware inventory.
    pub fn with_inventory(
        inventory: DiscoveredInventory,
        policy: &SelectionPolicy,
        precision_req: PrecisionMode,
    ) -> Result<Self, String> {
        let selected_cap = inventory.resolve_device(policy)?;

        let active_device: Arc<dyn ComputeDevice> = match selected_cap.kind {
            DeviceKind::Gpu => Arc::new(CudaDevice::new(selected_cap.clone())),
            DeviceKind::Tpu => Arc::new(TpuDevice::new(selected_cap.clone())),
            DeviceKind::Cpu => Arc::new(CpuDevice::new(selected_cap.clone())),
            DeviceKind::Custom(_) => {
                let mut ops = vec!["rmsnorm".to_string(), "rope".to_string()];
                if selected_cap.supports_matrix_multiply {
                    ops.push("matmul".to_string());
                    ops.push("gemm".to_string());
                }
                if selected_cap.supports_fused_attention {
                    ops.push("attention".to_string());
                }
                if selected_cap.supports_in_place_optimizer {
                    ops.push("adamw".to_string());
                }
                Arc::new(CustomDevice::new(selected_cap.clone(), ops))
            }
        };

        // Resolve effective precision
        let effective_precision = if precision_req == PrecisionMode::Auto {
            if selected_cap
                .supported_precisions
                .contains(&PrecisionMode::Bf16)
            {
                PrecisionMode::Bf16
            } else if selected_cap
                .supported_precisions
                .contains(&PrecisionMode::Fp16)
            {
                PrecisionMode::Fp16
            } else {
                PrecisionMode::Fp32
            }
        } else if selected_cap.supported_precisions.contains(&precision_req) {
            precision_req
        } else {
            log::warn!(
                "Requested precision {:?} not natively supported on {}. Defaulting to FP32.",
                precision_req,
                selected_cap.name
            );
            PrecisionMode::Fp32
        };

        Ok(Self {
            active_device,
            precision: effective_precision,
            inventory,
        })
    }

    /// Construct a ComputeChannel with an explicitly provided ComputeDevice implementation.
    pub fn with_device(
        device: Arc<dyn ComputeDevice>,
        precision: PrecisionMode,
        inventory: DiscoveredInventory,
    ) -> Self {
        Self {
            active_device: device,
            precision,
            inventory,
        }
    }

    /// Swap or update the active device implementation inside the channel.
    pub fn set_device(&mut self, device: Arc<dyn ComputeDevice>) {
        self.active_device = device;
    }

    /// Return reference to the active compute device.
    pub fn device(&self) -> &dyn ComputeDevice {
        self.active_device.as_ref()
    }

    /// Return reference to the CPU streaming helper if the active device is CPU.
    pub fn stream_helper(&self) -> Option<&CpuStreamHelper> {
        self.active_device.stream_helper()
    }

    /// Open a hardware-agnostic universal stream engine across CPU, GPU, or TPU.
    pub fn universal_stream(&self, max_in_flight_bytes: usize) -> crate::streaming::UniversalStreamEngine {
        crate::streaming::UniversalStreamEngine::new(max_in_flight_bytes)
    }

    /// Capture live dynamic telemetry for the active device.
    pub fn live_telemetry(&self) -> crate::telemetry::RuntimeTelemetrySnapshot {
        let vram_static = if self.is_gpu() {
            self.inventory.gpus.first().map(|g| g.total_memory_bytes as usize)
        } else {
            None
        };
        crate::telemetry::LiveTelemetryProvider::capture(self.device_kind().clone(), vram_static)
    }

    /// Evaluate memory placement (FullResident, PartialOffload, or CapacityExceeded)
    /// based on live telemetry and model budget.
    pub fn evaluate_placement(
        &self,
        budget: &crate::memory_governor::ModelMemoryBudget,
        mode: crate::memory_governor::MemoryExecutionMode,
        total_layers: usize,
    ) -> crate::memory_governor::PlacementDecision {
        let telemetry = self.live_telemetry();
        crate::memory_governor::MemoryGovernor::evaluate_placement(budget, &telemetry, mode, total_layers)
    }

    /// Return active device category.
    pub fn device_kind(&self) -> &DeviceKind {
        self.active_device.kind()
    }

    /// Return active device human-readable name.
    pub fn device_name(&self) -> &str {
        self.active_device.name()
    }

    /// Return active compute precision.
    pub fn precision(&self) -> PrecisionMode {
        self.precision
    }

    /// Returns true if execution is on a dedicated GPU accelerator.
    pub fn is_gpu(&self) -> bool {
        *self.device_kind() == DeviceKind::Gpu
    }

    /// Returns true if execution is on a TPU accelerator.
    pub fn is_tpu(&self) -> bool {
        *self.device_kind() == DeviceKind::Tpu
    }

    /// Returns true if execution is on host CPU.
    pub fn is_cpu(&self) -> bool {
        *self.device_kind() == DeviceKind::Cpu
    }

    /// Access the full inventory of all discovered hardware on the host.
    pub fn inventory(&self) -> &DiscoveredInventory {
        &self.inventory
    }

    /// Generate an authentic diagnostic summary string for logs and terminals.
    pub fn summary_report(&self) -> String {
        format!(
            "Device: {} | Backend: {} | Kind: {:?} | Precision: {:?} | Total Memory: {:.2} GB",
            self.device_name(),
            self.active_device.info().backend,
            self.device_kind(),
            self.precision(),
            self.active_device.info().total_memory_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
        )
    }

    /// Dispatch linear forward projection through the active compute device.
    pub fn forward_linear(
        &self,
        weight: &[f32],
        x: &[f32],
        seq_len: usize,
        in_dim: usize,
        out_dim: usize,
    ) -> Result<Vec<f32>, String> {
        self.active_device
            .forward_linear(weight, x, seq_len, in_dim, out_dim)
    }

    /// Dispatch RMSNorm forward pass through the active compute device.
    pub fn forward_rmsnorm(
        &self,
        weight: &[f32],
        x: &[f32],
        seq_len: usize,
        hidden_dim: usize,
        eps: f32,
    ) -> Result<Vec<f32>, String> {
        self.active_device
            .forward_rmsnorm(weight, x, seq_len, hidden_dim, eps)
    }

    /// Dispatch SwiGLU forward pass through the active compute device.
    pub fn forward_swiglu(&self, gate: &[f32], up: &[f32]) -> Result<Vec<f32>, String> {
        self.active_device.forward_swiglu(gate, up)
    }

    /// Dispatch residual addition through the active compute device.
    pub fn add_residual(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>, String> {
        self.active_device.add_residual(a, b)
    }

    /// Dispatch linear backward projection through the active compute device.
    pub fn backward_linear(
        &self,
        weight: &[f32],
        d_out: &[f32],
        x: &[f32],
        seq_len: usize,
        in_dim: usize,
        out_dim: usize,
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        self.active_device
            .backward_linear(weight, d_out, x, seq_len, in_dim, out_dim)
    }

    /// Dispatch named linear forward projection through the active compute device.
    pub fn forward_linear_named(
        &self,
        weight_name: &str,
        weight_slice: &[f32],
        x: &[f32],
        seq_len: usize,
        in_dim: usize,
        out_dim: usize,
    ) -> Result<Vec<f32>, String> {
        self.active_device.forward_linear_named(
            weight_name,
            weight_slice,
            x,
            seq_len,
            in_dim,
            out_dim,
        )
    }

    /// Dispatch named RMSNorm forward pass through the active compute device.
    pub fn forward_rmsnorm_named(
        &self,
        weight_name: &str,
        weight_slice: &[f32],
        x: &[f32],
        seq_len: usize,
        hidden_dim: usize,
        eps: f32,
    ) -> Result<Vec<f32>, String> {
        self.active_device.forward_rmsnorm_named(
            weight_name,
            weight_slice,
            x,
            seq_len,
            hidden_dim,
            eps,
        )
    }

    /// Dispatch named linear backward projection through the active compute device.
    pub fn backward_linear_named(
        &self,
        weight_name: &str,
        weight_slice: &[f32],
        d_out: &[f32],
        x: &[f32],
        seq_len: usize,
        in_dim: usize,
        out_dim: usize,
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        self.active_device.backward_linear_named(
            weight_name,
            weight_slice,
            d_out,
            x,
            seq_len,
            in_dim,
            out_dim,
        )
    }

    /// Dispatch embedding forward lookup through the active compute device.
    pub fn forward_embedding(
        &self,
        token_ids: &[u32],
        embed_table: &[f32],
        hidden_size: usize,
    ) -> Result<Vec<f32>, String> {
        self.active_device
            .forward_embedding(token_ids, embed_table, hidden_size)
    }

    /// Dispatch gradient accumulation into device parameter buffer.
    pub fn accumulate_gradient(&self, param_name: &str, gradient: &[f32]) -> Result<(), String> {
        self.active_device.accumulate_gradient(param_name, gradient)
    }
}
