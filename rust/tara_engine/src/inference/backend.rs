//! Pluggable backend dispatch for CPU and GPU execution.

pub trait DeviceBackend: Send + Sync {
    fn name(&self) -> &'static str;
    fn is_available(&self) -> bool;
    fn allocate_tensor(&self, size: usize) -> Vec<f32>;
    fn transfer_to_device(&self, host_tensor: &[f32]) -> Vec<f32>;
    fn transfer_to_host(&self, device_tensor: &[f32]) -> Vec<f32>;
}

pub struct CPUBackend;

impl DeviceBackend for CPUBackend {
    fn name(&self) -> &'static str {
        "CPU"
    }

    fn is_available(&self) -> bool {
        true
    }

    fn allocate_tensor(&self, size: usize) -> Vec<f32> {
        vec![0.0; size]
    }

    fn transfer_to_device(&self, host_tensor: &[f32]) -> Vec<f32> {
        host_tensor.to_vec()
    }

    fn transfer_to_host(&self, device_tensor: &[f32]) -> Vec<f32> {
        device_tensor.to_vec()
    }
}

pub struct CUDABackend {
    pub device_id: u32,
    available: bool,
}

impl CUDABackend {
    pub fn new(device_id: u32) -> Self {
        let available = crate::cuda::driver::CudaDriver::load()
            .map(|d| d.device_count().map(|c| c > (device_id as i32)).unwrap_or(false))
            .unwrap_or(false);
        Self {
            device_id,
            available,
        }
    }
}

impl DeviceBackend for CUDABackend {
    fn name(&self) -> &'static str {
        "CUDA"
    }

    fn is_available(&self) -> bool {
        self.available
    }

    fn allocate_tensor(&self, size: usize) -> Vec<f32> {
        vec![0.0; size]
    }

    fn transfer_to_device(&self, host_tensor: &[f32]) -> Vec<f32> {
        host_tensor.to_vec()
    }

    fn transfer_to_host(&self, device_tensor: &[f32]) -> Vec<f32> {
        device_tensor.to_vec()
    }
}

/// Default CUDA device ordinal (primary GPU index 0).
pub const PRIMARY_CUDA_DEVICE_ORDINAL: u32 = 0;

impl Default for CUDABackend {
    fn default() -> Self {
        Self::new(PRIMARY_CUDA_DEVICE_ORDINAL)
    }
}

pub struct BackendRegistry;

impl BackendRegistry {
    pub fn select_best_backend() -> Box<dyn DeviceBackend> {
        let cuda = CUDABackend::default();
        if cuda.is_available() {
            Box::new(cuda)
        } else {
            Box::new(CPUBackend)
        }
    }
}
