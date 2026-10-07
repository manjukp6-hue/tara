//! Native Dynamic CUDA Driver API Wrapper for TARA.
//!
//! Provides zero-dependency, runtime-linked CUDA Driver API bindings.
//! Automatically loads `nvcuda.dll` on Windows and `libcuda.so.1` / `libcuda.so` on Linux.
//! Does not require CUDA Toolkit / nvcc at compile time or link time.
//! Completely native, release-grade, and safe.

use std::ffi::{c_char, c_int, c_void, CStr, CString};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CudaError {
    #[error("CUDA Driver library not found (nvcuda.dll / libcuda.so.1): {0}")]
    LibraryNotFound(String),
    #[error("CUDA Driver symbol '{0}' not found in driver library")]
    SymbolNotFound(String),
    #[error("CUDA API error ({0}): {1}")]
    ApiError(i32, String),
    #[error("No CUDA-capable GPU detected")]
    NoDevice,
    #[error("CUDA Out of Memory: requested {requested_bytes} bytes, {free_bytes} bytes free")]
    OutOfMemory {
        requested_bytes: usize,
        free_bytes: usize,
    },
    #[error("CUDA kernel compilation / launch failed: {0}")]
    KernelError(String),
}

fn cuda_err_string(code: i32) -> String {
    match code {
        0 => "CUDA_SUCCESS".into(),
        1 => "CUDA_ERROR_INVALID_VALUE".into(),
        2 => "CUDA_ERROR_OUT_OF_MEMORY".into(),
        3 => "CUDA_ERROR_NOT_INITIALIZED".into(),
        4 => "CUDA_ERROR_DEINITIALIZED".into(),
        100 => "CUDA_ERROR_NO_DEVICE".into(),
        101 => "CUDA_ERROR_INVALID_DEVICE".into(),
        200 => "CUDA_ERROR_INVALID_IMAGE".into(),
        201 => "CUDA_ERROR_INVALID_CONTEXT".into(),
        209 => "CUDA_ERROR_NO_BINARY_FOR_GPU".into(),
        214 => "CUDA_ERROR_UNSUPPORTED_PTX_VERSION".into(),
        999 => "CUDA_ERROR_UNKNOWN".into(),
        other => format!("CUDA_ERROR_CODE_{other}"),
    }
}

macro_rules! check_cu {
    ($expr:expr) => {{
        let res = $expr;
        if res != 0 {
            return Err(CudaError::ApiError(res, cuda_err_string(res)));
        }
    }};
}

#[cfg(windows)]
extern "system" {
    fn LoadLibraryA(lpLibFileName: *const c_char) -> *mut c_void;
    fn GetProcAddress(hModule: *mut c_void, lpProcName: *const c_char) -> *mut c_void;
    fn FreeLibrary(hModule: *mut c_void) -> c_int;
}

#[cfg(unix)]
extern "C" {
    fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlclose(handle: *mut c_void) -> c_int;
    fn dlerror() -> *const c_char;
}

unsafe fn load_driver_lib() -> Result<*mut c_void, CudaError> {
    #[cfg(windows)]
    {
        let lib_name = b"nvcuda.dll\0";
        let handle = LoadLibraryA(lib_name.as_ptr() as *const c_char);
        if handle.is_null() {
            return Err(CudaError::LibraryNotFound(
                "nvcuda.dll could not be loaded. Ensure NVIDIA display driver is installed.".into(),
            ));
        }
        Ok(handle)
    }

    #[cfg(unix)]
    {
        use std::path::Path;

        let mut candidates: Vec<CString> = Vec::new();
        // Bare filenames (searches standard dynamic linker paths and LD_LIBRARY_PATH)
        if let Ok(c) = CString::new("libcuda.so.1") {
            candidates.push(c);
        }
        if let Ok(c) = CString::new("libcuda.so") {
            candidates.push(c);
        }

        // Standard cloud, distribution, and container driver locations
        let standard_paths = [
            "/usr/lib64-nvidia/libcuda.so.1",
            "/usr/lib64-nvidia/libcuda.so",
            "/usr/local/cuda/compat/libcuda.so.1",
            "/usr/local/cuda/compat/libcuda.so",
            "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
            "/usr/lib/x86_64-linux-gnu/libcuda.so",
            "/usr/lib64/libcuda.so.1",
            "/usr/lib64/libcuda.so",
            "/usr/lib/libcuda.so.1",
            "/usr/lib/libcuda.so",
        ];
        for path in standard_paths {
            if Path::new(path).exists() {
                if let Ok(c) = CString::new(path) {
                    candidates.push(c);
                }
            }
        }

        // Dynamically inspect all directories declared in LD_LIBRARY_PATH
        if let Ok(ld_path) = std::env::var("LD_LIBRARY_PATH") {
            for dir in ld_path.split(':') {
                if !dir.is_empty() {
                    let p1 = format!("{dir}/libcuda.so.1");
                    if Path::new(&p1).exists() {
                        if let Ok(c) = CString::new(p1) {
                            candidates.push(c);
                        }
                    }
                    let p2 = format!("{dir}/libcuda.so");
                    if Path::new(&p2).exists() {
                        if let Ok(c) = CString::new(p2) {
                            candidates.push(c);
                        }
                    }
                }
            }
        }

        // Dynamically probe /usr/local/cuda*/compat for version-specific drivers
        if let Ok(entries) = std::fs::read_dir("/usr/local") {
            for entry in entries.flatten() {
                let p = entry.path();
                if let Some(name) = p.file_name().and_then(|n| n.to_str()) {
                    if name.starts_with("cuda") {
                        let compat1 = p.join("compat").join("libcuda.so.1");
                        if compat1.exists() {
                            if let Ok(c) = CString::new(compat1.to_string_lossy().to_string()) {
                                candidates.push(c);
                            }
                        }
                        let compat2 = p.join("compat").join("libcuda.so");
                        if compat2.exists() {
                            if let Ok(c) = CString::new(compat2.to_string_lossy().to_string()) {
                                candidates.push(c);
                            }
                        }
                    }
                }
            }
        }

        let mut last_error = String::new();
        for candidate in &candidates {
            let handle = dlopen(candidate.as_ptr(), 1); // RTLD_LAZY
            if !handle.is_null() {
                return Ok(handle);
            } else {
                let err_ptr = dlerror();
                if !err_ptr.is_null() {
                    let c_str = std::ffi::CStr::from_ptr(err_ptr);
                    last_error = c_str.to_string_lossy().to_string();
                }
            }
        }

        let err_detail = if last_error.is_empty() {
            "No candidate matched or driver library not found in search paths".to_string()
        } else {
            last_error
        };

        Err(CudaError::LibraryNotFound(format!(
            "CUDA driver library (libcuda.so.1) could not be loaded: {err_detail}. Checked default linker paths, /usr/lib64-nvidia, /usr/local/cuda/compat, and LD_LIBRARY_PATH."
        )))
    }
}

unsafe fn get_proc(handle: *mut c_void, name: &str) -> Result<*mut c_void, CudaError> {
    let c_name = CString::new(name).map_err(|_| CudaError::SymbolNotFound(name.into()))?;
    #[cfg(windows)]
    let ptr = GetProcAddress(handle, c_name.as_ptr());
    #[cfg(unix)]
    let ptr = dlsym(handle, c_name.as_ptr());

    if ptr.is_null() {
        Err(CudaError::SymbolNotFound(name.into()))
    } else {
        Ok(ptr)
    }
}

unsafe fn get_proc_fn<F: Copy>(handle: *mut c_void, name: &str) -> Result<F, CudaError> {
    let ptr = get_proc(handle, name)?;
    Ok(std::mem::transmute_copy::<*mut c_void, F>(&ptr))
}

pub struct CudaDriver {
    lib_handle: *mut c_void,
    fn_init: extern "system" fn(c_int) -> c_int,
    fn_driver_get_version: extern "system" fn(*mut c_int) -> c_int,
    fn_device_get_count: extern "system" fn(*mut c_int) -> c_int,
    fn_device_get: extern "system" fn(*mut c_int, c_int) -> c_int,
    fn_device_get_name: extern "system" fn(*mut c_char, c_int, c_int) -> c_int,
    fn_device_total_mem: extern "system" fn(*mut usize, c_int) -> c_int,
    fn_device_get_attr: extern "system" fn(*mut c_int, c_int, c_int) -> c_int,
    fn_ctx_create: extern "system" fn(*mut *mut c_void, c_int, c_int) -> c_int,
    fn_ctx_destroy: extern "system" fn(*mut c_void) -> c_int,
    fn_ctx_synchronize: extern "system" fn() -> c_int,
    fn_mem_get_info: extern "system" fn(*mut usize, *mut usize) -> c_int,
    fn_mem_alloc: extern "system" fn(*mut u64, usize) -> c_int,
    fn_mem_free: extern "system" fn(u64) -> c_int,
    fn_memcpy_htod: extern "system" fn(u64, *const c_void, usize) -> c_int,
    fn_memcpy_dtoh: extern "system" fn(*mut c_void, u64, usize) -> c_int,
    fn_memcpy_dtod: extern "system" fn(u64, u64, usize) -> c_int,
    fn_module_load_data: extern "system" fn(*mut *mut c_void, *const c_char) -> c_int,
    fn_module_get_function:
        extern "system" fn(*mut *mut c_void, *mut c_void, *const c_char) -> c_int,
    fn_module_unload: extern "system" fn(*mut c_void) -> c_int,
    fn_launch_kernel: extern "system" fn(
        *mut c_void,
        u32,
        u32,
        u32,
        u32,
        u32,
        u32,
        u32,
        *mut c_void,
        *mut *mut c_void,
        *mut *mut c_void,
    ) -> c_int,
}

unsafe impl Send for CudaDriver {}
unsafe impl Sync for CudaDriver {}

impl CudaDriver {
    pub fn load() -> Result<Self, CudaError> {
        let handle = unsafe { load_driver_lib()? };

        unsafe {
            let fn_init = get_proc_fn(handle, "cuInit")?;
            let fn_driver_get_version = get_proc_fn(handle, "cuDriverGetVersion")?;
            let fn_device_get_count = get_proc_fn(handle, "cuDeviceGetCount")?;
            let fn_device_get = get_proc_fn(handle, "cuDeviceGet")?;
            let fn_device_get_name = get_proc_fn(handle, "cuDeviceGetName")?;
            let fn_device_total_mem = get_proc_fn(handle, "cuDeviceTotalMem_v2")?;
            let fn_device_get_attr = get_proc_fn(handle, "cuDeviceGetAttribute")?;
            let fn_ctx_create = get_proc_fn(handle, "cuCtxCreate_v2")?;
            let fn_ctx_destroy = get_proc_fn(handle, "cuCtxDestroy_v2")?;
            let fn_ctx_synchronize = get_proc_fn(handle, "cuCtxSynchronize")?;
            let fn_mem_get_info = get_proc_fn(handle, "cuMemGetInfo_v2")?;
            let fn_mem_alloc = get_proc_fn(handle, "cuMemAlloc_v2")?;
            let fn_mem_free = get_proc_fn(handle, "cuMemFree_v2")?;
            let fn_memcpy_htod = get_proc_fn(handle, "cuMemcpyHtoD_v2")?;
            let fn_memcpy_dtoh = get_proc_fn(handle, "cuMemcpyDtoH_v2")?;
            let fn_memcpy_dtod = get_proc_fn(handle, "cuMemcpyDtoD_v2")?;
            let fn_module_load_data = get_proc_fn(handle, "cuModuleLoadData")?;
            let fn_module_get_function = get_proc_fn(handle, "cuModuleGetFunction")?;
            let fn_module_unload = get_proc_fn(handle, "cuModuleUnload")?;
            let fn_launch_kernel = get_proc_fn(handle, "cuLaunchKernel")?;

            let driver = Self {
                lib_handle: handle,
                fn_init,
                fn_driver_get_version,
                fn_device_get_count,
                fn_device_get,
                fn_device_get_name,
                fn_device_total_mem,
                fn_device_get_attr,
                fn_ctx_create,
                fn_ctx_destroy,
                fn_ctx_synchronize,
                fn_mem_get_info,
                fn_mem_alloc,
                fn_mem_free,
                fn_memcpy_htod,
                fn_memcpy_dtoh,
                fn_memcpy_dtod,
                fn_module_load_data,
                fn_module_get_function,
                fn_module_unload,
                fn_launch_kernel,
            };

            // Call cuInit(0)
            let init_res = (driver.fn_init)(0);
            if init_res != 0 {
                return Err(CudaError::ApiError(
                    init_res,
                    format!("cuInit failed: {}", cuda_err_string(init_res)),
                ));
            }

            Ok(driver)
        }
    }

    pub fn driver_version(&self) -> Result<(i32, i32), CudaError> {
        let mut ver: c_int = 0;
        check_cu!((self.fn_driver_get_version)(&mut ver));
        let major = ver / 1000;
        let minor = (ver % 100) / 10;
        Ok((major, minor))
    }

    pub fn device_count(&self) -> Result<i32, CudaError> {
        let mut count: c_int = 0;
        check_cu!((self.fn_device_get_count)(&mut count));
        Ok(count)
    }

    pub fn get_device_info(&self, ordinal: i32) -> Result<CudaDeviceInfo, CudaError> {
        let mut dev: c_int = 0;
        check_cu!((self.fn_device_get)(&mut dev, ordinal));

        let mut name_buf = [0 as c_char; 256];
        check_cu!((self.fn_device_get_name)(name_buf.as_mut_ptr(), 256, dev));
        let name = unsafe {
            CStr::from_ptr(name_buf.as_ptr())
                .to_string_lossy()
                .into_owned()
        };

        let mut total_bytes = 0usize;
        check_cu!((self.fn_device_total_mem)(&mut total_bytes, dev));

        let mut major: c_int = 0;
        let mut minor: c_int = 0;
        check_cu!((self.fn_device_get_attr)(&mut major, 75, dev)); // CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR
        check_cu!((self.fn_device_get_attr)(&mut minor, 76, dev)); // CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR

        Ok(CudaDeviceInfo {
            ordinal,
            name,
            total_vram_bytes: total_bytes,
            compute_capability: (major, minor),
            raw_device: dev,
        })
    }

    pub fn create_context(&self, device: &CudaDeviceInfo) -> Result<CudaContextHandle, CudaError> {
        let mut ctx: *mut c_void = std::ptr::null_mut();
        check_cu!((self.fn_ctx_create)(&mut ctx, 0, device.raw_device));
        Ok(CudaContextHandle { ctx })
    }

    pub fn memcpy_dtod(&self, dst: u64, src: u64, bytes: usize) -> Result<(), CudaError> {
        check_cu!((self.fn_memcpy_dtod)(dst, src, bytes));
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct CudaDeviceInfo {
    pub ordinal: i32,
    pub name: String,
    pub total_vram_bytes: usize,
    pub compute_capability: (i32, i32),
    pub raw_device: i32,
}

pub struct CudaContextHandle {
    pub ctx: *mut c_void,
}

unsafe impl Send for CudaContextHandle {}
unsafe impl Sync for CudaContextHandle {}

pub struct CudaSession {
    driver: CudaDriver,
    device: CudaDeviceInfo,
    context: CudaContextHandle,
}

impl CudaSession {
    pub fn init(device_ordinal: i32) -> Result<Self, CudaError> {
        let driver = CudaDriver::load()?;
        let count = driver.device_count()?;
        if count <= 0 || device_ordinal >= count {
            return Err(CudaError::NoDevice);
        }
        let device = driver.get_device_info(device_ordinal)?;
        let context = driver.create_context(&device)?;
        Ok(Self {
            driver,
            device,
            context,
        })
    }

    pub fn device(&self) -> &CudaDeviceInfo {
        &self.device
    }

    pub fn get_memory_info(&self) -> Result<(usize, usize), CudaError> {
        let mut free = 0usize;
        let mut total = 0usize;
        check_cu!((self.driver.fn_mem_get_info)(&mut free, &mut total));
        Ok((free, total))
    }

    pub fn allocate_f32(&self, count: usize) -> Result<CudaBuffer, CudaError> {
        let bytes = count * std::mem::size_of::<f32>();
        let (free, _) = self.get_memory_info()?;
        if bytes > free {
            return Err(CudaError::OutOfMemory {
                requested_bytes: bytes,
                free_bytes: free,
            });
        }

        let mut dptr = 0u64;
        check_cu!((self.driver.fn_mem_alloc)(&mut dptr, bytes));
        Ok(CudaBuffer {
            dptr,
            len_elements: count,
            bytes,
            fn_mem_free: self.driver.fn_mem_free,
        })
    }

    pub fn upload_f32(&self, buffer: &CudaBuffer, data: &[f32]) -> Result<(), CudaError> {
        if data.len() != buffer.len_elements {
            return Err(CudaError::KernelError(format!(
                "Upload size mismatch: data has {} elements, buffer has {}",
                data.len(),
                buffer.len_elements
            )));
        }
        check_cu!((self.driver.fn_memcpy_htod)(
            buffer.dptr,
            data.as_ptr() as *const c_void,
            buffer.bytes
        ));
        Ok(())
    }

    /// Upload a slice of f32 data into a buffer whose capacity is >= data.len().
    /// Copies only data.len() * 4 bytes to the GPU buffer.
    pub fn upload_f32_slice(&self, buffer: &CudaBuffer, data: &[f32]) -> Result<(), CudaError> {
        if data.len() > buffer.len_elements {
            return Err(CudaError::KernelError(format!(
                "Upload overflow: data has {} elements, but buffer capacity is only {}",
                data.len(),
                buffer.len_elements
            )));
        }
        let bytes_to_copy = std::mem::size_of_val(data);
        check_cu!((self.driver.fn_memcpy_htod)(
            buffer.dptr,
            data.as_ptr() as *const c_void,
            bytes_to_copy
        ));
        Ok(())
    }

    pub fn download_f32(&self, buffer: &CudaBuffer, out: &mut [f32]) -> Result<(), CudaError> {
        if out.len() != buffer.len_elements {
            return Err(CudaError::KernelError(format!(
                "Download size mismatch: output slice has {} elements, buffer has {}",
                out.len(),
                buffer.len_elements
            )));
        }
        check_cu!((self.driver.fn_memcpy_dtoh)(
            out.as_mut_ptr() as *mut c_void,
            buffer.dptr,
            buffer.bytes
        ));
        Ok(())
    }

    /// Download a slice of f32 data from a buffer whose capacity is >= out.len().
    /// Copies only out.len() * 4 bytes from the GPU buffer.
    pub fn download_f32_slice(
        &self,
        buffer: &CudaBuffer,
        out: &mut [f32],
    ) -> Result<(), CudaError> {
        if out.len() > buffer.len_elements {
            return Err(CudaError::KernelError(format!(
                "Download overflow: output buffer has {} elements, but buffer capacity is only {}",
                out.len(),
                buffer.len_elements
            )));
        }
        let bytes_to_copy = std::mem::size_of_val(out);
        check_cu!((self.driver.fn_memcpy_dtoh)(
            out.as_mut_ptr() as *mut c_void,
            buffer.dptr,
            bytes_to_copy
        ));
        Ok(())
    }

    pub fn synchronize(&self) -> Result<(), CudaError> {
        check_cu!((self.driver.fn_ctx_synchronize)());
        Ok(())
    }

    /// Device-to-device copy between two CUDA buffers.
    pub fn copy_dtod(&self, dst: &mut CudaBuffer, src: &CudaBuffer) -> Result<(), CudaError> {
        let bytes_to_copy = dst.bytes.min(src.bytes);
        self.driver.memcpy_dtod(dst.dptr, src.dptr, bytes_to_copy)
    }

    pub fn load_ptx_module(&self, ptx_source: &str) -> Result<CudaModuleHandle, CudaError> {
        let c_ptx = CString::new(ptx_source).map_err(|e| CudaError::KernelError(e.to_string()))?;
        let mut module: *mut c_void = std::ptr::null_mut();
        check_cu!((self.driver.fn_module_load_data)(
            &mut module,
            c_ptx.as_ptr()
        ));
        Ok(CudaModuleHandle {
            module,
            fn_module_unload: self.driver.fn_module_unload,
            fn_get_func: self.driver.fn_module_get_function,
            fn_launch: self.driver.fn_launch_kernel,
        })
    }
}

impl Drop for CudaSession {
    fn drop(&mut self) {
        if !self.context.ctx.is_null() {
            (self.driver.fn_ctx_destroy)(self.context.ctx);
            self.context.ctx = std::ptr::null_mut();
        }
    }
}

pub struct CudaBuffer {
    pub dptr: u64,
    pub len_elements: usize,
    pub bytes: usize,
    fn_mem_free: extern "system" fn(u64) -> c_int,
}

unsafe impl Send for CudaBuffer {}
unsafe impl Sync for CudaBuffer {}

impl Drop for CudaBuffer {
    fn drop(&mut self) {
        if self.dptr != 0 {
            (self.fn_mem_free)(self.dptr);
            self.dptr = 0;
        }
    }
}

pub struct CudaModuleHandle {
    module: *mut c_void,
    fn_module_unload: extern "system" fn(*mut c_void) -> c_int,
    fn_get_func: extern "system" fn(*mut *mut c_void, *mut c_void, *const c_char) -> c_int,
    fn_launch: extern "system" fn(
        *mut c_void,
        u32,
        u32,
        u32,
        u32,
        u32,
        u32,
        u32,
        *mut c_void,
        *mut *mut c_void,
        *mut *mut c_void,
    ) -> c_int,
}

unsafe impl Send for CudaModuleHandle {}
unsafe impl Sync for CudaModuleHandle {}

impl CudaModuleHandle {
    pub fn get_kernel(&self, name: &str) -> Result<CudaKernel, CudaError> {
        let c_name = CString::new(name).map_err(|e| CudaError::KernelError(e.to_string()))?;
        let mut func: *mut c_void = std::ptr::null_mut();
        check_cu!((self.fn_get_func)(&mut func, self.module, c_name.as_ptr()));
        Ok(CudaKernel {
            func,
            fn_launch: self.fn_launch,
        })
    }
}

impl Drop for CudaModuleHandle {
    fn drop(&mut self) {
        if !self.module.is_null() {
            (self.fn_module_unload)(self.module);
            self.module = std::ptr::null_mut();
        }
    }
}

impl Drop for CudaDriver {
    fn drop(&mut self) {
        if !self.lib_handle.is_null() {
            #[cfg(windows)]
            unsafe {
                FreeLibrary(self.lib_handle);
            }
            #[cfg(unix)]
            unsafe {
                dlclose(self.lib_handle);
            }
            self.lib_handle = std::ptr::null_mut();
        }
    }
}

pub struct CudaKernel {
    func: *mut c_void,
    fn_launch: extern "system" fn(
        *mut c_void,
        u32,
        u32,
        u32,
        u32,
        u32,
        u32,
        u32,
        *mut c_void,
        *mut *mut c_void,
        *mut *mut c_void,
    ) -> c_int,
}

unsafe impl Send for CudaKernel {}
unsafe impl Sync for CudaKernel {}

impl CudaKernel {
    /// Launch the CUDA kernel.
    ///
    /// # Safety
    /// The caller must ensure that `params` contains valid device or host pointers matching the
    /// kernel argument signature, and that execution does not violate GPU memory safety.
    pub unsafe fn launch(
        &self,
        grid_dim: (u32, u32, u32),
        block_dim: (u32, u32, u32),
        shared_mem: u32,
        mut params: [*mut c_void; 16],
        _param_count: usize,
    ) -> Result<(), CudaError> {
        check_cu!((self.fn_launch)(
            self.func,
            grid_dim.0,
            grid_dim.1,
            grid_dim.2,
            block_dim.0,
            block_dim.1,
            block_dim.2,
            shared_mem,
            std::ptr::null_mut(),
            params.as_mut_ptr(),
            std::ptr::null_mut(),
        ));
        Ok(())
    }
}
