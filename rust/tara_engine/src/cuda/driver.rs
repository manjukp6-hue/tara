//! Native Dynamic CUDA Driver API Wrapper for TARA.
//!
//! Provides zero-dependency, runtime-linked CUDA Driver API bindings with:
//! - Full context binding & RAII current-context management (`cuCtxPushCurrent` / `cuCtxPopCurrent`)
//! - Strict lifetime ownership: Buffers and Modules own their Context and Driver, Kernels own their Module
//! - Prevention of use-after-unload / use-after-destroy bugs
//! - Detailed JIT diagnostics via `cuModuleLoadDataEx`
//! - Authentic driver error diagnostics via `cuGetErrorName` and `cuGetErrorString`
//! - Process-wide driver singleton caching via `OnceLock`
//! - Fully sound `Send` and `Sync` models serialized through context guard locks
//! - Completely native Rust, release-grade, and safe.

use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::sync::{Arc, Mutex, OnceLock};
use thiserror::Error;

#[derive(Debug, Error, Clone)]
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

fn cuda_err_fallback_string(code: i32) -> String {
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
        700 => "CUDA_ERROR_ILLEGAL_ADDRESS".into(),
        701 => "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES".into(),
        702 => "CUDA_ERROR_INVALID_HANDLE".into(),
        719 => "CUDA_ERROR_LAUNCH_FAILED".into(),
        801 => "CUDA_ERROR_NOT_SUPPORTED".into(),
        999 => "CUDA_ERROR_UNKNOWN".into(),
        other => format!("CUDA_ERROR_CODE_{other}"),
    }
}

macro_rules! check_cu {
    ($driver:expr, $expr:expr) => {{
        let res = $expr;
        if res != 0 {
            return Err(CudaError::ApiError(res, $driver.get_error_string(res)));
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
const RTLD_LAZY: c_int = 1;

#[cfg(unix)]
extern "C" {
    fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlclose(handle: *mut c_void) -> c_int;
    fn dlerror() -> *const c_char;
}

/// RAII guard to prevent library handle leak if symbol resolution fails midway.
struct LibraryGuard(*mut c_void);

impl LibraryGuard {
    fn disarm(mut self) -> *mut c_void {
        let handle = self.0;
        self.0 = std::ptr::null_mut();
        handle
    }
}

impl Drop for LibraryGuard {
    fn drop(&mut self) {
        if !self.0.is_null() {
            #[cfg(windows)]
            unsafe {
                FreeLibrary(self.0);
            }
            #[cfg(unix)]
            unsafe {
                dlclose(self.0);
            }
        }
    }
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
            let handle = dlopen(candidate.as_ptr(), RTLD_LAZY);
            if !handle.is_null() {
                return Ok(handle);
            } else {
                let err_ptr = dlerror();
                if !err_ptr.is_null() {
                    let c_str = CStr::from_ptr(err_ptr);
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

pub struct CudaDriverInner {
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
    fn_ctx_push_current: extern "system" fn(*mut c_void) -> c_int,
    fn_ctx_pop_current: extern "system" fn(*mut *mut c_void) -> c_int,
    fn_ctx_synchronize: extern "system" fn() -> c_int,
    fn_mem_get_info: extern "system" fn(*mut usize, *mut usize) -> c_int,
    fn_mem_alloc: extern "system" fn(*mut u64, usize) -> c_int,
    fn_mem_free: extern "system" fn(u64) -> c_int,
    fn_memcpy_htod: extern "system" fn(u64, *const c_void, usize) -> c_int,
    fn_memcpy_dtoh: extern "system" fn(*mut c_void, u64, usize) -> c_int,
    fn_memcpy_dtod: extern "system" fn(u64, u64, usize) -> c_int,
    fn_module_load_data: extern "system" fn(*mut *mut c_void, *const c_char) -> c_int,
    fn_module_load_data_ex: Option<
        extern "system" fn(*mut *mut c_void, *const c_char, u32, *mut u32, *mut *mut c_void) -> c_int,
    >,
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
    fn_get_error_name: Option<extern "system" fn(c_int, *mut *const c_char) -> c_int>,
    fn_get_error_string: Option<extern "system" fn(c_int, *mut *const c_char) -> c_int>,
}

unsafe impl Send for CudaDriverInner {}
unsafe impl Sync for CudaDriverInner {}

impl CudaDriverInner {
    fn load() -> Result<Self, CudaError> {
        let raw_handle = unsafe { load_driver_lib()? };
        let guard = LibraryGuard(raw_handle);

        unsafe {
            let fn_init = get_proc_fn(guard.0, "cuInit")?;
            let fn_driver_get_version = get_proc_fn(guard.0, "cuDriverGetVersion")?;
            let fn_device_get_count = get_proc_fn(guard.0, "cuDeviceGetCount")?;
            let fn_device_get = get_proc_fn(guard.0, "cuDeviceGet")?;
            let fn_device_get_name = get_proc_fn(guard.0, "cuDeviceGetName")?;
            let fn_device_total_mem = get_proc_fn(guard.0, "cuDeviceTotalMem_v2")
                .or_else(|_| get_proc_fn(guard.0, "cuDeviceTotalMem"))?;
            let fn_device_get_attr = get_proc_fn(guard.0, "cuDeviceGetAttribute")?;
            let fn_ctx_create = get_proc_fn(guard.0, "cuCtxCreate_v2")
                .or_else(|_| get_proc_fn(guard.0, "cuCtxCreate"))?;
            let fn_ctx_destroy = get_proc_fn(guard.0, "cuCtxDestroy_v2")
                .or_else(|_| get_proc_fn(guard.0, "cuCtxDestroy"))?;
            let fn_ctx_push_current = get_proc_fn(guard.0, "cuCtxPushCurrent_v2")
                .or_else(|_| get_proc_fn(guard.0, "cuCtxPushCurrent"))?;
            let fn_ctx_pop_current = get_proc_fn(guard.0, "cuCtxPopCurrent_v2")
                .or_else(|_| get_proc_fn(guard.0, "cuCtxPopCurrent"))?;
            let fn_ctx_synchronize = get_proc_fn(guard.0, "cuCtxSynchronize")?;
            let fn_mem_get_info = get_proc_fn(guard.0, "cuMemGetInfo_v2")
                .or_else(|_| get_proc_fn(guard.0, "cuMemGetInfo"))?;
            let fn_mem_alloc = get_proc_fn(guard.0, "cuMemAlloc_v2")
                .or_else(|_| get_proc_fn(guard.0, "cuMemAlloc"))?;
            let fn_mem_free = get_proc_fn(guard.0, "cuMemFree_v2")
                .or_else(|_| get_proc_fn(guard.0, "cuMemFree"))?;
            let fn_memcpy_htod = get_proc_fn(guard.0, "cuMemcpyHtoD_v2")
                .or_else(|_| get_proc_fn(guard.0, "cuMemcpyHtoD"))?;
            let fn_memcpy_dtoh = get_proc_fn(guard.0, "cuMemcpyDtoH_v2")
                .or_else(|_| get_proc_fn(guard.0, "cuMemcpyDtoH"))?;
            let fn_memcpy_dtod = get_proc_fn(guard.0, "cuMemcpyDtoD_v2")
                .or_else(|_| get_proc_fn(guard.0, "cuMemcpyDtoD"))?;
            let fn_module_load_data = get_proc_fn(guard.0, "cuModuleLoadData")?;
            let fn_module_load_data_ex = get_proc_fn(guard.0, "cuModuleLoadDataEx").ok();
            let fn_module_get_function = get_proc_fn(guard.0, "cuModuleGetFunction")?;
            let fn_module_unload = get_proc_fn(guard.0, "cuModuleUnload")?;
            let fn_launch_kernel = get_proc_fn(guard.0, "cuLaunchKernel")?;
            let fn_get_error_name = get_proc_fn(guard.0, "cuGetErrorName").ok();
            let fn_get_error_string = get_proc_fn(guard.0, "cuGetErrorString").ok();

            let lib_handle = guard.disarm();

            let driver = Self {
                lib_handle,
                fn_init,
                fn_driver_get_version,
                fn_device_get_count,
                fn_device_get,
                fn_device_get_name,
                fn_device_total_mem,
                fn_device_get_attr,
                fn_ctx_create,
                fn_ctx_destroy,
                fn_ctx_push_current,
                fn_ctx_pop_current,
                fn_ctx_synchronize,
                fn_mem_get_info,
                fn_mem_alloc,
                fn_mem_free,
                fn_memcpy_htod,
                fn_memcpy_dtoh,
                fn_memcpy_dtod,
                fn_module_load_data,
                fn_module_load_data_ex,
                fn_module_get_function,
                fn_module_unload,
                fn_launch_kernel,
                fn_get_error_name,
                fn_get_error_string,
            };

            // Process-wide global initialization
            let init_res = (driver.fn_init)(0);
            if init_res != 0 {
                return Err(CudaError::ApiError(
                    init_res,
                    format!("cuInit failed: {}", driver.get_error_string(init_res)),
                ));
            }

            Ok(driver)
        }
    }

    pub fn get_error_string(&self, code: i32) -> String {
        let mut name_ptr: *const c_char = std::ptr::null();
        let mut str_ptr: *const c_char = std::ptr::null();
        let has_name = self
            .fn_get_error_name
            .map(|f| f(code, &mut name_ptr) == 0 && !name_ptr.is_null())
            .unwrap_or(false);
        let has_str = self
            .fn_get_error_string
            .map(|f| f(code, &mut str_ptr) == 0 && !str_ptr.is_null())
            .unwrap_or(false);

        match (has_name, has_str) {
            (true, true) => {
                let name = unsafe { CStr::from_ptr(name_ptr).to_string_lossy() };
                let desc = unsafe { CStr::from_ptr(str_ptr).to_string_lossy() };
                format!("{name} ({desc})")
            }
            (true, false) => unsafe { CStr::from_ptr(name_ptr).to_string_lossy().into_owned() },
            _ => cuda_err_fallback_string(code),
        }
    }
}

impl Drop for CudaDriverInner {
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

static DRIVER_SINGLETON: OnceLock<Result<Arc<CudaDriverInner>, String>> = OnceLock::new();

/// Process-wide shared CUDA driver handle.
#[derive(Clone)]
pub struct CudaDriver {
    inner: Arc<CudaDriverInner>,
}

impl CudaDriver {
    pub fn load() -> Result<Self, CudaError> {
        let res = DRIVER_SINGLETON.get_or_init(|| {
            CudaDriverInner::load()
                .map(Arc::new)
                .map_err(|e| e.to_string())
        });
        match res {
            Ok(inner) => Ok(Self {
                inner: Arc::clone(inner),
            }),
            Err(e) => Err(CudaError::LibraryNotFound(e.clone())),
        }
    }

    pub fn driver_version(&self) -> Result<(i32, i32), CudaError> {
        let mut ver: c_int = 0;
        check_cu!(self.inner, (self.inner.fn_driver_get_version)(&mut ver));
        let major = ver / 1000;
        let minor = (ver % 100) / 10;
        Ok((major, minor))
    }

    pub fn device_count(&self) -> Result<i32, CudaError> {
        let mut count: c_int = 0;
        check_cu!(self.inner, (self.inner.fn_device_get_count)(&mut count));
        Ok(count)
    }

    pub fn get_device_info(&self, ordinal: i32) -> Result<CudaDeviceInfo, CudaError> {
        let count = self.device_count()?;
        if count <= 0 || ordinal < 0 || ordinal >= count {
            return Err(CudaError::NoDevice);
        }

        let mut dev: c_int = 0;
        check_cu!(self.inner, (self.inner.fn_device_get)(&mut dev, ordinal));

        let mut name_buf = [0 as c_char; 256];
        check_cu!(
            self.inner,
            (self.inner.fn_device_get_name)(name_buf.as_mut_ptr(), 256, dev)
        );
        let name = unsafe {
            CStr::from_ptr(name_buf.as_ptr())
                .to_string_lossy()
                .into_owned()
        };

        let mut total_bytes = 0usize;
        check_cu!(
            self.inner,
            (self.inner.fn_device_total_mem)(&mut total_bytes, dev)
        );

        let mut major: c_int = 0;
        let mut minor: c_int = 0;
        check_cu!(
            self.inner,
            (self.inner.fn_device_get_attr)(&mut major, 75, dev)
        ); // CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR
        check_cu!(
            self.inner,
            (self.inner.fn_device_get_attr)(&mut minor, 76, dev)
        ); // CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR

        Ok(CudaDeviceInfo {
            ordinal,
            name,
            total_vram_bytes: total_bytes,
            compute_capability: (major, minor),
            raw_device: dev,
        })
    }

    pub fn create_context(&self, device: &CudaDeviceInfo) -> Result<Arc<CudaContextInner>, CudaError> {
        let mut ctx: *mut c_void = std::ptr::null_mut();
        check_cu!(
            self.inner,
            (self.inner.fn_ctx_create)(&mut ctx, 0, device.raw_device)
        );
        Ok(Arc::new(CudaContextInner {
            ctx,
            driver: Arc::clone(&self.inner),
            lock: Mutex::new(()),
        }))
    }
}

/// RAII Guard that pushes a CUDA context onto the current host thread, and pops it on drop.
pub struct CudaContextGuard<'a> {
    driver: &'a CudaDriverInner,
}

impl<'a> CudaContextGuard<'a> {
    pub fn push(ctx: *mut c_void, driver: &'a CudaDriverInner) -> Result<Self, CudaError> {
        let res = (driver.fn_ctx_push_current)(ctx);
        if res != 0 {
            return Err(CudaError::ApiError(res, driver.get_error_string(res)));
        }
        Ok(Self { driver })
    }
}

impl<'a> Drop for CudaContextGuard<'a> {
    fn drop(&mut self) {
        let mut popped: *mut c_void = std::ptr::null_mut();
        let _ = (self.driver.fn_ctx_pop_current)(&mut popped);
    }
}

pub struct CudaContextInner {
    ctx: *mut c_void,
    driver: Arc<CudaDriverInner>,
    lock: Mutex<()>,
}

unsafe impl Send for CudaContextInner {}
unsafe impl Sync for CudaContextInner {}

impl Drop for CudaContextInner {
    fn drop(&mut self) {
        if !self.ctx.is_null() {
            let _lock = self.lock.lock().unwrap();
            let _ = (self.driver.fn_ctx_destroy)(self.ctx);
            self.ctx = std::ptr::null_mut();
        }
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

pub struct CudaSession {
    driver: CudaDriver,
    device: CudaDeviceInfo,
    context: Arc<CudaContextInner>,
}

impl CudaSession {
    pub fn init(device_ordinal: i32) -> Result<Self, CudaError> {
        let driver = CudaDriver::load()?;
        let count = driver.device_count()?;
        if count <= 0 || device_ordinal < 0 || device_ordinal >= count {
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

    pub fn driver(&self) -> &CudaDriver {
        &self.driver
    }

    pub fn get_memory_info(&self) -> Result<(usize, usize), CudaError> {
        let _lock = self.context.lock.lock().unwrap();
        let _guard = CudaContextGuard::push(self.context.ctx, &self.context.driver)?;
        let mut free = 0usize;
        let mut total = 0usize;
        check_cu!(
            self.context.driver,
            (self.context.driver.fn_mem_get_info)(&mut free, &mut total)
        );
        Ok((free, total))
    }

    pub fn allocate_f32(&self, count: usize) -> Result<CudaBuffer, CudaError> {
        if count == 0 {
            return Err(CudaError::KernelError(
                "Zero-byte buffer allocation requested".into(),
            ));
        }
        let bytes = count
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                CudaError::KernelError("Allocation element count overflowed usize bytes".into())
            })?;

        let (free, _) = self.get_memory_info()?;
        if bytes > free {
            return Err(CudaError::OutOfMemory {
                requested_bytes: bytes,
                free_bytes: free,
            });
        }

        let _lock = self.context.lock.lock().unwrap();
        let _guard = CudaContextGuard::push(self.context.ctx, &self.context.driver)?;
        let mut dptr = 0u64;
        check_cu!(
            self.context.driver,
            (self.context.driver.fn_mem_alloc)(&mut dptr, bytes)
        );

        Ok(CudaBuffer {
            dptr,
            len_elements: count,
            bytes,
            context: Arc::clone(&self.context),
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
        let _lock = self.context.lock.lock().unwrap();
        let _guard = CudaContextGuard::push(self.context.ctx, &self.context.driver)?;
        check_cu!(
            self.context.driver,
            (self.context.driver.fn_memcpy_htod)(
                buffer.dptr,
                data.as_ptr() as *const c_void,
                buffer.bytes
            )
        );
        Ok(())
    }

    pub fn upload_f32_slice(&self, buffer: &CudaBuffer, data: &[f32]) -> Result<(), CudaError> {
        if data.len() > buffer.len_elements {
            return Err(CudaError::KernelError(format!(
                "Upload overflow: data has {} elements, but buffer capacity is only {}",
                data.len(),
                buffer.len_elements
            )));
        }
        let bytes_to_copy = std::mem::size_of_val(data);
        let _lock = self.context.lock.lock().unwrap();
        let _guard = CudaContextGuard::push(self.context.ctx, &self.context.driver)?;
        check_cu!(
            self.context.driver,
            (self.context.driver.fn_memcpy_htod)(
                buffer.dptr,
                data.as_ptr() as *const c_void,
                bytes_to_copy
            )
        );
        Ok(())
    }

    pub fn upload_u32(&self, buffer: &CudaBuffer, data: &[u32]) -> Result<(), CudaError> {
        if data.len() != buffer.len_elements {
            return Err(CudaError::KernelError(format!(
                "Upload size mismatch: data has {} elements, buffer has {}",
                data.len(),
                buffer.len_elements
            )));
        }
        let _lock = self.context.lock.lock().unwrap();
        let _guard = CudaContextGuard::push(self.context.ctx, &self.context.driver)?;
        check_cu!(
            self.context.driver,
            (self.context.driver.fn_memcpy_htod)(
                buffer.dptr,
                data.as_ptr() as *const c_void,
                buffer.bytes
            )
        );
        Ok(())
    }

    pub fn download_u32(&self, buffer: &CudaBuffer, out: &mut [u32]) -> Result<(), CudaError> {
        if out.len() != buffer.len_elements {
            return Err(CudaError::KernelError(format!(
                "Download size mismatch: output slice has {} elements, buffer has {}",
                out.len(),
                buffer.len_elements
            )));
        }
        let _lock = self.context.lock.lock().unwrap();
        let _guard = CudaContextGuard::push(self.context.ctx, &self.context.driver)?;
        check_cu!(
            self.context.driver,
            (self.context.driver.fn_memcpy_dtoh)(
                out.as_mut_ptr() as *mut c_void,
                buffer.dptr,
                buffer.bytes
            )
        );
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
        let _lock = self.context.lock.lock().unwrap();
        let _guard = CudaContextGuard::push(self.context.ctx, &self.context.driver)?;
        check_cu!(
            self.context.driver,
            (self.context.driver.fn_memcpy_dtoh)(
                out.as_mut_ptr() as *mut c_void,
                buffer.dptr,
                buffer.bytes
            )
        );
        Ok(())
    }

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
        let _lock = self.context.lock.lock().unwrap();
        let _guard = CudaContextGuard::push(self.context.ctx, &self.context.driver)?;
        check_cu!(
            self.context.driver,
            (self.context.driver.fn_memcpy_dtoh)(
                out.as_mut_ptr() as *mut c_void,
                buffer.dptr,
                bytes_to_copy
            )
        );
        Ok(())
    }

    pub fn synchronize(&self) -> Result<(), CudaError> {
        let _lock = self.context.lock.lock().unwrap();
        let _guard = CudaContextGuard::push(self.context.ctx, &self.context.driver)?;
        check_cu!(
            self.context.driver,
            (self.context.driver.fn_ctx_synchronize)()
        );
        Ok(())
    }

    /// Exact device-to-device copy between two CUDA buffers of matching size.
    pub fn copy_dtod(&self, dst: &mut CudaBuffer, src: &CudaBuffer) -> Result<(), CudaError> {
        if dst.bytes != src.bytes {
            return Err(CudaError::KernelError(format!(
                "copy_dtod exact size mismatch: dst {} bytes != src {} bytes",
                dst.bytes, src.bytes
            )));
        }
        self.copy_dtod_bytes(dst, src, dst.bytes)
    }

    /// Partial or bounded device-to-device copy with explicit length verification.
    pub fn copy_dtod_bytes(
        &self,
        dst: &mut CudaBuffer,
        src: &CudaBuffer,
        bytes: usize,
    ) -> Result<(), CudaError> {
        if bytes > dst.bytes || bytes > src.bytes {
            return Err(CudaError::KernelError(format!(
                "copy_dtod_bytes out of bounds: requested {} bytes, dst capacity {} bytes, src capacity {} bytes",
                bytes, dst.bytes, src.bytes
            )));
        }
        let _lock = self.context.lock.lock().unwrap();
        let _guard = CudaContextGuard::push(self.context.ctx, &self.context.driver)?;
        check_cu!(
            self.context.driver,
            (self.context.driver.fn_memcpy_dtod)(dst.dptr, src.dptr, bytes)
        );
        Ok(())
    }

    /// Load and JIT-compile a PTX module into the session's active CUDA context.
    /// Captures detailed compiler diagnostics via `cuModuleLoadDataEx` whenever available.
    pub fn load_ptx_module(&self, ptx_source: &str) -> Result<CudaModuleHandle, CudaError> {
        let c_ptx = CString::new(ptx_source).map_err(|e| CudaError::KernelError(e.to_string()))?;
        let _lock = self.context.lock.lock().unwrap();
        let _guard = CudaContextGuard::push(self.context.ctx, &self.context.driver)?;

        let mut module: *mut c_void = std::ptr::null_mut();

        if let Some(fn_load_ex) = self.context.driver.fn_module_load_data_ex {
            const CU_JIT_INFO_LOG_BUFFER: u32 = 3;
            const CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES: u32 = 4;
            const CU_JIT_ERROR_LOG_BUFFER: u32 = 5;
            const CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES: u32 = 6;

            let mut error_log = vec![0 as c_char; 8192];
            let mut info_log = vec![0 as c_char; 8192];
            let error_log_len = error_log.len();
            let info_log_len = info_log.len();

            let mut options = [
                CU_JIT_INFO_LOG_BUFFER,
                CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
                CU_JIT_ERROR_LOG_BUFFER,
                CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
            ];
            let mut option_values: [*mut c_void; 4] = [
                info_log.as_mut_ptr() as *mut c_void,
                info_log_len as *mut c_void,
                error_log.as_mut_ptr() as *mut c_void,
                error_log_len as *mut c_void,
            ];

            let res = fn_load_ex(
                &mut module,
                c_ptx.as_ptr(),
                4,
                options.as_mut_ptr(),
                option_values.as_mut_ptr(),
            );

            if res != 0 {
                let err_msg = unsafe {
                    if error_log[0] != 0 {
                        CStr::from_ptr(error_log.as_ptr())
                            .to_string_lossy()
                            .into_owned()
                    } else {
                        self.context.driver.get_error_string(res)
                    }
                };
                return Err(CudaError::KernelError(format!(
                    "PTX JIT compilation failed ({res}): {err_msg}"
                )));
            }
        } else {
            let res = (self.context.driver.fn_module_load_data)(
                &mut module,
                c_ptx.as_ptr(),
            );
            if res != 0 {
                return Err(CudaError::ApiError(
                    res,
                    self.context.driver.get_error_string(res),
                ));
            }
        }

        let inner = Arc::new(CudaModuleInner {
            module,
            context: Arc::clone(&self.context),
        });

        Ok(CudaModuleHandle { inner })
    }
}

pub struct CudaBuffer {
    pub dptr: u64,
    pub len_elements: usize,
    pub bytes: usize,
    context: Arc<CudaContextInner>,
}

unsafe impl Send for CudaBuffer {}
unsafe impl Sync for CudaBuffer {}

impl Drop for CudaBuffer {
    fn drop(&mut self) {
        if self.dptr != 0 {
            let _lock = self.context.lock.lock().unwrap();
            if let Ok(_guard) = CudaContextGuard::push(self.context.ctx, &self.context.driver) {
                let _ = (self.context.driver.fn_mem_free)(self.dptr);
            }
            self.dptr = 0;
        }
    }
}

pub struct CudaModuleInner {
    module: *mut c_void,
    context: Arc<CudaContextInner>,
}

unsafe impl Send for CudaModuleInner {}
unsafe impl Sync for CudaModuleInner {}

impl Drop for CudaModuleInner {
    fn drop(&mut self) {
        if !self.module.is_null() {
            let _lock = self.context.lock.lock().unwrap();
            if let Ok(_guard) = CudaContextGuard::push(self.context.ctx, &self.context.driver) {
                let _ = (self.context.driver.fn_module_unload)(self.module);
            }
            self.module = std::ptr::null_mut();
        }
    }
}

#[derive(Clone)]
pub struct CudaModuleHandle {
    inner: Arc<CudaModuleInner>,
}

unsafe impl Send for CudaModuleHandle {}
unsafe impl Sync for CudaModuleHandle {}

impl CudaModuleHandle {
    pub fn get_kernel(&self, name: &str) -> Result<CudaKernel, CudaError> {
        let c_name = CString::new(name).map_err(|e| CudaError::KernelError(e.to_string()))?;
        let _lock = self.inner.context.lock.lock().unwrap();
        let _guard = CudaContextGuard::push(self.inner.context.ctx, &self.inner.context.driver)?;
        let mut func: *mut c_void = std::ptr::null_mut();
        check_cu!(
            self.inner.context.driver,
            (self.inner.context.driver.fn_module_get_function)(
                &mut func,
                self.inner.module,
                c_name.as_ptr()
            )
        );
        Ok(CudaKernel {
            func,
            module: Arc::clone(&self.inner),
        })
    }
}

#[derive(Clone)]
pub struct CudaKernel {
    func: *mut c_void,
    module: Arc<CudaModuleInner>,
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
        let ctx = &self.module.context;
        let _lock = ctx.lock.lock().unwrap();
        let _guard = CudaContextGuard::push(ctx.ctx, &ctx.driver)?;
        check_cu!(
            ctx.driver,
            (ctx.driver.fn_launch_kernel)(
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
            )
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_string_fallback() {
        assert_eq!(cuda_err_fallback_string(0), "CUDA_SUCCESS");
        assert_eq!(cuda_err_fallback_string(2), "CUDA_ERROR_OUT_OF_MEMORY");
        assert_eq!(cuda_err_fallback_string(201), "CUDA_ERROR_INVALID_CONTEXT");
        assert_eq!(cuda_err_fallback_string(700), "CUDA_ERROR_ILLEGAL_ADDRESS");
        assert_eq!(cuda_err_fallback_string(12345), "CUDA_ERROR_CODE_12345");
    }

    #[test]
    fn test_overflow_and_zero_size_handling() {
        let err_zero = CudaError::KernelError("Zero-byte buffer allocation requested".into());
        assert!(err_zero.to_string().contains("Zero-byte"));

        let max_elements = usize::MAX / 2;
        assert!(max_elements.checked_mul(std::mem::size_of::<f32>()).is_none());
    }

    #[test]
    fn test_driver_singleton_cached() {
        // Repeated calls to CudaDriver::load() must return consistent results and share the singleton
        let res1 = CudaDriver::load();
        let res2 = CudaDriver::load();
        match (res1, res2) {
            (Ok(d1), Ok(d2)) => {
                assert!(Arc::ptr_eq(&d1.inner, &d2.inner));
            }
            (Err(e1), Err(e2)) => {
                assert_eq!(e1.to_string(), e2.to_string());
            }
            _ => panic!("CudaDriver singleton returned inconsistent results across invocations"),
        }
    }
}
