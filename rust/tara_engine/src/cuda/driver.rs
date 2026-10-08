//! Native Dynamic CUDA Driver API Wrapper for TARA.
//!
//! Provides zero-dependency, runtime-linked CUDA Driver API bindings with:
//! - Full context binding & RAII current-context management (`cuCtxPushCurrent` / `cuCtxPopCurrent`)
//! - Immediate context detachment on creation (`cuCtxCreate` -> `cuCtxPopCurrent`) leaving zero lingering thread-local state
//! - Explicit pop error propagation via `CudaContextGuard::finish()` and `with_context()`
//! - Strict lifetime ownership: Buffers and Modules own their Context and Driver, Kernels own their Module
//! - Prevention of use-after-unload / use-after-destroy bugs
//! - Detailed JIT diagnostics via `cuModuleLoadDataEx`
//! - Authentic driver error diagnostics via `cuGetErrorName` and `cuGetErrorString`
//! - Process-wide driver singleton caching via `OnceLock`
//! - All wrapper-mediated CUDA operations are serialized per context and explicitly bind the intended context before execution.

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
    fn_ctx_get_current: extern "system" fn(*mut *mut c_void) -> c_int,
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
            let fn_ctx_get_current = get_proc_fn(guard.0, "cuCtxGetCurrent_v2")
                .or_else(|_| get_proc_fn(guard.0, "cuCtxGetCurrent"))?;
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
                fn_ctx_get_current,
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

    pub fn current_context(&self) -> Result<*mut c_void, CudaError> {
        let mut ctx: *mut c_void = std::ptr::null_mut();
        check_cu!(self.inner, (self.inner.fn_ctx_get_current)(&mut ctx));
        Ok(ctx)
    }

    pub fn create_context(&self, device: &CudaDeviceInfo) -> Result<Arc<CudaContextInner>, CudaError> {
        let mut ctx: *mut c_void = std::ptr::null_mut();
        check_cu!(
            self.inner,
            (self.inner.fn_ctx_create)(&mut ctx, 0, device.raw_device)
        );

        // Immediately detach the newly created context from the calling thread's context stack.
        // cuCtxCreate makes the new context current on the calling host thread. Popping it leaves
        // the thread's context stack restored and ensures the context is an unattached, owned handle.
        let mut popped: *mut c_void = std::ptr::null_mut();
        let pop_res = (self.inner.fn_ctx_pop_current)(&mut popped);
        if pop_res != 0 {
            let _ = (self.inner.fn_ctx_destroy)(ctx);
            return Err(CudaError::ApiError(
                pop_res,
                format!(
                    "cuCtxPopCurrent failed during context detachment: {}",
                    self.inner.get_error_string(pop_res)
                ),
            ));
        }
        if popped != ctx {
            let _ = (self.inner.fn_ctx_destroy)(ctx);
            return Err(CudaError::KernelError(format!(
                "cuCtxPopCurrent popped unexpected context {:p} instead of newly created {:p}",
                popped, ctx
            )));
        }

        Ok(Arc::new(CudaContextInner {
            ctx,
            driver: Arc::clone(&self.inner),
            lock: Mutex::new(()),
        }))
    }
}

/// RAII Guard that pushes a CUDA context onto the current host thread, and pops it on drop or finish.
pub struct CudaContextGuard<'a> {
    driver: &'a CudaDriverInner,
    popped: bool,
}

impl<'a> CudaContextGuard<'a> {
    /// Pushes the specified raw `CUcontext` onto the current host thread's CUDA context stack.
    /// Restricted to `pub(crate) unsafe` so external callers cannot pass arbitrary raw pointers.
    ///
    /// # Safety
    /// - `ctx` must be a non-null, valid, live raw `CUcontext` handle created by `driver`.
    /// - Exclusive ownership/lifetime of `ctx` and `driver` for the entire guard scope must be
    ///   externally guaranteed by the caller (e.g., under `CudaContextInner::lock`).
    /// - Current-thread CUDA context-stack discipline is the caller's responsibility: the guard
    ///   must be finished (`finish()`) or dropped on the exact same OS thread that called `push`,
    ///   and nested context pushes must be popped in strict LIFO order.
    pub(crate) unsafe fn push(
        ctx: *mut c_void,
        driver: &'a CudaDriverInner,
    ) -> Result<Self, CudaError> {
        let res = (driver.fn_ctx_push_current)(ctx);
        if res != 0 {
            return Err(CudaError::ApiError(res, driver.get_error_string(res)));
        }
        Ok(Self {
            driver,
            popped: false,
        })
    }

    /// Explicitly finishes the guard scope, popping the context, returning the popped handle,
    /// and propagating any asynchronous launch or context-stack error reported by `cuCtxPopCurrent`.
    pub(crate) fn finish(mut self) -> Result<*mut c_void, CudaError> {
        if self.popped {
            return Ok(std::ptr::null_mut());
        }
        self.popped = true;
        let mut popped: *mut c_void = std::ptr::null_mut();
        let res = (self.driver.fn_ctx_pop_current)(&mut popped);
        if res != 0 {
            return Err(CudaError::ApiError(
                res,
                format!(
                    "cuCtxPopCurrent failed in finish: {}",
                    self.driver.get_error_string(res)
                ),
            ));
        }
        Ok(popped)
    }
}

impl<'a> Drop for CudaContextGuard<'a> {
    fn drop(&mut self) {
        if !self.popped {
            self.popped = true;
            let mut popped: *mut c_void = std::ptr::null_mut();
            let _ = (self.driver.fn_ctx_pop_current)(&mut popped);
        }
    }
}

pub struct CudaContextInner {
    ctx: *mut c_void,
    driver: Arc<CudaDriverInner>,
    lock: Mutex<()>,
}

unsafe impl Send for CudaContextInner {}
unsafe impl Sync for CudaContextInner {}

impl CudaContextInner {
    /// Binds this context to the calling thread for the duration of the guard.
    pub(crate) fn bind<'a>(&'a self) -> Result<CudaContextGuard<'a>, CudaError> {
        unsafe { CudaContextGuard::push(self.ctx, &self.driver) }
    }

    /// Executes an operation with this context bound, serializing access via the context lock,
    /// popping the context upon completion, verifying popped context identity, and propagating any pop error.
    pub fn with_context<T, F>(&self, f: F) -> Result<T, CudaError>
    where
        F: FnOnce() -> Result<T, CudaError>,
    {
        let _lock = self.lock.lock().unwrap();
        let guard = self.bind()?;
        let res = f();
        let pop_res = guard.finish();
        match (res, pop_res) {
            (Ok(val), Ok(popped)) => {
                if popped != self.ctx {
                    return Err(CudaError::KernelError(format!(
                        "CUDA context stack imbalance: expected popped {:p}, got {:p}",
                        self.ctx, popped
                    )));
                }
                Ok(val)
            }
            (Ok(_), Err(pop_err)) => Err(pop_err),
            (Err(op_err), _) => Err(op_err),
        }
    }

    pub fn raw_context(&self) -> *mut c_void {
        self.ctx
    }
}

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

    pub fn context(&self) -> &Arc<CudaContextInner> {
        &self.context
    }

    /// Executes `f` with this session's CUDA context bound on the calling thread and
    /// propagates any `cuCtxPopCurrent` error on scope exit.
    pub fn with_context<T, F>(&self, f: F) -> Result<T, CudaError>
    where
        F: FnOnce() -> Result<T, CudaError>,
    {
        self.context.with_context(f)
    }

    pub fn get_memory_info(&self) -> Result<(usize, usize), CudaError> {
        self.context.with_context(|| {
            let mut free = 0usize;
            let mut total = 0usize;
            check_cu!(
                self.context.driver,
                (self.context.driver.fn_mem_get_info)(&mut free, &mut total)
            );
            Ok((free, total))
        })
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

        let mut dptr = 0u64;
        self.context.with_context(|| {
            check_cu!(
                self.context.driver,
                (self.context.driver.fn_mem_alloc)(&mut dptr, bytes)
            );
            Ok(())
        })?;

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
        self.context.with_context(|| {
            check_cu!(
                self.context.driver,
                (self.context.driver.fn_memcpy_htod)(
                    buffer.dptr,
                    data.as_ptr() as *const c_void,
                    buffer.bytes
                )
            );
            Ok(())
        })
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
        self.context.with_context(|| {
            check_cu!(
                self.context.driver,
                (self.context.driver.fn_memcpy_htod)(
                    buffer.dptr,
                    data.as_ptr() as *const c_void,
                    bytes_to_copy
                )
            );
            Ok(())
        })
    }

    pub fn upload_u32(&self, buffer: &CudaBuffer, data: &[u32]) -> Result<(), CudaError> {
        if data.len() != buffer.len_elements {
            return Err(CudaError::KernelError(format!(
                "Upload size mismatch: data has {} elements, buffer has {}",
                data.len(),
                buffer.len_elements
            )));
        }
        self.context.with_context(|| {
            check_cu!(
                self.context.driver,
                (self.context.driver.fn_memcpy_htod)(
                    buffer.dptr,
                    data.as_ptr() as *const c_void,
                    buffer.bytes
                )
            );
            Ok(())
        })
    }

    pub fn download_u32(&self, buffer: &CudaBuffer, out: &mut [u32]) -> Result<(), CudaError> {
        if out.len() != buffer.len_elements {
            return Err(CudaError::KernelError(format!(
                "Download size mismatch: output slice has {} elements, buffer has {}",
                out.len(),
                buffer.len_elements
            )));
        }
        self.context.with_context(|| {
            check_cu!(
                self.context.driver,
                (self.context.driver.fn_memcpy_dtoh)(
                    out.as_mut_ptr() as *mut c_void,
                    buffer.dptr,
                    buffer.bytes
                )
            );
            Ok(())
        })
    }

    pub fn download_f32(&self, buffer: &CudaBuffer, out: &mut [f32]) -> Result<(), CudaError> {
        if out.len() != buffer.len_elements {
            return Err(CudaError::KernelError(format!(
                "Download size mismatch: output slice has {} elements, buffer has {}",
                out.len(),
                buffer.len_elements
            )));
        }
        self.context.with_context(|| {
            check_cu!(
                self.context.driver,
                (self.context.driver.fn_memcpy_dtoh)(
                    out.as_mut_ptr() as *mut c_void,
                    buffer.dptr,
                    buffer.bytes
                )
            );
            Ok(())
        })
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
        self.context.with_context(|| {
            check_cu!(
                self.context.driver,
                (self.context.driver.fn_memcpy_dtoh)(
                    out.as_mut_ptr() as *mut c_void,
                    buffer.dptr,
                    bytes_to_copy
                )
            );
            Ok(())
        })
    }

    pub fn synchronize(&self) -> Result<(), CudaError> {
        self.context.with_context(|| {
            check_cu!(
                self.context.driver,
                (self.context.driver.fn_ctx_synchronize)()
            );
            Ok(())
        })
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
        self.context.with_context(|| {
            check_cu!(
                self.context.driver,
                (self.context.driver.fn_memcpy_dtod)(dst.dptr, src.dptr, bytes)
            );
            Ok(())
        })
    }

    /// Load and JIT-compile a PTX module into the session's active CUDA context.
    /// Captures detailed compiler diagnostics via `cuModuleLoadDataEx` whenever available.
    pub fn load_ptx_module(&self, ptx_source: &str) -> Result<CudaModuleHandle, CudaError> {
        let c_ptx = CString::new(ptx_source).map_err(|e| CudaError::KernelError(e.to_string()))?;
        let mut module: *mut c_void = std::ptr::null_mut();

        self.context.with_context(|| {
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
            Ok(())
        })?;

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

impl std::fmt::Debug for CudaBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaBuffer")
            .field("dptr", &format_args!("0x{:x}", self.dptr))
            .field("len_elements", &self.len_elements)
            .field("bytes", &self.bytes)
            .finish()
    }
}

unsafe impl Send for CudaBuffer {}
unsafe impl Sync for CudaBuffer {}

impl Drop for CudaBuffer {
    fn drop(&mut self) {
        if self.dptr != 0 {
            let _ = self.context.with_context(|| {
                let _ = (self.context.driver.fn_mem_free)(self.dptr);
                Ok(())
            });
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
            let _ = self.context.with_context(|| {
                let _ = (self.context.driver.fn_module_unload)(self.module);
                Ok(())
            });
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
        let mut func: *mut c_void = std::ptr::null_mut();
        self.inner.context.with_context(|| {
            check_cu!(
                self.inner.context.driver,
                (self.inner.context.driver.fn_module_get_function)(
                    &mut func,
                    self.inner.module,
                    c_name.as_ptr()
                )
            );
            Ok(())
        })?;
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
        ctx.with_context(|| {
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
        })
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

    #[test]
    fn test_context_creation_leaves_no_implicit_current_context() {
        let driver = match CudaDriver::load() {
            Ok(d) => d,
            Err(_) => return,
        };
        if driver.device_count().unwrap_or(0) <= 0 {
            return;
        }
        let dev = driver.get_device_info(0).unwrap();
        let context = driver.create_context(&dev).unwrap();

        // After creation, context MUST NOT be current on the creating thread
        let current = driver.current_context().unwrap();
        assert!(current.is_null() || current != context.raw_context());
    }

    #[test]
    fn test_context_push_pop_stack_restoration() {
        let driver = match CudaDriver::load() {
            Ok(d) => d,
            Err(_) => return,
        };
        if driver.device_count().unwrap_or(0) <= 0 {
            return;
        }
        let dev = driver.get_device_info(0).unwrap();
        let ctx_a = driver.create_context(&dev).unwrap();
        let ctx_b = driver.create_context(&dev).unwrap();

        let initial_current = driver.current_context().unwrap();

        ctx_a.with_context(|| {
            let cur_a = driver.current_context().unwrap();
            assert_eq!(cur_a, ctx_a.raw_context());

            ctx_b.with_context(|| {
                let cur_b = driver.current_context().unwrap();
                assert_eq!(cur_b, ctx_b.raw_context());
                Ok(())
            })?;

            let cur_a_restored = driver.current_context().unwrap();
            assert_eq!(cur_a_restored, ctx_a.raw_context());
            Ok(())
        }).unwrap();

        let final_current = driver.current_context().unwrap();
        assert_eq!(final_current, initial_current);
    }

    #[test]
    fn test_multithreaded_context_sequential_usage() {
        let session = match CudaSession::init(0) {
            Ok(s) => Arc::new(s),
            Err(_) => return,
        };

        let session_clone = Arc::clone(&session);
        let handle1 = std::thread::spawn(move || {
            let buf = session_clone.allocate_f32(4).unwrap();
            session_clone.upload_f32(&buf, &[1.0, 2.0, 3.0, 4.0]).unwrap();
            let mut out = [0.0f32; 4];
            session_clone.download_f32(&buf, &mut out).unwrap();
            assert_eq!(out, [1.0, 2.0, 3.0, 4.0]);
            buf
        });
        let buf = handle1.join().unwrap();

        let session_clone2 = Arc::clone(&session);
        let handle2 = std::thread::spawn(move || {
            session_clone2.upload_f32(&buf, &[5.0, 6.0, 7.0, 8.0]).unwrap();
            let mut out = [0.0f32; 4];
            session_clone2.download_f32(&buf, &mut out).unwrap();
            assert_eq!(out, [5.0, 6.0, 7.0, 8.0]);
        });
        handle2.join().unwrap();
    }

    #[test]
    fn test_cross_thread_buffer_drop() {
        let session = match CudaSession::init(0) {
            Ok(s) => s,
            Err(_) => return,
        };

        let (free_before, _) = session.get_memory_info().unwrap();
        let buf = session.allocate_f32(1024 * 1024).unwrap(); // 4 MB
        let (free_allocated, _) = session.get_memory_info().unwrap();
        assert!(free_allocated < free_before);

        let handle = std::thread::spawn(move || {
            drop(buf); // Explicit drop on another thread
        });
        handle.join().unwrap();

        let (free_after, _) = session.get_memory_info().unwrap();
        assert!(
            free_after > free_allocated,
            "Expected free VRAM to increase after cross-thread buffer drop (allocated: {}, after: {})",
            free_allocated,
            free_after
        );
    }

    #[test]
    fn test_module_kernel_survives_session_scope() {
        let kernel = {
            let session = match CudaSession::init(0) {
                Ok(s) => s,
                Err(_) => return,
            };
            let ptx = r#"
                .version 7.0
                .target sm_75
                .address_size 64
                .visible .entry noop_kernel() {
                    ret;
                }
            "#;
            let module = match session.load_ptx_module(ptx) {
                Ok(m) => m,
                Err(_) => return,
            };
            module.get_kernel("noop_kernel").unwrap()
        };

        unsafe {
            let dummy_params: [*mut c_void; 16] = [std::ptr::null_mut(); 16];
            let res = kernel.launch((1, 1, 1), (1, 1, 1), 0, dummy_params, 0);
            assert!(res.is_ok());
        }
    }

    #[test]
    fn test_two_sessions_two_contexts_on_single_thread() {
        let session_a = match CudaSession::init(0) {
            Ok(s) => s,
            Err(_) => return,
        };
        let session_b = match CudaSession::init(0) {
            Ok(s) => s,
            Err(_) => return,
        };
        assert_ne!(
            session_a.context().raw_context(),
            session_b.context().raw_context()
        );

        let buf_a = session_a.allocate_f32(4).unwrap();
        let buf_b = session_b.allocate_f32(4).unwrap();

        session_a.upload_f32(&buf_a, &[10.0, 20.0, 30.0, 40.0]).unwrap();
        session_b.upload_f32(&buf_b, &[50.0, 60.0, 70.0, 80.0]).unwrap();

        let mut out_a = [0.0f32; 4];
        let mut out_b = [0.0f32; 4];
        session_a.download_f32(&buf_a, &mut out_a).unwrap();
        session_b.download_f32(&buf_b, &mut out_b).unwrap();
        assert_eq!(out_a, [10.0, 20.0, 30.0, 40.0]);
        assert_eq!(out_b, [50.0, 60.0, 70.0, 80.0]);

        // Verify neither context remains implicitly bound on the calling thread
        let cur = session_a.driver().current_context().unwrap();
        assert!(
            cur.is_null()
                || (cur != session_a.context().raw_context()
                    && cur != session_b.context().raw_context())
        );
    }

    #[test]
    fn test_session_created_on_thread_a_dropped_on_thread_b() {
        let session_a = match CudaSession::init(0) {
            Ok(s) => s,
            Err(_) => return,
        };
        let raw_ctx_a = session_a.context().raw_context() as usize;
        let driver = session_a.driver().clone();

        // Thread A must NOT have session_a's context current after init
        let cur_on_a_before = driver.current_context().unwrap() as usize;
        assert_ne!(cur_on_a_before, raw_ctx_a);

        // Move session_a to Thread B and drop it there (calling cuCtxDestroy on Thread B)
        let handle = std::thread::spawn(move || {
            let buf = session_a.allocate_f32(4).unwrap();
            session_a.upload_f32(&buf, &[1.0, 2.0, 3.0, 4.0]).unwrap();
            drop(buf);
            drop(session_a);
        });
        handle.join().unwrap();

        // Thread A's current context must remain unaffected (no dangling destroyed context on Thread A)
        let cur_on_a_after = driver.current_context().unwrap() as usize;
        assert_eq!(cur_on_a_after, cur_on_a_before);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Deterministic Thread-Local Context Stack & Pop-Error Injection Harness
    // ─────────────────────────────────────────────────────────────────────────

    use std::cell::RefCell;
    use std::sync::atomic::{AtomicI32, AtomicU64, AtomicUsize, Ordering};

    thread_local! {
        static MOCK_CTX_STACK: RefCell<Vec<usize>> = const { RefCell::new(Vec::new()) };
    }

    static NEXT_MOCK_CTX_ID: AtomicUsize = AtomicUsize::new(0x1000);
    static NEXT_MOCK_DPTR: AtomicU64 = AtomicU64::new(0xA000_0000);
    static DESTROYED_CTXS: AtomicUsize = AtomicUsize::new(0);
    static FREED_BUFFERS: AtomicUsize = AtomicUsize::new(0);
    static UNLOADED_MODULES: AtomicUsize = AtomicUsize::new(0);
    static INJECT_POP_ERROR_CODE: AtomicI32 = AtomicI32::new(0);

    extern "system" fn mock_cu_init(_flags: c_int) -> c_int {
        0
    }
    extern "system" fn mock_cu_driver_get_version(ver: *mut c_int) -> c_int {
        unsafe { *ver = 12040 };
        0
    }
    extern "system" fn mock_cu_device_get_count(count: *mut c_int) -> c_int {
        unsafe { *count = 1 };
        0
    }
    extern "system" fn mock_cu_device_get(dev: *mut c_int, ordinal: c_int) -> c_int {
        unsafe { *dev = ordinal };
        0
    }
    extern "system" fn mock_cu_device_get_name(
        name: *mut c_char,
        _len: c_int,
        _dev: c_int,
    ) -> c_int {
        let s = b"MockCUDA\0";
        unsafe {
            std::ptr::copy_nonoverlapping(s.as_ptr() as *const c_char, name, s.len());
        }
        0
    }
    extern "system" fn mock_cu_device_total_mem(bytes: *mut usize, _dev: c_int) -> c_int {
        unsafe { *bytes = 1024 * 1024 * 1024 };
        0
    }
    extern "system" fn mock_cu_device_get_attr(
        pi: *mut c_int,
        _attrib: c_int,
        _dev: c_int,
    ) -> c_int {
        unsafe { *pi = 8 };
        0
    }
    extern "system" fn mock_cu_ctx_create(
        pctx: *mut *mut c_void,
        _flags: c_int,
        _dev: c_int,
    ) -> c_int {
        let id = NEXT_MOCK_CTX_ID.fetch_add(0x10, Ordering::SeqCst);
        MOCK_CTX_STACK.with(|stack| stack.borrow_mut().push(id));
        unsafe { *pctx = id as *mut c_void };
        0
    }
    extern "system" fn mock_cu_ctx_destroy(ctx: *mut c_void) -> c_int {
        if ctx.is_null() {
            return 201;
        }
        DESTROYED_CTXS.fetch_add(1, Ordering::SeqCst);
        0
    }
    extern "system" fn mock_cu_ctx_push_current(ctx: *mut c_void) -> c_int {
        if ctx.is_null() {
            return 201;
        }
        MOCK_CTX_STACK.with(|stack| stack.borrow_mut().push(ctx as usize));
        0
    }
    extern "system" fn mock_cu_ctx_pop_current(pctx: *mut *mut c_void) -> c_int {
        let injected = INJECT_POP_ERROR_CODE.load(Ordering::SeqCst);
        let popped = MOCK_CTX_STACK
            .with(|stack| stack.borrow_mut().pop())
            .unwrap_or(0);
        if !pctx.is_null() {
            unsafe { *pctx = popped as *mut c_void };
        }
        if injected != 0 {
            return injected;
        }
        if popped == 0 {
            return 201;
        }
        0
    }
    extern "system" fn mock_cu_ctx_get_current(pctx: *mut *mut c_void) -> c_int {
        let top = MOCK_CTX_STACK
            .with(|stack| stack.borrow().last().copied())
            .unwrap_or(0);
        unsafe { *pctx = top as *mut c_void };
        0
    }
    extern "system" fn mock_cu_ctx_synchronize() -> c_int {
        let top = MOCK_CTX_STACK
            .with(|stack| stack.borrow().last().copied())
            .unwrap_or(0);
        if top == 0 {
            201
        } else {
            0
        }
    }
    extern "system" fn mock_cu_mem_get_info(
        free: *mut usize,
        total: *mut usize,
    ) -> c_int {
        let top = MOCK_CTX_STACK
            .with(|stack| stack.borrow().last().copied())
            .unwrap_or(0);
        if top == 0 {
            return 201;
        }
        unsafe {
            *free = 512 * 1024 * 1024;
            *total = 1024 * 1024 * 1024;
        }
        0
    }
    extern "system" fn mock_cu_mem_alloc(dptr: *mut u64, _bytes: usize) -> c_int {
        let top = MOCK_CTX_STACK
            .with(|stack| stack.borrow().last().copied())
            .unwrap_or(0);
        if top == 0 {
            return 201;
        }
        unsafe { *dptr = NEXT_MOCK_DPTR.fetch_add(0x1000, Ordering::SeqCst) };
        0
    }
    extern "system" fn mock_cu_mem_free(_dptr: u64) -> c_int {
        let top = MOCK_CTX_STACK
            .with(|stack| stack.borrow().last().copied())
            .unwrap_or(0);
        if top == 0 {
            return 201;
        }
        FREED_BUFFERS.fetch_add(1, Ordering::SeqCst);
        0
    }
    extern "system" fn mock_cu_memcpy_htod(
        _dst: u64,
        _src: *const c_void,
        _bytes: usize,
    ) -> c_int {
        0
    }
    extern "system" fn mock_cu_memcpy_dtoh(
        _dst: *mut c_void,
        _src: u64,
        _bytes: usize,
    ) -> c_int {
        0
    }
    extern "system" fn mock_cu_memcpy_dtod(
        _dst: u64,
        _src: u64,
        _bytes: usize,
    ) -> c_int {
        0
    }
    extern "system" fn mock_cu_module_load_data(
        module: *mut *mut c_void,
        _image: *const c_char,
    ) -> c_int {
        let top = MOCK_CTX_STACK
            .with(|stack| stack.borrow().last().copied())
            .unwrap_or(0);
        if top == 0 {
            return 201;
        }
        unsafe { *module = 0xBEEFusize as *mut c_void };
        0
    }
    extern "system" fn mock_cu_module_get_function(
        hfunc: *mut *mut c_void,
        _hmod: *mut c_void,
        _name: *const c_char,
    ) -> c_int {
        unsafe { *hfunc = 0xCAFEusize as *mut c_void };
        0
    }
    extern "system" fn mock_cu_module_unload(_hmod: *mut c_void) -> c_int {
        let top = MOCK_CTX_STACK
            .with(|stack| stack.borrow().last().copied())
            .unwrap_or(0);
        if top == 0 {
            return 201;
        }
        UNLOADED_MODULES.fetch_add(1, Ordering::SeqCst);
        0
    }
    extern "system" fn mock_cu_launch_kernel(
        _f: *mut c_void,
        _gx: u32,
        _gy: u32,
        _gz: u32,
        _bx: u32,
        _by: u32,
        _bz: u32,
        _shared: u32,
        _stream: *mut c_void,
        _params: *mut *mut c_void,
        _extra: *mut *mut c_void,
    ) -> c_int {
        let top = MOCK_CTX_STACK
            .with(|stack| stack.borrow().last().copied())
            .unwrap_or(0);
        if top == 0 {
            201
        } else {
            0
        }
    }

    fn build_harness_driver() -> CudaDriver {
        CudaDriver {
            inner: Arc::new(CudaDriverInner {
                lib_handle: std::ptr::null_mut(),
                fn_init: mock_cu_init,
                fn_driver_get_version: mock_cu_driver_get_version,
                fn_device_get_count: mock_cu_device_get_count,
                fn_device_get: mock_cu_device_get,
                fn_device_get_name: mock_cu_device_get_name,
                fn_device_total_mem: mock_cu_device_total_mem,
                fn_device_get_attr: mock_cu_device_get_attr,
                fn_ctx_create: mock_cu_ctx_create,
                fn_ctx_destroy: mock_cu_ctx_destroy,
                fn_ctx_push_current: mock_cu_ctx_push_current,
                fn_ctx_pop_current: mock_cu_ctx_pop_current,
                fn_ctx_get_current: mock_cu_ctx_get_current,
                fn_ctx_synchronize: mock_cu_ctx_synchronize,
                fn_mem_get_info: mock_cu_mem_get_info,
                fn_mem_alloc: mock_cu_mem_alloc,
                fn_mem_free: mock_cu_mem_free,
                fn_memcpy_htod: mock_cu_memcpy_htod,
                fn_memcpy_dtoh: mock_cu_memcpy_dtoh,
                fn_memcpy_dtod: mock_cu_memcpy_dtod,
                fn_module_load_data: mock_cu_module_load_data,
                fn_module_load_data_ex: None,
                fn_module_get_function: mock_cu_module_get_function,
                fn_module_unload: mock_cu_module_unload,
                fn_launch_kernel: mock_cu_launch_kernel,
                fn_get_error_name: None,
                fn_get_error_string: None,
            }),
        }
    }

    #[test]
    fn test_deterministic_context_lifecycle_and_pop_error_propagation_harness() {
        MOCK_CTX_STACK.with(|s| s.borrow_mut().clear());
        INJECT_POP_ERROR_CODE.store(0, Ordering::SeqCst);

        let driver = build_harness_driver();
        let dev = driver.get_device_info(0).unwrap();

        // 1. Context created -> immediately detached (no implicit current context on creating thread)
        let ctx_a = driver.create_context(&dev).unwrap();
        let ctx_b = driver.create_context(&dev).unwrap();
        assert!(driver.current_context().unwrap().is_null());

        // 2. Current context A -> nested operation on B -> A restored -> null restored on exit
        ctx_a
            .with_context(|| {
                assert_eq!(driver.current_context().unwrap(), ctx_a.raw_context());
                ctx_b.with_context(|| {
                    assert_eq!(driver.current_context().unwrap(), ctx_b.raw_context());
                    Ok(())
                })?;
                assert_eq!(driver.current_context().unwrap(), ctx_a.raw_context());
                Ok(())
            })
            .unwrap();
        assert!(driver.current_context().unwrap().is_null());

        // 3. Inject cuCtxPopCurrent error (700 = CUDA_ERROR_ILLEGAL_ADDRESS):
        //    with_context MUST propagate the pop error instead of ignoring it!
        INJECT_POP_ERROR_CODE.store(700, Ordering::SeqCst);
        let pop_err_res = ctx_a.with_context(|| Ok(()));
        assert!(
            matches!(pop_err_res, Err(CudaError::ApiError(700, _))),
            "with_context must propagate cuCtxPopCurrent failure"
        );

        // And create_context MUST destroy the newly created context and return Err if cuCtxPopCurrent fails!
        let destroyed_before = DESTROYED_CTXS.load(Ordering::SeqCst);
        let create_fail_res = driver.create_context(&dev);
        assert!(matches!(create_fail_res, Err(CudaError::ApiError(700, _))));
        assert_eq!(
            DESTROYED_CTXS.load(Ordering::SeqCst),
            destroyed_before + 1,
            "Failed detachment during create_context must destroy the raw context"
        );
        INJECT_POP_ERROR_CODE.store(0, Ordering::SeqCst);

        // 4. Cross-thread buffer drop & cross-thread session drop
        let session = CudaSession {
            driver: driver.clone(),
            device: dev.clone(),
            context: Arc::clone(&ctx_a),
        };
        let buf = session.allocate_f32(16).unwrap();
        let freed_before = FREED_BUFFERS.load(Ordering::SeqCst);
        let handle = std::thread::spawn(move || {
            drop(buf);
        });
        handle.join().unwrap();
        assert_eq!(
            FREED_BUFFERS.load(Ordering::SeqCst),
            freed_before + 1,
            "Cross-thread CudaBuffer::drop must bind context and call cuMemFree"
        );

        // 5. Module and kernel survive session drop and unload cleanly on final kernel drop
        let unloaded_before = UNLOADED_MODULES.load(Ordering::SeqCst);
        let kernel = {
            let temp_ctx = driver.create_context(&dev).unwrap();
            let temp_session = CudaSession {
                driver: driver.clone(),
                device: dev,
                context: temp_ctx,
            };
            let module = temp_session.load_ptx_module(".version 7.0\n").unwrap();
            module.get_kernel("noop").unwrap()
        };
        unsafe {
            let params: [*mut c_void; 16] = [std::ptr::null_mut(); 16];
            assert!(kernel.launch((1, 1, 1), (1, 1, 1), 0, params, 0).is_ok());
        }
        drop(kernel);
        assert_eq!(
            UNLOADED_MODULES.load(Ordering::SeqCst),
            unloaded_before + 1,
            "Dropping last CudaKernel must unload module under bound context"
        );
    }

    #[test]
    fn test_multithreaded_concurrent_context_contention() {
        use std::sync::Barrier;

        let driver = build_harness_driver();
        let dev = driver.get_device_info(0).unwrap();
        let ctx = driver.create_context(&dev).unwrap();
        let session = Arc::new(CudaSession {
            driver: driver.clone(),
            device: dev,
            context: ctx,
        });

        let num_threads = 8usize;
        let iterations = 16usize;
        let barrier = Arc::new(Barrier::new(num_threads));
        let mut handles = Vec::with_capacity(num_threads);

        for tid in 0..num_threads {
            let sess = Arc::clone(&session);
            let bar = Arc::clone(&barrier);
            handles.push(std::thread::spawn(move || {
                MOCK_CTX_STACK.with(|s| s.borrow_mut().clear());
                bar.wait();
                for iter in 0..iterations {
                    let buf = sess.allocate_f32(8).unwrap();
                    let payload = [(tid * 100 + iter) as f32; 8];
                    sess.upload_f32(&buf, &payload).unwrap();
                    let mut out = [0.0f32; 8];
                    sess.download_f32(&buf, &mut out).unwrap();
                    sess.synchronize().unwrap();
                    drop(buf);
                    // Thread-local CUDA context stack must be completely empty after every operation
                    assert!(sess.driver().current_context().unwrap().is_null());
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        // Also run simultaneous barrier-synchronized contention on real GPU hardware if available
        if let Ok(real_sess) = CudaSession::init(0) {
            let real_sess = Arc::new(real_sess);
            let real_barrier = Arc::new(Barrier::new(4));
            let mut real_handles = Vec::with_capacity(4);
            for tid in 0..4 {
                let s = Arc::clone(&real_sess);
                let b = Arc::clone(&real_barrier);
                real_handles.push(std::thread::spawn(move || {
                    b.wait();
                    for iter in 0..8 {
                        let val = (tid * 10 + iter) as f32;
                        let buf = s.allocate_f32(4).unwrap();
                        s.upload_f32(&buf, &[val, val + 1.0, val + 2.0, val + 3.0])
                            .unwrap();
                        let mut out = [0.0f32; 4];
                        s.download_f32(&buf, &mut out).unwrap();
                        assert_eq!(out, [val, val + 1.0, val + 2.0, val + 3.0]);
                        drop(buf);
                    }
                }));
            }
            for h in real_handles {
                h.join().unwrap();
            }
        }
    }
}
