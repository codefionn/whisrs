//! NVIDIA: NVML from the driver's `libnvidia-ml.so.1`, opened at runtime so
//! whisrs neither links against nor requires the NVIDIA driver.

use std::ffi::{c_char, c_int, c_void, CString};
use std::sync::OnceLock;

use super::{VramUsage, NVIDIA_PROC_GPUS};

const NVML_SUCCESS: c_int = 0;

#[repr(C)]
#[derive(Default)]
struct NvmlMemory {
    total: u64,
    free: u64,
    used: u64,
}

type NvmlDevice = *mut c_void;

type InitFn = unsafe extern "C" fn() -> c_int;
type ByPciBusIdFn = unsafe extern "C" fn(*const c_char, *mut NvmlDevice) -> c_int;
type MemoryInfoFn = unsafe extern "C" fn(NvmlDevice, *mut NvmlMemory) -> c_int;

struct Nvml {
    by_pci_bus_id: ByPciBusIdFn,
    memory_info: MemoryInfoFn,
}

/// Load and initialize NVML once; `None` when the library is missing or
/// `nvmlInit` fails (no NVIDIA driver loaded). The library stays loaded for
/// the life of the process.
fn nvml() -> Option<&'static Nvml> {
    static NVML: OnceLock<Option<Nvml>> = OnceLock::new();
    NVML.get_or_init(|| {
        // SAFETY: dlopen/dlsym with NUL-terminated names; each symbol is cast
        // to the signature documented in nvml.h.
        unsafe {
            // NixOS keeps driver libraries out of the default search path.
            let lib = [
                c"libnvidia-ml.so.1",
                c"/run/opengl-driver/lib/libnvidia-ml.so.1",
            ]
            .iter()
            .map(|name| libc::dlopen(name.as_ptr(), libc::RTLD_NOW))
            .find(|lib| !lib.is_null())?;
            let sym = |name: &std::ffi::CStr| {
                let p = libc::dlsym(lib, name.as_ptr());
                (!p.is_null()).then_some(p)
            };
            let init = std::mem::transmute::<*mut c_void, InitFn>(sym(c"nvmlInit_v2")?);
            let nvml = Nvml {
                by_pci_bus_id: std::mem::transmute::<*mut c_void, ByPciBusIdFn>(sym(
                    c"nvmlDeviceGetHandleByPciBusId_v2",
                )?),
                memory_info: std::mem::transmute::<*mut c_void, MemoryInfoFn>(sym(
                    c"nvmlDeviceGetMemoryInfo",
                )?),
            };
            (init() == NVML_SUCCESS).then_some(nvml)
        }
    })
    .as_ref()
}

pub(super) fn usage(pci_addr: &str) -> Option<VramUsage> {
    let nvml = nvml()?;
    let bus_id = CString::new(pci_addr).ok()?;
    let mut dev: NvmlDevice = std::ptr::null_mut();
    let mut mem = NvmlMemory::default();
    // SAFETY: valid C string and out-pointers; NVML is initialized.
    unsafe {
        if (nvml.by_pci_bus_id)(bus_id.as_ptr(), &mut dev) != NVML_SUCCESS {
            return None;
        }
        if (nvml.memory_info)(dev, &mut mem) != NVML_SUCCESS {
            return None;
        }
    }
    (mem.total > 0).then_some(VramUsage {
        total: mem.total,
        // `used` includes driver-reserved memory; `total - free` is the
        // figure that matters for "will a new allocation fit".
        used: mem.total.saturating_sub(mem.free),
    })
}

/// The `Device Minor` from a `/proc/driver/nvidia/gpus/<addr>/information`.
fn device_minor(information: &str) -> Option<u32> {
    information
        .lines()
        .find_map(|l| l.strip_prefix("Device Minor:"))
        .and_then(|v| v.trim().parse().ok())
}

/// PCI addresses of NVIDIA GPUs this process has open (`/dev/nvidiaN`).
pub(super) fn own_devices() -> Vec<String> {
    let minors: Vec<u32> = std::fs::read_dir("/proc/self/fd")
        .into_iter()
        .flatten()
        .filter_map(Result::ok)
        .filter_map(|e| std::fs::read_link(e.path()).ok())
        .filter_map(|p| p.to_str()?.strip_prefix("/dev/nvidia")?.parse::<u32>().ok())
        .collect();
    if minors.is_empty() {
        return Vec::new();
    }
    std::fs::read_dir(NVIDIA_PROC_GPUS)
        .into_iter()
        .flatten()
        .filter_map(Result::ok)
        .filter(|e| {
            std::fs::read_to_string(e.path().join("information"))
                .ok()
                .and_then(|t| device_minor(&t))
                .is_some_and(|m| minors.contains(&m))
        })
        .map(|e| e.file_name().to_string_lossy().to_ascii_lowercase())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_device_minor() {
        let info = "Model: \t\t NVIDIA GeForce RTX 4070\nIRQ:   \t\t 180\n\
            GPU UUID: \t GPU-0000\nDevice Minor: \t 1\nBus Location: \t 0000:01:00.0\n";
        assert_eq!(device_minor(info), Some(1));
        assert_eq!(device_minor("Model: x\n"), None);
    }

    #[test]
    fn missing_nvml_is_not_an_error() {
        // On machines without the NVIDIA driver this must quietly be None.
        let _ = usage("0000:ff:00.0");
    }
}
