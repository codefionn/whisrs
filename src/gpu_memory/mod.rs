//! VRAM usage per GPU, read straight from the kernel drivers.
//!
//! Vulkan has no portable "how full is the card" query that works without
//! privileges, so each vendor gets its own reader:
//!
//! - AMD (`amdgpu`): `mem_info_vram_{total,used}` in sysfs.
//! - Intel Arc on `xe`: the `DRM_IOCTL_XE_DEVICE_QUERY` memory-region query.
//! - Intel Arc on `i915`: the `DRM_IOCTL_I915_QUERY` memory-region query.
//! - NVIDIA: NVML (`libnvidia-ml.so.1`), loaded at runtime with `dlopen`.
//!
//! The Intel queries only report usage to `CAP_PERFMON` on some kernels; when
//! they come back empty the used figure is summed from the `drm-resident-*`
//! keys in `/proc/*/fdinfo` instead (see [`fdinfo`]). That sum only sees
//! processes this user can read, so it undercounts other users' GPU clients.
//!
//! [`own_devices`] names the GPUs this process holds VRAM on, so the caller
//! watches the card the model actually lives on.

mod amd;
pub mod fdinfo;
mod intel;
mod nvidia;

use std::path::{Path, PathBuf};

const SYS_DRM: &str = "/sys/class/drm";
const NVIDIA_PROC_GPUS: &str = "/proc/driver/nvidia/gpus";

/// PCI vendor ids.
const VENDOR_AMD: u16 = 0x1002;
const VENDOR_INTEL: u16 = 0x8086;
const VENDOR_NVIDIA: u16 = 0x10de;

/// Which reader a device needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Driver {
    Amdgpu,
    Xe,
    I915,
    Nvidia,
    Other,
}

impl Driver {
    fn from_name(name: &str, vendor: u16) -> Self {
        match name {
            "amdgpu" => Self::Amdgpu,
            "xe" => Self::Xe,
            "i915" => Self::I915,
            "nvidia" => Self::Nvidia,
            _ if vendor == VENDOR_NVIDIA => Self::Nvidia,
            _ => Self::Other,
        }
    }
}

/// One GPU, identified by its PCI address.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuDevice {
    /// PCI address, lowercase, e.g. `0000:07:00.0`. Matches `drm-pdev` in
    /// fdinfo and the bus id NVML accepts.
    pub pci_addr: String,
    pub vendor: u16,
    pub driver: Driver,
    /// `/sys/bus/pci/devices/<addr>`.
    pub sysfs: PathBuf,
}

impl GpuDevice {
    pub fn vendor_name(&self) -> &'static str {
        match self.vendor {
            VENDOR_AMD => "AMD",
            VENDOR_INTEL => "Intel",
            VENDOR_NVIDIA => "NVIDIA",
            _ => "unknown",
        }
    }

    /// The DRM render node (`/dev/dri/renderD*`) for this device, if any.
    fn render_node(&self) -> Option<PathBuf> {
        std::fs::read_dir(self.sysfs.join("drm"))
            .ok()?
            .filter_map(Result::ok)
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .find(|n| n.starts_with("renderD"))
            .map(|n| Path::new("/dev/dri").join(n))
    }
}

/// Device-local memory of one GPU, in bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VramUsage {
    pub total: u64,
    pub used: u64,
}

impl VramUsage {
    pub fn free(&self) -> u64 {
        self.total.saturating_sub(self.used)
    }
}

/// Every GPU visible through DRM sysfs or the NVIDIA proc interface.
pub fn devices() -> Vec<GpuDevice> {
    let mut out: Vec<GpuDevice> = Vec::new();

    if let Ok(entries) = std::fs::read_dir(SYS_DRM) {
        for entry in entries.filter_map(Result::ok) {
            let name = entry.file_name().to_string_lossy().into_owned();
            // `card0` is a device; `card0-DP-1` is one of its connectors.
            if !name.starts_with("card") || !name[4..].bytes().all(|b| b.is_ascii_digit()) {
                continue;
            }
            if let Some(dev) = pci_device(&entry.path().join("device")) {
                if !out.iter().any(|d| d.pci_addr == dev.pci_addr) {
                    out.push(dev);
                }
            }
        }
    }

    // The proprietary NVIDIA driver registers DRM cards only with
    // `nvidia-drm.modeset=1`; its proc directory lists every GPU regardless.
    if let Ok(entries) = std::fs::read_dir(NVIDIA_PROC_GPUS) {
        for entry in entries.filter_map(Result::ok) {
            let addr = entry.file_name().to_string_lossy().to_ascii_lowercase();
            let sysfs = Path::new("/sys/bus/pci/devices").join(&addr);
            if !out.iter().any(|d| d.pci_addr == addr) {
                if let Some(dev) = pci_device(&sysfs) {
                    out.push(dev);
                }
            }
        }
    }

    out.sort_by(|a, b| a.pci_addr.cmp(&b.pci_addr));
    out
}

fn pci_device(sysfs: &Path) -> Option<GpuDevice> {
    let sysfs = std::fs::canonicalize(sysfs).ok()?;
    let pci_addr = sysfs.file_name()?.to_string_lossy().to_ascii_lowercase();
    let vendor = read_hex_u16(&sysfs.join("vendor"))?;
    let driver_name = std::fs::read_link(sysfs.join("driver"))
        .ok()
        .and_then(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()))
        .unwrap_or_default();
    Some(GpuDevice {
        pci_addr,
        vendor,
        driver: Driver::from_name(&driver_name, vendor),
        sysfs,
    })
}

fn read_hex_u16(path: &Path) -> Option<u16> {
    let s = std::fs::read_to_string(path).ok()?;
    u16::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok()
}

/// Current VRAM usage of `dev`, or `None` if no reader works for it (no
/// VRAM, missing driver interface, NVML not installed).
pub fn usage(dev: &GpuDevice) -> Option<VramUsage> {
    match dev.driver {
        Driver::Amdgpu => amd::usage(&dev.sysfs),
        Driver::Xe => {
            let q = intel::xe_usage(&dev.render_node()?)?;
            // `used` is 0 for callers without CAP_PERFMON on some kernels;
            // an empty VRAM is impossible while anything is on screen.
            Some(if q.used == 0 {
                with_fdinfo_used(dev, q)
            } else {
                q
            })
        }
        Driver::I915 => {
            let q = intel::i915_usage(&dev.render_node()?)?;
            // Without CAP_PERFMON i915 reports unallocated == probed.
            Some(if q.used == 0 {
                with_fdinfo_used(dev, q)
            } else {
                q
            })
        }
        Driver::Nvidia => nvidia::usage(&dev.pci_addr),
        Driver::Other => None,
    }
}

fn with_fdinfo_used(dev: &GpuDevice, q: VramUsage) -> VramUsage {
    VramUsage {
        total: q.total,
        used: fdinfo::system_vram_used(&dev.pci_addr),
    }
}

/// PCI addresses of the GPUs this process holds device memory on.
///
/// DRM drivers (amdgpu, xe, i915, nouveau) report it per open file in
/// `/proc/self/fdinfo`; the proprietary NVIDIA driver does not, so an open
/// `/dev/nvidiaN` is mapped to its card through `/proc/driver/nvidia`.
pub fn own_devices() -> Vec<String> {
    let mut addrs = fdinfo::own_vram_devices();
    for addr in nvidia::own_devices() {
        if !addrs.contains(&addr) {
            addrs.push(addr);
        }
    }
    addrs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn driver_names_map_to_readers() {
        assert_eq!(Driver::from_name("amdgpu", VENDOR_AMD), Driver::Amdgpu);
        assert_eq!(Driver::from_name("xe", VENDOR_INTEL), Driver::Xe);
        assert_eq!(Driver::from_name("i915", VENDOR_INTEL), Driver::I915);
        assert_eq!(Driver::from_name("nvidia", VENDOR_NVIDIA), Driver::Nvidia);
        // NVML works for the card no matter which kernel module is bound.
        assert_eq!(Driver::from_name("vfio-pci", VENDOR_NVIDIA), Driver::Nvidia);
        assert_eq!(Driver::from_name("nouveau", VENDOR_AMD), Driver::Other);
    }

    #[test]
    fn free_saturates() {
        let u = VramUsage {
            total: 10,
            used: 12,
        };
        assert_eq!(u.free(), 0);
    }
}
