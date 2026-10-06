//! Intel discrete GPUs (Arc): memory-region queries on the render node.
//!
//! Struct layouts and ioctl numbers mirror `drm/xe_drm.h` and
//! `drm/i915_drm.h` from the kernel uapi headers.

use std::os::fd::AsRawFd;
use std::path::Path;

use super::VramUsage;

/// `DRM_IOWR(DRM_COMMAND_BASE + DRM_XE_DEVICE_QUERY, struct drm_xe_device_query)`
const DRM_IOCTL_XE_DEVICE_QUERY: u64 = 0xc028_6440;
const DRM_XE_DEVICE_QUERY_MEM_REGIONS: u32 = 1;
const DRM_XE_MEM_REGION_CLASS_VRAM: u16 = 1;

/// `DRM_IOWR(DRM_COMMAND_BASE + DRM_I915_QUERY, struct drm_i915_query)`
const DRM_IOCTL_I915_QUERY: u64 = 0xc010_6479;
const DRM_I915_QUERY_MEMORY_REGIONS: u64 = 4;
const I915_MEMORY_CLASS_DEVICE: u16 = 1;

#[repr(C)]
#[derive(Default)]
struct DrmXeDeviceQuery {
    extensions: u64,
    query: u32,
    size: u32,
    data: u64,
    reserved: [u64; 2],
}

#[repr(C)]
#[derive(Default)]
struct DrmI915Query {
    num_items: u32,
    flags: u32,
    items_ptr: u64,
}

#[repr(C)]
#[derive(Default)]
struct DrmI915QueryItem {
    query_id: u64,
    length: i32,
    flags: u32,
    data_ptr: u64,
}

/// Size of `struct drm_xe_mem_region` / `struct drm_i915_memory_region_info`.
const REGION_SIZE: usize = 88;

fn open(node: &Path) -> Option<std::fs::File> {
    std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(node)
        .ok()
}

/// # Safety
/// `arg` must point to the struct the ioctl `request` expects.
unsafe fn ioctl<T>(file: &std::fs::File, request: u64, arg: &mut T) -> bool {
    // SAFETY: forwarded from the caller.
    unsafe { libc::ioctl(file.as_raw_fd(), request as _, arg as *mut T) == 0 }
}

fn u16_at(buf: &[u8], off: usize) -> u16 {
    u16::from_ne_bytes(buf[off..off + 2].try_into().unwrap())
}

fn u32_at(buf: &[u8], off: usize) -> u32 {
    u32::from_ne_bytes(buf[off..off + 4].try_into().unwrap())
}

fn u64_at(buf: &[u8], off: usize) -> u64 {
    u64::from_ne_bytes(buf[off..off + 8].try_into().unwrap())
}

/// A zeroed, 8-byte-aligned buffer for the kernel to fill.
fn aligned_buf(len: usize) -> Vec<u64> {
    vec![0u64; len.div_ceil(8)]
}

fn as_bytes(buf: &[u64], len: usize) -> &[u8] {
    // SAFETY: u64 storage is valid as bytes; `len` is within the allocation.
    unsafe { std::slice::from_raw_parts(buf.as_ptr().cast::<u8>(), len) }
}

pub(super) fn xe_usage(node: &Path) -> Option<VramUsage> {
    let file = open(node)?;
    let mut q = DrmXeDeviceQuery {
        query: DRM_XE_DEVICE_QUERY_MEM_REGIONS,
        ..Default::default()
    };
    // First call with size 0 asks the kernel for the reply size.
    // SAFETY: `q` is a `drm_xe_device_query`.
    if !unsafe { ioctl(&file, DRM_IOCTL_XE_DEVICE_QUERY, &mut q) } || q.size == 0 {
        return None;
    }
    let len = q.size as usize;
    let mut buf = aligned_buf(len);
    q.data = buf.as_mut_ptr() as u64;
    // SAFETY: `q.data` points at `q.size` writable bytes.
    if !unsafe { ioctl(&file, DRM_IOCTL_XE_DEVICE_QUERY, &mut q) } {
        return None;
    }
    parse_xe_regions(as_bytes(&buf, len))
}

/// Sum the VRAM regions of a `struct drm_xe_query_mem_regions` reply.
fn parse_xe_regions(buf: &[u8]) -> Option<VramUsage> {
    if buf.len() < 8 {
        return None;
    }
    let count = u32_at(buf, 0) as usize;
    let mut total = 0u64;
    let mut used = 0u64;
    for i in 0..count {
        let off = 8 + i * REGION_SIZE;
        if off + REGION_SIZE > buf.len() {
            break;
        }
        if u16_at(buf, off) == DRM_XE_MEM_REGION_CLASS_VRAM {
            total += u64_at(buf, off + 8);
            used += u64_at(buf, off + 16);
        }
    }
    (total > 0).then_some(VramUsage { total, used })
}

pub(super) fn i915_usage(node: &Path) -> Option<VramUsage> {
    let file = open(node)?;
    let mut item = DrmI915QueryItem {
        query_id: DRM_I915_QUERY_MEMORY_REGIONS,
        ..Default::default()
    };
    let mut q = DrmI915Query {
        num_items: 1,
        items_ptr: &mut item as *mut DrmI915QueryItem as u64,
        ..Default::default()
    };
    // First call with length 0 asks the kernel for the reply size.
    // SAFETY: `q` is a `drm_i915_query` whose `items_ptr` points at one item.
    if !unsafe { ioctl(&file, DRM_IOCTL_I915_QUERY, &mut q) } || item.length <= 0 {
        return None;
    }
    let len = item.length as usize;
    let mut buf = aligned_buf(len);
    item.data_ptr = buf.as_mut_ptr() as u64;
    // SAFETY: as above; `data_ptr` points at `length` writable bytes.
    if !unsafe { ioctl(&file, DRM_IOCTL_I915_QUERY, &mut q) } || item.length <= 0 {
        return None;
    }
    parse_i915_regions(as_bytes(&buf, len))
}

/// Sum the device-local regions of a `struct drm_i915_query_memory_regions`
/// reply. i915 reports what is left (`unallocated_size`), not what is used.
fn parse_i915_regions(buf: &[u8]) -> Option<VramUsage> {
    if buf.len() < 16 {
        return None;
    }
    let count = u32_at(buf, 0) as usize;
    let mut total = 0u64;
    let mut used = 0u64;
    for i in 0..count {
        let off = 16 + i * REGION_SIZE;
        if off + REGION_SIZE > buf.len() {
            break;
        }
        if u16_at(buf, off) == I915_MEMORY_CLASS_DEVICE {
            let probed = u64_at(buf, off + 8);
            let unallocated = u64_at(buf, off + 16);
            total += probed;
            used += probed.saturating_sub(unallocated);
        }
    }
    (total > 0).then_some(VramUsage { total, used })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn region(class: u16, a: u64, b: u64) -> Vec<u8> {
        let mut r = vec![0u8; REGION_SIZE];
        r[0..2].copy_from_slice(&class.to_ne_bytes());
        r[8..16].copy_from_slice(&a.to_ne_bytes());
        r[16..24].copy_from_slice(&b.to_ne_bytes());
        r
    }

    #[test]
    fn xe_sums_only_vram_regions() {
        let mut buf = vec![0u8; 8];
        buf[0..4].copy_from_slice(&2u32.to_ne_bytes());
        buf.extend(region(0, 64 << 30, 1 << 30)); // sysmem: ignored
        buf.extend(region(DRM_XE_MEM_REGION_CLASS_VRAM, 12 << 30, 4 << 30));
        assert_eq!(
            parse_xe_regions(&buf),
            Some(VramUsage {
                total: 12 << 30,
                used: 4 << 30
            })
        );
    }

    #[test]
    fn xe_without_vram_is_none() {
        let mut buf = vec![0u8; 8];
        buf[0..4].copy_from_slice(&1u32.to_ne_bytes());
        buf.extend(region(0, 64 << 30, 1 << 30));
        assert_eq!(parse_xe_regions(&buf), None);
    }

    #[test]
    fn i915_turns_unallocated_into_used() {
        let mut buf = vec![0u8; 16];
        buf[0..4].copy_from_slice(&2u32.to_ne_bytes());
        buf.extend(region(0, 32 << 30, 30 << 30));
        buf.extend(region(I915_MEMORY_CLASS_DEVICE, 16 << 30, 10 << 30));
        assert_eq!(
            parse_i915_regions(&buf),
            Some(VramUsage {
                total: 16 << 30,
                used: 6 << 30
            })
        );
    }

    #[test]
    fn truncated_reply_does_not_panic() {
        let mut buf = vec![0u8; 8];
        buf[0..4].copy_from_slice(&5u32.to_ne_bytes());
        assert_eq!(parse_xe_regions(&buf), None);
    }
}
