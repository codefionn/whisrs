//! AMD: `amdgpu` publishes VRAM totals in the PCI device's sysfs directory,
//! readable by any user.

use std::path::Path;

use super::VramUsage;

pub(super) fn usage(sysfs: &Path) -> Option<VramUsage> {
    let total = read_u64(&sysfs.join("mem_info_vram_total"))?;
    let used = read_u64(&sysfs.join("mem_info_vram_used"))?;
    (total > 0).then_some(VramUsage { total, used })
}

fn read_u64(path: &Path) -> Option<u64> {
    std::fs::read_to_string(path).ok()?.trim().parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_sysfs_files() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("mem_info_vram_total"), "17163091968\n").unwrap();
        std::fs::write(dir.path().join("mem_info_vram_used"), "1073741824\n").unwrap();
        assert_eq!(
            usage(dir.path()),
            Some(VramUsage {
                total: 17_163_091_968,
                used: 1_073_741_824
            })
        );
    }

    #[test]
    fn missing_files_mean_no_reading() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(usage(dir.path()), None);
    }
}
