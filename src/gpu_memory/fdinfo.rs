//! DRM client memory stats from `/proc/<pid>/fdinfo/<fd>`.
//!
//! Every open DRM file reports `drm-pdev`, `drm-client-id` and per-region
//! `drm-{total,resident}-<region>` keys (kernel `drm-usage-stats`). Device
//! memory regions are named `vram*` (amdgpu, xe) or `local*` (i915).

use std::collections::HashSet;
use std::path::Path;

/// Memory figures one DRM client file reports.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct DrmClient {
    pub pdev: String,
    pub client_id: Option<u64>,
    /// Device memory resident in VRAM, bytes.
    pub resident_vram: u64,
    /// Device memory allocated with VRAM placement, bytes.
    pub total_vram: u64,
}

fn is_device_region(region: &str) -> bool {
    region.starts_with("vram") || region.starts_with("local")
}

/// Parse `"19140 KiB"` / `"4096"` into bytes.
fn parse_size(value: &str) -> Option<u64> {
    let mut parts = value.split_whitespace();
    let n: u64 = parts.next()?.parse().ok()?;
    let mult = match parts.next() {
        None => 1,
        Some("KiB") => 1 << 10,
        Some("MiB") => 1 << 20,
        Some("GiB") => 1 << 30,
        Some(_) => return None,
    };
    Some(n * mult)
}

/// Parse one fdinfo file; `None` if it is not a DRM client.
pub fn parse(text: &str) -> Option<DrmClient> {
    let mut client = DrmClient::default();
    for line in text.lines() {
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        let value = value.trim();
        match key {
            "drm-pdev" => client.pdev = value.to_ascii_lowercase(),
            "drm-client-id" => client.client_id = value.parse().ok(),
            _ => {
                let (field, region) = if let Some(r) = key.strip_prefix("drm-resident-") {
                    (&mut client.resident_vram, r)
                } else if let Some(r) = key.strip_prefix("drm-total-") {
                    (&mut client.total_vram, r)
                } else if let Some(r) = key.strip_prefix("drm-memory-") {
                    // Pre-6.x amdgpu: one key per region, no total/resident split.
                    (&mut client.resident_vram, r)
                } else {
                    continue;
                };
                if is_device_region(region) {
                    *field += parse_size(value).unwrap_or(0);
                }
            }
        }
    }
    (!client.pdev.is_empty()).then_some(client)
}

fn clients_of(pid_dir: &Path) -> impl Iterator<Item = DrmClient> {
    std::fs::read_dir(pid_dir.join("fdinfo"))
        .into_iter()
        .flatten()
        .filter_map(Result::ok)
        .filter_map(|e| std::fs::read_to_string(e.path()).ok())
        .filter_map(|t| parse(&t))
}

/// PCI addresses this process holds VRAM on.
pub fn own_vram_devices() -> Vec<String> {
    let mut out = Vec::new();
    for c in clients_of(Path::new("/proc/self")) {
        if (c.total_vram > 0 || c.resident_vram > 0) && !out.contains(&c.pdev) {
            out.push(c.pdev);
        }
    }
    out
}

/// VRAM resident on `pdev`, summed over every DRM client this user can read.
///
/// Several fds can share one client (dup, fork), so each `drm-client-id`
/// (unique per device) is counted once.
pub fn system_vram_used(pdev: &str) -> u64 {
    let mut seen = HashSet::new();
    let mut used = 0;
    let Ok(procs) = std::fs::read_dir("/proc") else {
        return 0;
    };
    for entry in procs.filter_map(Result::ok) {
        if !entry
            .file_name()
            .to_string_lossy()
            .bytes()
            .all(|b| b.is_ascii_digit())
        {
            continue;
        }
        for c in clients_of(&entry.path()) {
            if c.pdev != pdev {
                continue;
            }
            if let Some(id) = c.client_id {
                if !seen.insert(id) {
                    continue;
                }
            }
            used += c.resident_vram;
        }
    }
    used
}

#[cfg(test)]
mod tests {
    use super::*;

    const XE: &str = "pos:\t0\nflags:\t02100002\ndrm-driver:\txe\ndrm-client-id:\t1215\n\
        drm-pdev:\t0000:07:00.0\ndrm-total-system:\t0\ndrm-total-gtt:\t8 KiB\n\
        drm-total-vram0:\t3100000 KiB\ndrm-resident-vram0:\t3000000 KiB\n\
        drm-total-stolen:\t0\ndrm-cycles-rcs:\t0\n";

    #[test]
    fn parses_xe_client() {
        let c = parse(XE).unwrap();
        assert_eq!(c.pdev, "0000:07:00.0");
        assert_eq!(c.client_id, Some(1215));
        assert_eq!(c.total_vram, 3_100_000 << 10);
        assert_eq!(c.resident_vram, 3_000_000 << 10);
    }

    #[test]
    fn parses_i915_local_and_old_amdgpu_keys() {
        let i915 =
            "drm-pdev:\t0000:03:00.0\ndrm-total-local0:\t2 GiB\ndrm-resident-local0:\t1 GiB\n\
            drm-total-system0:\t5 GiB\n";
        let c = parse(i915).unwrap();
        assert_eq!((c.total_vram, c.resident_vram), (2 << 30, 1 << 30));

        let amd = "drm-pdev:\t0000:C9:00.0\ndrm-memory-vram:\t512 MiB\ndrm-memory-gtt:\t1 GiB\n";
        let c = parse(amd).unwrap();
        assert_eq!(c.pdev, "0000:c9:00.0");
        assert_eq!(c.resident_vram, 512 << 20);
    }

    #[test]
    fn non_drm_file_is_none() {
        assert_eq!(parse("pos:\t0\nflags:\t0100000\nmnt_id:\t25\n"), None);
    }

    #[test]
    fn sizes() {
        assert_eq!(parse_size("0"), Some(0));
        assert_eq!(parse_size("3 KiB"), Some(3072));
        assert_eq!(parse_size("1 GiB"), Some(1 << 30));
        assert_eq!(parse_size("1 bogus"), None);
    }
}
