//! Pick the GPU whisper.cpp runs on (`[local-whisper] gpu`).
//!
//! whisper.cpp takes a GPU *index*: the n-th device of type GPU or IGPU in
//! ggml's backend registry, which is the `Vulkan0`, `Vulkan1`, … numbering
//! in the daemon log. This module turns the user's choice (an index, a kind,
//! a PCI address, or "cpu") into that index by listing the same registry.

use std::ffi::{c_char, CStr};

use tracing::{info, warn};
use whisper_rs::whisper_rs_sys as sys;

/// The parsed `gpu` setting.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum GpuChoice {
    /// whisper.cpp's default: the first GPU it lists.
    #[default]
    Auto,
    /// The n-th GPU (`Vulkan<n>`).
    Index(usize),
    /// The first discrete GPU.
    Dedicated,
    /// The first integrated GPU.
    Integrated,
    /// The GPU at this PCI address, e.g. `0000:07:00.0`.
    Pci(String),
    /// No GPU: decode on the CPU.
    Cpu,
}

impl GpuChoice {
    /// Parse a config value. `None` for anything unrecognised.
    pub fn parse(value: &crate::GpuSelect) -> Option<Self> {
        match value {
            crate::GpuSelect::Index(n) => Some(Self::Index(*n as usize)),
            crate::GpuSelect::Name(s) => {
                let s = s.trim().to_ascii_lowercase();
                match s.as_str() {
                    "" | "auto" => Some(Self::Auto),
                    "dedicated" | "discrete" | "dgpu" => Some(Self::Dedicated),
                    "integrated" | "igpu" => Some(Self::Integrated),
                    "cpu" | "none" | "off" => Some(Self::Cpu),
                    _ if s.bytes().all(|b| b.is_ascii_digit()) => s.parse().ok().map(Self::Index),
                    _ if is_pci_addr(&s) => Some(Self::Pci(s)),
                    _ => None,
                }
            }
        }
    }
}

/// `dddd:bb:dd.f` (the domain is optional).
fn is_pci_addr(s: &str) -> bool {
    let hex = |p: &str, n: usize| p.len() == n && p.bytes().all(|b| b.is_ascii_hexdigit());
    let (rest, func) = match s.rsplit_once('.') {
        Some(x) => x,
        None => return false,
    };
    if !(func.len() == 1 && func.bytes().all(|b| b.is_ascii_digit())) {
        return false;
    }
    match rest.split(':').collect::<Vec<_>>().as_slice() {
        [dom, bus, dev] => hex(dom, 4) && hex(bus, 2) && hex(dev, 2),
        [bus, dev] => hex(bus, 2) && hex(dev, 2),
        _ => false,
    }
}

/// One GPU as whisper.cpp sees it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WhisperGpu {
    /// Position among GPUs: the value whisper.cpp's `gpu_device` takes.
    pub index: usize,
    pub name: String,
    pub description: String,
    pub integrated: bool,
    /// PCI address, lowercase, when the backend reports one.
    pub pci_addr: Option<String>,
}

/// What [`resolve`] decided.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuSelection {
    pub use_gpu: bool,
    pub device: i32,
    /// The chosen GPU, if any GPU is used and it was found.
    pub gpu: Option<WhisperGpu>,
}

fn cstr(p: *const c_char) -> String {
    if p.is_null() {
        String::new()
    } else {
        // SAFETY: ggml returns NUL-terminated strings that live as long as
        // the (static) device registry.
        unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
    }
}

/// The GPUs in ggml's registry, in whisper.cpp's counting order.
pub fn list() -> Vec<WhisperGpu> {
    let mut out = Vec::new();
    // SAFETY: plain registry queries; the registry is initialized on first
    // use and devices live for the process.
    unsafe {
        for i in 0..sys::ggml_backend_dev_count() {
            let dev = sys::ggml_backend_dev_get(i);
            if dev.is_null() {
                continue;
            }
            let kind = sys::ggml_backend_dev_type(dev);
            let integrated = kind == sys::ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_IGPU;
            if kind != sys::ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_GPU && !integrated {
                continue;
            }
            let mut props: sys::ggml_backend_dev_props = std::mem::zeroed();
            sys::ggml_backend_dev_get_props(dev, &mut props);
            let pci = cstr(props.device_id).to_ascii_lowercase();
            out.push(WhisperGpu {
                index: out.len(),
                name: cstr(props.name),
                description: cstr(props.description),
                integrated,
                pci_addr: (!pci.is_empty()).then_some(pci),
            });
        }
    }
    out
}

/// Pick a GPU from `gpus` for `choice`. Falls back to whisper.cpp's default
/// (with a warning) when the requested GPU does not exist.
pub fn choose(choice: &GpuChoice, gpus: &[WhisperGpu]) -> GpuSelection {
    let auto = GpuSelection {
        use_gpu: true,
        device: 0,
        gpu: gpus.first().cloned(),
    };
    let found = match choice {
        GpuChoice::Cpu => {
            return GpuSelection {
                use_gpu: false,
                device: 0,
                gpu: None,
            }
        }
        GpuChoice::Auto => return auto,
        GpuChoice::Index(n) => gpus.get(*n),
        GpuChoice::Dedicated => gpus.iter().find(|g| !g.integrated),
        GpuChoice::Integrated => gpus.iter().find(|g| g.integrated),
        GpuChoice::Pci(addr) => gpus.iter().find(|g| {
            g.pci_addr
                .as_deref()
                .is_some_and(|p| p == addr || p.ends_with(&format!(":{addr}")))
        }),
    };
    match found {
        Some(g) => GpuSelection {
            use_gpu: true,
            device: g.index as i32,
            gpu: Some(g.clone()),
        },
        None => {
            if gpus.is_empty() {
                warn!("local-whisper gpu = {choice:?}: this build sees no GPU, using the CPU");
            } else {
                warn!("local-whisper gpu = {choice:?}: no such GPU, using the first one instead");
            }
            auto
        }
    }
}

/// Resolve the config value and log the result.
pub fn resolve(setting: Option<&crate::GpuSelect>) -> GpuSelection {
    let choice = match setting {
        None => GpuChoice::Auto,
        Some(v) => GpuChoice::parse(v).unwrap_or_else(|| {
            warn!("local-whisper gpu = {v:?} is not an index, \"dedicated\", \"integrated\", a PCI address or \"cpu\"; using auto");
            GpuChoice::Auto
        }),
    };
    if choice == GpuChoice::Cpu {
        info!("local-whisper: GPU disabled by config, decoding on the CPU");
        return choose(&choice, &[]);
    }
    let gpus = list();
    for g in &gpus {
        info!(
            "whisper GPU {}: {} ({}, {}{})",
            g.index,
            g.description,
            g.name,
            if g.integrated {
                "integrated"
            } else {
                "dedicated"
            },
            g.pci_addr
                .as_deref()
                .map(|p| format!(", {p}"))
                .unwrap_or_default()
        );
    }
    let sel = choose(&choice, &gpus);
    if let Some(g) = &sel.gpu {
        info!("whisper GPU selected: {} ({})", g.index, g.description);
    }
    sel
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GpuSelect;

    fn gpu(index: usize, integrated: bool, pci: &str) -> WhisperGpu {
        WhisperGpu {
            index,
            name: format!("Vulkan{index}"),
            description: String::new(),
            integrated,
            pci_addr: Some(pci.to_string()),
        }
    }

    #[test]
    fn parses_settings() {
        let name = |s: &str| GpuChoice::parse(&GpuSelect::Name(s.into()));
        assert_eq!(
            GpuChoice::parse(&GpuSelect::Index(1)),
            Some(GpuChoice::Index(1))
        );
        assert_eq!(name("2"), Some(GpuChoice::Index(2)));
        assert_eq!(name(" Auto "), Some(GpuChoice::Auto));
        assert_eq!(name("dedicated"), Some(GpuChoice::Dedicated));
        assert_eq!(name("discrete"), Some(GpuChoice::Dedicated));
        assert_eq!(name("iGPU"), Some(GpuChoice::Integrated));
        assert_eq!(name("cpu"), Some(GpuChoice::Cpu));
        assert_eq!(
            name("0000:07:00.0"),
            Some(GpuChoice::Pci("0000:07:00.0".into()))
        );
        assert_eq!(name("C9:00.0"), Some(GpuChoice::Pci("c9:00.0".into())));
        assert_eq!(name("fastest"), None);
        assert_eq!(name("07:00"), None);
    }

    #[test]
    fn chooses_by_kind_index_and_pci() {
        let gpus = [gpu(0, false, "0000:07:00.0"), gpu(1, true, "0000:c9:00.0")];
        assert_eq!(choose(&GpuChoice::Integrated, &gpus).device, 1);
        assert_eq!(choose(&GpuChoice::Dedicated, &gpus).device, 0);
        assert_eq!(choose(&GpuChoice::Index(1), &gpus).device, 1);
        assert_eq!(choose(&GpuChoice::Pci("c9:00.0".into()), &gpus).device, 1);
        assert_eq!(
            choose(&GpuChoice::Pci("0000:07:00.0".into()), &gpus).device,
            0
        );
        assert!(!choose(&GpuChoice::Cpu, &gpus).use_gpu);
    }

    #[test]
    fn missing_gpu_falls_back_to_first() {
        let gpus = [gpu(0, false, "0000:07:00.0")];
        let sel = choose(&GpuChoice::Integrated, &gpus);
        assert!(sel.use_gpu);
        assert_eq!(sel.device, 0);
        assert_eq!(choose(&GpuChoice::Index(5), &gpus).device, 0);
    }
}
