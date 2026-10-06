//! Local whisper.cpp transcription backend via `whisper-rs`.
//!
//! Streaming uses silence-delimited phrase segmentation by default
//! (`segmentation = "silence"`): incoming audio is split into phrases at
//! natural pauses (>= `phrase_silence_ms` of silence) and each phrase is
//! decoded exactly once — no overlapping windows, no deduplication, and no
//! feedback of prior output. Feeding each window's transcription back as
//! `initial_prompt` was the root cause of the repetition/invented-text loops
//! in issue #55, so the only prompt ever passed to the decoder is the static
//! vocabulary prompt from the user's config.
//!
//! Pause-free speech is force-cut by the splitter before it outgrows
//! whisper's 30 s context window (soft cap at a genuinely silent frame, hard
//! ceiling regardless); the phrase after such a cut is decoded on its own,
//! and its spurious leading capital — whisper capitalizes each decode
//! independently — is lowercased (see
//! [`Phrase::continuation`](super::phrase_split::Phrase::continuation)).
//!
//! The legacy sliding-window mode (`segmentation = "window"`) keeps the
//! 8s/2s overlapping windows with text-based n-gram overlap removal
//! (`asr_dedup::TextDedup`) for lower-latency partial output.
//!
//! `segmentation = "none"` turns streaming off: [`supports_streaming`]
//! returns `false`, so the daemon records the whole session and hands it to
//! [`transcribe`] in one call when recording stops. Nothing is typed while
//! you talk, and the batch-only dictation features (`[general]
//! llm_post_process`, `[input] paste`) apply.
//!
//! Both streaming modes create one decoder state per stream and reuse it
//! across decodes; see [`create_state`].
//!
//! A background thread can drop the model (and its GPU buffers) once no
//! transcription has run for `idle_unload_secs`, or when the GPU it lives on
//! runs below `gpu_min_free_mb` of free VRAM; the next dictation reloads it
//! from disk. See [`ModelSlot`] and [`UnloadPolicy`].
//!
//! [`supports_streaming`]: TranscriptionBackend::supports_streaming
//! [`transcribe`]: TranscriptionBackend::transcribe

use std::sync::{Arc, Mutex, Weak};
use std::time::{Duration, Instant};

use asr_dedup::TextDedup;
use async_trait::async_trait;
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

use super::phrase_split::PhraseSplitter;
use super::whisper_gpu::{self, GpuSelection};
use super::{ModelUnload, TranscriptionBackend, TranscriptionConfig};
use crate::audio::AudioChunk;
use crate::gpu_memory;

/// Sliding window parameters for the legacy "window" streaming mode.
const WINDOW_SECS: usize = 8;
const STEP_SECS: usize = 2;
const SAMPLE_RATE: usize = 16_000;
const WINDOW_SAMPLES: usize = WINDOW_SECS * SAMPLE_RATE;
const STEP_SAMPLES: usize = STEP_SECS * SAMPLE_RATE;
/// Shorter initial window for faster first result.
const INITIAL_WINDOW_SECS: usize = 4;
const INITIAL_WINDOW_SAMPLES: usize = INITIAL_WINDOW_SECS * SAMPLE_RATE;

/// Silence threshold — must match or be below the daemon's auto-stop threshold
/// (0.003) so we never skip audio that auto-stop considers speech.
const SILENCE_THRESHOLD: f64 = 0.003;

/// whisper.cpp produces no output for buffers shorter than ~1 s of audio;
/// short phrases are zero-padded up to this length before decoding.
const MIN_DECODE_SAMPLES: usize = SAMPLE_RATE + SAMPLE_RATE / 10; // 1.1 s

/// How streaming audio is segmented before decoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SegmentationMode {
    /// Split at natural pauses; decode each phrase exactly once (default).
    #[default]
    Silence,
    /// Legacy overlapping sliding window with text-based dedup.
    Window,
    /// No segmentation, no streaming: decode the whole recording once,
    /// after it stops (config value `"none"`).
    Unsegmented,
}

impl SegmentationMode {
    /// Parse a config string; unknown values warn and fall back to `Silence`.
    pub fn parse(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "silence" => Self::Silence,
            "window" => Self::Window,
            "none" => Self::Unsegmented,
            other => {
                warn!("unknown local-whisper segmentation {other:?}, using \"silence\"");
                Self::Silence
            }
        }
    }
}

/// The loaded model, shared between the backend and its idle reaper.
///
/// Users take a [`ModelLease`], which clones the context `Arc`; unloading
/// only drops the slot's own reference, so a decode in flight keeps its
/// context alive and the GPU memory is freed when the last lease ends.
struct ModelSlot {
    path: String,
    gpu: GpuSelection,
    state: Mutex<SlotState>,
}

struct SlotState {
    ctx: Option<Arc<whisper_rs::WhisperContext>>,
    /// Leases currently held. The reaper never unloads while this is > 0.
    active: usize,
    /// When the last lease ended (or the model was loaded).
    last_used: Instant,
}

impl ModelSlot {
    fn new(path: String, gpu: GpuSelection, ctx: Option<whisper_rs::WhisperContext>) -> Self {
        Self {
            path,
            gpu,
            state: Mutex::new(SlotState {
                ctx: ctx.map(Arc::new),
                active: 0,
                last_used: Instant::now(),
            }),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, SlotState> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Take a lease on the model, loading it first if it was unloaded.
    ///
    /// Blocking: a reload reads the whole model file. Call it off the async
    /// runtime (see [`LocalWhisperBackend::lease`]).
    fn acquire(self: &Arc<Self>) -> anyhow::Result<ModelLease> {
        let mut state = self.lock();
        let ctx = match &state.ctx {
            Some(ctx) => Arc::clone(ctx),
            None => {
                // Holding the lock across the load makes a concurrent
                // acquire wait for this load instead of starting a second one.
                let ctx = Arc::new(
                    LocalWhisperBackend::load_model(&self.path, &self.gpu).map_err(|e| {
                        anyhow::anyhow!("whisper model not loaded from {}: {e}", self.path)
                    })?,
                );
                info!("reloaded whisper model from {}", self.path);
                state.ctx = Some(Arc::clone(&ctx));
                ctx
            }
        };
        state.active += 1;
        Ok(ModelLease {
            ctx,
            slot: Arc::clone(self),
        })
    }

    /// Whether the model is loaded and nobody has used it for `timeout`.
    fn idle_for(&self, timeout: Duration) -> bool {
        let state = self.lock();
        state.ctx.is_some() && state.active == 0 && state.last_used.elapsed() >= timeout
    }

    /// Drop the model now. A lease in flight keeps its context until done.
    fn unload(&self) -> bool {
        self.lock().ctx.take().is_some()
    }

    /// Drop the model if it has been idle for at least `timeout`.
    fn unload_if_idle(&self, timeout: Duration) -> bool {
        let mut state = self.lock();
        if state.ctx.is_some() && state.active == 0 && state.last_used.elapsed() >= timeout {
            state.ctx = None;
            true
        } else {
            false
        }
    }
}

/// A held reference to the loaded model; marks the slot as in use until
/// dropped.
struct ModelLease {
    ctx: Arc<whisper_rs::WhisperContext>,
    slot: Arc<ModelSlot>,
}

impl Drop for ModelLease {
    fn drop(&mut self) {
        let mut state = self.slot.lock();
        state.active -= 1;
        state.last_used = Instant::now();
    }
}

/// When the reaper frees the model. Each rule is off when `None`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct UnloadPolicy {
    /// Unload after this long without a transcription.
    pub idle: Option<Duration>,
    /// Unload once the GPU holding the model has less free VRAM than this
    /// many bytes (and the model has sat unused for [`PRESSURE_GRACE`]).
    pub gpu_min_free: Option<u64>,
}

impl UnloadPolicy {
    /// From the `[local-whisper]` keys; `0` turns a rule off.
    pub fn from_config(idle_unload_secs: u64, gpu_min_free_mb: u64) -> Self {
        Self {
            idle: (idle_unload_secs > 0).then(|| Duration::from_secs(idle_unload_secs)),
            gpu_min_free: (gpu_min_free_mb > 0).then_some(gpu_min_free_mb.saturating_mul(1 << 20)),
        }
    }

    fn is_off(&self) -> bool {
        self.idle.is_none() && self.gpu_min_free.is_none()
    }

    /// How often the reaper wakes.
    fn tick(&self) -> Duration {
        let idle_tick = self
            .idle
            .map(|t| (t / 4).clamp(Duration::from_secs(1), Duration::from_secs(30)));
        let pressure_tick = self.gpu_min_free.map(|_| PRESSURE_POLL);
        match (idle_tick, pressure_tick) {
            (Some(a), Some(b)) => a.min(b),
            (a, b) => a.or(b).unwrap_or(Duration::from_secs(30)),
        }
    }
}

/// How often VRAM is sampled when `gpu_min_free_mb` is set.
const PRESSURE_POLL: Duration = Duration::from_secs(5);
/// Memory pressure never unloads a model used within this window, so
/// back-to-back dictations do not reload it each time.
const PRESSURE_GRACE: Duration = Duration::from_secs(10);

/// The GPU with the least free VRAM below `min_free` among those this
/// process holds memory on. Falls back to every readable GPU when ownership
/// cannot be told (e.g. a driver without fdinfo memory stats).
fn gpu_under_pressure(
    min_free: u64,
    selected: Option<&str>,
) -> Option<(gpu_memory::GpuDevice, gpu_memory::VramUsage)> {
    // The GPU whisper.cpp was told to use is authoritative; fdinfo is the
    // fallback when the backend reports no PCI address.
    let own = match selected {
        Some(addr) => vec![addr.to_string()],
        None => gpu_memory::own_devices(),
    };
    gpu_memory::devices()
        .into_iter()
        .filter(|d| own.is_empty() || own.contains(&d.pci_addr))
        .filter_map(|d| gpu_memory::usage(&d).map(|u| (d, u)))
        .filter(|(_, u)| u.free() < min_free)
        .min_by_key(|(_, u)| u.free())
}

/// Why the VRAM-pressure rule cannot apply to `gpu`, if it cannot.
///
/// An integrated GPU's "VRAM" is a small carve-out of system RAM that the
/// driver overflows into GTT, so it reads as nearly full whenever a model is
/// loaded; acting on it would unload after every dictation.
fn pressure_exempt(gpu: &GpuSelection) -> Option<&'static str> {
    if !gpu.use_gpu {
        Some("the model runs on the CPU")
    } else if gpu.gpu.as_ref().is_some_and(|g| g.integrated) {
        Some("the model runs on an integrated GPU, which shares system RAM")
    } else {
        None
    }
}

/// Log what the pressure rule can see, so an unsupported GPU is visible in
/// the journal instead of silently never triggering.
fn log_gpu_readers() {
    let devices = gpu_memory::devices();
    if devices.is_empty() {
        warn!("gpu_min_free_mb is set but no GPU was found in /sys/class/drm");
        return;
    }
    for d in &devices {
        match gpu_memory::usage(d) {
            Some(u) => info!(
                "GPU {} ({}, {:?}): {} MiB VRAM, {} MiB free",
                d.pci_addr,
                d.vendor_name(),
                d.driver,
                u.total >> 20,
                u.free() >> 20
            ),
            None => info!(
                "GPU {} ({}, {:?}): VRAM usage not readable; ignored for gpu_min_free_mb",
                d.pci_addr,
                d.vendor_name(),
                d.driver
            ),
        }
    }
}

/// Poll the slot and unload the model when `policy` says so. Exits once the
/// backend (the only strong owner besides leases) is gone.
fn spawn_reaper(slot: Weak<ModelSlot>, policy: UnloadPolicy) {
    let tick = policy.tick();
    let spawned = std::thread::Builder::new()
        .name("whisper-unload".into())
        .spawn(move || loop {
            std::thread::sleep(tick);
            let Some(slot) = slot.upgrade() else { break };
            if let Some(timeout) = policy.idle {
                if slot.unload_if_idle(timeout) {
                    info!(
                        "unloaded whisper model after {}s idle; next dictation reloads it",
                        timeout.as_secs()
                    );
                    continue;
                }
            }
            let Some(min_free) = policy.gpu_min_free else {
                continue;
            };
            // Cheap check first: sampling VRAM walks sysfs and /proc.
            if !slot.idle_for(PRESSURE_GRACE) {
                continue;
            }
            let selected = slot.gpu.gpu.as_ref().and_then(|g| g.pci_addr.as_deref());
            let Some((dev, usage)) = gpu_under_pressure(min_free, selected) else {
                continue;
            };
            if slot.unload_if_idle(PRESSURE_GRACE) {
                info!(
                    "unloaded whisper model: GPU {} ({}) has {} MiB of {} MiB free, \
                     below gpu_min_free_mb = {}; next dictation reloads it",
                    dev.pci_addr,
                    dev.vendor_name(),
                    usage.free() >> 20,
                    usage.total >> 20,
                    min_free >> 20
                );
            }
        });
    if let Err(e) = spawned {
        warn!("failed to start whisper unload thread: {e}");
    }
}

/// Local whisper.cpp transcription backend.
pub struct LocalWhisperBackend {
    slot: Arc<ModelSlot>,
    segmentation: SegmentationMode,
    phrase_silence_ms: u64,
}

impl LocalWhisperBackend {
    /// Create a new local whisper backend, eagerly loading the model.
    ///
    /// Defaults to silence-delimited phrase segmentation; see
    /// [`Self::with_segmentation`].
    pub fn new(model_path: String) -> Self {
        Self::with_gpu(model_path, None)
    }

    /// Like [`Self::new`], on the GPU the `[local-whisper] gpu` setting
    /// names (see [`whisper_gpu`]).
    pub fn with_gpu(model_path: String, gpu: Option<&crate::GpuSelect>) -> Self {
        let gpu = whisper_gpu::resolve(gpu);
        let ctx = match Self::load_model(&model_path, &gpu) {
            Ok(ctx) => {
                info!("loaded whisper model from {model_path}");
                Some(ctx)
            }
            Err(e) => {
                warn!("failed to load whisper model from {model_path}: {e}");
                None
            }
        };
        Self {
            slot: Arc::new(ModelSlot::new(model_path, gpu, ctx)),
            segmentation: SegmentationMode::default(),
            // Same default as `crate::LocalWhisperConfig` — lib.rs is the
            // single source of truth for segmentation defaults.
            phrase_silence_ms: crate::default_phrase_silence_ms(),
        }
    }

    /// Configure the streaming segmentation strategy.
    ///
    /// `mode` is `"silence"` (default), `"window"` or `"none"`;
    /// `phrase_silence_ms` is how much continuous silence ends a phrase in
    /// silence mode.
    pub fn with_segmentation(mut self, mode: &str, phrase_silence_ms: u64) -> Self {
        self.segmentation = SegmentationMode::parse(mode);
        self.phrase_silence_ms = phrase_silence_ms.max(1);
        self
    }

    /// Free the model from memory per `policy`; the next transcription
    /// reloads it. The default policy keeps it loaded for good.
    pub fn with_unload_policy(self, mut policy: UnloadPolicy) -> Self {
        if policy.gpu_min_free.is_some() {
            if let Some(why) = pressure_exempt(&self.slot.gpu) {
                info!("gpu_min_free_mb ignored: {why}");
                policy.gpu_min_free = None;
            } else {
                log_gpu_readers();
            }
        }
        if !policy.is_off() {
            spawn_reaper(Arc::downgrade(&self.slot), policy);
        }
        self
    }

    /// Lease the model, reloading it off the async runtime if it was
    /// unloaded for idleness.
    async fn lease(&self) -> anyhow::Result<ModelLease> {
        let slot = Arc::clone(&self.slot);
        tokio::task::spawn_blocking(move || slot.acquire()).await?
    }

    fn load_model(path: &str, gpu: &GpuSelection) -> anyhow::Result<whisper_rs::WhisperContext> {
        if !std::path::Path::new(path).exists() {
            anyhow::bail!("model file not found: {path}. Run 'whisrs setup' to download a model.");
        }

        let mut params = whisper_rs::WhisperContextParameters::default();
        // `use_gpu` defaults to whether this build has a GPU backend; only
        // ever turn it off, never on for a CPU-only build.
        params.use_gpu = params.use_gpu && gpu.use_gpu;
        params.gpu_device = gpu.device;

        whisper_rs::WhisperContext::new_with_params(path, params)
            .map_err(|e| anyhow::anyhow!("failed to initialize whisper context: {e}"))
    }

    /// Silence-delimited streaming: split audio into phrases at pauses and
    /// decode each phrase exactly once, in FIFO order.
    async fn stream_phrases(
        &self,
        ctx: &Arc<whisper_rs::WhisperContext>,
        mut audio_rx: mpsc::Receiver<AudioChunk>,
        text_tx: mpsc::Sender<String>,
        config: &TranscriptionConfig,
    ) -> anyhow::Result<()> {
        let mut splitter =
            PhraseSplitter::new(SAMPLE_RATE, SILENCE_THRESHOLD, self.phrase_silence_ms);
        let mut state = create_state(ctx)?;

        while let Some(chunk) = audio_rx.recv().await {
            for phrase in splitter.feed(&chunk) {
                state = decode_phrase(
                    ctx,
                    state,
                    phrase.samples,
                    phrase.continuation,
                    config,
                    &text_tx,
                )
                .await?;
            }
        }

        // End of stream: flush the trailing in-progress phrase (if any),
        // exactly once. No rewind, no overlap re-send.
        if let Some(phrase) = splitter.flush() {
            decode_phrase(
                ctx,
                state,
                phrase.samples,
                phrase.continuation,
                config,
                &text_tx,
            )
            .await?;
        }

        Ok(())
    }

    /// Legacy sliding-window streaming with text-based overlap removal.
    async fn stream_windows(
        &self,
        ctx: &Arc<whisper_rs::WhisperContext>,
        mut audio_rx: mpsc::Receiver<AudioChunk>,
        text_tx: mpsc::Sender<String>,
        config: &TranscriptionConfig,
    ) -> anyhow::Result<()> {
        let mut buffer: Vec<i16> = Vec::new();
        let mut dedup = TextDedup::new();
        let mut next_process_at = INITIAL_WINDOW_SAMPLES;
        let mut last_processed_end: usize = 0;
        // Static vocabulary prompt only — never prior transcription output.
        let prompt = config.prompt.clone().filter(|p| !p.is_empty());
        let mut state = create_state(ctx)?;

        while let Some(chunk) = audio_rx.recv().await {
            buffer.extend_from_slice(&chunk);

            while buffer.len() >= next_process_at {
                let window_size = if last_processed_end == 0 {
                    INITIAL_WINDOW_SAMPLES.min(buffer.len())
                } else {
                    WINDOW_SAMPLES.min(buffer.len())
                };

                let window_end = next_process_at;
                let window_start = window_end.saturating_sub(window_size);
                let window = buffer[window_start..window_end].to_vec();

                // Skip silent windows.
                if !audio_silence_gate::is_silent(&window, SILENCE_THRESHOLD) {
                    let samples_f32 = i16_to_f32(&window);
                    let lang = config.language.clone();
                    let prompt = prompt.clone();
                    let mut decode_state = state;

                    match tokio::task::spawn_blocking(move || {
                        let result = run_whisper_inference(
                            &mut decode_state,
                            &samples_f32,
                            &lang,
                            prompt.as_deref(),
                        );
                        (decode_state, result)
                    })
                    .await
                    {
                        Ok((returned, Ok(full_text))) => {
                            state = returned;
                            // Use text-based dedup to extract only the new portion.
                            let new_text = dedup.filter_text(&full_text);
                            if !new_text.trim().is_empty() {
                                debug!("streaming window produced: {:?}", new_text);
                                text_tx.send(new_text).await.ok();
                            }
                        }
                        Ok((stale, Err(e))) => {
                            warn!("whisper window inference failed: {e}");
                            // Free the failed state before allocating its
                            // replacement, so recovery never needs two states
                            // resident at once.
                            drop(stale);
                            state = create_state(ctx)?;
                        }
                        Err(e) => {
                            warn!("whisper task panicked: {e}");
                            // The state died with the panicked task; make a
                            // fresh one so later windows can still decode.
                            state = create_state(ctx)?;
                        }
                    }
                } else {
                    debug!(
                        "skipping silent window at samples {}..{}",
                        window_start, window_end
                    );
                }

                last_processed_end = window_end;
                next_process_at += STEP_SAMPLES;
            }
        }

        // Process remaining audio not covered by the last window.
        if buffer.len() > last_processed_end {
            let remaining_start = if buffer.len() - last_processed_end < SAMPLE_RATE {
                last_processed_end.saturating_sub(WINDOW_SAMPLES / 4)
            } else {
                last_processed_end
            };
            let remaining = &buffer[remaining_start..];

            if !remaining.is_empty() && !audio_silence_gate::is_silent(remaining, SILENCE_THRESHOLD)
            {
                let samples_f32 = i16_to_f32(remaining);
                let lang = config.language.clone();
                let prompt = prompt.clone();
                // Final decode of the stream — the state is consumed.
                let mut decode_state = state;

                match tokio::task::spawn_blocking(move || {
                    let result = run_whisper_inference(
                        &mut decode_state,
                        &samples_f32,
                        &lang,
                        prompt.as_deref(),
                    );
                    (decode_state, result)
                })
                .await
                {
                    Ok((_, Ok(full_text))) => {
                        let new_text = dedup.filter_text(&full_text);
                        if !new_text.trim().is_empty() {
                            text_tx.send(new_text).await.ok();
                        }
                    }
                    Ok((_, Err(e))) => warn!("whisper final inference failed: {e}"),
                    Err(e) => warn!("whisper final task panicked: {e}"),
                }
            }
        }

        Ok(())
    }

    /// Unsegmented mode over a stream: buffer everything, decode once at
    /// the end.
    ///
    /// The daemon never gets here, since this mode reports
    /// `supports_streaming() == false` and dictation takes the batch path
    /// instead. It exists so a direct `transcribe_stream` caller still gets
    /// batch semantics (one decode, one send) rather than a silent fallback
    /// to phrase streaming.
    async fn stream_unsegmented(
        &self,
        ctx: &Arc<whisper_rs::WhisperContext>,
        mut audio_rx: mpsc::Receiver<AudioChunk>,
        text_tx: mpsc::Sender<String>,
        config: &TranscriptionConfig,
    ) -> anyhow::Result<()> {
        let mut buffer: Vec<i16> = Vec::new();
        while let Some(chunk) = audio_rx.recv().await {
            buffer.extend_from_slice(&chunk);
        }

        if buffer.is_empty() || audio_silence_gate::is_silent(&buffer, SILENCE_THRESHOLD) {
            return Ok(());
        }

        decode_phrase(ctx, create_state(ctx)?, buffer, false, config, &text_tx).await?;
        Ok(())
    }
}

/// Convert i16 PCM samples to f32 in the range [-1.0, 1.0].
#[cfg(any(feature = "local-whisper", test))]
fn i16_to_f32(samples: &[i16]) -> Vec<f32> {
    samples
        .iter()
        .map(|&s| s as f32 / i16::MAX as f32)
        .collect()
}

/// Lowercase the first character of `text` if it is an uppercase letter.
///
/// Applied to continuation phrases (those starting at a forced mid-speech
/// cut): whisper capitalizes each decode independently, so a continuation's
/// leading capital is spurious mid-sentence. Char-safe for UTF-8;
/// `char::to_lowercase` may expand to multiple chars.
#[cfg(any(feature = "local-whisper", test))]
fn decapitalize_first(text: &str) -> String {
    let mut chars = text.chars();
    match chars.next() {
        // The standalone pronoun "I" (and contractions: "I'm", "I'll") is
        // correctly capitalized mid-sentence; lowercasing it is visibly wrong.
        Some('I') if !chars.clone().next().is_some_and(|c| c.is_alphabetic()) => text.to_string(),
        Some(c) if c.is_alphabetic() && c.is_uppercase() => {
            let mut out: String = c.to_lowercase().collect();
            out.push_str(chars.as_str());
            out
        }
        _ => text.to_string(),
    }
}

/// Create a decoder state for a stream (or a single batch call).
///
/// Creating a state allocates all of whisper.cpp's KV and compute buffers —
/// on GPU backends that is a full device buffer build — so it must happen
/// once per stream and be reused across decodes, never once per phrase.
/// Reuse carries no decoder context between phrases: every decode sets
/// `set_no_context(true)` (see [`run_whisper_inference`]).
fn create_state(ctx: &whisper_rs::WhisperContext) -> anyhow::Result<whisper_rs::WhisperState> {
    ctx.create_state()
        .map_err(|e| anyhow::anyhow!("failed to create whisper state: {e}"))
}

/// Decode a single phrase and send its trimmed text over `text_tx`.
///
/// Each phrase is decoded exactly once with only the static config prompt.
/// `continuation` marks a phrase starting at a forced mid-speech cut; its
/// spurious leading capital is lowercased before sending.
/// The stream's reusable `state` is taken by value (it has to move into the
/// blocking task) and handed back on return; if the decode fails or the task
/// panics, that state is discarded and a fresh one is created to keep the
/// stream going.
async fn decode_phrase(
    ctx: &Arc<whisper_rs::WhisperContext>,
    mut state: whisper_rs::WhisperState,
    mut phrase: Vec<i16>,
    continuation: bool,
    config: &TranscriptionConfig,
    text_tx: &mpsc::Sender<String>,
) -> anyhow::Result<whisper_rs::WhisperState> {
    debug!(
        "decoding phrase of {:.2}s",
        phrase.len() as f64 / SAMPLE_RATE as f64
    );
    // whisper.cpp silently produces nothing for <1 s buffers; pad with silence.
    if phrase.len() < MIN_DECODE_SAMPLES {
        phrase.resize(MIN_DECODE_SAMPLES, 0);
    }

    let samples_f32 = i16_to_f32(&phrase);
    let lang = config.language.clone();
    let prompt = config.prompt.clone().filter(|p| !p.is_empty());

    match tokio::task::spawn_blocking(move || {
        let result = run_whisper_inference(&mut state, &samples_f32, &lang, prompt.as_deref());
        (state, result)
    })
    .await
    {
        Ok((state, Ok(text))) => {
            let text = text.trim();
            if !text.is_empty() {
                let text = if continuation {
                    decapitalize_first(text)
                } else {
                    text.to_string()
                };
                debug!("phrase produced: {:?}", text);
                text_tx.send(text).await.ok();
            }
            Ok(state)
        }
        Ok((stale, Err(e))) => {
            warn!("whisper phrase inference failed: {e}");
            // Free the failed state before allocating its replacement, so
            // recovery never needs two states resident at once.
            drop(stale);
            create_state(ctx)
        }
        Err(e) => {
            warn!("whisper phrase task panicked: {e}");
            create_state(ctx)
        }
    }
}

/// Run whisper inference on an audio buffer.
///
/// - `state`: a reusable decoder state (see [`create_state`]). Reuse carries
///   no context between calls: `set_no_context(true)` below decodes every
///   buffer independently.
/// - `prompt`: the static vocabulary/context hint from the user's config.
///   Never prior transcription output — feeding decoded text back as the
///   prompt caused the repetition loops of issue #55.
fn run_whisper_inference(
    state: &mut whisper_rs::WhisperState,
    audio: &[f32],
    language: &str,
    prompt: Option<&str>,
) -> anyhow::Result<String> {
    use whisper_rs::{FullParams, SamplingStrategy};

    // `best_of: 1` is load-bearing beyond decode quality: it keeps whisper.cpp's
    // decoder count at 1, which is the only reason the library never takes the
    // path where it frees the state itself and hands back an error. Raising it,
    // or switching to beam search, needs the error arms above to stop dropping
    // the returned state.
    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });

    if language != "auto" {
        params.set_language(Some(language));
    }

    // Static vocabulary/context hint (config `prompt` + `vocabulary`).
    if let Some(prompt) = prompt {
        if !prompt.is_empty() {
            params.set_initial_prompt(prompt);
        }
    }
    // Decode every buffer independently: carrying decoder context across
    // buffers amplified repetition/invented text (issue #55). The static
    // initial_prompt above is still applied.
    // Reusing one state across decodes depends on this staying true: it is what
    // drops the previous decode's text context (see `create_state`).
    params.set_no_context(true);
    // Fall back to a higher temperature sooner when the decoder gets stuck in
    // low-entropy (repetitive) output. whisper.cpp default is 2.4.
    params.set_entropy_thold(2.6);
    // Suppress non-speech tokens (e.g. "(silence)", "♪") on pause-heavy audio.
    params.set_suppress_nst(true);

    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);

    let n_threads = std::thread::available_parallelism()
        .map(|n| n.get() as i32)
        .unwrap_or(4)
        .min(8);
    params.set_n_threads(n_threads);

    state
        .full(params, audio)
        .map_err(|e| anyhow::anyhow!("whisper inference failed: {e}"))?;

    let mut text = String::new();
    for segment in state.as_iter() {
        text.push_str(&format!("{}", segment));
    }

    Ok(text.trim().to_string())
}

#[async_trait]
impl TranscriptionBackend for LocalWhisperBackend {
    async fn transcribe(
        &self,
        audio: &[u8],
        config: &TranscriptionConfig,
    ) -> anyhow::Result<String> {
        let lease = self.lease().await?;

        // Decode WAV to i16 samples, then convert to f32.
        let cursor = std::io::Cursor::new(audio);
        let reader = hound::WavReader::new(cursor)?;
        let samples_i16: Vec<i16> = reader.into_samples::<i16>().collect::<Result<_, _>>()?;
        let mut samples_f32 = vec![0.0f32; samples_i16.len()];
        whisper_rs::convert_integer_to_float_audio(&samples_i16, &mut samples_f32)
            .map_err(|e| anyhow::anyhow!("failed to convert audio: {e}"))?;

        let ctx = Arc::clone(&lease.ctx);
        let language = config.language.clone();
        let prompt = config.prompt.clone();

        tokio::task::spawn_blocking(move || {
            let mut state = create_state(&ctx)?;
            run_whisper_inference(&mut state, &samples_f32, &language, prompt.as_deref())
        })
        .await?
        // `lease` drops here, after the decode, so idle time counts from now.
    }

    async fn transcribe_stream(
        &self,
        audio_rx: mpsc::Receiver<AudioChunk>,
        text_tx: mpsc::Sender<String>,
        config: &TranscriptionConfig,
    ) -> anyhow::Result<()> {
        let lease = self.lease().await?;
        let ctx = &lease.ctx;

        match self.segmentation {
            SegmentationMode::Silence => self.stream_phrases(ctx, audio_rx, text_tx, config).await,
            SegmentationMode::Window => self.stream_windows(ctx, audio_rx, text_tx, config).await,
            SegmentationMode::Unsegmented => {
                self.stream_unsegmented(ctx, audio_rx, text_tx, config)
                    .await
            }
        }
    }

    // Unsegmented mode opts out so the daemon records the whole session and calls
    // `transcribe` once on stop, instead of typing phrases as they end.
    fn supports_streaming(&self) -> bool {
        self.segmentation != SegmentationMode::Unsegmented
    }

    fn unload_model(&self) -> ModelUnload {
        if self.slot.unload() {
            info!("unloaded whisper model on request; next dictation reloads it");
            ModelUnload::Unloaded
        } else {
            ModelUnload::NotLoaded
        }
    }

    // `run_whisper_inference` hands the prompt to `set_initial_prompt`, so
    // whisper.cpp really does decode conditioned on it.
    fn sends_prompt(&self, _config: &TranscriptionConfig) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn stub_returns_error() {
        let backend = LocalWhisperBackend::new("/nonexistent/model.bin".to_string());
        let config = TranscriptionConfig {
            language: "en".to_string(),
            model: "base.en".to_string(),
            prompt: None,
            keyterms: Vec::new(),
        };
        let err = backend.transcribe(&[1, 2, 3], &config).await.unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("not available")
                || msg.contains("not loaded")
                || msg.contains("not found"),
            "unexpected error: {msg}"
        );
    }

    #[test]
    fn i16_to_f32_conversion() {
        let samples = vec![0i16, i16::MAX, i16::MIN];
        let f32_samples = i16_to_f32(&samples);
        assert_eq!(f32_samples[0], 0.0);
        assert!((f32_samples[1] - 1.0).abs() < 0.001);
        assert!((f32_samples[2] + 1.0).abs() < 0.001);
    }

    #[test]
    fn decapitalize_first_lowercases_ascii_uppercase() {
        assert_eq!(decapitalize_first("Hello world"), "hello world");
    }

    #[test]
    fn decapitalize_first_leaves_lowercase_unchanged() {
        assert_eq!(decapitalize_first("hello World"), "hello World");
    }

    #[test]
    fn decapitalize_first_leaves_digit_and_punct_unchanged() {
        assert_eq!(decapitalize_first("42 things"), "42 things");
        assert_eq!(decapitalize_first("...And then"), "...And then");
    }

    #[test]
    fn decapitalize_first_handles_multibyte_uppercase() {
        assert_eq!(decapitalize_first("École normale"), "école normale");
    }

    #[test]
    fn decapitalize_first_handles_empty_string() {
        assert_eq!(decapitalize_first(""), "");
    }

    #[test]
    fn decapitalize_first_keeps_pronoun_i() {
        assert_eq!(decapitalize_first("I think so"), "I think so");
        assert_eq!(decapitalize_first("I'm here"), "I'm here");
        assert_eq!(decapitalize_first("I"), "I");
    }

    #[test]
    fn decapitalize_first_still_lowercases_i_words() {
        assert_eq!(decapitalize_first("It works"), "it works");
        assert_eq!(decapitalize_first("Instead of that"), "instead of that");
    }

    #[test]
    fn unload_policy_from_config() {
        assert!(UnloadPolicy::from_config(0, 0).is_off());
        let p = UnloadPolicy::from_config(300, 1024);
        assert_eq!(p.idle, Some(Duration::from_secs(300)));
        assert_eq!(p.gpu_min_free, Some(1 << 30));
        // Pressure polling wins over the slower idle tick.
        assert_eq!(p.tick(), PRESSURE_POLL);
        assert_eq!(
            UnloadPolicy::from_config(8, 0).tick(),
            Duration::from_secs(2)
        );
        assert_eq!(UnloadPolicy::from_config(0, 512).tick(), PRESSURE_POLL);
    }

    #[test]
    fn unload_on_request_reports_whether_anything_was_freed() {
        let backend = LocalWhisperBackend::new("/nonexistent/model.bin".to_string());
        assert_eq!(backend.unload_model(), ModelUnload::NotLoaded);
    }

    #[test]
    fn pressure_rule_skips_cpu_and_integrated_gpus() {
        let on = |integrated: bool| GpuSelection {
            use_gpu: true,
            device: 0,
            gpu: Some(whisper_gpu::WhisperGpu {
                index: 0,
                name: "Vulkan0".into(),
                description: String::new(),
                integrated,
                pci_addr: None,
            }),
        };
        assert!(pressure_exempt(&on(false)).is_none());
        assert!(pressure_exempt(&on(true)).is_some());
        let cpu = GpuSelection {
            use_gpu: false,
            device: 0,
            gpu: None,
        };
        assert!(pressure_exempt(&cpu).is_some());
    }

    #[test]
    fn unloaded_slot_is_never_idle() {
        let gpu = GpuSelection {
            use_gpu: false,
            device: 0,
            gpu: None,
        };
        let slot = ModelSlot::new("/nonexistent".into(), gpu, None);
        assert!(!slot.idle_for(Duration::ZERO));
        assert!(!slot.unload_if_idle(Duration::ZERO));
    }

    #[test]
    fn segmentation_mode_parsing() {
        assert_eq!(
            SegmentationMode::parse("silence"),
            SegmentationMode::Silence
        );
        assert_eq!(SegmentationMode::parse("Window"), SegmentationMode::Window);
        assert_eq!(
            SegmentationMode::parse(" window "),
            SegmentationMode::Window
        );
        assert_eq!(
            SegmentationMode::parse("none"),
            SegmentationMode::Unsegmented
        );
        assert_eq!(
            SegmentationMode::parse(" NONE "),
            SegmentationMode::Unsegmented
        );
        // Unknown values fall back to the default.
        assert_eq!(SegmentationMode::parse("bogus"), SegmentationMode::Silence);
        assert_eq!(SegmentationMode::default(), SegmentationMode::Silence);
    }

    #[test]
    fn no_segmentation_disables_streaming() {
        // A missing model only logs a warning, which is all this test needs:
        // `supports_streaming` must not depend on the model being loaded.
        let backend = |mode: &str| {
            LocalWhisperBackend::new("/nonexistent/ggml.bin".to_string())
                .with_segmentation(mode, 400)
        };
        assert!(!backend("none").supports_streaming());
        assert!(backend("silence").supports_streaming());
        assert!(backend("window").supports_streaming());
        assert!(backend("bogus").supports_streaming());
    }
}
