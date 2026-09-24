//! Startup pack recording orchestration.
//!
//! After a snapshot is captured (and the source sandbox is running again),
//! boot one throwaway VM from the captured snapshot with a dedicated memory
//! device so the daemon-side recorder sees every first-touch read while all
//! memory layers are still node-local. The recorder emits a first-touch trace
//! file; the publisher expands it into the v3 startup manifest (exact-order
//! prefix plus merged ranges) and stores just that list. The whole flow is
//! best-effort: any failure returns `None` and the publish continues without
//! a manifest.

use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result};
use tracing::{debug, info, warn};
use uvm_ublk_daemon::protocol::PackRecordingState;

use super::{FirecrackerSandbox, FirecrackerSnapshotConfig};
use crate::cfg::ConfigManager;
use crate::sandbox::ublk::UblkDeviceManager;
use crate::snapshot::MEMORY_STARTUP_TRACE_ARTIFACT;

const UBLK_PREFETCH_CHUNK_BYTES: usize = 4 << 20;

/// Recording is repository-neutral. Each backend decides how to persist and
/// consume the resulting manifest.
fn recording_enabled_for(config: &crate::cfg::SnapshotConfig) -> bool {
    config.memory_startup_pack.enabled
}

fn recording_enabled() -> bool {
    recording_enabled_for(&ConfigManager::global_config().snapshot)
}

/// Synchronously warm the block-device page cache for the exact memory ublk
/// device Firecracker will mmap. This intentionally reads the full manifest
/// before restore; it is an experiment to distinguish final logical-page
/// cache effects from backing-file cache effects.
pub(super) async fn prefetch_ublk_startup_pages(
    device_path: &Path,
    pack: &crate::snapshot::ResolvedStartupPack,
) -> Result<()> {
    let crate::snapshot::StartupPackLocation::LocalPath(manifest_path) = &pack.location else {
        anyhow::bail!("ublk startup prefetch requires a local manifest");
    };
    let manifest_bytes = tokio::fs::read(manifest_path)
        .await
        .with_context(|| format!("read startup manifest {}", manifest_path.display()))?;
    anyhow::ensure!(
        manifest_bytes.len() as u64 == pack.pack_size,
        "startup manifest size mismatch: expected {}, got {}",
        pack.pack_size,
        manifest_bytes.len()
    );
    anyhow::ensure!(
        crate::snapshot::startup_pack::hex_sha256(&manifest_bytes) == pack.index_sha256,
        "startup manifest sha256 mismatch"
    );
    let manifest = overlaybd::startup_manifest::decode_manifest(&manifest_bytes)
        .context("decode startup manifest")?;
    anyhow::ensure!(
        manifest.mem_virtual_size == pack.mem_virtual_size,
        "startup manifest memory size mismatch"
    );

    let device_path = device_path.to_path_buf();
    let manifest_path = manifest_path.clone();
    tokio::task::spawn_blocking(move || -> Result<()> {
        use std::os::unix::fs::FileExt;

        let device = std::fs::File::open(&device_path)
            .with_context(|| format!("open memory ublk device {}", device_path.display()))?;
        let mut buffer = vec![0u8; UBLK_PREFETCH_CHUNK_BYTES];
        let mut bytes_read = 0u64;
        let mut read_count = 0u64;
        let spans = manifest
            .prefix_pages
            .into_iter()
            .map(|offset| (offset, overlaybd::startup_pack::PACK_PAGE_BYTES))
            .chain(manifest.ranges);
        for (start, len) in spans {
            let mut offset = start;
            let end = start.checked_add(len).context("startup range overflow")?;
            while offset < end {
                let chunk_len =
                    usize::try_from((end - offset).min(UBLK_PREFETCH_CHUNK_BYTES as u64))?;
                device
                    .read_exact_at(&mut buffer[..chunk_len], offset)
                    .with_context(|| {
                        format!(
                            "prefetch memory ublk {} at offset {offset:#x}",
                            device_path.display()
                        )
                    })?;
                offset += chunk_len as u64;
                bytes_read += chunk_len as u64;
                read_count += 1;
            }
        }
        info!(
            device_path = %device_path.display(),
            manifest_path = %manifest_path.display(),
            pages = bytes_read / overlaybd::startup_pack::PACK_PAGE_BYTES,
            bytes = bytes_read,
            reads = read_count,
            "memory ublk startup prefetch complete"
        );
        Ok(())
    })
    .await
    .context("memory ublk startup prefetch task failed to join")??;
    Ok(())
}

/// Record the first-touch trace for a just-captured snapshot.
///
/// Returns the trace path (`{snapshot_dir}/memory-startup.trace`) on success,
/// `None` on any failure or timeout. Cleanup (abort, VM stop, partial files)
/// always completes and is never bounded by the recording budget.
pub(crate) async fn record_startup_pack(
    mut config: FirecrackerSnapshotConfig,
    snapshot_dir: PathBuf,
) -> Option<PathBuf> {
    if !recording_enabled() {
        return None;
    }
    info!(dir = %snapshot_dir.display(), "startup pack recording started");
    let budget_secs = ConfigManager::global_config()
        .snapshot
        .memory_startup_pack
        .record_budget_secs;
    let trace_path = snapshot_dir.join(MEMORY_STARTUP_TRACE_ARTIFACT);

    let recording_config = match derive_recording_mem_config(
        &config.mem_overlaybd_config.image_config_path,
        &snapshot_dir,
    )
    .await
    {
        Ok(path) => path,
        Err(error) => {
            warn!(%error, "startup pack: derive recording mem config failed");
            return None;
        }
    };
    config.mem_overlaybd_config.image_config_path = recording_config.clone();
    config.pack_recording = true;

    let outcome = boot_and_wait(config, &trace_path, budget_secs).await;

    if outcome.is_none() {
        cleanup_pack_files(&trace_path, &recording_config).await;
        return None;
    }
    // Success: the derived recording config is no longer needed (the trace
    // stays next to the snapshot artifacts for the publisher).
    if let Err(error) = tokio::fs::remove_file(&recording_config).await {
        debug!(%error, "startup pack: remove derived recording config failed");
    }
    info!(path = %trace_path.display(), "startup trace recorded");
    Some(trace_path)
}

/// Boot the recording VM, wait (bounded) for the daemon-side window, then
/// clean up (unbounded). The VM handle and device id live outside the
/// timeout scope so the cleanup path always runs to completion.
async fn boot_and_wait(
    config: FirecrackerSnapshotConfig,
    trace_path: &Path,
    budget_secs: u64,
) -> Option<PathBuf> {
    let phase_t0 = std::time::Instant::now();
    let mut recording_vm = match FirecrackerSandbox::from_snapshot_config(&config) {
        Ok(vm) => vm,
        Err(error) => {
            warn!(%error, "startup pack: build recording VM failed");
            return None;
        }
    };
    if let Err(error) = recording_vm.start_nowait().await {
        warn!(%error, "startup pack: recording VM start failed");
        // Best-effort teardown of whatever start_nowait managed to create.
        if let Err(stop_error) = recording_vm.stop().await {
            warn!(%stop_error, "startup pack: recording VM stop after start failure failed");
        }
        return None;
    }
    let Some(dev_id) = recording_vm.dedicated_mem_device_id() else {
        warn!("startup pack: recording VM has no dedicated memory device");
        if let Err(stop_error) = recording_vm.stop().await {
            warn!(%stop_error, "startup pack: recording VM stop failed");
        }
        return None;
    };

    info!(
        elapsed_ms = phase_t0.elapsed().as_millis() as u64,
        "startup pack: recording VM started; status wait begins"
    );
    // Only this wait is bounded by the budget.
    let wait = async {
        loop {
            match UblkDeviceManager::global()
                .pack_recording_status(dev_id)
                .await
            {
                Ok(PackRecordingState::Done { pages, bytes, .. }) => {
                    break Some((pages, bytes));
                }
                Ok(PackRecordingState::Failed { reason }) => {
                    warn!(%reason, "startup pack: daemon recording failed");
                    break None;
                }
                Ok(PackRecordingState::Recording) => {}
                Err(error) => {
                    warn!(%error, "startup pack: recording status poll failed");
                    break None;
                }
            }
            tokio::time::sleep(Duration::from_millis(200)).await;
        }
    };
    let outcome = tokio::select! {
        outcome = tokio::time::timeout(Duration::from_secs(budget_secs), wait) => {
            match outcome {
                Ok(done) => WaitOutcome::Done(done),
                Err(_) => WaitOutcome::Timeout,
            }
        }
        _ = crate::snapshot::startup_pack::startup_manifest_abort_notify() => {
            WaitOutcome::Shutdown
        }
    };
    info!(
        elapsed_ms = phase_t0.elapsed().as_millis() as u64,
        "startup pack: status wait ended"
    );

    // Cleanup is never truncated by the budget on the normal paths — except
    // the daemon abort RPC, which is always time-boxed: it is best-effort
    // (stopping the VM deletes the device and makes the daemon abort the
    // recording anyway), and a dying daemon must never hang this task.
    let succeeded = matches!(outcome, WaitOutcome::Done(Some(_)));
    if !succeeded {
        let abort = async {
            if let Err(error) = UblkDeviceManager::global()
                .abort_pack_recording(dev_id)
                .await
            {
                warn!(%error, "startup pack: abort recording failed");
            }
        };
        let _ = tokio::time::timeout(Duration::from_secs(3), abort).await;
    }
    let stop = async {
        if let Err(error) = recording_vm.stop().await {
            warn!(%error, "startup pack: recording VM stop failed");
        }
    };
    if matches!(outcome, WaitOutcome::Shutdown) {
        let _ = tokio::time::timeout(Duration::from_secs(5), stop).await;
    } else {
        stop.await;
    }
    info!(
        elapsed_ms = phase_t0.elapsed().as_millis() as u64,
        "startup pack: recording VM stopped"
    );

    match outcome {
        WaitOutcome::Done(Some((pages, bytes))) => {
            debug!(pages, bytes, "startup pack recording finished");
            Some(trace_path.to_path_buf())
        }
        WaitOutcome::Done(None) => None,
        WaitOutcome::Timeout => {
            warn!(budget_secs, "startup pack: recording budget exceeded");
            None
        }
        WaitOutcome::Shutdown => None,
    }
}

enum WaitOutcome {
    Done(Option<(u32, u64)>),
    Timeout,
    Shutdown,
}

async fn cleanup_pack_files(trace_path: &Path, recording_config: &Path) {
    let mut tmp = trace_path.as_os_str().to_owned();
    tmp.push(".tmp");
    for path in [trace_path, Path::new(&tmp), recording_config] {
        if let Err(error) = tokio::fs::remove_file(path).await {
            if error.kind() != std::io::ErrorKind::NotFound {
                debug!(%error, path = %path.display(), "startup pack: cleanup remove failed");
            }
        }
    }
}

/// Derive a recording variant of the captured memory image config: same
/// lowers, background download force-disabled. Recording-time foreground
/// reads are still allowed (chain snapshots have remote parents fetched
/// on demand), but nothing should trigger a background bulk download.
async fn derive_recording_mem_config(src: &Path, snapshot_dir: &Path) -> Result<PathBuf> {
    let raw = tokio::fs::read(src)
        .await
        .with_context(|| format!("read memory image config {}", src.display()))?;
    let mut image_config: overlaybd::config::ImageConfig =
        serde_json::from_slice(&raw).context("parse memory image config")?;
    image_config.download_override = Some(overlaybd::config::DownloadConfig {
        enable: false,
        ..Default::default()
    });
    let derived = snapshot_dir.join("mem_image.pack-rec.json");
    tokio::fs::write(&derived, serde_json::to_vec_pretty(&image_config)?)
        .await
        .with_context(|| format!("write recording mem config {}", derived.display()))?;
    Ok(derived)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn ublk_prefetch_reads_manifest_ranges_from_device() -> Result<()> {
        let tmp = tempfile::TempDir::new()?;
        let device_path = tmp.path().join("ublk-device");
        let device = std::fs::File::create(&device_path)?;
        device.set_len(3 * overlaybd::startup_pack::PACK_PAGE_BYTES)?;

        let manifest = overlaybd::startup_manifest::StartupManifest {
            mem_virtual_size: 16 * overlaybd::startup_pack::PACK_PAGE_BYTES,
            prefix_pages: vec![0],
            ranges: vec![(
                overlaybd::startup_pack::PACK_PAGE_BYTES,
                2 * overlaybd::startup_pack::PACK_PAGE_BYTES,
            )],
        };
        let manifest_bytes = overlaybd::startup_manifest::encode_manifest(&manifest)?;
        let manifest_path = tmp
            .path()
            .join(crate::snapshot::MEMORY_STARTUP_PACK_ARTIFACT);
        tokio::fs::write(&manifest_path, &manifest_bytes).await?;
        let pack = crate::snapshot::ResolvedStartupPack {
            location: crate::snapshot::StartupPackLocation::LocalPath(manifest_path),
            pack_size: manifest_bytes.len() as u64,
            index_sha256: crate::snapshot::startup_pack::hex_sha256(&manifest_bytes),
            mem_virtual_size: manifest.mem_virtual_size,
        };

        prefetch_ublk_startup_pages(&device_path, &pack).await?;
        Ok(())
    }

    #[tokio::test]
    async fn recording_mem_config_disables_background_download() -> Result<()> {
        let tmp = tempfile::TempDir::new()?;
        let src = tmp.path().join("mem_image.json");
        let image_config = overlaybd::config::ImageConfig {
            repo_blob_url: "https://example/v2/repo/blobs".into(),
            lowers: vec![overlaybd::config::LayerConfig {
                file: "/layers/a.commit".into(),
                digest: "sha256:a".into(),
                size: 4096,
                ..Default::default()
            }],
            ..Default::default()
        };
        tokio::fs::write(&src, serde_json::to_vec_pretty(&image_config)?).await?;

        let derived = derive_recording_mem_config(&src, tmp.path()).await?;

        assert_eq!(derived, tmp.path().join("mem_image.pack-rec.json"));
        let written: overlaybd::config::ImageConfig =
            serde_json::from_slice(&tokio::fs::read(&derived).await?)?;
        let download = written.download_override.expect("download override");
        assert!(!download.enable);
        assert_eq!(written.lowers.len(), 1);
        assert_eq!(written.lowers[0].digest, "sha256:a");
        Ok(())
    }

    #[test]
    fn recording_gate_is_backend_neutral_and_disabled_by_default() {
        fn startup_pack_config(enabled: bool) -> crate::cfg::SnapshotStartupPackConfig {
            crate::cfg::SnapshotStartupPackConfig {
                enabled,
                record_min_window_ms: 200,
                record_quiet_ms: 300,
                record_max_window_ms: 2000,
                record_budget_secs: 10,
                max_pack_bytes: 1 << 30,
                consume_enabled: false,
                posix_ublk_prefetch_enabled: false,
                consume_timeout_secs: 30,
            }
        }

        let default_config = crate::cfg::SnapshotConfig::default();
        assert!(
            !recording_enabled_for(&default_config),
            "feature must be off by default"
        );

        let posix_enabled = crate::cfg::SnapshotConfig {
            repository_backend: crate::cfg::SnapshotRepositoryBackendKind::PosixFs,
            memory_startup_pack: startup_pack_config(true),
            ..Default::default()
        };
        assert!(
            recording_enabled_for(&posix_enabled),
            "enabled=true with posix_fs backend must record"
        );

        let oss_enabled = crate::cfg::SnapshotConfig {
            repository_backend: crate::cfg::SnapshotRepositoryBackendKind::Oss,
            memory_startup_pack: startup_pack_config(true),
            ..Default::default()
        };
        assert!(
            recording_enabled_for(&oss_enabled),
            "enabled=true with oss backend must record"
        );

        let oss_disabled = crate::cfg::SnapshotConfig {
            repository_backend: crate::cfg::SnapshotRepositoryBackendKind::Oss,
            memory_startup_pack: startup_pack_config(false),
            ..Default::default()
        };
        assert!(
            !recording_enabled_for(&oss_disabled),
            "enabled=false with oss backend must NOT record"
        );
    }
}
