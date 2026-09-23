//! Startup memory pack artifact constants and the committed-record
//! descriptor. The wire format itself lives in the overlaybd crate (shared
//! with the ublk daemon); this module re-exports what agentenv needs.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use tracing::{info, warn};

/// File name of the per-snapshot startup memory pack stored under
/// `artifacts/{snapshot_id}/` in the snapshot repository.
pub const MEMORY_STARTUP_PACK_ARTIFACT: &str = "memory-startup.pack";

/// File name of the local first-touch trace the recorder emits next to the
/// snapshot artifacts. The publisher expands it into the v3 startup manifest
/// (exact-order prefix plus merged ranges); it never leaves the node as-is.
pub const MEMORY_STARTUP_TRACE_ARTIFACT: &str = "memory-startup.trace";

/// A startup-manifest recording in flight plus the lease that keeps the
/// captured artifacts alive while the manifest is being built and stored.
/// Some repository backends may let that continuation outlive publication.
pub struct StartupRecording {
    /// Completes with the trace path on success, `None` on any failure.
    pub trace: tokio::task::JoinHandle<Option<std::path::PathBuf>>,
    /// Opaque hold-only lease (the capture root guard for sandbox captures;
    /// a unit placeholder for template builds).
    pub keep_alive: Box<dyn std::any::Any + Send>,
}

// ── Shutdown coordination for detached manifest work ────────────────────────

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::OnceLock;

/// Process-global coordination for detached startup-manifest work. Graceful
/// shutdown first stops NEW continuations from spawning, then drains the
/// in-flight ones (bounded): their recordings complete and manifests land
/// instead of being canceled outright. Only a drain timeout force-cancels
/// what is left.
struct StartupManifestShutdown {
    stop_new: AtomicBool,
    abort: AtomicBool,
    abort_notify: tokio::sync::Notify,
    in_flight: AtomicUsize,
    drained_notify: tokio::sync::Notify,
}

fn startup_manifest_shutdown() -> &'static StartupManifestShutdown {
    static SHUTDOWN: OnceLock<StartupManifestShutdown> = OnceLock::new();
    SHUTDOWN.get_or_init(|| StartupManifestShutdown {
        stop_new: AtomicBool::new(false),
        abort: AtomicBool::new(false),
        abort_notify: tokio::sync::Notify::new(),
        in_flight: AtomicUsize::new(0),
        drained_notify: tokio::sync::Notify::new(),
    })
}

/// True once shutdown was requested: the publish flow must not spawn new
/// startup-manifest continuations after this point.
pub fn startup_manifest_shutdown_requested() -> bool {
    startup_manifest_shutdown().stop_new.load(Ordering::SeqCst)
}

/// True once the drain timed out and remaining work was told to cancel.
pub fn startup_manifest_abort_requested() -> bool {
    startup_manifest_shutdown().abort.load(Ordering::SeqCst)
}

/// Notified once [`startup_manifest_abort_requested`] turns true.
pub fn startup_manifest_abort_notify() -> tokio::sync::futures::Notified<'static> {
    startup_manifest_shutdown().abort_notify.notified()
}

/// Request shutdown: stop spawning new startup-manifest continuations, then
/// wait (bounded by `timeout`) for the in-flight ones to drain so their
/// manifests still land. Whatever survives the timeout is force-canceled.
pub async fn drain_startup_manifest_tasks(timeout: std::time::Duration) {
    let state = startup_manifest_shutdown();
    state.stop_new.store(true, Ordering::SeqCst);
    let drained = async {
        while state.in_flight.load(Ordering::SeqCst) != 0 {
            state.drained_notify.notified().await;
        }
    };
    if tokio::time::timeout(timeout, drained).await.is_err()
        && state.in_flight.load(Ordering::SeqCst) != 0
    {
        state.abort.store(true, Ordering::SeqCst);
        state.abort_notify.notify_waiters();
    }
}

/// RAII guard counting one in-flight startup-manifest continuation.
pub(crate) struct StartupManifestTaskGuard {
    _private: (),
}

impl StartupManifestTaskGuard {
    pub(crate) fn new() -> Self {
        startup_manifest_shutdown()
            .in_flight
            .fetch_add(1, Ordering::SeqCst);
        Self { _private: () }
    }
}

impl Drop for StartupManifestTaskGuard {
    fn drop(&mut self) {
        let state = startup_manifest_shutdown();
        if state.in_flight.fetch_sub(1, Ordering::SeqCst) == 1 {
            state.drained_notify.notify_waiters();
        }
    }
}

/// Startup pack descriptor persisted in the committed record when — and only
/// when — the pack was recorded and stored successfully. Older snapshots do
/// not carry it and continue to resume on demand.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryStartupPackInfo {
    pub pack_size: u64,
    pub mem_virtual_size: u64,
    /// Hex-encoded sha256 of the v3 manifest bytes (older formats used their
    /// index section). The consumer verifies the received data against this
    /// digest before trusting any entry.
    pub index_sha256: String,
}

/// Backend-specific location of a resolved startup manifest.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StartupPackLocation {
    RemoteUrl(String),
    LocalPath(PathBuf),
}

/// Runtime-only startup pack reference handed from a repository resolver to
/// sandbox start: everything the daemon needs to register the prefetch. Never
/// trusted on its own — the daemon verifies the manifest against
/// `index_sha256` before acting on any entry.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResolvedStartupPack {
    pub location: StartupPackLocation,
    pub pack_size: u64,
    pub index_sha256: String,
    pub mem_virtual_size: u64,
}

/// Gate a committed descriptor into a runtime reference, only when
/// consumption is enabled. Missing descriptors and disabled consumption
/// fall back to plain on-demand resume; a manifest the consumer cannot
/// parse (including an unknown format version) is skipped the same way.
pub fn resolve_startup_pack_ref(
    info: Option<&MemoryStartupPackInfo>,
    consume_enabled: bool,
    location: impl FnOnce() -> StartupPackLocation,
) -> Option<ResolvedStartupPack> {
    if !consume_enabled {
        return None;
    }
    let info = info?;
    Some(ResolvedStartupPack {
        location: location(),
        pack_size: info.pack_size,
        index_sha256: info.index_sha256.clone(),
        mem_virtual_size: info.mem_virtual_size,
    })
}

/// Build the portable v3 startup manifest from a recorder trace. Repository
/// backends own only the final storage step (local atomic write or upload).
pub(crate) async fn build_manifest_from_trace(
    snapshot_id: &crate::snapshot::SnapshotId,
    trace_path: &Path,
) -> Option<(Vec<u8>, MemoryStartupPackInfo)> {
    let trace = match tokio::fs::read(trace_path).await {
        Ok(trace) => trace,
        Err(error) => {
            if error.kind() != std::io::ErrorKind::NotFound {
                tracing::debug!(
                    %error,
                    %snapshot_id,
                    "read startup trace failed; publishing without a manifest"
                );
            }
            return None;
        }
    };
    let (mem_virtual_size, offsets) = match overlaybd::startup_pack::decode_trace(&trace) {
        Ok(decoded) => decoded,
        Err(error) => {
            warn!(
                %error,
                %snapshot_id,
                "startup trace undecodable; publishing without a manifest"
            );
            return None;
        }
    };
    let manifest_doc = match overlaybd::startup_manifest::build_manifest(mem_virtual_size, &offsets)
    {
        Ok(doc) => doc,
        Err(error) => {
            warn!(%error, %snapshot_id, "build startup manifest failed (best-effort)");
            return None;
        }
    };
    let manifest_bytes = match overlaybd::startup_manifest::encode_manifest(&manifest_doc) {
        Ok(bytes) => bytes,
        Err(error) => {
            warn!(%error, %snapshot_id, "encode startup manifest failed (best-effort)");
            return None;
        }
    };
    info!(
        %snapshot_id,
        pages = offsets.len(),
        prefix_pages = manifest_doc.prefix_pages.len(),
        ranges = manifest_doc.ranges.len(),
        manifest_bytes = manifest_bytes.len(),
        "startup manifest built"
    );
    let info = MemoryStartupPackInfo {
        pack_size: manifest_bytes.len() as u64,
        mem_virtual_size,
        index_sha256: hex_sha256(&manifest_bytes),
    };
    Some((manifest_bytes, info))
}

pub(crate) fn hex_sha256(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write;
        let _ = write!(out, "{byte:02x}");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn committed_snapshot_without_memory_startup_deserializes() {
        // A record written before the field existed (mock() skips it) must
        // still parse, with the field defaulting to None.
        let json = serde_json::to_value(crate::snapshot::CommittedSnapshot::mock())
            .expect("serialize mock");
        assert!(json.get("memory_startup").is_none());
        let committed: crate::snapshot::CommittedSnapshot =
            serde_json::from_value(json).expect("old record must parse");
        assert!(committed.memory_startup.is_none());
    }

    #[test]
    fn committed_snapshot_with_memory_startup_roundtrips() {
        let mut committed = crate::snapshot::CommittedSnapshot::mock();
        committed.memory_startup = Some(MemoryStartupPackInfo {
            pack_size: 1024,
            mem_virtual_size: 1 << 30,
            index_sha256: "ab".repeat(32),
        });
        let json = serde_json::to_value(&committed).expect("serialize");
        let back: crate::snapshot::CommittedSnapshot = serde_json::from_value(json).expect("parse");
        assert_eq!(back.memory_startup, committed.memory_startup);
    }

    #[test]
    fn resolve_startup_pack_ref_gates_only_consumption() {
        let info = MemoryStartupPackInfo {
            pack_size: 4096,
            mem_virtual_size: 1 << 30,
            index_sha256: "ab".repeat(32),
        };
        // Consumption disabled → no reference.
        assert!(crate::snapshot::startup_pack::resolve_startup_pack_ref(
            Some(&info),
            false,
            || StartupPackLocation::RemoteUrl("s3://b/k".to_string())
        )
        .is_none());
        // No descriptor → no reference.
        assert!(
            crate::snapshot::startup_pack::resolve_startup_pack_ref(None, true, || {
                StartupPackLocation::RemoteUrl("s3://b/k".to_string())
            })
            .is_none()
        );
        // Any descriptor with consumption enabled resolves; the daemon
        // rejects non-manifest objects by magic instead.
        let resolved =
            crate::snapshot::startup_pack::resolve_startup_pack_ref(Some(&info), true, || {
                StartupPackLocation::RemoteUrl(
                    "s3://bucket/aenv-bk/artifacts/id/memory-startup.pack".to_string(),
                )
            })
            .expect("a descriptor with consumption enabled must resolve");
        assert_eq!(resolved.pack_size, 4096);
        assert_eq!(resolved.mem_virtual_size, 1 << 30);
        assert_eq!(resolved.index_sha256, "ab".repeat(32));
        assert!(matches!(
            resolved.location,
            StartupPackLocation::RemoteUrl(url) if url.ends_with("memory-startup.pack")
        ));
    }
}
