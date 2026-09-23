use std::fmt;
use std::net::{SocketAddr, TcpStream, ToSocketAddrs};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::sync::OnceLock;
use std::time::Duration;

use crate::backend::cache::{
    BkDownloadSubmitError, CacheFnTransFunc, CachedFile, FileCacheBackend, FileCacheBackendOptions,
    StartupPackHandle, StartupPackLayer, StartupPackSubmission,
};
use crate::backend::local::LocalFile;
use crate::backend::oss::OssBackend;
use crate::backend::registryfs_v2::RegistryFsV2;
use crate::config::{
    load_global_config, resolve_image_config_local_paths, validate_image_config, DownloadConfig,
    GlobalConfig, ImageConfig,
};
use crate::image::image_file::ImageFile;
use crate::io::dispatch_file::{build_remote_io_runtime, RuntimeDispatchFile};
use crate::io::virtual_file::VirtualFile;
use crate::lsmt::file::CommitArgs;
use anyhow::{bail, Context, Result};
use dashmap::DashSet;
use tokio::sync::OnceCell;
use uuid::Uuid;

const CONNECT_TIMEOUT: Duration = Duration::from_secs(1);

/// Upload shape for [`ImageService::export_upper_as_oss_sealed`].
///
/// Sized for the ceiling rather than for memory, because this entry point is
/// library surface: it can be handed an image of any size, and the part size is
/// what bounds how large that may be. S3 allows 10,000 parts, so 64 MiB caps a
/// single object at ~625 GiB, where 16 MiB would cap it at 156 GiB — and
/// exceeding the cap is not caught up front, it fails partway through the upload.
///
/// The price is memory: peak is `(2 * concurrency + 2) * part_size`, so ~640 MiB
/// here. Concurrency stays at 4 to keep that from doubling; past the point where
/// it covers the round-trip time, more parts in flight buy little. See
/// `backend::oss::upload_file_streaming` for both derivations.
const OSS_SEALED_UPLOAD_PART_SIZE: usize = 64 * 1024 * 1024;
const OSS_SEALED_UPLOAD_CONCURRENCY: usize = 4;

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum RemoteOpenMode {
    Direct,
    Cached,
}

struct ImageServiceInner {
    config_path: PathBuf,
    global_config: GlobalConfig,
    p2p_publish_url: Option<String>,
    remote_runtime: OnceCell<RemoteRuntime>,
    remote_mode: parking_lot::RwLock<RemoteOpenMode>,
    /// Handle to the runtime all remote network I/O is dispatched onto: the
    /// dedicated `remote_io_runtime` when the global config sets
    /// `remoteIoWorkers` > 0, otherwise the runtime that constructed this
    /// service (e.g. unit tests). Either way, remote I/O never runs on
    /// per-device ublk queue runtimes (dropped on device teardown).
    remote_io_handle: tokio::runtime::Handle,
    /// The dedicated remote-I/O runtime, created only when the global config
    /// sets `remoteIoWorkers` > 0. `Option` so `Drop` can move it out for a
    /// non-blocking shutdown.
    remote_io_runtime: Option<tokio::runtime::Runtime>,
}

impl Drop for ImageServiceInner {
    fn drop(&mut self) {
        if let Some(runtime) = self.remote_io_runtime.take() {
            // `Runtime::drop` blocks during shutdown and panics inside an
            // async context (tests, the daemon's main runtime); shut down in
            // the background instead. In-flight tasks are cancelled, and a
            // stale `RuntimeDispatchFile` handle spawning afterwards simply
            // yields cancelled join errors.
            runtime.shutdown_background();
        }
    }
}

struct RemoteRuntime {
    underlay_registryfs: RegistryFsV2,
    oss_backend: Option<OssBackend>,
    file_cache: Option<FileCacheBackend>,
}

pub(crate) struct CacheDownloadRequest {
    file: Arc<CachedFile>,
    config: DownloadConfig,
}

impl fmt::Debug for ImageServiceInner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ImageServiceInner")
            .field("config_path", &self.config_path)
            .field("global_config", &self.global_config)
            .field("p2p_publish_enabled", &self.p2p_publish_url.is_some())
            .field(
                "remote_runtime_initialized",
                &self.remote_runtime.get().is_some(),
            )
            .finish_non_exhaustive()
    }
}

#[derive(Clone)]
pub struct ImageService {
    inner: Arc<ImageServiceInner>,
}

impl fmt::Debug for ImageService {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ImageService")
            .field("config_path", &self.inner.config_path)
            .field("global_config", &self.inner.global_config)
            .finish_non_exhaustive()
    }
}

impl ImageService {
    async fn with_global_config(global_config: GlobalConfig, config_path: PathBuf) -> Result<Self> {
        Self::with_global_config_and_p2p_publish_url(global_config, config_path, None).await
    }

    async fn with_global_config_and_p2p_publish_url(
        global_config: GlobalConfig,
        config_path: PathBuf,
        p2p_publish_url: Option<String>,
    ) -> Result<Self> {
        let (remote_io_runtime, remote_io_handle) = if global_config.remote_io_workers > 0 {
            // Production deployments: dedicated runtime sized by the config.
            let runtime = build_remote_io_runtime(global_config.remote_io_workers)?;
            let handle = runtime.handle().clone();
            (Some(runtime), handle)
        } else {
            // No dedicated runtime requested: dispatch onto the runtime that
            // constructs this service (e.g. unit tests).
            (None, tokio::runtime::Handle::current())
        };

        Ok(Self {
            inner: Arc::new(ImageServiceInner {
                config_path,
                global_config,
                p2p_publish_url,
                remote_runtime: OnceCell::new(),
                remote_mode: parking_lot::RwLock::new(RemoteOpenMode::Cached),
                remote_io_handle,
                remote_io_runtime,
            }),
        })
    }

    /// Creates an `ImageService` from an in-memory [`GlobalConfig`].
    pub async fn new(global_config: GlobalConfig) -> Result<Self> {
        Self::with_global_config(global_config, PathBuf::new()).await
    }

    pub async fn from_config_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let global_config = load_global_config(&path)?;
        Self::with_global_config(global_config, path).await
    }

    pub async fn from_config_path_with_p2p_publish_url(
        path: impl AsRef<Path>,
        p2p_publish_url: Option<String>,
    ) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let global_config = load_global_config(&path)?;
        Self::with_global_config_and_p2p_publish_url(global_config, path, p2p_publish_url).await
    }

    /// Return the path from which this `ImageService` was loaded.
    ///
    /// Returns an empty path if the service was created from an in-memory
    /// [`GlobalConfig`] via [`ImageService::new`].
    pub fn config_path(&self) -> &Path {
        &self.inner.config_path
    }

    async fn build_file_cache(cfg: &GlobalConfig) -> Result<Option<FileCacheBackend>> {
        match cfg.cache_config.cache_type.as_str() {
            "" | "file" => {
                std::fs::create_dir_all(&cfg.cache_config.cache_dir)?;
                let mut options = FileCacheBackendOptions::from_cache_config(&cfg.cache_config)?;
                // The cache-owned background download scheduler takes its
                // node-level caps from the global download config; per-image
                // overrides never resize them.
                options.bk_download_max_inflight_blocks = cfg.download.max_inflight_blocks;
                options.bk_download_max_concurrent_files = cfg.download.max_concurrent_files;
                options.bk_download_block_size = cfg.download.block_size;
                let basename_transform: CacheFnTransFunc = Arc::new(|origin| {
                    let basename = Path::new(origin)
                        .file_name()
                        .and_then(|v| v.to_str())
                        .filter(|v| !v.is_empty())?;
                    Some(format!("/{basename}"))
                });
                let backend = FileCacheBackend::with_options_and_trans_func(
                    options,
                    Some(basename_transform),
                )
                .await?;
                Ok(Some(backend))
            }
            "ocf" | "download" => bail!(
                "cache type {} is not migrated in Rust image_service yet",
                cfg.cache_config.cache_type
            ),
            other => bail!("unknown cache type: {other}"),
        }
    }

    fn current_accelerate_address(&self) -> String {
        let remote_mode = *self.inner.remote_mode.read();
        if remote_mode == RemoteOpenMode::Direct {
            self.inner.global_config.p2p_config.address.clone()
        } else {
            String::new()
        }
    }

    async fn build_remote_runtime(&self) -> Result<RemoteRuntime> {
        let underlay_registryfs = RegistryFsV2::from_global_config(&self.inner.global_config)?;
        underlay_registryfs.set_accelerate_address(self.current_accelerate_address());
        let oss_backend = if self.inner.global_config.oss_config.enable {
            Some(OssBackend::new(&self.inner.global_config.oss_config)?)
        } else {
            None
        };
        let file_cache = Self::build_file_cache(&self.inner.global_config).await?;

        Ok(RemoteRuntime {
            underlay_registryfs,
            oss_backend,
            file_cache,
        })
    }

    async fn remote_runtime(&self) -> Result<&RemoteRuntime> {
        let service = self.clone();
        self.inner
            .remote_runtime
            .get_or_try_init(move || async move {
                // Run the whole construction on the service's dedicated
                // remote-I/O runtime: the cache spawns its background workers
                // (download scheduler, eviction/checkpoint loops) with a bare
                // `tokio::spawn` / `Handle::try_current` during construction,
                // which must bind to that runtime rather than the first
                // caller's — the remote runtime is initialized lazily and
                // the caller may be a per-device ublk queue runtime that is
                // dropped on device teardown.
                let handle = service.inner.remote_io_handle.clone();
                handle
                    .spawn(async move { service.build_remote_runtime().await })
                    .await
                    .context("remote runtime init task failed to join")?
            })
            .await
    }

    pub fn global_config(&self) -> &GlobalConfig {
        &self.inner.global_config
    }

    /// The object-store backend, or `None` when `ossConfig.enable` is false.
    ///
    /// Exposed so a caller that needs object storage for its own purposes — reading
    /// a metadata object, uploading a sealed layer — shares this one instead of
    /// constructing a second `OssBackend` from the same `ossConfig`. A second one
    /// would work, but it would carry its own operator cache and its own resolved
    /// credentials, and a `credentialProcess` costs a subprocess per cache miss.
    /// `OssBackend` is `Arc`-backed, so a clone of the returned value shares both.
    ///
    /// **This initializes the whole remote runtime**, including the registry client
    /// and the file cache, because they share one `OnceCell` with the backend. A
    /// caller serving only local images should check `ossConfig.enable` and not ask.
    pub async fn oss_backend(&self) -> Result<Option<&OssBackend>> {
        Ok(self.remote_runtime().await?.oss_backend.as_ref())
    }

    pub fn io_engine(&self) -> u32 {
        self.inner.global_config.io_engine
    }

    #[cfg(test)]
    pub(crate) async fn cached_file_stats(
        &self,
        source: &str,
    ) -> Result<Option<crate::backend::cache::CachedFileStats>> {
        Ok(self
            .remote_runtime()
            .await?
            .file_cache
            .as_ref()
            .and_then(|cache| cache.file_stats(source)))
    }

    #[cfg(test)]
    pub(crate) async fn file_cache_for_test(&self) -> Result<Option<FileCacheBackend>> {
        Ok(self.remote_runtime().await?.file_cache.clone())
    }

    #[cfg(test)]
    pub(crate) fn set_remote_mode_direct_for_test(&self) {
        *self.inner.remote_mode.write() = RemoteOpenMode::Direct;
    }

    pub(crate) fn p2p_uuid_address(&self) -> Option<String> {
        let p2p = &self.inner.global_config.p2p_config;
        if !p2p.enable {
            return None;
        }
        p2p.address
            .trim_end_matches('/')
            .strip_suffix("/p2p-http")
            .map(|base| format!("{base}/p2p-uuid"))
    }

    pub fn load_image_config(&self, path: impl AsRef<Path>) -> Result<ImageConfig> {
        let path = path.as_ref();
        let raw = std::fs::read_to_string(path)?;
        let mut cfg: ImageConfig =
            serde_json::from_str(&raw).context("parse image config json failed")?;
        resolve_image_config_local_paths(path, &mut cfg);
        validate_image_config(&cfg)?;
        Ok(cfg)
    }

    pub async fn create_image_file(&self, path: impl AsRef<Path>) -> Result<ImageFile> {
        let device_key =
            std::fs::canonicalize(path.as_ref()).unwrap_or_else(|_| path.as_ref().to_path_buf());
        let image_config = self.load_image_config(path)?;
        let result_file = image_config.result_file.clone();

        let _ = self.enable_acceleration();

        match ImageFile::open(image_config, self.clone(), Some(device_key)).await {
            Ok(image) => {
                self.set_result_file(&result_file, "success")?;
                Ok(image)
            }
            Err(err) => {
                let _ = self.set_result_file(&result_file, &format!("failed:{err}"));
                Err(err)
            }
        }
    }

    pub fn enable_acceleration(&self) -> bool {
        let p2p = &self.inner.global_config.p2p_config;
        let accelerate_address = if p2p.enable && check_accelerate_url(&p2p.address) {
            *self.inner.remote_mode.write() = RemoteOpenMode::Direct;
            p2p.address.clone()
        } else {
            *self.inner.remote_mode.write() = RemoteOpenMode::Cached;
            String::new()
        };

        if let Some(remote_runtime) = self.inner.remote_runtime.get() {
            remote_runtime
                .underlay_registryfs
                .set_accelerate_address(accelerate_address.clone());
        }

        !accelerate_address.is_empty()
    }

    pub async fn open_remote_blob_with_size(
        &self,
        url: &str,
        source_size: Option<u64>,
    ) -> Result<Arc<dyn VirtualFile>> {
        let remote_runtime = self.remote_runtime().await?;
        let source = self.open_backend_source_with_size(url, source_size).await?;
        // OSS blobs always go through the file cache (when available) regardless
        // of RemoteOpenMode. RemoteOpenMode::Direct is only meaningful for the
        // P2P accelerator path, which acts as its own cache; OSS has no P2P
        // channel, so we always want the local file cache as a read-ahead layer.
        if Self::is_oss_url(url) {
            if let Some(cache) = remote_runtime.file_cache.as_ref() {
                let cache_file = Self::open_cached_blob(cache, url, source, source_size).await?;
                return Ok(cache_file);
            }
            return Ok(source);
        }
        let remote_mode = *self.inner.remote_mode.read();
        match remote_mode {
            RemoteOpenMode::Direct => Ok(source),
            RemoteOpenMode::Cached => {
                if let Some(cache) = remote_runtime.file_cache.as_ref() {
                    let cache_file =
                        Self::open_cached_blob(cache, url, source, source_size).await?;
                    Ok(cache_file)
                } else {
                    Ok(source)
                }
            }
        }
    }

    pub(crate) async fn open_remote_blob_for_bk_download_with_size(
        &self,
        url: &str,
        source_size: Option<u64>,
        config: DownloadConfig,
    ) -> Result<(Arc<dyn VirtualFile>, CacheDownloadRequest)> {
        let remote_runtime = self.remote_runtime().await?;
        let cache = remote_runtime
            .file_cache
            .as_ref()
            .context("background download requires a file cache backend")?;
        let source = self.open_backend_source_with_size(url, source_size).await?;
        let cache_file = Self::open_cached_blob(cache, url, source, source_size).await?;
        Ok((
            cache_file.clone(),
            CacheDownloadRequest {
                file: cache_file,
                config,
            },
        ))
    }

    /// Submit background downloads for freshly opened layers.
    ///
    /// Background download is a best-effort accelerator: submission is
    /// registered with the cache scheduler and never fails due to execution
    /// pressure, so a busy scheduler cannot make an image open skip
    /// background download — tasks simply run later. A shut-down scheduler is
    /// skipped with a warning (foreground `CachedFile` reads refill missing
    /// blocks from the origin on demand); a missing file-cache backend is a
    /// configuration error and still fails.
    pub(crate) async fn submit_bk_downloads(
        &self,
        requests: Vec<CacheDownloadRequest>,
        device_key: Option<PathBuf>,
    ) -> Result<()> {
        if requests.is_empty() {
            return Ok(());
        }
        let remote_runtime = self.remote_runtime().await?;
        let cache = remote_runtime
            .file_cache
            .as_ref()
            .context("background download requires a file cache backend")?;
        let request_count = requests.len();
        // Submit on the service's dedicated remote-I/O runtime: the
        // scheduler spawns its per-task readiness timers with a bare
        // `tokio::spawn` during submission, which must bind to that runtime
        // rather than the caller's (a per-device ublk queue runtime is
        // dropped on device teardown, which would silently strand the
        // timers).
        let cache = cache.clone();
        let submit_device_key = device_key.clone();
        let result = self
            .inner
            .remote_io_handle
            .spawn(async move {
                cache.submit_bk_download_batch(
                    requests
                        .into_iter()
                        .map(|request| (request.file, request.config, submit_device_key.clone()))
                        .collect(),
                )
            })
            .await
            .context("background download submit task failed to join")?;
        match result {
            Ok(()) => Ok(()),
            Err(error) => match error.downcast_ref::<BkDownloadSubmitError>() {
                Some(submit_error) => {
                    // Log only the fixed category and the device key basename;
                    // the full path can expose tenant/host metadata.
                    let device_name = device_key
                        .as_ref()
                        .and_then(|key| key.file_name().map(|name| name.to_string_lossy()));
                    tracing::warn!(
                        error_category = submit_error.category(),
                        request_count,
                        device_key = device_name.as_deref().unwrap_or(""),
                        "skipping background download submission; foreground reads refill from origin"
                    );
                    Ok(())
                }
                None => Err(error).context("submit background downloads"),
            },
        }
    }

    /// Register a v3 startup memory manifest prefetch for the memory image at
    /// `image_config_path`. Remote objects are submitted to the shared cache
    /// scheduler; local POSIX layers are read through buffered I/O to warm the
    /// host page cache. Returns `None` when the image has no compatible
    /// layers or required remote cache. The prefetch is best-effort: normal
    /// on-demand reads always proceed regardless.
    pub async fn prefetch_startup_pack(
        &self,
        image_config_path: impl AsRef<Path>,
        pack: StartupPackPrefetch,
    ) -> Result<Option<StartupPackPrefetchHandle>> {
        let image_config = self.load_image_config(image_config_path.as_ref())?;
        if let StartupPackSource::LocalPath(manifest_path) = &pack.source {
            // Local memory layers are opened with buffered I/O under the
            // normal io_uring engine. libaio selects O_DIRECT, where warming
            // the host page cache would not accelerate the device path.
            if self.io_engine() == super::image_file::IO_ENGINE_LIBAIO {
                tracing::warn!(
                    path = %manifest_path.display(),
                    "skipping POSIX startup prefetch because the memory image uses direct I/O"
                );
                return Ok(None);
            }
            let Some(layers) = collect_local_startup_pack_layers(&image_config) else {
                return Ok(None);
            };
            let task_key = local_startup_task_key(&pack.index_sha256, &layers);
            if !local_startup_tasks().insert(task_key.clone()) {
                return Ok(Some(StartupPackPrefetchHandle::Local));
            }
            let runtime = self.inner.remote_io_handle.clone();
            runtime.spawn(async move {
                let timeout = pack.timeout;
                let result =
                    tokio::time::timeout(timeout, execute_local_startup_prefetch(&pack, &layers))
                        .await;
                match result {
                    Ok(Ok((ranges, bytes))) => tracing::info!(
                        task_key = %task_key,
                        ranges,
                        bytes,
                        "POSIX startup manifest prefetch done"
                    ),
                    Ok(Err(error)) => tracing::warn!(
                        task_key = %task_key,
                        %error,
                        "POSIX startup manifest prefetch failed"
                    ),
                    Err(_) => tracing::warn!(
                        task_key = %task_key,
                        timeout_secs = timeout.as_secs(),
                        "POSIX startup manifest prefetch timed out"
                    ),
                }
                local_startup_tasks().remove(&task_key);
            });
            return Ok(Some(StartupPackPrefetchHandle::Local));
        }

        let remote_runtime = self.remote_runtime().await?;
        let Some(cache) = remote_runtime.file_cache.as_ref() else {
            return Ok(None);
        };
        let Some(refs) = collect_startup_pack_layers(&image_config) else {
            return Ok(None);
        };
        let mut layers = Vec::with_capacity(refs.len());
        for (url, digest, size) in refs {
            let source = self.open_backend_source_with_size(&url, Some(size)).await?;
            let file = Self::open_cached_blob(cache, &url, source.clone(), Some(size)).await?;
            layers.push(StartupPackLayer {
                digest,
                size,
                file,
                source,
            });
        }
        // The pack object itself bypasses the file cache: it is consumed
        // exactly once, and only its layer blocks belong in the cache. The
        // size hint avoids a stat probe.
        let StartupPackSource::RemoteUrl(url) = &pack.source else {
            unreachable!("local startup packs returned above")
        };
        let pack_source = self
            .open_backend_source_with_size(url, Some(pack.pack_size))
            .await?;
        let handle = cache.submit_startup_pack(StartupPackSubmission {
            task_key: format!("startup-pack:{}", pack.index_sha256),
            pack_source,
            pack_size: pack.pack_size,
            index_sha256: pack.index_sha256,
            mem_virtual_size: pack.mem_virtual_size,
            layers,
            timeout: pack.timeout,
        })?;
        Ok(Some(StartupPackPrefetchHandle::Remote(handle)))
    }

    async fn open_cached_blob(
        cache: &FileCacheBackend,
        url: &str,
        source: Arc<dyn VirtualFile>,
        source_size: Option<u64>,
    ) -> Result<Arc<CachedFile>> {
        let source_size = match source_size {
            Some(size) => size,
            None => source.size().await?,
        };
        cache
            .open_file_with_source_size(url.to_string(), source, source_size)
            .await
    }

    pub async fn open_remote_blob(&self, url: &str) -> Result<Arc<dyn VirtualFile>> {
        self.open_remote_blob_with_size(url, None).await
    }

    pub async fn open_source_blob(&self, url: &str) -> Result<Arc<dyn VirtualFile>> {
        self.open_source_blob_with_size(url, None).await
    }

    pub(crate) async fn open_source_blob_with_size(
        &self,
        url: &str,
        source_size: Option<u64>,
    ) -> Result<Arc<dyn VirtualFile>> {
        self.open_backend_source_with_size(url, source_size).await
    }

    fn is_oss_url(url: &str) -> bool {
        match reqwest::Url::parse(url) {
            Ok(parsed) => matches!(parsed.scheme(), "s3" | "oss"),
            Err(_) => false,
        }
    }

    async fn open_backend_source_with_size(
        &self,
        url: &str,
        source_size: Option<u64>,
    ) -> Result<Arc<dyn VirtualFile>> {
        let remote_runtime = self.remote_runtime().await?;
        let oss_backend = remote_runtime.oss_backend.clone();
        let registryfs = remote_runtime.underlay_registryfs.clone();
        let url = url.to_string();
        // Run the open itself on the pinned runtime as well: opening may
        // issue requests (e.g. the eager size fetch in `RegistryFsV2::open`
        // when no hint is available), and any pooled connection used there
        // must live on the pinned runtime. Dispatching the whole open keeps
        // that guarantee regardless of what the open path does internally.
        let source: Arc<dyn VirtualFile> = self
            .inner
            .remote_io_handle
            .spawn(async move {
                if Self::is_oss_url(&url) {
                    let oss = oss_backend
                        .as_ref()
                        .ok_or_else(|| anyhow::anyhow!("OSS backend not enabled in config"))?;
                    oss.open_with_size_hint(&url, source_size)
                } else {
                    match source_size {
                        Some(size) => Ok(registryfs.open_with_size_hint(url, Some(size))),
                        None => registryfs.open(url).await,
                    }
                }
            })
            .await
            .context("remote blob open task failed to join")??;
        // Pin all subsequent remote network I/O (opendal/reqwest connection
        // tasks) to the service's remote-I/O runtime so per-device ublk
        // queue runtimes never own pooled HTTP connections. See
        // `io::dispatch_file`.
        let wrapped: Arc<dyn VirtualFile> =
            RuntimeDispatchFile::new(source, self.inner.remote_io_handle.clone());
        Ok(wrapped)
    }

    pub async fn export_upper_as_oss_sealed(
        &self,
        image: &ImageFile,
        dest_url: &str,
    ) -> Result<()> {
        if !Self::is_oss_url(dest_url) {
            bail!("destination url must use oss:// or s3://");
        }
        let remote_runtime = self.remote_runtime().await?;
        let oss = remote_runtime
            .oss_backend
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("OSS backend not enabled in config"))?;

        let stage_dir =
            Path::new(&self.inner.global_config.cache_config.cache_dir).join("oss-stage");
        std::fs::create_dir_all(&stage_dir)
            .with_context(|| format!("create oss stage dir {}", stage_dir.display()))?;
        let stage_path = stage_dir.join(format!("{}.lsmt", Uuid::new_v4()));
        let stage_file: Arc<dyn VirtualFile> = Arc::new(LocalFile::new(&stage_path)?);

        let export_result = image
            .export_upper_as_sealed(CommitArgs::new(stage_file.clone()))
            .await;
        if let Err(err) = export_result {
            let _ = tokio::fs::remove_file(&stage_path).await;
            return Err(err);
        }

        stage_file.sync().await?;
        let upload_result = oss
            .upload_path(
                dest_url,
                &stage_path,
                OSS_SEALED_UPLOAD_PART_SIZE,
                OSS_SEALED_UPLOAD_CONCURRENCY,
                None,
            )
            .await;
        // Always clean up the staging file regardless of upload outcome.
        // The staging file is a full copy of the sealed upper layer and can
        // be large; leaving it on disk across failures would accumulate waste.
        let _ = tokio::fs::remove_file(&stage_path).await;
        upload_result
    }

    fn set_result_file(&self, filename: &str, data: &str) -> Result<()> {
        if filename.is_empty() {
            return Ok(());
        }
        if let Some(parent) = Path::new(filename).parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        Ok(std::fs::write(filename, data.as_bytes())?)
    }
}

/// Parameters for one startup pack prefetch (see
/// [`ImageService::prefetch_startup_pack`]).
pub enum StartupPackSource {
    RemoteUrl(String),
    LocalPath(PathBuf),
}

pub struct StartupPackPrefetch {
    pub source: StartupPackSource,
    pub pack_size: u64,
    pub index_sha256: String,
    pub mem_virtual_size: u64,
    /// Hard bound on queueing plus remote downloading or local buffered reads.
    pub timeout: std::time::Duration,
}

pub enum StartupPackPrefetchHandle {
    Remote(StartupPackHandle),
    /// Local tasks are detached onto the daemon-owned runtime and deduplicated
    /// by manifest digest, so no keep-alive handle is required.
    Local,
}

fn local_startup_tasks() -> &'static DashSet<String> {
    static TASKS: OnceLock<DashSet<String>> = OnceLock::new();
    TASKS.get_or_init(DashSet::new)
}

fn local_startup_task_key(
    manifest_sha256: &str,
    layers: &[crate::pack_planner::FinalLayerSource],
) -> String {
    use sha2::Digest as _;

    let mut digest = sha2::Sha256::new();
    digest.update(manifest_sha256.as_bytes());
    for layer in layers {
        digest.update([0]);
        digest.update(layer.digest.as_bytes());
        digest.update(layer.size.to_le_bytes());
    }
    format!("startup-pack:{:x}", digest.finalize())
}

fn local_startup_read_slots() -> &'static Arc<tokio::sync::Semaphore> {
    static SLOTS: OnceLock<Arc<tokio::sync::Semaphore>> = OnceLock::new();
    SLOTS.get_or_init(|| Arc::new(tokio::sync::Semaphore::new(32)))
}

fn collect_local_startup_pack_layers(
    image_config: &ImageConfig,
) -> Option<Vec<crate::pack_planner::FinalLayerSource>> {
    if image_config.lowers.is_empty() {
        return None;
    }
    image_config
        .lowers
        .iter()
        .map(|lower| {
            if lower.file.is_empty() || lower.digest.is_empty() || lower.size == 0 {
                return None;
            }
            Some(crate::pack_planner::FinalLayerSource {
                digest: lower.digest.clone(),
                size: lower.size,
                source: crate::pack_planner::FinalLayerBytes::LocalPath(PathBuf::from(&lower.file)),
            })
        })
        .collect()
}

async fn execute_local_startup_prefetch(
    pack: &StartupPackPrefetch,
    layers: &[crate::pack_planner::FinalLayerSource],
) -> Result<(usize, u64)> {
    use sha2::Digest as _;

    let StartupPackSource::LocalPath(manifest_path) = &pack.source else {
        bail!("local startup prefetch requires a local manifest path");
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
    let digest = sha2::Sha256::digest(&manifest_bytes);
    let digest = digest
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    anyhow::ensure!(
        digest == pack.index_sha256,
        "startup manifest sha256 mismatch"
    );
    let manifest = crate::startup_manifest::decode_manifest(&manifest_bytes)
        .context("decode startup manifest")?;
    anyhow::ensure!(
        manifest.mem_virtual_size == pack.mem_virtual_size,
        "startup manifest memory size mismatch"
    );

    let mut spans = Vec::with_capacity(manifest.prefix_pages.len() + manifest.ranges.len());
    for &offset in &manifest.prefix_pages {
        spans.push((offset, crate::pack_planner::PLAN_PAGE_BYTES));
    }
    spans.extend_from_slice(&manifest.ranges);
    let plan =
        crate::pack_planner::plan_ranges(&spans, manifest.mem_virtual_size, layers, u64::MAX)
            .await
            .context("plan POSIX startup manifest ranges")?;
    let work = Arc::new(merge_local_prefetch_runs(&plan));
    let paths = layers
        .iter()
        .map(|layer| match &layer.source {
            crate::pack_planner::FinalLayerBytes::LocalPath(path) => Ok(path.clone()),
            crate::pack_planner::FinalLayerBytes::VFile(_) => {
                bail!("POSIX startup layer is not a local path")
            }
        })
        .collect::<Result<Vec<_>>>()?;
    let files = Arc::new(
        paths
            .iter()
            .map(|path| {
                LocalFile::open_ro(path)
                    .map(Arc::new)
                    .with_context(|| format!("open POSIX startup layer {}", path.display()))
            })
            .collect::<Result<Vec<_>>>()?,
    );
    let sizes = Arc::new(layers.iter().map(|layer| layer.size).collect::<Vec<_>>());
    let cursor = Arc::new(AtomicUsize::new(0));
    let bytes_read = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let mut workers = tokio::task::JoinSet::new();
    for _ in 0..4usize.min(work.len().max(1)) {
        let work = Arc::clone(&work);
        let files = Arc::clone(&files);
        let sizes = Arc::clone(&sizes);
        let cursor = Arc::clone(&cursor);
        let bytes_read = Arc::clone(&bytes_read);
        workers.spawn(async move {
            loop {
                let index = cursor.fetch_add(1, Ordering::Relaxed);
                let Some(&(layer, start_block, len_blocks)) = work.get(index) else {
                    return Ok::<(), anyhow::Error>(());
                };
                let _slot = local_startup_read_slots()
                    .acquire()
                    .await
                    .context("POSIX startup read semaphore closed")?;
                let offset = start_block
                    .checked_mul(crate::pack_planner::OBJECT_BLOCK_BYTES)
                    .context("POSIX startup range offset overflow")?;
                let len = sizes[layer]
                    .saturating_sub(offset)
                    .min(u64::from(len_blocks) * crate::pack_planner::OBJECT_BLOCK_BYTES);
                if len == 0 {
                    continue;
                }
                // `LocalFile::read_at` moves its owned allocation onto the
                // blocking pool. This keeps slow/shared POSIX storage from
                // pinning one of the daemon's async runtime workers.
                let expected = usize::try_from(len)?;
                let data = files[layer].read_at(offset, expected).await?;
                anyhow::ensure!(data.len() == expected, "short POSIX startup layer read");
                bytes_read.fetch_add(data.len() as u64, Ordering::Relaxed);
            }
        });
    }
    while let Some(result) = workers.join_next().await {
        result.context("POSIX startup prefetch worker join")??;
    }
    Ok((work.len(), bytes_read.load(Ordering::Relaxed)))
}

fn merge_local_prefetch_runs(plan: &crate::pack_planner::PackPlan) -> Vec<(usize, u64, u32)> {
    // Four MiB keeps each worker's read allocation bounded while still
    // coalescing the small random reads that dominate cold POSIX resume.
    const RUN_BLOCKS: u32 = 16;
    let mut items = plan
        .blocks
        .iter()
        .enumerate()
        .map(|(rank, block)| {
            (
                plan.objects[block.object as usize].layer,
                u64::from(block.block_id),
                1u32,
                rank,
            )
        })
        .collect::<Vec<_>>();
    items.sort_unstable_by_key(|(layer, block, _, _)| (*layer, *block));
    let mut runs: Vec<(usize, u64, u32, usize)> = Vec::new();
    for (layer, block, len, rank) in items {
        match runs.last_mut() {
            Some((last_layer, first, count, first_rank))
                if *last_layer == layer
                    && *first + u64::from(*count) == block
                    && *count + len <= RUN_BLOCKS =>
            {
                *count += len;
                *first_rank = (*first_rank).min(rank);
            }
            _ => runs.push((layer, block, len, rank)),
        }
    }
    runs.sort_unstable_by_key(|(_, _, _, rank)| *rank);
    runs.into_iter()
        .map(|(layer, block, len, _)| (layer, block, len))
        .collect()
}

/// Collect the OSS lowers of a memory image config as `(url, digest, size)`
/// in bottom-to-top order, with URLs built exactly like the image-open path
/// (`{repo_blob_url}/{digest}`). Returns `None` when any lower cannot be
/// bound (no remote URL, missing descriptor, or a non-OSS URL): the pack
/// binds by exact object-table match, so a partial binding is useless.
fn collect_startup_pack_layers(image_config: &ImageConfig) -> Option<Vec<(String, String, u64)>> {
    if image_config.lowers.is_empty() {
        return None;
    }
    let mut refs = Vec::with_capacity(image_config.lowers.len());
    for lower in &image_config.lowers {
        // A local-file lower is read from local bytes, not from the OSS
        // object: prefetching that object would warm a cache the device
        // never touches.
        if !lower.file.is_empty() {
            return None;
        }
        let base = lower.effective_repo_blob_url(&image_config.repo_blob_url);
        if base.is_empty() || lower.digest.is_empty() || lower.size == 0 {
            return None;
        }
        let url = format!("{}/{}", base.trim_end_matches('/'), lower.digest);
        if !ImageService::is_oss_url(&url) {
            return None;
        }
        refs.push((url, lower.digest.clone(), lower.size));
    }
    Some(refs)
}

fn check_accelerate_url(address: &str) -> bool {
    let Ok(url) = reqwest::Url::parse(address) else {
        return false;
    };
    let Some(host) = url.host_str() else {
        return false;
    };
    let port = url.port_or_known_default().unwrap_or(80);

    let addrs: Vec<SocketAddr> = match (host, port).to_socket_addrs() {
        Ok(v) => v.collect(),
        Err(_) => return false,
    };

    addrs
        .into_iter()
        .any(|addr| TcpStream::connect_timeout(&addr, CONNECT_TIMEOUT).is_ok())
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::extract::{Request, State};
    use axum::http::header::CONTENT_RANGE as CONTENT_RANGE_RAW;
    use axum::http::{HeaderMap as HttpHeaderMap, Response, StatusCode as HttpStatusCode};
    use axum::routing::any;
    use axum::Router;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use tempfile::TempDir;
    use tokio::net::TcpListener;

    fn write_json(path: &Path, value: &serde_json::Value) {
        std::fs::write(
            path,
            serde_json::to_vec_pretty(value).expect("serialize json"),
        )
        .expect("write json");
    }

    fn remote_lower(digest: &str, size: u64) -> crate::config::LayerConfig {
        crate::config::LayerConfig {
            digest: digest.to_string(),
            size,
            ..Default::default()
        }
    }

    async fn make_local_startup_layer(
        dir: &TempDir,
        name: &str,
        pages: &[(u32, u8)],
    ) -> crate::pack_planner::FinalLayerSource {
        use crate::lsmt::file::{CommitArgs, LSMTFile};

        let data = Arc::new(LocalFile::new(dir.path().join(format!("{name}.data"))).unwrap());
        let index = Arc::new(LocalFile::new(dir.path().join(format!("{name}.index"))).unwrap());
        let lsmt = LSMTFile::create(data, Some(index), 1 << 30, false)
            .await
            .unwrap();
        for (page, fill) in pages {
            lsmt.write_at(u64::from(*page) * 4096, &vec![*fill; 4096])
                .await
                .unwrap();
        }
        let path = dir.path().join(name);
        let destination: Arc<dyn VirtualFile> = Arc::new(LocalFile::new(&path).unwrap());
        lsmt.commit_with_args(CommitArgs::new(destination.clone()))
            .await
            .unwrap();
        crate::pack_planner::FinalLayerSource {
            digest: format!("sha256:{name}"),
            size: destination.size().await.unwrap(),
            source: crate::pack_planner::FinalLayerBytes::LocalPath(path),
        }
    }

    #[test]
    fn startup_pack_layers_builds_urls_like_image_open() {
        let config = ImageConfig {
            repo_blob_url: "s3://bucket/aenv-bk/managed-layers".to_string(),
            lowers: vec![
                remote_lower("sha256:aa", 100),
                remote_lower("sha256:bb", 200),
            ],
            ..Default::default()
        };
        let refs = collect_startup_pack_layers(&config).expect("all remote lowers bind");
        assert_eq!(
            refs,
            vec![
                (
                    "s3://bucket/aenv-bk/managed-layers/sha256:aa".to_string(),
                    "sha256:aa".to_string(),
                    100
                ),
                (
                    "s3://bucket/aenv-bk/managed-layers/sha256:bb".to_string(),
                    "sha256:bb".to_string(),
                    200
                ),
            ]
        );

        // A per-layer repo_blob_url wins over the config-level base.
        let mut layered = remote_lower("sha256:cc", 300);
        layered.repo_blob_url = "s3://bucket/other".to_string();
        let config = ImageConfig {
            repo_blob_url: "s3://bucket/aenv-bk/managed-layers".to_string(),
            lowers: vec![layered],
            ..Default::default()
        };
        let refs = collect_startup_pack_layers(&config).expect("layer-level url binds");
        assert_eq!(refs[0].0, "s3://bucket/other/sha256:cc");
    }

    #[test]
    fn startup_pack_layers_rejects_unbindable_lowers() {
        // Empty lowers.
        assert!(collect_startup_pack_layers(&ImageConfig::default()).is_none());
        // Local-file lower (no remote identity).
        let local = crate::config::LayerConfig {
            file: "/layers/a.commit".to_string(),
            digest: "sha256:aa".to_string(),
            size: 100,
            ..Default::default()
        };
        let config = ImageConfig {
            repo_blob_url: "s3://bucket/prefix".to_string(),
            lowers: vec![local],
            ..Default::default()
        };
        assert!(collect_startup_pack_layers(&config).is_none());
        // Missing descriptor.
        let config = ImageConfig {
            repo_blob_url: "s3://bucket/prefix".to_string(),
            lowers: vec![crate::config::LayerConfig::default()],
            ..Default::default()
        };
        assert!(collect_startup_pack_layers(&config).is_none());
        // Non-OSS URL scheme.
        let config = ImageConfig {
            repo_blob_url: "https://registry/v2/repo/blobs".to_string(),
            lowers: vec![remote_lower("sha256:aa", 100)],
            ..Default::default()
        };
        assert!(collect_startup_pack_layers(&config).is_none());
    }

    #[test]
    fn local_startup_pack_layers_require_complete_local_descriptors() {
        let config = ImageConfig {
            lowers: vec![crate::config::LayerConfig {
                file: "/layers/memory.commit".to_string(),
                digest: "sha256:aa".to_string(),
                size: 4096,
                ..Default::default()
            }],
            ..Default::default()
        };
        let layers = collect_local_startup_pack_layers(&config).expect("local layer binds");
        assert_eq!(layers.len(), 1);
        assert_eq!(layers[0].digest, "sha256:aa");
        assert!(matches!(
            &layers[0].source,
            crate::pack_planner::FinalLayerBytes::LocalPath(path)
                if path == Path::new("/layers/memory.commit")
        ));

        let incomplete = ImageConfig {
            lowers: vec![crate::config::LayerConfig {
                file: "/layers/memory.commit".to_string(),
                ..Default::default()
            }],
            ..Default::default()
        };
        assert!(collect_local_startup_pack_layers(&incomplete).is_none());
        assert!(collect_local_startup_pack_layers(&ImageConfig::default()).is_none());
    }

    #[test]
    fn local_startup_ranges_merge_physically_but_keep_first_need_priority() {
        let plan = crate::pack_planner::PackPlan {
            objects: vec![
                crate::pack_planner::PlannedObject {
                    digest: "sha256:a".to_string(),
                    size: 1 << 20,
                    layer: 0,
                },
                crate::pack_planner::PlannedObject {
                    digest: "sha256:b".to_string(),
                    size: 1 << 20,
                    layer: 1,
                },
            ],
            blocks: vec![
                crate::pack_planner::PlannedBlock {
                    object: 1,
                    block_id: 2,
                    metadata: false,
                },
                crate::pack_planner::PlannedBlock {
                    object: 0,
                    block_id: 7,
                    metadata: false,
                },
                crate::pack_planner::PlannedBlock {
                    object: 0,
                    block_id: 8,
                    metadata: false,
                },
            ],
            stats: Default::default(),
        };
        assert_eq!(merge_local_prefetch_runs(&plan), vec![(1, 2, 1), (0, 7, 2)]);
    }

    async fn spawn_server(app: Router) -> (String, tokio::task::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind server");
        let addr = listener.local_addr().expect("server addr");
        let handle = tokio::spawn(async move {
            axum::serve(listener, app).await.expect("run server");
        });
        (format!("http://{addr}"), handle)
    }

    fn parse_request_range(headers: &HttpHeaderMap) -> Option<(u64, u64)> {
        let raw = headers.get(reqwest::header::RANGE)?.to_str().ok()?.trim();
        let raw = raw.strip_prefix("bytes=")?;
        let (start, end) = raw.split_once('-')?;
        Some((start.parse().ok()?, end.parse().ok()?))
    }

    #[derive(Clone, Debug)]
    struct OssObjectState {
        blob: Arc<Vec<u8>>,
    }

    #[derive(Clone)]
    struct CountedBlobState {
        blob: Arc<Vec<u8>>,
        hits: Arc<AtomicUsize>,
    }

    async fn handle_counted_blob(
        State(state): State<CountedBlobState>,
        request: Request,
    ) -> Response<Body> {
        state.hits.fetch_add(1, Ordering::Relaxed);
        handle_oss_object(
            State(OssObjectState {
                blob: state.blob.clone(),
            }),
            request,
        )
        .await
    }

    async fn handle_oss_object(
        State(state): State<OssObjectState>,
        request: Request,
    ) -> Response<Body> {
        let headers = request.headers().clone();
        let body = state.blob.as_slice();
        let len = body.len() as u64;

        match *request.method() {
            axum::http::Method::HEAD => Response::builder()
                .status(HttpStatusCode::OK)
                .header(reqwest::header::CONTENT_LENGTH, len.to_string())
                .body(Body::empty())
                .expect("head response"),
            axum::http::Method::GET => {
                if let Some((start, end)) = parse_request_range(&headers) {
                    let start = start.min(len.saturating_sub(1));
                    let end = end.min(len.saturating_sub(1));
                    let chunk = body[start as usize..=end as usize].to_vec();
                    Response::builder()
                        .status(HttpStatusCode::PARTIAL_CONTENT)
                        .header(CONTENT_RANGE_RAW, format!("bytes {start}-{end}/{len}"))
                        .header(reqwest::header::CONTENT_LENGTH, chunk.len().to_string())
                        .body(Body::from(chunk))
                        .expect("range response")
                } else {
                    Response::builder()
                        .status(HttpStatusCode::OK)
                        .header(reqwest::header::CONTENT_LENGTH, len.to_string())
                        .body(Body::from(body.to_vec()))
                        .expect("get response")
                }
            }
            _ => Response::builder()
                .status(HttpStatusCode::METHOD_NOT_ALLOWED)
                .body(Body::empty())
                .expect("405 response"),
        }
    }

    async fn handle_aliyun_oss_object(
        State(state): State<OssObjectState>,
        request: Request,
    ) -> Response<Body> {
        let headers = request.headers();
        let auth = headers
            .get(reqwest::header::AUTHORIZATION)
            .and_then(|value| value.to_str().ok())
            .unwrap_or_default();
        let payload_header = headers
            .get("x-amz-content-sha256")
            .and_then(|value| value.to_str().ok())
            .unwrap_or_default();
        let signed_headers_ok = if request.method() == axum::http::Method::GET {
            auth.contains("SignedHeaders=host;range;x-amz-content-sha256;x-amz-date")
        } else {
            auth.contains("SignedHeaders=host;x-amz-content-sha256;x-amz-date")
        };
        if !auth.starts_with("AWS4-HMAC-SHA256 ")
            || payload_header != "UNSIGNED-PAYLOAD"
            || !signed_headers_ok
        {
            return Response::builder()
                .status(HttpStatusCode::FORBIDDEN)
                .body(Body::from("invalid aws sigv4 headers for aliyun oss"))
                .expect("403 response");
        }
        handle_oss_object(State(state), request).await
    }

    async fn handle_oss_get_only_object(
        State(state): State<OssObjectState>,
        request: Request,
    ) -> Response<Body> {
        if request.method() == axum::http::Method::HEAD {
            return Response::builder()
                .status(HttpStatusCode::METHOD_NOT_ALLOWED)
                .body(Body::from("head should not be called"))
                .expect("405 response");
        }
        handle_oss_object(State(state), request).await
    }

    #[tokio::test]
    async fn test_load_image_config_keeps_download_override_absent() {
        let tmp = TempDir::new().expect("tempdir");
        let global_path = tmp.path().join("overlaybd.json");
        let image_path = tmp.path().join("image.json");

        write_json(
            &global_path,
            &serde_json::json!({
                "registryFsVersion": "v2",
                "ioEngine": 0,
                "cacheConfig": {
                    "cacheType": "file",
                    "cacheDir": tmp.path().join("cache"),
                    "cacheSizeGB": 1,
                    "refillSize": 262144,
                    "blockSize": 65536
                },
                "download": {
                    "enable": true,
                    "delay": 12,
                    "delayExtra": 7,
                    "tryCnt": 9,
                    "blockSize": 131072
                }
            }),
        );

        write_json(
            &image_path,
            &serde_json::json!({
                "repoBlobUrl": "https://registry.example/v2/ns/repo/blobs",
                "lowers": [
                    {
                        "file": "/tmp/lower.data"
                    }
                ]
            }),
        );

        let service = ImageService::from_config_path(&global_path)
            .await
            .expect("service");
        let cfg = service.load_image_config(&image_path).expect("image cfg");
        let download = cfg.effective_download(service.global_config());

        assert!(cfg.download_override.is_none());
        assert!(download.enable);
        assert_eq!(download.delay, 12);
        assert_eq!(download.delay_extra, 7);
        assert_eq!(download.try_cnt, 9);
        assert_eq!(download.block_size, 131072);
    }

    #[tokio::test]
    async fn test_set_result_file_writes_status() {
        let tmp = TempDir::new().expect("tempdir");
        let global_path = tmp.path().join("overlaybd.json");
        let result_path = tmp.path().join("result.txt");

        write_json(
            &global_path,
            &serde_json::json!({
                "registryFsVersion": "v2",
                "ioEngine": 0,
                "cacheConfig": {
                    "cacheType": "file",
                    "cacheDir": tmp.path().join("cache"),
                    "cacheSizeGB": 1,
                    "refillSize": 262144,
                    "blockSize": 65536
                }
            }),
        );

        let service = ImageService::from_config_path(&global_path)
            .await
            .expect("service");
        service
            .set_result_file(result_path.to_string_lossy().as_ref(), "success")
            .expect("write result");

        let raw = std::fs::read_to_string(result_path).expect("read result");
        assert_eq!(raw, "success");
    }

    #[tokio::test]
    async fn test_create_image_file_with_local_upper_skips_eager_remote_init() {
        let tmp = TempDir::new().expect("tempdir");
        let global_path = tmp.path().join("overlaybd.json");
        let image_path = tmp.path().join("image.json");
        let upper_data = tmp.path().join("upper.data");
        let upper_index = tmp.path().join("upper.index");
        let result_path = tmp.path().join("result.txt");

        crate::helper::prepare_runtime_upper(
            &upper_data,
            Some(&upper_index),
            8192,
            crate::config::UpperMode::LogStructured,
        )
        .expect("prepare runtime upper");

        write_json(
            &global_path,
            &serde_json::json!({
                "registryFsVersion": "v2",
                "ioEngine": 0,
                "cacheConfig": {
                    "cacheType": "file",
                    "cacheDir": tmp.path().join("cache"),
                    "cacheSizeGB": 1,
                    "refillSize": 262144,
                    "blockSize": 65536
                },
                "certConfig": {
                    "certFile": tmp.path().join("missing-cert.pem"),
                    "keyFile": tmp.path().join("missing-key.pem")
                }
            }),
        );

        write_json(
            &image_path,
            &serde_json::json!({
                "upper": {
                    "data": upper_data,
                    "index": upper_index
                },
                "resultFile": result_path
            }),
        );

        let service = ImageService::from_config_path(&global_path)
            .await
            .expect("local-only service should not initialize remote runtime eagerly");
        let image = service
            .create_image_file(&image_path)
            .await
            .expect("local-only image should open without remote runtime");

        assert_eq!(image.size().await.expect("image size"), 8192);
    }

    #[tokio::test]
    async fn test_open_remote_blob_with_s3_url_reads_object() {
        let tmp = TempDir::new().expect("tempdir");
        let global_path = tmp.path().join("overlaybd.json");
        let object = b"hello from oss".to_vec();
        let app = Router::new()
            .route("/test-bucket/layers/lower", any(handle_oss_object))
            .with_state(OssObjectState {
                blob: Arc::new(object.clone()),
            });
        let (endpoint, server_handle) = spawn_server(app).await;

        write_json(
            &global_path,
            &serde_json::json!({
                "registryFsVersion": "v2",
                "ioEngine": 0,
                "cacheConfig": {
                    "cacheType": "file",
                    "cacheDir": tmp.path().join("cache"),
                    "cacheSizeGB": 1,
                    "refillSize": 262144,
                    "blockSize": 65536
                },
                "ossConfig": {
                    "enable": true,
                    "accessKeyId": "minioadmin",
                    "secretAccessKey": "minioadmin",
                    "defaultRegion": "us-east-1",
                    "defaultEndpoint": endpoint
                }
            }),
        );

        let service = ImageService::from_config_path(&global_path)
            .await
            .expect("service");
        let url = format!(
            "s3://test-bucket/layers/lower?endpoint={}&region=us-east-1",
            endpoint
        );

        let file = service
            .open_remote_blob_with_size(&url, None)
            .await
            .expect("open remote blob");
        let got = file.read_at(0, object.len()).await.expect("read object");
        assert_eq!(&got[..], object.as_slice());

        server_handle.abort();
    }

    #[tokio::test]
    async fn test_open_source_blob_with_size_hint_skips_head_for_oss() {
        let tmp = TempDir::new().expect("tempdir");
        let global_path = tmp.path().join("overlaybd.json");
        let object = b"hello hinted source blob".to_vec();
        let app = Router::new()
            .route(
                "/test-bucket/source/hinted",
                any(handle_oss_get_only_object),
            )
            .with_state(OssObjectState {
                blob: Arc::new(object.clone()),
            });
        let (endpoint, server_handle) = spawn_server(app).await;

        write_json(
            &global_path,
            &serde_json::json!({
                "registryFsVersion": "v2",
                "ioEngine": 0,
                "cacheConfig": {
                    "cacheType": "file",
                    "cacheDir": tmp.path().join("cache"),
                    "cacheSizeGB": 1,
                    "refillSize": 262144,
                    "blockSize": 65536
                },
                "ossConfig": {
                    "enable": true,
                    "accessKeyId": "minioadmin",
                    "secretAccessKey": "minioadmin",
                    "defaultRegion": "us-east-1",
                    "defaultEndpoint": endpoint
                }
            }),
        );

        let service = ImageService::from_config_path(&global_path)
            .await
            .expect("service");
        let url = format!(
            "s3://test-bucket/source/hinted?endpoint={}&region=us-east-1",
            endpoint
        );

        let file = service
            .open_source_blob_with_size(&url, Some(object.len() as u64))
            .await
            .expect("open source blob with size");
        let got = file.read_at(0, object.len()).await.expect("read object");
        assert_eq!(&got[..], object.as_slice());

        server_handle.abort();
    }

    #[tokio::test]
    async fn test_zero_remote_io_workers_uses_current_runtime() {
        let service = ImageService::new(GlobalConfig::default())
            .await
            .expect("service");
        assert!(service.inner.remote_io_runtime.is_none());
        assert_eq!(
            service.inner.remote_io_handle.id(),
            tokio::runtime::Handle::current().id()
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn local_startup_prefetch_reads_planned_posix_ranges() {
        let dir = TempDir::new().expect("tempdir");
        let pages = (0..200u32)
            .map(|page| (page, page as u8))
            .collect::<Vec<_>>();
        let layer = make_local_startup_layer(&dir, "memory.commit", &pages).await;
        let manifest = crate::startup_manifest::StartupManifest {
            mem_virtual_size: 1 << 30,
            prefix_pages: vec![0, 99 * 4096],
            ranges: vec![(150 * 4096, 4096)],
        };
        let manifest_bytes = crate::startup_manifest::encode_manifest(&manifest).unwrap();
        let manifest_path = dir.path().join("memory-startup.pack");
        std::fs::write(&manifest_path, &manifest_bytes).unwrap();
        let digest = {
            use sha2::Digest as _;
            format!("{:x}", sha2::Sha256::digest(&manifest_bytes))
        };
        let pack = StartupPackPrefetch {
            source: StartupPackSource::LocalPath(manifest_path),
            pack_size: manifest_bytes.len() as u64,
            index_sha256: digest,
            mem_virtual_size: 1 << 30,
            timeout: Duration::from_secs(5),
        };
        let (ranges, bytes) = execute_local_startup_prefetch(&pack, &[layer])
            .await
            .expect("prefetch local ranges");
        assert!(ranges > 0);
        assert!(bytes > 0);
    }

    #[tokio::test]
    async fn test_drop_service_in_async_context_shuts_down_remote_io() {
        let config = GlobalConfig {
            remote_io_workers: 2,
            ..GlobalConfig::default()
        };
        let service = ImageService::new(config).await.expect("service");
        let handle = service.inner.remote_io_handle.clone();
        // Dropping the service inside an async context must not panic: the
        // remote-io runtime is shut down in the background by
        // `ImageServiceInner`'s `Drop`.
        drop(service);
        // After shutdown, tasks spawned via a stale handle never run and
        // their join handles resolve to cancelled.
        let err = handle
            .spawn(async { 42 })
            .await
            .expect_err("spawn on a shut-down runtime must cancel");
        assert!(err.is_cancelled());
    }

    #[tokio::test]
    async fn test_open_source_blob_without_size_hint_dispatches_open() {
        let tmp = TempDir::new().expect("tempdir");
        let global_path = tmp.path().join("overlaybd.json");
        let object = b"dispatched registryfs open".to_vec();
        let hits = Arc::new(AtomicUsize::new(0));
        let app = Router::new()
            .route("/ns/repo/blobs/sha256:abc", any(handle_counted_blob))
            .with_state(CountedBlobState {
                blob: Arc::new(object.clone()),
                hits: hits.clone(),
            });
        let (endpoint, server_handle) = spawn_server(app).await;

        write_json(
            &global_path,
            &serde_json::json!({
                "registryFsVersion": "v2",
                "ioEngine": 0,
                "remoteIoWorkers": 2,
                "cacheConfig": {
                    "cacheType": "file",
                    "cacheDir": tmp.path().join("cache"),
                    "cacheSizeGB": 1,
                    "refillSize": 262144,
                    "blockSize": 65536
                }
            }),
        );

        let service = ImageService::from_config_path(&global_path)
            .await
            .expect("service");
        let url = format!("{endpoint}/ns/repo/blobs/sha256:abc");
        let file = service
            .open_source_blob_with_size(&url, None)
            .await
            .expect("open source blob without size hint");
        // Without a hint the open eagerly fetches the size — issued from
        // inside the pinned remote-io runtime, not the caller's runtime.
        // (A fresh registry origin may additionally probe for its auth
        // challenge, so only a lower bound is asserted.)
        let hits_after_open = hits.load(Ordering::Relaxed);
        assert!(hits_after_open >= 1);

        let size = file.size().await.expect("size");
        assert_eq!(size, object.len() as u64);
        // The size was fetched eagerly during open; asking for it again
        // must not hit the server.
        assert_eq!(hits.load(Ordering::Relaxed), hits_after_open);
        let got = file.read_at(0, object.len()).await.expect("read");
        assert_eq!(&got[..], object.as_slice());

        server_handle.abort();
    }

    #[tokio::test]
    async fn test_p2p_uuid_address_is_derived_from_http_facade_address() {
        let tmp = TempDir::new().expect("tempdir");
        let global_path = tmp.path().join("overlaybd.json");
        write_json(
            &global_path,
            &serde_json::json!({
                "registryFsVersion": "v2",
                "ioEngine": 0,
                "cacheConfig": {
                    "cacheType": "file",
                    "cacheDir": tmp.path().join("cache"),
                    "cacheSizeGB": 1,
                    "refillSize": 262144,
                    "blockSize": 65536
                },
                "p2pConfig": {
                    "enable": true,
                    "address": "http://127.0.0.1:9731/p2p-http/"
                }
            }),
        );

        let service = ImageService::from_config_path(&global_path)
            .await
            .expect("service");

        assert_eq!(
            service.p2p_uuid_address().as_deref(),
            Some("http://127.0.0.1:9731/p2p-uuid")
        );
    }

    #[tokio::test]
    async fn test_open_remote_blob_with_aliyun_endpoint_uses_unsigned_payload_sigv4() {
        let tmp = TempDir::new().expect("tempdir");
        let global_path = tmp.path().join("overlaybd.json");
        let object = b"hello aliyun oss".to_vec();
        let app = Router::new()
            .route(
                "/aliyun-bucket/objects/layer",
                any(handle_aliyun_oss_object),
            )
            .with_state(OssObjectState {
                blob: Arc::new(object.clone()),
            });
        let (endpoint, server_handle) = spawn_server(app).await;

        write_json(
            &global_path,
            &serde_json::json!({
                "registryFsVersion": "v2",
                "ioEngine": 0,
                "cacheConfig": {
                    "cacheType": "file",
                    "cacheDir": tmp.path().join("cache"),
                    "cacheSizeGB": 1,
                    "refillSize": 262144,
                    "blockSize": 65536
                },
                "ossConfig": {
                    "enable": true,
                    "accessKeyId": "aliyun-ak",
                    "secretAccessKey": "aliyun-sk",
                    "defaultRegion": "cn-hangzhou",
                    "defaultEndpoint": endpoint
                }
            }),
        );

        let service = ImageService::from_config_path(&global_path)
            .await
            .expect("service");
        let url = "oss://aliyun-bucket/objects/layer?region=cn-hangzhou".to_string();

        let file = service
            .open_remote_blob_with_size(&url, None)
            .await
            .expect("open remote blob");
        let got = file.read_at(0, object.len()).await.expect("read object");
        assert_eq!(&got[..], object.as_slice());

        server_handle.abort();
    }
}
