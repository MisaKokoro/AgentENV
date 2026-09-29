use std::future::Future;
use std::hash::Hash;
use std::sync::Arc;

use anyhow::{anyhow, Result};
use dashmap::mapref::entry::Entry;
use dashmap::DashMap;
use tokio::sync::watch;

#[derive(Clone)]
enum FlightState<T: Clone> {
    Running,
    Complete(std::result::Result<T, Arc<anyhow::Error>>),
}

struct Flight<T: Clone> {
    state: watch::Sender<FlightState<T>>,
}

impl<T: Clone> Flight<T> {
    fn new() -> Self {
        let (state, _) = watch::channel(FlightState::Running);
        Self { state }
    }

    fn complete(&self, result: std::result::Result<T, Arc<anyhow::Error>>) {
        self.state.send_replace(FlightState::Complete(result));
    }

    async fn wait(&self) -> Result<T> {
        let mut state = self.state.subscribe();
        loop {
            match state.borrow_and_update().clone() {
                FlightState::Running => {}
                FlightState::Complete(Ok(value)) => return Ok(value),
                FlightState::Complete(Err(error)) => return Err(anyhow!("{error:#}")),
            }
            state
                .changed()
                .await
                .map_err(|_| anyhow!("singleflight task ended without publishing a result"))?;
        }
    }
}

#[derive(Debug)]
pub(crate) struct SingleflightOutcome<T> {
    pub value: T,
    pub leader: bool,
}

/// Coalesces only operations that are currently running. Completed results
/// are removed before waiters are woken, so later calls execute again.
pub(crate) struct Singleflight<K, T>
where
    K: Eq + Hash,
    T: Clone,
{
    flights: Arc<DashMap<K, Arc<Flight<T>>>>,
}

impl<K, T> Default for Singleflight<K, T>
where
    K: Eq + Hash,
    T: Clone,
{
    fn default() -> Self {
        Self {
            flights: Arc::new(DashMap::new()),
        }
    }
}

impl<K, T> Singleflight<K, T>
where
    K: Eq + Hash + Clone + Send + Sync + 'static,
    T: Clone + Send + Sync + 'static,
{
    pub async fn run<F, Fut>(&self, key: K, operation: F) -> Result<SingleflightOutcome<T>>
    where
        F: FnOnce() -> Fut + Send + 'static,
        Fut: Future<Output = Result<T>> + Send + 'static,
    {
        let (flight, leader) = match self.flights.entry(key.clone()) {
            Entry::Occupied(entry) => (Arc::clone(entry.get()), false),
            Entry::Vacant(entry) => {
                let flight = Arc::new(Flight::new());
                entry.insert(Arc::clone(&flight));
                (flight, true)
            }
        };

        if leader {
            let flights = Arc::clone(&self.flights);
            let task_flight = Arc::clone(&flight);
            tokio::spawn(async move {
                // The nested task turns an operation panic into a result that
                // wakes every waiter. The supervisor itself owns no fallible
                // work before publishing that result.
                let result = match tokio::spawn(operation()).await {
                    Ok(Ok(value)) => Ok(value),
                    Ok(Err(error)) => Err(Arc::new(error)),
                    Err(error) => Err(Arc::new(anyhow!(
                        "singleflight operation task failed: {error}"
                    ))),
                };
                flights.remove_if(&key, |_, current| Arc::ptr_eq(current, &task_flight));
                task_flight.complete(result);
            });
        }

        let value = flight.wait().await?;
        Ok(SingleflightOutcome { value, leader })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use tokio::sync::{oneshot, Barrier};

    use super::*;

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_calls_share_one_running_operation() -> Result<()> {
        const CALLERS: usize = 100;
        let singleflight = Arc::new(Singleflight::<u64, u64>::default());
        let calls = Arc::new(AtomicUsize::new(0));
        let barrier = Arc::new(Barrier::new(CALLERS + 1));
        let (release, release_rx) = watch::channel(false);
        let mut tasks = Vec::new();

        for _ in 0..CALLERS {
            let singleflight = Arc::clone(&singleflight);
            let calls = Arc::clone(&calls);
            let barrier = Arc::clone(&barrier);
            let mut release_rx = release_rx.clone();
            tasks.push(tokio::spawn(async move {
                barrier.wait().await;
                singleflight
                    .run(7, move || async move {
                        calls.fetch_add(1, Ordering::SeqCst);
                        release_rx.wait_for(|released| *released).await?;
                        Ok(42)
                    })
                    .await
            }));
        }

        barrier.wait().await;
        while singleflight.flights.is_empty() {
            tokio::task::yield_now().await;
        }
        for _ in 0..CALLERS {
            tokio::task::yield_now().await;
        }
        release.send_replace(true);

        let mut leaders = 0;
        for task in tasks {
            let outcome = task.await??;
            assert_eq!(outcome.value, 42);
            leaders += usize::from(outcome.leader);
        }
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(leaders, 1);
        assert!(singleflight.flights.is_empty());
        Ok(())
    }

    #[tokio::test]
    async fn completed_result_is_not_cached() -> Result<()> {
        let singleflight = Singleflight::<u64, u64>::default();
        let calls = Arc::new(AtomicUsize::new(0));

        for _ in 0..2 {
            let calls = Arc::clone(&calls);
            let outcome = singleflight
                .run(7, move || async move {
                    calls.fetch_add(1, Ordering::SeqCst);
                    Ok(42)
                })
                .await?;
            assert!(outcome.leader);
            assert_eq!(outcome.value, 42);
        }
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        Ok(())
    }

    #[tokio::test]
    async fn leader_cancellation_does_not_cancel_shared_operation() -> Result<()> {
        let singleflight = Arc::new(Singleflight::<u64, u64>::default());
        let calls = Arc::new(AtomicUsize::new(0));
        let (started_tx, started_rx) = oneshot::channel();
        let (release, mut release_rx) = watch::channel(false);

        let leader_singleflight = Arc::clone(&singleflight);
        let leader_calls = Arc::clone(&calls);
        let leader = tokio::spawn(async move {
            leader_singleflight
                .run(7, move || async move {
                    leader_calls.fetch_add(1, Ordering::SeqCst);
                    let _ = started_tx.send(());
                    release_rx.wait_for(|released| *released).await?;
                    Ok(42)
                })
                .await
        });
        started_rx.await?;

        let follower_singleflight = Arc::clone(&singleflight);
        let follower = follower_singleflight.run(7, || async {
            anyhow::bail!("follower operation must not execute")
        });
        tokio::pin!(follower);
        tokio::select! {
            biased;
            _ = &mut follower => panic!("follower completed before release"),
            _ = tokio::task::yield_now() => {}
        }
        leader.abort();
        release.send_replace(true);

        let outcome = follower.await?;
        assert!(!outcome.leader);
        assert_eq!(outcome.value, 42);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        Ok(())
    }

    #[tokio::test]
    async fn operation_error_wakes_waiters_and_allows_retry() -> Result<()> {
        let singleflight = Arc::new(Singleflight::<u64, u64>::default());
        let (started_tx, started_rx) = oneshot::channel();
        let (release, mut release_rx) = watch::channel(false);

        let leader_singleflight = Arc::clone(&singleflight);
        let leader = tokio::spawn(async move {
            leader_singleflight
                .run(7, move || async move {
                    let _ = started_tx.send(());
                    release_rx.wait_for(|released| *released).await?;
                    anyhow::bail!("expected failure")
                })
                .await
        });
        started_rx.await?;

        let follower_singleflight = Arc::clone(&singleflight);
        let follower = follower_singleflight.run(7, || async {
            anyhow::bail!("follower operation must not execute")
        });
        tokio::pin!(follower);
        tokio::select! {
            biased;
            _ = &mut follower => panic!("follower completed before release"),
            _ = tokio::task::yield_now() => {}
        }
        release.send_replace(true);

        assert!(format!("{:#}", leader.await?.unwrap_err()).contains("expected failure"));
        assert!(format!("{:#}", follower.await.unwrap_err()).contains("expected failure"));

        let retry = singleflight.run(7, || async { Ok(42) }).await?;
        assert!(retry.leader);
        assert_eq!(retry.value, 42);
        Ok(())
    }
}
