//! Load balancing policies for SGLang router
//!
//! This module provides a unified abstraction for routing policies that work
//! across both regular and prefill-decode (PD) routing modes.

use crate::core::Worker;
use std::fmt::Debug;

mod cache_aware;
mod factory;
mod fixed_probability;
mod marginal_utility;
mod marginal_utility_recorder;
mod metrics;
mod power_of_two;
mod random;
mod round_robin;
mod shortest_queue;

pub use cache_aware::{CacheAwareConfig, CacheAwarePolicy};
pub use factory::PolicyFactory;
pub use fixed_probability::{FixedProbabilityConfig, FixedProbabilityPolicy};
pub use marginal_utility::{MarginalUtilityConfig, MarginalUtilityPolicy};
pub use marginal_utility_recorder::{MarginalUtilityRecorderConfig, MarginalUtilityRecorderPolicy};
pub use metrics::RequestMetrics;
pub use power_of_two::PowerOfTwoPolicy;
pub use random::RandomPolicy;
pub use round_robin::RoundRobinPolicy;
pub use shortest_queue::{ShortestQueueConfig, ShortestQueuePolicy};

/// Core trait for load balancing policies
///
/// This trait provides a unified interface for implementing routing algorithms
/// that can work with both regular single-worker selection and PD dual-worker selection.
pub trait LoadBalancingPolicy: Send + Sync + Debug {
    /// Select a single worker from the available workers
    ///
    /// This is used for regular routing mode where requests go to a single worker.
    fn select_worker(
        &self,
        workers: &[Box<dyn Worker>],
        request_text: Option<&str>,
    ) -> Option<usize>;

    /// Select a pair of workers (prefill and decode) for PD routing
    ///
    /// Returns indices of (prefill_worker, decode_worker) from their respective arrays.
    /// Default implementation uses select_worker for each array independently.
    fn select_worker_pair(
        &self,
        prefill_workers: &[Box<dyn Worker>],
        decode_workers: &[Box<dyn Worker>],
        request_text: Option<&str>,
    ) -> Option<(usize, usize)> {
        // Default implementation: independently select from each pool
        let prefill_idx = self.select_worker(prefill_workers, request_text)?;
        let decode_idx = self.select_worker(decode_workers, request_text)?;
        Some((prefill_idx, decode_idx))
    }

    /// Update policy state after request completion
    ///
    /// This is called when a request completes (successfully or not) to allow
    /// policies to update their internal state.
    fn on_request_complete(&self, _worker_url: &str, _success: bool) {
        // Default: no-op for stateless policies
    }

    /// Get policy name for metrics and debugging
    fn name(&self) -> &'static str;

    /// Update worker load information
    ///
    /// This is called periodically with current load information for load-aware policies.
    fn update_loads(&self, _loads: &std::collections::HashMap<String, isize>) {
        // Default: no-op for policies that don't use load information
    }

    /// Reset any internal state
    ///
    /// This is useful for policies that maintain state (e.g., round-robin counters).
    fn reset(&self) {
        // Default: no-op for stateless policies
    }

    /// Get as Any for downcasting
    fn as_any(&self) -> &dyn std::any::Any;
}

/// Extended trait for policies that can use detailed request metrics
///
/// This trait extends LoadBalancingPolicy to support richer performance data
/// while maintaining backward compatibility with existing policies.
pub trait LoadBalancingPolicyV2: LoadBalancingPolicy {
    /// Called when a request completes with detailed metrics
    ///
    /// Default implementation delegates to the legacy method for compatibility
    fn on_request_complete_v2(&self, metrics: &RequestMetrics) {
        let (worker_url, success) = metrics.to_legacy_params();
        self.on_request_complete(worker_url, success);
    }
}

// Automatically implement LoadBalancingPolicyV2 for all LoadBalancingPolicy types
impl<T: LoadBalancingPolicy + ?Sized> LoadBalancingPolicyV2 for T {}


/// Helper function to filter healthy workers and return their indices
pub(crate) fn get_healthy_worker_indices(workers: &[Box<dyn Worker>]) -> Vec<usize> {
    workers
        .iter()
        .enumerate()
        .filter(|(_, w)| w.is_healthy())
        .map(|(idx, _)| idx)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BasicWorker, WorkerType};

    #[test]
    fn test_get_healthy_worker_indices() {
        let workers: Vec<Box<dyn Worker>> = vec![
            Box::new(BasicWorker::new(
                "http://w1:8000".to_string(),
                WorkerType::Regular,
            )),
            Box::new(BasicWorker::new(
                "http://w2:8000".to_string(),
                WorkerType::Regular,
            )),
            Box::new(BasicWorker::new(
                "http://w3:8000".to_string(),
                WorkerType::Regular,
            )),
        ];

        // All healthy initially
        let indices = get_healthy_worker_indices(&workers);
        assert_eq!(indices, vec![0, 1, 2]);

        // Mark one unhealthy
        workers[1].set_healthy(false);
        let indices = get_healthy_worker_indices(&workers);
        assert_eq!(indices, vec![0, 2]);
    }
}
