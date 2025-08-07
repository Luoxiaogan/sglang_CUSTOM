//! Shortest Queue First routing policy
//! Routes requests to the worker with the shortest queue

use super::LoadBalancingPolicy;
use crate::core::Worker;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};
use tracing::{debug, info};

/// Configuration for the Shortest Queue policy
#[derive(Debug, Clone)]
pub struct ShortestQueueConfig {
    /// Whether to use fallback round-robin when no queue data is available
    pub enable_fallback: bool,
}

impl Default for ShortestQueueConfig {
    fn default() -> Self {
        Self {
            enable_fallback: true,
        }
    }
}

/// Shortest Queue First routing policy
#[derive(Debug)]
pub struct ShortestQueuePolicy {
    /// Mapping from worker URL to latest queue length
    worker_queue_lengths: Arc<RwLock<HashMap<String, usize>>>,
    /// Fallback counter for round-robin when no queue data available
    fallback_counter: Arc<AtomicUsize>,
    /// Configuration
    config: ShortestQueueConfig,
}

impl ShortestQueuePolicy {
    /// Create a new Shortest Queue policy with default configuration
    pub fn new() -> Self {
        Self::with_config(ShortestQueueConfig::default())
    }

    /// Create a new Shortest Queue policy with custom configuration
    pub fn with_config(config: ShortestQueueConfig) -> Self {
        info!("Creating ShortestQueuePolicy with config: {:?}", config);
        Self {
            worker_queue_lengths: Arc::new(RwLock::new(HashMap::new())),
            fallback_counter: Arc::new(AtomicUsize::new(0)),
            config,
        }
    }

    /// Update queue length for a worker
    pub fn update_queue_length(&self, worker_url: &str, queue_length: usize) {
        let mut queue_lengths = self.worker_queue_lengths.write().unwrap();
        queue_lengths.insert(worker_url.to_string(), queue_length);
        debug!(
            "Updated queue length for {}: {}",
            worker_url, queue_length
        );
    }
}

impl LoadBalancingPolicy for ShortestQueuePolicy {
    fn select_worker(
        &self,
        workers: &[Box<dyn Worker>],
        _request_text: Option<&str>,
    ) -> Option<usize> {
        // Get healthy workers
        let healthy_indices: Vec<usize> = workers
            .iter()
            .enumerate()
            .filter(|(_, w)| w.is_healthy())
            .map(|(idx, _)| idx)
            .collect();

        if healthy_indices.is_empty() {
            debug!("No healthy workers available");
            return None;
        }

        let queue_lengths = self.worker_queue_lengths.read().unwrap();

        // Find worker with shortest queue
        let mut best_idx = healthy_indices[0];
        let mut min_queue_length = usize::MAX;
        let mut has_queue_data = false;

        for &idx in &healthy_indices {
            let worker_url = workers[idx].url();
            
            if let Some(&queue_len) = queue_lengths.get(worker_url) {
                has_queue_data = true;
                debug!("Worker {} has queue length: {}", worker_url, queue_len);
                
                if queue_len < min_queue_length {
                    min_queue_length = queue_len;
                    best_idx = idx;
                }
            } else {
                debug!("No queue data for worker {}", worker_url);
            }
        }

        // If no queue data is available and fallback is enabled, use round-robin
        if !has_queue_data && self.config.enable_fallback {
            let counter = self.fallback_counter.fetch_add(1, Ordering::Relaxed);
            best_idx = healthy_indices[counter % healthy_indices.len()];
            debug!(
                "Using fallback round-robin, selected worker at index {}",
                best_idx
            );
        } else if has_queue_data {
            debug!(
                "Selected worker {} with queue length {}",
                workers[best_idx].url(),
                min_queue_length
            );
        }

        Some(best_idx)
    }

    fn select_worker_pair(
        &self,
        prefill_workers: &[Box<dyn Worker>],
        decode_workers: &[Box<dyn Worker>],
        request_text: Option<&str>,
    ) -> Option<(usize, usize)> {
        // For P/D mode, select shortest queue for both
        let prefill_idx = self.select_worker(prefill_workers, request_text)?;
        let decode_idx = self.select_worker(decode_workers, request_text)?;
        Some((prefill_idx, decode_idx))
    }

    fn on_request_complete(&self, _worker_url: &str, _success: bool) {
        // Legacy interface - do nothing
    }

    fn update_loads(&self, _loads: &HashMap<String, isize>) {
        // Not used by this policy
    }

    fn reset(&self) {
        let mut queue_lengths = self.worker_queue_lengths.write().unwrap();
        queue_lengths.clear();
        self.fallback_counter.store(0, Ordering::Relaxed);
        info!("ShortestQueuePolicy reset");
    }

    fn name(&self) -> &'static str {
        "shortest_queue"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

// Note: LoadBalancingPolicyV2 is automatically implemented via the blanket impl in mod.rs
// We handle queue_length updates through a different mechanism (see the actual update call sites)

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::MockWorker;

    fn create_test_workers(count: usize) -> Vec<Box<dyn Worker>> {
        (0..count)
            .map(|i| {
                let mut worker = MockWorker::new();
                worker.expect_url().return_const(format!("http://worker{}", i));
                worker.expect_is_healthy().return_const(true);
                Box::new(worker) as Box<dyn Worker>
            })
            .collect()
    }

    #[test]
    fn test_select_worker_with_queue_data() {
        let policy = ShortestQueuePolicy::new();
        let workers = create_test_workers(3);

        // Set up queue lengths
        policy.update_queue_length("http://worker0", 10);
        policy.update_queue_length("http://worker1", 5);
        policy.update_queue_length("http://worker2", 8);

        // Should select worker1 with shortest queue
        let selected = policy.select_worker(&workers, None);
        assert_eq!(selected, Some(1));
    }

    #[test]
    fn test_select_worker_without_queue_data() {
        let policy = ShortestQueuePolicy::new();
        let workers = create_test_workers(3);

        // No queue data, should use fallback round-robin
        let selected1 = policy.select_worker(&workers, None);
        let selected2 = policy.select_worker(&workers, None);
        let selected3 = policy.select_worker(&workers, None);

        assert!(selected1.is_some());
        assert!(selected2.is_some());
        assert!(selected3.is_some());

        // Check round-robin behavior
        assert_ne!(selected1, selected2);
    }

    #[test]
    fn test_select_worker_partial_queue_data() {
        let policy = ShortestQueuePolicy::new();
        let workers = create_test_workers(3);

        // Only set queue length for some workers
        policy.update_queue_length("http://worker0", 10);
        policy.update_queue_length("http://worker2", 5);

        // Should select worker2 with shortest known queue
        let selected = policy.select_worker(&workers, None);
        assert_eq!(selected, Some(2));
    }

    #[test]
    fn test_update_from_metrics() {
        let policy = ShortestQueuePolicy::new();
        
        // Directly test the update_queue_length method
        // In production, router.rs will call this based on RequestMetrics
        policy.update_queue_length("http://worker1", 7);

        let queue_lengths = policy.worker_queue_lengths.read().unwrap();
        assert_eq!(queue_lengths.get("http://worker1"), Some(&7));
    }

    #[test]
    fn test_reset() {
        let policy = ShortestQueuePolicy::new();
        
        policy.update_queue_length("http://worker0", 10);
        policy.update_queue_length("http://worker1", 5);

        policy.reset();

        let queue_lengths = policy.worker_queue_lengths.read().unwrap();
        assert!(queue_lengths.is_empty());
        assert_eq!(policy.fallback_counter.load(Ordering::Relaxed), 0);
    }
}