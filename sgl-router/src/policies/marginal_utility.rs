//! Marginal utility routing policy for SGLang router
//!
//! This policy routes requests based on performance trends, selecting workers
//! that show the best marginal improvement in throughput or latency.

use super::{get_healthy_worker_indices, LoadBalancingPolicy, RequestMetrics};
use crate::core::Worker;
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};
use tracing::{debug, info};

/// Configuration for marginal utility policy
#[derive(Debug, Clone)]
pub struct MarginalUtilityConfig {
    /// Size of the sliding window for historical metrics
    pub window_size: usize,
    /// Minimum history size needed for trend analysis
    pub min_history_for_trend: usize,
    /// Weight for throughput gradient (0.0 to 1.0)
    pub throughput_weight: f64,
    /// Weight for latency gradient (0.0 to 1.0)
    pub latency_weight: f64,
}

impl Default for MarginalUtilityConfig {
    fn default() -> Self {
        Self {
            window_size: 20,
            min_history_for_trend: 10,
            throughput_weight: 0.6,
            latency_weight: 0.4,
        }
    }
}

/// Worker state tracking for marginal utility calculations
#[derive(Debug)]
struct WorkerState {
    /// Worker URL
    url: String,
    /// Historical metrics (sliding window)
    history: VecDeque<RequestMetrics>,
    /// Number of outstanding requests
    outstanding_requests: AtomicUsize,
}

impl WorkerState {
    fn new(url: String, window_size: usize) -> Self {
        Self {
            url,
            history: VecDeque::with_capacity(window_size),
            outstanding_requests: AtomicUsize::new(0),
        }
    }
}

/// Marginal utility load balancing policy
#[derive(Debug)]
pub struct MarginalUtilityPolicy {
    /// Worker states indexed by URL
    workers_state: Arc<RwLock<HashMap<String, WorkerState>>>,
    /// Policy configuration
    config: MarginalUtilityConfig,
}

impl MarginalUtilityPolicy {
    /// Create a new marginal utility policy
    pub fn new(config: MarginalUtilityConfig) -> Self {
        info!("Creating MarginalUtilityPolicy with config: {:?}", config);
        Self {
            workers_state: Arc::new(RwLock::new(HashMap::new())),
            config,
        }
    }

    /// Calculate average throughput and latency for a set of metrics
    fn calculate_performance_metrics(history: &[RequestMetrics]) -> (f64, f64) {
        if history.is_empty() {
            return (0.0, 0.0);
        }

        // Calculate total tokens and time span
        let total_tokens: usize = history
            .iter()
            .filter_map(|m| m.actual_output_tokens)
            .sum();
        
        let time_span = if history.len() > 1 {
            history.last().unwrap().finish_time - history.first().unwrap().finish_time
        } else {
            1.0 // Avoid division by zero
        };
        
        let throughput = if time_span > 0.0 {
            total_tokens as f64 / time_span
        } else {
            0.0
        };

        // Calculate average latency
        let avg_latency = history.iter()
            .map(|m| m.server_latency)
            .sum::<f64>() / history.len() as f64;

        (throughput, avg_latency)
    }

    /// Calculate gradient-based score for a worker
    fn calculate_gradient_score(
        &self,
        state: &WorkerState,
    ) -> Option<f64> {
        let history: Vec<_> = state.history.iter().cloned().collect();
        
        if history.len() < self.config.min_history_for_trend {
            return None;
        }

        // Split history into two halves
        let mid = history.len() / 2;
        let h1 = &history[..mid];
        let h2 = &history[mid..];

        // Calculate performance metrics for each half
        let (t1, l1) = Self::calculate_performance_metrics(h1);
        let (t2, l2) = Self::calculate_performance_metrics(h2);

        // Calculate time delta
        let dt = if h2.last().unwrap().finish_time > h1.last().unwrap().finish_time {
            h2.last().unwrap().finish_time - h1.last().unwrap().finish_time
        } else {
            1.0 // Avoid division by zero
        };

        // Calculate gradients
        let grad_t = (t2 - t1) / dt;
        let grad_l = (l2 - l1) / dt;

        // Calculate score: higher throughput gradient is better, lower latency gradient is better
        let score = self.config.throughput_weight * grad_t - self.config.latency_weight * grad_l;
        
        debug!(
            "Worker {} gradient analysis: throughput_grad={:.3}, latency_grad={:.3}, score={:.3}",
            state.url, grad_t, grad_l, score
        );
        
        Some(score)
    }
}

impl LoadBalancingPolicy for MarginalUtilityPolicy {
    fn select_worker(
        &self,
        workers: &[Box<dyn Worker>],
        _request_text: Option<&str>,
    ) -> Option<usize> {
        let healthy_indices = get_healthy_worker_indices(workers);
        if healthy_indices.is_empty() {
            return None;
        }

        let states = self.workers_state.read().unwrap();
        let mut best_score = f64::NEG_INFINITY;
        let mut best_idx = healthy_indices[0];
        let mut used_gradient = false;

        for &idx in &healthy_indices {
            let worker = &workers[idx];
            let worker_url = worker.url();

            let score = if let Some(state) = states.get(worker_url) {
                // Try gradient-based scoring first
                if let Some(gradient_score) = self.calculate_gradient_score(state) {
                    used_gradient = true;
                    gradient_score
                } else {
                    // Fallback to load-based scoring (negative load = higher score)
                    debug!(
                        "Worker {} using load-based scoring: {} outstanding requests",
                        worker_url,
                        state.outstanding_requests.load(Ordering::Relaxed)
                    );
                    -(state.outstanding_requests.load(Ordering::Relaxed) as f64)
                }
            } else {
                // New worker - prefer it slightly
                debug!("Worker {} is new, giving neutral score", worker_url);
                0.0
            };

            if score > best_score {
                best_score = score;
                best_idx = idx;
            }
        }

        let selected_url = workers[best_idx].url();
        debug!(
            "Selected worker {} with score {:.3} (gradient={})",
            selected_url, best_score, used_gradient
        );

        // Update outstanding requests for selected worker
        drop(states); // Release read lock
        let mut states = self.workers_state.write().unwrap();
        
        // Create state if it doesn't exist
        let state = states.entry(selected_url.to_string()).or_insert_with(|| {
            WorkerState::new(selected_url.to_string(), self.config.window_size)
        });
        
        state.outstanding_requests.fetch_add(1, Ordering::Relaxed);

        Some(best_idx)
    }

    fn select_worker_pair(
        &self,
        prefill_workers: &[Box<dyn Worker>],
        decode_workers: &[Box<dyn Worker>],
        request_text: Option<&str>,
    ) -> Option<(usize, usize)> {
        // For PD mode, independently select best workers from each pool
        let prefill_idx = self.select_worker(prefill_workers, request_text)?;
        let decode_idx = self.select_worker(decode_workers, request_text)?;
        Some((prefill_idx, decode_idx))
    }

    fn on_request_complete(&self, worker_url: &str, success: bool) {
        // Legacy interface - just update outstanding count
        if let Ok(states) = self.workers_state.read() {
            if let Some(state) = states.get(worker_url) {
                if state.outstanding_requests.load(Ordering::Relaxed) > 0 {
                    state.outstanding_requests.fetch_sub(1, Ordering::Relaxed);
                }
            }
        }
        
        if !success {
            debug!("Request to {} failed", worker_url);
        }
    }

    fn name(&self) -> &'static str {
        "marginal_utility"
    }

    fn reset(&self) {
        let mut states = self.workers_state.write().unwrap();
        states.clear();
        info!("Reset marginal utility policy state");
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl MarginalUtilityPolicy {
    /// Handle request completion with detailed metrics
    pub fn handle_request_metrics(&self, metrics: &RequestMetrics) {
        let mut states = self.workers_state.write().unwrap();
        
        let state = states.entry(metrics.worker_url.clone()).or_insert_with(|| {
            WorkerState::new(metrics.worker_url.clone(), self.config.window_size)
        });
        
        // Add new metrics to history
        state.history.push_back(metrics.clone());
        
        // Maintain window size
        if state.history.len() > self.config.window_size {
            state.history.pop_front();
        }
        
        // Decrement outstanding requests
        if state.outstanding_requests.load(Ordering::Relaxed) > 0 {
            state.outstanding_requests.fetch_sub(1, Ordering::Relaxed);
        }
        
        debug!(
            "Updated metrics for {}: history_len={}, outstanding={}, throughput={:?}",
            metrics.worker_url,
            state.history.len(),
            state.outstanding_requests.load(Ordering::Relaxed),
            metrics.decode_throughput()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::BasicWorker;

    fn create_test_metrics(
        worker_url: &str,
        finish_time: f64,
        server_latency: f64,
        output_tokens: usize,
    ) -> RequestMetrics {
        RequestMetrics {
            worker_url: worker_url.to_string(),
            request_id: "test".to_string(),
            server_created_time: Some(finish_time - server_latency),
            server_first_token_time: None,
            queue_time_start: None,
            queue_time_end: None,
            finish_time,
            server_latency,
            total_latency: server_latency,
            actual_prompt_tokens: Some(100),
            actual_output_tokens: Some(output_tokens),
            actual_total_tokens: Some(100 + output_tokens),
            cached_tokens: None,
        }
    }

    #[test]
    fn test_new_worker_selection() {
        let config = MarginalUtilityConfig::default();
        let policy = MarginalUtilityPolicy::new(config);
        
        let workers: Vec<Box<dyn Worker>> = vec![
            Box::new(BasicWorker::new("http://w1:8000".to_string())),
            Box::new(BasicWorker::new("http://w2:8000".to_string())),
        ];
        
        // Should select from available workers
        let selected = policy.select_worker(&workers, None);
        assert!(selected.is_some());
        assert!(selected.unwrap() < workers.len());
    }

    #[test]
    fn test_gradient_calculation() {
        let config = MarginalUtilityConfig {
            window_size: 10,
            min_history_for_trend: 4,
            throughput_weight: 0.6,
            latency_weight: 0.4,
        };
        let policy = MarginalUtilityPolicy::new(config);
        
        // Simulate improving performance over time
        let worker_url = "http://w1:8000";
        for i in 0..10 {
            let time = 1000.0 + i as f64;
            let latency = 2.0 - (i as f64 * 0.1); // Decreasing latency
            let tokens = 100 + i * 10; // Increasing throughput
            
            let metrics = create_test_metrics(worker_url, time, latency, tokens);
            policy.on_request_complete_v2(&metrics);
        }
        
        // Check that history is maintained
        let states = policy.workers_state.read().unwrap();
        let state = states.get(worker_url).unwrap();
        assert_eq!(state.history.len(), 10);
        
        // Calculate gradient score
        let score = policy.calculate_gradient_score(state);
        assert!(score.is_some());
        // Score should be positive due to improving performance
        assert!(score.unwrap() > 0.0);
    }

    #[test]
    fn test_fallback_to_load_balancing() {
        let config = MarginalUtilityConfig {
            window_size: 20,
            min_history_for_trend: 10,
            throughput_weight: 0.6,
            latency_weight: 0.4,
        };
        let policy = MarginalUtilityPolicy::new(config);
        
        let workers: Vec<Box<dyn Worker>> = vec![
            Box::new(BasicWorker::new("http://w1:8000".to_string())),
            Box::new(BasicWorker::new("http://w2:8000".to_string())),
        ];
        
        // Add some history to w1 but not enough for gradient
        for i in 0..5 {
            let metrics = create_test_metrics("http://w1:8000", 1000.0 + i as f64, 1.0, 100);
            policy.on_request_complete_v2(&metrics);
        }
        
        // Select multiple times - should use load balancing
        let mut selections = HashMap::new();
        for _ in 0..20 {
            if let Some(idx) = policy.select_worker(&workers, None) {
                *selections.entry(idx).or_insert(0) += 1;
            }
        }
        
        // Both workers should be selected
        assert!(selections.len() > 1);
    }

    #[test]
    fn test_pd_mode_selection() {
        let config = MarginalUtilityConfig::default();
        let policy = MarginalUtilityPolicy::new(config);
        
        let prefill_workers: Vec<Box<dyn Worker>> = vec![
            Box::new(BasicWorker::new("http://p1:8000".to_string())),
            Box::new(BasicWorker::new("http://p2:8000".to_string())),
        ];
        
        let decode_workers: Vec<Box<dyn Worker>> = vec![
            Box::new(BasicWorker::new("http://d1:8000".to_string())),
            Box::new(BasicWorker::new("http://d2:8000".to_string())),
        ];
        
        let selected = policy.select_worker_pair(&prefill_workers, &decode_workers, None);
        assert!(selected.is_some());
        let (p_idx, d_idx) = selected.unwrap();
        assert!(p_idx < prefill_workers.len());
        assert!(d_idx < decode_workers.len());
    }
}