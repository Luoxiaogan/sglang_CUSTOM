//! Fixed probability load balancing policy
//! Routes requests based on pre-configured probability distribution

use super::{get_healthy_worker_indices, LoadBalancingPolicy};
use crate::core::Worker;
use rand::Rng;
use tracing::{info, warn};

/// Configuration for fixed probability policy
#[derive(Debug, Clone)]
pub struct FixedProbabilityConfig {
    /// Probability distribution for each worker (must sum to 1.0)
    pub probabilities: Vec<f64>,
}

/// Fixed probability routing policy
///
/// Routes requests to workers based on a pre-configured probability distribution.
/// Each request is routed independently (IID) according to the specified probabilities.
#[derive(Debug)]
pub struct FixedProbabilityPolicy {
    config: FixedProbabilityConfig,
    /// Cumulative probability distribution for efficient sampling
    cumulative_probs: Vec<f64>,
}

impl FixedProbabilityPolicy {
    /// Create a new fixed probability policy
    pub fn new(config: FixedProbabilityConfig) -> Self {
        // Calculate cumulative probability distribution
        let mut cumulative_probs = Vec::with_capacity(config.probabilities.len());
        let mut sum = 0.0;
        
        for prob in &config.probabilities {
            sum += prob;
            cumulative_probs.push(sum);
        }
        
        // Normalize if slightly off due to floating point errors
        if (sum - 1.0).abs() > 1e-9 && !config.probabilities.is_empty() {
            for cum_prob in &mut cumulative_probs {
                *cum_prob /= sum;
            }
        }
        
        info!(
            "Creating FixedProbabilityPolicy with probabilities: {:?}, cumulative: {:?}",
            config.probabilities, cumulative_probs
        );
        
        Self {
            config,
            cumulative_probs,
        }
    }
    
    /// Create with configuration
    pub fn with_config(config: FixedProbabilityConfig) -> Self {
        Self::new(config)
    }
}

impl LoadBalancingPolicy for FixedProbabilityPolicy {
    fn select_worker(
        &self,
        workers: &[Box<dyn Worker>],
        _request_text: Option<&str>,
    ) -> Option<usize> {
        let healthy_indices = get_healthy_worker_indices(workers);
        
        if healthy_indices.is_empty() {
            warn!("FixedProbabilityPolicy: No healthy workers available");
            return None;
        }
        
        // Check if we have probabilities for all workers
        if self.config.probabilities.len() != workers.len() {
            warn!(
                "FixedProbabilityPolicy: Probability count {} doesn't match worker count {}, using first healthy worker",
                self.config.probabilities.len(),
                workers.len()
            );
            return Some(healthy_indices[0]);
        }
        
        // Generate random value in [0, 1)
        let mut rng = rand::thread_rng();
        let rand_val: f64 = rng.gen();
        
        // Find the corresponding worker using cumulative probabilities
        for (i, &cum_prob) in self.cumulative_probs.iter().enumerate() {
            if rand_val < cum_prob {
                // Check if this worker is healthy
                if healthy_indices.contains(&i) {
                    info!(
                        "FixedProbabilityPolicy: Selected worker {} (prob={:.3}, rand={:.3})",
                        i, self.config.probabilities[i], rand_val
                    );
                    return Some(i);
                } else {
                    // Worker is unhealthy, need to redistribute
                    // For simplicity, select randomly from healthy workers
                    warn!(
                        "FixedProbabilityPolicy: Selected worker {} is unhealthy, falling back to random healthy worker",
                        i
                    );
                    let fallback_idx = rng.gen_range(0..healthy_indices.len());
                    return Some(healthy_indices[fallback_idx]);
                }
            }
        }
        
        // Should not reach here, but provide fallback
        warn!("FixedProbabilityPolicy: Failed to select worker, using first healthy worker");
        Some(healthy_indices[0])
    }
    
    fn select_worker_pair(
        &self,
        prefill_workers: &[Box<dyn Worker>],
        decode_workers: &[Box<dyn Worker>],
        request_text: Option<&str>,
    ) -> Option<(usize, usize)> {
        // For P/D mode, select independently for prefill and decode
        let prefill_idx = self.select_worker(prefill_workers, request_text)?;
        let decode_idx = self.select_worker(decode_workers, request_text)?;
        Some((prefill_idx, decode_idx))
    }
    
    fn on_request_complete(&self, _worker_url: &str, _success: bool) {
        // No state to update for fixed probability
    }
    
    fn update_loads(&self, _loads: &std::collections::HashMap<String, isize>) {
        // Not used by this policy
    }
    
    fn reset(&self) {
        // No state to reset
        info!("FixedProbabilityPolicy reset (no-op)");
    }
    
    fn name(&self) -> &'static str {
        "fixed_probability"
    }
    
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::MockWorker;
    use std::collections::HashMap;
    
    #[test]
    fn test_fixed_probability_distribution() {
        // Test with 3 workers: 50%, 30%, 20%
        let config = FixedProbabilityConfig {
            probabilities: vec![0.5, 0.3, 0.2],
        };
        
        let policy = FixedProbabilityPolicy::new(config);
        let workers: Vec<Box<dyn Worker>> = vec![
            Box::new(MockWorker::new("http://w1:8000".to_string())),
            Box::new(MockWorker::new("http://w2:8000".to_string())),
            Box::new(MockWorker::new("http://w3:8000".to_string())),
        ];
        
        // Run many selections to verify distribution
        let mut counts = HashMap::new();
        let iterations = 10000;
        
        for _ in 0..iterations {
            if let Some(idx) = policy.select_worker(&workers, None) {
                *counts.entry(idx).or_insert(0) += 1;
            }
        }
        
        // Check that distribution roughly matches configured probabilities
        let tolerance = 0.02; // 2% tolerance
        
        assert_eq!(counts.len(), 3, "All workers should be selected");
        
        let freq_0 = counts.get(&0).unwrap_or(&0) as &f64 / iterations as f64;
        let freq_1 = counts.get(&1).unwrap_or(&0) as &f64 / iterations as f64;
        let freq_2 = counts.get(&2).unwrap_or(&0) as &f64 / iterations as f64;
        
        assert!((freq_0 - 0.5).abs() < tolerance, "Worker 0 frequency {} should be close to 0.5", freq_0);
        assert!((freq_1 - 0.3).abs() < tolerance, "Worker 1 frequency {} should be close to 0.3", freq_1);
        assert!((freq_2 - 0.2).abs() < tolerance, "Worker 2 frequency {} should be close to 0.2", freq_2);
    }
    
    #[test]
    fn test_with_unhealthy_workers() {
        let config = FixedProbabilityConfig {
            probabilities: vec![0.7, 0.3], // 70%, 30%
        };
        
        let policy = FixedProbabilityPolicy::new(config);
        let workers: Vec<Box<dyn Worker>> = vec![
            Box::new(MockWorker::new("http://w1:8000".to_string())),
            Box::new(MockWorker::new("http://w2:8000".to_string())),
        ];
        
        // Mark first worker as unhealthy
        workers[0].set_healthy(false);
        
        // Should always select the healthy worker (index 1)
        for _ in 0..10 {
            assert_eq!(policy.select_worker(&workers, None), Some(1));
        }
    }
    
    #[test]
    fn test_no_healthy_workers() {
        let config = FixedProbabilityConfig {
            probabilities: vec![1.0],
        };
        
        let policy = FixedProbabilityPolicy::new(config);
        let workers: Vec<Box<dyn Worker>> = vec![
            Box::new(MockWorker::new("http://w1:8000".to_string())),
        ];
        
        workers[0].set_healthy(false);
        assert_eq!(policy.select_worker(&workers, None), None);
    }
    
    #[test]
    fn test_cumulative_distribution() {
        let config = FixedProbabilityConfig {
            probabilities: vec![0.25, 0.25, 0.25, 0.25],
        };
        
        let policy = FixedProbabilityPolicy::new(config);
        
        // Check cumulative probabilities
        assert_eq!(policy.cumulative_probs.len(), 4);
        assert!((policy.cumulative_probs[0] - 0.25).abs() < 1e-9);
        assert!((policy.cumulative_probs[1] - 0.50).abs() < 1e-9);
        assert!((policy.cumulative_probs[2] - 0.75).abs() < 1e-9);
        assert!((policy.cumulative_probs[3] - 1.00).abs() < 1e-9);
    }
}