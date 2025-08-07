//! Request metrics for performance-aware routing decisions

use serde::{Deserialize, Serialize};

/// Performance metrics collected from request completion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestMetrics {
    /// Worker URL that processed this request
    pub worker_url: String,
    
    /// Unique request identifier
    pub request_id: String,
    
    /// Server-side timestamps (Unix timestamps in seconds)
    pub server_created_time: Option<f64>,
    pub server_first_token_time: Option<f64>,
    pub queue_time_start: Option<f64>,
    pub queue_time_end: Option<f64>,
    
    /// Request completion time (Unix timestamp)
    pub finish_time: f64,
    
    /// Latency metrics (in seconds)
    pub server_latency: f64,
    pub total_latency: f64,
    
    /// Token counts
    pub actual_prompt_tokens: Option<usize>,
    pub actual_output_tokens: Option<usize>,
    pub actual_total_tokens: Option<usize>,
    
    /// Cached tokens (for cache-aware routing)
    pub cached_tokens: Option<usize>,
    
    /// Queue length information for routing decisions
    pub queue_length: Option<usize>,
}

impl RequestMetrics {
    /// Convert to legacy parameters for backward compatibility
    pub fn to_legacy_params(&self) -> (&str, bool) {
        // Consider request successful if latency is under 10 seconds
        (&self.worker_url, self.server_latency < 10.0)
    }
    
    /// Calculate decode token throughput (tokens per second)
    pub fn decode_throughput(&self) -> Option<f64> {
        match (self.actual_output_tokens, self.server_latency) {
            (Some(tokens), latency) if latency > 0.0 => Some(tokens as f64 / latency),
            _ => None,
        }
    }
    
    /// Get queue time if available
    pub fn queue_time(&self) -> Option<f64> {
        match (self.queue_time_start, self.queue_time_end) {
            (Some(start), Some(end)) => Some(end - start),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_legacy_params_conversion() {
        let metrics = RequestMetrics {
            worker_url: "http://worker1:8000".to_string(),
            request_id: "test-123".to_string(),
            server_created_time: Some(1000.0),
            server_first_token_time: Some(1000.5),
            queue_time_start: Some(1000.1),
            queue_time_end: Some(1000.2),
            finish_time: 1001.0,
            server_latency: 1.0,
            total_latency: 1.5,
            actual_prompt_tokens: Some(100),
            actual_output_tokens: Some(50),
            actual_total_tokens: Some(150),
            cached_tokens: Some(20),
            queue_length: None,
        };
        
        let (url, success) = metrics.to_legacy_params();
        assert_eq!(url, "http://worker1:8000");
        assert!(success);
    }
    
    #[test]
    fn test_decode_throughput() {
        let metrics = RequestMetrics {
            worker_url: "http://worker1:8000".to_string(),
            request_id: "test-123".to_string(),
            server_created_time: None,
            server_first_token_time: None,
            queue_time_start: None,
            queue_time_end: None,
            finish_time: 1001.0,
            server_latency: 2.0,
            total_latency: 2.5,
            actual_prompt_tokens: Some(100),
            actual_output_tokens: Some(100),
            actual_total_tokens: Some(200),
            cached_tokens: None,
            queue_length: None,
        };
        
        assert_eq!(metrics.decode_throughput(), Some(50.0));
    }
    
    #[test]
    fn test_queue_time() {
        let mut metrics = RequestMetrics {
            worker_url: "http://worker1:8000".to_string(),
            request_id: "test-123".to_string(),
            server_created_time: None,
            server_first_token_time: None,
            queue_time_start: Some(1000.1),
            queue_time_end: Some(1000.3),
            finish_time: 1001.0,
            server_latency: 0.9,
            total_latency: 1.0,
            actual_prompt_tokens: None,
            actual_output_tokens: None,
            actual_total_tokens: None,
            cached_tokens: None,
            queue_length: None,
        };
        
        assert!((metrics.queue_time().unwrap() - 0.2).abs() < 0.0001);
        
        // Test with missing queue times
        metrics.queue_time_start = None;
        assert!(metrics.queue_time().is_none());
    }
}