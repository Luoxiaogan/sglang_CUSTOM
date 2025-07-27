use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock};
use tracing::{debug, info};

/// Status of a tracked request
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum RequestStatus {
    Routed,
    Processing,
    Completed,
    Failed,
}

/// Information about a single request's routing
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RequestTrace {
    pub request_id: String,
    pub worker_url: String,
    pub node_id: String, // e.g., "gpu_0", "worker_0"
    pub timestamp: DateTime<Utc>,
    pub routing_policy: String,
    pub status: RequestStatus,
    pub route: String, // e.g., "/generate", "/v1/chat/completions"
    // Optional fields
    pub cache_hit: Option<bool>,
    pub input_tokens: Option<usize>,
    pub completion_time: Option<DateTime<Utc>>,
    pub error_message: Option<String>,
}

/// Configuration for request tracking
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RequestTrackingConfig {
    pub enabled: bool,
    pub max_entries: usize,
    pub ttl_seconds: i64,
    pub cleanup_interval_seconds: u64,
}

impl Default for RequestTrackingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_entries: 100_000,
            ttl_seconds: 3600, // 1 hour
            cleanup_interval_seconds: 60,
        }
    }
}

/// Request tracker that stores routing information
#[derive(Debug)]
pub struct RequestTracker {
    traces: Arc<RwLock<HashMap<String, RequestTrace>>>,
    trace_order: Arc<RwLock<VecDeque<String>>>, // For LRU eviction
    config: RequestTrackingConfig,
    last_cleanup: Arc<RwLock<DateTime<Utc>>>,
}

impl RequestTracker {
    pub fn new(config: RequestTrackingConfig) -> Self {
        info!(
            "Initializing request tracker with max_entries={}, ttl={}s",
            config.max_entries, config.ttl_seconds
        );

        Self {
            traces: Arc::new(RwLock::new(HashMap::new())),
            trace_order: Arc::new(RwLock::new(VecDeque::new())),
            config,
            last_cleanup: Arc::new(RwLock::new(Utc::now())),
        }
    }

    /// Record a new request trace
    pub fn record_trace(&self, trace: RequestTrace) {
        if !self.config.enabled {
            return;
        }

        let request_id = trace.request_id.clone();
        debug!("Recording trace for request {}", request_id);

        // Check if cleanup is needed
        self.maybe_cleanup();

        // Add trace
        {
            let mut traces = self.traces.write().unwrap();
            let mut order = self.trace_order.write().unwrap();

            // If we're at capacity, remove oldest
            if traces.len() >= self.config.max_entries {
                if let Some(oldest_id) = order.pop_front() {
                    traces.remove(&oldest_id);
                    debug!("Evicted oldest trace: {}", oldest_id);
                }
            }

            // Add new trace
            traces.insert(request_id.clone(), trace);
            order.push_back(request_id);
        }
    }

    /// Update the status of an existing trace
    pub fn update_trace_status(
        &self,
        request_id: &str,
        status: RequestStatus,
        error_message: Option<String>,
    ) {
        if !self.config.enabled {
            return;
        }

        let mut traces = self.traces.write().unwrap();
        if let Some(trace) = traces.get_mut(request_id) {
            trace.status = status;
            trace.completion_time = Some(Utc::now());
            if let Some(error) = error_message {
                trace.error_message = Some(error);
            }
            debug!("Updated trace status for request {}: {:?}", request_id, trace.status);
        }
    }

    /// Get a single trace by request ID
    pub fn get_trace(&self, request_id: &str) -> Option<RequestTrace> {
        self.traces.read().unwrap().get(request_id).cloned()
    }

    /// Get multiple traces by request IDs
    pub fn get_traces_batch(&self, request_ids: &[String]) -> HashMap<String, RequestTrace> {
        let traces = self.traces.read().unwrap();
        request_ids
            .iter()
            .filter_map(|id| traces.get(id).map(|trace| (id.clone(), trace.clone())))
            .collect()
    }

    /// Get recent traces with optional filtering
    pub fn get_recent_traces(
        &self,
        limit: usize,
        node_id: Option<&str>,
        status: Option<RequestStatus>,
    ) -> Vec<RequestTrace> {
        let traces = self.traces.read().unwrap();
        let order = self.trace_order.read().unwrap();

        let result: Vec<RequestTrace> = order
            .iter()
            .rev() // Most recent first
            .filter_map(|id| traces.get(id))
            .filter(|trace| {
                // Apply filters
                if let Some(nid) = node_id {
                    if trace.node_id != nid {
                        return false;
                    }
                }
                if let Some(s) = &status {
                    if &trace.status != s {
                        return false;
                    }
                }
                true
            })
            .take(limit)
            .cloned()
            .collect();

        result
    }

    /// Get statistics about tracked requests
    pub fn get_stats(&self) -> serde_json::Value {
        let traces = self.traces.read().unwrap();

        let mut node_counts: HashMap<String, usize> = HashMap::new();
        let mut status_counts: HashMap<String, usize> = HashMap::new();
        let mut route_counts: HashMap<String, usize> = HashMap::new();
        let mut oldest_time = None;
        let mut newest_time = None;

        for trace in traces.values() {
            // Count by node
            *node_counts.entry(trace.node_id.clone()).or_insert(0) += 1;

            // Count by status
            let status_str = format!("{:?}", trace.status);
            *status_counts.entry(status_str).or_insert(0) += 1;

            // Count by route
            *route_counts.entry(trace.route.clone()).or_insert(0) += 1;

            // Track time range
            match oldest_time {
                None => oldest_time = Some(trace.timestamp),
                Some(t) if trace.timestamp < t => oldest_time = Some(trace.timestamp),
                _ => {}
            }
            match newest_time {
                None => newest_time = Some(trace.timestamp),
                Some(t) if trace.timestamp > t => newest_time = Some(trace.timestamp),
                _ => {}
            }
        }

        serde_json::json!({
            "total_traces": traces.len(),
            "node_distribution": node_counts,
            "status_distribution": status_counts,
            "route_distribution": route_counts,
            "oldest_trace": oldest_time,
            "newest_trace": newest_time,
            "config": {
                "max_entries": self.config.max_entries,
                "ttl_seconds": self.config.ttl_seconds,
                "enabled": self.config.enabled,
            }
        })
    }

    /// Cleanup old traces based on TTL
    fn maybe_cleanup(&self) {
        let should_cleanup = {
            let last_cleanup = self.last_cleanup.read().unwrap();
            Utc::now() - *last_cleanup > Duration::seconds(self.config.cleanup_interval_seconds as i64)
        };

        if !should_cleanup {
            return;
        }

        let cutoff = Utc::now() - Duration::seconds(self.config.ttl_seconds);
        let mut traces = self.traces.write().unwrap();
        let mut order = self.trace_order.write().unwrap();
        let mut removed = 0;

        // Remove expired traces
        order.retain(|id| {
            if let Some(trace) = traces.get(id) {
                if trace.timestamp < cutoff {
                    traces.remove(id);
                    removed += 1;
                    false
                } else {
                    true
                }
            } else {
                false
            }
        });

        if removed > 0 {
            info!("Cleaned up {} expired request traces", removed);
        }

        *self.last_cleanup.write().unwrap() = Utc::now();
    }

    /// Clear all traces
    pub fn clear(&self) {
        let mut traces = self.traces.write().unwrap();
        let mut order = self.trace_order.write().unwrap();
        traces.clear();
        order.clear();
        info!("Cleared all request traces");
    }

    /// Get the current number of traces
    pub fn trace_count(&self) -> usize {
        self.traces.read().unwrap().len()
    }
}

/// Helper to convert worker URL to node ID
pub fn worker_url_to_node_id(worker_url: &str, worker_index: usize) -> String {
    // Extract port from URL if possible
    if let Some(port_str) = worker_url.split(':').last() {
        if let Ok(port) = port_str.trim_end_matches('/').parse::<u16>() {
            // Map common port ranges to GPU IDs
            let gpu_id = match port {
                30001..=30010 => port - 30001,
                8000..=8010 => port - 8000,
                _ => worker_index as u16,
            };
            return format!("gpu_{}", gpu_id);
        }
    }
    
    // Fallback to worker index
    format!("worker_{}", worker_index)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_tracker_basic() {
        let config = RequestTrackingConfig {
            enabled: true,
            max_entries: 10,
            ttl_seconds: 3600,
            cleanup_interval_seconds: 60,
        };
        
        let tracker = RequestTracker::new(config);
        
        // Add a trace
        let trace = RequestTrace {
            request_id: "test_123".to_string(),
            worker_url: "http://localhost:30001".to_string(),
            node_id: "gpu_0".to_string(),
            timestamp: Utc::now(),
            routing_policy: "round_robin".to_string(),
            status: RequestStatus::Routed,
            route: "/generate".to_string(),
            cache_hit: None,
            input_tokens: Some(100),
            completion_time: None,
            error_message: None,
        };
        
        tracker.record_trace(trace.clone());
        
        // Verify we can retrieve it
        let retrieved = tracker.get_trace("test_123").unwrap();
        assert_eq!(retrieved.request_id, "test_123");
        assert_eq!(retrieved.node_id, "gpu_0");
    }

    #[test]
    fn test_lru_eviction() {
        let config = RequestTrackingConfig {
            enabled: true,
            max_entries: 3,
            ttl_seconds: 3600,
            cleanup_interval_seconds: 60,
        };
        
        let tracker = RequestTracker::new(config);
        
        // Add 4 traces (one more than max)
        for i in 0..4 {
            let trace = RequestTrace {
                request_id: format!("req_{}", i),
                worker_url: "http://localhost:30001".to_string(),
                node_id: "gpu_0".to_string(),
                timestamp: Utc::now(),
                routing_policy: "round_robin".to_string(),
                status: RequestStatus::Routed,
                route: "/generate".to_string(),
                cache_hit: None,
                input_tokens: None,
                completion_time: None,
                error_message: None,
            };
            tracker.record_trace(trace);
        }
        
        // First request should be evicted
        assert!(tracker.get_trace("req_0").is_none());
        assert!(tracker.get_trace("req_1").is_some());
        assert!(tracker.get_trace("req_2").is_some());
        assert!(tracker.get_trace("req_3").is_some());
    }

    #[test]
    fn test_worker_url_to_node_id() {
        assert_eq!(worker_url_to_node_id("http://localhost:30001", 0), "gpu_0");
        assert_eq!(worker_url_to_node_id("http://localhost:30005", 0), "gpu_4");
        assert_eq!(worker_url_to_node_id("http://0.0.0.0:8001", 0), "gpu_1");
        assert_eq!(worker_url_to_node_id("invalid_url", 5), "worker_5");
    }
}