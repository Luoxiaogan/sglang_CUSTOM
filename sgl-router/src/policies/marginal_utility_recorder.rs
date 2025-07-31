//! Marginal utility routing policy with metrics recording
//!
//! This policy extends the MarginalUtilityPolicy to record routing decisions
//! and performance metrics to CSV files for analysis.

use super::{
    get_healthy_worker_indices, LoadBalancingPolicy, MarginalUtilityConfig, MarginalUtilityPolicy,
    RequestMetrics,
};
use crate::core::Worker;
use chrono::Local;
use serde::Serialize;
use std::fs::{create_dir_all, File, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use tracing::{debug, error, info};

/// Configuration for marginal utility recorder policy
#[derive(Debug, Clone)]
pub struct MarginalUtilityRecorderConfig {
    /// Base marginal utility configuration
    pub base_config: MarginalUtilityConfig,
    /// Directory to save CSV metrics
    pub output_dir: String,
    /// Buffer size for batch writing
    pub buffer_size: usize,
    /// Flush interval in seconds
    pub flush_interval_secs: u64,
}

/// Record of a routing decision
#[derive(Debug, Clone, Serialize)]
struct RoutingDecisionRecord {
    timestamp: String,
    worker_url: String,
    request_id: String,
    throughput_gradient: Option<f64>,
    latency_gradient: Option<f64>,
    score: Option<f64>,
    outstanding_requests: usize,
    avg_throughput: Option<f64>,
    avg_latency: Option<f64>,
    window_size: usize,
    selection_reason: String,
    actual_output_tokens: Option<usize>,
    server_latency: Option<f64>,
    queue_time: Option<f64>,
    ttft: Option<f64>,
    decision_time_ms: f64,
}

/// CSV writer with buffering
#[derive(Debug)] //CsvWriter 缺少 Debug，要么给它加，要么去掉外层 Debug
struct CsvWriter {
    file_path: PathBuf,
    buffer: Vec<RoutingDecisionRecord>,
    buffer_size: usize,
    last_flush: Instant,
    flush_interval: Duration,
}

impl CsvWriter {
    fn new(output_dir: &str, buffer_size: usize, flush_interval_secs: u64) -> Self {
        create_dir_all(output_dir).ok();
        
        let timestamp = Local::now().format("%Y%m%d_%H%M%S");
        let file_name = format!("marginal_utility_metrics_{}.csv", timestamp);
        let file_path = PathBuf::from(output_dir).join(file_name);
        
        // Write header if file doesn't exist
        if !file_path.exists() {
            if let Ok(mut file) = File::create(&file_path) {
                writeln!(
                    file,
                    "timestamp,worker_url,request_id,throughput_gradient,latency_gradient,\
                    score,outstanding_requests,avg_throughput,avg_latency,window_size,\
                    selection_reason,actual_output_tokens,server_latency,queue_time,\
                    ttft,decision_time_ms"
                ).ok();
            }
        }
        
        Self {
            file_path,
            buffer: Vec::with_capacity(buffer_size),
            buffer_size,
            last_flush: Instant::now(),
            flush_interval: Duration::from_secs(flush_interval_secs),
        }
    }
    
    fn add_record(&mut self, record: RoutingDecisionRecord) {
        self.buffer.push(record);
        
        // Check if we should flush
        if self.buffer.len() >= self.buffer_size 
            || self.last_flush.elapsed() >= self.flush_interval {
            self.flush();
        }
    }
    
    fn flush(&mut self) {
        if self.buffer.is_empty() {
            return;
        }
        
        if let Ok(mut file) = OpenOptions::new()
            .append(true)
            .create(true)
            .open(&self.file_path)
        {
            for record in &self.buffer {
                writeln!(
                    file,
                    "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
                    record.timestamp,
                    record.worker_url,
                    record.request_id,
                    record.throughput_gradient.map_or("".to_string(), |v| format!("{:.4}", v)),
                    record.latency_gradient.map_or("".to_string(), |v| format!("{:.4}", v)),
                    record.score.map_or("".to_string(), |v| format!("{:.4}", v)),
                    record.outstanding_requests,
                    record.avg_throughput.map_or("".to_string(), |v| format!("{:.2}", v)),
                    record.avg_latency.map_or("".to_string(), |v| format!("{:.4}", v)),
                    record.window_size,
                    record.selection_reason,
                    record.actual_output_tokens.map_or("".to_string(), |v| v.to_string()),
                    record.server_latency.map_or("".to_string(), |v| format!("{:.4}", v)),
                    record.queue_time.map_or("".to_string(), |v| format!("{:.4}", v)),
                    record.ttft.map_or("".to_string(), |v| format!("{:.4}", v)),
                    record.decision_time_ms
                ).ok();
            }
            
            self.buffer.clear();
            self.last_flush = Instant::now();
        } else {
            error!("Failed to open CSV file for writing: {:?}", self.file_path);
        }
    }
}

impl Drop for CsvWriter {
    fn drop(&mut self) {
        self.flush();
    }
}

/// Marginal utility policy with metrics recording
#[derive(Debug)]
pub struct MarginalUtilityRecorderPolicy {
    /// Base policy for routing decisions
    base_policy: MarginalUtilityPolicy,
    /// CSV writer for metrics
    csv_writer: Arc<Mutex<CsvWriter>>,
    /// Configuration
    config: MarginalUtilityRecorderConfig,
}

impl MarginalUtilityRecorderPolicy {
    /// Create a new marginal utility recorder policy
    pub fn new(config: MarginalUtilityRecorderConfig) -> Self {
        info!(
            "Creating MarginalUtilityRecorderPolicy with output_dir: {}",
            config.output_dir
        );
        
        let base_policy = MarginalUtilityPolicy::new(config.base_config.clone());
        let csv_writer = Arc::new(Mutex::new(CsvWriter::new(
            &config.output_dir,
            config.buffer_size,
            config.flush_interval_secs,
        )));
        
        // Start background flush thread
        let writer_clone = csv_writer.clone();
        let flush_interval = config.flush_interval_secs;
        thread::spawn(move || {
            loop {
                thread::sleep(Duration::from_secs(flush_interval));
                if let Ok(mut writer) = writer_clone.lock() {
                    writer.flush();
                }
            }
        });
        
        Self {
            base_policy,
            csv_writer,
            config,
        }
    }
    
    /// Record a routing decision
    fn record_decision(
        &self,
        worker_url: &str,
        score: Option<f64>,
        gradients: Option<(f64, f64)>,
        performance: Option<(f64, f64)>,
        outstanding: usize,
        window_size: usize,
        selection_reason: &str,
        decision_time: Duration,
    ) {
        let record = RoutingDecisionRecord {
            timestamp: Local::now().format("%Y-%m-%d %H:%M:%S%.3f").to_string(),
            worker_url: worker_url.to_string(),
            request_id: "pending".to_string(),
            throughput_gradient: gradients.map(|(t, _)| t),
            latency_gradient: gradients.map(|(_, l)| l),
            score,
            outstanding_requests: outstanding,
            avg_throughput: performance.map(|(t, _)| t),
            avg_latency: performance.map(|(_, l)| l),
            window_size,
            selection_reason: selection_reason.to_string(),
            actual_output_tokens: None,
            server_latency: None,
            queue_time: None,
            ttft: None,
            decision_time_ms: decision_time.as_secs_f64() * 1000.0,
        };
        
        if let Ok(mut writer) = self.csv_writer.lock() {
            writer.add_record(record);
        }
    }
}

impl LoadBalancingPolicy for MarginalUtilityRecorderPolicy {
    fn select_worker(
        &self,
        workers: &[Box<dyn Worker>],
        request_text: Option<&str>,
    ) -> Option<usize> {
        let start_time = Instant::now();
        
        // Use base policy to make the decision
        let selected = self.base_policy.select_worker(workers, request_text)?;
        
        // Record the decision
        let decision_time = start_time.elapsed();
        let worker_url = workers[selected].url();
        
        // Try to get detailed metrics from the base policy
        // This requires accessing the internal state
        let selection_reason = "gradient_based"; // TODO: Get actual reason from base policy
        
        self.record_decision(
            worker_url,
            None, // score
            None, // gradients
            None, // performance
            0,    // outstanding
            self.config.base_config.window_size,
            selection_reason,
            decision_time,
        );
        
        debug!(
            "Selected worker {} for recording (took {:?})",
            worker_url, decision_time
        );
        
        Some(selected)
    }
    
    fn select_worker_pair(
        &self,
        prefill_workers: &[Box<dyn Worker>],
        decode_workers: &[Box<dyn Worker>],
        request_text: Option<&str>,
    ) -> Option<(usize, usize)> {
        self.base_policy.select_worker_pair(prefill_workers, decode_workers, request_text)
    }
    
    fn on_request_complete(&self, worker_url: &str, success: bool) {
        self.base_policy.on_request_complete(worker_url, success)
    }
    
    fn name(&self) -> &'static str {
        "marginal_utility_recorder"
    }
    
    fn reset(&self) {
        self.base_policy.reset();
        if let Ok(mut writer) = self.csv_writer.lock() {
            writer.flush();
        }
    }
    
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl MarginalUtilityRecorderPolicy {
    /// Handle request completion with detailed metrics (called by router)
    pub fn handle_request_metrics(&self, metrics: &RequestMetrics) {
        // Forward to base policy
        self.base_policy.handle_request_metrics(metrics);
        
        // Record metrics
        let record = RoutingDecisionRecord {
            timestamp: Local::now().format("%Y-%m-%d %H:%M:%S%.3f").to_string(),
            worker_url: metrics.worker_url.clone(),
            request_id: metrics.request_id.clone(),
            throughput_gradient: None, // Will be filled in next selection
            latency_gradient: None,
            score: None,
            outstanding_requests: 0,
            avg_throughput: metrics.decode_throughput(),
            avg_latency: Some(metrics.server_latency),
            window_size: self.config.base_config.window_size,
            selection_reason: "completion_record".to_string(),
            actual_output_tokens: metrics.actual_output_tokens,
            server_latency: Some(metrics.server_latency),
            queue_time: metrics.queue_time(),
            ttft: metrics.server_first_token_time.map(|t| t - metrics.server_created_time.unwrap_or(0.0)),
            decision_time_ms: 0.0,
        };
        
        if let Ok(mut writer) = self.csv_writer.lock() {
            writer.add_record(record);
        }
    }
}