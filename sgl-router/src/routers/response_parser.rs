//! Response parser for extracting performance metrics from SGLang server responses

use crate::policies::RequestMetrics;
use serde_json::Value;
use std::time::Instant;
use tracing::debug;

/// Parser for extracting metrics from server responses
pub struct ResponseParser;

impl ResponseParser {
    /// Extract performance metrics from response body
    pub fn extract_metrics(
        body: &[u8],
        worker_url: &str,
        request_start_instant: Instant,
    ) -> Option<RequestMetrics> {
        // Try to parse JSON response
        let json: Value = match serde_json::from_slice(body) {
            Ok(v) => v,
            Err(e) => {
                debug!("Failed to parse response as JSON: {}", e);
                return None;
            }
        };

        // Extract meta_info (required for metrics)
        let meta_info = json.get("meta_info")?;

        // Extract timestamps (all optional)
        let server_created_time = meta_info
            .get("server_created_time")
            .and_then(|v| v.as_f64());
        let server_first_token_time = meta_info
            .get("server_first_token_time")
            .and_then(|v| v.as_f64());
        let queue_time_start = meta_info.get("queue_time_start").and_then(|v| v.as_f64());
        let queue_time_end = meta_info.get("queue_time_end").and_then(|v| v.as_f64());

        // Extract token counts from meta_info
        let completion_tokens = meta_info
            .get("completion_tokens")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);
        let cached_tokens = meta_info
            .get("cached_tokens")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);

        // prompt_tokens might be in meta_info
        let prompt_tokens = meta_info
            .get("prompt_tokens")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);

        // Also check usage field for OpenAI-compatible format
        let usage = json.get("usage");
        let actual_output_tokens = completion_tokens.or_else(|| {
            usage
                .and_then(|u| u.get("completion_tokens"))
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
        });
        let actual_prompt_tokens = prompt_tokens.or_else(|| {
            usage
                .and_then(|u| u.get("prompt_tokens"))
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
        });
        let actual_total_tokens = usage
            .and_then(|u| u.get("total_tokens"))
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);

        // Calculate latencies using router's timing
        let response_time = Instant::now();
        let server_latency = response_time
            .duration_since(request_start_instant)
            .as_secs_f64();
        let total_latency = server_latency; // At router level, these are the same

        // Get current time as finish_time
        let finish_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();

        // Extract request_id if available
        let request_id = json
            .get("request_id")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();

        // Log parsed data for debugging
        debug!(
            "Parsed metrics from response: output_tokens={:?}, server_created_time={:?}, queue_times=({:?}, {:?})",
            actual_output_tokens, server_created_time, queue_time_start, queue_time_end
        );

        Some(RequestMetrics {
            worker_url: worker_url.to_string(),
            request_id,
            server_created_time,
            server_first_token_time,
            queue_time_start,
            queue_time_end,
            finish_time,
            server_latency,
            total_latency,
            actual_prompt_tokens,
            actual_output_tokens,
            actual_total_tokens,
            cached_tokens,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_sglang_response() {
        let response_json = r#"{
            "text": "Hello world",
            "meta_info": {
                "completion_tokens": 50,
                "cached_tokens": 10,
                "server_created_time": 1000.0,
                "server_first_token_time": 1000.5,
                "queue_time_start": 1000.1,
                "queue_time_end": 1000.2
            }
        }"#;

        let start = Instant::now();
        let metrics = ResponseParser::extract_metrics(
            response_json.as_bytes(),
            "http://worker1:8000",
            start,
        );

        assert!(metrics.is_some());
        let metrics = metrics.unwrap();
        assert_eq!(metrics.worker_url, "http://worker1:8000");
        assert_eq!(metrics.actual_output_tokens, Some(50));
        assert_eq!(metrics.cached_tokens, Some(10));
        assert_eq!(metrics.server_created_time, Some(1000.0));
        assert_eq!(metrics.server_first_token_time, Some(1000.5));
        assert_eq!(metrics.queue_time_start, Some(1000.1));
        assert_eq!(metrics.queue_time_end, Some(1000.2));
    }

    #[test]
    fn test_parse_openai_compatible_response() {
        let response_json = r#"{
            "choices": [{"text": "Hello"}],
            "meta_info": {
                "e2e_latency": 1.5
            },
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }"#;

        let start = Instant::now();
        let metrics = ResponseParser::extract_metrics(
            response_json.as_bytes(),
            "http://worker2:8000",
            start,
        );

        assert!(metrics.is_some());
        let metrics = metrics.unwrap();
        assert_eq!(metrics.actual_prompt_tokens, Some(10));
        assert_eq!(metrics.actual_output_tokens, Some(5));
        assert_eq!(metrics.actual_total_tokens, Some(15));
    }

    #[test]
    fn test_parse_invalid_response() {
        let response = "Not JSON";
        let start = Instant::now();
        let metrics =
            ResponseParser::extract_metrics(response.as_bytes(), "http://worker1:8000", start);
        assert!(metrics.is_none());
    }

    #[test]
    fn test_parse_response_without_meta_info() {
        let response_json = r#"{"text": "Hello world"}"#;
        let start = Instant::now();
        let metrics = ResponseParser::extract_metrics(
            response_json.as_bytes(),
            "http://worker1:8000",
            start,
        );
        assert!(metrics.is_none());
    }
}