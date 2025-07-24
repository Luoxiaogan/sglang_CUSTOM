"""Result management and visualization for SGLang testing."""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from datetime import datetime
import logging

from .metrics_collector import AggregatedMetrics, MetricsCollector

logger = logging.getLogger(__name__)


class ResultManager:
    """Manages test results, analysis, and visualization."""
    
    def __init__(self, output_dir: str = "./results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set plotting style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
    def save_results(self, 
                    metrics_collector: MetricsCollector,
                    config: Dict[str, Any],
                    test_name: str = None) -> str:
        """Save test results to files."""
        if test_name is None:
            test_name = datetime.now().strftime("%Y%m%d_%H%M%S")
            
        test_dir = self.output_dir / test_name
        test_dir.mkdir(exist_ok=True)
        
        # Save configuration
        config_path = test_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2, default=str)
            
        # Save metrics in multiple formats
        metrics_path = test_dir / "metrics.json"
        metrics_collector.export_metrics("json", str(metrics_path))
        
        # Also save as CSV for detailed per-request analysis
        csv_path = test_dir / "results.csv"
        metrics_collector.export_metrics("csv", str(csv_path))
        logger.info(f"Exported per-request metrics to {csv_path}")
        
        # Save summary
        summary_path = test_dir / "summary.txt"
        with open(summary_path, 'w') as f:
            # Redirect print to file
            import sys
            old_stdout = sys.stdout
            sys.stdout = f
            metrics_collector.print_summary()
            sys.stdout = old_stdout
            
        logger.info(f"Results saved to {test_dir}")
        return str(test_dir)
    
    def create_visualizations(self,
                            metrics_collector: MetricsCollector,
                            save_dir: str = None) -> Dict[str, str]:
        """Create visualization plots for the test results."""
        if save_dir is None:
            save_dir = self.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
        else:
            save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        plots = {}
        
        # 1. Latency distribution plot
        plots['latency_dist'] = self._plot_latency_distribution(
            metrics_collector, save_dir / "latency_distribution.png"
        )
        
        # 2. Throughput over time plot
        plots['throughput_time'] = self._plot_throughput_over_time(
            metrics_collector, save_dir / "throughput_over_time.png"
        )
        
        # 3. Queue length over time
        plots['queue_length'] = self._plot_queue_length(
            metrics_collector, save_dir / "queue_length.png"
        )
        
        # 4. Request timeline
        plots['request_timeline'] = self._plot_request_timeline(
            metrics_collector, save_dir / "request_timeline.png"
        )
        
        # 5. Performance heatmap
        plots['perf_heatmap'] = self._plot_performance_heatmap(
            metrics_collector, save_dir / "performance_heatmap.png"
        )
        
        # 6. GPU utilization (if available)
        if metrics_collector.snapshots and any('gpu_utilization' in s.metrics for s in metrics_collector.snapshots):
            plots['gpu_util'] = self._plot_gpu_utilization(
                metrics_collector, save_dir / "gpu_utilization.png"
            )
        
        logger.info(f"Created {len(plots)} visualization plots in {save_dir}")
        return plots
    
    def _plot_latency_distribution(self, 
                                  metrics_collector: MetricsCollector,
                                  save_path: Path) -> str:
        """Plot latency distribution."""
        results = [r for r in metrics_collector.results if r.success]
        
        if not results:
            return ""
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Latency Distribution Analysis', fontsize=16)
        
        # Server latency
        server_latencies = [r.server_latency * 1000 for r in results]
        axes[0, 0].hist(server_latencies, bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 0].set_xlabel('Server Latency (ms)')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('Server Latency Distribution')
        axes[0, 0].axvline(np.mean(server_latencies), color='red', linestyle='--', label=f'Mean: {np.mean(server_latencies):.1f}ms')
        axes[0, 0].axvline(np.percentile(server_latencies, 95), color='orange', linestyle='--', label=f'P95: {np.percentile(server_latencies, 95):.1f}ms')
        axes[0, 0].legend()
        
        # Total latency
        total_latencies = [r.total_latency * 1000 for r in results]
        axes[0, 1].hist(total_latencies, bins=50, alpha=0.7, color='green', edgecolor='black')
        axes[0, 1].set_xlabel('Total Latency (ms)')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('Total Latency Distribution')
        axes[0, 1].axvline(np.mean(total_latencies), color='red', linestyle='--', label=f'Mean: {np.mean(total_latencies):.1f}ms')
        axes[0, 1].axvline(np.percentile(total_latencies, 95), color='orange', linestyle='--', label=f'P95: {np.percentile(total_latencies, 95):.1f}ms')
        axes[0, 1].legend()
        
        # Queue time
        queue_times = [r.queue_time * 1000 for r in results]
        axes[1, 0].hist(queue_times, bins=50, alpha=0.7, color='purple', edgecolor='black')
        axes[1, 0].set_xlabel('Queue Time (ms)')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Queue Time Distribution')
        axes[1, 0].axvline(np.mean(queue_times), color='red', linestyle='--', label=f'Mean: {np.mean(queue_times):.1f}ms')
        axes[1, 0].legend()
        
        # TTFT
        ttfts = [r.ttft * 1000 for r in results if r.ttft > 0]
        if ttfts:
            axes[1, 1].hist(ttfts, bins=50, alpha=0.7, color='orange', edgecolor='black')
            axes[1, 1].set_xlabel('Time to First Token (ms)')
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].set_title('TTFT Distribution')
            axes[1, 1].axvline(np.mean(ttfts), color='red', linestyle='--', label=f'Mean: {np.mean(ttfts):.1f}ms')
            axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        
        return str(save_path)
    
    def _plot_throughput_over_time(self,
                                  metrics_collector: MetricsCollector,
                                  save_path: Path) -> str:
        """Plot throughput over time."""
        results = [r for r in metrics_collector.results if r.success and r.request.completion_time]
        
        if not results:
            return ""
            
        # Sort by completion time
        results.sort(key=lambda r: r.request.completion_time)
        
        # Calculate throughput in windows
        window_size = 10  # seconds
        start_time = results[0].request.completion_time
        end_time = results[-1].request.completion_time
        
        times = []
        request_throughputs = []
        token_throughputs = []
        
        current_time = start_time
        while current_time < end_time:
            window_end = current_time + window_size
            
            # Count requests in window
            window_results = [r for r in results 
                            if current_time <= r.request.completion_time < window_end]
            
            if window_results:
                num_requests = len(window_results)
                total_tokens = sum(r.request.prompt_len + r.request.output_len for r in window_results)
                
                request_throughput = num_requests / window_size
                token_throughput = total_tokens / window_size
                
                times.append(current_time - start_time)
                request_throughputs.append(request_throughput)
                token_throughputs.append(token_throughput)
            
            current_time += window_size / 2  # Sliding window
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        fig.suptitle('Throughput Over Time', fontsize=16)
        
        # Request throughput
        ax1.plot(times, request_throughputs, 'b-', linewidth=2)
        ax1.fill_between(times, request_throughputs, alpha=0.3)
        ax1.set_ylabel('Requests/sec')
        ax1.set_title('Request Throughput')
        ax1.grid(True, alpha=0.3)
        
        # Token throughput
        ax2.plot(times, token_throughputs, 'g-', linewidth=2)
        ax2.fill_between(times, token_throughputs, alpha=0.3)
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Tokens/sec')
        ax2.set_title('Token Throughput')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        
        return str(save_path)
    
    def _plot_queue_length(self,
                          metrics_collector: MetricsCollector,
                          save_path: Path) -> str:
        """Plot queue length over time."""
        if not metrics_collector.queue_lengths:
            return ""
            
        times, lengths = zip(*metrics_collector.queue_lengths)
        
        # Convert to relative time
        start_time = times[0]
        times = [(t - start_time) for t in times]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(times, lengths, 'b-', linewidth=1, alpha=0.7)
        ax.fill_between(times, lengths, alpha=0.3)
        
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Queue Length (requests waiting)')
        ax.set_title('Queue Length Over Time')
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        ax.axhline(np.mean(lengths), color='red', linestyle='--', 
                  label=f'Mean: {np.mean(lengths):.1f}')
        ax.axhline(np.max(lengths), color='orange', linestyle='--', 
                  label=f'Max: {np.max(lengths)}')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        
        return str(save_path)
    
    def _plot_request_timeline(self,
                              metrics_collector: MetricsCollector,
                              save_path: Path) -> str:
        """Plot request timeline showing arrival, send, and completion times."""
        results = metrics_collector.results[:100]  # Limit to first 100 for readability
        
        if not results:
            return ""
            
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Sort by arrival time
        results.sort(key=lambda r: r.request.arrival_time)
        
        # Get time bounds
        min_time = min(r.request.arrival_time for r in results)
        
        for i, result in enumerate(results):
            req = result.request
            
            # Normalize times
            arrival = req.arrival_time - min_time
            send = (req.send_time - min_time) if req.send_time else arrival
            complete = (req.completion_time - min_time) if req.completion_time else send
            
            # Plot timeline
            color = 'green' if result.success else 'red'
            
            # Queue time (arrival to send)
            ax.barh(i, send - arrival, left=arrival, height=0.8, 
                   color='yellow', alpha=0.5, label='Queue' if i == 0 else "")
            
            # Processing time (send to complete)
            ax.barh(i, complete - send, left=send, height=0.8,
                   color=color, alpha=0.7, label='Processing' if i == 0 else "")
        
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Request ID')
        ax.set_title('Request Timeline (First 100 Requests)')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        
        return str(save_path)
    
    def _plot_performance_heatmap(self,
                                 metrics_collector: MetricsCollector,
                                 save_path: Path) -> str:
        """Plot performance heatmap."""
        metrics = metrics_collector.get_aggregated_metrics()
        
        # Create normalized metrics matrix
        data = {
            'Throughput': {
                'Request (req/s)': metrics.request_throughput,
                'Input Token (tok/s)': metrics.input_token_throughput / 1000,  # Scale to k
                'Output Token (tok/s)': metrics.output_token_throughput / 1000,
            },
            'Latency (ms)': {
                'Server Mean': metrics.mean_server_latency,
                'Server P95': metrics.p95_server_latency,
                'Total Mean': metrics.mean_total_latency,
                'Total P95': metrics.p95_total_latency,
            },
            'Queue': {
                'Mean Time (ms)': metrics.mean_queue_time,
                'Mean Length': metrics.mean_queue_length,
                'Max Length': float(metrics.max_queue_length),
            }
        }
        
        # Convert to DataFrame
        df_data = []
        for category, metrics_dict in data.items():
            for metric, value in metrics_dict.items():
                df_data.append({
                    'Category': category,
                    'Metric': metric,
                    'Value': value
                })
        
        df = pd.DataFrame(df_data)
        pivot_df = df.pivot(index='Metric', columns='Category', values='Value')
        
        # Normalize each column
        normalized_df = pivot_df.copy()
        for col in pivot_df.columns:
            col_values = pivot_df[col]
            if col_values.max() > 0:
                normalized_df[col] = (col_values - col_values.min()) / (col_values.max() - col_values.min())
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(normalized_df.T, annot=pivot_df.T, fmt='.1f', cmap='YlGnBu', 
                   cbar_kws={'label': 'Normalized Value'}, ax=ax)
        
        ax.set_title('Performance Metrics Heatmap')
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Categories')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        
        return str(save_path)
    
    def _plot_gpu_utilization(self,
                             metrics_collector: MetricsCollector,
                             save_path: Path) -> str:
        """Plot GPU utilization over time."""
        gpu_data = []
        
        for snapshot in metrics_collector.snapshots:
            if 'gpu_utilization' in snapshot.metrics:
                gpu_data.append({
                    'time': snapshot.timestamp,
                    'gpu_util': snapshot.metrics['gpu_utilization'],
                    'gpu_mem': snapshot.metrics.get('gpu_memory_utilization', 0)
                })
        
        if not gpu_data:
            return ""
            
        df = pd.DataFrame(gpu_data)
        df['time'] = df['time'] - df['time'].min()  # Relative time
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        fig.suptitle('GPU Utilization Over Time', fontsize=16)
        
        # GPU compute utilization
        ax1.plot(df['time'], df['gpu_util'], 'b-', linewidth=2)
        ax1.fill_between(df['time'], df['gpu_util'], alpha=0.3)
        ax1.set_ylabel('GPU Utilization (%)')
        ax1.set_ylim(0, 105)
        ax1.grid(True, alpha=0.3)
        ax1.axhline(df['gpu_util'].mean(), color='red', linestyle='--',
                   label=f'Mean: {df["gpu_util"].mean():.1f}%')
        ax1.legend()
        
        # GPU memory utilization
        ax2.plot(df['time'], df['gpu_mem'], 'g-', linewidth=2)
        ax2.fill_between(df['time'], df['gpu_mem'], alpha=0.3)
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('GPU Memory Utilization (%)')
        ax2.set_ylim(0, 105)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(df['gpu_mem'].mean(), color='red', linestyle='--',
                   label=f'Mean: {df["gpu_mem"].mean():.1f}%')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        
        return str(save_path)
    
    def compare_results(self,
                       result_dirs: List[str],
                       labels: List[str] = None,
                       save_dir: str = None) -> Dict[str, str]:
        """Compare results from multiple test runs."""
        if labels is None:
            labels = [f"Test {i+1}" for i in range(len(result_dirs))]
            
        if save_dir is None:
            save_dir = self.output_dir / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Load metrics from each test
        all_metrics = []
        for result_dir, label in zip(result_dirs, labels):
            metrics_path = Path(result_dir) / "metrics.json"
            if metrics_path.exists():
                with open(metrics_path, 'r') as f:
                    data = json.load(f)
                    data['label'] = label
                    all_metrics.append(data)
        
        if not all_metrics:
            logger.warning("No valid metrics found for comparison")
            return {}
            
        plots = {}
        
        # 1. Throughput comparison
        plots['throughput_comp'] = self._plot_throughput_comparison(
            all_metrics, save_dir / "throughput_comparison.png"
        )
        
        # 2. Latency comparison
        plots['latency_comp'] = self._plot_latency_comparison(
            all_metrics, save_dir / "latency_comparison.png"
        )
        
        # 3. Summary table
        plots['summary_table'] = self._create_comparison_table(
            all_metrics, save_dir / "comparison_summary.csv"
        )
        
        return plots
    
    def _plot_throughput_comparison(self,
                                   all_metrics: List[Dict],
                                   save_path: Path) -> str:
        """Plot throughput comparison."""
        labels = []
        request_throughputs = []
        token_throughputs = []
        
        for metrics in all_metrics:
            summary = metrics['summary']
            labels.append(metrics['label'])
            request_throughputs.append(summary['request_throughput'])
            token_throughputs.append(summary['total_token_throughput'])
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Request throughput
        bars1 = ax1.bar(labels, request_throughputs, color='skyblue', edgecolor='navy')
        ax1.set_ylabel('Requests/sec')
        ax1.set_title('Request Throughput Comparison')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, value in zip(bars1, request_throughputs):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{value:.1f}', ha='center', va='bottom')
        
        # Token throughput
        bars2 = ax2.bar(labels, token_throughputs, color='lightgreen', edgecolor='darkgreen')
        ax2.set_ylabel('Tokens/sec')
        ax2.set_title('Token Throughput Comparison')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, value in zip(bars2, token_throughputs):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                    f'{value:.0f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        
        return str(save_path)
    
    def _plot_latency_comparison(self,
                                all_metrics: List[Dict],
                                save_path: Path) -> str:
        """Plot latency comparison."""
        labels = []
        mean_latencies = []
        p95_latencies = []
        p99_latencies = []
        
        for metrics in all_metrics:
            summary = metrics['summary']
            labels.append(metrics['label'])
            mean_latencies.append(summary['mean_total_latency'])
            p95_latencies.append(summary['p95_total_latency'])
            p99_latencies.append(summary['p99_total_latency'])
        
        x = np.arange(len(labels))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars1 = ax.bar(x - width, mean_latencies, width, label='Mean', color='blue', alpha=0.7)
        bars2 = ax.bar(x, p95_latencies, width, label='P95', color='orange', alpha=0.7)
        bars3 = ax.bar(x + width, p99_latencies, width, label='P99', color='red', alpha=0.7)
        
        ax.set_xlabel('Test')
        ax.set_ylabel('Latency (ms)')
        ax.set_title('Latency Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height + 5,
                       f'{height:.0f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        
        return str(save_path)
    
    def _create_comparison_table(self,
                                all_metrics: List[Dict],
                                save_path: Path) -> str:
        """Create comparison summary table."""
        rows = []
        
        for metrics in all_metrics:
            summary = metrics['summary']
            row = {
                'Test': metrics['label'],
                'Total Requests': summary['total_requests'],
                'Success Rate (%)': summary['completed_requests'] / summary['total_requests'] * 100,
                'Request Throughput (req/s)': summary['request_throughput'],
                'Token Throughput (tok/s)': summary['total_token_throughput'],
                'Mean Latency (ms)': summary['mean_total_latency'],
                'P95 Latency (ms)': summary['p95_total_latency'],
                'P99 Latency (ms)': summary['p99_total_latency'],
                'Mean Queue Time (ms)': summary['mean_queue_time'],
                'Max Queue Length': summary.get('max_queue_length', 0)
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(save_path, index=False)
        
        # Also print to console
        print("\nComparison Summary:")
        print(df.to_string(index=False))
        
        return str(save_path)