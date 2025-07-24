"""Visualization utilities for SGLang testing framework."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
import seaborn as sns
from pathlib import Path


def plot_latency_distribution(
    latencies: Dict[str, List[float]],
    title: str = "Latency Distribution",
    output_path: Optional[str] = None
):
    """Plot latency distribution as box plots.
    
    Args:
        latencies: Dictionary mapping latency type to list of values
        title: Plot title
        output_path: Optional path to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Prepare data for box plot
    data = []
    labels = []
    for latency_type, values in latencies.items():
        data.append(values)
        labels.append(latency_type)
    
    # Create box plot
    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    
    # Color the boxes
    colors = plt.cm.Set3(np.linspace(0, 1, len(data)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.set_ylabel("Latency (ms)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    # Add mean values
    for i, (label, values) in enumerate(zip(labels, data)):
        mean_val = np.mean(values)
        ax.text(i + 1, mean_val, f"={mean_val:.1f}", 
                ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def plot_throughput_timeline(
    timestamps: List[float],
    throughputs: List[float],
    title: str = "Throughput Over Time",
    ylabel: str = "Throughput (req/s)",
    output_path: Optional[str] = None
):
    """Plot throughput over time.
    
    Args:
        timestamps: List of timestamps
        throughputs: List of throughput values
        title: Plot title
        ylabel: Y-axis label
        output_path: Optional path to save the plot
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Convert timestamps to relative time
    start_time = min(timestamps) if timestamps else 0
    relative_times = [t - start_time for t in timestamps]
    
    # Plot throughput
    ax.plot(relative_times, throughputs, 'b-', alpha=0.7, linewidth=1.5)
    
    # Add rolling average
    window_size = min(20, len(throughputs) // 10)
    if window_size > 1:
        rolling_avg = pd.Series(throughputs).rolling(window=window_size, center=True).mean()
        ax.plot(relative_times, rolling_avg, 'r-', linewidth=2, label=f'Rolling Avg (window={window_size})')
    
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def plot_concurrency_heatmap(
    concurrency_data: pd.DataFrame,
    title: str = "Concurrency Heatmap",
    output_path: Optional[str] = None
):
    """Plot concurrency as a heatmap over time.
    
    Args:
        concurrency_data: DataFrame with columns: timestamp, server_id, concurrency
        title: Plot title
        output_path: Optional path to save the plot
    """
    # Pivot data for heatmap
    if 'server_id' in concurrency_data.columns:
        pivot_data = concurrency_data.pivot_table(
            values='concurrency',
            index='server_id',
            columns=pd.cut(concurrency_data['timestamp'], bins=50),
            aggfunc='mean'
        )
    else:
        # Single server case
        time_bins = pd.cut(concurrency_data['timestamp'], bins=50)
        pivot_data = pd.DataFrame({
            'concurrency': concurrency_data.groupby(time_bins)['concurrency'].mean()
        }).T
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create heatmap
    sns.heatmap(pivot_data, cmap='YlOrRd', cbar_kws={'label': 'Concurrency'}, ax=ax)
    
    ax.set_xlabel("Time")
    ax.set_ylabel("Server ID")
    ax.set_title(title)
    
    # Rotate x-axis labels
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def create_summary_report(
    results: Dict[str, Any],
    output_dir: str,
    test_name: str = "sglang_test"
):
    """Create a comprehensive summary report with multiple visualizations.
    
    Args:
        results: Test results dictionary
        output_dir: Directory to save report files
        test_name: Name prefix for output files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Latency distribution
    ax1 = fig.add_subplot(gs[0, :2])
    if 'latencies' in results:
        latency_data = []
        latency_labels = []
        for lat_type, values in results['latencies'].items():
            if values:
                latency_data.append(values)
                latency_labels.append(lat_type)
        
        if latency_data:
            bp = ax1.boxplot(latency_data, labels=latency_labels, patch_artist=True)
            ax1.set_ylabel("Latency (ms)")
            ax1.set_title("Latency Distribution")
            ax1.grid(True, alpha=0.3)
    
    # 2. Throughput timeline
    ax2 = fig.add_subplot(gs[1, :])
    if 'throughput_timeline' in results:
        timeline = results['throughput_timeline']
        ax2.plot(timeline['timestamps'], timeline['values'], 'b-', alpha=0.7)
        ax2.set_xlabel("Time (seconds)")
        ax2.set_ylabel("Throughput (req/s)")
        ax2.set_title("Throughput Over Time")
        ax2.grid(True, alpha=0.3)
    
    # 3. Request success rate pie chart
    ax3 = fig.add_subplot(gs[0, 2])
    if 'metrics' in results:
        metrics = results['metrics']
        success = metrics.get('completed_requests', 0)
        failed = metrics.get('failed_requests', 0)
        
        if success + failed > 0:
            ax3.pie([success, failed], labels=['Success', 'Failed'], 
                   autopct='%1.1f%%', colors=['green', 'red'])
            ax3.set_title("Request Success Rate")
    
    # 4. Summary text
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')
    
    summary_text = f"Test Summary: {test_name}\n"
    summary_text += "=" * 50 + "\n"
    
    if 'config' in results:
        config = results['config']
        summary_text += f"Model: {config.get('model_path', 'N/A')}\n"
        summary_text += f"Request Rate: {config.get('request_rate', 0)} req/s\n"
        summary_text += f"Total Requests: {config.get('num_prompts', 0)}\n"
    
    if 'metrics' in results:
        metrics = results['metrics']
        summary_text += f"\nPerformance Metrics:\n"
        summary_text += f"  Request Throughput: {metrics.get('request_throughput', 0):.2f} req/s\n"
        summary_text += f"  Mean Server Latency: {metrics.get('mean_server_latency', 0):.1f} ms\n"
        summary_text += f"  P99 Server Latency: {metrics.get('p99_server_latency', 0):.1f} ms\n"
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    # Save the report
    report_path = output_dir / f"{test_name}_summary_report.png"
    plt.savefig(report_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return str(report_path)