#!/usr/bin/env python3
"""
从日志数据直接生成优化分析图表
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime

# 直接从日志中提取的数据
data = {
    'baseline': {
        'probabilities': [0.5, 0.5],
        'prefill': 5556.88,
        'decode': 886.32,
        'latency': 0.062,
        'objective': 6443.1948
    },
    'perturbation_1': {
        'probabilities': [0.504950495049505, 0.49504950495049505],
        'prefill': 3819.16,
        'decode': 615.07,
        'latency': 0.089,
        'gradient': -200896.685035
    },
    'perturbation_2': {
        'probabilities': [0.49504950495049505, 0.504950495049505],
        'prefill': 4830.32,
        'decode': 764.18,
        'latency': 0.070,
        'gradient': -84869.404829
    },
    'final': {
        'probabilities': [0.5, 0.5],
        'prefill': 4055.40,
        'decode': 676.51,
        'latency': 0.080407,
        'objective': 4731.9175
    }
}

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def create_optimization_plots():
    """创建优化过程的可视化图表"""
    
    # 创建图表
    fig = plt.figure(figsize=(16, 10))
    
    # 1. 吞吐量对比
    ax1 = plt.subplot(2, 3, 1)
    scenarios = ['Baseline', 'Perturb +W1', 'Perturb +W2', 'Final']
    prefill_values = [
        data['baseline']['prefill'],
        data['perturbation_1']['prefill'],
        data['perturbation_2']['prefill'],
        data['final']['prefill']
    ]
    decode_values = [
        data['baseline']['decode'],
        data['perturbation_1']['decode'],
        data['perturbation_2']['decode'],
        data['final']['decode']
    ]
    
    x = np.arange(len(scenarios))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, prefill_values, width, label='Prefill', color='#2E86AB')
    bars2 = ax1.bar(x + width/2, decode_values, width, label='Decode', color='#A23B72')
    
    ax1.set_xlabel('Test Scenario')
    ax1.set_ylabel('Throughput (tok/s)')
    ax1.set_title('Throughput Comparison Across Different Probability Configurations')
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenarios, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 在柱子上添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}', ha='center', va='bottom', fontsize=8)
    
    # 2. 延迟对比
    ax2 = plt.subplot(2, 3, 2)
    latency_values = [
        data['baseline']['latency'],
        data['perturbation_1']['latency'],
        data['perturbation_2']['latency'],
        data['final']['latency']
    ]
    
    bars = ax2.bar(scenarios, latency_values, color=['#27AE60', '#E74C3C', '#F39C12', '#3498DB'])
    ax2.set_xlabel('Test Scenario')
    ax2.set_ylabel('Average Latency (seconds)')
    ax2.set_title('Latency Comparison Across Different Probability Configurations')
    ax2.grid(True, alpha=0.3)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom')
    
    # 3. 总吞吐量（Prefill + Decode）
    ax3 = plt.subplot(2, 3, 3)
    total_throughput = [p + d for p, d in zip(prefill_values, decode_values)]
    
    line = ax3.plot(scenarios, total_throughput, 'o-', linewidth=2, markersize=10, color='#E74C3C')
    ax3.set_xlabel('Test Scenario')
    ax3.set_ylabel('Total Throughput (tok/s)')
    ax3.set_title('Total Throughput Trend')
    ax3.grid(True, alpha=0.3)
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
    
    # 添加数值标签
    for i, (x_pos, y_pos) in enumerate(zip(scenarios, total_throughput)):
        ax3.annotate(f'{y_pos:.0f}', 
                    xy=(i, y_pos), 
                    xytext=(0, 10),
                    textcoords='offset points',
                    ha='center',
                    fontsize=9)
    
    # 4. 概率分布变化
    ax4 = plt.subplot(2, 3, 4)
    worker1_probs = [
        data['baseline']['probabilities'][0],
        data['perturbation_1']['probabilities'][0],
        data['perturbation_2']['probabilities'][0],
        data['final']['probabilities'][0]
    ]
    worker2_probs = [
        data['baseline']['probabilities'][1],
        data['perturbation_1']['probabilities'][1],
        data['perturbation_2']['probabilities'][1],
        data['final']['probabilities'][1]
    ]
    
    ax4.plot(scenarios, worker1_probs, 'o-', label='Worker 1', linewidth=2, markersize=8)
    ax4.plot(scenarios, worker2_probs, 's-', label='Worker 2', linewidth=2, markersize=8)
    ax4.set_xlabel('Test Scenario')
    ax4.set_ylabel('Probability')
    ax4.set_title('Probability Distribution Evolution')
    ax4.set_ylim([0.48, 0.52])
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
    
    # 5. 梯度值
    ax5 = plt.subplot(2, 3, 5)
    gradients = [
        data['perturbation_1']['gradient'],
        data['perturbation_2']['gradient']
    ]
    gradient_labels = ['Worker 1\n(+0.01 perturbation)', 'Worker 2\n(+0.01 perturbation)']
    
    bars = ax5.bar(gradient_labels, gradients, color=['#E74C3C', '#3498DB'])
    ax5.set_ylabel('Gradient Value')
    ax5.set_title('Numerical Gradient Calculation Results')
    ax5.grid(True, alpha=0.3)
    ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}', ha='center', va='top' if height < 0 else 'bottom')
    
    # 6. 性能指标雷达图
    ax6 = plt.subplot(2, 3, 6, projection='polar')
    
    # 准备雷达图数据
    categories = ['Prefill\nThroughput', 'Decode\nThroughput', 'Low\nLatency', 'Total\nThroughput']
    
    # 归一化数据（将所有值缩放到0-1之间）
    baseline_values = [
        data['baseline']['prefill'] / 6000,  # 归一化到6000
        data['baseline']['decode'] / 1000,   # 归一化到1000
        1 - data['baseline']['latency'] / 0.1,  # 反转延迟（低延迟=高分）
        (data['baseline']['prefill'] + data['baseline']['decode']) / 7000
    ]
    final_values = [
        data['final']['prefill'] / 6000,
        data['final']['decode'] / 1000,
        1 - data['final']['latency'] / 0.1,
        (data['final']['prefill'] + data['final']['decode']) / 7000
    ]
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    baseline_values += baseline_values[:1]  # 闭合图形
    final_values += final_values[:1]
    angles += angles[:1]
    
    ax6.plot(angles, baseline_values, 'o-', linewidth=2, label='Baseline', color='#2E86AB')
    ax6.fill(angles, baseline_values, alpha=0.25, color='#2E86AB')
    ax6.plot(angles, final_values, 's-', linewidth=2, label='Final', color='#E74C3C')
    ax6.fill(angles, final_values, alpha=0.25, color='#E74C3C')
    
    ax6.set_xticks(angles[:-1])
    ax6.set_xticklabels(categories)
    ax6.set_ylim(0, 1)
    ax6.set_title('Performance Metrics Comparison (Normalized)')
    ax6.legend(loc='upper right')
    ax6.grid(True)
    
    # Main title
    fig.suptitle('SGLang Router Gradient Optimization Analysis\nInitial Prob [0.5, 0.5] → Final Prob [0.5, 0.5]', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f'/Users/luogan/Code/sglang/optimization_analysis_{timestamp}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Chart saved to: {output_file}")
    
    plt.show()

def create_summary_table():
    """Create summary table"""
    
    # Create dataframe
    summary_data = {
        'Scenario': ['Baseline', 'Perturb +W1', 'Perturb +W2', 'Final'],
        'Worker1 Prob': [0.500, 0.505, 0.495, 0.500],
        'Worker2 Prob': [0.500, 0.495, 0.505, 0.500],
        'Prefill Throughput': [5556.88, 3819.16, 4830.32, 4055.40],
        'Decode Throughput': [886.32, 615.07, 764.18, 676.51],
        'Total Throughput': [6443.20, 4434.23, 5594.50, 4731.91],
        'Avg Latency(ms)': [62, 89, 70, 80.4],
        'Objective Value': [6443.19, '-', '-', 4731.92]
    }
    
    df = pd.DataFrame(summary_data)
    
    # Print table
    print("\n" + "="*80)
    print("Optimization Process Data Summary")
    print("="*80)
    print(df.to_string(index=False))
    
    # Calculate and print key findings
    print("\n" + "="*80)
    print("Key Findings")
    print("="*80)
    
    # Gradient analysis
    print("\nGradient Analysis:")
    print(f"  Worker 1 gradient: {data['perturbation_1']['gradient']:.2f}")
    print(f"  Worker 2 gradient: {data['perturbation_2']['gradient']:.2f}")
    print("  Conclusion: Both gradients are negative, indicating that increasing")
    print("              either worker's probability reduces total throughput.")
    print("              This suggests [0.5, 0.5] uniform distribution may be a local optimum.")
    
    # Performance changes
    print("\nPerformance Changes:")
    baseline_total = data['baseline']['prefill'] + data['baseline']['decode']
    final_total = data['final']['prefill'] + data['final']['decode']
    change_pct = (final_total - baseline_total) / baseline_total * 100
    
    print(f"  Baseline total throughput: {baseline_total:.2f} tok/s")
    print(f"  Final total throughput: {final_total:.2f} tok/s")
    print(f"  Change: {change_pct:.1f}%")
    
    print("\nLatency Changes:")
    latency_change = (data['final']['latency'] - data['baseline']['latency']) / data['baseline']['latency'] * 100
    print(f"  Baseline latency: {data['baseline']['latency']*1000:.1f} ms")
    print(f"  Final latency: {data['final']['latency']*1000:.1f} ms")
    print(f"  Change: {latency_change:.1f}%")

def main():
    """Main function"""
    print("Generating optimization analysis charts...")
    
    # Create summary table
    create_summary_table()
    
    # Create visualization charts
    create_optimization_plots()
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()