import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_process_data(filepath: str) -> pd.DataFrame:
    """读取JSONL文件，处理数据，并计算关键指标。"""
    records = []
    with open(filepath, 'r') as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    
    df = pd.DataFrame(records)
    
    # --- 核心修改：在调用 .mean() 时，加入 numeric_only=True 参数 ---
    df = df.groupby("request_rate").mean(numeric_only=True).reset_index()
    
    # 计算平均服务时间 S (秒)
    # 我们现在从已经取过均值的DataFrame中计算，结果是一样的
    avg_s = df['mean_server_latency_ms'].mean() / 1000.0
    print(f"根据所有数据计算出的全局平均服务时间 S = {avg_s:.6f} 秒")
    
    # 计算流量强度 ρ 和 实验队列长度 N_experimental
    df['traffic_intensity_rho'] = df['request_rate'] * avg_s / df['batch_size']
    df['n_experimental'] = df['request_rate'] * (df['mean_total_latency_ms'] / 1000.0)
    
    # 筛选掉不稳定的点 (rho >= 1) 以便绘图
    stable_df = df[df['traffic_intensity_rho'] < 1].copy()
    
    return stable_df

# ... (脚本的其余部分保持不变, 为保持回答简洁此处省略) ...

def plot_results(df: pd.DataFrame):
    if df.empty:
        print("没有稳定的数据点可供绘图。")
        return

    sns.set_theme(style="whitegrid")
    
    # --- 图 1: 吞吐量 vs 请求速率 ---
    plt.figure(figsize=(10, 6))
    # 使用 request_throughput 作为Y轴
    plt.plot(df['request_rate'], df['request_throughput'], marker='o', linestyle='-', label='Experimental Throughput')
    # 计算理论饱和点
    # 理论最大吞吐率 = K/S (每秒处理的批次数 * 每批大小)
    # 对应的请求率 λ = K/S
    avg_s = df['mean_server_latency_ms'].mean() / 1000.0
    batch_size = df['batch_size'].iloc[0]
    theoretical_max_rate = batch_size / avg_s
    plt.axvline(x=theoretical_max_rate, color='r', linestyle='--', label=f'Theoretical Saturation Rate (ρ=1) ≈ {theoretical_max_rate:.2f} req/s')
    plt.title('System Throughput vs. Arrival Rate', fontsize=16)
    plt.xlabel('Arrival Rate (λ, requests/sec)', fontsize=12)
    plt.ylabel('Request Throughput (requests/sec)', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.savefig('throughput_vs_rate.png')
    print("已保存图表: throughput_vs_rate.png")

    # --- 图 2: 延迟 vs 请求速率 ---
    plt.figure(figsize=(10, 6))
    plt.plot(df['request_rate'], df['mean_total_latency_ms'], marker='o', linestyle='-', label='Mean Total Latency (End-to-End)')
    plt.plot(df['request_rate'], df['mean_server_latency_ms'], marker='s', linestyle='--', label='Mean Server-Side Latency')
    plt.title('Latency vs. Arrival Rate', fontsize=16)
    plt.xlabel('Arrival Rate (λ, requests/sec)', fontsize=12)
    plt.ylabel('Latency (ms)', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.savefig('latency_vs_rate.png')
    print("已保存图表: latency_vs_rate.png")

    # --- 图 3: 队列长度 vs 系统负载 (最关键的理论验证图) ---
    plt.figure(figsize=(10, 6))
    plt.plot(df['traffic_intensity_rho'], df['n_experimental'], marker='o', linestyle='-', label='Experimental Queue Length (N_exp)')
    plt.title('Queue Length vs. Traffic Intensity (ρ)', fontsize=16)
    plt.xlabel('Traffic Intensity (ρ)', fontsize=12)
    plt.ylabel('Average Number of Requests in System (N)', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.savefig('queue_length_vs_rho.png')
    print("已保存图表: queue_length_vs_rho.png")
    
    # 在Jupyter Notebook等环境中，取消下一行注释可以直接显示图片
    # plt.show()



# 1. 加载和处理数据
processed_df = load_and_process_data("/home/lg/sglang/static_batch_results.jsonl")

# 打印处理后的数据概览
print("\n--- Processed Data Overview (Stable Points where ρ < 1) ---")
print(processed_df[['request_rate', 'traffic_intensity_rho', 'n_experimental', 'request_throughput', 'mean_total_latency_ms']].round(3))

# 2. 绘图
print("\n--- Generating Plots ---")
plot_results(processed_df)