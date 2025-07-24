import numpy as np

def generate_lengths(mean_input_len, mean_output_len, variance=1.0, num_samples=1):
    """
    生成服从正态分布的输入和输出长度
    
    参数:
    - mean_input_len: 输入长度的均值
    - mean_output_len: 输出长度的均值
    - variance: 方差，默认为1.0
    - num_samples: 需要生成的样本数量
    
    返回:
    - input_lengths: 输入长度列表
    - output_lengths: 输出长度列表
    """
    # 生成正态分布的长度
    input_lengths = np.random.normal(mean_input_len, np.sqrt(variance), num_samples)
    output_lengths = np.random.normal(mean_output_len, np.sqrt(variance), num_samples)
    
    # 确保长度为正整数
    input_lengths = np.maximum(1, np.round(input_lengths)).astype(int)
    output_lengths = np.maximum(1, np.round(output_lengths)).astype(int)
    
    return input_lengths, output_lengths

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--mean-input-len", type=int, required=True)
    parser.add_argument("--mean-output-len", type=int, required=True)
    parser.add_argument("--variance", type=float, default=1.0)
    parser.add_argument("--num-samples", type=int, default=1)
    
    args = parser.parse_args()
    
    input_lens, output_lens = generate_lengths(
        args.mean_input_len,
        args.mean_output_len,
        args.variance,
        args.num_samples
    )
    
    # 打印结果，每行一对长度值
    for in_len, out_len in zip(input_lens, output_lens):
        print(f"{in_len} {out_len}") 