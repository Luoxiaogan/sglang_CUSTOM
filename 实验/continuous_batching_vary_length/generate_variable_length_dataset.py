# generate_variable_length_dataset.py
import json
import argparse
import numpy as np

def generate_dataset(num_prompts, mean_input_len, mean_output_len, std_dev_ratio, output_file):
    """
    生成一个包含可变长度prompt和output_len的JSON数据集。
    整个文件是一个大的JSON数组，使用 'value' 键以匹配sglang的期望。
    """
    input_lens = np.random.lognormal(mean=np.log(mean_input_len), sigma=std_dev_ratio, size=num_prompts)
    output_lens = np.random.lognormal(mean=np.log(mean_output_len), sigma=std_dev_ratio, size=num_prompts)

    all_records = []
    for i in range(num_prompts):
        prompt_len = max(1, int(input_lens[i]))
        output_len = max(1, int(output_lens[i]))
        
        prompt_text = "A" * prompt_len 
        
        # --- 核心改动在这里 ---
        record = {
            "conversations": [
                {"role": "user", "value": prompt_text},          # <--- 'content' 改为 'value'
                {"role": "assistant", "value": "B" * output_len} # <--- 'content' 改为 'value'
            ]
        }
        all_records.append(record)

    with open(output_file, "w") as f:
        json.dump(all_records, f) 

    print(f"成功生成数据集，包含 {num_prompts} 条记录，已保存到 {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-prompts", type=int, required=True)
    parser.add_argument("--mean-input-len", type=int, required=True)
    parser.add_argument("--mean-output-len", type=int, required=True)
    parser.add_argument("--std-dev-ratio", type=float, default=0.3, help="长度分布的标准差与均值的比例")
    parser.add_argument("--output-file", type=str, required=True)
    args = parser.parse_args()

    generate_dataset(args.num_prompts, args.mean_input_len, args.mean_output_len, args.std_dev_ratio, args.output_file)