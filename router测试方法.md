# SGLang Router 测试

### 0. 安装rust环境和编译
本地测试安装sglang:
```bash
pip install -e "python[all]"
```
然后安装rust和相关的编译依赖
```bash
apt install -y pkgconf
apt install -y libssl-dev # 可以不用

conda install -c conda-forge openssl #可能conda的openssl比apt跟不容易报错
export OPENSSL_DIR=$CONDA_PREFIX
export OPENSSL_INCLUDE_DIR=$CONDA_PREFIX/include
export OPENSSL_LIB_DIR=$CONDA_PREFIX/lib
export PKG_CONFIG_PATH=$CONDA_PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH

pkg-config --version
conda install -c conda-forge clang
conda install -c conda-forge gcc_impl_linux=12.0
conda install -c conda-forge gcc_linux-64 gxx_linux-64 gfortran_linux-64 # 还有一个原因出在system的gcc版本太低了, 并且后续可能还需要which gcc来测试使用
conda install -c conda-forge rust
conda update -c conda-forge rust
pip install maturin
rustc --version
cargo --version
```
以及配置rust的临时镜像
```bash
export CARGO_REGISTRIES_CRATES_IO_REGISTRY=https://mirrors.tuna.tsinghua.edu.cn/crates.io-index
echo $CARGO_HOME
cd $HOME/.cargo

touch config.toml

cat << EOF | tee -a ${CARGO_HOME:-$HOME/.cargo}/config.toml
[source.crates-io]
replace-with = 'mirror'

[source.mirror]
registry = "sparse+https://mirrors.tuna.tsinghua.edu.cn/crates.io-index/"

[registries.mirror]
index = "sparse+https://mirrors.tuna.tsinghua.edu.cn/crates.io-index/"
EOF

cat config.toml

conda remove -c conda-forge rust # 删除重装
```
编译
```bash 
cargo build --release
maturin build --release
```

### 1. 启动 SGLang 服务器

```bash
# 终端 1 - GPU 2
python -m sglang.launch_server \
--model-path "/nas/models/Meta-Llama-3-8B-Instruct" \
--host "0.0.0.0" \
--port 60005 \
--base-gpu-id 2 \
--enable-metrics \  # 添加这个参数,collect_metrics 只在 self.enable_metrics 为 True 时调用
--log-level debug # 这个可以在server的log里面检查相应的全部的log

# 终端 2 - GPU 3
python -m sglang.launch_server \
--model-path "/nas/models/Meta-Llama-3-8B-Instruct" \
--host "0.0.0.0" \
--port 60006 \
--base-gpu-id 3 \
--enable-metrics
```

### 2. 启动路由器（启用请求追踪）

```bash
bash /nas/ganluo/sglang/bash_start_router.sh
```
停止的方法是:
```bash
ctrl+Z
kill %1
clear
```

### 3. 测试router
```bash
bash /nas/ganluo/sglang/bash_send_req.sh
```

### 4. 故障排除
当遇到29000端口被占据的情况, 是因为之前的router端口没有释放, 需要这样做
```bash
lsof -i :29000
```
然后`kill -9`这个pid.

### 5. 参数解读
目前记录在csv中的参数有（按照新的顺序）:

**基础信息：**
req_id,success,error,host,has_generated_text

**长度相关字段：**
input_length,expected_output_length,actual_prompt_tokens,actual_output_tokens,actual_total_tokens,output_tokens_from_trace,decode_length

**时间戳（按时间顺序）：**
arrival_time,router_send_time,server_created_time,queue_time_start,queue_time_end,server_first_token_time,finish_time

**延迟指标：**
server_latency,total_latency,ttft,queue_time_in_router,queue_time_in_server,tokenize_time,pure_queue_time

#### 参数含义详解：

**基础信息：**
- `req_id`: 请求的唯一标识符
- `success`: 请求是否成功（True/False）
- `error`: 错误信息（如果有）
- `host`: 实际处理请求的服务器地址
- `has_generated_text`: 是否有生成的文本（用于调试）

**长度相关字段说明（重要）：**
- `input_length`: 输入文本的**单词数**（由 `prompt.split()` 计算，以空格分割）
- `expected_output_length`: 请求时设置的期望输出 **token 数**（max_new_tokens 参数）
- `actual_prompt_tokens`: 输入文本经过 tokenizer 处理后的实际 **token 数**（从服务器返回）
- `actual_output_tokens`: 实际生成的 **token 数**（从服务器返回）
- `actual_total_tokens`: 实际总 **token 数**（输入+输出，从服务器返回）
- `output_tokens_from_trace`: 从路由器追踪信息获取的输出 **token 数**（通常为 0）
- `decode_length`: 综合字段，表示生成的 **token 数**（优先使用 actual_output_tokens）

**注意事项：**
1. **单词数 vs Token 数**：
   - `input_length` 是按空格分割的单词数，不是 token 数
   - 对于英文：一个单词可能对应 1-3 个 tokens
   - 对于中文：由于中文通常不使用空格分割，`input_length` 可能不准确
   - 因此 `actual_prompt_tokens` 才是真实的输入 token 数量

2. **准确的 Token 计数**：
   - **必须使用** `actual_prompt_tokens` 而不是 `input_length` 来计算输入 token 吞吐量
   - **必须使用** `actual_output_tokens` 而不是 `decode_length` 来计算输出 token 吞吐量
   
3. **示例对比**：
   - 英文文本 "Hello world": `input_length`=2（2个单词），`actual_prompt_tokens` 可能是 2-3
   - 中文文本 "你好世界": `input_length`=1（没有空格），`actual_prompt_tokens` 可能是 4-6

**时间戳（按时间顺序）：**
- `arrival_time`: 请求到达router的时刻（从测试开始计时）
- `router_send_time`: router将请求转发到server的时刻（原名 to_server_time）
- `server_created_time`: 请求在server端被tokenizer_manager接收的时刻
  - 根据`python/sglang/srt/entrypoints/http_server.py`, server_created_time 是请求被 tokenizer_manager 接收的时刻，而不是 HTTP 请求刚到达服务器的时刻
  - 但是请求到达服务器后基本上是立即转发到 tokenizer_manager 的，中间没有排队
- `queue_time_start`: 请求进入scheduler waiting_queue的时刻
- `queue_time_end`: 请求从waiting_queue取出准备处理的时刻
- `server_first_token_time`: server生成第一个token的时刻（prefill完成+第一次decode）
- `finish_time`: 请求完成的时刻

**延迟指标：**
- `server_latency`: 服务器中的总时间 = `finish_time - router_send_time`
- `total_latency`: 总延迟 = `finish_time - arrival_time`
- `ttft` (Time To First Token): 第一个token生成时间 = `server_first_token_time - arrival_time`（从客户端发送时刻算起）
- `queue_time_in_router`: router中的排队时间 = `router_send_time - arrival_time`（通常很小）
- `queue_time_in_server`: server端总排队时间 = `server_first_token_time - server_created_time`
- `tokenize_time`: server端tokenize的时间 = `queue_time_start - server_created_time`
- `pure_queue_time`: scheduler纯排队时间 = `queue_time_end - queue_time_start`（不含tokenize时间）

**Throughput(吞吐量计算)计算方法, 注意这里记录的是不同的host上的, 分开计算**
+ 首先计算的时候，需要按照不同的 host 进行group
+ 之后把每个group按照 finish_time 从小到大排序
+ 之后，对于每个group, 对于一个req, 我们知道这个req的finish_time, 然后计算是这样的:
  + prefill_token_throughput: (所有小于等于finish_time的这个group的reqs的`actual_prompt_tokens`的求和) / (finish_time)
  + decode_token_throughput: (所有小于等于finish_time的这个group的reqs的`actual_output_tokens`的求和) / (finish_time)
+ 这两个参数在csv里面的位置是，在ttft和queue_time_in_router之间。

**最后还需要添加对于整个系统的评判：**
+ 整个df按照 finish_time 从小到大排序
+ 对于一个req, 我们知道这个req的finish_time
+ PREFILL_THROUGHPUT: (所有小于等于finish_time(不考虑host了)的reqs的`actual_prompt_tokens`的求和) / (finish_time)
+ DECODE_THROUGHPUT: (所有小于等于finish_time(不考虑host了)的reqs的`actual_output_tokens`的求和) / (finish_time)
+ AVG_LATENCY: finish_time/(所有小于等于finish_time(不考虑host了)的reqs的数目)
+ 这3个参数在csv里面的位置时req_id 最前面.


**各阶段耗时计算：**
1. **Router转发延迟**: `router_send_time - arrival_time` （通常<0.1ms）
2. **Tokenize时间**: `queue_time_start - server_created_time`
3. **Scheduler排队时间**: `queue_time_end - queue_time_start`（即 pure_queue_time）
4. **Prefill时间**: `server_first_token_time - queue_time_end`
5. **网络传输时间**: `server_created_time - router_send_time`

### 6. 关于 decode_length 和 token 计数说明（已修复）

#### 问题背景
之前测试中 `decode_length` 显示为 0 是因为依赖路由器追踪信息，但该信息可能不完整。

#### 修复方案（已实施）
现在测试代码已经更新，采用与 SGLang 官方 bench_serving.py 相同的方法获取实际生成的 token 数：

1. **从 meta_info 提取**（SGLang native API）：
   - `completion_tokens`: 实际生成的 token 数
   - `prompt_tokens`: 实际输入的 token 数
   - `total_tokens`: 总 token 数

2. **从 usage 字段提取**（OpenAI-compatible API）：
   - 作为 meta_info 的备选方案

3. **CSV 中记录的所有 token 相关字段**：
   - `expected_output_length`: 请求时设置的期望输出长度
   - `actual_output_tokens`: 从 meta_info/usage 获取的实际生成 token 数
   - `actual_prompt_tokens`: 从 meta_info/usage 获取的实际输入 token 数
   - `actual_total_tokens`: 从 meta_info/usage 获取的总 token 数
   - `output_tokens_from_trace`: 从路由器追踪信息获取的输出 token 数
   - `decode_length`: 综合字段（优先使用 actual_output_tokens，保持向后兼容）
   - `has_generated_text`: 是否有生成的文本（用于调试）

#### 使用说明
- 分析时优先使用 `actual_output_tokens` 字段，这是最准确的实际生成长度
- `decode_length` 保留用于向后兼容
- 如果 `actual_output_tokens` 为空，可能是服务器版本较旧或未启用相关功能