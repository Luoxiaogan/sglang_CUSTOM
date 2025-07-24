1. SGLANG的知识库文档: `.DUCUMENT`
   1. 将它们加入/memory
   2. 文档中的github网址对应的文件，都可以本地找到，因为`https://github.com/sgl-project/sglang/tree/b045841baeff37a5601fcde23fa98bd09d942c36/`; `https://github.com/sgl-project/sglang/tree/main/`; 等都对应了本项目的根目录
   3. 本项目在服务器上进行测试，但本地进行claude code编写.
   4. sglang的所有API使用都**必须**查询`.DUCUMENT`
2. 项目的目的，利用sglang的借口, 构建一个易用的测试框架
   1. 暴露的借口除了GPU，inference用的数据(或者直接随机生成)之外——这些都可以传到sgalng的api，他们已经考虑了这些
   2. 所有测试使用最简单的`llama-2-7b-hf`, 单个模型在单个GPU上启动
   3. 我们主要考虑以下的测试:
      1. node level: 只启动一个server on a single GPU.
         1. 可以选择的策略: static batching, continuous batching(sglang只支持这个方法).
            1. static batching 可以看我自己实现的`实验/static_length_with_distribution`
               1. 本质上是启动server之后, 按照static的方法来发送请求.
      2. routing level: 例如启动多个server on 多个GPU(一个model在一个GPU上, num_servers = num_gpus)
         1. 可以调试routing policy
         2. 有custom的路由策略(提前给一个base class), 后面可以定制
            1. 典型的policy有: uniform 随机路由; shortest job first 路由; parameters aware routing(router can get the parameters from the servers, like throughput, latency.)
   4. 保存的metric, 主要考虑throughput and latency.
      1. 查看`python/sglang/bench_serving.py` and `python/sglang/bench_serving_new.py`(我自己写的) for knowledge
      2. 特点澄清:
         1. sglang 记录的 latency 是 reqeust 被传入server时刻，到reqeust完成推理的时刻，的时间
            1. 我称为server_latency
         2. 我还需要记录的是, reqeust 本身的生成是根据泊松分布生成的, 那么有以下的三个时刻
            1. reqeust到达的时刻(生成时刻), 模拟的是用户的request到达router.
            2. reqeust被送到服务器, 模拟的是用户的reqeust在router处排队之后，被发送 to a server.
            3. reqeust完成推理.
         3. 你需要查看`.DUCUMENT/Router_for_Data_Parallelism.md`了解实现方法
   5. 补充说明：
      1. max_concurrency 不是 SGLang 服务器的启动参数，而是用于基准测试脚本（如 sglang.bench_serving）的参数，用于限制客户端并发请求数。在服务端，并发能力主要通过 --max-running-requests 控制。需要详细区分：服务端用 --max-running-requests，客户端基准测试用 --max-concurrency。
      2. --max-running-requests 控制 SGLang 服务器上同时运行的最大请求数。它限制了在解码和生成阶段能并发处理的请求数量，从而影响并发能力和显存占用。设置过大可能导致显存溢出（OOM），设置过小则会降低吞吐量。推荐根据显存容量和实际负载调整，通常与 --mem-fraction-static 联合优化。
      3. --max-running-requests 需要在启动 SGLang 服务器（launch server）时作为命令行参数设置，该参数不能在服务器启动后动态修改，必须在启动命令中指定。
         1. 但是，实际上，是可以修改的，只需要做一个APP参数下发。node 的到更新的--max-running-requests之后，如果当前的running-requests < --max-running-requests那么自动调整上限, 如果大于, 那么就对running的arrival进行停止，直到running-requests降低到上限
   6. 每个node可以修改的参数
      1. 如果是node测试, 那么基本和sglang自己的launch server的参数一样
         1. 逻辑基本和我在`实验`里面实现的一样
      2. 如果是routing测试, 那么是routing policy & --max-running-requests(这个是可以在serve的过程中进行修改)

# instruction

🔍 代码修复对比改进分析指令 

说明部分请都说中文

📚 知识库`.DUCUMENT`一致性验证
必须执行的知识库`.DUCUMENT`查询
1. **查询相关文件**：
   - 确认文件的最新版本
   - 验证错误行号与实际代码的对应关系
   - 禁止使用防御性编程，及时报错发现错误

2. **接口依赖检查**：
   - 查找所有调用该函数/类的位置
   - 确认接口签名（参数、返回值）
   - 验证与其他模块的依赖关系
   - **特别注意**：记录所有可用的工具函数和API接口

3. **数据流追踪**：
   - 追踪错误数据的来源
   - 确认数据类型和格式要求
   - 检查相关的数据验证逻辑

4. **版本一致性**：
   - 确认代码版本与错误日志匹配
   - 检查是否有未同步的修改
   - 验证所有相关文件的一致性
🎯 根本原因分析
系统性问题诊断
## 🔬 深度原因分析框架
**直接原因**：[导致错误的直接代码问题]

**根本原因**：[更深层的设计或逻辑问题]
- 接口设计缺陷
- 数据结构不匹配
- 状态管理错误
- 异步/并发问题
- 资源管理问题

**一致性影响**：
- 修复后需要更新的接口
- 需要同步修改的相关文件
- 可能破坏的现有功能
📝 最佳解决方案
单一最优方案设计
## 🏆 推荐解决方案
**方案概述**：[简洁描述修复策略]

**关键原则**：
1. 使用已有的接口和函数
2. 遵循项目现有的设计模式
3. 避免硬编码，使用配置或常量
4. 保持代码的可扩展性
5. 禁止使用try except! 我要碰见错误就直接显示traceback，并且退出终止运行程序，方便我从本质上解决问题
6. 禁止采用备用fallback等方案，如缺少属性直接报错返回！

**具体实施**：
1. 核心修复：[解决直接错误]
2. 接口调整：[保持一致性的必要修改]
3. 验证步骤：[确保修复有效]

**一致性保证**：
- 保持的接口：[列出不变的接口]
- 更新的接口：[列出需要更新的接口及原因]
- 兼容性处理：[如何保证向后兼容]
🔧 代码修改对比
⚠️ 完整性要求
## 📋 代码输出规范
1. **必须输出完整函数**：
   - 不允许使用"前面代码不变"等省略表达
   - 从函数定义开始到函数结束的完整代码
   - 包含所有必要的导入语句
   - 在每一处可能出错的地方（比如try except, if等）添加print作为报错点，如果采用了备用方案则打印出具体位置

2. **保持上下文**：
   - 显示函数的完整签名
   - 保留原有的注释和文档字符串

关键修改说明
## 📋 修改要点
1. **修复内容**：[具体修改了什么]
2. **使用的知识接口**：
   - [函数1]：来自[文件名]，用于[目的]
   - [函数2]：来自[文件名]，用于[目的]
3. **避免的硬编码**：
   - 原本：硬编码的值
   - 现在：使用配置/常量
4. **接口保持**：[哪些接口签名保持不变]
5. **必要更新**：[哪些调用方需要更新]
✅ 一致性验证清单
修复后验证项
## 🔍 一致性检查项
□ 完整的函数/方法代码
□ 没有使用"省略"或"不变"等表达
□ 没有硬编码的魔数或字符串
□ 接口签名变更已记录
□ 数据格式兼容性已验证
□ 相关测试用例已考虑
□ 文档注释已同步

