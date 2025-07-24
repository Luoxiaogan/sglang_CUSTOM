# demand_2_check

**INSTRUCTION: 指示中提到的.md文件和.py文件, 以及提到的文件夹下面的所有文件, 都必须阅读一遍**

## 1. API兼容性检查和修改

1. 目前是初始化了`CLAUDE.md`
2. 并且在`sglang_test_framework`里面制定了初步的文件结构以及目标
   1. 你可以查看`sglang_test_framework/说明.md`
目前.
```
sglang_test_framework/
├── config/               # 配置模块
│   ├── base.py          # 基础配置类
│   ├── node_config.py   # 节点测试配置
│   └── routing_config.py # 路由测试配置
├── core/                # 核心功能模块
│   ├── server_manager.py    # 服务器管理
│   ├── request_generator.py # 请求生成
│   ├── metrics_collector.py # 指标收集
│   └── result_manager.py    # 结果管理
├── strategies/          # 策略实现（待开发）
│   ├── batching/       # 批处理策略
│   └── routing/        # 路由策略
├── tests/              # 测试运行器（待开发）
├── utils/              # 工具函数（待开发）
└── requirements.txt    # 依赖列表
```
config and core部分已经完成了开发

3. 本次任务不需要对后面的strategies, tests, utils进行开发. 而是要检查config, core是否适配和对齐了sglang api
   1. 对于可以直接复用的sglang的代码和模块的, 可以直接copy过来, 不需要自己实现. 但是需要在.py文件最前面注释来源
   2. 我思考了一下: 可能主要能够复用的是:
      1. server 模块: 直接使用server的api, 看文档: `.DUCUMENT/Server_Arguments.md`, `.DUCUMENT/Sampling_Parameters.md`.
      2. 记录metric模块: 目前记录metric
         1. 我需要的是, 对于每一个request, 记录相关的metric, 你可以参考
            1. `python/sglang/bench_serving_new.py`, `python/sglang/bench_serving.py`
            2. 但是这个是benchmark. 我是要在serving过程中记录per request的metric
      3. routing 模块: in `sgl-router`, 看文档: `.DUCUMENT/SGLang_Router_详解.md`

**本次任务不需要对后面的strategies, tests, utils进行开发. 而是要检查config, core是否适配和对齐了sglang api**

**INSTRUCTION: 指示中提到的.md文件和.py文件, 以及提到的文件夹下面的所有文件, 都必须阅读一遍**