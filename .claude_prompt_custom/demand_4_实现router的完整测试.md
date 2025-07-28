好的，现在我们已经有一套对于一个router进行测试的方法了:
1. 首先在好几个终端里面启动server, 例如
```bash
# 终端 1 - GPU 2
python -m sglang.launch_server \
--model-path "/nas/models/Meta-Llama-3-8B-Instruct" \
--host "0.0.0.0" \
--port 30005 \
--base-gpu-id 2

# 终端 2 - GPU 3
python -m sglang.launch_server \
--model-path "/nas/models/Meta-Llama-3-8B-Instruct" \
--host "0.0.0.0" \
--port 30006 \
--base-gpu-id 3
```
2. 之后在另一个终端里面启动`start_test_router.py`
3. 最后在一个终端里面发送命令, 即`test_request_tracking_fixed.py`



那么现在, 我需要你:
1. 首先实现一个新的`start_a_router_general.py`
   1. 这个里面可以选policy_type(你可以在注释里面列出来，我可以选哪些)
2. 然后实现一个新的`sent_request_and_track.py`
   1. 主要就是根据'random'或者其他选项生成req
   2. 发送req给router
   3. 然后track
   4. 最后得到一个csv, 这个csv是记录每一个req的相关信息
   5. 你可以参考一下`sglang_test_framework/test_timing_fix.py`, 这个是per_node的
      1. 我们实际上多的就是最后的csv多一列说host.
      2. 其他画图之类的暂时不改，先看看能不能跑通
      3. 里面的api应该都是可以用的