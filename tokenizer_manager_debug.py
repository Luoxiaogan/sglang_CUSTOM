#!/usr/bin/env python3
"""
调试版本的 tokenizer_manager.py 修改
用于诊断为什么时间戳没有出现在响应中

使用方法：
1. 将这些修改应用到服务器端的 tokenizer_manager.py
2. 重启服务器
3. 运行测试查看日志输出
"""

# 版本标记 - 添加到文件开头
TOKENIZER_MANAGER_VERSION = "v2_with_timestamps_debug"

# 在 TokenizerManager.__init__ 方法中添加
print(f"=== TokenizerManager initialized with version: {TOKENIZER_MANAGER_VERSION} ===")

# 修改 _handle_batch_output 方法，添加详细日志
def _handle_batch_output_with_debug(
    self,
    recv_obj: Union[
        BatchStrOut, BatchEmbeddingOut, BatchMultimodalOut, BatchTokenIDOut
    ],
):
    logger.info(f"=== DEBUG: _handle_batch_output called with {type(recv_obj).__name__} ===")
    
    for i, rid in enumerate(recv_obj.rids):
        state = self.rid_to_state.get(rid, None)
        if state is None:
            logger.error(
                f"Received output for {rid=} but the state was deleted in TokenizerManager."
            )
            continue

        logger.info(f"=== DEBUG: Processing rid={rid}, index={i} ===")
        logger.info(f"=== DEBUG: state.created_time={state.created_time}, state.first_token_time={state.first_token_time} ===")

        # Build meta_info and return value
        meta_info = {
            "id": rid,
            "finish_reason": recv_obj.finished_reasons[i],
            "prompt_tokens": recv_obj.prompt_tokens[i],
        }
        
        logger.info(f"=== DEBUG: Initial meta_info: {meta_info} ===")

        if getattr(state.obj, "return_logprob", False):
            self.convert_logprob_style(
                meta_info,
                state,
                state.obj.top_logprobs_num,
                state.obj.token_ids_logprob,
                state.obj.return_text_in_logprobs
                and not self.server_args.skip_tokenizer_init,
                recv_obj,
                i,
            )

        logger.info(f"=== DEBUG: isinstance(recv_obj, BatchEmbeddingOut) = {isinstance(recv_obj, BatchEmbeddingOut)} ===")
        
        if not isinstance(recv_obj, BatchEmbeddingOut):
            logger.info("=== DEBUG: Adding timestamps to meta_info ===")
            meta_info.update(
                {
                    "completion_tokens": recv_obj.completion_tokens[i],
                    "cached_tokens": recv_obj.cached_tokens[i],
                    # Add server-side timestamps for accurate queue time measurement
                    "server_created_time": state.created_time,
                    "server_first_token_time": state.first_token_time if state.first_token_time > 0 else None,
                    # 添加版本标记以验证代码生效
                    "_debug_version": TOKENIZER_MANAGER_VERSION,
                }
            )
            logger.info(f"=== DEBUG: meta_info after timestamp update: {meta_info} ===")
        else:
            logger.info("=== DEBUG: Skipping timestamp update for BatchEmbeddingOut ===")

        if getattr(recv_obj, "output_hidden_states", None):
            meta_info["hidden_states"] = recv_obj.output_hidden_states[i]

        # 处理不同类型的输出
        if isinstance(recv_obj, BatchStrOut):
            logger.info("=== DEBUG: Processing BatchStrOut ===")
            state.text += recv_obj.output_strs[i]
            out_dict = {
                "text": state.text,
                "meta_info": meta_info,
            }
        elif isinstance(recv_obj, BatchTokenIDOut):
            logger.info("=== DEBUG: Processing BatchTokenIDOut ===")
            if self.server_args.stream_output and state.obj.stream:
                state.output_ids.extend(recv_obj.output_ids[i])
                output_token_ids = state.output_ids[state.last_output_offset :]
                state.last_output_offset = len(state.output_ids)
            else:
                state.output_ids.extend(recv_obj.output_ids[i])
                output_token_ids = state.output_ids.copy()

            out_dict = {
                "output_ids": output_token_ids,
                "meta_info": meta_info,
            }
        elif isinstance(recv_obj, BatchMultimodalOut):
            raise NotImplementedError("BatchMultimodalOut not implemented")
        else:
            assert isinstance(recv_obj, BatchEmbeddingOut)
            logger.info("=== DEBUG: Processing BatchEmbeddingOut ===")
            out_dict = {
                "embedding": recv_obj.embeddings[i],
                "meta_info": meta_info,
            }

        state.finished = recv_obj.finished_reasons[i] is not None
        if state.finished:
            logger.info(f"=== DEBUG: Request finished, adding e2e_latency ===")
            if self.server_args.speculative_algorithm:
                meta_info["spec_verify_ct"] = recv_obj.spec_verify_ct[i]
            state.finished_time = time.time()
            meta_info["e2e_latency"] = state.finished_time - state.created_time
            logger.info(f"=== DEBUG: Final meta_info for finished request: {meta_info} ===")
            del self.rid_to_state[rid]

        state.out_list.append(out_dict)
        state.event.set()

        # Log metrics and dump
        if self.enable_metrics and state.obj.log_metrics:
            self.collect_metrics(state, recv_obj, i)
        if self.dump_requests_folder and state.finished and state.obj.log_metrics:
            self.dump_requests(state, out_dict)
        if self.crash_dump_folder and state.finished and state.obj.log_metrics:
            self.record_request_for_crash_dump(state, out_dict)


# 在 generate_request 方法中添加日志
def generate_request_with_debug(...):
    """带调试日志的 generate_request 方法"""
    # 在方法开始时
    logger.info(f"=== DEBUG: generate_request called, stream={obj.stream} ===")
    
    # 在创建 ReqState 后
    created_time = time.time()
    state = ReqState(
        out_list=[],
        finished=False,
        event=asyncio.Event(),
        obj=obj,
        created_time=created_time,
    )
    logger.info(f"=== DEBUG: Created ReqState with created_time={created_time} ===")
    
    # ... 其余代码 ...

# 另外，在 collect_metrics 方法中检查 first_token_time 的设置
def collect_metrics_with_debug(self, state: ReqState, recv_obj: Any, i: int):
    """带调试日志的 collect_metrics 方法"""
    # Check whether it is the first token
    if (
        state.first_token_time == 0.0
        and getattr(recv_obj, "output_token_logprobs_val", None) is not None
    ):
        state.first_token_time = state.last_time = time.time()
        logger.info(f"=== DEBUG: Set first_token_time={state.first_token_time} for rid={state.obj.id} ===")
        if self.enable_metrics:
            self.metrics_collector.observe_time_to_first_token(
                state.first_token_time - state.created_time
            )
    # ... 其余代码 ...