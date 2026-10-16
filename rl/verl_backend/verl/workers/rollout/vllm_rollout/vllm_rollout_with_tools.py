# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import concurrent.futures
import importlib
import logging
import os
import re
import time
import random
from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Counter

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict

from verl import DataProto
from verl.third_party.vllm import vllm_version
from verl.utils.debug import GPUMemoryLogger
from verl.utils.torch_functional import get_response_mask, pad_sequence_to_length
from verl.workers.rollout.tools.base_tool import BaseTool
from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import vLLMRollout, _pre_process_inputs, _repeat_interleave
import math
from verl.workers.rollout.vllm_rollout.appo_branching import APPOBrancher
logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _load_tool_from_config(tool_config: DictConfig) -> BaseTool:
    """Dynamically loads a tool from its configuration."""
    module_path, class_name = tool_config.class_path.rsplit('.', 1)
    try:
        module = importlib.import_module(module_path)
        
        tool_class = getattr(module, class_name)
        
        tool_params = OmegaConf.to_container(tool_config.get('params', {}), resolve=True)
        
        tool_instance = tool_class(**tool_params)
        
        return tool_instance
    except ImportError as e:
        logger.error(f"Failed to import module {module_path}: {e}")
        raise
    except AttributeError as e:
        logger.error(f"Failed to find class {class_name} in module {module_path}: {e}")
        raise
    except TypeError as e:
        logger.error(f"Failed to instantiate {class_name} with provided parameters: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading tool from {tool_config.class_path}: {e}")
        raise


class vLLMRolloutWithTools(vLLMRollout):
    """
    An advanced vLLM rollout engine capable of handling multiple tools like
    code interpreters and search engines during generation.

    This class extends vLLMRollout by orchestrating a multi-step generation
    process where the language model can emit special tokens to trigger external
    tools. The tool outputs are then fed back into the model to continue
    generation.
    """

    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        super().__init__(model_path, config, tokenizer, model_hf_config, **kwargs)
        self.tokenizer = tokenizer

        self.initial_rollouts = self.config.get("initial_rollouts", self.config['n'])
        
        # 从配置中获取工具设置
        tools_config = self.config.get("tools", OmegaConf.create({}))

        # 获取工具通用配置
        self.tool_call_limit = tools_config.get("call_limit", 5)
        self.max_tool_workers = tools_config.get("max_workers", 64)
        self.tool_timeout = tools_config.get("timeout", 120)

        # 其他可能的工具通用配置
        self.tool_retry_count = tools_config.get("retry_count", 3)
        self.tool_verbose_logging = tools_config.get("verbose_logging", False)

        
        self.tools: Dict[str, BaseTool] = {}
        if "tool_instances" in tools_config:
            for tool_name, tool_config in tools_config.tool_instances.items():
                logger.info(f"Loading tool '{tool_name}' from {tool_config.class_path}")
                try:
                    tool_instance = _load_tool_from_config(tool_config)
                    self.tools[tool_instance.trigger_tag] = tool_instance
                except Exception as e:
                    logger.error(f"Could not initialize tool '{tool_name}'. Please check your configuration. Error: {e}")
                    if tools_config.get("fail_on_error", False):
                        raise

        self.tool_stop_sequences = [f"</{tag}>" for tag in self.tools.keys()]
        # Stop on </answer> so the model cannot ramble / emit extra answers after boxing.
        self.stop_sequences = self.tool_stop_sequences + ["</answer>"]

        if not self.tools:
            logger.warning(
                "vLLMRolloutWithTools initialized, but no tools were configured.")

        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_tool_workers)

        # ============================================================
        # APPO: APPO procedural branching 相关参数
        # ============================================================
        # 缓存 COGNITIVE_ADDON 的 token ids 和 <|im_end|> 的 token id
        self._cognitive_addon_token_ids: Optional[List[int]] = None
        self._im_end_token_id: Optional[int] = None
        # APPO brancher 实例（懒加载）
        self._appo_brancher: Optional[APPOBrancher] = None
        # reward 缩放因子 M 的距离折扣系数
        self.reward_scale_discount = self.config.get("reward_scale_discount", 0.9)
        self.appo_dynamic_branching = self.config.get("appo_dynamic_branching", True)

    def __del__(self):
        self.executor.shutdown(wait=False)

    def _extract_content(self, text: str, tag: str) -> str:
        """Extracts content from within the last <tag>...</tag> block."""
        try:
            start_tag = f"<{tag}>"
            end_tag = f"</{tag}>"
            end_pos = text.rindex(end_tag)
            start_pos = text.rindex(start_tag, 0, end_pos)
            return text[start_pos + len(start_tag):end_pos].strip()
        except ValueError:
            logger.warning(
                f"Could not extract content for tag '{tag}' from text: {text}")
            return ""

    def _execute_tool_with_retry(self, tool, content, sample_id=None):
        """Execute *tool* with *content*, retrying up to ``tool_retry_count`` times.

        Args:
            tool:      Tool instance (must implement ``execute(code, sample_id=...)``)
            content:   Code / query string to pass to the tool.
            sample_id: Identifier for the current rollout sample.  Passed through
                       to tools that support persistent namespaces (e.g. PythonTool)
                       so that variables defined in one <python> block are visible
                       in subsequent blocks of the same sample.
        """
        retry_count = 0
        start_time = time.time()

        while retry_count < self.tool_retry_count:
            try:
                # Pass sample_id if the tool's execute() accepts it; fall back
                # gracefully for tools that don't have the parameter yet.
                try:
                    result_text = tool.execute(content, sample_id=sample_id)
                except TypeError:
                    result_text = tool.execute(content)

                if result_text:
                    execution_time = time.time() - start_time
                    return {
                        "success": True,
                        "retry_count": retry_count,
                        "execution_time": execution_time,
                        "result": result_text,
                    }
                else:
                    logger.warning(
                        f"Tool({tool.trigger_tag}) returned empty output. "
                        f"Retrying {retry_count + 1}/{self.tool_retry_count}"
                    )
                    retry_count += 1
            except Exception as e:
                logger.error(
                    f"Tool({tool.trigger_tag}) execution failed. "
                    f"Retrying {retry_count + 1}/{self.tool_retry_count}: {e}"
                )
                retry_count += 1

        execution_time = time.time() - start_time
        logger.warning(
            f"Tool({tool.trigger_tag}) execution failed after {self.tool_retry_count} retries. Appending EOS."
        )
        return {
            "success": False,
            "retry_count": retry_count,
            "execution_time": execution_time,
            "result": "",
        }

    def _calc_entropy(self, logprobs):
        if not logprobs:
            return 0.0
        p_list = [math.exp(l) for l in logprobs]
        entropy = -sum(p * l for p, l in zip(p_list, logprobs))
        return entropy

    @staticmethod
    def _calc_entropy_from_logprob_dict(logprob_dict) -> float:
        if not logprob_dict:
            return 0.0
        log_probs = [float(v.logprob) for v in logprob_dict.values() if hasattr(v, "logprob")]
        if not log_probs:
            return 0.0
        probs = [math.exp(lp) for lp in log_probs]
        return max(0.0, -sum(p * lp for p, lp in zip(probs, log_probs) if p > 0))

    @staticmethod
    def _extract_sample_logprob(logprob_dict, token_id: int) -> float:
        if not logprob_dict:
            return 0.0
        entry = logprob_dict.get(token_id)
        if entry is None:
            entry = logprob_dict.get(str(token_id))
        if entry is not None and hasattr(entry, "logprob"):
            return float(entry.logprob)
        return 0.0

    # ============================================================
    # APPO: 认知思维模式分裂相关方法
    # ============================================================

    def _get_appo_brancher(self) -> APPOBrancher:
        """懒加载 APPOBrancher 实例。"""
        if self._appo_brancher is None:
            self._appo_brancher = APPOBrancher(
                tokenizer=self.tokenizer,
                reward_scale_discount=self.reward_scale_discount,
            )
        return self._appo_brancher

    def _appo_branch_after_rollout(
        self,
        completed_rollouts: List[List[int]],
        completed_logprobs: List[List[Optional[dict]]],
        prompt_token_ids_list: List[List[int]],
        rollouts_per_sample: List[int],
        sample_to_indices: Dict[int, List[int]],
        num_samples: int,
        max_len: int,
    ):
        """APPO 核心：在所有初始 rollout 完成后，对每条 rollout 的生成 token
        计算 rollout-level Branching Score 并进行 branching。

        Returns:
            (new_rollout_token_ids, new_result_masks, new_sample_origins, branch_meta_list)
            branch_meta_list: List of {
                "orig_rollout_idx": int,
                "orig_sample": int,
                "branch_token_response_idx": int,  # 分裂点在 response 中的索引
                "branch_score": float,              # 分裂点的 Branching Score
            }
        """
        brancher = self._get_appo_brancher()
        rollout_to_sample: Dict[int, int] = {}
        for sample_idx, indices in sample_to_indices.items():
            for rollout_idx in indices:
                rollout_to_sample[rollout_idx] = sample_idx

        # 收集每条 rollout 的候选分裂点
        all_rollout_candidates = []  # (rollout_idx, orig_sample, prompt_len, candidates)
        for rollout_idx, rollout_token_ids in enumerate(completed_rollouts):
            orig_sample = rollout_to_sample.get(rollout_idx)
            if orig_sample is None:
                continue
            prompt_len = len(prompt_token_ids_list[orig_sample])
            response_token_ids = rollout_token_ids[prompt_len:]
            response_logprobs = completed_logprobs[rollout_idx] if rollout_idx < len(completed_logprobs) else []

            candidates = brancher.compute_saliency_scores(response_token_ids, response_logprobs)
            if candidates:
                all_rollout_candidates.append((rollout_idx, orig_sample, prompt_len, candidates))

        if not all_rollout_candidates:
            return [], [], [], []

        # 对每条 rollout 按预算抽样分裂点，生成 branch 前缀
        branch_prefixes: List[List[int]] = []
        branch_meta: List[Dict] = []

        for rollout_idx, orig_sample, prompt_len, candidates in all_rollout_candidates:
            budget = num_samples - rollouts_per_sample[orig_sample]
            if budget <= 0:
                continue

            selected_points = brancher.sample_branch_points(candidates, budget)
            response_token_ids = completed_rollouts[rollout_idx][prompt_len:]

            for point in selected_points:
                branch_token_idx = point["response_token_idx"]
                # 分裂前缀 = prompt + response[:branch_token_idx+1]
                split_prefix = list(prompt_token_ids_list[orig_sample]) + response_token_ids[:branch_token_idx + 1]
                remaining_len = max_len - (branch_token_idx + 1)
                if remaining_len <= 0:
                    continue

                branch_prefixes.append(split_prefix)
                branch_meta.append({
                    "orig_rollout_idx": rollout_idx,
                    "orig_sample": orig_sample,
                    "prompt_len": prompt_len,
                    "branch_token_response_idx": branch_token_idx,
                    "branch_score": float(point.get("saliency", 0.0)),
                    "remaining_len": remaining_len,
                    "response_token_ids": response_token_ids,
                })
                rollouts_per_sample[orig_sample] += 1

        if not branch_prefixes:
            return [], [], [], []

        global_max_remaining = max(meta["remaining_len"] for meta in branch_meta)

        try:
            with self.update_sampling_params(
                n=1,
                stop=self.stop_sequences,
                max_tokens=global_max_remaining,
                detokenize=True,
                logprobs=0,
            ):
                branch_outputs = self.inference_engine.generate(
                    prompt_token_ids=branch_prefixes,
                    sampling_params=self.sampling_params,
                    use_tqdm=False,
                )
        except Exception as exc:
            logger.warning(f"APPO batch branch generation failed: {exc}")
            return [], [], [], []

        new_rollout_inputs: List[List[int]] = []
        new_result_masks: List[List[int]] = []
        new_sample_origins: List[int] = []
        new_branch_meta: List[Dict] = []

        for branch_idx, (meta, branch_output) in enumerate(zip(branch_meta, branch_outputs)):
            try:
                orig_sample = meta["orig_sample"]
                prompt_len = meta["prompt_len"]
                branch_token_idx = meta["branch_token_response_idx"]
                response_token_ids = meta["response_token_ids"]
                remaining_len = meta["remaining_len"]

                branch_generated = list(branch_output.outputs[0].token_ids)[:remaining_len]

                full_token_ids, result_mask = brancher.build_branch_rollout(
                    prompt_token_ids=prompt_token_ids_list[orig_sample],
                    response_token_ids=response_token_ids,
                    branch_token_idx=branch_token_idx,
                    branch_generated_ids=branch_generated,
                    max_response_len=max_len,
                )

                new_rollout_inputs.append(full_token_ids)
                new_result_masks.append(result_mask)
                new_sample_origins.append(orig_sample)
                new_branch_meta.append({
                    "orig_rollout_idx": meta["orig_rollout_idx"],  # curr_inputs 中原始 rollout 的索引
                    "orig_sample": orig_sample,
                    "branch_token_response_idx": branch_token_idx,
                    "branch_score": meta["branch_score"],
                    # orig_batch_row_idx 在 output 阶段写入（通过 curr_idx_to_batch_row 查表）
                })

                logger.info(
                    f"APPO: branch at rollout={meta['orig_rollout_idx']}, "
                    f"token_idx={branch_token_idx}, "
                    f"branch_score={meta['branch_score']:.4f}"
                )
            except Exception as exc:
                logger.warning(f"APPO branch assembly failed at index {branch_idx}: {exc}")

        return new_rollout_inputs, new_result_masks, new_sample_origins, new_branch_meta

    def _generate_appo_selected_branches(self, prompts: DataProto, eos_token_id: int) -> DataProto:
        prefix_ids_list = [list(x) for x in prompts.non_tensor_batch["appo_branch_prefix_ids"]]
        prompt_ids_list = [list(x) for x in prompts.non_tensor_batch["appo_branch_prompt_ids"]]
        remaining_lens = [int(x) for x in prompts.non_tensor_batch["appo_branch_remaining_len"]]
        branch_count = len(prefix_ids_list)

        max_prompt_len = prompts.batch["input_ids"].size(1)
        max_len = self.config.response_length
        curr_inputs = [prefix_ids.copy() for prefix_ids in prefix_ids_list]
        init_inputs = [prompt_ids.copy() for prompt_ids in prompt_ids_list]
        result_masks = [
            [0] * min(max_len, max(0, len(prefix_ids) - len(prompt_ids)))
            for prefix_ids, prompt_ids in zip(prefix_ids_list, prompt_ids_list)
        ]
        call_counters = []
        for prefix_ids in prefix_ids_list:
            prefix_text = self.tokenizer.decode(prefix_ids)
            call_counters.append(sum(prefix_text.count(stop_seq) for stop_seq in self.tool_stop_sequences))

        active_indices = [
            idx
            for idx, (prefix_ids, prompt_ids) in enumerate(zip(prefix_ids_list, prompt_ids_list))
            if len(prefix_ids) - len(prompt_ids) < max_len and remaining_lens[idx] > 0
        ]
        tool_metrics = {
            "tools/total_calls": 0,
            "tools/successful_calls": 0,
            "tools/failed_calls": 0,
            "tools/total_execution_time": 0.0,
            "tools/avg_execution_time": 0.0,
            "tools/max_execution_time": 0.0,
            "tools/max_retries": 0,
            "tools/total_retries": 0,
            "tools/call_limit_reached_count": 0,
        }
        calls_per_tool = Counter()
        success_per_tool = Counter()
        total_time_per_tool = Counter()

        while active_indices:
            active_prompts = [curr_inputs[i] for i in active_indices]
            max_tokens = max(
                1,
                max(max_len - (len(curr_inputs[i]) - len(init_inputs[i])) for i in active_indices),
            )
            with self.update_sampling_params(
                n=1,
                stop=self.stop_sequences,
                max_tokens=max_tokens,
                detokenize=True,
                logprobs=0,
                allowed_token_ids=list(self.tokenizer.get_vocab().values()),
            ):
                outputs = self.inference_engine.generate(
                    prompt_token_ids=active_prompts,
                    sampling_params=self.sampling_params,
                    use_tqdm=False,
                )

            tool_requests: Dict[str, List[Dict]] = {tag: [] for tag in self.tools}
            next_active_indices = []

            for i, out_idx in enumerate(active_indices):
                output = outputs[i]
                generated_tokens = list(output.outputs[0].token_ids)
                curr_inputs[out_idx].extend(generated_tokens)
                result_masks[out_idx].extend([0] * len(generated_tokens))

                finish_reason = output.outputs[0].finish_reason
                stop_reason = output.outputs[0].stop_reason
                is_tool_call = finish_reason == "stop" and stop_reason in self.tool_stop_sequences

                if is_tool_call:
                    tag = stop_reason.strip("</>")
                    if call_counters[out_idx] < self.tool_call_limit:
                        call_counters[out_idx] += 1
                        full_text = self.tokenizer.decode(curr_inputs[out_idx])
                        content = self._extract_content(full_text, tag)
                        if content:
                            tool_requests[tag].append({"index": out_idx, "content": content})
                            next_active_indices.append(out_idx)
                            tool_metrics["tools/total_calls"] += 1
                            calls_per_tool[tag] += 1
                    else:
                        logger.warning(f"Tool call limit reached for APPO branch sample {out_idx}. Appending EOS.")
                        curr_inputs[out_idx].append(eos_token_id)
                        result_masks[out_idx].append(0)
                        tool_metrics["tools/call_limit_reached_count"] += 1
                elif finish_reason == "length":
                    if len(curr_inputs[out_idx]) - len(init_inputs[out_idx]) < max_len:
                        next_active_indices.append(out_idx)

            if any(tool_requests.values()):
                futures = {}
                for tag, requests in tool_requests.items():
                    if not requests:
                        continue
                    tool = self.tools[tag]
                    for req in requests:
                        future = self.executor.submit(
                            self._execute_tool_with_retry, tool, req["content"], req["index"]
                        )
                        futures[future] = {"index": req["index"], "tag": tag}

                for future in concurrent.futures.as_completed(futures):
                    fut_info = futures[future]
                    idx = fut_info["index"]
                    tag = fut_info["tag"]
                    try:
                        result = future.result(timeout=self.tool_timeout)
                        success = result["success"]
                        retry_count = result["retry_count"]
                        execution_time = result["execution_time"]
                        result_text = result["result"]
                        if success:
                            tool_metrics["tools/successful_calls"] += 1
                            success_per_tool[tag] += 1
                        else:
                            tool_metrics["tools/failed_calls"] += 1
                            result_text = f"Tool({tag}) returned empty output."

                        tool_metrics["tools/total_execution_time"] += execution_time
                        tool_metrics["tools/max_execution_time"] = max(tool_metrics["tools/max_execution_time"], execution_time)
                        tool_metrics["tools/total_retries"] += retry_count
                        tool_metrics["tools/max_retries"] = max(tool_metrics["tools/max_retries"], retry_count)
                        total_time_per_tool[tag] += execution_time
                        if not result_text:
                            result_text = f"Tool({tag}) returned empty output."
                    except Exception as e:
                        logger.error(f"Tool({tag}) execution failed for APPO branch sample {idx}: {e}")
                        result_text = f"Error: Tool({tag}) execution failed with message: {e}"
                        tool_metrics["tools/failed_calls"] += 1

                    formatted_result = f" <result>\n{result_text}\n</result>"
                    result_tokens = self.tokenizer.encode(formatted_result, add_special_tokens=False)
                    curr_inputs[idx].extend(result_tokens)
                    result_masks[idx].extend([0] * len(result_tokens))

            final_active_indices = []
            for idx in next_active_indices:
                response_len = len(curr_inputs[idx]) - len(init_inputs[idx])
                if response_len < max_len:
                    final_active_indices.append(idx)
            active_indices = final_active_indices

        prompt_tensors = []
        response_tensors = []
        loss_mask_tensors = []
        for curr_ids, prompt_ids, result_mask in zip(curr_inputs, prompt_ids_list, result_masks):
            response_ids = curr_ids[len(prompt_ids):][:max_len]
            result_mask = result_mask[:len(response_ids)]
            prompt_tensor = torch.tensor(prompt_ids, dtype=prompts.batch["input_ids"].dtype, device=prompts.batch["input_ids"].device)
            if prompt_tensor.size(0) > max_prompt_len:
                prompt_tensor = prompt_tensor[-max_prompt_len:]
            elif prompt_tensor.size(0) < max_prompt_len:
                pad_len = max_prompt_len - prompt_tensor.size(0)
                prompt_tensor = torch.cat([
                    torch.full((pad_len,), self.pad_token_id, dtype=prompt_tensor.dtype, device=prompt_tensor.device),
                    prompt_tensor,
                ])
            prompt_tensors.append(prompt_tensor)

            response_tensor = torch.tensor(response_ids, dtype=prompts.batch["input_ids"].dtype, device=prompts.batch["input_ids"].device)
            response_tensor = pad_sequence_to_length(response_tensor, self.config.response_length, self.pad_token_id)
            response_tensors.append(response_tensor)
            result_mask_tensor = torch.tensor(result_mask, dtype=torch.long, device=prompts.batch["input_ids"].device)
            result_mask_tensor = pad_sequence_to_length(result_mask_tensor, self.config.response_length, 0)
            loss_mask_tensors.append(result_mask_tensor)

        input_ids = torch.stack(prompt_tensors, dim=0)
        response = torch.stack(response_tensors, dim=0)
        loss_mask = torch.stack(loss_mask_tensors, dim=0)
        attention_mask = (input_ids != self.pad_token_id).to(prompts.batch["attention_mask"].dtype)
        if prompts.batch["position_ids"].dim() == 3:
            position_ids = prompts.batch["position_ids"]
        else:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids = position_ids.masked_fill(attention_mask == 0, 0)

        response_attention_mask = get_response_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        final_attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)
        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device).unsqueeze(0).expand(branch_count, -1)
        if position_ids.dim() == 3:
            delta_position_id = delta_position_id.view(branch_count, 1, -1).expand(branch_count, position_ids.size(1), -1)
            response_position_ids = position_ids[..., -1:].expand(-1, position_ids.size(1), -1) + delta_position_id
        else:
            response_position_ids = position_ids[..., -1:] + delta_position_id
        final_position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        seq = torch.cat([input_ids, response], dim=-1)
        loss_mask = loss_mask * response_attention_mask

        batch = TensorDict({
            "prompts": input_ids,
            "responses": response,
            "input_ids": seq,
            "attention_mask": final_attention_mask,
            "loss_mask": loss_mask,
            "position_ids": final_position_ids,
        }, batch_size=branch_count)

        branch_meta = np.empty(branch_count, dtype=object)
        for i in range(branch_count):
            branch_meta[i] = {
                "orig_batch_row_idx": int(prompts.non_tensor_batch["appo_orig_batch_row_idx"][i]),
                "batch_row_idx": -1,
                "branch_token_response_idx": int(prompts.non_tensor_batch["appo_branch_token_response_idx"][i]),
                "branch_score": float(prompts.non_tensor_batch["appo_branch_score"][i]),
            }
        non_tensor_batch = {
            "appo_branch_meta": branch_meta,
            "appo_rollout_kind": np.array(["branch"] * branch_count, dtype=object),
        }
        if tool_metrics["tools/total_calls"] > 0:
            tool_metrics["tools/avg_execution_time"] = tool_metrics["tools/total_execution_time"] / tool_metrics["tools/total_calls"]
        tool_specific_metrics = {}
        for tag in self.tools.keys():
            calls = calls_per_tool[tag]
            if calls > 0:
                tool_specific_metrics[f"tools/{tag}/calls"] = calls
                tool_specific_metrics[f"tools/{tag}/avg_time"] = total_time_per_tool[tag] / calls
                tool_specific_metrics[f"tools/{tag}/success_rate"] = success_per_tool[tag] / calls
            else:
                tool_specific_metrics[f"tools/{tag}/calls"] = 0
                tool_specific_metrics[f"tools/{tag}/avg_time"] = 0
                tool_specific_metrics[f"tools/{tag}/success_rate"] = 0

        meta_info = deepcopy(prompts.meta_info)
        meta_info["metrics"] = {**tool_metrics, **tool_specific_metrics}
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=meta_info)

    @GPUMemoryLogger(role="vllm rollout spmd with tools", logger=logger)
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        if vllm_version in ('0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()

        input_ids = prompts.batch['input_ids']
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']
        eos_token_id = self.tokenizer.eos_token_id
        batch_size = input_ids.size(0)

        if prompts.meta_info.get("appo_branch_mode", False):
            data_proto = self._generate_appo_selected_branches(prompts, eos_token_id)
            if vllm_version in ('0.5.4', '0.6.3') and self.config.free_cache_engine:
                self.inference_engine.free_cache_engine()
            for sample_idx in range(batch_size):
                for tool in self.tools.values():
                    if hasattr(tool, "reset_sample"):
                        tool.reset_sample(sample_idx)
            return data_proto
        
        # 初始化工具调用统计信息
        tool_metrics = {
            "tools/total_calls": 0,
            "tools/successful_calls": 0,
            "tools/failed_calls": 0,
            "tools/total_execution_time": 0.0,
            "tools/avg_execution_time": 0.0,
            "tools/max_execution_time": 0.0,
            "tools/max_retries": 0,
            "tools/total_retries": 0,
            "tools/call_limit_reached_count": 0,
        }
        
        # 每个工具的统计信息
        calls_per_tool = Counter()
        success_per_tool = Counter()
        total_time_per_tool = Counter()

        do_sample = prompts.meta_info.get('do_sample', True)
        is_validate = prompts.meta_info.get('validate', False)
        
        # 更新采样参数设置
        if not do_sample:
            kwargs.update({
                'best_of': 1, 'top_p': 1.0, 'top_k': -1,
                'min_p': 0.0, 'temperature': 0, 'n': 1
            })
        elif is_validate:
            kwargs.update({
                'top_k': self.config.val_kwargs.top_k,
                'top_p': self.config.val_kwargs.top_p,
                'temperature': self.config.val_kwargs.temperature,
                'n': 1  # 验证模式下使用单个样本
            })
        
        # fix oov error: 允许 tokenizer 词表里的所有 token id（包括特殊 token），
        # 防止搜索结果或 COGNITIVE_ADDON 里的特殊 token 字符串被编码成超出 vLLM vocab_size 的 id
        kwargs["allowed_token_ids"] = list(self.tokenizer.get_vocab().values())

        with self.update_sampling_params(**kwargs):
            num_samples = self.sampling_params.n

            raw_prompt_token_ids_list = [_pre_process_inputs(self.pad_token_id, prompt) for prompt in input_ids]

            # ==== APPO: 动态注入 COGNITIVE_SYSTEM_PROMPT_ADDON ====
            # 默认关闭：未做认知 SFT 时 extra_0 会干扰 <think>/<search> 协议。
            # 需要论文形态时设置 APPO_INJECT_COGNITIVE=1。
            inject_cognitive = os.getenv("APPO_INJECT_COGNITIVE", "0").lower() not in (
                "0", "false", "no", "off",
            )
            if inject_cognitive:
                if self._cognitive_addon_token_ids is None:
                    # cognitive_pipeline is an APPO-only package; MAS never installs it by default.
                    from cognitive_pipeline.config import COGNITIVE_SYSTEM_PROMPT_ADDON as _COGNITIVE_ADDON

                    self._cognitive_addon_token_ids = self.tokenizer.encode(
                        _COGNITIVE_ADDON, add_special_tokens=False
                    )
                if self._im_end_token_id is None:
                    im_end_tokens = self.tokenizer.encode("<|im_end|>", add_special_tokens=False)
                    self._im_end_token_id = im_end_tokens[0] if im_end_tokens else -1

                cognitive_addon_token_ids: List[int] = self._cognitive_addon_token_ids or []
                im_end_token_id: int = self._im_end_token_id if self._im_end_token_id is not None else -1

                prompt_token_ids_list = []
                for raw_ids in raw_prompt_token_ids_list:
                    if im_end_token_id >= 0 and im_end_token_id in raw_ids:
                        insert_pos = raw_ids.index(im_end_token_id)
                        new_ids = (
                            list(raw_ids[:insert_pos])
                            + cognitive_addon_token_ids
                            + list(raw_ids[insert_pos:])
                        )
                    else:
                        new_ids = list(raw_ids)
                        logger.warning("APPO: could not find <|im_end|> token, skipping COGNITIVE_ADDON injection")
                    prompt_token_ids_list.append(new_ids)
            else:
                prompt_token_ids_list = [list(ids) for ids in raw_prompt_token_ids_list]
            # ==== END APPO ====

            # State for each sample in the batch
            # 为每个样本创建初始rollout，数量由initial_rollouts控制
            initial_rollouts = self.initial_rollouts
            initial_rollouts = min(initial_rollouts, num_samples)  # 但不超过num_samples

            curr_inputs = []
            init_inputs = []
            result_masks = []
            rollout_logprobs = []  # APPO: 收集每条 rollout 的 token-level logprobs
            call_counters = []
            active_indices = []
            
            # 创建初始样本
            for i, ids in enumerate(prompt_token_ids_list):
                for _ in range(initial_rollouts):
                    curr_inputs.append(ids.copy())
                    init_inputs.append(ids.copy())
                    result_masks.append([])
                    rollout_logprobs.append([])  # APPO: 初始化 logprobs 列表
                    call_counters.append(0)
                    active_indices.append(len(curr_inputs) - 1)
            
            # Track rollouts per original sample
            rollouts_per_sample = [initial_rollouts] * batch_size  # 每个样本初始有initial_rollouts个rollout
            # 初始时每个样本有多个索引
            sample_to_indices = {i: [i * initial_rollouts + j for j in range(initial_rollouts)] for i in range(batch_size)}

            max_len = self.config.response_length

            while active_indices:
                active_prompts = [curr_inputs[i] for i in active_indices]
                logger.debug(f"rollouts_per_sample: {rollouts_per_sample}")
                logger.debug(f"active_indices: {active_indices}")
                logger.debug(f"active_prompts: {active_prompts}")

                # Update max_tokens for each active sample
                with self.update_sampling_params(
                    n=1,
                    stop=self.stop_sequences,
                    max_tokens=max(1, max((max_len - (len(curr_inputs[i]) - len(init_inputs[i])) for i in active_indices))),
                    detokenize=True,
                    logprobs=10,  # APPO: 收集 logprobs 用于熵计算
                ):
                    outputs = self.inference_engine.generate(
                        prompt_token_ids=active_prompts,
                        sampling_params=self.sampling_params,
                        use_tqdm=False
                    )

                tool_requests: Dict[str, List[Dict]] = {tag: [] for tag in self.tools}
                next_active_indices = []

                for i, out_idx in enumerate(active_indices):
                    output = outputs[i]
                    generated_tokens = output.outputs[0].token_ids
                    



                    curr_inputs[out_idx].extend(generated_tokens)
                    result_masks[out_idx].extend([1] * len(generated_tokens))
                    # APPO: 收集 logprobs（对齐到 response token 位置）
                    if output.outputs[0].logprobs:
                        rollout_logprobs[out_idx].extend(output.outputs[0].logprobs)
                    else:
                        rollout_logprobs[out_idx].extend([None] * len(generated_tokens))

                    finish_reason = output.outputs[0].finish_reason
                    stop_reason = output.outputs[0].stop_reason



                    is_tool_call = finish_reason == 'stop' and stop_reason in self.tool_stop_sequences
                    
                    # Debug information
                    decoded_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                    logger.debug(f"  Sample {out_idx} output:")
                    logger.debug(f"  Token IDs: {generated_tokens}")
                    logger.debug(f"  Text: {decoded_text}")
                    logger.debug(f"  Finish reason: {finish_reason}")
                    logger.debug(f"  Stop reason: {stop_reason}")
                    logger.debug(f"  Is tool call: {is_tool_call}")
                    logger.debug(f"  Tool: {stop_reason.strip('</>') if is_tool_call else 'No tool call'}")

                    if is_tool_call:
                        tag = stop_reason.strip("</>")
                        if call_counters[out_idx] < self.tool_call_limit:
                            call_counters[out_idx] += 1
                            full_text = self.tokenizer.decode(curr_inputs[out_idx])
                            content = self._extract_content(full_text, tag)
                            if content:
                                tool_requests[tag].append({"index": out_idx, "content": content})
                                next_active_indices.append(out_idx)
                                # 更新工具调用计数统计
                                tool_metrics["tools/total_calls"] += 1
                                calls_per_tool[tag] += 1
                        else:
                            logger.warning(f"Tool call limit reached for sample {out_idx}. Appending EOS.")
                            curr_inputs[out_idx].append(eos_token_id)
                            result_masks[out_idx].append(1)
                            tool_metrics["tools/call_limit_reached_count"] += 1

                    elif finish_reason == 'length':
                        if len(curr_inputs[out_idx]) - len(init_inputs[out_idx]) < max_len:
                            next_active_indices.append(out_idx)

                    elif finish_reason == 'stop':  # EOS
                        pass

                if any(tool_requests.values()):
                    logger.info(f"Processing tool requests: {sum(len(reqs) for reqs in tool_requests.values())} total requests")
                    futures = {}
                    for tag, requests in tool_requests.items():
                        if not requests:
                            continue
                        logger.debug(f"Processing {len(requests)} requests for tool '{tag}'")
                        tool = self.tools[tag]
                        for req in requests:
                            logger.debug(f"Submitting tool request: tool={tag}, idx={req['index']}, content={req['content']}")
                            future = self.executor.submit(
                                self._execute_tool_with_retry, tool, req["content"], req["index"]
                            )
                            futures[future] = {"index": req["index"], "tag": tag}

                    total_futures = len(futures)
                    completed_futures = 0
                    logger.debug(f"Submitted {total_futures} tool requests for execution")
                    for future in concurrent.futures.as_completed(futures):
                        completed_futures += 1
                        fut_info = futures[future]
                        idx = fut_info["index"]
                        tag = fut_info["tag"]
                        try:
                            result = future.result(timeout=self.tool_timeout)
                            # 解析工具执行结果
                            success = result["success"]
                            retry_count = result["retry_count"]
                            execution_time = result["execution_time"]
                            result_text = result["result"]
                            
                            # 更新统计信息
                            if success:
                                tool_metrics["tools/successful_calls"] += 1
                                success_per_tool[tag] += 1
                                logger.info(f"Tool({tag}) for sample {idx} completed successfully in {execution_time:.2f}s, result length: {len(result_text)}")
                            else:
                                tool_metrics["tools/failed_calls"] += 1
                                result_text = f"Tool({tag}) returned empty output."
                                logger.warning(f"Tool({tag}) for sample {idx} failed after {retry_count} retries, execution time: {execution_time:.2f}s")
                            
                            tool_metrics["tools/total_execution_time"] += execution_time
                            tool_metrics["tools/max_execution_time"] = max(tool_metrics["tools/max_execution_time"], execution_time)
                            tool_metrics["tools/total_retries"] += retry_count
                            tool_metrics["tools/max_retries"] = max(tool_metrics["tools/max_retries"], retry_count)
                            
                            # 更新每个工具的时间统计
                            total_time_per_tool[tag] += execution_time
                            
                            if not result_text:
                                result_text = f"Tool({tag}) returned empty output."
                                logger.warning(f"Tool({tag}) for sample {idx} returned empty output, execution time: {execution_time:.2f}s")
                            else:
                                logger.debug(f"Tool({tag}) result: {result_text}")
                                
                        except Exception as e:
                            logger.error(f"Tool({tag}) execution failed for sample {idx}: {e}")
                            result_text = f"Error: Tool({tag}) execution failed with message: {e}"
                            tool_metrics["tools/failed_calls"] += 1
                        
                        logger.debug(f"Tool completion progress: {completed_futures}/{total_futures} ({completed_futures/total_futures*100:.1f}%)")
                        formatted_result = f" <result>\n{result_text}\n</result>"
                        # add_special_tokens=False 防止搜索结果里包含的特殊 token 字符串
                        # （如 <|im_start|>、<|endoftext|> 等）被编码成 OOV id，导致 vLLM 校验报错
                        result_tokens = self.tokenizer.encode(formatted_result, add_special_tokens=False)
                        logger.debug(f"Result for tool({tag}), sample {idx} tokenized to {len(result_tokens)} tokens")
                        curr_inputs[idx].extend(result_tokens)
                        result_masks[idx].extend([0] * len(result_tokens))

                final_active_indices = []
                for idx in next_active_indices:
                    response_len = len(curr_inputs[idx]) - len(init_inputs[idx])
                    if response_len < max_len:
                        final_active_indices.append(idx)

                active_indices = final_active_indices

            # ==== APPO: 在所有初始 rollout 完成后，进行 procedure-level branching ====
            branch_meta_list = []
            if do_sample and not is_validate and not self.appo_dynamic_branching:
                branch_inputs, branch_masks, branch_origins, branch_meta_list = self._appo_branch_after_rollout(
                    completed_rollouts=curr_inputs,
                    completed_logprobs=rollout_logprobs,
                    prompt_token_ids_list=prompt_token_ids_list,
                    rollouts_per_sample=rollouts_per_sample,
                    sample_to_indices=sample_to_indices,
                    num_samples=num_samples,
                    max_len=max_len,
                )
                if branch_inputs:
                    start_idx = len(curr_inputs)
                    curr_inputs.extend(branch_inputs)
                    init_inputs.extend([
                        prompt_token_ids_list[orig_sample]
                        for orig_sample in branch_origins
                    ])
                    result_masks.extend(branch_masks)
                    # Fix 1: rollout_logprobs 需要同步扩展，否则与 curr_inputs 长度不一致
                    rollout_logprobs.extend([[] for _ in branch_inputs])
                    for new_idx_offset, orig_sample in enumerate(branch_origins):
                        curr_inputs_idx = start_idx + new_idx_offset
                        sample_to_indices.setdefault(orig_sample, []).append(curr_inputs_idx)
                        # Fix 4: 直接把 curr_inputs 索引写入 branch_meta，供输出阶段查表
                        if new_idx_offset < len(branch_meta_list):
                            branch_meta_list[new_idx_offset]["curr_inputs_idx"] = curr_inputs_idx
                    logger.info(f"APPO: created {len(branch_inputs)} token-level branches in total")
            # ==== END APPO ====

            # 确保所有序列不超过max_len
            for idx in range(len(curr_inputs)):
                response_len = len(curr_inputs[idx]) - len(init_inputs[idx])
                if response_len > max_len:
                    offset = len(init_inputs[idx])
                    curr_inputs[idx] = curr_inputs[idx][:offset + max_len]
                    result_masks[idx] = result_masks[idx][:max_len]
            
            # Reorganize outputs to match original batch structure and select up to num_samples per sample
            output_sequences = []
            output_result_masks = []
            output_rollout_kinds = []
            output_rollout_log_probs = []
            output_rollout_entropys = []
            output_repeat_times = initial_rollouts if self.appo_dynamic_branching and do_sample and not is_validate else num_samples
            # Fix 4: 记录每个 curr_inputs 索引在最终 batch 中的行号，供 branch_meta 使用
            curr_idx_to_batch_row = {}
            branch_curr_indices = {
                meta.get("curr_inputs_idx")
                for meta in branch_meta_list
                if isinstance(meta, dict) and meta.get("curr_inputs_idx", -1) >= 0
            }
            batch_row_counter = 0
            for i in range(batch_size):
                # Get all indices for this sample
                sample_indices = sample_to_indices.get(i, [])
                # Ensure we have exactly num_samples outputs per sample
                selected_indices = sample_indices[:output_repeat_times]
                
                # If we have fewer rollouts than requested, duplicate the last one
                while len(selected_indices) < output_repeat_times:
                    if selected_indices:
                        selected_indices.append(selected_indices[-1])
                    else:
                        break  # Should not happen but just in case
                        
                # Extract outputs for selected indices
                for idx in selected_indices:
                    output_sequences.append(curr_inputs[idx][len(prompt_token_ids_list[i]):])
                    output_result_masks.append(result_masks[idx])
                    output_rollout_kinds.append("branch" if idx in branch_curr_indices else "init")
                    selected_logprobs = rollout_logprobs[idx] if idx < len(rollout_logprobs) else []
                    response_token_ids = curr_inputs[idx][len(prompt_token_ids_list[i]):]
                    token_log_probs = []
                    token_entropys = []
                    for token_id, logprob_dict in zip(response_token_ids, selected_logprobs):
                        token_log_probs.append(self._extract_sample_logprob(logprob_dict, token_id))
                        token_entropys.append(self._calc_entropy_from_logprob_dict(logprob_dict))
                    while len(token_log_probs) < len(response_token_ids):
                        token_log_probs.append(0.0)
                        token_entropys.append(0.0)
                    output_rollout_log_probs.append(token_log_probs[:len(response_token_ids)])
                    output_rollout_entropys.append(token_entropys[:len(response_token_ids)])
                    # Fix 25: 只记录第一次出现的 batch_row，避免重复 idx（补齐时）覆盖正确映射
                    curr_idx_to_batch_row.setdefault(idx, batch_row_counter)
                    batch_row_counter += 1

            # Fix 4: 用 curr_inputs_idx 直接查表，精确写入 batch_row_idx 和 orig_batch_row_idx
            for meta in branch_meta_list:
                curr_inputs_idx = meta.get("curr_inputs_idx", -1)
                if curr_inputs_idx >= 0 and curr_inputs_idx in curr_idx_to_batch_row:
                    meta["batch_row_idx"] = curr_idx_to_batch_row[curr_inputs_idx]
                # 同时写入原始 rollout 的 batch 行索引，供 reward scaling 使用
                orig_curr_inputs_idx = meta.get("orig_rollout_idx", -1)
                if orig_curr_inputs_idx >= 0 and orig_curr_inputs_idx in curr_idx_to_batch_row:
                    meta["orig_batch_row_idx"] = curr_idx_to_batch_row[orig_curr_inputs_idx]

            padded_response_list = []
            padded_result_mask_list = []
            padded_rollout_log_prob_list = []
            padded_rollout_entropy_list = []
            for output_ids, result_mask in zip(output_sequences, output_result_masks):
                logger.debug(f"len(output_ids): {len(output_ids)}, len(result_mask): {len(result_mask)}, output_ids: {output_ids}, result_mask: {result_mask}")
                
                assert len(output_ids) == len(result_mask), f"output_ids: {len(output_ids)}, result_mask: {len(result_mask)}"
                
                response = torch.tensor(output_ids)
                response = pad_sequence_to_length(response, self.config.response_length, self.pad_token_id)
                
                result_mask_tensor = torch.tensor(result_mask)
                result_mask_tensor = pad_sequence_to_length(result_mask_tensor, self.config.response_length, 0)
                
                padded_response_list.append(response)
                padded_result_mask_list.append(result_mask_tensor)

            for token_log_probs, token_entropys in zip(output_rollout_log_probs, output_rollout_entropys):
                log_prob_tensor = torch.tensor(token_log_probs, dtype=torch.float32)
                entropy_tensor = torch.tensor(token_entropys, dtype=torch.float32)
                padded_rollout_log_prob_list.append(pad_sequence_to_length(log_prob_tensor, self.config.response_length, 0.0))
                padded_rollout_entropy_list.append(pad_sequence_to_length(entropy_tensor, self.config.response_length, 0.0))
            
            response = torch.stack(padded_response_list, dim=0).to(input_ids.device)
            loss_mask = torch.stack(padded_result_mask_list, dim=0).to(input_ids.device)
            rollout_log_probs_tensor = torch.stack(padded_rollout_log_prob_list, dim=0).to(input_ids.device)
            rollout_entropys_tensor = torch.stack(padded_rollout_entropy_list, dim=0).to(input_ids.device)
            
            non_tensor_batch = deepcopy(prompts.non_tensor_batch)
            # Fix 5: appo_branch_meta 单独保存，不参与 repeat_interleave 展开
            # 在 repeat_interleave 之后再写入，避免被重复 num_samples 次
            _appo_branch_meta_to_attach = branch_meta_list

            # ==== P0 fix: 用注入了 COGNITIVE_ADDON 的 prompt 替换原始 input_ids ====
            # APPO 在 rollout 时动态注入了 COGNITIVE_ADDON，prompt_token_ids_list 是注入后的长 prompt。
            # response 是从注入后的长 prompt 之后切出来的。
            # 若不替换，seq = cat([原始短 input_ids, response]) 中 prompt/response 边界错位，
            # 导致 old_log_prob / ref_log_prob / KL 全部算错。
            # 修复：把 input_ids 替换成 padded 的注入后 prompt tensor，与 response 边界对齐。
            max_prompt_len = input_ids.size(1)  # 原始 prompt padding 长度（来自 dataloader）
            cognitive_prompt_tensors = []
            for cognitive_ids in prompt_token_ids_list:
                t = torch.tensor(cognitive_ids, dtype=input_ids.dtype, device=input_ids.device)
                if t.size(0) > max_prompt_len:
                    # COGNITIVE_ADDON 注入后超出 max_prompt_len：截断头部（保留最新 token）
                    t = t[-max_prompt_len:]
                elif t.size(0) < max_prompt_len:
                    # 不足则左侧 pad（与 dataloader 的 left-padding 对齐）
                    pad_len = max_prompt_len - t.size(0)
                    t = torch.cat([torch.full((pad_len,), self.pad_token_id, dtype=t.dtype, device=t.device), t])
                cognitive_prompt_tensors.append(t)
            input_ids = torch.stack(cognitive_prompt_tensors, dim=0)  # (batch_size, max_prompt_len)
            # attention_mask 同步更新：非 pad 位置为 1
            attention_mask = (input_ids != self.pad_token_id).to(attention_mask.dtype)
            # position_ids 同步更新：基于新 attention_mask 重新计算累积位置
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids = position_ids.masked_fill(attention_mask == 0, 0)
            # ==== END P0 fix ====

            if output_repeat_times > 1 and do_sample:
                input_ids = _repeat_interleave(input_ids, output_repeat_times)
                attention_mask = _repeat_interleave(attention_mask, output_repeat_times)
                position_ids = _repeat_interleave(position_ids, output_repeat_times)
                if non_tensor_batch:
                    for key, value in non_tensor_batch.items():
                        if isinstance(value, np.ndarray):
                            non_tensor_batch[key] = np.repeat(value, output_repeat_times, axis=0)
                        elif isinstance(value, list):
                            non_tensor_batch[key] = [item for item in value for _ in range(output_repeat_times)]

            final_batch_size = input_ids.size(0)
            # Fix 5: repeat_interleave 之后再写入 branch_meta，避免被重复展开。
            # DataProto.check_consistency 要求 non_tensor_batch 的值为 np.ndarray 且长度等于 batch_size。
            # 这里将 branch_meta 填充/截断为与 final_batch_size 一致的 object 数组，
            # trainer 侧会过滤掉 None，仅保留 dict 元素。
            branch_meta_arr = np.empty(final_batch_size, dtype=object)
            branch_meta_arr[:] = None
            meta_count = min(len(_appo_branch_meta_to_attach), final_batch_size)
            if meta_count > 0:
                branch_meta_arr[:meta_count] = _appo_branch_meta_to_attach[:meta_count]
            non_tensor_batch["appo_branch_meta"] = branch_meta_arr
            non_tensor_batch["appo_rollout_kind"] = np.array(output_rollout_kinds, dtype=object)

            seq = torch.cat([input_ids, response], dim=-1)

            response_length = response.size(1)
            delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device).unsqueeze(0).expand(final_batch_size, -1)

            if position_ids.dim() == 3:  # for RoPE scaling like qwen2vl mrope
                delta_position_id = delta_position_id.view(final_batch_size, 1, -1).expand(final_batch_size, position_ids.size(1), -1)
                response_position_ids = position_ids[..., -1:].expand(-1, position_ids.size(1), -1) + delta_position_id
            else:
                response_position_ids = position_ids[..., -1:] + delta_position_id

            final_position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

            response_attention_mask = get_response_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
            final_attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

            loss_mask = loss_mask * response_attention_mask

            # 计算平均执行时间
            if tool_metrics["tools/total_calls"] > 0:
                tool_metrics["tools/avg_execution_time"] = tool_metrics["tools/total_execution_time"] / tool_metrics["tools/total_calls"]
                
            # 计算每个工具的平均执行时间和成功率
            tool_specific_metrics = {}
            for tag in self.tools.keys():
                calls = calls_per_tool[tag]
                if calls > 0:
                    tool_specific_metrics[f"tools/{tag}/calls"] = calls
                    tool_specific_metrics[f"tools/{tag}/avg_time"] = total_time_per_tool[tag] / calls
                    tool_specific_metrics[f"tools/{tag}/success_rate"] = success_per_tool[tag] / calls
                else:
                    tool_specific_metrics[f"tools/{tag}/calls"] = 0
                    tool_specific_metrics[f"tools/{tag}/avg_time"] = 0
                    tool_specific_metrics[f"tools/{tag}/success_rate"] = 0

            batch = TensorDict({
                "prompts": input_ids,
                "responses": response,
                "input_ids": seq,
                "attention_mask": final_attention_mask,
                "loss_mask": loss_mask,
                "position_ids": final_position_ids,
                "rollout_log_probs": rollout_log_probs_tensor,
                "rollout_entropys": rollout_entropys_tensor,
            }, batch_size=final_batch_size)

        if vllm_version in ('0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()

        # 清理本次 rollout 中各 sample 的持久化 namespace（避免内存泄漏）。
        # 只对支持 reset_sample 接口的工具（如 PythonTool）执行清理。
        for sample_idx in range(batch_size):
            for tool in self.tools.values():
                if hasattr(tool, "reset_sample"):
                    tool.reset_sample(sample_idx)

        # 合并所有metrics
        all_metrics = {**tool_metrics, **tool_specific_metrics}
        
        # 将metrics添加到meta_info中
        meta_info = deepcopy(prompts.meta_info) if prompts.meta_info else {}
        meta_info["metrics"] = all_metrics

        data_proto = DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=meta_info)

        return data_proto
