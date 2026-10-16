# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import csv
import json
import os
import uuid
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Dict, Optional, Type

import numpy as np
import ray
import torch
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tensordict import TensorDict
from tqdm import tqdm

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.metric import (
    reduce_metrics,
)
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from verl.workers.rollout.async_server import AsyncLLMServerManager

WorkerType = Type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


class AdvantageEstimator(str, Enum):
    """
    Using an enumeration class to avoid spelling errors in adv_estimator
    """

    GAE = "gae"
    GRPO = "grpo"
    APPO = "appo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REINFORCE_PLUS_PLUS_BASELINE = "reinforce_plus_plus_baseline"
    REMAX = "remax"
    RLOO = "rloo"
    OPO = "opo"
    GRPO_PASSK = "grpo_passk"


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0) for node, node_info in node_available_resources.items()}

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])
        if total_available_gpus < total_required_gpus:
            raise ValueError(f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}")

        # check each resource pool can be satisfied, O(#resource_pools * #nodes)
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes}" + "cannot be satisfied in this ray cluster")


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl", multi_turn=False):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    if multi_turn:
        loss_mask = data.batch["loss_mask"]
        response_mask = loss_mask[:, -response_length:]
    else:
        attention_mask = data.batch["attention_mask"]
        response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty)  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def apply_appo_reward_scale_M(data, branch_meta_list, reward_scale_discount=0.9):
    """
    对 token_level_rewards 应用 APPO 的 reward 缩放因子 M。

    branch_meta_list: List of {
        "orig_rollout_idx": int,       # 原始 rollout 在 batch 中的索引
        "orig_sample": int,            # 所属 prompt 的索引
        "branch_token_response_idx": int,  # 分裂点在 response 中的索引
        "branch_score": float,         # 选中分裂点的 Branching Score
    }
    """
    import numpy as np
    from verl.workers.rollout.vllm_rollout.appo_branching import compute_reward_scale_M

    if branch_meta_list is None:
        return data
    if isinstance(branch_meta_list, np.ndarray):
        branch_meta_list = branch_meta_list.tolist()
    # 过滤非 dict 占位元素（如 None），只保留有效 meta
    branch_meta_list = [m for m in branch_meta_list if isinstance(m, dict)]
    if len(branch_meta_list) == 0:
        return data

    token_level_rewards = data.batch["token_level_rewards"]  # (bs, response_length)
    response_length = token_level_rewards.size(1)

    # 只用有效的 branch_meta（有 batch_row_idx 的）计算 max score。
    # 避免未落地的 branch 拉大归一化分母导致 M 偏小。
    valid_metas = [m for m in branch_meta_list if m.get("batch_row_idx", -1) >= 0]
    if not valid_metas:
        return data
    max_branch_score = max(float(meta.get("branch_score", 0.0)) for meta in valid_metas)
    if max_branch_score <= 0:
        return data

    rewards_np = token_level_rewards.cpu().numpy()

    # orig rollout：可能有多个 branch，取所有配对 M 的平均值，避免被某一个 M 偏置。
    # branch rollout 是辅助信号，不直接优化 actor；保持其 reward 原值用于对比估计。
    orig_M_accumulator = {}  # orig_batch_row -> List[np.ndarray]

    for meta in valid_metas:
        orig_batch_row = meta.get("orig_batch_row_idx", -1)

        M = compute_reward_scale_M(
            branch_token_response_idx=meta["branch_token_response_idx"],
            branch_score=float(meta.get("branch_score", 0.0)),
            max_branch_score=max_branch_score,
            response_length=response_length,
            discount=reward_scale_discount,
        )

        # 收集 orig rollout 的 M，供后续取平均
        if orig_batch_row >= 0 and orig_batch_row < rewards_np.shape[0]:
            orig_M_accumulator.setdefault(orig_batch_row, []).append(M)

    # orig rollout 取所有配对 M 的平均值后应用
    for orig_batch_row, M_list in orig_M_accumulator.items():
        M_avg = np.mean(np.stack(M_list, axis=0), axis=0)
        rewards_np[orig_batch_row] *= M_avg

    data.batch["token_level_rewards"] = torch.tensor(
        rewards_np, dtype=token_level_rewards.dtype, device=token_level_rewards.device
    )
    return data


def compute_appo_advantage(data, norm_by_std=True):
    """
    APPO 双组 GRPO advantage：
    - T_init：同一 prompt 的 initial rollouts 单独成组
    - T_branch：同一 prompt 的 branch rollouts 单独成组

    Branch rows are kept in the batch so their rewards can provide auxiliary
    contrastive signals. Their final token advantages are mapped back to the
    corresponding initial rollout branch prefixes, while branch rows themselves
    keep zero final advantage/loss contribution.
    """
    import numpy as np

    if "response_mask" not in data.batch:
        data.batch["response_mask"] = compute_response_mask(data)

    token_level_rewards = data.batch["token_level_rewards"]
    response_mask = data.batch["response_mask"]
    scores = token_level_rewards.sum(dim=-1)  # (bs,)

    tree_ids = data.non_tensor_batch["uid"]  # (bs,) prompt uid，同一 prompt 的 rollout uid 相同
    rollout_kind = data.non_tensor_batch.get("appo_rollout_kind", None)
    if rollout_kind is None:
        rollout_kind = np.array(["init"] * len(tree_ids), dtype=object)
    elif isinstance(rollout_kind, np.ndarray):
        rollout_kind = rollout_kind.astype(object)
    else:
        rollout_kind = np.array(rollout_kind, dtype=object)

    bsz = scores.shape[0]
    base_adv_values = torch.zeros(bsz, dtype=scores.dtype, device=scores.device)
    group_to_indices = defaultdict(list)
    for idx in range(bsz):
        group_key = (str(tree_ids[idx]), str(rollout_kind[idx]))
        group_to_indices[group_key].append(idx)

    for indices in group_to_indices.values():
        group_scores = scores[indices]
        mean_val = group_scores.mean()
        std_val = group_scores.std(correction=0) if len(indices) > 1 else torch.tensor(0.0, device=scores.device)
        for idx in indices:
            if norm_by_std:
                base_adv_values[idx] = (scores[idx] - mean_val) / (std_val + 1e-6)
            else:
                base_adv_values[idx] = scores[idx] - mean_val

    response_length = response_mask.size(1)
    advantages = torch.zeros_like(response_mask, dtype=scores.dtype)

    init_rows = np.array([str(kind) == "init" for kind in rollout_kind], dtype=bool)
    if init_rows.any():
        init_mask = torch.tensor(init_rows, dtype=torch.bool, device=scores.device)
        advantages[init_mask] = base_adv_values[init_mask].unsqueeze(-1) * response_mask[init_mask]

    branch_meta_list = data.non_tensor_batch.get("appo_branch_meta", [])
    if isinstance(branch_meta_list, np.ndarray):
        branch_meta_list = branch_meta_list.tolist()

    aux_sum = torch.zeros_like(advantages)
    aux_count = torch.zeros_like(advantages)
    for meta in branch_meta_list:
        if not isinstance(meta, dict):
            continue
        orig_row = int(meta.get("orig_batch_row_idx", -1))
        branch_row = int(meta.get("batch_row_idx", -1))
        split_idx = int(meta.get("branch_token_response_idx", -1))
        if orig_row < 0 or orig_row >= bsz or branch_row < 0 or branch_row >= bsz or split_idx < 0:
            continue
        if str(rollout_kind[orig_row]) != "init" or str(rollout_kind[branch_row]) != "branch":
            continue

        end = min(split_idx + 1, response_length)
        if end <= 0:
            continue
        valid_prefix = response_mask[orig_row, :end].to(dtype=scores.dtype)
        aux_sum[orig_row, :end] += base_adv_values[branch_row] * valid_prefix
        aux_count[orig_row, :end] += valid_prefix

    has_aux = aux_count > 0
    advantages = torch.where(has_aux, advantages + aux_sum / torch.clamp(aux_count, min=1.0), advantages)
    advantages = advantages * response_mask

    data.batch["advantages"] = advantages
    data.batch["returns"] = advantages
    return data


def _tensor_zscore(values: torch.Tensor) -> torch.Tensor:
    if values.numel() == 0:
        return values
    return (values - values.mean()) / (values.std(correction=0) + 1e-6)


def build_appo_branch_request(data, tokenizer, max_branches_per_prompt: int, gamma: float, clip_eps: float):
    if max_branches_per_prompt <= 0:
        return None
    if any(key not in data.batch.keys() for key in ("rollout_log_probs", "rollout_entropys", "old_log_probs")):
        return None

    responses = data.batch["responses"]
    prompts = data.batch["prompts"]
    attention_mask = data.batch["attention_mask"]
    response_mask = data.batch["loss_mask"] if "loss_mask" in data.batch.keys() else data.batch["response_mask"]
    response_length = responses.size(1)
    if response_mask.size(1) != response_length:
        response_mask = response_mask[:, -response_length:]

    current_log_probs = data.batch["old_log_probs"]
    behavior_log_probs = data.batch["rollout_log_probs"]
    rollout_entropys = data.batch["rollout_entropys"]
    tree_ids = data.non_tensor_batch["uid"]
    rollout_kind = data.non_tensor_batch.get("appo_rollout_kind", np.array(["init"] * len(tree_ids), dtype=object))

    grouped_candidates = defaultdict(list)
    grouped_fallback_candidates = defaultdict(list)
    for row in range(responses.size(0)):
        if str(rollout_kind[row]) != "init":
            continue
        valid_positions = response_mask[row].bool()
        if not valid_positions.any():
            continue

        log_ratio = current_log_probs[row] - behavior_log_probs[row]
        positions = torch.arange(response_length, device=log_ratio.device)
        distance = positions.unsqueeze(0) - positions.unsqueeze(1)
        gamma_t = torch.tensor(gamma, dtype=log_ratio.dtype, device=log_ratio.device)
        decay = torch.where(
            distance >= 0,
            torch.pow(gamma_t, distance.clamp(min=0)),
            torch.zeros((), dtype=log_ratio.dtype, device=log_ratio.device),
        )
        future_log_ratio = torch.matmul((log_ratio * valid_positions.to(log_ratio.dtype)).unsqueeze(0), decay.t()).squeeze(0)
        future_values = torch.exp(torch.clamp(future_log_ratio, min=-20.0, max=20.0))
        if clip_eps > 0:
            future_values = torch.clamp(future_values, min=1.0 - clip_eps, max=1.0 + clip_eps)

        entropy_z = torch.zeros_like(rollout_entropys[row])
        future_z = torch.zeros_like(future_values)
        entropy_z[valid_positions] = _tensor_zscore(rollout_entropys[row][valid_positions])
        future_z[valid_positions] = _tensor_zscore(future_values[valid_positions])
        branch_scores = entropy_z * future_z

        for token_idx in torch.nonzero(valid_positions, as_tuple=False).flatten().tolist():
            token_id = int(responses[row, token_idx].item())
            token_str = tokenizer.decode([token_id])
            if not token_str.strip():
                continue
            candidate = (float(branch_scores[token_idx].item()), row, token_idx)
            grouped_fallback_candidates[str(tree_ids[row])].append(candidate)
            if entropy_z[token_idx] <= 0 or future_z[token_idx] <= 0 or branch_scores[token_idx] <= 0:
                continue
            grouped_candidates[str(tree_ids[row])].append(candidate)

    prefix_ids = []
    prompt_ids = []
    orig_rows = []
    branch_token_indices = []
    branch_scores = []
    remaining_lens = []
    input_rows = []
    for tree_id, fallback_candidates in grouped_fallback_candidates.items():
        strict_candidates = grouped_candidates.get(tree_id, [])
        strict_candidates.sort(key=lambda x: x[0], reverse=True)
        fallback_candidates.sort(key=lambda x: x[0], reverse=True)
        selected = strict_candidates[:max_branches_per_prompt]
        if len(selected) < max_branches_per_prompt:
            selected_keys = {(row, token_idx) for _, row, token_idx in selected}
            for candidate in fallback_candidates:
                _, row, token_idx = candidate
                if (row, token_idx) in selected_keys:
                    continue
                selected.append(candidate)
                selected_keys.add((row, token_idx))
                if len(selected) >= max_branches_per_prompt:
                    break
        fill_idx = 0
        while selected and len(selected) < max_branches_per_prompt:
            selected.append(selected[fill_idx % len(selected)])
            fill_idx += 1

        for score, row, token_idx in selected:
            prompt = prompts[row]
            prompt_valid = prompt[prompt != tokenizer.pad_token_id].detach().cpu().tolist()
            response_prefix = responses[row, : token_idx + 1].detach().cpu().tolist()
            response_prefix = [tid for tid in response_prefix if tid != tokenizer.pad_token_id]
            remaining_len = response_length - len(response_prefix)
            if remaining_len <= 0:
                continue
            prefix_ids.append(prompt_valid + response_prefix)
            prompt_ids.append(prompt_valid)
            orig_rows.append(row)
            branch_token_indices.append(token_idx)
            branch_scores.append(score)
            remaining_lens.append(remaining_len)
            input_rows.append(row)

    if not prefix_ids:
        return None

    request_attention_mask = data.batch["attention_mask"][input_rows, : data.batch["prompts"].size(1)]
    request_position_ids = data.batch["position_ids"]
    if request_position_ids.dim() == 3:
        request_position_ids = request_position_ids[input_rows, :, : data.batch["prompts"].size(1)]
    else:
        request_position_ids = request_position_ids[input_rows, : data.batch["prompts"].size(1)]
    request_batch = TensorDict({
        "input_ids": data.batch["prompts"][input_rows],
        "attention_mask": request_attention_mask,
        "position_ids": request_position_ids,
    }, batch_size=len(prefix_ids))
    return DataProto(
        batch=request_batch,
        non_tensor_batch={
            "appo_branch_prefix_ids": np.array(prefix_ids, dtype=object),
            "appo_branch_prompt_ids": np.array(prompt_ids, dtype=object),
            "appo_orig_batch_row_idx": np.array(orig_rows, dtype=np.int64),
            "appo_branch_token_response_idx": np.array(branch_token_indices, dtype=np.int64),
            "appo_branch_score": np.array(branch_scores, dtype=np.float32),
            "appo_branch_remaining_len": np.array(remaining_lens, dtype=np.int64),
        },
        meta_info={"appo_branch_mode": True, "do_sample": True},
    )


def align_appo_branch_batch(init_batch: DataProto, branch_batch: DataProto):
    init_bsz = len(init_batch.batch)
    branch_bsz = len(branch_batch.batch)
    if branch_bsz == 0:
        return init_batch

    branch_batch.batch["response_mask"] = compute_response_mask(branch_batch)
    response_length = branch_batch.batch["responses"].size(1)
    for key in ("old_log_probs", "rollout_log_probs", "rollout_entropys"):
        if key in init_batch.batch and key not in branch_batch.batch:
            branch_batch.batch[key] = torch.zeros(
                branch_bsz,
                response_length,
                dtype=init_batch.batch[key].dtype,
                device=branch_batch.batch["responses"].device,
            )

    orig_rows = [
        int(meta.get("orig_batch_row_idx", -1))
        for meta in branch_batch.non_tensor_batch.get("appo_branch_meta", [])
    ]
    for key, values in init_batch.non_tensor_batch.items():
        if key == "appo_rollout_kind":
            branch_batch.non_tensor_batch[key] = np.array(["branch"] * branch_bsz, dtype=object)
        elif key == "appo_branch_meta":
            metas = branch_batch.non_tensor_batch.get("appo_branch_meta", np.empty(branch_bsz, dtype=object))
            for i, meta in enumerate(metas):
                if isinstance(meta, dict):
                    meta["batch_row_idx"] = init_bsz + i
            branch_batch.non_tensor_batch[key] = metas
        elif key not in branch_batch.non_tensor_batch:
            branch_batch.non_tensor_batch[key] = values[np.array(orig_rows, dtype=np.int64)]

    for key in list(branch_batch.non_tensor_batch.keys()):
        if key not in init_batch.non_tensor_batch:
            branch_batch.non_tensor_batch.pop(key)

    return DataProto.concat([init_batch, branch_batch])


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1, multi_turn=False, norm_adv_by_std_in_grpo=True, **kwargs):
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator: The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in GRPO. Defaults to True.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch:
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == AdvantageEstimator.GAE:
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if kwargs.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                kwargs.get("pf_ppo_reweight_method", "pow"),
                kwargs.get("pf_ppo_weight_pow", 2.0),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # TODO: test on more adv estimator type
        grpo_calculation_mask = data.batch["response_mask"]
        if multi_turn:
            # If multi-turn, replace the mask with the relevant part of loss_mask
            response_length = grpo_calculation_mask.size(1)  # Get length from the initial response mask
            grpo_calculation_mask = data.batch["loss_mask"][:, -response_length:]  # This mask is the one intended for GRPO
        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.GRPO_PASSK:
        advantages, returns = core_algos.compute_grpo_passk_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE:
        advantages, returns = core_algos.compute_reinforce_plus_plus_baseline_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REMAX:
        advantages, returns = core_algos.compute_remax_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            reward_baselines=data.batch["reward_baselines"],
            response_mask=data.batch["response_mask"],
        )

        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.RLOO:
        advantages, returns = core_algos.compute_rloo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.OPO:
        advantages, returns = core_algos.compute_opo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.APPO:
        data = compute_appo_advantage(data, norm_by_std=norm_adv_by_std_in_grpo)
    else:
        raise NotImplementedError
    return data


# 论文对比曲线的固定列。缺的算法（例如 ARPO 没有 FIPO）写空，方便直接叠图。
_PAPER_COLUMNS = [
    "step",
    "algo",
    "experiment",
    "adv_estimator",
    "branching",
    "loss_mode",
    "reward_mean",
    "reward_max",
    "reward_min",
    "token_reward_mean",
    "advantage_mean",
    "advantage_max",
    "advantage_min",
    "pg_loss",
    "entropy",
    "ppo_kl",
    "grad_norm",
    "pg_clipfrac",
    "response_length_mean",
    "prompt_length_mean",
    "tools_total_calls",
    "tools_successful_calls",
    "tools_failed_calls",
    "search_calls",
    "search_success_rate",
    "python_calls",
    "python_success_rate",
    "fipo_iw_mean",
    "branch_count",
    "branch_score_mean",
    "branch_improve_rate",
    "val_reward",
    "val_score",
    "val_f1",
    "gen_time_s",
    "step_time_s",
    "throughput",
]


def _paper_float(metrics: dict, key: str):
    if key not in metrics:
        return None
    try:
        return float(metrics[key])
    except (TypeError, ValueError):
        return None


def _paper_mean_where(metrics: dict, pred) -> Optional[float]:
    vals = []
    for key, value in metrics.items():
        if not pred(key):
            continue
        try:
            vals.append(float(value))
        except (TypeError, ValueError):
            continue
    if not vals:
        return None
    return sum(vals) / len(vals)


def _infer_algo_name(experiment: str, adv_estimator: str) -> str:
    name = (experiment or "").upper()
    if name.startswith("APPO"):
        return "APPO"
    if name.startswith("ARPO"):
        return "ARPO"
    return str(adv_estimator or "unknown")


def build_paper_row(metrics: dict, step: int, experiment: str, adv_estimator: str, branching: bool, loss_mode: str) -> dict:
    """把 verl 的扁平时序指标收成论文对比用的一行。"""
    row = {
        "step": int(step),
        "algo": _infer_algo_name(experiment, adv_estimator),
        "experiment": experiment,
        "adv_estimator": str(adv_estimator),
        "branching": int(bool(branching)),
        "loss_mode": str(loss_mode),
        "reward_mean": _paper_float(metrics, "critic/score/mean"),
        "reward_max": _paper_float(metrics, "critic/score/max"),
        "reward_min": _paper_float(metrics, "critic/score/min"),
        "token_reward_mean": _paper_float(metrics, "critic/rewards/mean"),
        "advantage_mean": _paper_float(metrics, "critic/advantages/mean"),
        "advantage_max": _paper_float(metrics, "critic/advantages/max"),
        "advantage_min": _paper_float(metrics, "critic/advantages/min"),
        "pg_loss": _paper_float(metrics, "actor/pg_loss"),
        "entropy": _paper_float(metrics, "actor/entropy_loss"),
        "ppo_kl": _paper_float(metrics, "actor/ppo_kl"),
        "grad_norm": _paper_float(metrics, "actor/grad_norm"),
        "pg_clipfrac": _paper_float(metrics, "actor/pg_clipfrac"),
        "response_length_mean": _paper_float(metrics, "response_length/mean"),
        "prompt_length_mean": _paper_float(metrics, "prompt_length/mean"),
        "tools_total_calls": _paper_float(metrics, "tools/total_calls"),
        "tools_successful_calls": _paper_float(metrics, "tools/successful_calls"),
        "tools_failed_calls": _paper_float(metrics, "tools/failed_calls"),
        "search_calls": _paper_float(metrics, "tools/search/calls"),
        "search_success_rate": _paper_float(metrics, "tools/search/success_rate"),
        "python_calls": _paper_float(metrics, "tools/python/calls"),
        "python_success_rate": _paper_float(metrics, "tools/python/success_rate"),
        "fipo_iw_mean": _paper_float(metrics, "actor/fipo_iw_mean"),
        "branch_count": _paper_float(metrics, "branch/count"),
        "branch_score_mean": _paper_float(metrics, "branch/score_mean"),
        "branch_improve_rate": _paper_float(metrics, "branch/improve_rate"),
        "val_reward": _paper_mean_where(metrics, lambda k: k.startswith("val-core/") and k.endswith("/reward/mean@1")),
        "val_score": _paper_mean_where(metrics, lambda k: k.startswith("val-aux/") and "/score/mean@1" in k),
        "val_f1": _paper_mean_where(metrics, lambda k: k.startswith("val-aux/") and "f1_score/mean@1" in k),
        "gen_time_s": _paper_float(metrics, "timing_s/gen"),
        "step_time_s": _paper_float(metrics, "timing_s/step"),
        "throughput": _paper_float(metrics, "perf/throughput"),
    }
    return {col: row.get(col) for col in _PAPER_COLUMNS}


def append_paper_metrics(dump_dir: str, row: dict) -> None:
    """追加 paper_metrics.jsonl 与同列的 paper_metrics.csv。"""
    os.makedirs(dump_dir, exist_ok=True)
    jsonl_path = os.path.join(dump_dir, "paper_metrics.jsonl")
    csv_path = os.path.join(dump_dir, "paper_metrics.csv")
    with open(jsonl_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    with open(csv_path, "a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_PAPER_COLUMNS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow({col: "" if row.get(col) is None else row.get(col) for col in _PAPER_COLUMNS})


_SAMPLE_BRANCH_COLUMNS = [
    "step",
    "algo",
    "experiment",
    "adv_estimator",
    "branching",
    "loss_mode",
    "row_idx",
    "uid",
    "sample_index",
    "data_source",
    "rollout_kind",
    "parent_row",
    "split_token_idx",
    "split_frac",
    "branch_score",
    "m_base",
    "split_entropy",
    "score",
    "f1_score",
    "answer",
    "reason",
    "prompt_length",
    "response_length",
    "advantage_mean",
    "n_search",
    "n_python",
    "parent_score",
    "parent_f1",
    "parent_answer",
    "parent_response_length",
    "parent_advantage_mean",
    "parent_n_search",
    "score_delta",
    "f1_delta",
    "group_n_init",
    "group_n_branch",
    "group_best_score",
    "group_mean_branch_score",
    "group_improve_rate",
    "split_preview",
]


def _blank_num(value):
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:
        return None
    return number


def build_sample_branch_rows(samples: list, step: int, experiment: str, adv_estimator: str, branching: bool, loss_mode: str) -> tuple:
    """把一个 step 的 rollout 收成「每个样本一行」的分支对照表。

    init 与 branch 都保留。branch 行带上父轨迹的 score / f1 / 答案 / 长度，
    这样不用再回查 rollout jsonl 就能比「从哪条轨迹、哪个 token 分出去、结果变好还是变差」。
    返回 (rows, step_stats)。
    """
    algo = _infer_algo_name(experiment, adv_estimator)
    parents = {}
    for sample in samples:
        parents[int(sample["row_idx"])] = sample

    branch_scores = []
    for sample in samples:
        meta = sample.get("branch_meta") or {}
        if str(sample.get("rollout_kind")) != "branch":
            continue
        score = _blank_num(meta.get("branch_score"))
        if score is not None:
            branch_scores.append(score)
    max_branch_score = max(branch_scores) if branch_scores else 0.0

    groups = {}
    for sample in samples:
        groups.setdefault(str(sample.get("uid") or ""), []).append(sample)

    group_stats = {}
    for uid, members in groups.items():
        inits = [m for m in members if str(m.get("rollout_kind")) != "branch"]
        branches = [m for m in members if str(m.get("rollout_kind")) == "branch"]
        scores = [_blank_num(m.get("score")) for m in members]
        scores = [s for s in scores if s is not None]
        b_scores = []
        improved = 0
        comparable = 0
        for member in branches:
            meta = member.get("branch_meta") or {}
            b_score = _blank_num(meta.get("branch_score"))
            if b_score is not None:
                b_scores.append(b_score)
            parent = parents.get(int(meta.get("orig_batch_row_idx", -1)))
            child_score = _blank_num(member.get("score"))
            parent_score = _blank_num(parent.get("score")) if parent else None
            if child_score is None or parent_score is None:
                continue
            comparable += 1
            if child_score > parent_score:
                improved += 1
        group_stats[uid] = {
            "group_n_init": len(inits),
            "group_n_branch": len(branches),
            "group_best_score": max(scores) if scores else None,
            "group_mean_branch_score": (sum(b_scores) / len(b_scores)) if b_scores else None,
            "group_improve_rate": (improved / comparable) if comparable else None,
        }

    rows = []
    for sample in samples:
        meta = sample.get("branch_meta") if isinstance(sample.get("branch_meta"), dict) else {}
        kind = str(sample.get("rollout_kind") or "init")
        parent = None
        parent_row = None
        split_idx = None
        branch_score = None
        m_base = None
        split_frac = None
        if kind == "branch":
            parent_row = meta.get("orig_batch_row_idx")
            try:
                parent_row = int(parent_row)
            except (TypeError, ValueError):
                parent_row = None
            if parent_row is not None and parent_row < 0:
                parent_row = None
            parent = parents.get(parent_row) if parent_row is not None else None
            split_idx = _blank_num(meta.get("branch_token_response_idx"))
            if split_idx is not None:
                split_idx = int(split_idx)
            branch_score = _blank_num(meta.get("branch_score"))
            if branch_score is not None and max_branch_score > 0:
                m_base = branch_score / max_branch_score
            parent_len = _blank_num(parent.get("response_length")) if parent else None
            if split_idx is not None and parent_len and parent_len > 0:
                split_frac = split_idx / parent_len

        score = _blank_num(sample.get("score"))
        f1 = _blank_num(sample.get("f1_score"))
        parent_score = _blank_num(parent.get("score")) if parent else None
        parent_f1 = _blank_num(parent.get("f1_score")) if parent else None
        uid = str(sample.get("uid") or "")
        stats = group_stats.get(uid, {})
        row = {
            "step": int(step),
            "algo": algo,
            "experiment": experiment,
            "adv_estimator": str(adv_estimator),
            "branching": int(bool(branching)),
            "loss_mode": str(loss_mode),
            "row_idx": int(sample["row_idx"]),
            "uid": uid,
            "sample_index": "" if sample.get("sample_index") is None else str(sample.get("sample_index")),
            "data_source": "" if sample.get("data_source") is None else str(sample.get("data_source")),
            "rollout_kind": kind,
            "parent_row": parent_row,
            "split_token_idx": split_idx,
            "split_frac": split_frac,
            "branch_score": branch_score,
            "m_base": m_base,
            "split_entropy": _blank_num(sample.get("split_entropy")) if kind == "branch" else None,
            "score": score,
            "f1_score": f1,
            "answer": "" if sample.get("answer") is None else str(sample.get("answer")),
            "reason": "" if sample.get("reason") is None else str(sample.get("reason")),
            "prompt_length": _blank_num(sample.get("prompt_length")),
            "response_length": _blank_num(sample.get("response_length")),
            "advantage_mean": _blank_num(sample.get("advantage_mean")),
            "n_search": _blank_num(sample.get("n_search")),
            "n_python": _blank_num(sample.get("n_python")),
            "parent_score": parent_score,
            "parent_f1": parent_f1,
            "parent_answer": "" if not parent or parent.get("answer") is None else str(parent.get("answer")),
            "parent_response_length": _blank_num(parent.get("response_length")) if parent else None,
            "parent_advantage_mean": _blank_num(parent.get("advantage_mean")) if parent else None,
            "parent_n_search": _blank_num(parent.get("n_search")) if parent else None,
            "score_delta": None if score is None or parent_score is None else score - parent_score,
            "f1_delta": None if f1 is None or parent_f1 is None else f1 - parent_f1,
            "group_n_init": stats.get("group_n_init"),
            "group_n_branch": stats.get("group_n_branch"),
            "group_best_score": stats.get("group_best_score"),
            "group_mean_branch_score": stats.get("group_mean_branch_score"),
            "group_improve_rate": stats.get("group_improve_rate"),
            "split_preview": "" if kind != "branch" or not sample.get("split_preview") else str(sample.get("split_preview")),
        }
        rows.append({col: row.get(col) for col in _SAMPLE_BRANCH_COLUMNS})

    branch_rows = [row for row in rows if row["rollout_kind"] == "branch"]
    comparable = [row for row in branch_rows if row["score_delta"] is not None]
    improved = [row for row in comparable if row["score_delta"] > 0]
    step_stats = {
        "branch/count": float(len(branch_rows)),
        "branch/score_mean": (sum(row["branch_score"] for row in branch_rows if row["branch_score"] is not None) / len(branch_rows)) if branch_rows else None,
        "branch/improve_rate": (len(improved) / len(comparable)) if comparable else None,
    }
    return rows, step_stats


def write_sample_branches(run_dir: str, rows: list) -> None:
    """按 step 覆盖写入 branches/{step}.jsonl，并重建 run 根目录的汇总 csv/jsonl。"""
    if not rows:
        return
    step = int(rows[0]["step"])
    branch_dir = os.path.join(run_dir, "branches")
    os.makedirs(branch_dir, exist_ok=True)
    step_path = os.path.join(branch_dir, f"{step}.jsonl")
    with open(step_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    combined = []
    step_files = []
    for name in os.listdir(branch_dir):
        if name.endswith(".jsonl") and name[:-6].isdigit():
            step_files.append((int(name[:-6]), os.path.join(branch_dir, name)))
    for _, path in sorted(step_files):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    combined.append(json.loads(line))

    jsonl_path = os.path.join(run_dir, "sample_branches.jsonl")
    csv_path = os.path.join(run_dir, "sample_branches.csv")
    with open(jsonl_path, "w", encoding="utf-8") as handle:
        for row in combined:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_SAMPLE_BRANCH_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in combined:
            writer.writerow({col: "" if row.get(col) is None else row.get(col) for col in _SAMPLE_BRANCH_COLUMNS})


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    """Context manager for timing code execution.

    This utility function measures the execution time of code within its context
    and accumulates the timing information in the provided dictionary.

    Args:
        name (str): The name/identifier for this timing measurement.
        timing_raw (Dict[str, float]): Dictionary to store timing information.

    Yields:
        None: This is a context manager that yields control back to the code block.
    """
    with Timer(name=name, logger=None) as timer:
        yield
    if name not in timing_raw:
        timing_raw[name] = 0
    timing_raw[name] += timer.last


class RayPPOTrainer:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name="cuda",
    ):
        """Initialize distributed PPO trainer with Ray backend."""

        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name
        self.validation_generations_logger = ValidationGenerationsLogger()

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.actor_rollout_ref.model.get("lora_rank", 0) > 0

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(config.algorithm.kl_ctrl)

        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
            AdvantageEstimator.GRPO,
            AdvantageEstimator.GRPO_PASSK,
            AdvantageEstimator.REINFORCE_PLUS_PLUS,
            AdvantageEstimator.REMAX,
            AdvantageEstimator.RLOO,
            AdvantageEstimator.OPO,
            AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE,
            AdvantageEstimator.APPO,
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError

        self._validate_config()
        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % n_gpus == 0, f"real_train_batch_size ({real_train_batch_size}) must be divisible by total n_gpus ({n_gpus})."

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            settings = {
                "actor_rollout_ref.actor": "micro_batch_size",
                "critic": "micro_batch_size",
                "reward_model": "micro_batch_size",
                "actor_rollout_ref.ref": "log_prob_micro_batch_size",
                "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
            }

            if name in settings:
                param = settings[name]
                param_per_gpu = f"{param}_per_gpu"

                if mbs is None and mbs_per_gpu is None:
                    raise ValueError(f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove '{name}.{param}' because only '*_{param_per_gpu}'" + "is supported (the former is deprecated).")

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.actor.ppo_micro_batch_size,
                config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                "actor_rollout_ref.actor",
            )

            if self.use_reference_policy:
                # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
                check_mutually_exclusive(
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                    "actor_rollout_ref.ref",
                )

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.rollout",
            )

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu, "critic")

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu, "reward_model")

        # Actor
        # check if train_batch_size is larger than ppo_mini_batch_size
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.actor_rollout_ref.actor.ppo_mini_batch_size
            sp_size = config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert config.actor_rollout_ref.actor.ppo_mini_batch_size % config.actor_rollout_ref.actor.ppo_micro_batch_size == 0
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        assert config.actor_rollout_ref.actor.loss_agg_mode in [
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
            "seq-mean-token-sum-norm",
        ], f"Invalid loss_agg_mode: {config.actor_rollout_ref.actor.loss_agg_mode}"

        if config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
            print("NOTICE: You have both enabled in-reward kl and kl loss.")

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.critic.ppo_mini_batch_size
            sp_size = config.critic.get("ulysses_sequence_parallel_size", 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == "fsdp" and (config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1) > 1 or config.actor_rollout_ref.ref.get("ulysses_sequence_parallel_size", 1) > 1):
            assert config.actor_rollout_ref.model.use_remove_padding, "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."

        if self.use_critic and config.critic.strategy == "fsdp":
            if config.critic.get("ulysses_sequence_parallel_size", 1) > 1:
                assert config.critic.model.use_remove_padding, "When using sequence parallelism for critic, you must enable `use_remove_padding`."

        if config.data.get("val_batch_size", None) is not None:
            print("WARNING: val_batch_size is deprecated." + " Validation datasets are sent to inference engines as a whole batch," + " which will schedule the memory themselves.")

        # check eval config
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, "validation gen temperature should be greater than 0 when enabling do_sample"

        # check multi_turn with tool config
        if config.actor_rollout_ref.rollout.multi_turn.enable:
            assert config.actor_rollout_ref.rollout.multi_turn.tool_config_path is not None, "tool_config_path must be set when enabling multi_turn with tool, due to no role-playing support"
            assert config.algorithm.adv_estimator in [AdvantageEstimator.GRPO, AdvantageEstimator.APPO], "only GRPO/APPO is tested for multi-turn with tool"

        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(self.config.data.train_files, self.config.data, self.tokenizer, self.processor)
        if val_dataset is None:
            val_dataset = create_rl_dataset(self.config.data.val_files, self.config.data, self.tokenizer, self.processor)
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: {len(self.val_dataloader)}")

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    @staticmethod
    def _dash_safe(obj):
        """把 batch 里可能出现的 numpy/张量/None 元素转成 json 可序列化的值。"""
        import numpy as np
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    def _dump_generations(self, inputs, outputs, scores, reward_extra_infos_dict, dump_path, batch=None):
        """Dump rollout/validation samples as JSONL.

        dashboard 扩展：当传入 batch 时，额外写入 rollout tree 所需的分支元数据
        （uid 分组 id / appo_rollout_kind / appo_branch_meta / token 长度等），
        全部为 additive 字段，不影响旧读取方。
        """
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        if batch is not None:
            try:
                ntb = batch.non_tensor_batch
                bsz = len(batch.batch)
                # uid：prompt 分组 id，rollout tree 的树根 key（branch 与其 init 共享）
                if "uid" in ntb:
                    base_data["uid"] = [str(self._dash_safe(x)) for x in list(ntb["uid"])[:bsz]]
                # init | branch，branch 行的 loss_mask 全 0、不进 actor loss
                if "appo_rollout_kind" in ntb:
                    base_data["rollout_kind"] = [str(self._dash_safe(x)) for x in list(ntb["appo_rollout_kind"])[:bsz]]
                # 分支元数据：分裂点位置/分数与 init↔branch 行号映射
                if "appo_branch_meta" in ntb:
                    metas = []
                    for m in list(ntb["appo_branch_meta"])[:bsz]:
                        metas.append(m if isinstance(m, dict) else None)
                    base_data["branch_meta"] = [
                        {k: self._dash_safe(v) for k, v in m.items()} if m else None for m in metas
                    ]
                # 数据源与原始 sample index
                for key, out_key in (("data_source", "data_source"), ("index", "sample_index")):
                    if key in ntb:
                        base_data[out_key] = [str(self._dash_safe(x)) for x in list(ntb[key])[:bsz]]
                # token 长度（prompt/response 实际非 pad token 数）
                if "attention_mask" in batch.batch:
                    base_data["prompt_length"] = batch.batch["attention_mask"].sum(-1).cpu().tolist()
                resp_key = "response_mask" if "response_mask" in batch.batch else None
                if resp_key is not None:
                    base_data["response_length"] = batch.batch[resp_key].sum(-1).cpu().tolist()
                elif "loss_mask" in batch.batch:
                    base_data["response_length"] = batch.batch["loss_mask"].sum(-1).cpu().tolist()
            except Exception as e:
                print(f"[dashboard] enriched dump failed (fallback to plain fields): {e}")

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        with open(filename, "w") as f:
            for i in range(n):
                entry = {k: v[i] for k, v in base_data.items()}
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"Dumped generations to {filename}")

        # 训练 rollout（带 batch）另写一份按样本展开的分支对照表。验证集 dump 不带 batch，跳过。
        if batch is not None:
            try:
                extras = self._branch_row_extras(batch, base_data.get("branch_meta"), n)
                samples = []
                for i in range(n):
                    output = outputs[i] if i < len(outputs) else ""
                    samples.append({
                        "row_idx": i,
                        "uid": (base_data.get("uid") or [""] * n)[i] if "uid" in base_data else "",
                        "sample_index": (base_data.get("sample_index") or [None] * n)[i] if "sample_index" in base_data else None,
                        "data_source": (base_data.get("data_source") or [None] * n)[i] if "data_source" in base_data else None,
                        "rollout_kind": (base_data.get("rollout_kind") or ["init"] * n)[i] if "rollout_kind" in base_data else "init",
                        "branch_meta": (base_data.get("branch_meta") or [None] * n)[i] if "branch_meta" in base_data else None,
                        "score": scores[i] if i < len(scores) else None,
                        "f1_score": (base_data.get("f1_score") or [None] * n)[i] if "f1_score" in base_data else None,
                        "answer": (base_data.get("answer") or [None] * n)[i] if "answer" in base_data else None,
                        "reason": (base_data.get("reason") or [None] * n)[i] if "reason" in base_data else None,
                        "prompt_length": (base_data.get("prompt_length") or [None] * n)[i] if "prompt_length" in base_data else None,
                        "response_length": (base_data.get("response_length") or [None] * n)[i] if "response_length" in base_data else None,
                        "advantage_mean": extras["advantage_mean"][i],
                        "split_entropy": extras["split_entropy"][i],
                        "split_preview": extras["split_preview"][i],
                        "n_search": str(output).count("<search>"),
                        "n_python": str(output).count("<python>"),
                    })
                experiment = str(self.config.trainer.get("experiment_name", ""))
                adv_estimator = str(self.config.algorithm.get("adv_estimator", ""))
                branching = bool(self.config.actor_rollout_ref.rollout.get("appo_dynamic_branching", False))
                loss_mode = str(self.config.actor_rollout_ref.actor.policy_loss.get("loss_mode", "vanilla"))
                branch_rows, branch_stats = build_sample_branch_rows(
                    samples,
                    step=self.global_steps,
                    experiment=experiment,
                    adv_estimator=adv_estimator,
                    branching=branching,
                    loss_mode=loss_mode,
                )
                run_dir = self.config.trainer.get("default_local_dir", None) or os.path.dirname(os.path.abspath(dump_path))
                write_sample_branches(run_dir, branch_rows)
                self._branch_step_stats = branch_stats
                print(f"Dumped sample branches to {run_dir}/sample_branches.csv ({branch_stats['branch/count']:.0f} branches)")
            except Exception as e:
                print(f"[dashboard] sample branch dump failed: {e}")

    def _branch_row_extras(self, batch, branch_meta, n: int) -> dict:
        """每行的 advantage 均值，以及 branch 行在父轨迹分裂点上的熵和短预览。"""
        advantage_mean = [None] * n
        split_entropy = [None] * n
        split_preview = [""] * n
        try:
            if "advantages" in batch.batch:
                advantages = batch.batch["advantages"]
                mask = batch.batch["response_mask"] if "response_mask" in batch.batch else None
                limit = min(n, advantages.size(0))
                for i in range(limit):
                    if mask is not None:
                        weight = mask[i].float()
                        denom = float(weight.sum().item())
                        if denom <= 0:
                            advantage_mean[i] = None
                        else:
                            advantage_mean[i] = float((advantages[i].float() * weight).sum().item() / denom)
                    else:
                        advantage_mean[i] = float(advantages[i].float().mean().item())
        except Exception as e:
            print(f"[dashboard] advantage extract failed: {e}")

        if not branch_meta:
            return {"advantage_mean": advantage_mean, "split_entropy": split_entropy, "split_preview": split_preview}

        entropys = batch.batch["rollout_entropys"] if "rollout_entropys" in batch.batch else None
        responses = batch.batch["responses"] if "responses" in batch.batch else None
        for i, meta in enumerate(branch_meta[:n]):
            if not isinstance(meta, dict):
                continue
            try:
                parent_row = int(meta.get("orig_batch_row_idx", -1))
                token_idx = int(meta.get("branch_token_response_idx", -1))
            except (TypeError, ValueError):
                continue
            if parent_row < 0 or token_idx < 0:
                continue
            if entropys is not None and parent_row < entropys.size(0) and token_idx < entropys.size(1):
                split_entropy[i] = float(entropys[parent_row, token_idx].item())
            if responses is not None and parent_row < responses.size(0) and token_idx < responses.size(1):
                start = max(0, token_idx - 16)
                window = responses[parent_row, start: token_idx + 1].detach().cpu().tolist()
                try:
                    text = self.tokenizer.decode(window, skip_special_tokens=True)
                except Exception:
                    text = ""
                split_preview[i] = text[-120:]
        return {"advantage_mean": advantage_mean, "split_entropy": split_entropy, "split_preview": split_preview}

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            # repeat test batch
            test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True)

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            test_gen_batch = test_batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                self.async_rollout_manager.wake_up()
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)
                self.async_rollout_manager.sleep()

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_inputs, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (var_name == core_var) and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"]) and (f"@{n_max}" in metric_name):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        return metric_dict

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy], config=self.config.actor_rollout_ref, role="ref")
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls, device_name=self.device_name, **wg_kwargs)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            self.async_rollout_mode = True
            self.async_rollout_manager = AsyncLLMServerManager(
                config=self.config.actor_rollout_ref,
                worker_group=self.actor_rollout_wg,
            )

    def _save_checkpoint(self):
        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print("Warning: remove_previous_ckpt_in_save is deprecated," + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead")
        max_actor_ckpt_to_keep = self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        max_critic_ckpt_to_keep = self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1

        self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep)

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep)

        # save dataloader
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt")
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, "resume ckpt must specify the global_steps"
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")
        # load actor
        self.actor_rollout_wg.load_checkpoint(actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens.
        Returns global_idx (the reorder permutation) so callers can remap index-based metadata."""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst, k_partitions=world_size, equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix)
        metrics.update(global_balance_stats)
        return global_idx

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # pop those keys for generation
                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
                if "multi_modal_data" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )

                is_last_step = self.global_steps >= self.total_training_steps

                with _timer("step", timing_raw):
                    # generate a batch
                    with _timer("gen", timing_raw):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        else:
                            self.async_rollout_manager.wake_up()
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                            self.async_rollout_manager.sleep()

                        if gen_batch_output.meta_info and "metrics" in gen_batch_output.meta_info:
                            metrics.update(gen_batch_output.meta_info["metrics"])

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with _timer("gen_max", timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
                    appo_dynamic_branching = (
                        self.config.algorithm.adv_estimator == AdvantageEstimator.APPO
                        and self.config.actor_rollout_ref.rollout.get("appo_dynamic_branching", True)
                    )
                    rollout_repeat = self.config.actor_rollout_ref.rollout.n
                    if appo_dynamic_branching:
                        rollout_repeat = len(gen_batch_output.batch) // len(batch.batch)
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=rollout_repeat, interleave=True)
                    batch = batch.union(gen_batch_output)
                    if appo_dynamic_branching:
                        missing_branch_keys = [
                            key
                            for key in ("rollout_log_probs", "rollout_entropys")
                            if key not in batch.batch.keys()
                        ]
                        if missing_branch_keys:
                            raise RuntimeError(
                                "APPO dynamic branching requires a rollout backend that returns "
                                f"token log-probs and entropies; missing {missing_branch_keys}. "
                                "Use rollout.mode=sync_with_tool or set "
                                "actor_rollout_ref.rollout.appo_dynamic_branching=False."
                            )

                    batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        balance_global_idx = self._balance_batch(batch, metrics=metrics)
                        # Fix 37: _balance_batch 重排了 batch 行顺序，需要用逆映射更新 appo_branch_meta 里的行索引
                        if self.config.algorithm.adv_estimator == AdvantageEstimator.APPO and balance_global_idx is not None:
                            # old_row -> new_row 的映射：new_row = position of old_row in global_idx
                            old_to_new = {int(old_row): new_row for new_row, old_row in enumerate(balance_global_idx.tolist())}
                            branch_meta_after_balance = batch.non_tensor_batch.get("appo_branch_meta", [])
                            if isinstance(branch_meta_after_balance, np.ndarray):
                                branch_meta_after_balance = branch_meta_after_balance.tolist()
                            for meta in branch_meta_after_balance:
                                if not isinstance(meta, dict):
                                    continue
                                for key in ("batch_row_idx", "orig_batch_row_idx"):
                                    old_row = meta.get(key, -1)
                                    if old_row >= 0:
                                        meta[key] = old_to_new.get(old_row, old_row)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    if not appo_dynamic_branching:
                        with _timer("reward", timing_raw):
                            # compute reward model score
                            if self.use_rm:
                                reward_tensor = self.rm_wg.compute_rm_score(batch)
                                batch = batch.union(reward_tensor)

                            if self.config.reward_model.launch_reward_fn_async:
                                future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer)
                            else:
                                reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    with _timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        if "loss_mask" in batch.batch.keys():
                            loss_mask = batch.batch["loss_mask"]
                        else:
                            loss_mask = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_loss = agg_loss(loss_mat=entropys, loss_mask=loss_mask, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy_loss": entropy_loss.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                    if appo_dynamic_branching:
                        with _timer("appo_branch", timing_raw):
                            branch_budget = self.config.actor_rollout_ref.rollout.n - rollout_repeat
                            expected_appo_bsz = len(batch.batch) // rollout_repeat * self.config.actor_rollout_ref.rollout.n
                            branch_request = build_appo_branch_request(
                                batch,
                                tokenizer=self.tokenizer,
                                max_branches_per_prompt=branch_budget,
                                gamma=self.config.algorithm.get("gamma", 1.0),
                                clip_eps=self.config.actor_rollout_ref.actor.policy_loss.get("future_kl_clip_ratio", 0.2),
                            )
                            if branch_request is not None:
                                branch_output = self.actor_rollout_wg.generate_sequences(branch_request)
                                batch = align_appo_branch_batch(batch, branch_output)
                                batch.batch["response_mask"] = compute_response_mask(batch)
                                batch.batch["critic_loss_mask"] = batch.batch["loss_mask"]
                                batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                            if branch_budget > 0 and len(batch.batch) != expected_appo_bsz:
                                raise RuntimeError(
                                    "APPO dynamic branching could not create enough branch rollouts to "
                                    f"preserve rollout.n: got batch_size={len(batch.batch)}, "
                                    f"expected={expected_appo_bsz}. This usually means some initial "
                                    "rollouts had no valid generated token for branching."
                                )

                        with _timer("reward", timing_raw):
                            # compute reward model score
                            if self.use_rm:
                                reward_tensor = self.rm_wg.compute_rm_score(batch)
                                batch = batch.union(reward_tensor)

                            if self.config.reward_model.launch_reward_fn_async:
                                future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer)
                            else:
                                reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    if "rollout_log_probs" in batch.batch.keys():
                        # TODO: we may want to add diff of probs too.
                        rollout_old_log_probs = batch.batch["rollout_log_probs"]
                        actor_old_log_probs = batch.batch["old_log_probs"]
                        attention_mask = batch.batch["attention_mask"]
                        responses = batch.batch["responses"]
                        response_length = responses.size(1)
                        response_mask = attention_mask[:, -response_length:]

                        rollout_probs = torch.exp(rollout_old_log_probs)
                        actor_probs = torch.exp(actor_old_log_probs)
                        rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
                        rollout_probs_diff = torch.masked_select(rollout_probs_diff, response_mask.bool())
                        rollout_probs_diff_max = torch.max(rollout_probs_diff) if rollout_probs_diff.numel() > 0 else torch.tensor(0.0)
                        rollout_probs_diff_mean = torch.mean(rollout_probs_diff) if rollout_probs_diff.numel() > 0 else torch.tensor(0.0)
                        rollout_probs_diff_std = torch.std(rollout_probs_diff) if rollout_probs_diff.numel() > 1 else torch.tensor(0.0)
                        metrics.update(
                            {
                                "training/rollout_probs_diff_max": rollout_probs_diff_max.detach().item(),
                                "training/rollout_probs_diff_mean": rollout_probs_diff_mean.detach().item(),
                                "training/rollout_probs_diff_std": rollout_probs_diff_std.detach().item(),
                            }
                        )

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer("ref", timing_raw):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer("adv", timing_raw):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        print(f"{list(reward_extra_infos_dict.keys())=}")
                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # compute advantages, executed on the driver process

                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)  # GRPO adv normalization factor

                        # APPO: 在 compute_advantage 之前应用 reward 缩放因子 M
                        if self.config.algorithm.adv_estimator == AdvantageEstimator.APPO:
                            branch_meta_list = batch.non_tensor_batch.get("appo_branch_meta", [])
                            reward_scale_discount = self.config.algorithm.get("reward_scale_discount", 0.9)
                            batch = apply_appo_reward_scale_M(batch, branch_meta_list, reward_scale_discount)

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                            use_pf_ppo=self.config.algorithm.use_pf_ppo,
                            pf_ppo_reweight_method=self.config.algorithm.pf_ppo.reweight_method,
                            pf_ppo_weight_pow=self.config.algorithm.pf_ppo.weight_pow,
                        )

                    # update critic
                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer("update_actor", timing_raw):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            # Debug batch information
                            print("Batch keys:", batch.batch.keys())
                            # print("Batch batch_size:", batch.batch_size)
                            for key in batch.batch.keys():
                                if hasattr(batch.batch[key], "shape"):
                                    print(f"Batch[{key}] shape:", batch.batch[key].shape)
                                elif isinstance(batch.batch[key], list):
                                    print(f"Batch[{key}] length:", len(batch.batch[key]))
                                    # If it's the 'answer' field, print more details
                                    if key == 'answer':
                                        print(f"Batch[{key}] content type:", type(batch.batch[key]))
                                        # Print first few items if possible
                                        if len(batch.batch[key]) > 0:
                                            print(f"First item type:", type(batch.batch[key][0]))
                            
                            # Debug non_tensor_batch information
                            print("\nNon-tensor batch keys:", batch.non_tensor_batch.keys())
                            for key in batch.non_tensor_batch.keys():
                                if isinstance(batch.non_tensor_batch[key], list):
                                    print(f"Non-tensor batch[{key}] length:", len(batch.non_tensor_batch[key]))
                                    if key == 'answer':
                                        print(f"Non-tensor batch[{key}] content type:", type(batch.non_tensor_batch[key]))
                                        if len(batch.non_tensor_batch[key]) > 0:
                                            print(f"First item type:", type(batch.non_tensor_batch[key][0]))
                            
                            # Debug meta_info
                            print("\nMeta info keys:", batch.meta_info.keys())
                            print("Batch size in meta_info if exists:", batch.meta_info.get("batch_size", "Not found"))
                            
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with _timer("dump_rollout_generations", timing_raw):
                            print(batch.batch.keys())
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                                batch=batch,
                            )

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                # dashboard: 逐 step 追加 metrics.jsonl（挂在 rollout_data_dir 同一开关下，
                # 文件落在 run 根目录，即 default_local_dir）
                try:
                    metrics_dump_dir = self.config.trainer.get("default_local_dir", None)
                    if metrics_dump_dir:
                        entry = {"step": self.global_steps}
                        for k, v in metrics.items():
                            try:
                                entry[k] = float(v)
                            except (TypeError, ValueError):
                                entry[k] = str(v)
                        for k, v in getattr(self, "_branch_step_stats", {}).items():
                            if v is not None:
                                entry[k] = float(v)
                        os.makedirs(metrics_dump_dir, exist_ok=True)
                        with open(os.path.join(metrics_dump_dir, "metrics.jsonl"), "a") as mf:
                            mf.write(json.dumps(entry, ensure_ascii=False) + "\n")
                        experiment = str(self.config.trainer.get("experiment_name", ""))
                        adv_estimator = str(self.config.algorithm.get("adv_estimator", ""))
                        branching = bool(self.config.actor_rollout_ref.rollout.get("appo_dynamic_branching", False))
                        loss_mode = str(self.config.actor_rollout_ref.actor.policy_loss.get("loss_mode", "vanilla"))
                        paper_row = build_paper_row(
                            entry,
                            step=self.global_steps,
                            experiment=experiment,
                            adv_estimator=adv_estimator,
                            branching=branching,
                            loss_mode=loss_mode,
                        )
                        append_paper_metrics(metrics_dump_dir, paper_row)
                except Exception as e:
                    print(f"[dashboard] metrics.jsonl append failed: {e}")

                progress_bar.update(1)
                self.global_steps += 1
                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return
