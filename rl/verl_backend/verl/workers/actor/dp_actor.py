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
Single Process Actor
"""

import itertools
import logging
import os
from typing import Tuple

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss, compute_policy_loss_future_kl, kl_penalty
from verl.utils.debug import GPUMemoryLogger
from verl.utils.device import get_device_name, get_torch_device, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from rl.fcr import binary_bce_from_scores, binary_brier_from_scores, combine_pred_losses

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input


__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _find_subsequence(sequence, needle):
    needle = [int(value) for value in needle]
    if not needle:
        return -1
    values = [int(value) for value in sequence]
    for start in range(len(values) - len(needle) + 1):
        if values[start : start + len(needle)] == needle:
            return start
    return -1


class DataParallelPPOActor(BasePPOActor):
    def __init__(self, config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"Actor use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"Actor use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.compute_entropy_from_logits = (
            torch.compile(verl_F.entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  #  use torch compile by default
            else verl_F.entropy_from_logits
        )
        self.device_name = get_device_name()

    def _forward_micro_batch(self, micro_batch, temperature, calculate_entropy=False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch:
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0)

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices).transpose(0, 1).unsqueeze(1)  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = "multi_modal_inputs" in micro_batch
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outpus_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
        elif use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature, calculate_entropy=calculate_entropy)
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]

        return log_probs, entropys

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        multi_turn = data.meta_info.get("multi_turn", False)

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages"]
        if multi_turn or "loss_mask" in data.batch.keys():
            select_keys.append("loss_mask")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        fcr_non_tensor_keys = [
            "satisfy_token_ids_list", "unsatisfy_token_ids_list",
            "prediction_token_start_list", "predicted_label_list",
            "actual_satisfied_list",
        ]
        fcr_mode = str(data.meta_info.get("fcr_brier_mode") or "magnitude")
        fcr_enabled = bool(data.meta_info.get("fcr_brier_enabled", False))
        fcr_fixed_beta = float(data.meta_info.get("fcr_brier_beta", 0.1) or 0.0)
        has_fcr_labels = fcr_enabled and all(
            key in data.non_tensor_batch for key in fcr_non_tensor_keys
        )
        non_tensor_select_keys = []
        if has_multi_modal_inputs:
            non_tensor_select_keys.append("multi_modal_inputs")
        if has_fcr_labels:
            non_tensor_select_keys.extend(fcr_non_tensor_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        if non_tensor_select_keys:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        fcr_gradient_diagnostic_done = False
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                # split batch into micro_batches
                mini_batch = data
                if non_tensor_select_keys:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    # split batch into micro_batches
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for data in micro_batches:
                    # Support all hardwares
                    if isinstance(data, DataProto):
                        data = {**data.batch.to(get_torch_device().current_device()), **data.non_tensor_batch}
                    else:
                        data = data.to(get_torch_device().current_device())  # actor device is cpu when using offload
                    responses = data["responses"]
                    response_length = responses.size(1)
                    attention_mask = data["attention_mask"]
                    if multi_turn or "loss_mask" in data.keys():
                        response_mask = data["loss_mask"][:, -response_length:]
                    else:
                        response_mask = attention_mask[:, -response_length:]

                    old_log_prob = data["old_log_probs"]
                    advantages = data["advantages"]

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    entropy, log_prob = self._forward_micro_batch(micro_batch=data, temperature=temperature, calculate_entropy=calculate_entropy)

                    # Read FIPO config (nested under actor.policy_loss in ppo_trainer.yaml)
                    fipo_cfg = self.config.get("policy_loss", {})
                    use_future_kl = fipo_cfg.get("loss_mode", "vanilla") == "future_kl"

                    if use_future_kl:
                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower, fipo_metrics = compute_policy_loss_future_kl(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            clip_ratio=clip_ratio,
                            clip_ratio_low=clip_ratio_low,
                            clip_ratio_high=clip_ratio_high,
                            clip_ratio_c=clip_ratio_c,
                            loss_agg_mode=loss_agg_mode,
                            chunk_size=fipo_cfg.get("chunk_size", 128),
                            decay_rate=fipo_cfg.get("decay_rate", 128.0),
                            future_kl_clip_ratio=fipo_cfg.get("future_kl_clip_ratio", 0.2),
                            future_kl_clip_high_only=fipo_cfg.get("future_kl_clip_high_only", False),
                            future_kl_weight=fipo_cfg.get("future_kl_weight", 1.0),
                            safety_thresh=fipo_cfg.get("safety_thresh", 4.0),
                        )
                        append_to_dict(metrics, fipo_metrics)
                    else:
                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            cliprange=clip_ratio,
                            cliprange_low=clip_ratio_low,
                            cliprange_high=clip_ratio_high,
                            clip_ratio_c=clip_ratio_c,
                            loss_agg_mode=loss_agg_mode,
                        )

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = data["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type)
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] = kl_loss.detach().item()
                        metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if has_fcr_labels:
                        pred_loss, brier_meta = self._fcr_brier_loss(data, temperature=temperature)
                        if pred_loss is not None:
                            diagnostics_on = os.environ.get("FCR_GRAD_DIAGNOSTICS", "0").lower() not in (
                                "0", "false", "off",
                            )
                            if not fcr_gradient_diagnostic_done and diagnostics_on:
                                diagnostics = self._fcr_gradient_diagnostics(policy_loss, pred_loss)
                                append_to_dict(metrics, diagnostics)
                                fcr_gradient_diagnostic_done = True
                            elif "actor/fcr_loss_abs_ratio" not in metrics:
                                pg_abs = pg_loss.detach().abs().clamp_min(1e-12)
                                # Cheap proxy only — not a gradient norm ratio.
                                metrics["actor/fcr_loss_abs_ratio"] = (
                                    pred_loss.detach().abs() / pg_abs
                                ).item()
                            pred_abs = pred_loss.detach().abs()
                            pg_abs = pg_loss.detach().abs()
                            if fcr_mode == "fixed":
                                scale = torch.tensor(
                                    fcr_fixed_beta,
                                    device=pred_loss.device,
                                    dtype=pred_loss.dtype,
                                )
                            else:
                                scale = pg_abs / (pg_abs + pred_abs).clamp_min(1e-12)
                            policy_loss = policy_loss + scale * pred_loss
                            metrics["actor/fcr_brier_loss"] = float(
                                brier_meta.get("actor/fcr_brier_component", pred_loss.detach().item())
                            )
                            metrics["actor/fcr_pred_loss"] = pred_loss.detach().item()
                            metrics["actor/fcr_brier_beta"] = float(scale.detach().item())
                            metrics["actor/fcr_brier_scale"] = float(scale.detach().item())
                            metrics["actor/fcr_loss_ratio"] = float(
                                (pred_abs / pg_abs.clamp_min(1e-12)).item()
                            )
                            append_to_dict(metrics, brier_meta)

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * (len(data) / self.config.ppo_mini_batch_size)
                    else:
                        loss = policy_loss / self.gradient_accumulation
                    loss.backward()

                    append_to_dict(metrics, {
                        "actor/pg_loss": pg_loss.detach().item(),
                        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                        "actor/ppo_kl": ppo_kl.detach().item(),
                        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                    })

                grad_norm = self._optimizer_step()
                data = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, data)
        self.actor_optimizer.zero_grad()
        return metrics

    def _fcr_gradient_diagnostics(self, policy_loss, brier_loss):
        params = [parameter for parameter in self.actor_module.parameters() if parameter.requires_grad]
        policy_grads = torch.autograd.grad(
            policy_loss, params, retain_graph=True, allow_unused=True
        )
        brier_grads = torch.autograd.grad(
            brier_loss, params, retain_graph=True, allow_unused=True
        )
        policy_sq = policy_loss.new_zeros(())
        brier_sq = policy_loss.new_zeros(())
        dot = policy_loss.new_zeros(())
        for policy_grad, brier_grad in zip(policy_grads, brier_grads):
            if policy_grad is not None:
                policy_sq = policy_sq + policy_grad.float().square().sum()
            if brier_grad is not None:
                brier_sq = brier_sq + brier_grad.float().square().sum()
            if policy_grad is not None and brier_grad is not None:
                dot = dot + (policy_grad.float() * brier_grad.float()).sum()
        policy_norm = policy_sq.sqrt()
        brier_norm = brier_sq.sqrt()
        denominator = (policy_norm * brier_norm).clamp_min(1e-12)
        return {
            "actor/fcr_grad_norm_ratio": (brier_norm / policy_norm.clamp_min(1e-12)).detach().item(),
            "actor/fcr_grad_cosine": (dot / denominator).detach().item(),
        }

    def _fcr_brier_loss(self, data, *, temperature: float):
        """Teacher-force YES/NO with class-balanced row selection.

        Returns ``(loss, meta)`` where ``meta`` carries sampling diagnostics for
        metrics.jsonl. Failure-dominated batches previously collapsed to always-NO
        when only the first valid row (usually Y=0) was supervised.
        """
        empty = {
            "actor/fcr_brier_n_pos": 0.0,
            "actor/fcr_brier_n_neg": 0.0,
            "actor/fcr_brier_pos_weight": 1.0,
            "actor/fcr_brier_mean_p": 0.0,
            "actor/fcr_brier_skipped_unbalanced": 0.0,
        }
        if self.use_ulysses_sp:
            return None, empty
        max_rows = max(0, int(os.environ.get("FCR_BRIER_MAX_ROWS", "1")))
        if max_rows == 0:
            return None, empty

        responses = data["responses"]
        input_ids = data["input_ids"]
        attention_mask = data["attention_mask"]
        satisfy_ids = data["satisfy_token_ids_list"]
        unsatisfy_ids = data["unsatisfy_token_ids_list"]
        labels = data["actual_satisfied_list"]
        written_labels = data["predicted_label_list"]
        prediction_starts = data["prediction_token_start_list"]

        candidates = []
        for row in range(int(responses.size(0))):
            positive = list(satisfy_ids[row] or [])
            negative = list(unsatisfy_ids[row] or [])
            label = labels[row]
            if not positive or not negative or not isinstance(label, (bool,)):
                continue
            response_tokens = responses[row].detach().cpu().tolist()
            written = str(written_labels[row] or "").strip().upper()
            actual_tokens = positive if written in {"YES", "满足", "Y"} else negative
            try:
                preferred_offset = int(prediction_starts[row])
            except (TypeError, ValueError):
                preferred_offset = -1
            if (
                preferred_offset >= 0
                and response_tokens[
                    preferred_offset : preferred_offset + len(actual_tokens)
                ] == [int(value) for value in actual_tokens]
            ):
                response_offset = preferred_offset
            else:
                response_offset = _find_subsequence(response_tokens, actual_tokens)
            if response_offset < 0:
                continue
            response_start = int(input_ids.size(1) - responses.size(1))
            label_start = response_start + response_offset
            positions = torch.arange(input_ids.size(1), device=input_ids.device)
            prefix = input_ids[row][attention_mask[row].bool() & (positions < label_start)]
            if int(prefix.numel()) <= 0:
                continue
            candidates.append(
                {
                    "row": row,
                    "label": bool(label),
                    "positive": positive,
                    "negative": negative,
                    "prefix": prefix,
                }
            )

        n_pos = sum(1 for item in candidates if item["label"])
        n_neg = sum(1 for item in candidates if not item["label"])
        pos_weight = float(min(8.0, max(1.0, n_neg / max(n_pos, 1))))
        skipped_unbalanced = 0.0
        if n_pos == 0 or n_neg == 0:
            skipped_unbalanced = 1.0

        selected = []
        if not candidates:
            return None, {
                **empty,
                "actor/fcr_brier_skipped_unbalanced": skipped_unbalanced,
            }
        if max_rows == 1:
            minority = n_pos <= n_neg
            preferred = [item for item in candidates if item["label"] == minority]
            pool = preferred or candidates
            selected = [pool[0]]
        else:
            pos_pool = [item for item in candidates if item["label"]]
            neg_pool = [item for item in candidates if not item["label"]]
            if pos_pool:
                selected.append(pos_pool[0])
            if neg_pool and len(selected) < max_rows:
                selected.append(neg_pool[0])
            used = {id(item) for item in selected}
            for item in candidates:
                if len(selected) >= max_rows:
                    break
                if id(item) in used:
                    continue
                selected.append(item)
                used.add(id(item))

        losses = []
        probabilities = []
        brier_components = []
        bce_components = []
        for item in selected:
            prefix = item["prefix"]
            prefix_len = int(prefix.numel())
            scores = []
            for candidate in (item["positive"], item["negative"]):
                candidate_tensor = torch.tensor(
                    candidate, dtype=prefix.dtype, device=prefix.device
                )
                candidate_input = torch.cat((prefix, candidate_tensor), dim=0).unsqueeze(0)
                candidate_mask = torch.ones_like(candidate_input)
                candidate_positions = torch.arange(
                    candidate_input.size(1), device=prefix.device, dtype=torch.long
                ).unsqueeze(0)
                with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
                    output = self.actor_module(
                        input_ids=candidate_input,
                        attention_mask=candidate_mask,
                        position_ids=candidate_positions,
                        use_cache=False,
                    )
                    logits = getattr(output, "logits", None)
                    if logits is None:
                        scores = []
                        break
                    logits = logits.float() / float(temperature)
                    start = prefix_len - 1
                    token_logits = logits[0, start : start + len(candidate)]
                    token_targets = candidate_tensor
                    scores.append(
                        torch.log_softmax(token_logits, dim=-1)
                        .gather(-1, token_targets.unsqueeze(-1))
                        .sum()
                    )
            if len(scores) != 2:
                continue
            use_bce = os.environ.get("FCR_PRED_BCE", "0").lower() not in ("0", "false", "off")
            bce_w = float(os.environ.get("FCR_BCE_WEIGHT", "1.0"))
            brier_w = float(os.environ.get("FCR_BRIER_WEIGHT", "1.0" if not use_bce else "0.1"))
            brier_loss, probability = binary_brier_from_scores(
                scores[0],
                scores[1],
                bool(item["label"]),
                pos_weight=pos_weight,
            )
            bce_loss = None
            if use_bce:
                bce_loss, probability = binary_bce_from_scores(
                    scores[0],
                    scores[1],
                    bool(item["label"]),
                    pos_weight=pos_weight,
                )
            loss = combine_pred_losses(
                bce_loss, brier_loss, bce_weight=bce_w if use_bce else 0.0, brier_weight=brier_w,
            )
            if loss is None:
                continue
            losses.append(loss)
            probabilities.append(probability.detach())
            brier_components.append(brier_loss.detach())
            if bce_loss is not None:
                bce_components.append(bce_loss.detach())

        meta = {
            "actor/fcr_brier_n_pos": float(n_pos),
            "actor/fcr_brier_n_neg": float(n_neg),
            "actor/fcr_brier_pos_weight": float(pos_weight),
            "actor/fcr_brier_mean_p": float(
                torch.stack(probabilities).mean().item()
            )
            if probabilities
            else 0.0,
            "actor/fcr_brier_skipped_unbalanced": skipped_unbalanced,
            "actor/fcr_brier_component": float(
                torch.stack(brier_components).mean().item()
            )
            if brier_components
            else 0.0,
            "actor/fcr_bce_component": float(
                torch.stack(bce_components).mean().item()
            )
            if bce_components
            else 0.0,
        }
        if not losses:
            return None, meta
        return torch.stack(losses).mean(), meta
