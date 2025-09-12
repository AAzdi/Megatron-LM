# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
import torch
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Union

# ===== Router logits capture utilities (optimized pre-allocated buffer) =====
# 全局开关与预分配缓冲区（仅在use_router_logits=True时使用）
_ROUTER_LOGITS_CAPTURE: bool = False
_ROUTER_LOGITS_BUFFER: Optional[torch.Tensor] = None  # 预分配: [num_layers, bsz, seq_len, num_experts]
_BUFFER_LAYER_COUNT: int = 0  # 已写入的层数

def enable_router_logits_capture(
    num_layers: int, 
    batch_size: int, 
    seq_len: int, 
    num_experts: int, 
    device: torch.device, 
    dtype: torch.dtype = torch.float16,
    clear: bool = True
):
    """开启router logits捕获，预分配缓冲区。
    Args:
        num_layers: 总层数
        batch_size: 批次大小  
        seq_len: 序列长度
        num_experts: 专家数量
        device: 设备
        dtype: 数据类型，建议 fp16/bf16 节省内存
        clear: 是否清空计数器
    """
    global _ROUTER_LOGITS_CAPTURE, _ROUTER_LOGITS_BUFFER, _BUFFER_LAYER_COUNT
    _ROUTER_LOGITS_CAPTURE = True
    _ROUTER_LOGITS_BUFFER = torch.empty(
        num_layers, batch_size, seq_len, num_experts, 
        device=device, dtype=dtype
    )
    if clear:
        _BUFFER_LAYER_COUNT = 0

def disable_router_logits_capture():
    """关闭捕获并清理缓冲区。"""
    global _ROUTER_LOGITS_CAPTURE, _ROUTER_LOGITS_BUFFER, _BUFFER_LAYER_COUNT
    _ROUTER_LOGITS_CAPTURE = False
    _ROUTER_LOGITS_BUFFER = None
    _BUFFER_LAYER_COUNT = 0

def get_captured_router_logits(detach: bool = True, cpu: bool = False):
    """获取捕获到的router logits，返回 [bsz, seq_len, num_layers, num_experts] 格式。
    Args:
        detach: 返回前是否detach，避免梯度跟踪
        cpu: 是否转移到CPU
    Returns:
        torch.Tensor: [batch_size, seq_len, num_layers, num_experts]
    """
    global _ROUTER_LOGITS_BUFFER, _BUFFER_LAYER_COUNT
    if _ROUTER_LOGITS_BUFFER is None:
        return torch.empty(0, 0, 0, 0)
    
    # 只返回已写入的层: [written_layers, B, T, E] -> [B, T, written_layers, E]
    result = _ROUTER_LOGITS_BUFFER[:_BUFFER_LAYER_COUNT].permute(1, 2, 0, 3)
    
    if detach:
        result = result.detach()
    if cpu:
        result = result.cpu()
    
    return result.contiguous()

def _capture_router_logits(layer_number: int, logits: torch.Tensor):
    """写入已标准化 (B,T,E) logits。"""
    global _ROUTER_LOGITS_BUFFER, _BUFFER_LAYER_COUNT
    if _ROUTER_LOGITS_BUFFER is not None and _ROUTER_LOGITS_CAPTURE:
        with torch.no_grad():
            # 确保layer_number在有效范围内 (0-based indexing)
            if layer_number >= _ROUTER_LOGITS_BUFFER.shape[0]:
                # layer_number可能从1开始，转换为0-based
                layer_idx = layer_number - 1 if layer_number > 0 else layer_number
                if layer_idx >= _ROUTER_LOGITS_BUFFER.shape[0] or layer_idx < 0:
                    raise RuntimeError(
                        f"Layer number {layer_number} (idx={layer_idx}) out of bounds for buffer with {_ROUTER_LOGITS_BUFFER.shape[0]} layers"
                    )
            else:
                layer_idx = layer_number
            
            target = _ROUTER_LOGITS_BUFFER[layer_idx]
            if logits.shape != target.shape:
                raise RuntimeError(
                    f"Router logits canonical shape mismatch layer={layer_number} (idx={layer_idx}): got {tuple(logits.shape)} expected {tuple(target.shape)}"
                )
            target.copy_(logits, non_blocking=True)
            _BUFFER_LAYER_COUNT = max(_BUFFER_LAYER_COUNT, layer_idx + 1)

import torch

from megatron.core import parallel_state, tensor_parallel
from megatron.core.process_groups_config import ModelCommProcessGroups
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.moe_utils import get_default_model_comm_pgs
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.moe.token_dispatcher import (
    MoEAllGatherTokenDispatcher,
    MoEAlltoAllTokenDispatcher,
    MoEFlexTokenDispatcher,
    MoETokenDispatcher,
)
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig

try:
    import transformer_engine as te  # pylint: disable=unused-import

    from megatron.core.extensions.transformer_engine import te_checkpoint

    HAVE_TE = True
except ImportError:
    HAVE_TE = False


@dataclass
class MoESubmodules:
    """MoE Layer Submodule spec"""

    experts: Union[ModuleSpec, type] = None
    shared_experts: Union[ModuleSpec, type] = None


class BaseMoELayer(MegatronModule, ABC):
    """Base class for a mixture of experts layer.

    Args:
        config (TransformerConfig): Configuration object for the transformer model.
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: Optional[int] = None,
        model_comm_pgs: Optional[ModelCommProcessGroups] = None,
    ):
        super(BaseMoELayer, self).__init__(config)
        self.config = config
        self.layer_number = layer_number
        self.ep_group = model_comm_pgs.ep
        # use model_comm_pgs.expt_tp_group as tensor parallel group in this module.
        self.attn_tp_group = model_comm_pgs.tp
        ep_size = self.ep_group.size()
        ep_rank = self.ep_group.rank()
        assert ep_size > 0, "Expected non-negative expert parallel size"

        assert self.config.num_moe_experts % ep_size == 0
        self.num_local_experts = self.config.num_moe_experts // ep_size
        local_expert_indices_offset = ep_rank * self.num_local_experts

        self.use_shared_expert = self.config.moe_shared_expert_intermediate_size is not None
        self.shared_expert_overlap = self.config.moe_shared_expert_overlap

        self.local_expert_indices = [
            local_expert_indices_offset + i for i in range(self.num_local_experts)
        ]
        assert all(map(lambda x: x < self.config.num_moe_experts, self.local_expert_indices))
        self.router: TopKRouter = None
        self.experts = None
        self.shared_experts = None
        self.token_dispatcher: Optional[MoETokenDispatcher] = None
        self.layer_number = layer_number

    @abstractmethod
    def forward(self, hidden_states):
        """Forward method for the MoE layer."""
        pass

    def set_layer_number(self, layer_number: int):
        """Set the layer number for the MoE layer."""
        self.layer_number = layer_number
        self.router.set_layer_number(layer_number)


class MoELayer(BaseMoELayer):
    """Mixture of Experts layer.

    This layer implements a Mixture of Experts model, where each token is routed to a
    subset of experts. This implementation supports different token dispatching
    strategies such as All-to-All and All-Gather.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: Optional[MoESubmodules] = None,
        layer_number: Optional[int] = None,
        model_comm_pgs: Optional[ModelCommProcessGroups] = None,
    ):
        self.submodules = submodules
        # TODO(Hepteract): delete the usage of the global parallel_state.
        # Initialize process groups with the global parallel_state.
        if model_comm_pgs is None:
            model_comm_pgs = get_default_model_comm_pgs()
        super(MoELayer, self).__init__(
            config=config, layer_number=layer_number, model_comm_pgs=model_comm_pgs
        )
        self.moe_layer_recompute = (
            config.recompute_granularity == 'selective' and "moe" in config.recompute_modules
        )

        # Initialize router
        self.router = TopKRouter(config=self.config, model_comm_pgs=model_comm_pgs)

        # Initialize token dispatcher
        if config.moe_token_dispatcher_type == "allgather":
            self.token_dispatcher = MoEAllGatherTokenDispatcher(
                self.num_local_experts,
                self.local_expert_indices,
                config=self.config,
                model_comm_pgs=model_comm_pgs,
            )
        elif config.moe_token_dispatcher_type == "alltoall":
            self.token_dispatcher = MoEAlltoAllTokenDispatcher(
                self.num_local_experts,
                self.local_expert_indices,
                config=self.config,
                model_comm_pgs=model_comm_pgs,
            )
        elif config.moe_token_dispatcher_type == "flex":
            self.token_dispatcher = MoEFlexTokenDispatcher(
                self.num_local_experts,
                self.local_expert_indices,
                config=self.config,
                model_comm_pgs=model_comm_pgs,
            )
        else:
            raise ValueError(
                f"Unsupported token dispatcher type: {config.moe_token_dispatcher_type}"
            )

        # Initialize experts
        self.experts = build_module(
            self.submodules.experts,
            self.num_local_experts,
            self.config,
            model_comm_pgs=model_comm_pgs,
        )

        # Initialize shared experts
        if self.use_shared_expert:
            self.shared_experts = build_module(
                self.submodules.shared_experts, config=self.config, model_comm_pgs=model_comm_pgs
            )
            if self.shared_expert_overlap:
                self.token_dispatcher.set_shared_experts(self.shared_experts)

    def router_and_preprocess(self, hidden_states: torch.Tensor):
        """Compute and preprocess token routing for dispatch.

        This method uses the router to determine which experts to send each token to,
        producing routing probabilities and a mapping. It then preprocesses the
        hidden states and probabilities for the token dispatcher. The original
        hidden states are returned as a residual connection.
        """
        residual = hidden_states
        # probs, routing_map, logits = self.router(hidden_states)
        probs, routing_map, logits = self.router(hidden_states)
        # ---- 标准化 logits 形状为 (B,T,E) ----
        # 可能输入 (T,B,E) / (T,1,E) / (T,E) / (B,T,E)
        if logits.dim() == 2:
            # (T,E) -> (1,T,E)
            logits = logits.unsqueeze(0)
        elif logits.dim() == 3:
            # 明确检查 (T,1,E) -> (1,T,E) 的情况
            if logits.shape[1] == 1 and logits.shape[0] > 1:
                # (T,1,E) -> (1,T,E)
                logits = logits.permute(1,0,2).contiguous()
            elif logits.shape[0] > 1 and logits.shape[1] > 1:
                # 可能是 (T,B,E) -> (B,T,E) 或已经是 (B,T,E)
                # 根据 hidden_states 的 batch 维判断
                if logits.shape[1] == hidden_states.shape[0]:
                    # (T,B,E) -> permute
                    logits = logits.permute(1,0,2).contiguous()
                # 否则假设已是 (B,T,E)
            # 其他情况（如已是 (1,T,E)）直接通过
        # 捕获
        if _ROUTER_LOGITS_CAPTURE:
            _capture_router_logits(self.layer_number, logits)
        hidden_states, probs = self.token_dispatcher.dispatch_preprocess(
            hidden_states, routing_map, probs
        )
        return hidden_states, probs, residual

    def dispatch(self, hidden_states: torch.Tensor, probs: torch.Tensor):
        """Dispatches tokens to assigned expert ranks via communication.
        This method performs the actual communication (e.g., All-to-All) to distribute
        tokens and their associated probabilities to the devices hosting their assigned
        experts.
        """
        return self.token_dispatcher.token_dispatch(hidden_states, probs)

    def experts_compute(
        self, hidden_states: torch.Tensor, probs: torch.Tensor, residual: torch.Tensor
    ):
        """Computes the output of the experts on the dispatched tokens.

        This method first post-processes the dispatched input to get permuted tokens
        for each expert. It then passes the tokens through the local experts.
        If a shared expert is configured and not overlapped with communication,
        it is also applied. The output from the experts is preprocessed for the
        combine step.
        """
        shared_expert_output = None
        if self.use_shared_expert and not self.shared_expert_overlap:
            # Compute the shared expert separately when not overlapped with communication.
            shared_expert_output = self.shared_experts(residual)
        dispatched_input, tokens_per_expert, permuted_probs = (
            self.token_dispatcher.dispatch_postprocess(hidden_states, probs)
        )
        expert_output, mlp_bias = self.experts(dispatched_input, tokens_per_expert, permuted_probs)
        assert mlp_bias is None, f"mlp_bias is not supported for {type(self.token_dispatcher)}"
        output = self.token_dispatcher.combine_preprocess(expert_output)

        return output, shared_expert_output, mlp_bias

    def combine(self, output: torch.Tensor, shared_expert_output: Optional[torch.Tensor]):
        """Combines expert outputs via communication and adds shared expert output.

        This method uses the token dispatcher to combine the outputs from different
        experts (e.g., via an All-to-All communication). It then adds the output
        from the shared expert if it exists.
        """
        output = self.token_dispatcher.token_combine(output)
        output = self.token_dispatcher.combine_postprocess(output)
        if shared_expert_output is not None:
            output = output + shared_expert_output
        return output

    def forward(self, hidden_states: torch.Tensor):
        """Forward pass for the MoE layer.

        The forward pass comprises four main steps:
        1. Routing & Preprocessing: Route tokens to the assigned experts and prepare for dispatch.
        2. Dispatch: Tokens are sent to the expert devices using communication collectives.
        3. Expert Computation: Experts process the dispatched tokens.
        4. Combine: The outputs from the experts are combined and returned.

        Args:
            hidden_states (torch.Tensor): The input tensor to the MoE layer.

        Returns:
            A tuple containing the output tensor and the MLP bias, if any.
        """
        if self.training and self.attn_tp_group.size() > 1 and not self.config.sequence_parallel:
            raise ValueError(
                "During training, performance may degrade if MoE and tensor parallelism"
                "are enabled without also enabling sequence parallelism."
            )

        # MoE forward: route -> dispatch -> compute -> combine
        def custom_forward(hidden_states):
            hidden_states, probs, residual = self.router_and_preprocess(hidden_states)
            dispatched_input, probs = self.dispatch(hidden_states, probs)
            output, shared_expert_output, mlp_bias = self.experts_compute(
                dispatched_input, probs, residual
            )
            output = self.combine(output, shared_expert_output)
            return output, mlp_bias

        if self.moe_layer_recompute:
            if self.config.fp8:
                output, mlp_bias = te_checkpoint(
                    custom_forward,
                    False,
                    tensor_parallel.random.get_cuda_rng_tracker,
                    parallel_state.get_tensor_model_parallel_group(),
                    hidden_states,
                )
            else:
                output, mlp_bias = tensor_parallel.checkpoint(custom_forward, False, hidden_states)
        else:
            output, mlp_bias = custom_forward(hidden_states)

        return output, mlp_bias

    def backward_dw(self):
        """Compute weight gradients for experts and shared experts."""
        self.experts.backward_dw()
        if self.use_shared_expert and not self.shared_expert_overlap:
            self.shared_experts.backward_dw()
