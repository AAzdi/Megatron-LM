#!/usr/bin/env python3

"""
Test script to verify that MoE routing maps are correctly collected and returned
by the GPTModel forward function.
"""

import torch
import sys
import os

# Add the Megatron-LM directory to the Python path
sys.path.append('/home/shaohanh/dizhang/Megatron-LM')

from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core import parallel_state

def test_moe_routing_maps():
    """Test that MoE routing maps are correctly collected and returned."""
    
    # Initialize torch distributed for single process
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend='gloo',  # Use gloo for CPU testing
            init_method='tcp://localhost:12345',
            rank=0,
            world_size=1
        )
    
    # Initialize parallel state for single GPU
    if not parallel_state.is_initialized():
        parallel_state.initialize_model_parallel()
    
    # Create a simple transformer config with MoE layers
    config = TransformerConfig(
        num_layers=4,                    # 4 layers total
        hidden_size=128,                 # Small hidden size for testing
        num_attention_heads=8,
        num_moe_experts=4,               # 4 experts for MoE layers
        moe_router_topk=2,               # Top-2 routing
        moe_layer_freq=2,                # Every 2nd layer is MoE (layers 0, 2)
        moe_router_load_balancing_type="aux_loss",
        add_bias_linear=False,           # Required for MoE
        use_cpu_initialization=True,
        sequence_parallel=False,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=1,
    )
    
    # Create decoder block spec for MoE
    decoder_block_spec = get_gpt_decoder_block_spec(
        config=config,
        use_transformer_engine=False,  # Use local spec for simplicity
    )
    
    # Create GPT model
    model = GPTModel(
        config=config,
        transformer_layer_spec=decoder_block_spec,
        vocab_size=1000,
        max_sequence_length=512,
        pre_process=True,
        post_process=True,
    )
    
    # Create dummy input
    batch_size = 2
    seq_length = 8
    
    input_ids = torch.randint(0, 1000, (batch_size, seq_length))
    position_ids = torch.arange(seq_length).unsqueeze(0).expand(batch_size, -1)
    attention_mask = torch.ones(batch_size, 1, seq_length, seq_length)
    
    print("Testing GPTModel with MoE layers...")
    print(f"Model config: {config.num_layers} layers, MoE every {config.moe_layer_freq} layers")
    print(f"Expected MoE layers: {[i for i in range(config.num_layers) if i % config.moe_layer_freq == 0]}")
    
    # Forward pass
    with torch.no_grad():
        result = model.forward(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )
    
    # Check if routing maps are returned
    if isinstance(result, tuple) and len(result) == 2:
        output, moe_routing_maps = result
        print(f"\n✅ SUCCESS: Model returned routing maps!")
        print(f"Output shape: {output.shape}")
        print(f"Number of MoE layers with routing maps: {len(moe_routing_maps)}")
        
        for i, routing_info in enumerate(moe_routing_maps):
            layer_num = routing_info['layer_number']
            routing_map = routing_info['routing_map']
            print(f"  MoE Layer {layer_num}: routing_map shape = {routing_map.shape}")
            print(f"    Routing map dtype: {routing_map.dtype}")
            print(f"    Number of selected tokens per expert: {routing_map.sum(dim=0)}")
            
    else:
        print(f"\n❌ Model returned only output: {type(result)}")
        if hasattr(result, 'shape'):
            print(f"Output shape: {result.shape}")
        print("No MoE routing maps found. This might indicate:")
        print("1. No MoE layers in the model")
        print("2. MoE layers not returning routing_map")
        print("3. TransformerBlock not collecting routing maps properly")

if __name__ == "__main__":
    test_moe_routing_maps()
