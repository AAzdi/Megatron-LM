#!/usr/bin/env python3

"""
Example script showing how to use the modified GPTModel to collect and analyze
MoE routing maps during forward pass.
"""

import sys
import os
sys.path.append('/home/shaohanh/dizhang/Megatron-LM')

def example_usage():
    """Demonstrate how to use the modified GPTModel with MoE routing maps."""
    
    print("="*70)
    print("EXAMPLE: Using Modified GPTModel with MoE Routing Maps")
    print("="*70)
    
    print("""
🔧 SETUP:
Your moe_layer.py forward function should now return:
    return output, mlp_bias, routing_map
    
Where routing_map is a torch.Tensor of shape [num_tokens, num_experts]
with boolean values indicating which experts were selected for each token.

📋 USAGE PATTERNS:

1. BASIC USAGE:
""")
    
    print("""
# Import required modules
from megatron.core.models.gpt.gpt_model import GPTModel
import torch

# Create your model (assumes you have proper config)
# model = GPTModel(config=config, transformer_layer_spec=spec, ...)

# Forward pass
forward_result = model(input_ids, position_ids, attention_mask)

# Check if MoE routing maps are returned
if isinstance(forward_result, tuple) and len(forward_result) == 2:
    # Model has MoE layers
    result, moe_routing_maps = forward_result
    print(f"Found {len(moe_routing_maps)} MoE layers")
    
    # Analyze each MoE layer's routing
    for routing_info in moe_routing_maps:
        layer_num = routing_info['layer_number']
        routing_map = routing_info['routing_map']
        
        print(f"Layer {layer_num}: {routing_map.shape}")
        print(f"  Tokens per expert: {routing_map.sum(dim=0)}")
        print(f"  Load balance: {routing_map.sum(dim=0).std().item():.4f}")
        
else:
    # Model has no MoE layers
    result = forward_result
    print("No MoE layers found in model")
""")
    
    print("""
2. ADVANCED ANALYSIS:
""")
    
    print("""
def analyze_routing_patterns(moe_routing_maps):
    \"\"\"Analyze routing patterns across all MoE layers.\"\"\"
    
    for i, routing_info in enumerate(moe_routing_maps):
        layer_num = routing_info['layer_number']
        routing_map = routing_info['routing_map']  # [num_tokens, num_experts]
        
        # Basic statistics
        num_tokens, num_experts = routing_map.shape
        tokens_per_expert = routing_map.sum(dim=0)  # [num_experts]
        experts_per_token = routing_map.sum(dim=1)  # [num_tokens]
        
        print(f"\\n📊 MoE Layer {layer_num} Analysis:")
        print(f"  📏 Shape: {routing_map.shape}")
        print(f"  🎯 Top-K: {experts_per_token[0].item()} experts per token")
        print(f"  ⚖️  Load Balance:")
        print(f"    - Mean tokens per expert: {tokens_per_expert.float().mean():.2f}")
        print(f"    - Std tokens per expert: {tokens_per_expert.float().std():.2f}")
        print(f"    - Min tokens per expert: {tokens_per_expert.min().item()}")
        print(f"    - Max tokens per expert: {tokens_per_expert.max().item()}")
        
        # Expert utilization
        active_experts = (tokens_per_expert > 0).sum().item()
        print(f"  🔥 Active experts: {active_experts}/{num_experts}")
        print(f"  📈 Utilization rate: {active_experts/num_experts*100:.1f}%")
        
        # Routing entropy (diversity measure)
        expert_probs = tokens_per_expert.float() / tokens_per_expert.sum()
        entropy = -(expert_probs * torch.log(expert_probs + 1e-8)).sum()
        max_entropy = torch.log(torch.tensor(num_experts, dtype=torch.float))
        normalized_entropy = entropy / max_entropy
        print(f"  🎲 Routing entropy: {entropy:.3f} (normalized: {normalized_entropy:.3f})")

# Usage
if isinstance(forward_result, tuple):
    result, moe_routing_maps = forward_result
    analyze_routing_patterns(moe_routing_maps)
""")
    
    print("""
3. INTEGRATION WITH TRAINING:
""")
    
    print("""
# In your training loop
for batch in dataloader:
    input_ids, position_ids, attention_mask, labels = batch
    
    # Forward pass
    forward_result = model(input_ids, position_ids, attention_mask, labels=labels)
    
    if isinstance(forward_result, tuple):
        # Model with MoE layers
        loss, moe_routing_maps = forward_result
        
        # Log routing statistics for monitoring
        for routing_info in moe_routing_maps:
            layer_num = routing_info['layer_number']
            routing_map = routing_info['routing_map']
            
            # Calculate load balancing metrics
            tokens_per_expert = routing_map.sum(dim=0)
            load_balance_loss = tokens_per_expert.float().var()
            
            # Log to tensorboard/wandb
            logger.log(f"moe_layer_{layer_num}/load_balance_var", load_balance_loss)
            logger.log(f"moe_layer_{layer_num}/active_experts", 
                      (tokens_per_expert > 0).sum().item())
    else:
        # Model without MoE layers
        loss = forward_result
    
    # Continue with backward pass
    loss.backward()
    optimizer.step()
""")

    print("""
4. DEBUGGING AND VISUALIZATION:
""")
    
    print("""
def visualize_routing_map(routing_map, layer_num, save_path=None):
    \"\"\"Create a heatmap visualization of routing patterns.\"\"\"
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 8))
    plt.imshow(routing_map.cpu().numpy(), aspect='auto', cmap='Blues')
    plt.xlabel('Expert Index')
    plt.ylabel('Token Index')
    plt.title(f'MoE Layer {layer_num} Routing Map')
    plt.colorbar(label='Selected (1) / Not Selected (0)')
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

# Usage
if isinstance(forward_result, tuple):
    result, moe_routing_maps = forward_result
    for routing_info in moe_routing_maps:
        visualize_routing_map(
            routing_info['routing_map'], 
            routing_info['layer_number'],
            f"routing_layer_{routing_info['layer_number']}.png"
        )
""")

    print("""
🎯 KEY BENEFITS:

1. 📊 Monitor Load Balancing: Track how evenly tokens are distributed across experts
2. 🔍 Debug Routing Issues: Identify experts that are under/over-utilized  
3. 📈 Optimize Training: Adjust aux loss coefficients based on routing patterns
4. 🎲 Analyze Diversity: Measure routing entropy to ensure expert diversity
5. 💾 Save Routing Data: Store routing patterns for offline analysis

🚨 IMPORTANT NOTES:

- routing_map is only available during non-checkpointed forward passes
- In gradient checkpointing mode, routing maps are not collected
- routing_map shape: [num_tokens, num_experts] with boolean values
- Each MoE layer will have its own entry in moe_routing_maps list

🎉 Your modified GPTModel is now ready to provide detailed MoE routing insights!
""")

if __name__ == "__main__":
    example_usage()
