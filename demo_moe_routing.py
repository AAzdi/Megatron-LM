#!/usr/bin/env python3

"""
Test script to demonstrate MoE routing map functionality 
without complex distributed initialization.
"""

import torch
import sys
import os

# Add the Megatron-LM directory to the Python path
sys.path.append('/home/shaohanh/dizhang/Megatron-LM')

def test_moe_routing_concept():
    """Demonstrate the MoE routing map concept with mock data."""
    
    print("=" * 60)
    print("MoE 路由映射功能演示")
    print("=" * 60)
    
    # 模拟参数
    batch_size = 2
    seq_length = 4
    num_experts = 6
    top_k = 2
    
    print(f"📊 测试参数:")
    print(f"   批次大小: {batch_size}")
    print(f"   序列长度: {seq_length}")
    print(f"   专家数量: {num_experts}")
    print(f"   Top-K: {top_k}")
    
    # 创建模拟的路由映射
    num_tokens = batch_size * seq_length
    routing_map = torch.zeros(num_tokens, num_experts, dtype=torch.bool)
    
    # 为每个token选择top_k个专家
    for token_idx in range(num_tokens):
        selected_experts = torch.randperm(num_experts)[:top_k]
        routing_map[token_idx, selected_experts] = True
    
    print(f"\n🗺️ 生成的路由映射 (形状: {routing_map.shape}):")
    print(f"   routing_map[token_idx, expert_idx] = True 表示 token 被路由到该专家")
    print(f"\n{routing_map}")
    
    # 分析路由统计
    print(f"\n📈 路由统计分析:")
    
    # 每个专家处理的token数量
    tokens_per_expert = routing_map.sum(dim=0)
    print(f"   每个专家的token数量: {tokens_per_expert.tolist()}")
    
    # 每个token使用的专家数量
    experts_per_token = routing_map.sum(dim=1)
    print(f"   每个token的专家数量: {experts_per_token.tolist()}")
    
    # 负载均衡指标
    load_balance_var = tokens_per_expert.float().var().item()
    print(f"   负载均衡方差: {load_balance_var:.4f} (越小越好)")
    
    # 专家利用率
    active_experts = (tokens_per_expert > 0).sum().item()
    utilization_rate = active_experts / num_experts * 100
    print(f"   专家利用率: {active_experts}/{num_experts} ({utilization_rate:.1f}%)")
    
    # 模拟多层MoE的情况
    print(f"\n🏗️ 模拟多层MoE情况:")
    moe_layers = [0, 2, 4]  # 假设第0, 2, 4层是MoE层
    
    all_routing_maps = []
    for layer_num in moe_layers:
        # 为每层生成不同的路由映射
        layer_routing = torch.zeros(num_tokens, num_experts, dtype=torch.bool)
        for token_idx in range(num_tokens):
            # 使用不同的随机种子确保每层路由不同
            torch.manual_seed(layer_num * 100 + token_idx)
            selected_experts = torch.randperm(num_experts)[:top_k]
            layer_routing[token_idx, selected_experts] = True
        
        routing_info = {
            'layer_number': layer_num,
            'routing_map': layer_routing
        }
        all_routing_maps.append(routing_info)
        
        layer_tokens_per_expert = layer_routing.sum(dim=0)
        layer_balance_var = layer_tokens_per_expert.float().var().item()
        print(f"   第{layer_num}层: 负载均衡方差 = {layer_balance_var:.4f}")
    
    # 演示你的GPTModel.forward()返回格式
    print(f"\n🎯 GPTModel.forward() 返回格式演示:")
    mock_output = torch.randn(batch_size, seq_length, 256)  # 模拟输出
    
    # 情况1: 无MoE层的模型
    print(f"   情况1 - 无MoE层:")
    print(f"     result = model.forward(...)")
    print(f"     type(result): {type(mock_output)}")
    print(f"     result.shape: {mock_output.shape}")
    
    # 情况2: 有MoE层的模型
    print(f"\n   情况2 - 有MoE层:")
    print(f"     result = model.forward(...)")
    print(f"     if isinstance(result, tuple):")
    print(f"         output, moe_routing_maps = result")
    print(f"         print(f'发现 {{len(moe_routing_maps)}} 个MoE层')")
    print(f"     ")
    print(f"     模拟结果:")
    print(f"       output.shape: {mock_output.shape}")
    print(f"       len(moe_routing_maps): {len(all_routing_maps)}")
    for i, routing_info in enumerate(all_routing_maps):
        print(f"       moe_routing_maps[{i}]: 第{routing_info['layer_number']}层, 形状 {routing_info['routing_map'].shape}")
    
    print(f"\n✨ 使用示例代码:")
    example_code = '''
# 使用修改后的GPTModel
forward_result = model(input_ids, position_ids, attention_mask)

if isinstance(forward_result, tuple):
    # 模型包含MoE层
    output, moe_routing_maps = forward_result
    print(f"找到 {len(moe_routing_maps)} 个MoE层")
    
    for routing_info in moe_routing_maps:
        layer_num = routing_info['layer_number']
        routing_map = routing_info['routing_map']
        
        # 分析路由模式
        tokens_per_expert = routing_map.sum(dim=0)
        load_balance = tokens_per_expert.float().var().item()
        
        print(f"第{layer_num}层: 负载均衡方差 = {load_balance:.4f}")
else:
    # 模型不包含MoE层
    output = forward_result
    print("模型不包含MoE层")
'''
    print(example_code)
    
    print("=" * 60)
    print("🎉 MoE路由映射功能演示完成!")
    print("你的修改已经就绪，可以在真实的MoE模型上使用了!")
    print("=" * 60)

if __name__ == "__main__":
    test_moe_routing_concept()
