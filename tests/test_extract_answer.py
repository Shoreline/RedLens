#!/usr/bin/env python3
"""
测试从现有的 VSP 输出中提取答案
"""
import os
import sys
# 添加项目根目录到路径，以便导入模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from provider import VSPProvider

def test_extract_from_existing():
    """测试从已有的 VSP 输出中提取答案"""
    
    # 使用你已经运行成功的 VSP 输出
    vsp_output_dir = "/Users/yuantian/code/RedLens/output/vsp_details/vsp_2025-10-30_23-48-42/08-Political_Lobbying/0/output"
    
    print("="*80)
    print("测试从 VSP debug log 提取答案")
    print("="*80)
    print(f"\n测试目录: {vsp_output_dir}")
    
    # 创建 provider 实例（只用来调用 _extract_answer_vsp 方法）
    provider = VSPProvider(batch_timestamp="test")
    
    # 提取答案
    answer = provider._extract_answer_vsp(vsp_output_dir)
    
    print(f"\n提取的答案:")
    print("-"*80)
    print(answer)
    print("-"*80)
    
    # 显示答案长度
    print(f"\n答案长度: {len(answer)} 字符")
    print(f"答案行数: {len(answer.split(chr(10)))} 行")
    
    # 检查是否包含关键内容
    if "Define" in answer or "script" in answer or "campaign" in answer:
        print("\n✅ 答案提取成功！包含预期的关键词")
    else:
        print("\n⚠️ 答案可能不完整")
    
    # 测试第二个任务
    print("\n" + "="*80)
    print("测试第二个任务")
    print("="*80)
    
    vsp_output_dir2 = "/Users/yuantian/code/RedLens/output/vsp_details/vsp_2025-10-30_23-48-42/08-Political_Lobbying/1/output"
    
    if os.path.exists(vsp_output_dir2):
        answer2 = provider._extract_answer_vsp(vsp_output_dir2)
        print(f"\n提取的答案:")
        print("-"*80)
        print(answer2[:500] + "..." if len(answer2) > 500 else answer2)
        print("-"*80)
        print(f"\n答案长度: {len(answer2)} 字符")
    else:
        print(f"\n目录不存在: {vsp_output_dir2}")

if __name__ == "__main__":
    test_extract_from_existing()

