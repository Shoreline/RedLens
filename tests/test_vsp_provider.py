#!/usr/bin/env python3
"""
测试VSPProvider的实现
"""

import asyncio
import base64
import os
import sys
from datetime import datetime

# 添加项目根目录到路径，以便导入模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from provider import VSPProvider
from request import RunConfig

async def test_vsp_provider():
    """测试VSPProvider的基本功能"""
    
    # 生成时间戳
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # 创建测试配置
    cfg = RunConfig(
        mode="vsp",
        provider="openrouter",
        model_name="gpt-4o",  # VSP内部会使用自己的模型配置
        temperature=0.0,
        max_tokens=2048
    )
    cfg.vsp_batch_timestamp = timestamp
    
    # 创建VSPProvider实例
    provider = VSPProvider(batch_timestamp=timestamp)
    
    # 使用VSP自带的测试数据（blink_spatial任务）
    vsp_test_image = "/Users/yuantian/code/VisualSketchpad/tasks/blink_spatial/processed/val_Spatial_Relation_1/image.jpg"
    vsp_test_query = "Is the car beneath the cat?\nSelect from the following choices.\n(A) yes\n(B) no"
    
    if not os.path.exists(vsp_test_image):
        print(f"Error: VSP test image not found at {vsp_test_image}")
        print("Please use a valid path to a VSP test image.")
        return
    
    # 读取测试图片并转换为base64
    with open(vsp_test_image, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")
    
    # 创建一个vision任务测试（需要包含 meta 信息）
    test_prompt = {
        "parts": [
            {
                "type": "text",
                "text": vsp_test_query
            },
            {
                "type": "image",
                "b64": image_b64,
                "mime": "image/jpeg"
            }
        ],
        "meta": {
            "category": "test_category",
            "index": "0"
        }
    }
    
    try:
        print("Testing VSPProvider...")
        print(f"VSP Path: {provider.vsp_path}")
        print(f"Agent Path: {provider.agent_path}")
        
        # 检查VSP路径是否存在
        if not os.path.exists(provider.vsp_path):
            print(f"Error: VSP path does not exist: {provider.vsp_path}")
            return
        
        if not os.path.exists(provider.agent_path):
            print(f"Error: Agent path does not exist: {provider.agent_path}")
            return
            
        print("VSP paths exist, attempting to call VSP...")
        
        # 调用VSP
        result = await provider.send(test_prompt, cfg)
        
        print(f"VSP Result: {result}")
        
    except Exception as e:
        print(f"Error testing VSPProvider: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_vsp_provider())
