#!/usr/bin/env python3
"""
测试 VSP 批量模式的目录结构
模拟 request.py 调用 VSP 处理多个任务
"""
import asyncio
import base64
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any

# 添加项目根目录到路径，以便导入模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from provider import VSPProvider


@dataclass
class RunConfig:
    """简化的运行配置"""
    mode: str = "vsp"
    provider: str = "openrouter"
    model_name: str = "gpt-4o"
    temperature: float = 0.0
    max_tokens: int = 4096
    consumer_size: int = 2
    vsp_batch_mode: bool = True
    vsp_batch_timestamp: str = None


def load_image_as_b64(image_path: str) -> str:
    """加载图片并转换为 base64"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')


def create_test_prompt(category: str, index: str, query: str, image_path: str) -> Dict[str, Any]:
    """创建测试 prompt"""
    return {
        "parts": [
            {"type": "text", "text": query},
            {"type": "image", "b64": load_image_as_b64(image_path)}
        ],
        "meta": {
            "category": category,
            "index": index
        }
    }


async def test_batch_mode():
    """测试批量模式"""
    print("="*80)
    print("测试 VSP 批量模式目录结构")
    print("="*80)
    
    # 生成批量时间戳
    batch_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"\n批量时间戳: {batch_timestamp}")
    
    # 初始化 VSP Provider（批量模式）
    vsp_path = os.environ.get("VSP_PATH", "/Users/yuantian/code/VisualSketchpad")
    provider = VSPProvider(
        vsp_path=vsp_path,
        batch_mode=True,
        batch_timestamp=batch_timestamp
    )
    
    # 配置
    cfg = RunConfig(
        vsp_batch_mode=True,
        vsp_batch_timestamp=batch_timestamp
    )
    
    # 准备测试任务（模拟 MM-SafetyBench 的结构）
    test_tasks = [
        {
            "category": "01-Illegal_Activity",
            "index": "0",
            "query": "Is the car beneath the cat?\nSelect from the following choices.\n(A) yes\n(B) no",
            "image_path": "/Users/yuantian/code/VisualSketchpad/tasks/blink_spatial/processed/val_Spatial_Relation_1/image.jpg"
        },
        {
            "category": "08-Political_Lobbying",
            "index": "5",
            "query": "Two points are circled on the image, labeled by A and B beside each circle. Which point is closer to the camera?\nSelect from the following choices.\n(A) A is closer\n(B) B is closer",
            "image_path": "/Users/yuantian/code/VisualSketchpad/tasks/blink_depth/processed/val_Relative_Depth_1/image.jpg"
        }
    ]
    
    print(f"\n将处理 {len(test_tasks)} 个测试任务:")
    for task in test_tasks:
        print(f"  - {task['category']}/{task['index']}")
    
    print(f"\n预期目录结构:")
    print(f"output/vsp_details/")
    print(f"└── vsp_{batch_timestamp}/")
    for task in test_tasks:
        print(f"    ├── {task['category']}/")
        print(f"    │   └── {task['index']}/")
        print(f"    │       ├── input/")
        print(f"    │       └── output/")
    
    print(f"\n预期输出文件:")
    print(f"output/vsp_{batch_timestamp}.jsonl")
    
    print("\n" + "-"*80)
    print("开始处理任务...")
    print("-"*80 + "\n")
    
    results = []
    for i, task in enumerate(test_tasks, 1):
        print(f"[{i}/{len(test_tasks)}] 处理 {task['category']}/{task['index']}...")
        
        try:
            prompt = create_test_prompt(
                task["category"],
                task["index"],
                task["query"],
                task["image_path"]
            )
            
            answer = await provider.send(prompt, cfg)
            results.append({
                "category": task["category"],
                "index": task["index"],
                "answer": answer,
                "status": "success"
            })
            print(f"  ✅ 完成: {answer}")
        
        except Exception as e:
            results.append({
                "category": task["category"],
                "index": task["index"],
                "error": str(e),
                "status": "failed"
            })
            print(f"  ❌ 失败: {str(e)[:100]}")
    
    # 显示结果摘要
    print("\n" + "="*80)
    print("结果摘要")
    print("="*80)
    
    success_count = sum(1 for r in results if r["status"] == "success")
    print(f"成功: {success_count}/{len(results)}")
    
    for result in results:
        status_icon = "✅" if result["status"] == "success" else "❌"
        print(f"{status_icon} {result['category']}/{result['index']}: ", end="")
        if result["status"] == "success":
            print(result["answer"])
        else:
            print(f"Error - {result['error'][:50]}")
    
    # 显示生成的目录结构
    print(f"\n生成的目录结构:")
    vsp_root = f"output/vsp_details/vsp_{batch_timestamp}"
    if os.path.exists(vsp_root):
        for category in os.listdir(vsp_root):
            category_path = os.path.join(vsp_root, category)
            if os.path.isdir(category_path):
                print(f"  {category}/")
                for index in os.listdir(category_path):
                    index_path = os.path.join(category_path, index)
                    if os.path.isdir(index_path):
                        print(f"    {index}/")
                        for subdir in ["input", "output"]:
                            subdir_path = os.path.join(index_path, subdir)
                            if os.path.exists(subdir_path):
                                print(f"      {subdir}/")
    else:
        print(f"  ⚠️ 目录不存在: {vsp_root}")
    
    print(f"\n✨ 测试完成！")
    print(f"详细输出目录: output/vsp_details/vsp_{batch_timestamp}/")
    print(f"汇总文件应为: output/vsp_{batch_timestamp}.jsonl")


if __name__ == "__main__":
    asyncio.run(test_batch_mode())

