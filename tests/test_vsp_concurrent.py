#!/usr/bin/env python3
"""
测试 VSPProvider 的并发能力
同时运行两个不同的 vision 任务
"""
import asyncio
import base64
import os
import sys
import time
from dataclasses import dataclass
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
    consumer_size: int = 2  # 允许2个并发


def load_image_as_b64(image_path: str) -> str:
    """加载图片并转换为 base64"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')


def create_prompt(query: str, image_path: str) -> Dict[str, Any]:
    """创建 prompt 结构"""
    # 清理 query 中的 <img> 标签
    import re
    query_clean = re.sub(r"<img src='[^']*'>", "", query).strip()
    
    return {
        "parts": [
            {"type": "text", "text": query_clean},
            {"type": "image", "b64": load_image_as_b64(image_path)}
        ]
    }


async def run_task(provider: VSPProvider, cfg: RunConfig, task_name: str, 
                   query: str, image_path: str) -> tuple:
    """运行单个任务"""
    print(f"\n[{task_name}] Starting...")
    start_time = time.time()
    
    try:
        prompt = create_prompt(query, image_path)
        result = await provider.send(prompt, cfg)
        elapsed = time.time() - start_time
        
        print(f"[{task_name}] ✅ Completed in {elapsed:.1f}s")
        print(f"[{task_name}] Result: {result}")
        return task_name, "success", result, elapsed
    
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"[{task_name}] ❌ Failed in {elapsed:.1f}s: {str(e)[:100]}")
        return task_name, "failed", str(e), elapsed


async def test_concurrent():
    """测试并发执行"""
    print("="*60)
    print("Testing VSPProvider Concurrent Execution")
    print("="*60)
    
    # 初始化 provider
    vsp_path = os.environ.get("VSP_PATH", "/Users/yuantian/code/VisualSketchpad")
    provider = VSPProvider(vsp_path=vsp_path)
    
    # 配置
    cfg = RunConfig(consumer_size=2)
    
    # 任务1: Spatial Relation (猫和车)
    task1_query = """Is the car beneath the cat?
Select from the following choices.
(A) yes
(B) no"""
    task1_image = "/Users/yuantian/code/VisualSketchpad/tasks/blink_spatial/processed/val_Spatial_Relation_1/image.jpg"
    
    # 任务2: Relative Depth (深度判断)
    task2_query = """Two points are circled on the image, labeled by A and B beside each circle. Which point is closer to the camera?
Select from the following choices.
(A) A is closer
(B) B is closer"""
    task2_image = "/Users/yuantian/code/VisualSketchpad/tasks/blink_depth/processed/val_Relative_Depth_1/image.jpg"
    
    # 检查文件是否存在
    for name, path in [("Task1", task1_image), ("Task2", task2_image)]:
        if not os.path.exists(path):
            print(f"❌ {name} image not found: {path}")
            return
    
    print(f"\nTask 1: Spatial Relation (car beneath cat?)")
    print(f"Task 2: Relative Depth (which point closer?)")
    print(f"\nRunning both tasks concurrently...")
    print("-"*60)
    
    # 记录开始时间
    overall_start = time.time()
    
    # 并发执行两个任务
    results = await asyncio.gather(
        run_task(provider, cfg, "Task1-Spatial", task1_query, task1_image),
        run_task(provider, cfg, "Task2-Depth", task2_query, task2_image),
        return_exceptions=True
    )
    
    overall_elapsed = time.time() - overall_start
    
    # 汇总结果
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    success_count = 0
    for result in results:
        if isinstance(result, Exception):
            print(f"❌ Exception: {result}")
        else:
            task_name, status, output, elapsed = result
            if status == "success":
                success_count += 1
                print(f"✅ {task_name}: {output} ({elapsed:.1f}s)")
            else:
                print(f"❌ {task_name}: Failed - {output[:100]} ({elapsed:.1f}s)")
    
    print(f"\nTotal time: {overall_elapsed:.1f}s")
    print(f"Success rate: {success_count}/{len(results)}")
    
    if success_count == len(results):
        print("\n🎉 All tasks completed successfully!")
        print("✅ VSP concurrent execution is working!")
    else:
        print("\n⚠️ Some tasks failed. Check the output above for details.")


if __name__ == "__main__":
    asyncio.run(test_concurrent())

