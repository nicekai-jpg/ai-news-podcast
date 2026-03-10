import os
import logging
from pathlib import Path
from dotenv import load_dotenv
import sys

# 设置日志
logging.basicConfig(level=logging.INFO)

# 加载环境变量
load_dotenv()

# 将 src 目录加入路径
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from ai_news_podcast.pipeline.scriptwriter import _call_llm

def test_rest_api():
    print("--- 正在测试重写后的 REST API 调用逻辑 ---")
    
    # 模拟配置
    llm_cfg = {
        "model": "gemini-flash-lite-latest",
        "temperature": 0.7,
        "max_output_tokens": 100
    }
    
    prompt = "你好，确认你收到了这条消息。"
    
    result = _call_llm(prompt, llm_cfg)
    
    if result:
        print(f"✅ 成功拿到回复: {result}")
    else:
        print("❌ 依然失败")

if __name__ == "__main__":
    test_rest_api()
