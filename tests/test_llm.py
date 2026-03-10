import os
import logging
from openai import OpenAI
from pathlib import Path
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_llm")

def test():
    root = Path(__file__).resolve().parents[1]
    with open(root / "config" / "config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    
    llm_cfg = cfg.get("llm", {})
    env_key_name = llm_cfg.get("api_key_env", "OPENAI_API_KEY")
    api_key = os.environ.get(env_key_name)
    
    if not api_key:
        logger.error(f"环境变量中未找到 {env_key_name}。请在 .env 中设置真实的值。")
        return

    model = llm_cfg.get("model", "deepseek-chat")
    base_url = llm_cfg.get("base_url")
    
    logger.info(f"正在测试 LLM: model={model}, base_url={base_url}")
    
    client = OpenAI(api_key=api_key, base_url=base_url)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "你好，如果你能收到这条消息，请回复 'OK'。"}],
            max_tokens=10
        )
        logger.info(f"✅ LLM 响应成功: {response.choices[0].message.content}")
    except Exception as e:
        logger.error(f"❌ LLM 调用失败: {e}")

if __name__ == "__main__":
    test()
