import os

from dotenv import load_dotenv
from openai import OpenAI

# 加载 .env 拿 API Key 和代理
load_dotenv()


def test_gemini_via_openai(model_name):
    print(f"\n--- [OpenAI协议] 测试模型: {model_name} ---")
    try:
        # 从环境变量拿 Key 和 Base URL
        api_key = os.environ.get("GEMINI_API_KEY")
        base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"

        # 创建客户端（它会自动识别系统的 HTTP_PROXY）
        client = OpenAI(api_key=api_key, base_url=base_url)

        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "你好，请回复'收到'"}],
            max_tokens=10,
        )
        print(f"✅ 成功! 返回内容: {response.choices[0].message.content}")
        return True
    except Exception as e:
        print(f"❌ 失败! 错误信息: {e}")
        return False


if __name__ == "__main__":
    # 在 OpenAI 协议下，模型名字通常不需要加 models/ 前缀
    models_to_test = [
        "gemini-2.0-flash",
        "gemini-1.5-flash",
        "gemini-flash-lite-latest",
    ]

    for m in models_to_test:
        test_gemini_via_openai(m)
