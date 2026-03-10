import os
from google import genai

def list_models():
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Error: No API key found")
        return

    client = genai.Client(api_key=api_key)
    print("Available models:")
    try:
        # 尝试列出模型。注意：新版 SDK 的 API 可能有所不同
        for model in client.models.list():
            print(f"- {model.name} (Supported: {model.supported_actions})")
    except Exception as e:
        print(f"Failed to list models: {e}")

if __name__ == "__main__":
    list_models()
