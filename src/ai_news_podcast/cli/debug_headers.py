import os

import requests
from dotenv import load_dotenv

load_dotenv()


def debug_raw_request():
    api_key = os.environ.get("GEMINI_API_KEY")
    proxy = os.environ.get("HTTPS_PROXY")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-lite-latest:generateContent?key={api_key}"

    proxies = {"https": proxy} if proxy else None

    payload = {"contents": [{"parts": [{"text": "hi"}]}]}

    print(f"正在发送请求: {url}")

    try:
        response = requests.post(
            url,
            json=payload,
            proxies=proxies,
            headers={"Content-Type": "application/json"},
        )

        print(f"\nStatus: {response.status_code}")
        print("\n--- Headers ---")
        for k, v in response.headers.items():
            print(f"{k}: {v}")

        print("\n--- Body ---")
        print(response.text)

    except Exception as e:
        print(f"Error: {e}")


def entrypoint() -> None:
    """Synchronous entrypoint for console_scripts."""
    debug_raw_request()


if __name__ == "__main__":
    entrypoint()
