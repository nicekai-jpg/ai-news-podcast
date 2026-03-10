
import asyncio
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ai_news_podcast.pipeline.tts_engine import synthesize

async def main():
    text = "[mood:hook] 欢迎收听 AI 每日先锋。[mood:excited] 今天我们要测试的是语音合成系统！[mood:calm] 听起来效果还不错吧？"
    output = Path("docs/test_tts.mp3")
    
    print(f"正在合成音频到: {output}")
    try:
        await synthesize(
            text,
            backend="edge-tts",
            voice="zh-CN-XiaoxiaoNeural",
            output_path=output,
            rate="+0%",
            pitch="+0Hz",
            volume="+0%",
            chunk_silence_ms=500
        )
        print("✅ 合成成功！")
    except Exception as e:
        print(f"❌ 合成失败: {e}")

if __name__ == "__main__":
    asyncio.run(main())
