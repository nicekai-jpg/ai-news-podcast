import asyncio
from pathlib import Path
from ai_news_podcast.pipeline import tts_engine

dummy_script = """
[Host A] 听众朋友大家好，今天我们来测试一下双主播和背景混音的新功能。
[Host B] 没错！我现在的声音是由第二个音色生成的。
[Host A] 听起来非常清晰。同时，大家应该能听到垫在咱们声音底下的背景音乐。
[Host B] 当我们说话的时候，背景音乐的声音会被压低，这就是经典的播客“压限”或者叫 Ducking 效果。
[Host A] 非常棒。那今天的测试就到这里，感谢收听！
"""

async def test_tts_only():
    print("1. 解析双人剧本...")
    chunks = tts_engine.parse_dialogue_chunks(dummy_script)
    for c in chunks:
        print(f"[{c.host}] {c.text}")

    print("\n2. 进行 TTS 音频合成与 BGM 混音...")
    out_path = Path("data/assets/v2_test_output.mp3")
    bgm_path = Path("data/assets/bgm_placeholder.wav")
    
    await tts_engine.synthesize(
        dummy_script,
        backend="edge-tts",
        voices=("zh-CN-YunxiNeural", "zh-CN-XiaoxiaoNeural"),
        output_path=str(out_path),
        bgm_path=str(bgm_path) if bgm_path.exists() else None
    )
    print(f"3. 完成！请检查 {out_path}")

if __name__ == "__main__":
    asyncio.run(test_tts_only())
