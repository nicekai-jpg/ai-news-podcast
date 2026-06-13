#!/usr/bin/env python3
"""One-off: generate CosyVoice reference WAVs from Edge-TTS, then commit to asset/refs/."""

import asyncio
import subprocess
from pathlib import Path

import edge_tts

ROOT = Path(__file__).resolve().parents[1]
REFS = ROOT / "asset" / "refs"

HOST_A_TEXT = (
    "云阳，咱们今天这选题，你看啊，从底层神经网络的突破，"
    "到应用端那些草根团队基于协议和开源模型做的创新，真的非常有意思。"
)
HOST_B_TEXT = (
    "云阳老师，你又带上了技术极客的视角了啊！"
    "你能不能给咱们听众翻译翻译，这个机制被 AI 破解到底意味着什么？"
)


async def _edge_to_wav(voice: str, text: str, out_wav: Path) -> None:
    mp3 = out_wav.with_suffix(".mp3")
    await edge_tts.Communicate(text, voice=voice).save(str(mp3))
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(mp3), "-ar", "16000", "-ac", "1", str(out_wav)],
        check=True,
    )
    mp3.unlink(missing_ok=True)


async def main() -> None:
    REFS.mkdir(parents=True, exist_ok=True)
    await _edge_to_wav("zh-CN-YunjianNeural", HOST_A_TEXT, REFS / "host_a_ref.wav")
    await _edge_to_wav("zh-CN-XiaoxiaoNeural", HOST_B_TEXT, REFS / "host_b_ref.wav")
    (REFS / "host_a_ref.txt").write_text(HOST_A_TEXT, encoding="utf-8")
    (REFS / "host_b_ref.txt").write_text(HOST_B_TEXT, encoding="utf-8")
    print("Reference audio written to", REFS)


if __name__ == "__main__":
    asyncio.run(main())
