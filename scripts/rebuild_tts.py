import sys
import asyncio
import yaml
from pathlib import Path

# Add project src to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from ai_news_podcast.pipeline.tts_engine import synthesize
from ai_news_podcast.utils import read_yaml

async def main():
    print("Starting TTS reconstruction on existing script...")
    cfg = read_yaml(project_root / "config/config.yaml")
    tts_cfg = cfg.get("tts", {})
    build_cfg = cfg.get("build", {})
    
    backend = str(tts_cfg.get("backend") or "edge-tts")
    voice = str(tts_cfg.get("voice") or "zh-CN-XiaoxiaoNeural")
    rate = str(tts_cfg.get("rate") or "+0%")
    volume = str(tts_cfg.get("volume") or "+0%")
    pitch = str(tts_cfg.get("pitch") or "+0Hz")
    mood_presets = tts_cfg.get("mood_presets")
    chunk_silence_ms = int(tts_cfg.get("chunk_silence_ms", 500))
    
    episode_id = "2026-06-04"
    episodes_dir = project_root / str(build_cfg.get("episodes_dir") or "site/episodes")
    
    mp3_path = episodes_dir / f"{episode_id}.mp3"
    transcript_path = episodes_dir / f"{episode_id}.txt"
    
    if not transcript_path.exists():
        print(f"Error: script file {transcript_path} does not exist!")
        return 1
        
    print(f"Reading existing script from {transcript_path}...")
    script_text = transcript_path.read_text(encoding="utf-8")
    
    print("Synthesizing audio and computing timestamps...")
    # Pass voices if dual voice Yunxi and Xiaoxiao is preferred
    # Yunxi is Host A, Xiaoxiao is Host B
    voices = ("zh-CN-YunxiNeural", "zh-CN-XiaoxiaoNeural")
    
    # Run synthesis
    await synthesize(
        script_text,
        backend=backend,
        voices=voices,
        output_path=mp3_path,
        rate=rate,
        volume=volume,
        pitch=pitch,
        mood_presets=(mood_presets if isinstance(mood_presets, dict) else None),
        chunk_silence_ms=chunk_silence_ms,
        transcript_path=transcript_path,
    )
    
    print(f"Audio saved to: {mp3_path}")
    print(f"Timestamped script saved to: {transcript_path}")
    
    # Run the rebuild script to regenerate index.html and sync to data/_preview
    print("Regenerating site and syncing preview...")
    import rebuild_site
    rebuild_site.rebuild()
    print("All tasks complete!")

if __name__ == "__main__":
    asyncio.run(main())
