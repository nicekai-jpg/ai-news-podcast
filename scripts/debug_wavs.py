import soundfile as sf
import numpy as np
from pathlib import Path

out_dir = Path("/Users/limingkai/nas/project/ai-news-podcast/site/episodes/benchmark")
files = sorted(list(out_dir.glob("moss_*.wav")))

print("Analyzing MOSS-TTS-Nano generated wav files:")
for f in files:
    try:
        data, samplerate = sf.read(str(f))
        duration = len(data) / samplerate
        
        # Calculate stats
        max_val = np.max(np.abs(data)) if len(data) > 0 else 0
        mean_val = np.mean(np.abs(data)) if len(data) > 0 else 0
        # Count non-zero samples (e.g. absolute value > 1e-4)
        non_zero_ratio = np.mean(np.abs(data) > 1e-4) if len(data) > 0 else 0
        
        print(f"{f.name}:")
        print(f"  Sample Rate: {samplerate} Hz")
        print(f"  Duration: {duration:.2f} s")
        print(f"  Max Amplitude: {max_val:.5f}")
        print(f"  Mean Absolute Amplitude: {mean_val:.5f}")
        print(f"  Non-zero ratio: {non_zero_ratio*100:.2f}%")
        
        # Check if it is silence or noise
        if len(data) == 0:
            print("  STATUS: EMPTY FILE (Zero samples)")
        elif max_val < 0.01:
            print("  STATUS: SILENT (Maximum amplitude too low)")
        elif mean_val < 0.001:
            print("  STATUS: ALMOST SILENT / GLITCHED")
        elif non_zero_ratio > 0.95 and mean_val > 0.3:
            print("  STATUS: CONSTANT LOUD NOISE (Possible glitch)")
        else:
            print("  STATUS: Dynamic sound data (normal range)")
            
    except Exception as e:
        print(f"Error reading {f.name}: {e}")
