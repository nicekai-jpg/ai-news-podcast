
import re
import sys
from pathlib import Path

# Mocking the functions from scriptwriter.py to test them
def _sanitize_for_tts(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\\n", "\n")
    text = re.sub(r"\[(?:FACT|INFERENCE|OPINION)\]\s*", "", text)
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text

def _normalize_mood_tags(text: str) -> str:
    valid_moods = {"hook", "excited", "serious", "calm", "emphasis", "closing"}
    lines = text.split("\n")
    result = []
    for line in lines:
        line = line.strip()
        if not line: continue
        m = re.match(r"\[mood:(\w+)\]\s*(.*)", line)
        if m:
            mood, content = m.group(1), m.group(2)
            if mood not in valid_moods: mood = "calm"
            if content.strip(): result.append(f"[mood:{mood}] {content.strip()}")
        else:
            if line.strip(): result.append(f"[mood:calm] {line.strip()}")
    return "\n\n".join(result) + "\n" if result else "[mood:calm] 暂无内容。\n"

test_input = "[mood:hook] Hello\\nWorld!  This is a   test.\n\n[mood:excited] Amazing!\\n\\nNew line here.\nSome extra text without tag."
print("--- INPUT ---")
print(repr(test_input))
print("--- OUTPUT ---")
sanitized = _sanitize_for_tts(test_input)
normalized = _normalize_mood_tags(sanitized)
print(normalized)
print("--- REPR OUTPUT ---")
print(repr(normalized))
