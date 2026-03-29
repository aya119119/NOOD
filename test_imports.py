#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
from pathlib import Path

root = Path.cwd()
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / 'Streamlit + Whisper'))
sys.path.insert(0, str(root / 'Speech Analysis'))

print(f"Root: {root}")
print(f"\nPath entries:")
for i, p in enumerate(sys.path[:5]):
    print(f"  {i}: {p}")

print("\n" + "="*60)
print("Testing imports...")
print("="*60)

try:
    from combined_analyzer import SpeechAndBodyLanguageAnalyzer
    print("✓ combined_analyzer imported successfully")
except Exception as e:
    print(f"✗ combined_analyzer failed: {e}")

try:
    from speech_analyzer import load_asr
    print("✓ speech_analyzer imported successfully")
except Exception as e:
    print(f"✗ speech_analyzer failed: {e}")

print("\nDone!")
