"""
Extract transcript from audio/video file.

Usage:
    python extract_transcript.py my_speech.mp4
    python extract_transcript.py my_speech.mp4 --output transcript.txt
"""

import argparse
import sys
from pathlib import Path

# Add Speech Analysis to path
sys.path.insert(0, str(Path(__file__).parent / "Speech Analysis"))

from speech_analyzer import analyze_speech_content, load_asr


def extract_transcript(audio_path: str, output_file: str = None) -> str:
    """
    Extract transcript from audio/video file.
    
    Args:
        audio_path: Path to audio/video file
        output_file: Optional path to save transcript
    
    Returns:
        Extracted transcript text
    """
    audio_path = Path(audio_path)
    
    if not audio_path.exists():
        print(f"✗ Error: File not found: {audio_path}")
        sys.exit(1)
    
    # Load ASR model
    print(f"Loading speech recognition model…")
    load_asr()
    
    # Extract transcript
    print(f"\nExtracting transcript from: {audio_path.name}")
    print("(This may take a few minutes on first run)…\n")
    
    try:
        asr = load_asr()
        transcript = asr.transcribe_file(str(audio_path))
        
        # Handle different return types
        if isinstance(transcript, (list, tuple)):
            transcript = transcript[0] if transcript else ""
        else:
            transcript = str(transcript)
        
        if not transcript or not transcript.strip():
            print("✗ Error: No speech detected or transcript is empty.")
            sys.exit(1)
        
        # Save to file if specified
        if output_file:
            output_path = Path(output_file)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(transcript)
            print(f"✓ Transcript saved to: {output_path}")
        
        # Show preview
        preview = transcript[:300] + ("…" if len(transcript) > 300 else "")
        print(f"\nTranscript Preview:")
        print(f"{'─' * 70}")
        print(preview)
        print(f"{'─' * 70}")
        print(f"Total length: {len(transcript)} characters, {len(transcript.split())} words\n")
        
        return transcript
    
    except Exception as e:
        print(f"✗ Error extracting transcript: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Extract transcript from audio/video file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        "audio",
        help="Path to audio/video file (.mp4, .wav, .mp3, .m4a, etc.)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save transcript (default: audio_filename.txt)",
    )
    
    args = parser.parse_args()
    
    # Determine output file
    if not args.output:
        audio_path = Path(args.audio)
        args.output = audio_path.stem + ".txt"
    
    # Extract transcript
    transcript = extract_transcript(args.audio, args.output)
    
    # Next steps
    if args.output:
        print(f"\nNext step:")
        print(f"  python combined_analyzer.py --transcript {args.output} --output report.json\n")


if __name__ == "__main__":
    main()
