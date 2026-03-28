"""
Combined Speech, Body Language, and Grammar Analysis Script

INSTALLATION REQUIRED:
Before running, install dependencies:
    pip install transformers torch torchaudio librosa soundfile speechbrain numpy scipy gradio

Grammar Correction Dependencies:
    pip install transformers sentencepiece

For GPU support (optional, enables faster processing):
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

Pipeline:
    ┌─────────────────────────────────────────────────────────┐
    │  INPUT: video.mp4 or transcript                         │
    │         ┌─────────────────┐   ┌──────────────────────┐  │
    │  video ─┤ Body Language   ├─┐ │ Speech -> Grammar    │  │
    │         │ Detector        │ │ │ Analyzer  Correction │  │
    │         └─────────────────┘ │ └──────────────────────┘  │
    │                             │                           │
    │                    MERGE + SCORE                        │
    │                            │                            │
    │                    combined_report.json                 │
    └─────────────────────────────────────────────────────────┘

Models Used:
    - Transcription: openai/whisper-large-v3-turbo or speechbrain ASR
    - Grammar Correction: AventIQ-AI/T5-small-grammar-correction  
      (Can be changed to any small T5 grammar model)
    - Body Language: TFLite model
"""

import argparse
import json
import sys
import warnings
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from datetime import datetime

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION & DEVICE SETUP
# ============================================================================

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

# Grammar Correction Model ID - Easy to swap
# IMPORTANT: Model ID is AventIQ-AI/T5-small-grammar-correction
# You can change this to any other T5 grammar correction model:
#   - prithivida/grammar_error_corretor_v1
#   - pszemraj/long-t5-tglobal-base-sci-simplify
#   - grammarly/coedit-large
GRAMMAR_MODEL_ID = "AventIQ-AI/T5-small-grammar-correction"

print(f"✓ Device: {DEVICE}")
print(f"✓ PyTorch dtype: {TORCH_DTYPE}")


# ============================================================================
# DATA CLASSES FOR GRAMMAR ANALYSIS
# ============================================================================

@dataclass
class GrammarError:
    """A single grammar error with correction example."""
    original: str
    corrected: str
    error_type: str  # e.g., "subject-verb agreement", "tense", "article"
    position: int    # approximate position in sentence


@dataclass
class GrammarReport:
    """Grammar correction analysis report."""
    error_count: int
    grammar_score: float  # 0-10 scale
    errors: List[GrammarError]
    error_examples: List[Dict[str, str]]  # [{"original": "...", "corrected": "..."}, ...]
    feedback: str
    model_used: str
    transcript_preview: str


@dataclass
class SentenceStructureReport:
    """Sentence structure analysis report."""
    avg_sentence_length: float  # Average number of words per sentence
    sentence_length_std: float  # Standard deviation (variety measure)
    total_sentences: int
    short_sentences: int  # Sentences <= 5 words
    long_sentences: int   # Sentences >= 20 words
    variety_level: str    # "low", "moderate", "high"
    sentence_length_category: str  # "choppy", "balanced", "dense"
    feedback: str
    suggestions: List[str]


# ============================================================================
# GRAMMAR CORRECTION MODULE
# ============================================================================

_grammar_tokenizer = None
_grammar_model = None


def load_grammar_models():
    """Load grammar correction model (lazy loading)."""
    global _grammar_tokenizer, _grammar_model
    
    if _grammar_tokenizer is None or _grammar_model is None:
        print(f"  Loading grammar model ({GRAMMAR_MODEL_ID})…", flush=True)
        try:
            _grammar_tokenizer = AutoTokenizer.from_pretrained(GRAMMAR_MODEL_ID)
            _grammar_model = AutoModelForSeq2SeqLM.from_pretrained(GRAMMAR_MODEL_ID).to(DEVICE)
            _grammar_model.eval()
        except Exception as e:
            print(f"  [WARNING] Failed to load grammar model ({GRAMMAR_MODEL_ID}): {e}", flush=True)
            print(f"  Falling back to rule-based detection.", flush=True)
            return False
    
    return True


def correct_text(text: str, max_length: int = 512) -> str:
    """
    Use the grammar correction model to correct a sentence.
    
    Args:
        text: Input sentence/text to correct
        max_length: Maximum token length for the model
    
    Returns:
        Corrected text string
    """
    if not text or len(text.strip()) == 0:
        return text
    
    try:
        inputs = _grammar_tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True).to(DEVICE)
        
        with torch.no_grad():
            outputs = _grammar_model.generate(
                **inputs,
                max_length=max_length,
                num_beams=5,
                early_stopping=True,
                temperature=0.7,
            )
        
        corrected = _grammar_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return corrected.strip()
    
    except Exception as e:
        print(f"  [WARNING] Grammar correction failed for text: {e}", flush=True)
        return text


def analyze_grammar(transcript: str, verbose: bool = False) -> GrammarReport:
    """
    Analyze grammar in transcript using T5-based grammar correction model.
    
    Args:
        transcript: Full speech transcript text
        verbose: Whether to print progress
    
    Returns:
        GrammarReport object with errors, examples, and score
    """
    if not transcript or len(transcript.strip()) == 0:
        # Empty transcript
        return GrammarReport(
            error_count=0,
            grammar_score=10.0,
            errors=[],
            error_examples=[],
            feedback="No transcript provided for grammar analysis.",
            model_used=GRAMMAR_MODEL_ID,
            transcript_preview="[empty]",
        )
    
    if verbose:
        print("\n  Analyzing grammar…", flush=True)
    
    # Load models
    model_loaded = load_grammar_models()
    if not model_loaded:
        return _fallback_grammar_analysis(transcript)
    
    # Split transcript into sentences for analysis
    sentences = _split_sentences(transcript)
    
    if verbose:
        print(f"    Processing {len(sentences)} sentences…", flush=True)
    
    errors = []
    error_examples = []
    
    for i, sentence in enumerate(sentences):
        if len(sentence.strip()) < 3:
            continue
        
        # Correct the sentence
        corrected = correct_text(sentence)
        
        # Check if correction differs from original
        if corrected.lower() != sentence.lower():
            # Extract the differences (simplified)
            error = GrammarError(
                original=sentence,
                corrected=corrected,
                error_type="grammar",
                position=i
            )
            errors.append(error)
            
            error_examples.append({
                "original": sentence,
                "corrected": corrected,
            })
    
    # Calculate grammar score (0-10)
    # Score decreases with more errors
    error_rate = len(errors) / max(len(sentences), 1)
    # Map error_rate [0, 1] to score [10, 0] with some smoothing
    grammar_score = max(0.0, 10.0 - (error_rate * 8.0))
    grammar_score = round(grammar_score, 1)
    
    # Generate feedback
    feedback = _generate_grammar_feedback(len(errors), grammar_score, error_rate)
    
    # Limit examples to top 3 for brevity
    top_examples = error_examples[:3]
    
    if verbose:
        print(f"    Found {len(errors)} grammar issues. Score: {grammar_score}/10", flush=True)
    
    return GrammarReport(
        error_count=len(errors),
        grammar_score=grammar_score,
        errors=errors,
        error_examples=top_examples,
        feedback=feedback,
        model_used=GRAMMAR_MODEL_ID,
        transcript_preview=transcript[:100] + ("…" if len(transcript) > 100 else ""),
    )


def _split_sentences(text: str) -> List[str]:
    """Simple sentence splitter (split on . ! ?)"""
    import re
    # Split on sentence-ending punctuation
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def _generate_grammar_feedback(error_count: int, score: float, error_rate: float) -> str:
    """Generate actionable feedback based on grammar analysis."""
    if error_count == 0:
        return "Excellent grammar throughout. No errors detected."
    
    if score >= 8.5:
        return f"Minor grammar issues ({error_count} errors, {error_rate:.1%} error rate). Overall strong language use."
    elif score >= 7.0:
        return f"Moderate grammar issues ({error_count} errors, {error_rate:.1%} error rate). Review and correct sentences for clarity."
    elif score >= 5.0:
        return f"Several grammar issues ({error_count} errors, {error_rate:.1%} error rate). Consider proofreading and revision."
    else:
        return f"Significant grammar issues ({error_count} errors, {error_rate:.1%} error rate). Substantial revision recommended."


def _fallback_grammar_analysis(transcript: str) -> GrammarReport:
    """Fallback rule-based grammar analysis if model loading fails."""
    print("  Using rule-based grammar detection (model load failed)…", flush=True)
    
    sentences = _split_sentences(transcript)
    errors = []
    
    # Simple rule-based checks
    for sentence in sentences:
        if _has_simple_grammar_issues(sentence):
            errors.append(GrammarError(
                original=sentence,
                corrected="[suggested correction]",
                error_type="possible_issue",
                position=0,
            ))
    
    error_rate = len(errors) / max(len(sentences), 1)
    grammar_score = max(0.0, 10.0 - (error_rate * 8.0))
    
    return GrammarReport(
        error_count=len(errors),
        grammar_score=round(grammar_score, 1),
        errors=errors[:3],
        error_examples=[],
        feedback=_generate_grammar_feedback(len(errors), grammar_score, error_rate),
        model_used=f"{GRAMMAR_MODEL_ID} (fallback)",
        transcript_preview=transcript[:100] + ("…" if len(transcript) > 100 else ""),
    )


def _has_simple_grammar_issues(sentence: str) -> bool:
    """Very basic grammar rule checks."""
    lower = sentence.lower()
    
    # Check for common patterns (very basic)
    if "was going" in lower or "were going" in lower:
        return True
    if sentence.count('"') % 2 != 0:  # Mismatched quotes
        return True
    if "  " in sentence:  # Double spaces
        return True
    
    return False


# ============================================================================
# SENTENCE STRUCTURE ANALYSIS MODULE
# ============================================================================

def analyze_sentence_structure(transcript: str, verbose: bool = False) -> SentenceStructureReport:
    """
    Analyze sentence structure in transcript.
    
    Metrics:
      - Average sentence length (words)
      - Sentence length variety (standard deviation)
      - Proportion of short vs. long sentences
      - Overall classification (choppy, balanced, dense)
    
    Args:
        transcript: Full speech transcript text
        verbose: Whether to print progress
    
    Returns:
        SentenceStructureReport object with metrics and feedback
    """
    if not transcript or len(transcript.strip()) == 0:
        return SentenceStructureReport(
            avg_sentence_length=0.0,
            sentence_length_std=0.0,
            total_sentences=0,
            short_sentences=0,
            long_sentences=0,
            variety_level="none",
            sentence_length_category="empty",
            feedback="No transcript provided for sentence structure analysis.",
            suggestions=[],
        )
    
    if verbose:
        print("\n  Analyzing sentence structure…", flush=True)
    
    # Split into sentences
    sentences = _split_sentences(transcript)
    
    if len(sentences) == 0:
        return SentenceStructureReport(
            avg_sentence_length=0.0,
            sentence_length_std=0.0,
            total_sentences=0,
            short_sentences=0,
            long_sentences=0,
            variety_level="none",
            sentence_length_category="empty",
            feedback="No sentences found in transcript.",
            suggestions=[],
        )
    
    # Calculate sentence lengths (in words)
    sentence_lengths = []
    for sentence in sentences:
        # Count words (split by whitespace, filter empty)
        words = [w for w in sentence.split() if w.strip()]
        sentence_lengths.append(len(words))
    
    # Calculate statistics
    avg_length = np.mean(sentence_lengths)
    length_std = np.std(sentence_lengths)
    
    # Count short and long sentences
    short_count = sum(1 for length in sentence_lengths if length <= 5)
    long_count = sum(1 for length in sentence_lengths if length >= 20)
    
    # Determine variety level (based on std dev)
    if length_std < 3.0:
        variety_level = "low"
    elif length_std < 6.0:
        variety_level = "moderate"
    else:
        variety_level = "high"
    
    # Determine sentence structure category
    short_pct = short_count / len(sentences) * 100
    long_pct = long_count / len(sentences) * 100
    
    if short_pct > 40:
        category = "choppy"
    elif long_pct > 30:
        category = "dense"
    else:
        category = "balanced"
    
    # Generate feedback and suggestions
    feedback, suggestions = _generate_sentence_feedback(
        avg_length, length_std, variety_level, category,
        short_count, long_count, len(sentences)
    )
    
    if verbose:
        print(f"    Avg length: {avg_length:.1f} words, Variety: {variety_level}, Category: {category}", flush=True)
    
    return SentenceStructureReport(
        avg_sentence_length=round(avg_length, 1),
        sentence_length_std=round(length_std, 1),
        total_sentences=len(sentences),
        short_sentences=short_count,
        long_sentences=long_count,
        variety_level=variety_level,
        sentence_length_category=category,
        feedback=feedback,
        suggestions=suggestions,
    )


def _generate_sentence_feedback(
    avg_length: float,
    std_dev: float,
    variety_level: str,
    category: str,
    short_count: int,
    long_count: int,
    total_sentences: int,
) -> Tuple[str, List[str]]:
    """Generate feedback and actionable suggestions for sentence structure."""
    suggestions = []
    feedback_parts = []
    
    # Feedback on average length
    if avg_length < 10:
        feedback_parts.append(f"Short average sentence length ({avg_length:.1f} words)")
    elif avg_length > 25:
        feedback_parts.append(f"Long average sentence length ({avg_length:.1f} words)")
    else:
        feedback_parts.append(f"Moderate average sentence length ({avg_length:.1f} words)")
    
    # Feedback on variety
    short_pct = short_count / total_sentences * 100 if total_sentences > 0 else 0
    long_pct = long_count / total_sentences * 100 if total_sentences > 0 else 0
    
    if variety_level == "low":
        feedback_parts.append(f"Low variety in sentence length (σ={std_dev:.1f})")
        suggestions.append("Vary your sentence length to maintain audience interest.")
        suggestions.append("Alternate between short punchy sentences and longer complex ones.")
    elif variety_level == "moderate":
        feedback_parts.append(f"Moderate variety in sentence length (σ={std_dev:.1f})")
    else:
        feedback_parts.append(f"High variety in sentence length (σ={std_dev:.1f})")
        suggestions.append("Good balance in sentence length variation.")
    
    # Feedback on category
    if category == "choppy":
        feedback_parts.append(f"Many short sentences ({short_pct:.0f}% are ≤5 words)")
        if "Vary your sentence length" not in suggestions:
            suggestions.append("Combine some short sentences to create more complex ideas.")
        suggestions.append("Use longer sentences for ideas that deserve elaboration.")
    elif category == "dense":
        feedback_parts.append(f"Many long sentences ({long_pct:.0f}% are ≥20 words)")
        suggestions.append("Break up very long sentences into shorter, digestible segments.")
        suggestions.append("Use periodic sentences: short sentence at the end for impact.")
    else:
        feedback_parts.append(f"Balanced sentence structure")
        suggestions.append("Maintain this balance between short and long sentences.")
    
    feedback = ". ".join(feedback_parts) + "."
    
    return feedback, suggestions


# ============================================================================
# COMBINED ANALYZER CLASS
# ============================================================================

class SpeechAndBodyLanguageAnalyzer:
    """Main analyzer combining speech, body language, grammar, and sentence structure analysis."""
    
    def __init__(self, transcript: str = "", speech_report: Optional[dict] = None, 
                 body_language_report: Optional[dict] = None):
        """
        Initialize analyzer.
        
        Args:
            transcript: Speech transcript (used for grammar and sentence structure analysis)
            speech_report: Pre-computed speech analysis report (optional)
            body_language_report: Pre-computed body language analysis report (optional)
        """
        self.transcript = transcript
        self.speech_report = speech_report or {}
        self.body_language_report = body_language_report or {}
        self.grammar_report = None
        self.sentence_structure_report = None
    
    def analyze_grammar(self, verbose: bool = False) -> GrammarReport:
        """Analyze grammar in the transcript."""
        print("\n══ Grammar Analysis starting…", flush=True)
        t0 = __import__('time').time()
        
        self.grammar_report = analyze_grammar(self.transcript, verbose=verbose)
        
        elapsed = __import__('time').time() - t0
        print(f"══ Grammar analysis done ({elapsed:.1f}s)", flush=True)
        
        return self.grammar_report
    
    def analyze_sentence_structure(self, verbose: bool = False) -> SentenceStructureReport:
        """Analyze sentence structure in the transcript."""
        print("\n══ Sentence Structure Analysis starting…", flush=True)
        t0 = __import__('time').time()
        
        self.sentence_structure_report = analyze_sentence_structure(self.transcript, verbose=verbose)
        
        elapsed = __import__('time').time() - t0
        print(f"══ Sentence structure analysis done ({elapsed:.1f}s)", flush=True)
        
        return self.sentence_structure_report
    
    def analyze_speech(self) -> Dict:
        """
        Placeholder for speech analysis.
        In production, this would call speech_analyzer.py
        """
        print("  Analyzing speech data…", flush=True)
        return self.speech_report
    
    def analyze_body_language(self) -> Dict:
        """
        Placeholder for body language analysis.
        In production, this would call body_language_detector.py
        """
        print("  Analyzing body language data…", flush=True)
        return self.body_language_report
    
    def run_analysis(self) -> Dict:
        """Run all analyses and generate combined report."""
        print("\n" + "=" * 70)
        print("  COMBINED ANALYSIS - Speech, Body Language, Grammar, & Structure")
        print("=" * 70)
        
        # Run analyses
        self.analyze_speech()
        self.analyze_body_language()
        self.analyze_grammar(verbose=True)
        self.analyze_sentence_structure(verbose=True)
        
        # Generate combined report
        report = self.generate_combined_report()
        
        return report
    
    def generate_combined_report(self) -> Dict:
        """
        Generate combined report with all analyses.
        
        Format:
        - Speech Performance (WPM, filler rate, enthusiasm, etc.)
        - Body Language (emotions, confidence)
        - Language & Content:
          - Grammar (errors, score)
          - Sentence Structure (length, variety, suggestions)
        - Final Confidence Score
        """
        if self.grammar_report is None:
            self.analyze_grammar()
        if self.sentence_structure_report is None:
            self.analyze_sentence_structure()
        
        # Build the combined report
        report = {
            "meta": {
                "generated_at": datetime.now().isoformat(),
                "transcript_preview": self.transcript[:150] + ("…" if len(self.transcript) > 150 else ""),
            },
            "speech": self.speech_report,
            "body_language": self.body_language_report,
            "language_and_content": {
                "grammar": {
                    "errors_detected": self.grammar_report.error_count,
                    "examples": self.grammar_report.error_examples,
                    "grammar_score": self.grammar_report.grammar_score,
                    "feedback": self.grammar_report.feedback,
                    "model_used": self.grammar_report.model_used,
                },
                "sentence_structure": {
                    "average_length": self.sentence_structure_report.avg_sentence_length,
                    "length_variety": self.sentence_structure_report.sentence_length_std,
                    "total_sentences": self.sentence_structure_report.total_sentences,
                    "short_sentences": self.sentence_structure_report.short_sentences,
                    "long_sentences": self.sentence_structure_report.long_sentences,
                    "variety_level": self.sentence_structure_report.variety_level,
                    "category": self.sentence_structure_report.sentence_length_category,
                    "feedback": self.sentence_structure_report.feedback,
                    "suggestions": self.sentence_structure_report.suggestions,
                }
            },
            "final_confidence_score": 0.0,  # Placeholder for overall confidence
        }
        
        return report
    
    def print_summary(self):
        """Print human-readable summary to console."""
        if self.grammar_report is None or self.sentence_structure_report is None:
            print("\n  [Warning] Not all analyses completed.")
            return
        
        sep = "─" * 70
        
        print(f"\n{'=' * 70}")
        print("  COMBINED ANALYSIS SUMMARY")
        print(f"{'=' * 70}")
        
        # Speech section
        if self.speech_report:
            print(f"\n  ▸ Speech Performance")
            print(f"    [Speech metrics would appear here]")
        
        # Body language section
        if self.body_language_report:
            print(f"\n  ▸ Body Language")
            print(f"    [Body language metrics would appear here]")
        
        # Language & Content section
        print(f"\n  ▸ Language & Content")
        
        # Grammar subsection
        print(f"\n    Grammar:")
        print(f"      Errors detected      : {self.grammar_report.error_count}")
        
        if self.grammar_report.error_examples:
            print(f"      Examples:")
            for i, example in enumerate(self.grammar_report.error_examples, 1):
                print(f"        {i}. \"{example['original']}\"")
                print(f"           → \"{example['corrected']}\"")
        
        print(f"      Grammar score        : {self.grammar_report.grammar_score}/10")
        print(f"      Feedback             : {self.grammar_report.feedback}")
        
        # Sentence structure subsection
        print(f"\n    Sentence Structure:")
        print(f"      Average length       : {self.sentence_structure_report.avg_sentence_length} words")
        print(f"      Length variety       : {self.sentence_structure_report.sentence_length_std} σ ({self.sentence_structure_report.variety_level})")
        print(f"      Total sentences      : {self.sentence_structure_report.total_sentences}")
        print(f"      Short sentences      : {self.sentence_structure_report.short_sentences} (≤5 words)")
        print(f"      Long sentences       : {self.sentence_structure_report.long_sentences} (≥20 words)")
        print(f"      Category             : {self.sentence_structure_report.sentence_length_category}")
        print(f"      Feedback             : {self.sentence_structure_report.feedback}")
        
        if self.sentence_structure_report.suggestions:
            print(f"      Suggestions:")
            for i, suggestion in enumerate(self.sentence_structure_report.suggestions, 1):
                print(f"        {i}. {suggestion}")
        
        print(f"\n  ▸ Final Confidence Score")
        print(f"    [Overall confidence score would appear here]")
        
        print(f"\n{'=' * 70}\n")


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    """Command-line interface for combined analysis."""
    parser = argparse.ArgumentParser(
        description="Combined Speech, Body Language, and Grammar Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    # Input options
    parser.add_argument(
        "--transcript",
        type=str,
        required=True,
        help="Path to transcript file or raw text",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="combined_analysis_report.json",
        help="Path for output JSON report",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output report as JSON",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    
    args = parser.parse_args()
    
    # Load transcript
    if Path(args.transcript).exists():
        with open(args.transcript, "r", encoding="utf-8") as f:
            transcript = f.read()
    else:
        transcript = args.transcript
    
    # Run analysis
    analyzer = SpeechAndBodyLanguageAnalyzer(transcript=transcript)
    
    # Run grammar analysis (main feature)
    analyzer.analyze_grammar(verbose=args.verbose)
    
    # Generate report
    report = analyzer.generate_combined_report()
    
    # Output results
    if args.json or args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Report saved to: {args.output}")
    
    # Print summary to console
    analyzer.print_summary()
    
    return report


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """
    CLI Usage:
        python combined_analyzer.py --transcript "transcript.txt" --output report.json --verbose
        python combined_analyzer.py --transcript "You was going to the store." --verbose
    
    Programmatic Usage:
        from combined_analyzer import SpeechAndBodyLanguageAnalyzer
        
        analyzer = SpeechAndBodyLanguageAnalyzer(
            transcript="Your speech text here..."
        )
        analyzer.analyze_grammar(verbose=True)
        analyzer.analyze_sentence_structure(verbose=True)
        report = analyzer.generate_combined_report()
        analyzer.print_summary()
    """
    
    main()