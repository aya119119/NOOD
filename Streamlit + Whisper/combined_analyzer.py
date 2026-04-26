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


@dataclass
class VocabularyReport:
    """Vocabulary and lexical diversity analysis report."""
    total_words: int
    unique_words: int
    type_token_ratio: float  # TTR: unique_words / total_words (0.0-1.0)
    vocabulary_level: str   # "basic", "intermediate", "advanced"
    richness_score: float   # 0-10 scale
    common_words_count: int  # Words in top 1000 most common
    rare_words_count: int    # Words not in top 1000
    feedback: str
    suggestions: List[str]


@dataclass
class FluencyReport:
    """Fluency and speech flow analysis report."""
    repetition_count: int     # Number of unique repetitions detected
    repetition_examples: List[Dict[str, any]]  # [{"word": "...", "count": N}, ...]
    fluency_score: float      # 0-10 scale
    fluency_level: str        # "poor", "fair", "good", "excellent"
    primary_issues: List[str] # Main fluency problems identified
    feedback: str
    suggestions: List[str]


@dataclass
class LanguageAndContentReport:
    """Consolidated Language & Content feedback combining all sub-analyses."""
    grammar_score: float
    sentence_structure_score: float
    vocabulary_score: float
    fluency_score: float
    overall_language_score: float  # Weighted average of above 4
    strengths: List[str]
    areas_for_improvement: List[str]
    top_recommendations: List[str]  # Prioritized action items
    consolidated_feedback: str


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
# VOCABULARY ANALYSIS MODULE
# ============================================================================

# Top 1000 most common English words (frequency-based, for richness scoring)
# This is a simplified list; source: https://en.wiktionary.org/wiki/Wiktionary:Frequency_lists
COMMON_WORDS = {
    'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
    'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
    'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
    'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what',
    'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go', 'me',
    'when', 'make', 'can', 'like', 'time', 'no', 'just', 'him', 'know', 'take',
    'people', 'into', 'year', 'your', 'good', 'some', 'could', 'them', 'see', 'other',
    'than', 'then', 'now', 'look', 'only', 'come', 'its', 'over', 'think', 'also',
    'back', 'after', 'use', 'two', 'how', 'our', 'work', 'first', 'well', 'way',
    'even', 'new', 'want', 'because', 'any', 'these', 'give', 'day', 'most', 'us',
    'is', 'was', 'are', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
    'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'shall',
    'am', 'are', 'is', 'was', 'were', 'where', 'why', 'how', 'what', 'when', 'which'
}


def analyze_vocabulary(transcript: str, verbose: bool = False) -> VocabularyReport:
    """
    Analyze vocabulary richness and lexical diversity in transcript.
    
    Metrics:
      - Type-Token Ratio (TTR): unique_words / total_words
        * Low TTR (< 0.40): less diverse vocabulary
        * High TTR (> 0.60): more diverse, sophisticated vocabulary
      - Unique word count
      - Rare vs. common word ratio
      - Overall vocabulary richness score (0-10)
    
    Args:
        transcript: Full speech transcript text
        verbose: Whether to print progress
    
    Returns:
        VocabularyReport object with metrics and suggestions
    """
    if not transcript or len(transcript.strip()) == 0:
        return VocabularyReport(
            total_words=0,
            unique_words=0,
            type_token_ratio=0.0,
            vocabulary_level="none",
            richness_score=0.0,
            common_words_count=0,
            rare_words_count=0,
            feedback="No transcript provided for vocabulary analysis.",
            suggestions=[],
        )
    
    if verbose:
        print("\n  Analyzing vocabulary…", flush=True)
    
    # Extract words (lowercase, alphanumeric only)
    import re
    words = re.findall(r'\b[a-z]+\b', transcript.lower())
    
    if len(words) == 0:
        return VocabularyReport(
            total_words=0,
            unique_words=0,
            type_token_ratio=0.0,
            vocabulary_level="none",
            richness_score=0.0,
            common_words_count=0,
            rare_words_count=0,
            feedback="No words found in transcript.",
            suggestions=[],
        )
    
    # Calculate basic metrics
    total_words = len(words)
    unique_words = len(set(words))
    type_token_ratio = unique_words / total_words if total_words > 0 else 0.0
    
    # Count common vs. rare words
    unique_word_set = set(words)
    common_count = sum(1 for w in unique_word_set if w in COMMON_WORDS)
    rare_count = unique_words - common_count
    
    # Determine vocabulary level based on TTR
    if type_token_ratio < 0.40:
        vocab_level = "basic"
    elif type_token_ratio < 0.60:
        vocab_level = "intermediate"
    else:
        vocab_level = "advanced"
    
    # Calculate richness score (0-10)
    # Formula: combine TTR, unique word count, and rare word ratio
    ttr_score = min(10.0, type_token_ratio * 15)  # Scale 0-1 to 0-15, cap at 10
    unique_score = min(10.0, (unique_words / max(total_words * 0.5, 1)) * 10)  # Reward high unique count
    rare_score = (rare_count / unique_words * 10) if unique_words > 0 else 0  # Reward rare words
    
    richness_score = (ttr_score * 0.4 + unique_score * 0.3 + rare_score * 0.3)
    richness_score = round(min(10.0, richness_score), 1)
    
    # Generate feedback and suggestions
    feedback, suggestions = _generate_vocabulary_feedback(
        type_token_ratio, vocab_level, richness_score, unique_words, total_words
    )
    
    if verbose:
        print(f"    TTR: {type_token_ratio:.2f}, Level: {vocab_level}, Score: {richness_score}/10", flush=True)
    
    return VocabularyReport(
        total_words=total_words,
        unique_words=unique_words,
        type_token_ratio=round(type_token_ratio, 3),
        vocabulary_level=vocab_level,
        richness_score=richness_score,
        common_words_count=common_count,
        rare_words_count=rare_count,
        feedback=feedback,
        suggestions=suggestions,
    )


def _generate_vocabulary_feedback(
    ttr: float,
    vocab_level: str,
    richness_score: float,
    unique_words: int,
    total_words: int,
) -> Tuple[str, List[str]]:
    """Generate feedback and suggestions for vocabulary richness."""
    suggestions = []
    feedback_parts = []
    
    # Level description
    if vocab_level == "basic":
        feedback_parts.append(f"Basic vocabulary level (TTR={ttr:.2f})")
        suggestions.append("Use more varied and precise words to enhance sophistication.")
        suggestions.append("Replace common words with synonyms to increase diversity.")
        suggestions.append("Explore subject-specific terminology relevant to your topic.")
    elif vocab_level == "intermediate":
        feedback_parts.append(f"Intermediate vocabulary level (TTR={ttr:.2f})")
        suggestions.append("Good vocabulary variety, but room for enhancement.")
        suggestions.append("Introduce more domain-specific or less common words where appropriate.")
    else:
        feedback_parts.append(f"Advanced vocabulary level (TTR={ttr:.2f})")
        suggestions.append("Excellent lexical diversity. Maintain this level of vocabulary richness.")
    
    # Score description
    if richness_score >= 8.0:
        feedback_parts.append(f"Excellent richness score ({richness_score}/10)")
    elif richness_score >= 6.0:
        feedback_parts.append(f"Good richness score ({richness_score}/10)")
    elif richness_score >= 4.0:
        feedback_parts.append(f"Moderate richness score ({richness_score}/10)")
    else:
        feedback_parts.append(f"Low richness score ({richness_score}/10)")
    
    # Unique word feedback
    unique_rate = unique_words / total_words if total_words > 0 else 0
    if unique_rate > 0.6:
        feedback_parts.append(f"High word uniqueness ({unique_words} unique from {total_words} total)")
    elif unique_rate > 0.4:
        feedback_parts.append(f"Moderate word uniqueness ({unique_words} unique from {total_words} total)")
    else:
        feedback_parts.append(f"Low word uniqueness ({unique_words} unique from {total_words} total)")
        if "words to enhance" not in suggestions[0].lower():
            suggestions.insert(0, "Reduce word repetition by using synonyms and varied phrasing.")
    
    feedback = ". ".join(feedback_parts) + "."
    
    return feedback, suggestions


# ============================================================================
# FLUENCY ANALYSIS MODULE
# ============================================================================

def analyze_fluency(transcript: str, verbose: bool = False) -> FluencyReport:
    """
    Analyze fluency and speech flow smoothness in transcript.
    
    Metrics:
      - Word repetitions (how many times same words repeated)
      - Fluency score based on repetition frequency
      - Identification of primary fluency issues
    
    Args:
        transcript: Full speech transcript text
        verbose: Whether to print progress
    
    Returns:
        FluencyReport object with repetition analysis and fluency score
    """
    if not transcript or len(transcript.strip()) == 0:
        return FluencyReport(
            repetition_count=0,
            repetition_examples=[],
            fluency_score=10.0,
            fluency_level="excellent",
            primary_issues=[],
            feedback="No transcript provided for fluency analysis.",
            suggestions=[],
        )
    
    if verbose:
        print("\n  Analyzing fluency…", flush=True)
    
    # Extract words
    import re
    words = re.findall(r'\b[a-z]+\b', transcript.lower())
    
    if len(words) == 0:
        return FluencyReport(
            repetition_count=0,
            repetition_examples=[],
            fluency_score=10.0,
            fluency_level="excellent",
            primary_issues=[],
            feedback="No words found in transcript.",
            suggestions=[],
        )
    
    # Identify repetitions (words that appear very close together)
    repetitions = {}
    window_size = 20  # Check within 20-word windows
    
    for i in range(len(words) - window_size):
        window = words[i:i + window_size]
        word_counts = {}
        for word in window:
            # Skip very common filler words
            if word not in {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'and', 'or', 'but', 'in', 'on', 'at'}:
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Flag words repeated 3+ times in a window
        for word, count in word_counts.items():
            if count >= 3:
                if word not in repetitions:
                    repetitions[word] = 0
                repetitions[word] += 1
    
    # Sort by frequency
    sorted_reps = sorted(repetitions.items(), key=lambda x: x[1], reverse=True)[:5]
    repetition_examples = [{"word": word, "count": count} for word, count in sorted_reps]
    repetition_count = len(repetitions)
    
    # Calculate fluency score (0-10)
    # Fewer repetitions = higher fluency
    rep_rate = repetition_count / max((len(words) / 20), 1)  # Normalize by 20-word chunks
    fluency_score = max(0.0, 10.0 - (rep_rate * 2.0))
    fluency_score = round(fluency_score, 1)
    
    # Determine fluency level
    if fluency_score >= 8.5:
        fluency_level = "excellent"
    elif fluency_score >= 7.0:
        fluency_level = "good"
    elif fluency_score >= 5.0:
        fluency_level = "fair"
    else:
        fluency_level = "poor"
    
    # Generate feedback
    feedback, suggestions, primary_issues = _generate_fluency_feedback(
        repetition_count, fluency_score, repetition_examples
    )
    
    if verbose:
        print(f"    Repetitions: {repetition_count}, Score: {fluency_score}/10, Level: {fluency_level}", flush=True)
    
    return FluencyReport(
        repetition_count=repetition_count,
        repetition_examples=repetition_examples,
        fluency_score=fluency_score,
        fluency_level=fluency_level,
        primary_issues=primary_issues,
        feedback=feedback,
        suggestions=suggestions,
    )


def _generate_fluency_feedback(
    repetition_count: int,
    fluency_score: float,
    examples: List[Dict[str, any]],
) -> Tuple[str, List[str], List[str]]:
    """Generate feedback and suggestions for fluency."""
    suggestions = []
    primary_issues = []
    feedback_parts = []
    
    if fluency_score >= 8.5:
        feedback_parts.append(f"Excellent fluency (score: {fluency_score}/10)")
        suggestions.append("Your speech flows naturally with minimal repetition. Maintain this smooth delivery.")
    elif fluency_score >= 7.0:
        feedback_parts.append(f"Good fluency (score: {fluency_score}/10)")
        feedback_parts.append(f"Some word repetition detected ({repetition_count} unique repeated words)")
        primary_issues.append("Minor repetition patterns")
        suggestions.append("Consider varying your word choices slightly to avoid repetition patterns.")
        if examples:
            suggestions.append(f"Watch out for repeating: {', '.join([e['word'] for e in examples[:2]])}")
    elif fluency_score >= 5.0:
        feedback_parts.append(f"Fair fluency (score: {fluency_score}/10)")
        feedback_parts.append(f"Notable word repetition detected ({repetition_count} unique repeated words)")
        primary_issues.append("Moderate repetition patterns")
        primary_issues.append("Speech flow interrupted by repeated phrases")
        suggestions.append("Reduce word repetitions by using synonyms or restructuring sentences.")
        suggestions.append("Practice pausing naturally instead of repeating words for filler.")
        if examples:
            suggestions.append(f"Priority words to replace: {', '.join([e['word'] for e in examples[:3]])}")
    else:
        feedback_parts.append(f"Needs improvement ({fluency_score}/10)")
        feedback_parts.append(f"Heavy repetition detected ({repetition_count} unique repeated words)")
        primary_issues.append("Significant repetition throughout")
        primary_issues.append("Flow frequently interrupted by repeated words/phrases")
        suggestions.append("Record and listen to yourself—identify repetitive patterns.")
        suggestions.append("Use synonyms or rephrase sentences to break repetition habits.")
        suggestions.append("Practice slow, deliberate speech with intentional pauses.")
    
    feedback = ". ".join(feedback_parts) + "."
    
    return feedback, suggestions, primary_issues


# ============================================================================
# CONSOLIDATED LANGUAGE & CONTENT FEEDBACK
# ============================================================================

def generate_language_and_content_report(
    grammar_report: GrammarReport,
    sentence_structure_report: SentenceStructureReport,
    vocabulary_report: VocabularyReport,
    fluency_report: FluencyReport,
) -> LanguageAndContentReport:
    """
    Generate consolidated Language & Content feedback combining all sub-analyses.
    
    Weighting (all equally weighted at 25% each):
      - Grammar: 25%
      - Sentence Structure: 25%
      - Vocabulary: 25%
      - Fluency: 25%
    
    Adjust weights below if needed:
    """
    # WEIGHT CONFIGURATION - Adjust these to change the importance of each component
    GRAMMAR_WEIGHT = 0.25      # 25% - Grammatical correctness
    STRUCTURE_WEIGHT = 0.25    # 25% - Sentence construction & variety
    VOCABULARY_WEIGHT = 0.25   # 25% - Word choice & richness
    FLUENCY_WEIGHT = 0.25      # 25% - Speech flow & smoothness
    
    # Convert sentence structure category to score (0-10)
    structure_score = 7.0  # Default
    if sentence_structure_report.sentence_length_category == "choppy":
        structure_score = 6.0
    elif sentence_structure_report.sentence_length_category == "balanced":
        structure_score = 8.5
    elif sentence_structure_report.sentence_length_category == "dense":
        structure_score = 6.5
    
    # Adjust for variety level
    if sentence_structure_report.variety_level == "low":
        structure_score -= 1.5
    elif sentence_structure_report.variety_level == "high":
        structure_score += 0.5
    
    structure_score = max(0.0, min(10.0, structure_score))
    
    # Calculate overall language score
    overall_score = (
        grammar_report.grammar_score * GRAMMAR_WEIGHT +
        structure_score * STRUCTURE_WEIGHT +
        vocabulary_report.richness_score * VOCABULARY_WEIGHT +
        fluency_report.fluency_score * FLUENCY_WEIGHT
    )
    overall_score = round(overall_score, 1)
    
    # Identify strengths and areas for improvement
    strengths = []
    areas_for_improvement = []
    
    if grammar_report.grammar_score >= 8.0:
        strengths.append("Strong grammatical accuracy")
    elif grammar_report.grammar_score < 6.0:
        areas_for_improvement.append("Grammar needs attention (multiple errors detected)")
    
    if sentence_structure_report.variety_level == "high":
        strengths.append("Excellent sentence variety and structure")
    elif sentence_structure_report.variety_level == "low":
        areas_for_improvement.append("Sentence structure lacks variety (repetitive patterns)")
    
    if vocabulary_report.vocabulary_level == "advanced":
        strengths.append("Rich and sophisticated vocabulary")
    elif vocabulary_report.vocabulary_level == "basic":
        areas_for_improvement.append("Vocabulary is basic (limited word diversity)")
    
    if fluency_report.fluency_level == "excellent":
        strengths.append("Excellent speech fluency and flow")
    elif fluency_report.fluency_level in ["poor", "fair"]:
        areas_for_improvement.append(f"Fluency issues ({', '.join(fluency_report.primary_issues)})")
    
    # Generate top recommendations (prioritized)
    top_recommendations = []
    
    # Priority 1: Grammar errors
    if grammar_report.grammar_score < 7.0:
        top_recommendations.append(f"Priority: Fix grammar errors ({grammar_report.error_count} issues found)")
    
    # Priority 2: Fluency
    if fluency_report.fluency_score < 7.0:
        top_recommendations.append("Important: Reduce word repetitions (practice smoother delivery)")
    
    # Priority 3: Sentence structure
    if sentence_structure_report.variety_level == "low":
        top_recommendations.append("Vary your sentence lengths and structures")
    
    # Priority 4: Vocabulary
    if vocabulary_report.vocabulary_level == "basic":
        top_recommendations.append("Expand vocabulary with more precise and varied words")
    
    # Keep recommendations to top 3-4
    top_recommendations = top_recommendations[:4]
    
    # Generate consolidated feedback message
    feedback_lines = [
        f"Overall Language & Content Score: {overall_score}/10"
    ]
    
    if strengths:
        feedback_lines.append(f"Strengths: {', '.join(strengths)}")
    
    if areas_for_improvement:
        feedback_lines.append(f"Areas to improve: {', '.join(areas_for_improvement)}")
    
    consolidated_feedback = "\n".join(feedback_lines)
    
    return LanguageAndContentReport(
        grammar_score=grammar_report.grammar_score,
        sentence_structure_score=round(structure_score, 1),
        vocabulary_score=vocabulary_report.richness_score,
        fluency_score=fluency_report.fluency_score,
        overall_language_score=overall_score,
        strengths=strengths,
        areas_for_improvement=areas_for_improvement,
        top_recommendations=top_recommendations,
        consolidated_feedback=consolidated_feedback,
    )


# ============================================================================
# OVERALL CONFIDENCE SCORE CALCULATION
# ============================================================================

def calculate_overall_confidence_score(
    language_and_content_score: float,
    speech_metrics: Optional[Dict] = None,
    body_language_metrics: Optional[Dict] = None,
) -> Tuple[float, str, str]:
    """
    Calculate overall presentation confidence score (1-10).
    
    SCORING WEIGHTS:
    - Language & Content: 25%   (Grammar, Sentence Structure, Vocabulary, Fluency)
    - Speech Metrics:     40%   (WPM, filler rate, prosody, emotion)
    - Body Language:      35%   (Emotions, confidence, gestures)
    
    Adjust weights below if needed. For example, to emphasize speech more:
    Change to LANGUAGE_WEIGHT=0.20, SPEECH_WEIGHT=0.50, BODY_WEIGHT=0.30
    """
    # SCORE WEIGHT CONFIGURATION - Main tuning knobs
    LANGUAGE_WEIGHT = 0.25   # 25% - Content quality
    SPEECH_WEIGHT = 0.40     # 40% - Delivery & vocal performance
    BODY_WEIGHT = 0.35       # 35% - Non-verbal communication
    
    # Start with language & content score
    overall = language_and_content_score * LANGUAGE_WEIGHT
    
    # Add speech metrics (if available)
    if speech_metrics and isinstance(speech_metrics, dict):
        speech_score = speech_metrics.get('overall', 5.0)
        if isinstance(speech_score, dict) and 'score' in speech_score:
            speech_score = speech_score['score']
        # Normalize speech score to 0-10 if needed
        if speech_score < 0:
            speech_score = (speech_score + 1) / 2 * 10  # Convert from [-1, 1] to [0, 10]
        overall += (speech_score * SPEECH_WEIGHT)
    else:
        overall += (5.0 * SPEECH_WEIGHT)  # Neutral default
    
    # Add body language metrics (if available)
    if body_language_metrics and isinstance(body_language_metrics, dict):
        # Try to extract a confidence/score value
        body_score = 5.0
        if 'confidence' in body_language_metrics:
            body_score = body_language_metrics['confidence'] * 10
        elif 'score' in body_language_metrics:
            body_score = body_language_metrics['score']
        overall += (body_score * BODY_WEIGHT)
    else:
        overall += (5.0 * BODY_WEIGHT)  # Neutral default
    
    overall = round(min(10.0, max(0.0, overall)), 1)
    
    # Generate letter grade
    if overall >= 9.0:
        grade = "A+"
        proficiency = "Excellent"
    elif overall >= 8.0:
        grade = "A"
        proficiency = "Excellent"
    elif overall >= 7.0:
        grade = "B+"
        proficiency = "Very Good"
    elif overall >= 6.0:
        grade = "B"
        proficiency = "Good"
    elif overall >= 5.0:
        grade = "C+"
        proficiency = "Satisfactory"
    elif overall >= 4.0:
        grade = "C"
        proficiency = "Needs Improvement"
    elif overall >= 3.0:
        grade = "D"
        proficiency = "Significant Work Needed"
    else:
        grade = "F"
        proficiency = "Requires Major Revision"
    
    return overall, grade, proficiency


# ============================================================================
# COMBINED ANALYZER CLASS
# ============================================================================

class SpeechAndBodyLanguageAnalyzer:
    """Main analyzer combining speech, body language, grammar, sentence structure, vocabulary, and fluency."""
    
    def __init__(self, transcript: str = "", speech_report: Optional[dict] = None, 
                 body_language_report: Optional[dict] = None):
        """
        Initialize analyzer.
        
        Args:
            transcript: Speech transcript (used for language & content analysis)
            speech_report: Pre-computed speech analysis report (optional)
            body_language_report: Pre-computed body language analysis report (optional)
        """
        self.transcript = transcript
        self.speech_report = speech_report or {}
        self.body_language_report = body_language_report or {}
        self.grammar_report = None
        self.sentence_structure_report = None
        self.vocabulary_report = None
        self.fluency_report = None
        self.language_and_content_report = None
        self.overall_score = 0.0
        self.overall_grade = ""
        self.overall_proficiency = ""
    
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
    
    def analyze_vocabulary(self, verbose: bool = False) -> VocabularyReport:
        """Analyze vocabulary and lexical diversity in the transcript."""
        print("\n══ Vocabulary Analysis starting…", flush=True)
        t0 = __import__('time').time()
        
        self.vocabulary_report = analyze_vocabulary(self.transcript, verbose=verbose)
        
        elapsed = __import__('time').time() - t0
        print(f"══ Vocabulary analysis done ({elapsed:.1f}s)", flush=True)
        
        return self.vocabulary_report
    
    def analyze_fluency(self, verbose: bool = False) -> FluencyReport:
        """Analyze fluency and speech smoothness in the transcript."""
        print("\n══ Fluency Analysis starting…", flush=True)
        t0 = __import__('time').time()
        
        self.fluency_report = analyze_fluency(self.transcript, verbose=verbose)
        
        elapsed = __import__('time').time() - t0
        print(f"══ Fluency analysis done ({elapsed:.1f}s)", flush=True)
        
        return self.fluency_report
    
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
        print("  COMPREHENSIVE COMMUNICATION ANALYSIS")
        print("=" * 70)
        
        # Run analyses
        self.analyze_speech()
        self.analyze_body_language()
        self.analyze_grammar(verbose=True)
        self.analyze_sentence_structure(verbose=True)
        self.analyze_vocabulary(verbose=True)
        self.analyze_fluency(verbose=True)
        
        # Generate consolidated Language & Content report
        if self.grammar_report and self.sentence_structure_report and \
           self.vocabulary_report and self.fluency_report:
            self.language_and_content_report = generate_language_and_content_report(
                self.grammar_report,
                self.sentence_structure_report,
                self.vocabulary_report,
                self.fluency_report,
            )
        
        # Calculate overall confidence score
        self.overall_score, self.overall_grade, self.overall_proficiency = \
            calculate_overall_confidence_score(
                self.language_and_content_report.overall_language_score if self.language_and_content_report else 5.0,
                self.speech_report,
                self.body_language_report,
            )
        
        # Generate combined report
        report = self.generate_combined_report()
        
        return report
    
    def generate_combined_report(self) -> Dict:
        """
        Generate comprehensive combined report with all analyses.
        """
        if self.grammar_report is None:
            self.analyze_grammar()
        if self.sentence_structure_report is None:
            self.analyze_sentence_structure()
        if self.vocabulary_report is None:
            self.analyze_vocabulary()
        if self.fluency_report is None:
            self.analyze_fluency()
        if self.language_and_content_report is None:
            self.language_and_content_report = generate_language_and_content_report(
                self.grammar_report,
                self.sentence_structure_report,
                self.vocabulary_report,
                self.fluency_report,
            )
        
        # Build the comprehensive report
        report = {
            "meta": {
                "generated_at": datetime.now().isoformat(),
                "transcript_preview": self.transcript[:200] + ("…" if len(self.transcript) > 200 else ""),
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
                },
                "vocabulary": {
                    "total_words": self.vocabulary_report.total_words,
                    "unique_words": self.vocabulary_report.unique_words,
                    "type_token_ratio": self.vocabulary_report.type_token_ratio,
                    "vocabulary_level": self.vocabulary_report.vocabulary_level,
                    "richness_score": self.vocabulary_report.richness_score,
                    "common_words": self.vocabulary_report.common_words_count,
                    "rare_words": self.vocabulary_report.rare_words_count,
                    "feedback": self.vocabulary_report.feedback,
                    "suggestions": self.vocabulary_report.suggestions,
                },
                "fluency": {
                    "repetition_count": self.fluency_report.repetition_count,
                    "repetition_examples": self.fluency_report.repetition_examples,
                    "fluency_score": self.fluency_report.fluency_score,
                    "fluency_level": self.fluency_report.fluency_level,
                    "primary_issues": self.fluency_report.primary_issues,
                    "feedback": self.fluency_report.feedback,
                    "suggestions": self.fluency_report.suggestions,
                },
                "consolidated": {
                    "grammar_score": self.language_and_content_report.grammar_score,
                    "sentence_structure_score": self.language_and_content_report.sentence_structure_score,
                    "vocabulary_score": self.language_and_content_report.vocabulary_score,
                    "fluency_score": self.language_and_content_report.fluency_score,
                    "overall_language_score": self.language_and_content_report.overall_language_score,
                    "strengths": self.language_and_content_report.strengths,
                    "areas_for_improvement": self.language_and_content_report.areas_for_improvement,
                    "top_recommendations": self.language_and_content_report.top_recommendations,
                    "feedback": self.language_and_content_report.consolidated_feedback,
                }
            },
            "overall_confidence_score": self.overall_score,
            "overall_grade": self.overall_grade,
            "overall_proficiency": self.overall_proficiency,
        }
        
        return report
    
    def print_summary(self):
        """Print professional, encouraging coach-style summary to console."""
        if self.grammar_report is None or self.language_and_content_report is None:
            print("\n  [Warning] Not all analyses completed.")
            return
        
        print(f"\n{'═' * 75}")
        print("                    PRESENTATION ANALYSIS REPORT")
        print(f"                        Professional Feedback")
        print(f"{'═' * 75}")
        
        # Overall Score - Big and Prominent
        print(f"\n  ╔═══════════════════════════════════════════════════════════════╗")
        print(f"  ║                                                               ║")
        print(f"  ║  OVERALL CONFIDENCE SCORE:  {self.overall_score}/10  ({self.overall_grade})                         ║")
        print(f"  ║  PROFICIENCY: {self.overall_proficiency:<49} ║")
        print(f"  ║                                                               ║")
        print(f"  ╚═══════════════════════════════════════════════════════════════╝")
        
        # Speech section
        if self.speech_report:
            print(f"\n  ▸ Speech Performance")
            print(f"    [Voice quality, pace, and vocal delivery metrics]")
        
        # Body language section
        if self.body_language_report:
            print(f"\n  ▸ Body Language & Non-Verbal Communication")
            print(f"    [Posture, gestures, and visual presence metrics]")
        
        # Language & Content - Main Focus
        print(f"\n  ▸ LANGUAGE & CONTENT (Score: {self.language_and_content_report.overall_language_score}/10)")
        print(f"    {'─' * 70}")
        
        # Report grammar metrics
        print(f"\n    ① Grammar Accuracy (Score: {self.grammar_report.grammar_score}/10)")
        if self.grammar_report.error_count == 0:
            print(f"       ✓ Excellent! No grammar errors detected.")
        else:
            print(f"       • {self.grammar_report.error_count} error(s) found")
            if self.grammar_report.error_examples:
                print(f"       Examples:")
                for i, example in enumerate(self.grammar_report.error_examples[:2], 1):
                    print(f"         {i}. \"{example['original']}\" → \"{example['corrected']}\"")
        print(f"       Feedback: {self.grammar_report.feedback}")
        
        # Sentence structure
        print(f"\n    ② Sentence Structure (Score: {self.language_and_content_report.sentence_structure_score}/10)")
        print(f"       • Length: {self.sentence_structure_report.avg_sentence_length} words average")
        print(f"       • Variety: {self.sentence_structure_report.variety_level.title()} (σ={self.sentence_structure_report.sentence_length_std})")
        print(f"       • Pattern: {self.sentence_structure_report.sentence_length_category.title()}")
        print(f"       Feedback: {self.sentence_structure_report.feedback}")
        if self.sentence_structure_report.suggestions:
            print(f"       →", self.sentence_structure_report.suggestions[0])
        
        # Vocabulary
        print(f"\n    ③ Vocabulary & Word Choice (Score: {self.vocabulary_report.richness_score}/10)")
        print(f"       • Level: {self.vocabulary_report.vocabulary_level.title()}")
        print(f"       • Diversity: TTR = {self.vocabulary_report.type_token_ratio:.3f}")
        print(f"       • Word Count: {self.vocabulary_report.unique_words} unique / {self.vocabulary_report.total_words} total")
        print(f"       Feedback: {self.vocabulary_report.feedback}")
        if self.vocabulary_report.suggestions:
            print(f"       →", self.vocabulary_report.suggestions[0])
        
        # Fluency
        print(f"\n    ④ Fluency & Flow (Score: {self.fluency_report.fluency_score}/10)")
        print(f"       • Level: {self.fluency_report.fluency_level.title()}")
        print(f"       • Repetitions Detected: {self.fluency_report.repetition_count}")
        if self.fluency_report.repetition_examples:
            reps = ', '.join([f"{e['word']} ({e['count']}x)" for e in self.fluency_report.repetition_examples[:3]])
            print(f"       Notable repetitions: {reps}")
        print(f"       Feedback: {self.fluency_report.feedback}")
        if self.fluency_report.suggestions:
            print(f"       →", self.fluency_report.suggestions[0])
        
        # Strengths
        print(f"\n  ▸ KEY STRENGTHS")
        print(f"    {'─' * 70}")
        if self.language_and_content_report.strengths:
            for i, strength in enumerate(self.language_and_content_report.strengths, 1):
                print(f"    ✓ {strength}")
        else:
            print(f"    [Continue working on all areas]")
        
        # Areas for improvement
        print(f"\n  ▸ AREAS FOR IMPROVEMENT")
        print(f"    {'─' * 70}")
        if self.language_and_content_report.areas_for_improvement:
            for i, area in enumerate(self.language_and_content_report.areas_for_improvement, 1):
                print(f"    • {area}")
        else:
            print(f"    [Great work! Maintain your current performance]")
        
        # Top recommendations
        print(f"\n  ▸ TOP RECOMMENDED ACTIONS (Prioritized)")
        print(f"    {'─' * 70}")
        if self.language_and_content_report.top_recommendations:
            for i, rec in enumerate(self.language_and_content_report.top_recommendations, 1):
                print(f"    {i}. {rec}")
        else:
            print(f"    You're doing great! Keep refining your presentation skills.")
        
        # Closing coach-style message
        print(f"\n  ▸ COACH'S SUMMARY")
        print(f"    {'─' * 70}")
        if self.overall_score >= 8.0:
            print(f"    Excellent work! Your presentation demonstrates strong command of the")
            print(f"    material and clear communication. You're well-prepared and confident.")
            print(f"    Focus on fine-tuning the details for an even more polished delivery.")
        elif self.overall_score >= 6.0:
            print(f"    Good effort! You have a solid foundation. By addressing the areas")
            print(f"    highlighted above, your next presentation will be significantly stronger.")
            print(f"    Focus on your top recommended actions for the most impact.")
        else:
            print(f"    Keep practicing! You have room to grow. Focus on the recommended actions")
            print(f"    in order to build your presentation skills. Each practice session will")
            print(f"    help you improve. You've got this!")
        
        print(f"\n{'═' * 75}\n")


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
    transcript = None
    input_path = Path(args.transcript)
    
    if input_path.exists():
        # Check if it's a video/audio file (not a text transcript)
        audio_video_extensions = {'.mp4', '.mp3', '.wav', '.m4a', '.flac', '.aac', '.mov', '.avi'}
        if input_path.suffix.lower() in audio_video_extensions:
            print(f"\n✗ Error: '{args.transcript}' is a media file, not a transcript.")
            print(f"  Please extract the transcript first using speech_analyzer.py")
            print(f"  Then run: python combined_analyzer.py --transcript transcript.txt --output report.json")
            sys.exit(1)
        
        # Try to read as text transcript
        try:
            with open(args.transcript, "r", encoding="utf-8") as f:
                transcript = f.read()
        except UnicodeDecodeError:
            print(f"\n✗ Error: Could not read '{args.transcript}' as text.")
            print(f"  Make sure the file is a valid UTF-8 encoded text transcript.")
            sys.exit(1)
    else:
        # Treat input as raw text
        transcript = args.transcript
    
    if not transcript or not transcript.strip():
        print("\n✗ Error: Transcript is empty.")
        sys.exit(1)
    
    
    # Run analysis
    analyzer = SpeechAndBodyLanguageAnalyzer(transcript=transcript)
    
    # Run complete analysis (grammar, sentence structure, vocabulary, fluency)
    report = analyzer.run_analysis()
    
    # Output results
    if args.json or args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Report saved to: {args.output}")
    
    # Print professional summary to console
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
        analyzer.analyze_vocabulary(verbose=True)
        report = analyzer.generate_combined_report()
        analyzer.print_summary()
    """
    
    main()