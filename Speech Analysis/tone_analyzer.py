
#Prompt: look at this python file and comment on it.

"""

tone_analyzer.py
-------------------------------------------------------------------------------
LLM-powered tone-content appropriateness analyzer.
 
Takes the output of speech_analyzer.py (SpeechReport) and asks an LLM to:
  1. Identify the topic / context of the speech from the transcript.
  2. Assess whether the detected vocal emotion + prosody fits that context.
  3. Flag specific mismatches (e.g. cheerful delivery during a eulogy).
  4. Return prioritised, actionable coaching tips.
 
Uses the Pollinations AI text API — no API key required.
  Endpoint: https://text.pollinations.ai/
  Docs:     https://github.com/pollinations/pollinations
 
Usage (standalone):
    python tone_analyzer.py report.json          # pass a saved JSON report
    python tone_analyzer.py --demo               # run with a synthetic demo report
 
Usage (programmatic):
    from speech_analyzer import analyze
    from tone_analyzer import analyze_tone, print_tone_report
 
    speech_report = analyze("talk.wav")
    tone_report   = analyze_tone(speech_report)
    print_tone_report(tone_report)
-------------------------------------------------------------------------------
"""
 
import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, asdict, field
from typing import Optional
 
 
# Pollinations API config
POLLINATIONS_URL = "https://text.pollinations.ai/"
 
DEFAULT_MODEL = "openai"
 
REQUEST_TIMEOUT = 60    
MAX_RETRIES     = 3
RETRY_DELAY     = 4     
 
#output data class 
@dataclass
class ToneMismatch:
    severity: str           # "high" | "medium" | "low"
    observed_tone: str      # "cheerful / high energy"
    expected_tone: str      # "solemn / subdued"
    reason: str             # one-sentence explanation
    moment: str             # where in the speech ("throughout" or "opening")
 
 
@dataclass
class ToneReport:
    detected_topic: str             # LLM's inference of speech subject
    detected_context: str           # "memorial service", "sales pitch"
    overall_tone_fit: str           # "appropriate" | "partially appropriate" | "inappropriate"
    tone_fit_score: float           # 0.0 (completely wrong) ----> 1.0 (perfect match)
    mismatches: list                # list of ToneMismatch dicts
    coaching_tips: list             # ordered list of tip strings (most impactful first)
    model_used: str
    raw_response: str               # full LLM output for debugging
 
 
# Prompt builder

#TODO:add additional language prompts
SYSTEM_PROMPT = """You are an expert public speaking coach with deep knowledge of 
rhetoric, emotional communication, and audience psychology.
 
You will receive:
- A transcript (or transcript preview) of a speech
- Detected prosodic/emotional markers from audio analysis
 
Your job is to assess whether the speaker's VOCAL TONE matches the CONTENT AND CONTEXT
of their speech, then give prioritised coaching tips.
 
Rules:
- Be specific and concrete. Reference actual content from the transcript.
- Mismatches vary in severity: a eulogy delivered cheerfully is HIGH severity;
  a motivational talk with slightly flat energy is LOW severity.
- Tips should be actionable (what to DO differently), not just observations.
- Never invent content not present in the transcript or markers.
- Respond ONLY with valid JSON. No preamble, no markdown fences.
 
JSON schema:
{
  "detected_topic": "string — 1 sentence describing what the speech is about",
  "detected_context": "string — inferred occasion/setting (e.g. 'corporate presentation', 'eulogy', 'TED-style talk')",
  "overall_tone_fit": "appropriate | partially appropriate | inappropriate",
  "tone_fit_score": float between 0.0 and 1.0,
  "mismatches": [
    {
      "severity": "high | medium | low",
      "observed_tone": "string",
      "expected_tone": "string",
      "reason": "string",
      "moment": "string"
    }
  ],
  "coaching_tips": [
    "string — specific, actionable tip"
  ]
}"""
 
 
def build_user_message(speech_report: dict) -> str:
    """
    Constructs the user message from a SpeechReport dict
    Keeps it concise
    """
    #pull key values - handle both dataclass and plain dict inputs
    transcript  = speech_report.get("transcript_preview", "")
    overall     = speech_report.get("overall", 0)
    wpm_raw     = speech_report.get("wpm", {}).get("raw", 0)
    filler      = speech_report.get("filler_rate", {}).get("raw", 0)
    pitch_score = speech_report.get("pitch_variation", {}).get("score", 0)
    pitch_raw   = speech_report.get("pitch_variation", {}).get("raw", 0)
    energy_score= speech_report.get("energy_variation", {}).get("score", 0)
    pause_raw   = speech_report.get("pause_ratio", {}).get("raw", 0)
    emo_label   = speech_report.get("vocal_emotion", {}).get("label", "Unknown")
    emo_score   = speech_report.get("vocal_emotion", {}).get("score", 0)
    grade       = speech_report.get("grade", "?")
 
    # Map scores to plain-English descriptors for the LLM
    def score_to_descriptor(score: float) -> str:
        if score >= 0.5:  return "strong / above average"
        if score >= 0.1:  return "moderate / average"
        if score >= -0.2: return "slightly below average"
        if score >= -0.5: return "weak / below average"
        return "very poor"
 
    pitch_desc  = score_to_descriptor(pitch_score)
    energy_desc = score_to_descriptor(energy_score)
 
    return f"""TRANSCRIPT (first 300 chars):
\"\"\"{transcript}\"\"\"
 
DETECTED VOCAL MARKERS:
- Speaking rate    : {wpm_raw:.0f} wpm
- Filler word rate : {filler:.1f}% of words
- Pitch variation  : {pitch_raw:.1f} Hz std-dev  ({pitch_desc})
- Energy variation : {energy_desc}
- Pause ratio      : {pause_raw:.1f}% of total duration
- Vocal emotion    : {emo_label}  (score: {emo_score:+.2f})
- Overall grade    : {grade}  ({overall:+.2f})
 
Analyze the tone-content appropriateness and return JSON as specified."""
 
 
# ──────────────────────────────────────────────────────────────────────────────
# Pollinations API call
# ──────────────────────────────────────────────────────────────────────────────
 
def call_pollinations(
    user_message: str,
    model: str = DEFAULT_MODEL,
    seed: int = 42,
) -> str:
    """
    Calls the Pollinations text API and returns the raw response string.
    Retries on transient failures.
    """
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "user", "content": user_message}
        ],
        "system": SYSTEM_PROMPT,
        "seed": seed,
        "json_mode": True,      # instructs the model to return valid JSON
        "temperature": 0.3,     # low temperature for consistent structured output
    }).encode("utf-8")
 
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
 
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            req = urllib.request.Request(
                POLLINATIONS_URL,
                data=payload,
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                raw = resp.read().decode("utf-8")
                return raw
 
        except urllib.error.HTTPError as e:
            last_error = f"HTTP {e.code}: {e.reason}"
            print(f"  [Pollinations] Attempt {attempt}/{MAX_RETRIES} failed: {last_error}", flush=True)
        except urllib.error.URLError as e:
            last_error = f"URLError: {e.reason}"
            print(f"  [Pollinations] Attempt {attempt}/{MAX_RETRIES} failed: {last_error}", flush=True)
        except TimeoutError:
            last_error = "Request timed out"
            print(f"  [Pollinations] Attempt {attempt}/{MAX_RETRIES} timed out.", flush=True)
 
        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY)
 
    raise RuntimeError(f"Pollinations API failed after {MAX_RETRIES} attempts. Last error: {last_error}")
 
 
# Response parser
 
def parse_llm_response(raw: str, model: str) -> ToneReport:
    """
    Parses the Pollinations response into a ToneReport.
    Handles both direct JSON responses and responses wrapped in a 'text' field.
    """
    # Pollinations wraps responses differently depending on the model.
    text_content = raw.strip()
 
    try:
        outer = json.loads(text_content)
        # Some models return {"text": "...json...", "model": "..."}
        if isinstance(outer, dict) and "text" in outer and isinstance(outer["text"], str):
            text_content = outer["text"].strip()
        elif isinstance(outer, dict) and "detected_topic" in outer:
            # Already the structured response
            data = outer
            return _build_report(data, model, raw)
    except json.JSONDecodeError:
        pass
 
    # Strip accidental markdown fences
    if text_content.startswith("```"):
        lines = text_content.split("\n")
        text_content = "\n".join(
            line for line in lines
            if not line.strip().startswith("```")
        ).strip()
 
    try:
        data = json.loads(text_content)
    except json.JSONDecodeError as e:
        # Last-resort: extract JSON substring
        start = text_content.find("{")
        end   = text_content.rfind("}") + 1
        if start != -1 and end > start:
            try:
                data = json.loads(text_content[start:end])
            except json.JSONDecodeError:
                raise ValueError(
                    f"Could not parse LLM response as JSON.\n"
                    f"Parse error: {e}\n"
                    f"Raw content (first 500 chars): {text_content[:500]}"
                )
        else:
            raise ValueError(
                f"No JSON object found in LLM response.\n"
                f"Raw content (first 500 chars): {text_content[:500]}"
            )
 
    return _build_report(data, model, raw)
 
 
def _build_report(data: dict, model: str, raw: str) -> ToneReport:
    mismatches = [
        ToneMismatch(
            severity     = m.get("severity", "low"),
            observed_tone= m.get("observed_tone", ""),
            expected_tone= m.get("expected_tone", ""),
            reason       = m.get("reason", ""),
            moment       = m.get("moment", ""),
        )
        for m in data.get("mismatches", [])
    ]
 
    return ToneReport(
        detected_topic   = data.get("detected_topic", "Unknown"),
        detected_context = data.get("detected_context", "Unknown"),
        overall_tone_fit = data.get("overall_tone_fit", "unknown"),
        tone_fit_score   = float(data.get("tone_fit_score", 0.5)),
        mismatches       = [asdict(m) for m in mismatches],
        coaching_tips    = data.get("coaching_tips", []),
        model_used       = model,
        raw_response     = raw,
    )
 
 
#main public function
def analyze_tone(
    speech_report,                      # SpeechReport dataclass OR dict
    model: str = DEFAULT_MODEL,
    verbose: bool = True,
) -> ToneReport:
    """
    Primary entry point. Pass the SpeechReport from speech_analyzer.analyze().
 
    Args:
        speech_report : SpeechReport dataclass or equivalent dict.
        model         : Pollinations model to use (default: "openai").
        verbose       : Print progress messages.
 
    Returns:
        ToneReport dataclass.
    """
    # Accept both dataclass and dict
    if hasattr(speech_report, "__dataclass_fields__"):
        report_dict = asdict(speech_report)
    elif isinstance(speech_report, dict):
        report_dict = speech_report
    else:
        raise TypeError(f"speech_report must be a SpeechReport dataclass or dict, got {type(speech_report)}")
 
    if verbose:
        print("\n── Tone-content analysis ──")
        print(f"  Model : {model}", flush=True)
 
    user_msg = build_user_message(report_dict)
 
    if verbose:
        print("  Calling Pollinations AI…", flush=True)
 
    raw = call_pollinations(user_msg, model=model)
 
    if verbose:
        print("  Parsing response…", flush=True)
 
    tone_report = parse_llm_response(raw, model)
 
    return tone_report
 
 
# Pretty printer
SEVERITY_ICON = {"high": "[!!]", "medium": "[! ]", "low": "[ .]"}
FIT_ICON      = {"appropriate": "[OK]", "partially appropriate": "[~~]", "inappropriate": "[XX]"}
 
 
def print_tone_report(report: ToneReport):
    sep = "─" * 62
 
    print(f"\n{'═' * 62}")
    print(f"  TONE-CONTENT ANALYSIS")
    print(f"{'═' * 62}")
    print(f"  Topic    : {report.detected_topic}")
    print(f"  Context  : {report.detected_context}")
    fit_icon = FIT_ICON.get(report.overall_tone_fit, "[??]")
    print(f"  Tone fit : {fit_icon} {report.overall_tone_fit.upper()}  ({report.tone_fit_score:.2f} / 1.00)")
    print(f"  Model    : {report.model_used}")
 
    if report.mismatches:
        print(f"\n{sep}")
        print(f"  Tone mismatches ({len(report.mismatches)} found)")
        for m in report.mismatches:
            icon = SEVERITY_ICON.get(m["severity"], "[ ]")
            print(f"\n  {icon} {m['severity'].upper()} — {m['moment']}")
            print(f"      Observed : {m['observed_tone']}")
            print(f"      Expected : {m['expected_tone']}")
            print(f"      Why      : {m['reason']}")
    else:
        print(f"\n{sep}")
        print("  No significant tone mismatches detected.")
 
    if report.coaching_tips:
        print(f"\n{sep}")
        print(f"  Coaching tips (most impactful first)")
        for i, tip in enumerate(report.coaching_tips, 1):
            # Word-wrap at 56 chars
            words = tip.split()
            lines, line = [], []
            for w in words:
                if sum(len(x) + 1 for x in line) + len(w) > 56:
                    lines.append(" ".join(line))
                    line = [w]
                else:
                    line.append(w)
            if line:
                lines.append(" ".join(line))
            print(f"\n  {i}. {lines[0]}")
            for l in lines[1:]:
                print(f"     {l}")
 
    print(f"\n{'═' * 62}\n")
 
 
#demo / synthetic report for testing without running the full pipeline
DEMO_REPORT = {
    "overall": -0.21,
    "grade": "D",
    "transcript_preview": (
        "Today we gather to remember our beloved colleague, John, who passed away "
        "last Tuesday after a brief illness. I'm super excited to be here and share "
        "some amazing memories with you all! John was just the best, honestly. "
        "He really crushed it in the marketing department every single quarter."
    ),
    "wpm":             {"raw": 152, "score": 0.4},
    "filler_rate":     {"raw": 4.2, "score": -0.3, "label": "Filler words"},
    "pitch_variation": {"raw": 38.5, "score": 0.6, "label": "Pitch variation"},
    "energy_variation":{"raw": 0.051, "score": 0.3, "label": "Energy variation"},
    "pause_ratio":     {"raw": 8.1, "score": -0.2, "label": "Pause ratio"},
    "vocal_emotion":   {"raw": 0.82, "score": 0.8, "label": "Vocal emotion (Positive / engaged)"},
    "segments": [],
}
 
 
# CLI
def main():
    parser = argparse.ArgumentParser(
        description="LLM tone-content appropriateness analyzer for public speaking.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "report",
        nargs="?",
        help="Path to a JSON file produced by speech_analyzer.py (--json flag)",
    )
    group.add_argument(
        "--demo",
        action="store_true",
        help="Run with a built-in synthetic demo report (eulogy with mismatched happy tone)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        choices=["openai", "openai-large", "mistral"],
        help=f"Pollinations model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    args = parser.parse_args()
 
    if args.demo:
        report_dict = DEMO_REPORT
        print("  [demo mode] Using synthetic eulogy report with mismatched cheerful tone.")
    else:
        try:
            with open(args.report, "r") as f:
                report_dict = json.load(f)
        except FileNotFoundError:
            print(f"Error: file not found: {args.report}", file=sys.stderr)
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"Error: invalid JSON in {args.report}: {e}", file=sys.stderr)
            sys.exit(1)
 
    tone_report = analyze_tone(report_dict, model=args.model)
 
    if args.json:
        print(json.dumps(asdict(tone_report), indent=2))
    else:
        print_tone_report(tone_report)
 
 
if __name__ == "__main__":
    main()
 
