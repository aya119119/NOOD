"""
# NOOD - AI Public Speaking Coach
# Color: #0F0866

A clean, modern Streamlit app for analyzing presentation videos with instant feedback.
Inspired by Odyssers minimal and elegant design.

Installation:
    pip install -r requirements.txt

Run locally:
    streamlit run streamlit_app.py

Dependencies:
    - combined_analyzer.py (main analysis engine)
    - Speech Analysis/speech_analyzer.py (ASR model)
"""

import streamlit as st
import tempfile
import json
import sys
from pathlib import Path

# ============================================================================
# Streamlit Config (MUST be first)
# ============================================================================

st.set_page_config(
    page_title="NOOD - AI Public Speaking Coach",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Custom CSS for elegant design
st.markdown("""
<style>
/* Root styles */
:root {
    --primary-color: #0F0866;
    --white: #FFFFFF;
    --light-gray: #F8F9FA;
    --text-dark: #1A1A1A;
    --text-light: #666666;
    --border-light: #E0E0E0;
}

/* Remove default padding and margins */
body {
    margin: 0;
    padding: 0;
}

/* Main container */
.main {
    padding: 0;
}

/* Custom container styles */
.header-container {
    background-color: var(--white);
    padding: 1.5rem 2rem;
    border-bottom: 1px solid var(--border-light);
    margin-bottom: 3rem;
}

.hero-container {
    padding: 4rem 2rem;
    text-align: center;
    background: linear-gradient(135deg, #FFFFFF 0%, #F8F9FA 100%);
    margin-bottom: 3rem;
}

.section-title {
    font-size: 2.5rem;
    font-weight: 700;
    color: var(--text-dark);
    margin-bottom: 1rem;
    letter-spacing: -0.5px;
}

.section-subtitle {
    font-size: 1.15rem;
    color: var(--text-light);
    margin-bottom: 2.5rem;
    line-height: 1.6;
}

.feature-card {
    background: var(--white);
    border: 1px solid var(--border-light);
    border-radius: 12px;
    padding: 2rem;
    text-align: center;
    transition: all 0.3s ease;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.03);
}

.feature-card:hover {
    border-color: var(--primary-color);
    box-shadow: 0 8px 24px rgba(15, 8, 102, 0.08);
    transform: translateY(-2px);
}

.feature-icon {
    font-size: 2.5rem;
    margin-bottom: 1rem;
}

.feature-title {
    font-size: 1.3rem;
    font-weight: 600;
    color: var(--text-dark);
    margin-bottom: 0.75rem;
}

.feature-description {
    color: var(--text-light);
    font-size: 0.95rem;
    line-height: 1.6;
}

.steps-container {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin: 3rem 0;
    gap: 2rem;
}

.step-item {
    flex: 1;
    text-align: center;
}

.step-number {
    background-color: var(--primary-color);
    color: var(--white);
    width: 50px;
    height: 50px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.5rem;
    font-weight: 700;
    margin: 0 auto 1.5rem;
}

.step-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--text-dark);
    margin-bottom: 0.5rem;
}

.step-description {
    color: var(--text-light);
    font-size: 0.9rem;
}

.upload-area {
    border: 2px dashed var(--primary-color);
    border-radius: 12px;
    padding: 3rem;
    text-align: center;
    background-color: rgba(15, 8, 102, 0.02);
    margin: 2rem 0;
}

.report-card {
    background: var(--white);
    border: 1px solid var(--border-light);
    border-radius: 12px;
    padding: 2rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.03);
}

.confidence-score {
    text-align: center;
    padding: 2rem;
    background: linear-gradient(135deg, var(--primary-color) 0%, #2D1B69 100%);
    color: var(--white);
    border-radius: 12px;
    margin-bottom: 2rem;
}

.score-number {
    font-size: 3.5rem;
    font-weight: 700;
    margin: 0.5rem 0;
}

.score-label {
    font-size: 0.95rem;
    opacity: 0.9;
}

.metric-row {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1.5rem;
    margin-bottom: 2rem;
}

.metric-box {
    background: var(--light-gray);
    padding: 1.5rem;
    border-radius: 8px;
    border-left: 4px solid var(--primary-color);
}

.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--primary-color);
    margin-bottom: 0.5rem;
}

.metric-label {
    font-size: 0.85rem;
    color: var(--text-light);
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.tip-item {
    background-color: var(--light-gray);
    border-left: 4px solid var(--primary-color);
    padding: 1rem 1.5rem;
    margin-bottom: 1rem;
    border-radius: 4px;
}

.tip-title {
    font-weight: 600;
    color: var(--text-dark);
    margin-bottom: 0.5rem;
}

.tip-text {
    color: var(--text-light);
    font-size: 0.9rem;
}

/* Button styles */
.stButton > button {
    background-color: var(--primary-color);
    color: var(--white);
    border: none;
    border-radius: 8px;
    padding: 0.75rem 2rem;
    font-weight: 600;
    font-size: 1rem;
    cursor: pointer;
    transition: all 0.3s ease;
    width: 100%;
}

.stButton > button:hover {
    background-color: #1A1272;
    box-shadow: 0 8px 24px rgba(15, 8, 102, 0.2);
    transform: translateY(-2px);
}

/* Text input and file uploader */
.stTextInput > div > div > input,
.stFileUploader > div > div > div > label {
    border-radius: 8px;
    border: 1px solid var(--border-light);
}

/* Success message */
.success-banner {
    background-color: #F0F8F5;
    border-left: 4px solid #10B981;
    padding: 1rem 1.5rem;
    border-radius: 4px;
    margin-bottom: 1.5rem;
}

.nav-link {
    color: var(--text-dark);
    text-decoration: none;
    font-weight: 500;
    font-size: 0.95rem;
    margin: 0 1.5rem;
    cursor: pointer;
    transition: color 0.3s ease;
}

.nav-link:hover {
    color: var(--primary-color);
}

.sign-in-btn {
    color: var(--primary-color);
    text-decoration: none;
    font-weight: 500;
    margin: 0 1rem;
    cursor: pointer;
}

.try-demo-btn {
    background-color: var(--primary-color);
    color: var(--white);
    padding: 0.6rem 1.5rem;
    border-radius: 6px;
    text-decoration: none;
    font-weight: 600;
    cursor: pointer;
    border: none;
}

.try-demo-btn:hover {
    background-color: #1A1272;
}

/* Footer */
.footer {
    text-align: center;
    padding: 2rem;
    color: var(--text-light);
    font-size: 0.9rem;
    margin-top: 4rem;
    border-top: 1px solid var(--border-light);
}

/* Responsive */
@media (max-width: 768px) {
    .section-title {
        font-size: 1.8rem;
    }
    
    .score-number {
        font-size: 2.5rem;
    }
    
    .steps-container {
        flex-direction: column;
    }
}
</style>
""", unsafe_allow_html=True)


# ============================================================================
# Imports
# ============================================================================

# Add project modules to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "Streamlit + Whisper"))
sys.path.insert(0, str(Path(__file__).parent / "Speech Analysis"))

try:
    from combined_analyzer import SpeechAndBodyLanguageAnalyzer
except ImportError as e:
    st.error(f"❌ Import Error: {str(e)}")
    st.markdown(f"""
    **Import Error:** {str(e)}
    
    **To fix:**
    1. Run `pip install -r requirements.txt`
    2. Make sure combined_analyzer.py exists in the Streamlit + Whisper folder
    """)
    st.stop()


# ============================================================================
# Model Caching
# ============================================================================

@st.cache_resource(show_spinner=False)
def get_asr_model():
    """Load ASR model once and cache it."""
    try:
        import speech_analyzer
        return speech_analyzer.load_asr()
    except Exception as e:
        st.error(f"Failed to load speech recognition model: {str(e)}")
        return None


# ============================================================================
# Helper Functions
# ============================================================================

def extract_transcript_from_video(video_path: str) -> str:
    """Extract transcript from video using cached ASR."""
    try:
        asr = get_asr_model()
        if asr is None:
            return None
        
        transcript = asr.transcribe_file(video_path)
        
        # Handle different return types
        if isinstance(transcript, (list, tuple)):
            transcript = transcript[0] if transcript else ""
        else:
            transcript = str(transcript)
        
        return transcript if transcript and transcript.strip() else None
    except Exception as e:
        st.error(f"Error extracting transcript: {str(e)}")
        return None


def run_analysis(transcript: str):
    """Run complete analysis on transcript."""
    try:
        analyzer = SpeechAndBodyLanguageAnalyzer(transcript=transcript)
        report = analyzer.run_analysis()
        return report, analyzer
    except Exception as e:
        st.error(f"Error during analysis: {str(e)}")
        return None, None


def format_score(score: float) -> str:
    """Format score for display."""
    if score >= 8:
        emoji = "🌟"
    elif score >= 6:
        emoji = "✨"
    elif score >= 4:
        emoji = "💪"
    else:
        emoji = "📈"
    return f"{emoji} {score:.1f}/10"


# ============================================================================
# Main UI
# ============================================================================

# Header (render once)
st.markdown("<h1 style='text-align: center;'>🎤 NOOD - Public Speaking Coach</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'><b>Upload your presentation video and get instant feedback</b></p>", unsafe_allow_html=True)
st.markdown("---")

# File uploader with unique key
uploaded_file = st.file_uploader(
    "Upload your presentation video",
    type=["mp4", "mov", "avi", "mkv", "flv", "wmv", "webm"],
    help="Supported formats: MP4, MOV, AVI, MKV, FLV, WMV, WebM",
    key="video_uploader_nood",
)


if uploaded_file is not None:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        video_path = tmp_file.name
    
    # Show video
    st.markdown("### 📹 Your Video")
    st.video(uploaded_file)
    
    # Extract transcript
    st.markdown("---")
    st.markdown("### Processing...")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("🎙️ Loading speech recognition model...")
    progress_bar.progress(20)
    
    status_text.text("🎙️ Extracting transcript from video...")
    progress_bar.progress(50)
    
    transcript = extract_transcript_from_video(video_path)
    progress_bar.progress(100)
    status_text.empty()
    progress_bar.empty()
    
    if transcript:
        st.success("✓ Transcript extracted!")
        
        # Show transcript
        with st.expander("📄 View Full Transcript", expanded=False):
            st.text_area(
                "Transcript:",
                value=transcript,
                height=150,
                disabled=True,
                key="transcript_text",
            )
        
        # Run analysis
        st.markdown("---")
        st.markdown("### Analyzing...")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("🔍 Analyzing Grammar, Structure, Vocabulary, and Fluency...")
        progress_bar.progress(50)
        
        report, analyzer = run_analysis(transcript)
        
        progress_bar.progress(100)
        status_text.empty()
        progress_bar.empty()
        
        if report and analyzer:
            st.success("✓ Analysis complete!")
            st.markdown("---")
            
            # SECTION 1: Overall Score
            st.markdown("## 📊 Overall Confidence Score")
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown(f"<h2 style='text-align: center; color: #2E86AB;'>{format_score(analyzer.overall_score)}</h2>", unsafe_allow_html=True)
            with col2:
                st.markdown(f"**Grade:** {analyzer.overall_grade}")
                st.markdown(f"**Proficiency:** {analyzer.overall_proficiency}")
            
            st.markdown("---")
            
            # SECTION 2: Speech
            st.markdown("## 🗣️ Speech Observations")
            if analyzer.speech_report and isinstance(analyzer.speech_report, dict):
                speech = analyzer.speech_report
                col1, col2, col3 = st.columns(3)
                with col1:
                    if "wpm" in speech:
                        st.metric("Speaking Rate", f"{speech['wpm'].get('raw', 0):.0f} WPM")
                with col2:
                    if "filler_rate" in speech:
                        st.metric("Filler Words", f"{speech['filler_rate'].get('raw', 0):.1f}%")
                with col3:
                    if "pause_ratio" in speech:
                        st.metric("Pause Ratio", f"{speech['pause_ratio'].get('raw', 0):.1f}%")
            else:
                st.info("No speech data available.")
            st.markdown("---")
            
            # SECTION 3: Language & Content
            st.markdown("## 📚 Language & Content Feedback")
            if analyzer.language_and_content_report:
                lac = analyzer.language_and_content_report
                
                # Scores
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Grammar", f"{lac.grammar_score:.1f}/10")
                with col2:
                    st.metric("Structure", f"{lac.sentence_structure_score:.1f}/10")
                with col3:
                    st.metric("Vocabulary", f"{lac.vocabulary_score:.1f}/10")
                with col4:
                    st.metric("Fluency", f"{lac.fluency_score:.1f}/10")
                
                # Grammar details
                st.markdown("### Grammar & Accuracy")
                if analyzer.grammar_report:
                    if analyzer.grammar_report.error_count == 0:
                        st.success("✓ No grammar errors!")
                    else:
                        st.warning(f"Found {analyzer.grammar_report.error_count} error(s)")
                        if analyzer.grammar_report.error_examples:
                            for example in analyzer.grammar_report.error_examples[:3]:
                                st.caption(f"→ \"{example['original']}\" should be \"{example['corrected']}\"")
                
                # Strengths & Improvements
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### 💪 Strengths")
                    if lac.strengths:
                        for s in lac.strengths:
                            st.success(f"✓ {s}")
                    else:
                        st.info("Keep working on your skills!")
                
                with col2:
                    st.markdown("### 📈 Areas to Improve")
                    if lac.areas_for_improvement:
                        for a in lac.areas_for_improvement:
                            st.warning(f"• {a}")
                    else:
                        st.success("Great job!")
            
            st.markdown("---")
            
            # SECTION 4: Recommendations
            st.markdown("## 🎯 Top Recommended Actions")
            if analyzer.language_and_content_report and analyzer.language_and_content_report.top_recommendations:
                for i, rec in enumerate(analyzer.language_and_content_report.top_recommendations, 1):
                    st.markdown(f"**{i}. {rec}**")
            else:
                st.info("You're doing great!")
            
            st.markdown("---")
            
            # SECTION 5: Coach Message
            st.markdown("## 💬 Coach's Message")
            if analyzer.overall_score >= 8.0:
                st.success("**Excellent work!** You're well-prepared and confident. Focus on fine-tuning details.")
            elif analyzer.overall_score >= 6.0:
                st.info("**Good effort!** You have a solid foundation. Address the areas above for next time.")
            else:
                st.warning("**Keep practicing!** Each session helps you improve. You've got this! 💪")
            
            st.markdown("---")
            
            # Download report
            st.markdown("### 📥 Download Full Report")
            report_json = json.dumps(report, indent=2, ensure_ascii=False)
            st.download_button(
                label="📊 Download JSON Report",
                data=report_json,
                file_name="presentation_analysis_report.json",
                mime="application/json",
                key="download_report",
            )
    
    else:
        st.error("⚠️ Could not extract transcript.")
        st.markdown("**Make sure:** Video has clear audio, format is supported, audio quality is good")

st.markdown("---")
st.markdown("""
**⏱️ Processing Time:** 1-3 minutes (first run: 3-5 min for model download)

**📝 Note:** Optimized for presentations in English.
""")

st.markdown("""
---
<div style='text-align: center; color: gray; font-size: 0.8em;'>
Made with 💙 by NOOD - Your AI Public Speaking Coach
</div>
""", unsafe_allow_html=True)
