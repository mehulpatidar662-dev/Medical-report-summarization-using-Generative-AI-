import streamlit as st
import os
import re
import nest_asyncio
from dotenv import load_dotenv

load_dotenv()
from observability import logger, start_metrics_server
from preprocessor import Preprocessor
from ner import NEREngine
from summarizer import SummarizationEngine
from evaluator import Evaluator

# ── Config ────────────────────────────────────────────────
APP_TITLE = "LLM Medical Report Summarizer"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ── Page setup ────────────────────────────────────────────
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Start metrics server safely
start_metrics_server()

# Patch asyncio for Streamlit
nest_asyncio.apply()

# ── Universal High-Contrast Medical Theme ──────────────────
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    /* Layout Max Width */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3.5rem;
        max-width: 1250px;
    }
    
    /* Hero Header Banner */
    .hero-banner {
        background: linear-gradient(135deg, #07172c 0%, #172e61 50%, #0369a1 100%);
        border: 1px solid rgba(56, 189, 248, 0.35);
        border-radius: 18px;
        padding: 30px 36px;
        color: #ffffff;
        margin-bottom: 24px;
        box-shadow: 0 12px 30px -8px rgba(2, 132, 199, 0.35);
        position: relative;
        overflow: hidden;
    }
    .hero-banner::after {
        content: "";
        position: absolute;
        top: -50px;
        right: -50px;
        width: 220px;
        height: 220px;
        background: radial-gradient(circle, rgba(56, 189, 248, 0.22) 0%, rgba(2, 132, 199, 0) 70%);
        border-radius: 50%;
        pointer-events: none;
    }
    .hero-title {
        color: #ffffff !important;
        font-size: 2.2rem !important;
        font-weight: 800 !important;
        letter-spacing: -0.6px;
        margin: 0 !important;
        line-height: 1.2 !important;
    }
    .hero-subtitle {
        color: #cbd5e1 !important;
        font-size: 1.05rem !important;
        font-weight: 400 !important;
        margin: 8px 0 0 0 !important;
        line-height: 1.5 !important;
    }
    
    /* Badge Pills */
    .badge-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(255, 255, 255, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.28);
        color: #ffffff !important;
        padding: 6px 14px;
        border-radius: 30px;
        font-size: 0.84rem;
        font-weight: 600;
        backdrop-filter: blur(8px);
    }
    
    /* Section Headings (High Contrast) */
    .section-title {
        font-size: 1.35rem !important;
        font-weight: 700 !important;
        color: #f8fafc !important;
        margin-bottom: 14px !important;
        display: flex !important;
        align-items: center !important;
        gap: 10px !important;
        letter-spacing: -0.3px !important;
    }
    
    /* Secondary Action Buttons (Sample Buttons) */
    div.stButton > button:not([kind="primary"]) {
        background-color: #1e293b !important;
        color: #f8fafc !important;
        border: 1px solid #334155 !important;
        border-radius: 10px !important;
        font-size: 0.94rem !important;
        font-weight: 600 !important;
        padding: 0.55rem 1rem !important;
        white-space: nowrap !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    }
    div.stButton > button:not([kind="primary"]):hover {
        background-color: #334155 !important;
        border-color: #0284c7 !important;
        color: #38bdf8 !important;
        transform: translateY(-1px);
    }
    
    /* Primary Action Button (Generate Summary) */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 1.6rem !important;
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.2px !important;
        box-shadow: 0 4px 16px rgba(2, 132, 199, 0.45) !important;
    }
    div.stButton > button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(2, 132, 199, 0.6) !important;
    }

    /* Metric Cards (High Contrast Dark Slate) */
    div[data-testid="stMetric"] {
        background: #1e293b !important;
        border: 1px solid #334155 !important;
        border-radius: 14px !important;
        padding: 16px 20px !important;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.2) !important;
        border-top: 4px solid #0284c7 !important;
    }
    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] [data-testid="stMetricLabel"],
    div[data-testid="stMetric"] [data-testid="stMetricLabel"] *,
    div[data-testid="stMetricLabel"],
    div[data-testid="stMetricLabel"] p,
    div[data-testid="stMetricLabel"] span {
        color: #94a3b8 !important;
        font-size: 0.88rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.6px !important;
        opacity: 1 !important;
    }
    div[data-testid="stMetricValue"], div[data-testid="stMetricValue"] * {
        color: #38bdf8 !important;
        font-size: 1.85rem !important;
        font-weight: 800 !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        opacity: 1 !important;
    }
    
    /* Tab Styling (High Contrast) */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #1e293b !important;
        padding: 6px 8px;
        border-radius: 12px;
        border: 1px solid #334155 !important;
    }
    .stTabs [data-baseweb="tab"] {
        height: 42px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.94rem;
        color: #94a3b8 !important;
        padding: 0 16px;
        background-color: transparent;
        border: none !important;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #0284c7 !important;
        color: #ffffff !important;
        box-shadow: 0 2px 8px rgba(2, 132, 199, 0.4) !important;
    }

    /* Patient Summary Container (High Contrast) */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #1e293b !important;
        border-radius: 14px !important;
        border: 1px solid #334155 !important;
        border-left: 6px solid #0284c7 !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25) !important;
        padding: 26px 30px !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] h3 {
        color: #38bdf8 !important;
        font-weight: 700 !important;
        font-size: 1.3rem !important;
        margin-top: 1.3rem !important;
        margin-bottom: 0.6rem !important;
        padding-bottom: 6px !important;
        border-bottom: 1px solid #334155 !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] strong {
        color: #ffffff !important;
        font-weight: 700 !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] p,
    div[data-testid="stVerticalBlockBorderWrapper"] li,
    div[data-testid="stVerticalBlockBorderWrapper"] div {
        color: #e2e8f0 !important;
        font-size: 1.05rem !important;
        line-height: 1.8 !important;
    }
    
    /* Biomedical Entity Badges */
    .entity-tag {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 7px 15px;
        border-radius: 20px;
        font-size: 0.88rem;
        font-weight: 600;
        margin: 4px 6px 4px 0;
        transition: transform 0.15s ease;
    }
    .entity-tag:hover {
        transform: scale(1.04);
    }
    .tag-disease {
        background: rgba(244, 63, 94, 0.18) !important;
        color: #fda4af !important;
        border: 1px solid rgba(244, 63, 94, 0.45) !important;
    }
    .tag-chemical {
        background: rgba(14, 165, 233, 0.18) !important;
        color: #7dd3fc !important;
        border: 1px solid rgba(14, 165, 233, 0.45) !important;
    }
    
    /* Sidebar enhancements & full contrast */
    [data-testid="stSidebar"] {
        background-color: #0f172a !important;
        border-right: 1px solid #1e293b !important;
    }
    [data-testid="stSidebar"] * {
        color: #f8fafc !important;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4 {
        color: #ffffff !important;
        font-weight: 700 !important;
    }
    [data-testid="stSidebar"] .stCaption, [data-testid="stSidebar"] p, [data-testid="stSidebar"] span {
        color: #94a3b8 !important;
    }
    [data-testid="stSidebar"] code {
        background-color: #1e293b !important;
        color: #38bdf8 !important;
        font-weight: 600 !important;
        padding: 3px 8px !important;
        border-radius: 6px !important;
        border: 1px solid #334155 !important;
    }
    [data-testid="stSidebar"] input {
        background-color: #1e293b !important;
        color: #f8fafc !important;
        border: 1.5px solid #334155 !important;
        border-radius: 8px !important;
    }
    [data-testid="stSidebar"] input:focus {
        border-color: #0284c7 !important;
    }
    
    /* Text Area Focus */
    .stTextArea textarea {
        border-radius: 12px !important;
        border: 1.5px solid #334155 !important;
        font-family: inherit !important;
        font-size: 0.95rem !important;
        background-color: #1e293b !important;
        color: #f8fafc !important;
    }
    .stTextArea textarea:focus {
        border-color: #0284c7 !important;
        box-shadow: 0 0 0 3px rgba(2, 132, 199, 0.25) !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Load models once ──────────────────────────────────────
@st.cache_resource(show_spinner="Initializing clinical models...")
def load_models():
    return (
        Preprocessor(),
        NEREngine(),
        SummarizationEngine(GROQ_API_KEY),
        Evaluator(),
    )

preprocessor, ner, summarizer, evaluator = load_models()

# ── Sample Medical Reports ────────────────────────────────
SAMPLE_LAB_REPORT = """PATIENT NAME: Rajesh Kumar | AGE: 54 | GENDER: Male
DATE OF TEST: 12-Nov-2026 | DR: Dr. A. Sharma, MD

COMPREHENSIVE METABOLIC & LIPID PROFILE:
- Fasting Blood Sugar (FBS): 148 mg/dL (Reference: 70 - 100 mg/dL) [HIGH]
- Post Prandial Blood Sugar (PPBS): 215 mg/dL (Reference: < 140 mg/dL) [HIGH]
- HbA1c (Glycated Hemoglobin): 7.8 % (Reference: < 5.7 % Normal, 5.7-6.4 % Prediabetes, >=6.5 % Diabetes) [HIGH]
- Total Cholesterol: 242 mg/dL (Reference: < 200 mg/dL) [HIGH]
- Triglycerides: 210 mg/dL (Reference: < 150 mg/dL) [HIGH]
- HDL Cholesterol: 38 mg/dL (Reference: > 40 mg/dL) [LOW]
- LDL Cholesterol: 162 mg/dL (Reference: < 100 mg/dL) [HIGH]
- VLDL Cholesterol: 42 mg/dL (Reference: < 30 mg/dL) [HIGH]
- Serum Creatinine: 0.9 mg/dL (Reference: 0.7 - 1.3 mg/dL) [NORMAL]
- Blood Urea Nitrogen (BUN): 16 mg/dL (Reference: 7 - 20 mg/dL) [NORMAL]
- Estimated GFR: > 90 mL/min/1.73m2 [NORMAL]"""

SAMPLE_CT_REPORT = """PATIENT NAME: Sunita Devi | AGE: 62 | GENDER: Female
PROCEDURE: HRCT Chest (High Resolution Computed Tomography) - Non Contrast
CLINICAL INDICATION: Persistent dry cough and mild exertional breathlessness for 3 weeks.

FINDINGS:
Trachea and major bronchi are patent with normal caliber.
Subpleural ground-glass opacities noted predominantly in the bilateral lower lobes with minimal reticulation.
No evidence of dense consolidation, cavitation, or pleural effusion.
Cardiac size is within normal limits. Pericardium appears normal.
Mediastinal and hilar lymph nodes show no significant enlargement (short axis < 10 mm).
Visualized upper abdominal structures are unremarkable.
Bone windows show age-related degenerative changes in the dorsal spine.

IMPRESSION:
1. Mild bilateral subpleural ground-glass opacities in lower lobes, likely representing resolving viral/atypical pneumonitis or early interstitial inflammatory changes.
2. Clinical correlation and follow-up HRCT in 6 to 8 weeks is suggested."""

# ── Helper Functions ──────────────────────────────────────
def extract_pdf(file_obj) -> str:
    temp_filename = "temp_upload.pdf"
    with open(temp_filename, "wb") as f:
        f.write(file_obj.read())

    try:
        llama_key = os.getenv("LLAMA_PARSE_API_KEY")
        if llama_key and "PASTE" not in llama_key:
            try:
                from llama_parse import LlamaParse
                parser = LlamaParse(
                    api_key=llama_key,
                    result_type="text",
                    verbose=True
                )
                documents = parser.load_data(temp_filename)
                full_text = "\n\n".join([doc.text for doc in documents])
                if full_text.strip():
                    return full_text
            except Exception as e:
                logger.warning(f"LlamaParse parsing error: {e}, falling back to pdfplumber")

        import pdfplumber
        pages_text = []
        with pdfplumber.open(temp_filename) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    pages_text.append(extracted)
        
        full_text = "\n\n".join(pages_text).strip()
        if not full_text:
            raise ValueError("Could not extract readable text from PDF.")
        return full_text

    except Exception as e:
        raise ValueError(f"PDF extraction failed: {str(e)}")
        
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

def read_txt(file_obj) -> str:
    text = file_obj.read().decode("utf-8", errors="ignore").strip()
    if not text:
        raise ValueError("Uploaded TXT file is empty.")
    return text

# ── Sidebar Controls ─────────────────────────────────────
with st.sidebar:
    st.markdown("### 🩺 Clinical Control Panel")
    
    # Engine status pill
    is_connected = bool(summarizer.client)
    status_bg = "rgba(16, 185, 129, 0.2)" if is_connected else "rgba(245, 158, 11, 0.2)"
    status_text_color = "#34d399" if is_connected else "#fbbf24"
    status_dot = "#10b981" if is_connected else "#f59e0b"
    status_label = "Groq Engine Active" if is_connected else "API Key Required"
    
    st.markdown(f"""
    <div style="background: {status_bg}; color: {status_text_color}; padding: 10px 14px; border-radius: 10px; font-size: 0.88rem; font-weight: 600; display: flex; align-items: center; gap: 8px; margin-bottom: 16px; border: 1px solid {status_text_color}40;">
        <span style="height: 10px; width: 10px; background-color: {status_dot}; border-radius: 50%; display: inline-block;"></span>
        {status_label}
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("#### ⚙️ Pipeline Settings")
    show_entities = st.checkbox("Extract Medical Entities", value=True, help="Extracts Diseases & Medications via biomedical scispaCy")
    run_eval = st.checkbox("ROUGE Evaluation Mode", value=False, help="Computes ROUGE-1, ROUGE-2, and ROUGE-L against a reference summary")
    
    st.divider()
    
    st.markdown("#### 🔑 API Authentication")
    groq_key_input = st.text_input(
        "Groq API Key", 
        value=os.getenv("GROQ_API_KEY", ""), 
        type="password",
        help="Groq API key from console.groq.com"
    )
    if groq_key_input:
        os.environ["GROQ_API_KEY"] = groq_key_input
        summarizer.set_api_key(groq_key_input)
        
    llama_key_input = st.text_input(
        "LlamaParse Key (Optional)", 
        value=os.getenv("LLAMA_PARSE_API_KEY", ""), 
        type="password",
        help="Optional: For cloud table parsing. Falls back to pdfplumber if omitted."
    )
    if llama_key_input:
        os.environ["LLAMA_PARSE_API_KEY"] = llama_key_input

    st.divider()
    
    st.markdown("#### 🧠 Model Specifications")
    st.markdown(f"**LLM:** `{summarizer.model_name}`")
    st.markdown(f"**NER:** `scispaCy en_ner_bc5cdr_md`")
    st.markdown(f"**Parser:** `pdfplumber / LlamaParse`")
    
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    st.caption("🔒 **Stateless & Confidential:** Medical reports are processed in-memory and never logged or stored.")

# ── Main Header Banner ───────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 16px;">
        <div>
            <div style="display: flex; align-items: center; gap: 14px;">
                <span style="font-size: 2.5rem; filter: drop-shadow(0 2px 6px rgba(0,0,0,0.3));">🩺</span>
                <div>
                    <h1 class="hero-title">LLM Medical Report Summarizer</h1>
                    <p class="hero-subtitle">Transform complex laboratory and radiology findings into clear, patient-friendly insights.</p>
                </div>
            </div>
        </div>
        <div style="display: flex; gap: 8px; flex-wrap: wrap;">
            <span class="badge-pill">⚡ Groq LLM Acceleration</span>
            <span class="badge-pill">🧬 Biomedical NER</span>
            <span class="badge-pill">🛡️ HIPAA Conscious & Stateless</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Session State for Report Text ─────────────────────────
if "input_text" not in st.session_state:
    st.session_state["input_text"] = ""

# ── Quick Test Samples Bar (High Contrast & Flexible) ────
st.markdown("""
<div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px; flex-wrap: wrap; gap: 8px;">
    <span style="font-weight: 700; color: #f8fafc; font-size: 1.05rem;">🚀 Instant Test Samples</span>
    <span style="color: #94a3b8; font-size: 0.85rem;">Click to load sample reports for one-click testing</span>
</div>
""", unsafe_allow_html=True)

col_s1, col_s2, col_s3, _ = st.columns([1.5, 1.5, 1.2, 2.5])
with col_s1:
    if st.button("🧪 Lab Report", use_container_width=True, help="Load complete metabolic and lipid profile"):
        st.session_state["input_text"] = SAMPLE_LAB_REPORT
        st.rerun()
with col_s2:
    if st.button("🩻 Chest CT", use_container_width=True, help="Load HRCT Chest radiology report"):
        st.session_state["input_text"] = SAMPLE_CT_REPORT
        st.rerun()
with col_s3:
    if st.button("🔄 Reset", use_container_width=True, help="Clear input report"):
        st.session_state["input_text"] = ""
        st.rerun()

st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

# ── Step 1: Input Medical Report ──────────────────────────
st.markdown("<div class='section-title'><span>📄</span> Step 1 — Input Medical Report</div>", unsafe_allow_html=True)

tab_paste, tab_pdf, tab_txt = st.tabs(["📝 Paste Report Text", "📄 Upload Medical PDF", "📑 Upload Text File"])

raw_text = ""

with tab_paste:
    pasted = st.text_area(
        label="Medical report content:",
        label_visibility="collapsed",
        value=st.session_state["input_text"],
        height=250,
        placeholder="Paste complete blood test, lipid profile, metabolic panel, or imaging report text here..."
    )
    if pasted.strip():
        raw_text = pasted.strip()

with tab_pdf:
    uploaded_pdf = st.file_uploader("Upload text-based PDF report", type=["pdf"], key="pdf_uploader")
    if uploaded_pdf:
        try:
            with st.spinner("Extracting clinical text from PDF..."):
                extracted_pdf_text = extract_pdf(uploaded_pdf)
                raw_text = extracted_pdf_text
                st.success(f"Extracted {len(extracted_pdf_text.split())} words from PDF successfully.")
                with st.expander("👁️ Inspect Extracted Text"):
                    st.text_area("Extracted Content", extracted_pdf_text, height=180, disabled=True)
        except Exception as e:
            st.error(str(e))

with tab_txt:
    uploaded_txt = st.file_uploader("Upload medical .txt file", type=["txt"], key="txt_uploader")
    if uploaded_txt:
        try:
            extracted_txt = read_txt(uploaded_txt)
            raw_text = extracted_txt
            st.success(f"Loaded {len(extracted_txt.split())} words successfully.")
            with st.expander("👁️ Inspect TXT Content"):
                st.text_area("TXT Content", extracted_txt, height=180, disabled=True)
        except Exception as e:
            st.error(str(e))

# Evaluation reference summary
ref_summary = ""
if run_eval:
    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'><span>🎯</span> Reference Patient Summary (for ROUGE)</div>", unsafe_allow_html=True)
    ref_summary = st.text_area(
        label="Reference summary:",
        label_visibility="collapsed",
        height=100,
        placeholder="Provide the gold-standard summary to quantitatively evaluate the model's output..."
    )

st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

# ── Action Bar ────────────────────────────────────────────
col_act_btn, col_act_meta = st.columns([1.5, 2.2])
with col_act_btn:
    generate_clicked = st.button(
        "✨ Generate Patient Summary",
        type="primary",
        disabled=not raw_text,
        use_container_width=True
    )
with col_act_meta:
    if raw_text:
        word_count = len(raw_text.split())
        char_count = len(raw_text)
        st.markdown(
            f'<div style="color: #cbd5e1; font-size: 0.95rem; padding-top: 10px;">'
            f'📊 <b style="color: #38bdf8;">{word_count}</b> words &nbsp;|&nbsp; <b style="color: #38bdf8;">{char_count}</b> characters ready for summarization'
            f'</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div style="color: #94a3b8; font-size: 0.95rem; padding-top: 10px;">'
            '👈 Paste a report or select a sample above to generate a summary.'
            '</div>',
            unsafe_allow_html=True
        )

# ── Pipeline Execution ────────────────────────────────────
if generate_clicked and raw_text:
    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    st.markdown("<hr style='border: none; border-top: 1px solid #334155; margin: 20px 0;'>", unsafe_allow_html=True)
    
    # 1. Preprocessing
    with st.spinner("🧹 Normalizing medical terminology & formatting..."):
        try:
            preprocessed = preprocessor.process(raw_text)
        except Exception as e:
            st.error(f"Preprocessing error: {e}")
            st.stop()

    # 2. Biomedical NER Extraction
    entities = []
    grouped = {}
    if show_entities:
        with st.spinner("🧬 Extracting biomedical entities via scispaCy (en_ner_bc5cdr_md)..."):
            try:
                entities = ner.extract(preprocessed["clean_text"])
                grouped = ner.group_by_type(entities)
            except Exception as e:
                st.error(f"NER extraction error: {e}")
                st.stop()

    # 3. LLM Summarization
    with st.spinner(f"🤖 Generating patient-friendly summary via Groq ({summarizer.model_name})..."):
        try:
            current_key = groq_key_input or os.getenv("GROQ_API_KEY")
            primary = summarizer.generate(preprocessed["clean_text"], api_key=current_key)
        except Exception as e:
            st.error(f"Summarization error: {e}")
            st.stop()

    # ── Step 2: Output Presentation ───────────────────────
    st.markdown("<div class='section-title'><span>📋</span> Step 2 — Analysis & Summary Results</div>", unsafe_allow_html=True)
    
    # Medical Dashboard Metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("⏱️ API Latency", f"{primary['latency_seconds']} s")
    with m2:
        st.metric("🧬 Entities Found", f"{len(entities)}")
    with m3:
        st.metric("🤖 LLM Engine", primary["model_used"].split('/')[-1])
    with m4:
        st.metric("📝 Words Processed", f"{len(preprocessed['clean_text'].split())}")

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # Output Tabs
    res_tab_summary, res_tab_ner, res_tab_clean = st.tabs([
        "📋 Patient Summary", 
        "🧬 Biomedical Entities", 
        "🔍 Structured Clinical Input"
    ])

    with res_tab_summary:
        if primary.get("fallback_triggered"):
            st.warning(primary["patient"])
        else:
            summary_content = primary["patient"] or "No summary generated."
            
            # Render with rich native markdown inside custom styled border container
            with st.container(border=True):
                st.markdown(summary_content)
            
            st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
            
            # Download & Action buttons
            down_col, _ = st.columns([1.5, 3])
            with down_col:
                st.download_button(
                    label="📥 Download Patient Summary (.txt)",
                    data=summary_content,
                    file_name="patient_medical_summary.txt",
                    mime="text/plain",
                    help="Save this generated patient summary as a text file"
                )

    with res_tab_ner:
        if not show_entities:
            st.info("Medical entity extraction is disabled in the sidebar options.")
        elif not grouped:
            st.info("No specific diseases or medications were detected in this report.")
        else:
            st.markdown("<div style='color: #cbd5e1; margin-bottom: 12px;'>Biomedical entities extracted via <b>scispaCy en_ner_bc5cdr_md</b>:</div>", unsafe_allow_html=True)
            
            for label, terms in grouped.items():
                unique_terms = sorted(list(set(terms)))
                is_disease = "DISEASE" in label.upper()
                badge_class = "tag-disease" if is_disease else "tag-chemical"
                icon = "🦠" if is_disease else "💊"
                card_title = "Diseases & Conditions" if is_disease else "Chemicals & Medications"
                
                st.markdown(f"""
                <div style="background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 20px; margin-bottom: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);">
                    <div style="font-weight: 700; color: #f8fafc; font-size: 1.05rem; margin-bottom: 10px; display: flex; align-items: center; gap: 8px;">
                        <span>{icon}</span> {card_title}
                        <span style="background: #334155; color: #cbd5e1; padding: 2px 10px; border-radius: 12px; font-size: 0.8rem; font-weight: 600;">{len(unique_terms)} items</span>
                    </div>
                    <div>
                        {' '.join([f'<span class="entity-tag {badge_class}">{term}</span>' for term in unique_terms])}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    with res_tab_clean:
        st.markdown("<div style='color: #cbd5e1; margin-bottom: 8px;'>Preprocessed and normalized report text delivered to the LLM:</div>", unsafe_allow_html=True)
        st.text_area(
            label="Preprocessed text:",
            label_visibility="collapsed",
            value=preprocessed["clean_text"],
            height=260,
            disabled=True
        )

    # ── Step 3: Quantitative ROUGE Evaluation ─────────────
    if run_eval and ref_summary.strip():
        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'><span>📊</span> Step 3 — Quantitative Evaluation Metrics</div>", unsafe_allow_html=True)
        
        with st.spinner("Computing ROUGE evaluation scores against reference summary..."):
            try:
                metrics = evaluator.evaluate(
                    prediction=primary["patient"],
                    reference=ref_summary,
                    entities=entities,
                    ner_engine=ner,
                )
                r = metrics["rouge"]
                
                eval_c1, eval_c2, eval_c3 = st.columns(3)
                with eval_c1:
                    st.metric("ROUGE-1 F1 (Unigrams)", f"{r['rouge1']:.4f}")
                with eval_c2:
                    st.metric("ROUGE-2 F1 (Bigrams)", f"{r['rouge2']:.4f}")
                with eval_c3:
                    st.metric("ROUGE-L F1 (Longest Sequence)", f"{r['rougeL']:.4f}")
            except Exception as e:
                st.error(f"Evaluation error: {e}")

# ── Footer ────────────────────────────────────────────────
st.markdown("""
<br><hr style='border: none; border-top: 1px solid #334155;'>
<div style='text-align: center; color: #94a3b8; font-size: 0.86rem; padding-bottom: 24px;'>
    <b>LLM Medical Report Summarization System</b> &nbsp;•&nbsp; Free & CPU-Compatible &nbsp;•&nbsp; For Informational Assistance Only &nbsp;•&nbsp; Consult a Healthcare Professional for Medical Decisions
</div>
""", unsafe_allow_html=True)
