# LLM-Based Medical Report Summarization System

---

## Overview

An end-to-end LLM pipeline that generates **patient-friendly summaries** from clinical medical reports — laboratory reports and CT scan reports — using a medically-prompted Large Language Model via the Groq API.

The system bridges the gap between clinical NLP extraction tools (which identify entities but don't summarize) and general LLMs (which summarize but lack medical domain awareness).

---

## Key Features

- **Intelligent PDF extraction** via LlamaParse — preserves tabular structure in lab reports
- **Biomedical NER** via scispaCy `en_ner_bc5cdr_md` — extracts diseases and medications
- **Chain-of-Thought prompting** — identifies report type before summarizing
- **Patient-friendly output** — Key Findings, Interpretation, Recommendations
- **ROUGE evaluation** — optional, requires reference summary
- **Observability** — Prometheus metrics + structured logging
- **Stateless by design** — no patient data stored on disk
- **Free, CPU-only** — no GPU required

---

## Tech Stack

| Component | Technology |
|---|---|
| LLM Summarization | Groq API — `llama-3.3-70b-versatile` |
| PDF Extraction | LlamaParse |
| Biomedical NER | scispaCy `en_ner_bc5cdr_md` |
| Preprocessing | spaCy |
| Evaluation | rouge-score |
| UI | Streamlit |
| Observability | Prometheus Client + Custom Logger |
| API Key Management | python-dotenv |

---

## Project Structure

```
├── app.py              # Streamlit UI + pipeline orchestration
├── preprocessor.py     # Text cleaning and section detection
├── ner.py              # scispaCy NER entity extraction
├── summarizer.py       # Groq API + Chain-of-Thought prompt
├── evaluator.py        # ROUGE metric computation
├── observability.py    # Structured logging + Prometheus metrics
├── requirements.txt    # Project dependencies
└── .env.example        # API key template
```

---

## Setup and Installation

### Prerequisites
- Python 3.11 (required — 3.12+ has spaCy compatibility issues)
- Internet connection (for Groq and LlamaParse APIs)

### Step 1 — Clone the repository
```bash
git clone https://github.com/Kirannpatil/your-repo-name
cd your-repo-name
```

### Step 2 — Create virtual environment
```bash
py -3.11 -m venv venv
venv\Scripts\activate.bat        # Windows
source venv/bin/activate         # macOS / Linux
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Install scispaCy biomedical model
```bash
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz
```

### Step 5 — Configure API keys
Create a `.env` file in the project folder:
```
GROQ_API_KEY=your_groq_api_key_here
LLAMA_PARSE_API_KEY=your_llamaparse_api_key_here
```

- **Groq API key**: [console.groq.com](https://console.groq.com) → Settings → API Keys
- **LlamaParse API key**: [cloud.llamaindex.ai](https://cloud.llamaindex.ai) → API Keys

### Step 6 — Run the application
```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## Usage

1. Select input method: **Paste Text**, **Upload PDF**, or **Upload TXT**
2. Provide your medical report
3. Click **Generate Summary**
4. View extracted entities and patient-friendly summary
5. Optionally enable evaluation metrics with a reference summary

---

## Evaluation Results

Tested on a real Krsnaa Diagnostics laboratory report (Lipid Profile + FBG + PPBG):

| Metric | Score |
|---|---|
| ROUGE-1 | 0.5960 |
| ROUGE-2 | 0.3200 |
| ROUGE-L | 0.5770 |
| Avg. API Latency | 2.05 seconds |

> **Note:** Lower ROUGE scores on CT scan reports reflect appropriate paraphrasing of clinical findings into plain language — not poor summarization quality.

---

## Supported Report Types

- Laboratory reports (CBC, Lipid Profile, Blood Glucose, LFT, etc.)
- CT Scan reports

---

## Limitations

- Scanned/image-based PDFs are not supported
- ROUGE metrics are insufficient for evaluating patient-friendly summarization — human evaluation by a clinical expert is recommended
- System requires internet connection for API calls
- Not a substitute for professional medical advice

---

## Future Scope

- Human evaluation framework with clinical experts
- Multi-language support (Hindi, Marathi)
- Dual output: clinical + patient summaries
- Mobile application
- EHR integration via HL7 FHIR

---

## Disclaimer

This system is built for informational purposes. The generated summaries are AI-generated and intended for general understanding only. Always consult a qualified healthcare professional for medical advice.
