import time
import os
from groq import Groq

class SummarizationEngine:
    def __init__(self, api_key: str = None):
        """
        Initialize Groq client.
        """
        self.api_key = api_key
        self.client = None
        self.model_name = "openai/gpt-oss-120b" 
        if api_key and "PASTE" not in api_key:
            try:
                self.client = Groq(api_key=api_key)
                self._resolve_model()
            except Exception:
                self.client = None

    def _resolve_model(self):
        if not self.client:
            return
        preferred_models = [
            "openai/gpt-oss-120b",
            "qwen/qwen3.8-27b",
            "openai/gpt-oss-20b",
            "llama-3.3-70b-versatile",
            "llama-3.1-70b-versatile",
            "llama-3.1-8b-instant"
        ]
        try:
            available = [m.id for m in self.client.models.list().data]
            for pref in preferred_models:
                if pref in available:
                    self.model_name = pref
                    return
            for m in available:
                if "whisper" not in m and "guard" not in m:
                    self.model_name = m
                    return
        except Exception:
            pass

    def set_api_key(self, api_key: str):
        if api_key and "PASTE" not in api_key:
            self.api_key = api_key
            try:
                self.client = Groq(api_key=api_key)
                self._resolve_model()
            except Exception:
                self.client = None

    def generate(self, text: str, api_key: str = None) -> dict:
        """
        Generates a patient-friendly summary using Groq API.
        """
        if api_key:
            self.set_api_key(api_key)

        if not self.client:
            return {
                "patient": "⚠️ Groq API key is missing. Please enter your Groq API key in the sidebar or add GROQ_API_KEY to your .env file.",
                "clinical": None,
                "model_used": "none",
                "latency_seconds": 0.0,
                "fallback_triggered": True
            }

        start_time = time.time()
        
        system_prompt = (
            "You are an empathetic, expert medical AI assistant.\n\n"

            "STEP 1 — Identify report type from input:\n"
            "- Laboratory Report: contains test names, numeric values, units, reference ranges.\n"
            "- Radiology Report: contains descriptive scan findings (CT, MRI, X-ray, Ultrasound).\n\n"

            "STEP 2 — Generate a patient-friendly summary in this exact structure:\n\n"

            "Dear Patient,\n\n"
            "Below is a clear, easy-to-understand summary of your medical report:\n\n"

            "### 1. Key Findings\n\n"
            "FOR LAB REPORT: List each test with its value, unit, and reference range. "
            "State clearly whether it is Normal, High, or Low. "
            "Example: 'Your Hemoglobin is 11.0 g/dL, which is below the normal range of 12.0–16.0 g/dL.'\n"
            "FOR RADIOLOGY REPORT: Summarize main scan observations in simple language. "
            "Include clinical indications if mentioned. Do NOT add values or infer beyond what is stated.\n\n"

            "### 2. Clinical Interpretation\n\n"
            "Explain what these findings mean in everyday, jargon-free language. "
            "Clearly distinguish normal findings from values that require attention.\n\n"

            "### 3. Recommendations & Next Steps\n\n"
            "Suggest practical next steps (e.g. consult physician, follow-up tests, lifestyle or dietary measures). "
            "Do NOT diagnose conditions beyond the provided report data.\n\n"

            "If you have any symptoms or immediate concerns, please consult your doctor promptly.\n\n"
            "Wishing you good health.\n\n"

            "RULES:\n"
            "- Maintain an empathetic, professional, and reassuring tone.\n"
            "- Never skip any test or key finding from the original report.\n"
            "- Never assume abnormality unless clearly outside normal reference limits.\n"
            "- Never speculate or fabricate values.\n"
        )

        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ],
                model=self.model_name,
                temperature=0.5,
                max_tokens=900,
            )

            summary = chat_completion.choices[0].message.content

            return {
                "patient": summary,
                "clinical": None,  # Explicitly None as requested
                "model_used": self.model_name,
                "latency_seconds": round(time.time() - start_time, 2),
                "fallback_triggered": False
            }

        except Exception as e:
            return {
                "patient": f"Error generating summary: {str(e)}",
                "clinical": None,
                "model_used": "error",
                "latency_seconds": 0.0,
                "fallback_triggered": True
            }
