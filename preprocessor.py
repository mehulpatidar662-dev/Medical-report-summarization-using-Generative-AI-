import re
from observability import logger

class Preprocessor:
    def __init__(self):
        # No heavy Spacy model loading needed for Llama 3
        logger.info("preprocessor_initialized_lightweight")

    def clean(self, text: str) -> str:
        """
        Cleans text while PRESERVING paragraph structure (newlines).
        """
        
        # 1. Remove non-printable characters (garbage from PDF extraction)

        text = re.sub(r"[^\x00-\x7F]+", " ", text)

        # 2. Fix weird spacing issues but KEEP NEWLINES (\n)

        # This regex replaces multiple spaces on a single line with one space,

        # but respects the newline character so paragraphs stay intact.

        text = re.sub(r"[ \t]+", " ", text)
    
        # 3. Normalize multiple newlines to max 2 (paragraph breaks)

        text = re.sub(r"(\n\s*){3,}", "\n\n", text)

        # Normalize units by removing unnecessary punctuation between letters

        text = re.sub(r"([a-zA-Z])\.(?=[a-zA-Z])", r"\1", text)

        # Normalize slash spacing

        text = re.sub(r"\s*/\s*", "/", text)

        # Normalize percentage spacing

        text = re.sub(r"(\d)\s*%", r"\1%", text)

        # Ensure space between number and unit

        text = re.sub(r"(\d)([a-zA-Z%])", r"\1 \2", text)

        text = re.sub(r"\.\s*/", "/", text)  # current (good)

        text = re.sub(r"/\s*", "/", text)    # normalize slash spacing
        
        return text.strip()

    def process(self, raw_text: str) -> dict:
        """
        Passes the full, structured text to the LLM.
        """
        # We skip manual abbreviation expansion. 
        # Llama 3 is smart enough to understand medical acronyms in context.
        clean_text = self.clean(raw_text)
        
        # Simple heuristic for sections (optional, just for metadata)
        # We don't split the text for the model; we give the model the WHOLE text.
        sections = {}
        if "IMPRESSION:" in clean_text.upper():
            sections["IMPRESSION"] = "Present"
        if "HISTORY:" in clean_text.upper():
            sections["HISTORY"] = "Present"

        logger.info("preprocessing_done", char_count=len(clean_text))

        return {
            "clean_text": clean_text,  # The full, structured report
            "sentences": [],           # Not needed for LLM approach
            "sections": sections,      # Metadata only
        }
