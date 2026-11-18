import spacy
from observability import logger, ENTITY_COUNT

class NEREngine:
    def __init__(self):
        self.nlp = None
        self.model_name = "en_ner_bc5cdr_md"
        logger.info("ner_engine_created_lazy")

    def _load_model(self):
        if self.nlp is not None:
            return
        models = ["en_ner_bc5cdr_md", "en_core_web_sm"]
        for model in models:
            try:
                self.nlp = spacy.load(model)
                self.model_name = model
                break
            except OSError:
                continue
        
        if not self.nlp:
            logger.warning("downloading_fallback_model")
            spacy.cli.download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
            self.model_name = "en_core_web_sm"
            
        logger.info("ner_initialized", model=self.model_name)

    def extract(self, text: str) -> list:
        """
        Returns a list of (text, label) tuples.
        """
        self._load_model()
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            entities.append((ent.text, ent.label_))
            ENTITY_COUNT.labels(entity_type=ent.label_).inc()
            
        return entities

    def group_by_type(self, entities: list) -> dict:
        """
        Groups entities by their label for the UI.
        """
        grouped = {}
        for text, label in entities:
            if label not in grouped:
                grouped[label] = []
            grouped[label].append(text)
        return grouped
    
    def entity_recall(self, entities: list, summary: str) -> float:
        if not entities:
            return 0.0
        self._load_model()
        summary_lower = summary.lower()
        matched = sum(1 for text, label in entities if text.lower() in summary_lower)
        recall  = round(matched / len(entities), 4)
        logger.info("entity_recall", recall=recall, matched=matched, total=len(entities))
        return recall
