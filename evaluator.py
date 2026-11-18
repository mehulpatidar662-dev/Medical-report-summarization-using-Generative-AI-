from rouge_score import rouge_scorer
from observability import logger


class Evaluator:
    def __init__(self):
        self._rouge = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )
        logger.info("evaluator_initialized")

    def compute_rouge(self, prediction: str, reference: str) -> dict:
        scores = self._rouge.score(reference, prediction)
        result = {
            "rouge1": round(scores["rouge1"].fmeasure, 4),
            "rouge2": round(scores["rouge2"].fmeasure, 4),
            "rougeL": round(scores["rougeL"].fmeasure, 4),
        }
        logger.info("rouge_computed", **result)
        return result

    def evaluate(self, prediction: str, reference: str,
                 entities: list, ner_engine) -> dict:
        rouge         = self.compute_rouge(prediction, reference)
        return {
            "rouge":         rouge
        }
