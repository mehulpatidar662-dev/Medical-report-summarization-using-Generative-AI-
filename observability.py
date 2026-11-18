import time
import logging
from prometheus_client import Counter, start_http_server, REGISTRY

# ── 1. Structured Logger (Fixes TypeError) ────────────────
class StructuredLogger:
    """
    A wrapper around standard logging that accepts keyword arguments
    (e.g., model="llama3") and formats them into the log string.
    """
    def __init__(self, name):
        self.logger = logging.getLogger(name)
        # Prevent adding multiple handlers if reloaded
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def _format(self, msg, kwargs):
        if kwargs:
            # Converts ("msg", model="x") -> "msg model=x"
            extra_info = " ".join(f"{k}={v}" for k, v in kwargs.items())
            return f"{msg} [{extra_info}]"
        return msg

    def info(self, msg, **kwargs):
        self.logger.info(self._format(msg, kwargs))

    def warning(self, msg, **kwargs):
        self.logger.warning(self._format(msg, kwargs))

    def error(self, msg, **kwargs):
        self.logger.error(self._format(msg, kwargs))

# Initialize the logger
logger = StructuredLogger("medical_summarizer")


# ── 2. Metrics Safety (Fixes Duplicated Timeseries) ───────
def get_or_create_metric(metric_class, name, documentation, labels=None):
    """
    Safely retrieves a metric or creates it if it doesn't exist.
    Prevents 'Duplicated timeseries' error on Streamlit reload.
    """
    # Attempt to unregister existing metric to avoid conflicts
    if name in REGISTRY._names_to_collectors:
        try:
            REGISTRY.unregister(REGISTRY._names_to_collectors[name])
        except Exception:
            pass # Ignore unregister errors
            
    if labels:
        return metric_class(name, documentation, labels)
    return metric_class(name, documentation)


# ── 3. Define Metrics (Restores Missing ENTITY_COUNT) ─────

REQUEST_COUNT = get_or_create_metric(
    Counter, 
    'summarizer_requests', 
    'Total number of summarization requests', 
    ['model', 'status']
)

FALLBACK_COUNT = get_or_create_metric(
    Counter,
    'fallback_events',
    'Total number of times fallback model was triggered'
)

ENTITY_COUNT = get_or_create_metric(
    Counter,
    'entity_extraction_count',
    'Total number of medical entities extracted',
    ['entity_type']
)


# ── 4. Timer Utility ──────────────────────────────────────
class Timer:
    def __init__(self, model_name):
        self.model = model_name
        self.start = 0
        self.elapsed = 0

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = time.time() - self.start


# ── 5. Server Start ───────────────────────────────────────
def start_metrics_server(port=8000):
    try:
        start_http_server(port)
        logger.info(f"Prometheus metrics server started on port {port}")
    except OSError:
        # Port is already in use (expected during Streamlit reload)
        pass
