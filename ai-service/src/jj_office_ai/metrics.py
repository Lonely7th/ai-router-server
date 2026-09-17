from prometheus_client import Counter, Gauge, Histogram

REQUESTS = Counter(
    "jj_office_ai_requests_total",
    "AI completion requests",
    ("model", "stream", "status"),
)
REQUEST_DURATION = Histogram(
    "jj_office_ai_request_duration_seconds",
    "End-to-end AI gateway request duration",
    ("model", "stream"),
    buckets=(0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300, 600),
)
ACTIVE_REQUESTS = Gauge(
    "jj_office_ai_active_requests",
    "Currently active upstream requests",
    ("model", "stream"),
)
TOKENS = Counter(
    "jj_office_ai_tokens_total",
    "Provider-reported token usage",
    ("model", "kind"),
)
UPSTREAM_ERRORS = Counter(
    "jj_office_ai_upstream_errors_total",
    "Upstream errors grouped by status code",
    ("status_code",),
)
QUOTA_ERRORS = Counter(
    "jj_office_ai_quota_errors_total",
    "Quota reserve or settlement failures",
    ("operation",),
)


def observe_tokens(model: str, prompt_tokens: int, completion_tokens: int) -> None:
    if prompt_tokens:
        TOKENS.labels(model=model, kind="input").inc(prompt_tokens)
    if completion_tokens:
        TOKENS.labels(model=model, kind="output").inc(completion_tokens)

