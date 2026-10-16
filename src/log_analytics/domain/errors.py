"""Domain exceptions. The API layer maps each family to an HTTP status."""


class LogAnalyticsError(Exception):
    """Base class for all expected (non-bug) failures."""


class DataSourceError(LogAnalyticsError):
    """The log source is missing, unreadable or has an unexpected schema."""


class NotFoundError(LogAnalyticsError):
    """A requested resource (run, report) does not exist."""


class ConflictError(LogAnalyticsError):
    """The resource exists but is not in a state that allows the operation (e.g. report not ready)."""


class CapacityError(LogAnalyticsError):
    """Every analysis slot is busy or the job queue is full; the caller should retry later."""


class FeatureUnavailableError(LogAnalyticsError):
    """An optional capability (vector store, LLM) is not configured or not reachable."""


class LLMError(LogAnalyticsError):
    """Base class for language-model failures."""


class LLMUnavailableError(LLMError):
    """The model server is unreachable, overloaded, or the model is not installed."""


class LLMResponseError(LLMError):
    """The model answered, but not with valid output for the requested schema."""
