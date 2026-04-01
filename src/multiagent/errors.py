class MultiAgentError(Exception):
    """Base exception for the project."""


class ConfigurationError(MultiAgentError):
    """Raised when configuration is invalid or incomplete."""


class BudgetExceededError(MultiAgentError):
    """Raised when a budget guardrail blocks new work."""


class SchemaValidationError(MultiAgentError):
    """Raised when a model response cannot be validated."""


class SchedulerError(MultiAgentError):
    """Raised when the DAG or scheduler state is invalid."""


class RepoMutationError(MultiAgentError):
    """Raised when guarded repo mutation fails."""
