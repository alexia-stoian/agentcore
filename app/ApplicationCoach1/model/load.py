from strands.models.bedrock import BedrockModel


def load_model() -> BedrockModel:
    """Get Bedrock model client using IAM credentials.

    Perf: Sonnet (much faster/cheaper than Opus) + system-prompt caching so the large
    static system prompt is not re-prefilled every turn.
    """
    return BedrockModel(
        model_id="eu.anthropic.claude-sonnet-4-6",
        cache_prompt="default",
    )
