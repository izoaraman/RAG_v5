"""Provider configuration for Azure OpenAI embedding models."""

import os

from dotenv import load_dotenv
from openai import AsyncAzureOpenAI

load_dotenv()


def get_embedding_client() -> AsyncAzureOpenAI:
    """
    Get Azure OpenAI client for embeddings.

    Required environment variables:
        AZURE_OPENAI_API_KEY: Azure OpenAI API key
        AZURE_OPENAI_ENDPOINT: Azure OpenAI endpoint URL
        AZURE_OPENAI_API_VERSION: API version (e.g., "2024-02-01")

    Returns:
        Configured Azure OpenAI client for embeddings.

    Raises:
        ValueError: If required environment variables are not set.
    """
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

    if not api_key:
        raise ValueError("AZURE_OPENAI_API_KEY environment variable is required")
    if not endpoint:
        raise ValueError("AZURE_OPENAI_ENDPOINT environment variable is required")

    return AsyncAzureOpenAI(
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version=api_version,
    )


def get_embedding_model() -> str:
    """
    Get Azure OpenAI embedding deployment name.

    Returns:
        Embedding deployment name from env or default.
    """
    return os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")


def validate_configuration() -> bool:
    """
    Validate that required Azure OpenAI environment variables are set.

    Returns:
        True if configuration is valid.
    """
    required_vars = [
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "DATABASE_URL",
    ]

    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        print(f"Missing required environment variables: {', '.join(missing_vars)}")
        return False

    return True


def get_model_info() -> dict:
    """
    Get information about current Azure OpenAI model configuration.

    Returns:
        Dictionary with model configuration info.
    """
    return {
        "provider": "azure_openai",
        "endpoint": os.getenv("AZURE_OPENAI_ENDPOINT", "not set"),
        "api_version": os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        "embedding_deployment": get_embedding_model(),
    }
