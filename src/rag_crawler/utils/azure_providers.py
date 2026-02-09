"""
Azure AI Provider Configuration for RAG_v5.

Centralized configuration for all Azure OpenAI services including:
- Chat/Completion models (GPT-4o)
- Embedding models (text-embedding-3-small, ada-002)
- Azure Database for PostgreSQL connection
"""

import logging
import os
from functools import lru_cache
from typing import Any

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class AzureConfig:
    """
    Centralized Azure configuration.

    Loads all Azure credentials from environment variables with validation.
    """

    def __init__(self):
        """Initialize Azure configuration from environment."""
        # Azure OpenAI
        self.openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

        # Deployment names
        self.chat_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")
        self.embedding_deployment = os.getenv(
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small"
        )

        # Azure Database for PostgreSQL
        self.database_url = os.getenv("DATABASE_URL")

        # Validate required fields
        self._validate()

    def _validate(self) -> None:
        """Validate required configuration."""
        missing = []

        if not self.openai_api_key:
            missing.append("AZURE_OPENAI_API_KEY")
        if not self.openai_endpoint:
            missing.append("AZURE_OPENAI_ENDPOINT")
        if not self.database_url:
            missing.append("DATABASE_URL")

        if missing:
            logger.warning(f"Missing Azure configuration: {', '.join(missing)}")

    @property
    def is_valid(self) -> bool:
        """Check if all required configuration is present."""
        return bool(self.openai_api_key and self.openai_endpoint and self.database_url)

    def to_dict(self) -> dict[str, Any]:
        """Export configuration (redacting sensitive values)."""
        return {
            "openai_endpoint": self.openai_endpoint,
            "openai_api_version": self.openai_api_version,
            "chat_deployment": self.chat_deployment,
            "embedding_deployment": self.embedding_deployment,
            "database_url": self.database_url[:20] + "..." if self.database_url else None,
            "is_valid": self.is_valid,
        }


@lru_cache()
def get_azure_config() -> AzureConfig:
    """Get cached Azure configuration."""
    return AzureConfig()


def get_chat_llm(
    temperature: float = 0.3,
    max_tokens: int = 2000,
    **kwargs: Any,
):
    """
    Get Azure ChatOpenAI instance for chat/completion.

    Args:
        temperature: Response creativity (0.0-1.0).
        max_tokens: Maximum response tokens.
        **kwargs: Additional LangChain parameters.

    Returns:
        Configured AzureChatOpenAI instance.
    """
    from langchain_openai import AzureChatOpenAI

    config = get_azure_config()

    return AzureChatOpenAI(
        azure_deployment=config.chat_deployment,
        azure_endpoint=config.openai_endpoint,
        api_key=config.openai_api_key,
        api_version=config.openai_api_version,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )


def get_embedding_model():
    """
    Get Azure OpenAI embedding model name.

    Returns:
        Embedding deployment name.
    """
    config = get_azure_config()
    return config.embedding_deployment


def get_embedding_client():
    """
    Get Azure OpenAI async client for embeddings.

    Returns:
        Configured AsyncAzureOpenAI client.
    """
    from openai import AsyncAzureOpenAI

    config = get_azure_config()

    return AsyncAzureOpenAI(
        api_key=config.openai_api_key,
        azure_endpoint=config.openai_endpoint,
        api_version=config.openai_api_version,
    )


def get_database_url() -> str:
    """
    Get Azure Database for PostgreSQL connection URL.

    Returns:
        Database connection string.

    Raises:
        ValueError: If DATABASE_URL is not configured.
    """
    config = get_azure_config()

    if not config.database_url:
        raise ValueError("DATABASE_URL environment variable is required")

    return config.database_url


def validate_azure_configuration() -> dict[str, Any]:
    """
    Validate all Azure configuration and return status.

    Returns:
        Dictionary with validation results.
    """
    config = get_azure_config()
    results = {
        "is_valid": config.is_valid,
        "checks": {},
    }

    # Check OpenAI
    results["checks"]["openai_api_key"] = bool(config.openai_api_key)
    results["checks"]["openai_endpoint"] = bool(config.openai_endpoint)
    results["checks"]["chat_deployment"] = config.chat_deployment
    results["checks"]["embedding_deployment"] = config.embedding_deployment

    # Check Database
    results["checks"]["database_url"] = bool(config.database_url)

    # Test OpenAI connection
    if config.openai_api_key and config.openai_endpoint:
        try:
            llm = get_chat_llm(temperature=0, max_tokens=10)
            # Just verify we can create the client
            results["checks"]["openai_connection"] = True
        except Exception as e:
            results["checks"]["openai_connection"] = f"Error: {e}"

    return results


# Environment variable template for documentation
ENV_TEMPLATE = """
# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY=your-azure-openai-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-01
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-small

# Azure Database for PostgreSQL
DATABASE_URL=postgresql://user:password@your-server.postgres.database.azure.com:5432/ragdb?sslmode=require
"""


def print_env_template():
    """Print environment variable template for setup."""
    print(ENV_TEMPLATE)
