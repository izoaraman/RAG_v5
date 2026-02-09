"""Document-level summary generation for enhanced retrieval."""

import json
import logging
import os
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
from pydantic import SecretStr

from rag_crawler.utils.db_utils import store_document_summary

logger = logging.getLogger(__name__)


class DocumentSummarizer:
    """
    Generates document-level summaries at ingestion time.

    Enables LogicRAG-style warm-up retrieval by providing broad document-level
    context before drilling into specific chunks.
    """

    def __init__(self, model_name: str | None = None, temperature: float = 0.0):
        """Initialize summarizer configuration with lazy LLM setup."""
        self.model_name = model_name or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")
        self.temperature = temperature
        self._llm: AzureChatOpenAI | None = None

    @property
    def llm(self) -> AzureChatOpenAI:
        """Lazy-initialize Azure OpenAI client."""
        if self._llm is None:
            api_key = os.getenv("AZURE_OPENAI_API_KEY")
            self._llm = AzureChatOpenAI(
                azure_deployment=self.model_name,
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=SecretStr(api_key) if api_key else None,
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
                temperature=self.temperature,
            )
            logger.info(f"DocumentSummarizer initialized with model: {self.model_name}")
        return self._llm

    async def summarize_document(
        self, title: str, content: str, source: str = ""
    ) -> dict[str, Any]:
        """Generate summary, entities, and topics for a document."""
        truncated_content = (content or "")[:8000]

        system_prompt = (
            "You analyze regulatory and policy documents and return structured JSON. "
            "Be factual, concise, and avoid speculation."
        )
        user_prompt = (
            "Analyze the following document and provide:\n"
            "1. A concise summary (200-300 words) capturing the key information\n"
            "2. A list of key entities (people, organizations, laws, regulations, dates)\n"
            "3. A list of key topics/themes\n\n"
            f"Document Title: {title}\n"
            f"Source: {source}\n"
            f"Content: {truncated_content}\n\n"
            "Respond as JSON: "
            '{"summary": str, "key_entities": [str], "key_topics": [str]}'
        )

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
            response = await self.llm.ainvoke(messages)
            response_text = str(response.content).strip()

            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
                response_text = response_text.strip()

            try:
                parsed = json.loads(response_text)
                summary = str(parsed.get("summary", "")).strip()
                key_entities = parsed.get("key_entities", [])
                key_topics = parsed.get("key_topics", [])

                if not isinstance(key_entities, list):
                    key_entities = []
                if not isinstance(key_topics, list):
                    key_topics = []

                return {
                    "summary": summary or response_text,
                    "key_entities": [str(item) for item in key_entities if str(item).strip()],
                    "key_topics": [str(item) for item in key_topics if str(item).strip()],
                }
            except json.JSONDecodeError:
                logger.warning("Failed to parse summary JSON response, using fallback")
                return {
                    "summary": response_text,
                    "key_entities": [],
                    "key_topics": [],
                }
        except Exception as e:
            logger.error(f"Document summarization failed: {e}")
            return {
                "summary": truncated_content,
                "key_entities": [],
                "key_topics": [],
            }

    async def summarize_and_store(
        self,
        document_id: str,
        title: str,
        content: str,
        source: str = "",
        embedder=None,
    ) -> dict[str, Any]:
        """Summarize a document and store the result in the database."""
        summary_data = await self.summarize_document(title=title, content=content, source=source)

        summary_embedding: list[float] | None = None
        if embedder and summary_data.get("summary"):
            try:
                summary_embedding = await embedder.generate_embedding(summary_data["summary"])
            except Exception as e:
                logger.warning(f"Failed to generate summary embedding for {document_id}: {e}")

        summary_id = await store_document_summary(
            document_id=document_id,
            summary=summary_data["summary"],
            key_entities=summary_data["key_entities"],
            key_topics=summary_data["key_topics"],
            embedding=summary_embedding,
        )

        return {
            "id": summary_id,
            **summary_data,
        }
