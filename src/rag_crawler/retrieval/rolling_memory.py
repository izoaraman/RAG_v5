"""
LogicRAG-style rolling memory for iterative multi-hop retrieval.

Maintains a compressed, progressively refined information summary
across retrieval rounds.
"""

import json
import logging
import os
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
from pydantic import SecretStr

logger = logging.getLogger(__name__)


class RollingMemory:
    """LogicRAG-inspired rolling memory that maintains a compressed context summary."""

    def __init__(self, model_name: str | None = None, temperature: float = 0.0):
        self.model_name = model_name or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")
        self.temperature = temperature
        self.llm = AzureChatOpenAI(
            azure_deployment=self.model_name,
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=SecretStr(os.getenv("AZURE_OPENAI_API_KEY", "")),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
            temperature=temperature,
        )
        logger.info(f"RollingMemory initialized with model: {self.model_name}")

    async def create_initial_summary(self, question: str, contexts: list[str]) -> str:
        """Create initial info summary from first retrieval round."""
        if not contexts:
            return ""

        context_text = "\n\n".join(contexts)
        messages = [
            SystemMessage(
                content=(
                    "You create concise, faithful summaries from retrieved evidence. "
                    "Do not add facts not present in the evidence."
                )
            ),
            HumanMessage(
                content=(
                    "Create a concise summary of the following information as it relates to "
                    "answering this question:\n"
                    f"Question: {question}\n"
                    f"Information: {context_text}\n\n"
                    "Your summary should: 1. Include all relevant facts "
                    "2. Exclude irrelevant info 3. Be clear and concise "
                    "4. Preserve specific details, dates, numbers, names"
                )
            ),
        ]

        try:
            response = await self.llm.ainvoke(messages)
            content = response.content
            return content.strip() if isinstance(content, str) else str(content).strip()
        except Exception as e:
            logger.error(f"Failed to create initial summary: {e}")
            return context_text[:2000]

    async def refine_summary(
        self,
        question: str,
        current_summary: str,
        new_contexts: list[str],
    ) -> str:
        """Refine existing summary with new retrieved contexts."""
        if not new_contexts:
            return current_summary

        context_text = "\n\n".join(new_contexts)
        messages = [
            SystemMessage(
                content=(
                    "You refine evidence summaries for question answering. Keep only relevant facts, "
                    "remove redundancy, and preserve specific details."
                )
            ),
            HumanMessage(
                content=(
                    "Refine the following information summary using newly retrieved information.\n"
                    f"Question: {question}\n"
                    f"Current summary: {current_summary}\n"
                    f"New information: {context_text}\n\n"
                    "Your refined summary should: 1. Integrate new relevant facts "
                    "2. Remove redundancies 3. Remain concise "
                    "4. Prioritize info that helps answer the question "
                    "5. Maintain specific details"
                )
            ),
        ]

        try:
            response = await self.llm.ainvoke(messages)
            content = response.content
            return content.strip() if isinstance(content, str) else str(content).strip()
        except Exception as e:
            logger.error(f"Failed to refine summary: {e}")
            return f"{current_summary}\n\n{context_text}".strip()[:3000]

    async def check_sufficiency(
        self,
        question: str,
        info_summary: str,
        subproblems: list[str] | None = None,
    ) -> dict[str, Any]:
        """Check if we have enough information to answer the question."""
        subproblem_text = "\n".join(f"- {s}" for s in (subproblems or []))
        messages = [
            SystemMessage(
                content=(
                    "You judge whether current evidence is sufficient to answer a question. "
                    "Return strict JSON only."
                )
            ),
            HumanMessage(
                content=(
                    "Analyze whether the info_summary is sufficient to answer the question.\n\n"
                    f"Question: {question}\n\n"
                    f"Information summary: {info_summary}\n\n"
                    "Subproblems:\n"
                    f"{subproblem_text if subproblem_text else '- None'}\n\n"
                    "Return JSON with this schema:\n"
                    "{\n"
                    '  "can_answer": true or false,\n'
                    '  "missing_info": "string",\n'
                    '  "current_understanding": "string"\n'
                    "}"
                )
            ),
        ]

        fallback = {
            "can_answer": False,
            "missing_info": "Could not confidently determine sufficiency.",
            "current_understanding": info_summary[:500],
        }

        try:
            response = await self.llm.ainvoke(messages)
            content = response.content
            response_text = content.strip() if isinstance(content, str) else str(content).strip()

            if response_text.startswith("```"):
                response_text = response_text.split("```", maxsplit=2)[1].strip()
                if response_text.startswith("json"):
                    response_text = response_text[4:].strip()

            parsed = json.loads(response_text)
            return {
                "can_answer": bool(parsed.get("can_answer", False)),
                "missing_info": str(parsed.get("missing_info", "")),
                "current_understanding": str(parsed.get("current_understanding", "")),
            }
        except Exception as e:
            logger.warning(f"Failed to parse sufficiency response, using fallback: {e}")
            return fallback
