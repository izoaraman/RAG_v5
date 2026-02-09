"""Lightweight entity and topic extraction for chunk metadata enrichment."""

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


class EntityExtractor:
    """
    Extracts entities and topics from text chunks using regex patterns.

    Designed to be lightweight (no LLM calls) for use during ingestion.
    Uses pattern matching for common entity types in the ACCC/AER domain.
    """

    def __init__(self):
        """Initialize entity extractor with domain-specific patterns."""
        self._org_patterns = [
            r"\b(?:ACCC|AER|AEMC|AEMO|NCC)\b",
            r"\b(?:Australian Competition and Consumer Commission)\b",
            r"\b(?:Australian Energy Regulator)\b",
            r"\b(?:Australian Energy Market Commission)\b",
            r"\b(?:Australian Energy Market Operator)\b",
            r"\b(?:National Competition Council)\b",
        ]
        self._legal_patterns = [
            r"\b(?:Competition and Consumer Act)\b",
            r"\b(?:National Electricity Law)\b",
            r"\b(?:National Gas Law)\b",
            r"\b(?:Consumer Data Right)\b",
            r"\b(?:Memorandum of Understanding)\b",
            r"\bAct\s+\d{4}\b",
            r"\bRegulation\s+\d+\b",
            r"\b[Ss]ection\s+\d+[A-Za-z]?\b",
        ]
        self._date_patterns = [
            r"\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b",
            r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b",
            r"\b\d{4}[-/]\d{2}[-/]\d{2}\b",
            r"\bFY\s*\d{2,4}\b",
        ]
        self._money_patterns = [
            r"\$[\d,]+(?:\.\d{2})?\s*(?:million|billion|m|b)?\b",
            r"\b\d+(?:\.\d+)?\s*(?:million|billion)\s*dollars?\b",
        ]

    def extract_entities(self, text: str) -> list[str]:
        """Extract named entities from text using patterns."""
        entities = set()

        for pattern_list in [
            self._org_patterns,
            self._legal_patterns,
            self._date_patterns,
            self._money_patterns,
        ]:
            for pattern in pattern_list:
                matches = re.findall(pattern, text)
                entities.update(matches)

        return sorted(entities)

    def extract_topics(self, text: str) -> list[str]:
        """Extract key topics/themes from text using keyword frequency."""
        topic_keywords = {
            "enforcement": [
                "enforcement",
                "penalty",
                "fine",
                "court",
                "prosecution",
                "compliance",
            ],
            "consumer_protection": [
                "consumer",
                "protection",
                "fair trading",
                "misleading",
                "deceptive",
            ],
            "competition": [
                "competition",
                "merger",
                "acquisition",
                "market power",
                "monopoly",
                "cartel",
            ],
            "energy_regulation": [
                "energy",
                "electricity",
                "gas",
                "network",
                "tariff",
                "distribution",
            ],
            "pricing": ["price", "pricing", "cost", "tariff", "fee", "charge"],
            "governance": ["governance", "board", "committee", "chairman", "commissioner"],
            "policy": ["policy", "regulation", "framework", "guideline", "standard", "rule"],
            "data_rights": ["data", "privacy", "CDR", "consumer data right", "open banking"],
        }

        text_lower = text.lower()
        detected_topics = []

        for topic, keywords in topic_keywords.items():
            if any(kw.lower() in text_lower for kw in keywords):
                detected_topics.append(topic)

        return detected_topics

    def enrich_chunk_metadata(self, content: str, metadata: dict[str, Any]) -> dict[str, Any]:
        """Add extracted entities and topics to chunk metadata."""
        entities = self.extract_entities(content)
        topics = self.extract_topics(content)

        return {
            **metadata,
            "entities": entities,
            "topics": topics,
        }


_extractor: EntityExtractor | None = None


def get_entity_extractor() -> EntityExtractor:
    """Get or create the global entity extractor instance."""
    global _extractor
    if _extractor is None:
        _extractor = EntityExtractor()
    return _extractor
