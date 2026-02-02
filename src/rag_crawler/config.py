"""
Configuration module for RAG Data Ingestion Crawler.

Based on patterns from crawl4AI-examples.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal
from urllib.parse import urlparse


class ScopeMode(str, Enum):
    """Crawling scope mode."""

    DOMAIN_ONLY = "domain_only"
    SUBPATH_ONLY = "subpath_only"
    ALLOW_LIST = "allow_list"


@dataclass
class CrawlerConfig:
    """
    User-configurable settings for the RAG Data Ingestion Crawler.

    Based on crawl4ai patterns with BrowserConfig and CrawlerRunConfig.
    """

    # Target configuration
    target_url: str
    scope_mode: ScopeMode = ScopeMode.DOMAIN_ONLY

    # Crawl limits
    max_depth: int = 3
    max_pages: int = 200
    max_concurrent: int = 10

    # Document extraction
    include_documents: bool = True
    document_types: list[str] = field(
        default_factory=lambda: [
            "pdf",
            "docx",
            "doc",
            "rtf",
            "txt",
            "pptx",
            "ppt",
            "xlsx",
            "xls",
            "csv",
            "odt",
            "odp",
            "ods",
        ]
    )

    # URL filtering
    exclude_paths: list[str] = field(
        default_factory=lambda: [
            "/cart",
            "/checkout",
            "/login",
            "/account",
            "/wp-admin",
            "/admin",
            "/api",
            "/search",
        ]
    )

    # Crawl behavior
    respect_robots: bool = True
    rate_limit_ms: int = 300
    render_js: bool = False

    # Resource management (for MemoryAdaptiveDispatcher)
    memory_threshold_percent: float = 70.0

    # Output
    output_dir: str = "./output"

    # Browser config options (passed to BrowserConfig)
    headless: bool = True
    verbose: bool = False
    extra_args: list[str] = field(
        default_factory=lambda: ["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"]
    )

    def get_domain(self) -> str:
        """Extract domain from target URL."""
        parsed = urlparse(self.target_url)
        return parsed.netloc

    def get_base_path(self) -> str:
        """Extract base path from target URL for subpath_only mode."""
        parsed = urlparse(self.target_url)
        return parsed.path if parsed.path else "/"

    def is_url_in_scope(self, url: str) -> bool:
        """Check if a URL is within the configured crawl scope."""
        parsed = urlparse(url)

        if self.scope_mode == ScopeMode.DOMAIN_ONLY:
            return parsed.netloc == self.get_domain()
        elif self.scope_mode == ScopeMode.SUBPATH_ONLY:
            return (
                parsed.netloc == self.get_domain()
                and parsed.path.startswith(self.get_base_path())
            )
        return True

    def is_excluded_path(self, url: str) -> bool:
        """Check if URL path matches any excluded patterns."""
        parsed = urlparse(url)
        path_lower = parsed.path.lower()
        return any(excluded.lower() in path_lower for excluded in self.exclude_paths)

    def is_document_url(self, url: str) -> bool:
        """Check if URL points to a downloadable document."""
        parsed = urlparse(url)
        path_lower = parsed.path.lower()
        return any(path_lower.endswith(f".{ext}") for ext in self.document_types)
