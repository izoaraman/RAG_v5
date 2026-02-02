"""
Core RAG Crawler module using Crawl4AI.

Based on patterns from crawl4AI-examples:
- 5-crawl_site_recursively.py for recursive crawling
- 3-crawl_sitemap_in_parallel.py for parallel batch crawling
"""

import asyncio
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from urllib.parse import urldefrag

import psutil
from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CacheMode,
    CrawlerRunConfig,
    MemoryAdaptiveDispatcher,
)

from rag_crawler.config import CrawlerConfig


def safe_print(text: str) -> None:
    """Print text safely, handling Unicode encoding errors on Windows."""
    try:
        print(text)
    except UnicodeEncodeError:
        # Replace problematic characters for Windows console
        safe_text = text.encode(sys.stdout.encoding, errors="replace").decode(
            sys.stdout.encoding
        )
        print(safe_text)


@dataclass
class CrawlResult:
    """Result of crawling a single page or document."""

    url: str
    title: str
    content_type: str  # "page" | "pdf" | "docx" | etc.
    markdown: str
    retrieved_at: datetime
    extraction_quality: str  # "full" | "partial" | "failed"
    error_message: str | None = None
    links_found: int = 0


@dataclass
class CrawlReport:
    """Complete crawl report for RAG output."""

    target_url: str
    scope_mode: str
    max_depth: int
    documents_included: bool
    started_at: datetime
    completed_at: datetime | None = None
    results: list[CrawlResult] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)  # (url, reason)
    peak_memory_mb: int = 0


class RAGCrawler:
    """
    RAG Data Ingestion Crawler using Crawl4AI.

    Recursively crawls a website and extracts clean markdown for embedding.
    Based on patterns from crawl4AI-examples.

    Example:
        >>> config = CrawlerConfig(target_url="https://example.com")
        >>> crawler = RAGCrawler(config)
        >>> report = await crawler.crawl()
    """

    def __init__(self, config: CrawlerConfig):
        self.config = config
        self.visited: set[str] = set()
        self.document_urls: set[str] = set()
        self.peak_memory: int = 0
        self.process = psutil.Process(os.getpid())

    def _normalize_url(self, url: str) -> str:
        """Remove fragment (part after #) from URL."""
        return urldefrag(url)[0]

    def _log_memory(self, prefix: str = "") -> None:
        """Track peak memory usage for observability."""
        current_mem = self.process.memory_info().rss
        if current_mem > self.peak_memory:
            self.peak_memory = current_mem
        safe_print(
            f"{prefix} Memory: {current_mem // (1024 * 1024)} MB, "
            f"Peak: {self.peak_memory // (1024 * 1024)} MB"
        )

    def _get_markdown(self, result) -> str:
        """Extract markdown content from crawl result, handling different API versions."""
        if result.markdown is None:
            return ""
        # Handle both string and object with raw_markdown attribute
        if isinstance(result.markdown, str):
            return result.markdown
        if hasattr(result.markdown, "raw_markdown"):
            return result.markdown.raw_markdown or ""
        return str(result.markdown)

    def _create_browser_config(self) -> BrowserConfig:
        """Create BrowserConfig from crawler settings."""
        return BrowserConfig(
            headless=self.config.headless,
            verbose=self.config.verbose,
            extra_args=self.config.extra_args,
        )

    def _create_run_config(self) -> CrawlerRunConfig:
        """Create CrawlerRunConfig for crawling."""
        return CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            stream=False,
        )

    def _create_dispatcher(self) -> MemoryAdaptiveDispatcher:
        """Create MemoryAdaptiveDispatcher for resource management."""
        return MemoryAdaptiveDispatcher(
            memory_threshold_percent=self.config.memory_threshold_percent,
            check_interval=1.0,
            max_session_permit=self.config.max_concurrent,
        )

    def _filter_urls(self, urls: list[str]) -> list[str]:
        """Filter URLs based on scope and exclusion rules."""
        filtered = []
        for url in urls:
            normalized = self._normalize_url(url)

            # Skip already visited
            if normalized in self.visited:
                continue

            # Check scope
            if not self.config.is_url_in_scope(normalized):
                continue

            # Check exclusions
            if self.config.is_excluded_path(normalized):
                continue

            filtered.append(normalized)

        return filtered

    def _extract_links(self, result) -> tuple[list[str], list[str]]:
        """
        Extract internal links and document URLs from crawl result.

        Returns:
            Tuple of (page_urls, document_urls)
        """
        page_urls = []
        doc_urls = []

        internal_links = result.links.get("internal", [])
        for link in internal_links:
            href = link.get("href", "")
            if not href:
                continue

            normalized = self._normalize_url(href)

            if self.config.is_document_url(normalized):
                doc_urls.append(normalized)
            else:
                page_urls.append(normalized)

        return page_urls, doc_urls

    def _extract_title(self, result) -> str:
        """Extract page title from crawl result."""
        if hasattr(result, "metadata") and result.metadata:
            return result.metadata.get("title", "Untitled")
        return "Untitled"

    async def crawl(self) -> CrawlReport:
        """
        Perform recursive crawl starting from target URL.

        Returns:
            CrawlReport with all extracted content.
        """
        report = CrawlReport(
            target_url=str(self.config.target_url),
            scope_mode=self.config.scope_mode.value,
            max_depth=self.config.max_depth,
            documents_included=self.config.include_documents,
            started_at=datetime.now(timezone.utc),
        )

        browser_config = self._create_browser_config()
        run_config = self._create_run_config()
        dispatcher = self._create_dispatcher()

        start_url = self._normalize_url(str(self.config.target_url))
        current_urls = {start_url}

        safe_print(f"\n=== RAG Crawler Starting ===")
        safe_print(f"Target: {start_url}")
        safe_print(f"Max depth: {self.config.max_depth}")
        safe_print(f"Max pages: {self.config.max_pages}")
        self._log_memory("Initial")

        async with AsyncWebCrawler(config=browser_config) as crawler:
            for depth in range(self.config.max_depth):
                # Check page limit
                if len(report.results) >= self.config.max_pages:
                    safe_print(f"Reached max pages limit ({self.config.max_pages})")
                    break

                # Filter URLs to crawl
                urls_to_crawl = self._filter_urls(list(current_urls))

                if not urls_to_crawl:
                    safe_print(f"No more URLs to crawl at depth {depth + 1}")
                    break

                # Limit to remaining page budget
                remaining = self.config.max_pages - len(report.results)
                urls_to_crawl = urls_to_crawl[:remaining]

                safe_print(f"\n=== Depth {depth + 1}: Crawling {len(urls_to_crawl)} URLs ===")

                # Rate limiting delay
                if self.config.rate_limit_ms > 0:
                    await asyncio.sleep(self.config.rate_limit_ms / 1000)

                # Batch crawl all URLs at this depth
                results = await crawler.arun_many(
                    urls=urls_to_crawl,
                    config=run_config,
                    dispatcher=dispatcher,
                )

                next_level_urls = set()

                for result in results:
                    normalized_url = self._normalize_url(result.url)
                    self.visited.add(normalized_url)

                    if result.success:
                        markdown = self._get_markdown(result)
                        title = self._extract_title(result)
                        page_links, doc_links = self._extract_links(result)

                        crawl_result = CrawlResult(
                            url=normalized_url,
                            title=title,
                            content_type="page",
                            markdown=markdown,
                            retrieved_at=datetime.now(timezone.utc),
                            extraction_quality="full" if len(markdown) > 100 else "partial",
                            links_found=len(page_links) + len(doc_links),
                        )
                        report.results.append(crawl_result)

                        # Collect document URLs
                        if self.config.include_documents:
                            self.document_urls.update(doc_links)

                        # Collect next level URLs
                        for link in page_links:
                            if link not in self.visited:
                                next_level_urls.add(link)

                        safe_print(
                            f"[OK] {normalized_url} | "
                            f"Markdown: {len(markdown)} chars | "
                            f"Links: {len(page_links)} pages, {len(doc_links)} docs"
                        )
                    else:
                        error_msg = str(result.error_message or "Unknown error")
                        report.skipped.append((normalized_url, error_msg))
                        safe_print(f"[ERROR] {normalized_url}: {error_msg}")

                current_urls = next_level_urls
                self._log_memory(f"After depth {depth + 1}")

            # Crawl discovered documents
            if self.config.include_documents and self.document_urls:
                await self._crawl_documents(crawler, run_config, dispatcher, report)

        report.completed_at = datetime.now(timezone.utc)
        report.peak_memory_mb = self.peak_memory // (1024 * 1024)

        safe_print(f"\n=== Crawl Complete ===")
        safe_print(f"Pages crawled: {len(report.results)}")
        safe_print(f"Skipped: {len(report.skipped)}")
        safe_print(f"Peak memory: {report.peak_memory_mb} MB")

        return report

    async def _crawl_documents(
        self,
        crawler: AsyncWebCrawler,
        run_config: CrawlerRunConfig,
        dispatcher: MemoryAdaptiveDispatcher,
        report: CrawlReport,
    ) -> None:
        """
        Extract content from discovered document URLs using Docling.

        Uses DoclingExtractor for robust document extraction including:
        - PDF (text-based and scanned/image with OCR)
        - DOCX, PPTX, XLSX
        - HTML, Images

        Docling handles complex documents that pypdf/python-docx cannot.
        """
        from rag_crawler.extractors.docling_extractor import DoclingExtractor

        doc_urls = [url for url in self.document_urls if url not in self.visited]

        if not doc_urls:
            return

        safe_print(f"\n=== Extracting {len(doc_urls)} Documents (Docling) ===")

        async with DoclingExtractor() as extractor:
            for url in doc_urls:
                normalized_url = self._normalize_url(url)
                self.visited.add(normalized_url)

                try:
                    markdown, doc_type = await extractor.extract(url)
                    title = extractor.get_filename_from_url(url)

                    crawl_result = CrawlResult(
                        url=normalized_url,
                        title=title,
                        content_type=doc_type,
                        markdown=markdown,
                        retrieved_at=datetime.now(timezone.utc),
                        extraction_quality="full" if len(markdown) > 100 else "partial",
                    )
                    report.results.append(crawl_result)
                    safe_print(f"[DOC] {title} | Type: {doc_type} | {len(markdown)} chars")
                except Exception as e:
                    error_msg = str(e)
                    report.skipped.append((normalized_url, error_msg))
                    safe_print(f"[ERROR] {normalized_url}: {error_msg}")
