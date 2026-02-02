"""
RAG Data Ingestion Crawler - Main Entry Point.

Usage:
    python -m rag_crawler.main --url https://example.com
    python -m rag_crawler.main --url https://example.com --depth 5 --max-pages 100
"""

import argparse
import asyncio
import io
import os
import sys

# Fix Windows console encoding for Unicode characters (crawl4ai uses Unicode symbols)
if sys.platform == "win32":
    # Set UTF-8 mode for Python
    os.environ.setdefault("PYTHONUTF8", "1")
    # Reconfigure stdout/stderr to use UTF-8 with error replacement
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from rag_crawler.config import CrawlerConfig, ScopeMode
from rag_crawler.crawler import RAGCrawler, safe_print
from rag_crawler.output.markdown import MarkdownGenerator


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="RAG Data Ingestion Crawler - Extract website content for embedding",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--url",
        "-u",
        type=str,
        required=True,
        help="Target website URL to crawl",
    )

    parser.add_argument(
        "--scope",
        "-s",
        type=str,
        choices=["domain_only", "subpath_only", "allow_list"],
        default="domain_only",
        help="Crawling scope mode",
    )

    parser.add_argument(
        "--depth",
        "-d",
        type=int,
        default=3,
        help="Maximum crawl depth",
    )

    parser.add_argument(
        "--max-pages",
        "-m",
        type=int,
        default=200,
        help="Maximum number of pages to crawl",
    )

    parser.add_argument(
        "--concurrent",
        "-c",
        type=int,
        default=10,
        help="Maximum concurrent browser sessions",
    )

    parser.add_argument(
        "--no-documents",
        action="store_true",
        help="Skip document extraction (PDF, DOCX, etc.)",
    )

    parser.add_argument(
        "--render-js",
        action="store_true",
        help="Render JavaScript (slower, for JS-heavy sites)",
    )

    parser.add_argument(
        "--rate-limit",
        type=int,
        default=300,
        help="Delay between requests in milliseconds",
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="./output",
        help="Directory to save output files",
    )

    parser.add_argument(
        "--save-individual",
        action="store_true",
        help="Save each page as individual markdown files",
    )

    return parser.parse_args()


async def run_crawler(args: argparse.Namespace) -> int:
    """Run the crawler with given arguments."""
    # Create config from arguments
    config = CrawlerConfig(
        target_url=args.url,
        scope_mode=ScopeMode(args.scope),
        max_depth=args.depth,
        max_pages=args.max_pages,
        max_concurrent=args.concurrent,
        include_documents=not args.no_documents,
        render_js=args.render_js,
        rate_limit_ms=args.rate_limit,
        output_dir=args.output_dir,
    )

    # Initialize crawler
    crawler = RAGCrawler(config)

    # Execute crawl
    print("=" * 60)
    print("RAG Data Ingestion Crawler")
    print("=" * 60)
    print(f"Target URL: {config.target_url}")
    print(f"Scope: {config.scope_mode.value}")
    print(f"Max depth: {config.max_depth}")
    print(f"Max pages: {config.max_pages}")
    print(f"Include documents: {config.include_documents}")
    print("=" * 60)

    try:
        report = await crawler.crawl()
    except Exception as e:
        safe_print(f"\nCrawl failed with error: {e}")
        return 1

    # Generate output
    generator = MarkdownGenerator(output_dir=config.output_dir)

    # Save main report
    report_path = generator.save_report(report)

    # Optionally save individual pages
    if args.save_individual:
        generator.save_individual_pages(report)

    print("\n" + "=" * 60)
    print("Crawl Summary")
    print("=" * 60)
    print(f"Pages crawled: {len(report.results)}")
    print(f"Pages skipped: {len(report.skipped)}")
    print(f"Peak memory: {report.peak_memory_mb} MB")
    print(f"Report saved: {report_path}")
    print("=" * 60)

    return 0


def main() -> int:
    """Main entry point."""
    args = parse_args()
    return asyncio.run(run_crawler(args))


if __name__ == "__main__":
    sys.exit(main())
