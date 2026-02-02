"""
Docling-based document extractor for multiple formats.

Uses Docling for robust document extraction including:
- PDFs (text-based and scanned/image with OCR)
- Word documents (DOCX, DOC)
- PowerPoint (PPTX, PPT)
- Excel (XLSX, XLS)
- HTML
- Images (PNG, JPG with OCR)
- Markdown, Text files

Docling advantages:
- Built-in OCR for scanned PDFs (EasyOCR)
- Preserves document structure (tables, sections, hierarchies)
- Consistent markdown output across all formats
- Battle-tested by IBM Research
"""

import os
import sys
import tempfile
from pathlib import Path
from urllib.parse import unquote, urlparse

# Fix Windows symlink issues with HuggingFace model cache
# Must be set BEFORE importing docling/transformers
if sys.platform == "win32":
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    # Force copy mode instead of symlinks for model caching
    os.environ["HF_HUB_LOCAL_DIR_AUTO_SYMLINK_THRESHOLD"] = "0"
    # Disable progress bars that can cause issues
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import httpx

# Import huggingface_hub and patch symlink behavior before Docling loads models
try:
    from huggingface_hub import constants
    # Force the hub to not use symlinks on Windows
    constants.HF_HUB_ENABLE_HF_TRANSFER = False
except ImportError:
    pass

from docling.document_converter import DocumentConverter


class DoclingExtractor:
    """
    Document extractor using Docling for multi-format support.

    Handles PDF (including scanned), DOCX, PPTX, XLSX, HTML, and more.
    Converts all documents to clean, LLM-friendly markdown.

    Example:
        >>> async with DoclingExtractor() as extractor:
        ...     text, doc_type = await extractor.extract(url)
    """

    # Supported file extensions
    SUPPORTED_EXTENSIONS = {
        "pdf": "pdf",
        "docx": "docx",
        "doc": "doc",
        "pptx": "pptx",
        "ppt": "ppt",
        "xlsx": "xlsx",
        "xls": "xls",
        "html": "html",
        "htm": "html",
        "md": "markdown",
        "markdown": "markdown",
        "txt": "text",
        "png": "image",
        "jpg": "image",
        "jpeg": "image",
    }

    def __init__(self, timeout: float = 60.0):
        """
        Initialize DoclingExtractor.

        Args:
            timeout: HTTP timeout for downloading documents.
        """
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=timeout, follow_redirects=True)
        self._converter = None

    @property
    def converter(self) -> DocumentConverter:
        """Lazy-load DocumentConverter (heavy initialization)."""
        if self._converter is None:
            self._converter = DocumentConverter()
        return self._converter

    async def close(self) -> None:
        """Close the HTTP client."""
        await self.client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    def _get_document_type(self, url: str) -> str | None:
        """Determine document type from URL."""
        parsed = urlparse(url)
        path = parsed.path.lower()

        for ext, doc_type in self.SUPPORTED_EXTENSIONS.items():
            if path.endswith(f".{ext}"):
                return doc_type
        return None

    def _get_extension(self, url: str) -> str:
        """Get file extension from URL."""
        parsed = urlparse(url)
        path = parsed.path.lower()
        return Path(path).suffix.lstrip(".")

    async def extract(self, url: str) -> tuple[str, str]:
        """
        Download and extract text from a document URL using Docling.

        Args:
            url: URL of the document to extract.

        Returns:
            Tuple of (extracted_markdown, document_type).

        Raises:
            ValueError: If document type is not supported.
            httpx.HTTPError: If download fails.
        """
        doc_type = self._get_document_type(url)
        if not doc_type:
            raise ValueError(f"Unsupported document type for URL: {url}")

        extension = self._get_extension(url)

        # Download the document
        content = await self._download(url)

        # Save to temporary file for Docling processing
        with tempfile.NamedTemporaryFile(
            suffix=f".{extension}", delete=False
        ) as tmp_file:
            tmp_file.write(content)
            tmp_path = tmp_file.name

        try:
            # Convert using Docling
            result = self.converter.convert(tmp_path)
            markdown = result.document.export_to_markdown()

            if not markdown.strip():
                markdown = "[No text extracted from document]"

            return markdown, doc_type

        except Exception as e:
            return f"[Document extraction error: {e}]", doc_type

        finally:
            # Clean up temporary file
            Path(tmp_path).unlink(missing_ok=True)

    async def extract_local(self, file_path: str | Path) -> tuple[str, str]:
        """
        Extract text from a local document file using Docling.

        Args:
            file_path: Path to the local document file.

        Returns:
            Tuple of (extracted_markdown, document_type).
        """
        file_path = Path(file_path)
        extension = file_path.suffix.lower().lstrip(".")
        doc_type = self.SUPPORTED_EXTENSIONS.get(extension, "unknown")

        try:
            result = self.converter.convert(str(file_path))
            markdown = result.document.export_to_markdown()

            if not markdown.strip():
                markdown = "[No text extracted from document]"

            return markdown, doc_type

        except Exception as e:
            return f"[Document extraction error: {e}]", doc_type

    async def _download(self, url: str) -> bytes:
        """Download document content from URL."""
        response = await self.client.get(url)
        response.raise_for_status()
        return response.content

    def get_filename_from_url(self, url: str) -> str:
        """Extract readable filename from URL."""
        parsed = urlparse(url)
        filename = parsed.path.split("/")[-1]
        return unquote(filename)
