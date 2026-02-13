"""
RAG_v5 Streamlit Application.

Agentic RAG chat interface matching RAG_v4 UI with:
- Ask Mode: Current documents / New document
- URL Crawler integration
- Document upload
- Sources tab with clickable citations
"""

import asyncio
import base64
import html
import json
import logging
import os
import re
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path

# Windows UTF-8 encoding fix
if os.name == "nt":
    try:
        import codecs

        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")
        sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, "strict")
    except (AttributeError, TypeError):
        pass

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ---------- PATHS ----------
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
DOCUMENTS_DIR = ROOT / "documents"
NEW_DOC_DIR = ROOT / "data" / "new_uploads"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
NEW_DOC_DIR.mkdir(parents=True, exist_ok=True)

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Knowledge Assistant",
    layout="wide",
    page_icon="",
    initial_sidebar_state="expanded",
)


# ---------- THEMED AVATARS (matching RAG_v4) ----------
def _svg_circle_data_uri(color_hex: str) -> str:
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10" fill="{color_hex}"/></svg>'
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("ascii")


AVATAR_USER = _svg_circle_data_uri("#1CCFC9")  # Opal
AVATAR_ASSISTANT = _svg_circle_data_uri("#342D8C")  # Violet

# ---------- CSS STYLING (from RAG_v4) ----------
CUSTOM_CSS = """
<style>
:root {
    --violet: #342D8C;
    --opal: #1CCFC9;
    --navy: #131838;
}

html, body, [data-testid="stAppViewContainer"], .main {background: #ffffff;}
.main {padding-top: 0.5rem !important;}
.element-container {margin-bottom: 0.4rem !important;}
[data-testid="stSidebar"] {padding: 0.75rem 0.75rem 0.5rem !important;}
[data-testid="stSidebar"] h2 {font-size: 1.05rem; margin: 0.25rem 0 0.5rem 0; color: var(--navy);}
[data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4 {font-size: 0.95rem; margin: 0.25rem 0 0.5rem 0; color: var(--navy);}

.app-header {background: var(--violet); padding: 0.6rem 0.8rem; border-radius: 10px;}
.app-header h1 {color: #fff; margin: 0; font-weight: 700; font-size: 1.4rem;}

.answer-card {
    padding: 0.2rem 0;
    color: #1a1a2e;
    font-size: 15px;
    line-height: 1.7;
}
.answer-card p { margin: 0 0 0.6em 0; }
.answer-card ul, .answer-card ol { margin: 0.3em 0 0.6em 1.4em; padding: 0; }
.answer-card li { margin-bottom: 0.25em; }
.answer-card strong { font-weight: 600; }
.answer-card h1, .answer-card h2, .answer-card h3 {
    margin: 0.8em 0 0.3em 0;
    color: var(--navy);
    font-weight: 700;
}
.answer-card h3 { font-size: 1.05em; }
.sources-card {padding: 0.9rem; border-radius: 10px; border: 1px solid #CBEFED; background: #F7FFFE;}

.citation-link {
    color: #1f77b4 !important;
    text-decoration: none !important;
    font-weight: 500;
    font-size: 0.75em !important;
    cursor: pointer !important;
    border-bottom: 1px dotted #1f77b4;
    padding: 0px 1px;
    border-radius: 2px;
    background-color: rgba(31, 119, 180, 0.08);
}
.citation-link:hover {
    color: #0d5aa7 !important;
    background-color: rgba(31, 119, 180, 0.15) !important;
}

.source-card {
    background: linear-gradient(135deg, #ffffff 0%, #f8fffe 100%);
    border: 2px solid #e6f9f7;
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 20px;
    transition: all 0.3s ease;
    box-shadow: 0 2px 4px rgba(28, 207, 201, 0.1);
}
.source-card:hover {
    box-shadow: 0 4px 12px rgba(28, 207, 201, 0.2);
    border-color: #1CCFC9;
    transform: translateY(-2px);
}
.source-header {
    display: flex;
    align-items: center;
    margin-bottom: 12px;
    border-bottom: 1px solid #e6f9f7;
    padding-bottom: 8px;
}
.source-filename {
    color: #131838;
    font-weight: 700;
    font-size: 17px;
    flex-grow: 1;
}
.source-page {
    background: linear-gradient(135deg, #1CCFC9 0%, #17b0aa 100%);
    color: white;
    font-weight: 600;
    font-size: 13px;
    padding: 4px 10px;
    border-radius: 20px;
    margin-left: 12px;
}
.source-snippet {
    background: #f7fffe;
    color: #4a5168;
    font-size: 14px;
    line-height: 1.6;
    border-left: 4px solid #1CCFC9;
    padding: 12px 16px;
    margin-top: 12px;
    border-radius: 4px;
}
.source-detail-number {
    display: inline-block;
    color: #9ca3af;
    font-weight: 500;
    font-size: 13px;
    margin-right: 8px;
}

.stButton button:not([kind="tertiary"]), button[kind="primary"], [data-testid="baseButton-primary"] {
    background: var(--opal) !important;
    color: var(--navy) !important;
    border: 1px solid var(--opal) !important;
}
.stButton button:not([kind="tertiary"]):hover, button[kind="primary"]:hover {
    background: #18bab4 !important;
    border-color: var(--violet) !important;
}
button[kind="secondary"] {
    background: #f5f7fb !important;
    color: var(--navy) !important;
    border: 1px solid #e6e9f2 !important;
}

.stSlider [role="slider"] {
    background-color: var(--opal) !important;
}

.sources-list {
    font-family: monospace;
    font-size: 12px;
    line-height: 1.6;
    padding: 12px;
    background: #f9fafb;
    border-radius: 8px;
    max-height: 500px;
    overflow-y: auto;
}
.source-item {
    margin-bottom: 6px;
    color: #374151;
    font-size: 12px;
}
.source-number {
    color: #6b7280;
    font-weight: bold;
}
/* Sources button — light grey */
[data-testid="stBaseButton-tertiary"] {
    color: #b0b0b0 !important;
    font-size: 0.8rem !important;
    background: transparent !important;
    border: none !important;
    padding: 2px 0 !important;
    box-shadow: none !important;
}
[data-testid="stBaseButton-tertiary"]:hover {
    color: #888 !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------- SESSION STATE ----------
def init_session_state():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "rag_option" not in st.session_state:
        st.session_state.rag_option = "Current documents"
    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.2
    if "debug_mode" not in st.session_state:
        st.session_state.debug_mode = False
    if "uploaded_file_names" not in st.session_state:
        st.session_state.uploaded_file_names = []
    # Background crawler state
    if "crawler_process" not in st.session_state:
        st.session_state.crawler_process = None  # subprocess.Popen handle
    if "crawler_start_time" not in st.session_state:
        st.session_state.crawler_start_time = None  # datetime when crawl started
    if "crawler_url" not in st.session_state:
        st.session_state.crawler_url = ""  # URL being crawled
    if "crawler_status" not in st.session_state:
        st.session_state.crawler_status = "idle"  # idle | crawling | ingesting | done | error
    if "crawler_message" not in st.session_state:
        st.session_state.crawler_message = ""  # Status/error message
    if "crawler_timeout" not in st.session_state:
        st.session_state.crawler_timeout = 600  # 10 minutes default
    # Background ingestion state (for crawl results)
    if "ingest_process" not in st.session_state:
        st.session_state.ingest_process = None  # subprocess.Popen handle for ingestion


init_session_state()


# ---------- RAG ROUTER ----------
@st.cache_resource
def get_rag_router():
    try:
        from rag_crawler.router.router_graph import RAGRouter

        return RAGRouter()
    except Exception as e:
        logger.error(f"Failed to initialize RAG router: {e}")
        return None


def process_query(query: str, temperature: float = 0.0, source_filter: str | None = None) -> dict:
    router = get_rag_router()
    if router is None:
        return {
            "response": "RAG system is not available. Please check configuration and database connection.",
            "sources": [],
            "agent_used": "error",
            "error": "Router initialization failed",
        }
    try:
        result = router.process_query_sync(
            query=query,
            session_id=st.session_state.session_id,
            source_filter=source_filter,
            temperature=temperature,
        )
        return result
    except Exception as e:
        logger.error(f"Query processing error: {e}")
        return {
            "response": f"Error processing query: {str(e)}",
            "sources": [],
            "agent_used": "error",
            "error": str(e),
        }


# ---------- CITATION & SOURCE RENDERING ----------
def make_citations_clickable(text: str) -> str:
    """Inject clickable citation links into response text.

    Keeps the original markdown intact so Streamlit renders it natively.
    Only citation brackets like [1], [2-3] become HTML links.
    Bold markers (**text**) are stripped for a clean, uniform look.
    """
    # Strip bold markers
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)

    citation_pattern = r"\[(\d+(?:-\d+)?(?:\]\[\d+(?:-\d+)?)*)\]"

    def replace_citation(match):
        citation_text = match.group(0)
        numbers = re.findall(r"\d+", citation_text)
        if numbers:
            first_number = numbers[0]
            return f'<a class="citation-link" title="Source {first_number}">{citation_text}</a>'
        return citation_text

    return re.sub(citation_pattern, replace_citation, text)


def _resolve_source_file(doc_source: str) -> Path | None:
    """Resolve a document source path to its file on disk."""
    if not doc_source or doc_source.startswith("http"):
        return None

    # New uploads: source = "new_uploads/filename.ext"
    if doc_source.startswith("new_uploads/"):
        filename = doc_source[len("new_uploads/") :]
        file_path = NEW_DOC_DIR / filename
        if file_path.exists():
            return file_path

    # Current documents: source = relative path
    file_path = DOCUMENTS_DIR / doc_source
    if file_path.exists():
        return file_path

    # Fallback: try from project root
    file_path = ROOT / doc_source
    if file_path.exists():
        return file_path

    return None


def _format_page(page) -> str:
    """Format page number(s) for display."""
    if page is None or page == "N/A":
        return "N/A"
    if isinstance(page, list):
        if len(page) == 0:
            return "N/A"
        if len(page) == 1:
            return str(page[0])
        return f"{page[0]}-{page[-1]}"
    return str(page)


def render_sources_detailed(sources: list, key_prefix: str = "detail"):
    """Render sources in detailed card format with links to originals."""
    if not sources:
        st.info("No source documents found for this query.")
        return

    for i, source in enumerate(sources, 1):
        if isinstance(source, dict):
            content = source.get("content", source.get("text", ""))
            doc_title = source.get("document_title", source.get("title", f"Source {i}"))
            doc_source = source.get("document_source", source.get("source", ""))
            score = source.get("score", 0)
            chunk_index = source.get("chunk_index", 0)
            page = source.get("page", source.get("metadata", {}).get("page"))
        else:
            content = getattr(source, "content", str(source))
            doc_title = getattr(source, "document_title", f"Source {i}")
            doc_source = getattr(source, "document_source", "")
            score = getattr(source, "score", 0)
            chunk_index = getattr(source, "chunk_index", 0)
            page = getattr(source, "page", None)

        snippet = content[:500] + "..." if len(content) > 500 else content
        is_url = doc_source.startswith("http") if doc_source else False

        # Build title HTML — clickable link for URLs, plain text for files
        if is_url:
            title_html = (
                f'<a href="{html.escape(doc_source)}" target="_blank" '
                f'style="color: #131838; text-decoration: none; '
                f'border-bottom: 2px solid #1CCFC9;">'
                f"{html.escape(doc_title)}</a>"
            )
            source_link = (
                f'<a href="{html.escape(doc_source)}" target="_blank" '
                f'style="color: #7a8199; word-break: break-all;">'
                f"{html.escape(doc_source[:100])}</a>"
            )
        else:
            title_html = html.escape(doc_title)
            source_link = html.escape(doc_source[:80]) if doc_source else "N/A"

        source_html = f"""
        <div class="source-card" id="source-{i}">
            <div class="source-header">
                <span class="source-detail-number">[{i}]</span>
                <span class="source-filename">{title_html}</span>
                <span class="source-page">Page: {_format_page(page)}</span>
            </div>
            <div style="color: #7a8199; font-size: 13px; margin-bottom: 10px;">
                Chunk: {chunk_index} | Source: {source_link}
            </div>
            <div class="source-snippet">{html.escape(snippet)}</div>
        </div>
        """
        st.markdown(source_html, unsafe_allow_html=True)

        # Add download button for file-based sources (PDF, DOCX, etc.)
        if not is_url and doc_source:
            file_path = _resolve_source_file(doc_source)
            if file_path and file_path.exists():
                with open(file_path, "rb") as f:
                    file_data = f.read()
                st.download_button(
                    label=f"Download {file_path.name}",
                    data=file_data,
                    file_name=file_path.name,
                    key=f"download_{key_prefix}_{i}",
                )


def render_sources_inline(sources: list, button_key: str):
    """Brief unique-document list with a muted 'View details' button."""
    if not sources:
        return

    # Deduplicate by document title, preserving order
    seen = set()
    unique_titles = []
    for src in sources:
        title = (
            src.get("document_title", src.get("title", ""))
            if isinstance(src, dict)
            else getattr(src, "document_title", "")
        )
        if title and title not in seen:
            seen.add(title)
            unique_titles.append(title)

    if not unique_titles:
        return

    items = "".join(
        f'<div class="source-item"><span class="source-number">{i}.</span> {html.escape(t)}</div>'
        for i, t in enumerate(unique_titles, 1)
    )
    st.markdown(
        f'<div class="sources-list" style="padding:8px 12px;margin-top:6px;">{items}</div>',
        unsafe_allow_html=True,
    )

    if st.button(
        "Sources",
        key=button_key,
        type="tertiary",
    ):
        show_sources_dialog(sources)


@st.dialog("Document Sources", width="large")
def show_sources_dialog(sources):
    """Show detailed sources in a native Streamlit dialog."""
    if not sources:
        st.info("No source documents found.")
        return
    render_sources_detailed(sources)


# ---------- DOCUMENT & CRAWLER FUNCTIONS ----------
def handle_document_upload(uploaded_files, save_dir: Path) -> list:
    if not uploaded_files:
        return []

    saved_files = []
    for uploaded_file in uploaded_files:
        output_path = save_dir / uploaded_file.name
        with open(output_path, "wb") as f:
            f.write(uploaded_file.read())
        saved_files.append(uploaded_file.name)
    return saved_files


def run_ingestion(
    documents_dir: str | None = None,
    source_prefix: str = "",
    skip_summary: bool = False,
) -> tuple[bool, str]:
    """Run document ingestion.

    Args:
        documents_dir: Directory containing documents.
        source_prefix: Prefix to add to source paths (e.g., "new_uploads/").
        skip_summary: Skip LLM summary generation for faster upload.
    """
    try:
        cmd = ["uv", "run", "rag-ingest"]
        if documents_dir:
            cmd.extend(["--documents", documents_dir])
        else:
            cmd.extend(["--documents", str(DOCUMENTS_DIR)])

        if source_prefix:
            cmd.extend(["--source-prefix", source_prefix])

        if skip_summary:
            cmd.append("--skip-summary")

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT), timeout=600)
        if result.returncode == 0:
            return True, "Documents ingested successfully!"
        else:
            return False, f"Ingestion failed: {result.stderr[:200]}"
    except subprocess.TimeoutExpired:
        return False, "Ingestion timed out after 10 minutes. Try uploading a smaller document."
    except Exception as e:
        return False, f"Error: {str(e)}"


def start_crawler_background(url: str, max_pages: int = 10) -> None:
    """Start the RAG crawler as a non-blocking background process.

    Updates st.session_state with process handle and status.
    """
    output_dir = ROOT / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Record existing reports to detect new ones later
    st.session_state._crawler_existing_reports = set(
        str(p) for p in output_dir.glob("crawl_report_*.json")
    )

    cmd = ["uv", "run", "rag-crawler", "--url", url, "--max-pages", str(max_pages)]
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(ROOT),
        )
        st.session_state.crawler_process = process
        st.session_state.crawler_start_time = datetime.now()
        st.session_state.crawler_url = url
        st.session_state.crawler_status = "crawling"
        st.session_state.crawler_message = ""
    except Exception as e:
        st.session_state.crawler_status = "error"
        st.session_state.crawler_message = f"Failed to start crawler: {e}"


def start_ingestion_background(report_path: str) -> None:
    """Start crawl ingestion as a non-blocking background process."""
    cmd = ["uv", "run", "rag-ingest", "--crawl-report", report_path]
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(ROOT),
        )
        st.session_state.ingest_process = process
        st.session_state.crawler_status = "ingesting"
        st.session_state.crawler_message = ""
    except Exception as e:
        st.session_state.crawler_status = "error"
        st.session_state.crawler_message = f"Failed to start ingestion: {e}"


def _find_new_crawl_report() -> str | None:
    """Find the new crawl report created by the background crawler."""
    output_dir = ROOT / "output"
    existing = st.session_state.get("_crawler_existing_reports", set())
    current_reports = set(str(p) for p in output_dir.glob("crawl_report_*.json"))
    new_reports = current_reports - existing
    if new_reports:
        # Return the most recently modified new report
        return max(new_reports, key=lambda p: Path(p).stat().st_mtime)
    # Fallback: most recent report
    all_reports = list(output_dir.glob("crawl_report_*.json"))
    if all_reports:
        return str(max(all_reports, key=lambda p: p.stat().st_mtime))
    return None


def poll_crawler_status() -> None:
    """Poll background crawler/ingestion processes. Call on every Streamlit rerun.

    Handles transitions: crawling → ingesting → done, plus timeout enforcement.
    """
    status = st.session_state.crawler_status

    # --- Phase 1: Crawling ---
    if status == "crawling" and st.session_state.crawler_process is not None:
        proc = st.session_state.crawler_process
        elapsed = (datetime.now() - st.session_state.crawler_start_time).total_seconds()

        # Check timeout
        if elapsed > st.session_state.crawler_timeout:
            proc.kill()
            proc.wait()
            timeout_min = st.session_state.crawler_timeout // 60
            st.session_state.crawler_process = None
            st.session_state.crawler_status = "error"
            st.session_state.crawler_message = (
                f"Crawler timed out after {timeout_min} minutes. "
                f"Try reducing 'Max Pages' or using a more specific URL."
            )
            return

        # Check if process finished
        retcode = proc.poll()
        if retcode is not None:
            st.session_state.crawler_process = None
            if retcode == 0:
                # Crawl succeeded → find report → start ingestion
                report_path = _find_new_crawl_report()
                if report_path:
                    start_ingestion_background(report_path)
                else:
                    st.session_state.crawler_status = "error"
                    st.session_state.crawler_message = "Crawl completed but no report file found."
            else:
                stderr = proc.stderr.read() if proc.stderr else ""
                st.session_state.crawler_status = "error"
                st.session_state.crawler_message = (
                    f"Crawler failed: {stderr[:300]}" if stderr else "Crawler failed"
                )

    # --- Phase 2: Ingesting ---
    elif status == "ingesting" and st.session_state.ingest_process is not None:
        proc = st.session_state.ingest_process
        elapsed = (datetime.now() - st.session_state.crawler_start_time).total_seconds()

        # Ingestion timeout (same as crawler timeout)
        if elapsed > st.session_state.crawler_timeout + 600:  # extra 10 min for ingestion
            proc.kill()
            proc.wait()
            st.session_state.ingest_process = None
            st.session_state.crawler_status = "error"
            st.session_state.crawler_message = "Ingestion timed out."
            return

        retcode = proc.poll()
        if retcode is not None:
            st.session_state.ingest_process = None
            if retcode == 0:
                st.session_state.crawler_status = "done"
                st.session_state.crawler_message = (
                    f"Successfully crawled and ingested {st.session_state.crawler_url}"
                )
                # Auto-switch to "Current documents" so crawled content is queryable
                if st.session_state.rag_option != "Current documents":
                    st.session_state.rag_option = "Current documents"
            else:
                stderr = proc.stderr.read() if proc.stderr else ""
                st.session_state.crawler_status = "error"
                st.session_state.crawler_message = (
                    f"Ingestion failed: {stderr[:300]}" if stderr else "Ingestion failed"
                )


def stop_crawler() -> None:
    """Kill the background crawler/ingestion process."""
    for key in ("crawler_process", "ingest_process"):
        proc = st.session_state.get(key)
        if proc is not None:
            try:
                proc.kill()
                proc.wait()
            except Exception:
                pass
            st.session_state[key] = None

    st.session_state.crawler_status = "idle"
    st.session_state.crawler_start_time = None
    st.session_state.crawler_url = ""
    st.session_state.crawler_message = ""


# ---------- CONFIG CHECK ----------
def check_configuration() -> tuple[bool, list[str]]:
    issues = []
    if not os.getenv("AZURE_OPENAI_API_KEY"):
        issues.append("AZURE_OPENAI_API_KEY is not set")
    if not os.getenv("AZURE_OPENAI_ENDPOINT"):
        issues.append("AZURE_OPENAI_ENDPOINT is not set")
    if not os.getenv("DATABASE_URL"):
        issues.append("DATABASE_URL is not set")
    return len(issues) == 0, issues


def check_database_connection() -> tuple[bool, str]:
    try:
        import socket
        from urllib.parse import urlparse

        db_url = os.getenv("DATABASE_URL", "")
        if not db_url:
            return False, "DATABASE_URL not set"

        parsed = urlparse(db_url)
        host = parsed.hostname
        port = parsed.port or 5432

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex((host, port))
        sock.close()

        if result == 0:
            return True, "Database connection OK"
        else:
            return False, "Cannot connect to database. Add your IP to Azure PostgreSQL firewall."
    except Exception as e:
        return False, f"Database check failed: {str(e)}"


# ---------- CRAWLER STATUS FRAGMENT ----------
@st.fragment(run_every=5)
def _render_crawler_status():
    """Auto-refreshing fragment that shows crawler status and handles transitions."""
    # Re-poll on each fragment refresh
    poll_crawler_status()

    status = st.session_state.crawler_status
    crawler_active = status in ("crawling", "ingesting")

    if crawler_active and st.session_state.crawler_start_time:
        elapsed = (datetime.now() - st.session_state.crawler_start_time).total_seconds()
        elapsed_min = int(elapsed // 60)
        elapsed_sec = int(elapsed % 60)
        phase = "Crawling" if status == "crawling" else "Ingesting"
        timeout_min = st.session_state.crawler_timeout // 60
        st.info(
            f"**{phase}** {st.session_state.crawler_url}\n\n"
            f"⏱ {elapsed_min}m {elapsed_sec}s elapsed "
            f"(timeout: {timeout_min}m)"
        )
    elif status == "done":
        st.success(st.session_state.crawler_message)
        if st.session_state.rag_option == "Current documents":
            st.caption(
                "Mode set to **Current documents** — crawled content is available in this mode."
            )
        if st.button("Dismiss", key="crawler_dismiss_done", type="tertiary"):
            st.session_state.crawler_status = "idle"
            st.rerun()
    elif status == "error":
        st.error(st.session_state.crawler_message)
        if st.button("Dismiss", key="crawler_dismiss_error", type="tertiary"):
            st.session_state.crawler_status = "idle"
            st.rerun()


# ---------- MAIN APP ----------
def main():
    # Header
    st.markdown(
        """
    <div class="app-header">
        <h1>Knowledge Management Assistant</h1>
    </div>
    """,
        unsafe_allow_html=True,
    )

    config_ok, config_issues = check_configuration()

    # ========== SIDEBAR ==========
    with st.sidebar:
        st.markdown("## Settings")

        # Ask Mode (matching RAG_v4)
        st.session_state.rag_option = st.selectbox(
            "Ask Mode",
            ["Current documents", "New document"],
            index=0 if st.session_state.rag_option == "Current documents" else 1,
            help="Choose where answers come from",
        )

        # Temperature slider
        st.session_state.temperature = st.slider(
            "Answer Style",
            min_value=0.0,
            max_value=1.0,
            value=float(st.session_state.temperature),
            step=0.1,
            help="0 = Factual & precise | 1 = More creative & exploratory",
        )

        st.divider()

        # Document Upload Section
        st.markdown("#### Upload Documents")

        save_dir = NEW_DOC_DIR if st.session_state.rag_option == "New document" else DOCUMENTS_DIR

        uploaded_files = st.file_uploader(
            "Choose files",
            type=["pdf", "doc", "docx", "txt", "md", "html", "csv"],
            accept_multiple_files=True,
            help="Add documents to the knowledge base",
            label_visibility="collapsed",
        )

        if uploaded_files:
            current_names = [f.name for f in uploaded_files]
            if current_names != st.session_state.uploaded_file_names:
                saved = handle_document_upload(uploaded_files, save_dir)
                if saved:
                    st.success(f"Saved {len(saved)} file(s)")
                st.session_state.uploaded_file_names = current_names

        # Run Data Upload button
        if st.button("Run Data Upload", use_container_width=True, type="secondary"):
            with st.spinner("Processing documents..."):
                # Add source_prefix for "New document" mode to enable filtering
                source_prefix = (
                    "new_uploads/" if st.session_state.rag_option == "New document" else ""
                )
                success, message = run_ingestion(
                    str(save_dir), source_prefix=source_prefix, skip_summary=True
                )
                if success:
                    st.success(message)
                else:
                    st.error(message)

        st.divider()

        # URL Crawler Section
        st.markdown("#### Crawl Website")

        crawler_active = st.session_state.crawler_status in ("crawling", "ingesting")

        crawler_url = st.text_input(
            "Website URL",
            placeholder="https://example.com",
            help="Enter a URL to crawl and ingest into the knowledge base",
            disabled=crawler_active,
        )
        crawler_max_pages = st.number_input(
            "Max Pages",
            min_value=1,
            max_value=500,
            value=10,
            help="Maximum number of pages to crawl (lower = faster)",
            disabled=crawler_active,
        )

        if not crawler_active:
            # Show Run button
            if st.button("Run Crawler", use_container_width=True, type="secondary"):
                if crawler_url:
                    start_crawler_background(crawler_url, crawler_max_pages)
                    st.rerun()
                else:
                    st.warning("Please enter a URL")
        else:
            # Show Stop button + status
            if st.button("Stop Crawler", use_container_width=True, type="primary"):
                stop_crawler()
                st.rerun()

        # Crawler status display (auto-refreshing fragment)
        _render_crawler_status()

        st.divider()

        # Advanced Features
        with st.expander("Advanced Features", expanded=False):
            st.session_state.debug_mode = st.checkbox(
                "Debug Mode",
                value=st.session_state.debug_mode,
                help="Show detailed processing information",
            )

        # Configuration status
        if config_ok:
            st.success("Configuration OK")
            db_ok, db_msg = check_database_connection()
            if db_ok:
                st.success("Database OK")
            else:
                st.error("Database Issue")
                st.warning(db_msg)
        else:
            st.error("Configuration Issues")
            for issue in config_issues:
                st.warning(issue)

        # Clear chat
        if st.button("Clear Chat", use_container_width=True, type="secondary"):
            st.session_state.chat_history = []
            st.session_state.session_id = str(uuid.uuid4())
            st.rerun()

        # Quick Guide
        with st.expander("Quick Guide", expanded=False):
            st.markdown("""
            - **Ask Mode**: Choose 'Current documents' or 'New document'
            - **Upload documents**: Add files (PDF, DOCX, TXT, etc.)
            - **Crawl Website**: Enter URL to crawl and ingest web content
            - **Run Data Upload**: Build/update the index
            - **Ask**: Type your question below
            """)

    # ========== MAIN CONTENT ==========
    if not config_ok:
        st.warning("Please configure the application. See sidebar for details.")
        st.code("""
# Create .env file with:
AZURE_OPENAI_API_KEY=your-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-ada-002
DATABASE_URL=postgresql://user:pass@server:5432/db?sslmode=require
        """)
        return

    # Chat container
    chat_container = st.container()
    with chat_container:
        for i, message in enumerate(st.session_state.chat_history):
            _avatar = AVATAR_ASSISTANT if message["role"] == "assistant" else AVATAR_USER
            with st.chat_message(message["role"], avatar=_avatar):
                if message["role"] == "assistant":
                    clickable_content = make_citations_clickable(message["content"])
                    st.markdown(clickable_content, unsafe_allow_html=True)
                    # Show agent badge for LogicRAG multi-hop queries (persisted)
                    agent_used = message.get("agent_used", "")
                    if agent_used == "logic_rag":
                        meta = message.get("metadata", {})
                        round_count = meta.get("round_count", 0)
                        subproblems = meta.get("subproblems", [])
                        st.caption(
                            f"Multi-hop reasoning | {round_count} rounds | "
                            f"{len(subproblems)} sub-problems"
                        )
                    # Inline source list + details button
                    sources = message.get("sources", [])
                    if sources:
                        render_sources_inline(sources, f"sources_history_{i}")
                else:
                    st.markdown(message["content"])

    # ========== CHAT INPUT ==========
    if prompt := st.chat_input("Ask a question about your documents..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        with chat_container:
            with st.chat_message("user", avatar=AVATAR_USER):
                st.markdown(prompt)

            with st.chat_message("assistant", avatar=AVATAR_ASSISTANT):
                with st.spinner("Thinking..."):
                    # Determine source filter based on Ask Mode
                    source_filter = None
                    if st.session_state.rag_option == "New document":
                        # Filter to only search documents from new_uploads directory
                        source_filter = "new_uploads/"

                    result = process_query(prompt, st.session_state.temperature, source_filter)

                response = result.get("response", "No response generated.")
                sources = result.get("sources", [])
                agent_used = result.get("agent_used", "unknown")

                clickable_response = make_citations_clickable(response)
                st.markdown(clickable_response, unsafe_allow_html=True)

                # Show agent badge for LogicRAG multi-hop queries
                if agent_used == "logic_rag":
                    metadata = result.get("metadata", {})
                    round_count = metadata.get("round_count", 0) if metadata else 0
                    subproblems = metadata.get("subproblems", []) if metadata else []
                    st.caption(
                        f"Multi-hop reasoning | {round_count} rounds | "
                        f"{len(subproblems)} sub-problems"
                    )

                # Inline source list + details button
                if sources:
                    render_sources_inline(sources, "sources_live")

                if st.session_state.debug_mode:
                    with st.expander("Debug Info", expanded=False):
                        debug_data = {
                            "agent_used": agent_used,
                            "documents_retrieved": result.get("documents_retrieved", 0),
                            "query_analysis": result.get("query_analysis", {}),
                            "error": result.get("error"),
                        }

                        # Add LogicRAG multi-hop reasoning metadata
                        if agent_used == "logic_rag":
                            metadata = result.get("metadata", {})
                            if metadata:
                                debug_data["logic_rag"] = {
                                    "subproblems": metadata.get("subproblems", []),
                                    "round_count": metadata.get("round_count", 0),
                                    "max_rounds": metadata.get("max_rounds", 5),
                                }

                        st.json(debug_data)

                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": response,
                        "sources": sources,
                        "agent_used": agent_used,
                        "metadata": result.get("metadata", {}),
                    }
                )

        st.rerun()


if __name__ == "__main__":
    main()
