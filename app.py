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

.answer-card {padding: 0.9rem; border-radius: 10px; border: 1px solid #E6F7F6; background: #FFFFFF;}
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
.source-score {
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

.stButton button, button[kind="primary"], [data-testid="baseButton-primary"] {
    background: var(--opal) !important;
    color: var(--navy) !important;
    border: 1px solid var(--opal) !important;
}
.stButton button:hover, button[kind="primary"]:hover {
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
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------- SESSION STATE ----------
def init_session_state():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "references_last" not in st.session_state:
        st.session_state.references_last = []
    if "answer_last" not in st.session_state:
        st.session_state.answer_last = ""
    if "rag_option" not in st.session_state:
        st.session_state.rag_option = "Current documents"
    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.0
    if "debug_mode" not in st.session_state:
        st.session_state.debug_mode = False
    if "uploaded_file_names" not in st.session_state:
        st.session_state.uploaded_file_names = []


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
    escaped_text = html.escape(text)
    citation_pattern = r"\[(\d+(?:-\d+)?(?:\]\[\d+(?:-\d+)?)*)\]"

    def replace_citation(match):
        citation_text = match.group(0)
        numbers = re.findall(r"\d+", citation_text)
        if numbers:
            first_number = numbers[0]
            return f'<a href="#source-{first_number}" class="citation-link" data-source="{first_number}" title="View source {first_number}">{citation_text}</a>'
        return citation_text

    return re.sub(citation_pattern, replace_citation, escaped_text)


def extract_cited_numbers(answer_text: str) -> list:
    citation_pattern = r"\[(\d+)\]"
    return sorted(set(int(m) for m in re.findall(citation_pattern, answer_text)))


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


def render_sources_detailed(sources: list):
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
        else:
            content = getattr(source, "content", str(source))
            doc_title = getattr(source, "document_title", f"Source {i}")
            doc_source = getattr(source, "document_source", "")
            score = getattr(source, "score", 0)
            chunk_index = getattr(source, "chunk_index", 0)

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
                <span class="source-score">Score: {score:.3f}</span>
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
                    key=f"download_source_{i}",
                )


def render_sources_compact(sources: list, answer_text: str):
    """Render compact source list below answer."""
    if not sources:
        return

    cited_numbers = extract_cited_numbers(answer_text)
    if not cited_numbers:
        return

    st.markdown("**Sources:**")
    sources_html = '<div class="sources-list">'

    for i, source in enumerate(sources, 1):
        if i not in cited_numbers:
            continue

        if isinstance(source, dict):
            doc_title = source.get("document_title", source.get("title", f"Source {i}"))
        else:
            doc_title = getattr(source, "document_title", f"Source {i}")

        sources_html += f'<div class="source-item" id="source-{i}"><span class="source-number">[{i}]</span> {html.escape(doc_title)}</div>'

    sources_html += "</div>"
    st.markdown(sources_html, unsafe_allow_html=True)


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


def run_ingestion(documents_dir: str | None = None, source_prefix: str = "") -> tuple[bool, str]:
    """Run document ingestion.

    Args:
        documents_dir: Directory containing documents.
        source_prefix: Prefix to add to source paths (e.g., "new_uploads/").
    """
    try:
        cmd = ["uv", "run", "rag-ingest"]
        if documents_dir:
            cmd.extend(["--documents", documents_dir])
        else:
            cmd.extend(["--documents", str(DOCUMENTS_DIR)])

        if source_prefix:
            cmd.extend(["--source-prefix", source_prefix])

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT), timeout=300)
        if result.returncode == 0:
            return True, "Documents ingested successfully!"
        else:
            return False, f"Ingestion failed: {result.stderr[:200]}"
    except Exception as e:
        return False, f"Error: {str(e)}"


def run_crawler(url: str, max_pages: int = 50) -> tuple[bool, str, str | None]:
    """Run the RAG crawler on a URL.

    Returns:
        Tuple of (success, message, json_report_path or None).
    """
    try:
        # Find or create output directory
        output_dir = ROOT / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get list of existing JSON reports before crawl
        existing_reports = set(output_dir.glob("crawl_report_*.json"))

        cmd = ["uv", "run", "rag-crawler", "--url", url, "--max-pages", str(max_pages)]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT), timeout=600)

        if result.returncode == 0:
            # Find the new JSON report (created after crawl)
            new_reports = set(output_dir.glob("crawl_report_*.json")) - existing_reports
            if new_reports:
                report_path = max(new_reports, key=lambda p: p.stat().st_mtime)
                return True, f"Successfully crawled {url}", str(report_path)
            else:
                # Fallback: find most recent JSON report
                all_reports = list(output_dir.glob("crawl_report_*.json"))
                if all_reports:
                    report_path = max(all_reports, key=lambda p: p.stat().st_mtime)
                    return True, f"Successfully crawled {url}", str(report_path)
                return True, f"Successfully crawled {url}", None
        else:
            return False, f"Crawler failed: {result.stderr[:200]}", None
    except subprocess.TimeoutExpired:
        return False, "Crawler timed out after 10 minutes", None
    except Exception as e:
        return False, f"Error: {str(e)}", None


def run_crawl_ingestion(report_path: str) -> tuple[bool, str]:
    """Ingest documents from a crawl report.

    Args:
        report_path: Path to the crawler JSON report.

    Returns:
        Tuple of (success, message).
    """
    try:
        cmd = ["uv", "run", "rag-ingest", "--crawl-report", report_path]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT), timeout=300)
        if result.returncode == 0:
            return True, "Crawled content ingested successfully!"
        else:
            return False, f"Ingestion failed: {result.stderr[:200]}"
    except Exception as e:
        return False, f"Error: {str(e)}"


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
                success, message = run_ingestion(str(save_dir), source_prefix=source_prefix)
                if success:
                    st.success(message)
                else:
                    st.error(message)

        st.divider()

        # URL Crawler Section
        st.markdown("#### Crawl Website")
        crawler_url = st.text_input(
            "Website URL",
            placeholder="https://example.com",
            help="Enter a URL to crawl and ingest into the knowledge base",
        )
        crawler_max_pages = st.number_input(
            "Max Pages",
            min_value=1,
            max_value=500,
            value=50,
            help="Maximum number of pages to crawl",
        )

        if st.button("Run Crawler", use_container_width=True, type="secondary"):
            if crawler_url:
                with st.spinner(f"Crawling {crawler_url}..."):
                    success, message, report_path = run_crawler(crawler_url, crawler_max_pages)
                    if success:
                        st.success(message)
                        # Auto-ingest crawled content using the report
                        if report_path:
                            with st.spinner("Ingesting crawled content..."):
                                ingest_success, ingest_msg = run_crawl_ingestion(report_path)
                                if ingest_success:
                                    st.success("Content ingested!")
                                else:
                                    st.warning(ingest_msg)
                        else:
                            st.warning("Crawl report not found. Run manual ingestion.")
                    else:
                        st.error(message)
            else:
                st.warning("Please enter a URL")

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
            st.session_state.references_last = []
            st.session_state.answer_last = ""
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
        for message in st.session_state.chat_history:
            _avatar = AVATAR_ASSISTANT if message["role"] == "assistant" else AVATAR_USER
            with st.chat_message(message["role"], avatar=_avatar):
                if message["role"] == "assistant":
                    clickable_content = make_citations_clickable(message["content"])
                    st.markdown(
                        f'<div class="answer-card">{clickable_content}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(message["content"])

    # ========== SOURCE SECTION ==========
    if st.session_state.chat_history:
        st.markdown("---")

        # JavaScript for citation click handling
        st.markdown(
            """
        <script>
        document.addEventListener('click', function(e) {
            if (e.target && e.target.classList && e.target.classList.contains('citation-link')) {
                e.preventDefault();
                const sourceNum = e.target.getAttribute('data-source');
                if (sourceNum) {
                    const sourceElement = document.querySelector('#source-' + sourceNum);
                    if (sourceElement) {
                        sourceElement.scrollIntoView({ behavior: 'smooth', block: 'center' });
                        sourceElement.style.transition = 'all 0.3s ease';
                        sourceElement.style.backgroundColor = '#e6f9f7';
                        sourceElement.style.border = '2px solid #1CCFC9';
                        setTimeout(() => {
                            sourceElement.style.backgroundColor = '';
                            sourceElement.style.border = '';
                        }, 2500);
                    }
                }
            }
        });
        </script>
        """,
            unsafe_allow_html=True,
        )

        st.markdown("### Source")

        if st.session_state.references_last:
            st.markdown("*Click on citations [1], [2] in the answer to jump to sources*")
            st.markdown(
                '<div style="max-height: 600px; overflow-y: auto; padding: 1rem;">',
                unsafe_allow_html=True,
            )
            render_sources_detailed(st.session_state.references_last)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("No sources available. Sources will appear here after you ask a question.")

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
                st.markdown(
                    f'<div class="answer-card">{clickable_response}</div>', unsafe_allow_html=True
                )

                # Show agent badge for LogicRAG multi-hop queries
                if agent_used == "logic_rag":
                    metadata = result.get("metadata", {})
                    round_count = metadata.get("round_count", 0) if metadata else 0
                    subproblems = metadata.get("subproblems", []) if metadata else []
                    st.caption(
                        f"Multi-hop reasoning | {round_count} rounds | "
                        f"{len(subproblems)} sub-problems"
                    )

                # Show compact sources below answer
                if sources:
                    render_sources_compact(sources, response)

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
                    }
                )
                st.session_state.references_last = sources
                st.session_state.answer_last = response

        st.rerun()


if __name__ == "__main__":
    main()
