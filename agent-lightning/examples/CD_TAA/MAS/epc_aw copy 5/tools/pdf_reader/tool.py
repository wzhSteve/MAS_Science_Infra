from MAS.epc_aw.tools.base import BaseTool
import os
import io
import re
import json
import base64
import requests
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

TOOL_NAME = "PDF_Reader_Tool"

LIMITATION = f"""
The {TOOL_NAME} has several limitations:
1. Requires a text-layer PDF. Scanned-only PDFs return no text; use mode="figures"
   to extract page images and hand them to Vision_OCR_Tool.
2. Large PDFs (>100 pages) should be queried with mode="search" or "pages" by
   page index rather than "full_text" to stay within token budgets.
3. Online PDFs are fetched via requests; JS-protected hosts should be downloaded
   first by Browser_Tool and passed as a local path.
4. Figure extraction is best-effort (PyMuPDF embedded-image enumeration); it does
   not reconstruct complex multi-panel figures.
5. Not for computation on numbers extracted from the PDF — pipe the result to
   Python_Coder_Tool.
"""

BEST_PRACTICE = f"""
For optimal results with the {TOOL_NAME}:
1. mode="search" with a term locates the relevant page(s) cheaply before reading.
2. mode="count" counts pages or matches of a selector-like term.
3. mode="figures" returns base64 page-region images → feed to Vision_OCR_Tool.
4. mode="tables" attempts structural table extraction (best with PyMuPDF).
5. For "how many articles listed in <section> on <date> had <format>" tasks,
   combine Browser_Tool (navigate listing) with PDF_Reader_Tool (per-paper checks).
6. Always pass url OR file_path (one is enough); url is normalized (arxiv pdf→abs
   is NOT done here — that normalization lives in Web_Search_Tool).
"""


class PDF_Reader_Tool(BaseTool):
    """Read PDFs by page / search / count / figures / tables / full_text."""

    require_llm_engine = False

    def __init__(self, output_dir: Optional[str] = None, max_pages_full_text: int = 50):
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description=(
                "A dedicated PDF reader that extracts text, figures, tables, and counts "
                "from a PDF URL or local path. Modes: pages, full_text, search, count, "
                "figures, tables. Use this for GAIA tasks involving arXiv papers, reports, "
                "scanned documents (figures→Vision_OCR_Tool), and 'count items in PDF' "
                "questions. For HTML pages use Web_Search_Tool; for interactive listings "
                "use Browser_Tool."
            ),
            tool_version="1.0.0",
            input_types={
                "url": "str - optional URL to a PDF file",
                "file_path": "str - optional local path to a PDF file",
                "mode": "str - pages | full_text | search | count | figures | tables (default pages)",
                "page_indices": "list[int] - 0-based page indices for mode='pages'",
                "term": "str - search/count term for mode='search'/'count'",
                "max_len": "int - per-page char cap (default 4000)",
            },
            output_type="dict - extracted text/figures/counts with success status",
            demo_commands=[
                {
                    "command": 'execution = tool.execute(url="https://arxiv.org/pdf/2106.12345", mode="search", term="egalitarian")',
                    "description": "Find pages mentioning a term in an arXiv PDF."
                },
                {
                    "command": 'execution = tool.execute(file_path="/tmp/paper.pdf", mode="figures", page_indices=[0])',
                    "description": "Extract page-0 figures as base64 for Vision_OCR_Tool."
                },
                {
                    "command": 'execution = tool.execute(url="https://example.com/list.pdf", mode="count", term="ps versions")',
                    "description": "Count term matches in a listing PDF."
                },
            ],
            user_metadata={"limitations": LIMITATION, "best_practices": BEST_PRACTICE},
        )
        self.output_dir = output_dir
        self.max_pages_full_text = max_pages_full_text
        self._headers = {
            "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                           "AppleWebKit/537.36 (KHTML, like Gecko) "
                           "Chrome/123.0.0.0 Safari/537.36"),
        }

    # ----------------- load PDF bytes -----------------
    def _load_pdf_bytes(self, url: Optional[str], file_path: Optional[str]) -> bytes:
        if file_path:
            with open(file_path, "rb") as f:
                return f.read()
        if not url:
            raise ValueError("Provide either `url` or `file_path`.")
        resp = requests.get(url, headers=self._headers, timeout=30)
        resp.raise_for_status()
        return resp.content

    def _open_reader(self, pdf_bytes: bytes):
        # Prefer PyMuPDF (fitz) for figures/tables; fall back to PyPDF2 for text.
        try:
            import fitz  # PyMuPDF
            return ("fitz", fitz.open(stream=pdf_bytes, filetype="pdf"))
        except ImportError:
            pass
        from PyPDF2 import PdfReader
        return ("pypdf2", PdfReader(io.BytesIO(pdf_bytes)))

    # ----------------- execute -----------------
    def execute(self, url: Optional[str] = None, file_path: Optional[str] = None,
                mode: str = "pages", page_indices: Optional[List[int]] = None,
                term: Optional[str] = None, max_len: int = 4000) -> Dict[str, Any]:
        try:
            pdf_bytes = self._load_pdf_bytes(url, file_path)
            backend, doc = self._open_reader(pdf_bytes)
            n_pages = doc.page_count if backend == "fitz" else len(doc.pages)

            if mode == "full_text":
                return self._mode_full_text(backend, doc, n_pages, max_len)
            if mode == "pages":
                return self._mode_pages(backend, doc, page_indices or [0], max_len)
            if mode == "search":
                return self._mode_search(backend, doc, term or "", n_pages, max_len)
            if mode == "count":
                return self._mode_count(backend, doc, term, n_pages)
            if mode == "figures":
                return self._mode_figures(backend, doc, page_indices or [0])
            if mode == "tables":
                return self._mode_tables(backend, doc, page_indices or [0], max_len)
            return {"success": False, "error": f"unknown mode: {mode}"}
        except Exception as e:
            return {"success": False, "error": f"{type(e).__name__}: {e}"}

    # ----------------- text helpers -----------------
    def _page_text(self, backend, doc, idx: int) -> str:
        if backend == "fitz":
            return doc[idx].get_text("text") or ""
        try:
            return doc.pages[idx].extract_text() or ""
        except Exception:
            return ""

    def _mode_full_text(self, backend, doc, n_pages: int, max_len: int) -> Dict[str, Any]:
        cap = min(n_pages, self.max_pages_full_text)
        parts: List[str] = []
        for i in range(cap):
            t = self._page_text(backend, doc, i)
            if t:
                parts.append(f"--- page {i} ---\n{t}")
        joined = "\n".join(parts)
        return {"success": True, "mode": "full_text", "page_count": n_pages,
                "text": joined[: max_len * 4], "truncated": len(joined) > max_len * 4}

    def _mode_pages(self, backend, doc, page_indices: List[int], max_len: int) -> Dict[str, Any]:
        out: Dict[str, Any] = {"success": True, "mode": "pages", "pages": {}}
        for idx in page_indices:
            try:
                out["pages"][str(idx)] = self._page_text(backend, doc, idx)[:max_len]
            except Exception as e:
                out["pages"][str(idx)] = f"[error: {e}]"
        return out

    def _mode_search(self, backend, doc, term: str, n_pages: int, max_len: int) -> Dict[str, Any]:
        if not term:
            return {"success": False, "error": "mode='search' requires `term`"}
        pat = re.compile(re.escape(term), re.IGNORECASE)
        hits: List[Dict[str, Any]] = []
        for i in range(n_pages):
            t = self._page_text(backend, doc, i)
            if pat.search(t):
                # pull a context window around first match
                m = pat.search(t)
                start = max(0, m.start() - 200)
                end = min(len(t), m.end() + 600)
                hits.append({"page": i, "context": t[start:end][:max_len]})
                if len(hits) >= 20:
                    break
        return {"success": True, "mode": "search", "term": term,
                "hit_count": len(hits), "hits": hits}

    def _mode_count(self, backend, doc, term: Optional[str], n_pages: int) -> Dict[str, Any]:
        if term:
            pat = re.compile(re.escape(term), re.IGNORECASE)
            total = 0
            per_page: List[int] = []
            for i in range(n_pages):
                t = self._page_text(backend, doc, i)
                c = len(pat.findall(t))
                per_page.append(c)
                total += c
            return {"success": True, "mode": "count", "term": term,
                    "page_count": n_pages, "match_count": total, "per_page": per_page}
        return {"success": True, "mode": "count", "page_count": n_pages}

    def _mode_figures(self, backend, doc, page_indices: List[int]) -> Dict[str, Any]:
        if backend != "fitz":
            return {"success": False, "error": "figures mode requires PyMuPDF (fitz)"}
        out: Dict[str, Any] = {"success": True, "mode": "figures", "pages": {}}
        for idx in page_indices:
            page = doc[idx]
            images = page.get_images(full=True)
            page_imgs: List[str] = []
            for img_index, img in enumerate(images):
                xref = img[0]
                try:
                    base_img = doc.extract_image(xref)
                    b64 = base64.b64encode(base_img["image"]).decode("utf-8")
                    page_imgs.append(b64)
                except Exception:
                    continue
            # Also render the full page as an image (catches vector figures)
            try:
                pix = page.get_pixmap(dpi=150)
                page_b64 = base64.b64encode(pix.tobytes("png")).decode("utf-8")
                page_imgs.append(page_b64)
            except Exception:
                pass
            out["pages"][str(idx)] = page_imgs
        return out

    def _mode_tables(self, backend, doc, page_indices: List[int], max_len: int) -> Dict[str, Any]:
        if backend != "fitz":
            return {"success": False, "error": "tables mode requires PyMuPDF (fitz)"}
        out: Dict[str, Any] = {"success": True, "mode": "tables", "pages": {}}
        for idx in page_indices:
            try:
                tabs = doc[idx].find_tables()
                tables = []
                for t in tabs.tables:
                    try:
                        tables.append(t.extract())
                    except Exception:
                        continue
                out["pages"][str(idx)] = json.dumps(tables, ensure_ascii=False)[: max_len * 2]
            except Exception as e:
                out["pages"][str(idx)] = f"[error: {e}]"
        return out

    def get_metadata(self):
        return super().get_metadata()


if __name__ == "__main__":
    tool = PDF_Reader_Tool()
    print(json.dumps(tool.get_metadata(), indent=2, ensure_ascii=False))
    print("Done!")
