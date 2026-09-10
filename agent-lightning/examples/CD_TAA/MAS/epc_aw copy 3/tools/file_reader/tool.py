from MAS.epc_aw.tools.base import BaseTool
import os
import json
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

TOOL_NAME = "File_Reader_Tool"

LIMITATION = f"""
The {TOOL_NAME} has several limitations:
1. Reads files from the local filesystem (GAIA `file_path` attachments).
2. For PDFs, delegates to PDF_Reader_Tool; for images, to Vision_OCR_Tool;
   for audio, to Audio_Tool. Ensure those tools are importable.
3. CSV/JSON/text are read in-process; very large files are truncated.
4. No network fetch — download remote attachments first via Browser_Tool/requests.
"""

BEST_PRACTICE = f"""
For optimal results with the {TOOL_NAME}:
1. mode="auto" infers type by extension and dispatches to the right tool.
2. For CSV, mode="csv" returns header + rows (capped) for Python_Coder_Tool.
3. For images, mode="image" returns base64 → Vision_OCR_Tool.
4. For PDFs, mode="pdf" delegates to PDF_Reader_Tool (pass through page_indices/term).
5. Always check the `file_path`/`file_name` fields on the GAIA task before calling.
"""


class File_Reader_Tool(BaseTool):
    """Read user-attached files (CSV/JSON/text/PDF/image/audio) and dispatch to specialists."""

    require_llm_engine = False

    def __init__(self):
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description=(
                "Read a local attachment file and return its content, dispatching by "
                "type: CSV/JSON/text are parsed in-process; PDF delegates to "
                "PDF_Reader_Tool; images delegate to Vision_OCR_Tool; audio delegates "
                "to Audio_Tool. Use for GAIA tasks that ship a file_name/file_path "
                "attachment. mode='auto' infers the type from the extension."
            ),
            tool_version="1.0.0",
            input_types={
                "file_path": "str - path to the attachment file",
                "mode": "str - auto | csv | json | text | pdf | image | audio (default auto)",
                "max_rows": "int - row cap for CSV (default 200)",
                "max_len": "int - char cap for text (default 8000)",
            },
            output_type="dict - file content / delegated tool result",
            demo_commands=[
                {
                    "command": 'execution = tool.execute(file_path="/tmp/attachment.csv", mode="auto")',
                    "description": "Auto-detect and read a CSV attachment."
                },
                {
                    "command": 'execution = tool.execute(file_path="/tmp/scan.png", mode="image")',
                    "description": "Read an image attachment via Vision_OCR_Tool."
                },
            ],
            user_metadata={"limitations": LIMITATION, "best_practices": BEST_PRACTICE},
        )

    def execute(self, file_path: str, mode: str = "auto",
                max_rows: int = 200, max_len: int = 8000) -> Dict[str, Any]:
        p = Path(file_path)
        if not p.exists():
            return {"success": False, "error": f"file not found: {file_path}"}
        if mode == "auto":
            mode = self._infer_mode(p)
        try:
            if mode == "csv":
                return self._read_csv(p, max_rows)
            if mode == "json":
                return self._read_json(p, max_len)
            if mode == "text":
                return self._read_text(p, max_len)
            if mode == "pdf":
                return self._delegate_pdf(p)
            if mode == "image":
                return self._delegate_image(p)
            if mode == "audio":
                return self._delegate_audio(p)
            return {"success": False, "error": f"unknown mode: {mode}"}
        except Exception as e:
            return {"success": False, "error": f"{type(e).__name__}: {e}"}

    @staticmethod
    def _infer_mode(p: Path) -> str:
        ext = p.suffix.lower()
        if ext in {".csv"}:
            return "csv"
        if ext in {".json"}:
            return "json"
        if ext in {".txt", ".md", ".log", ".tsv"}:
            return "text"
        if ext in {".pdf"}:
            return "pdf"
        if ext in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}:
            return "image"
        if ext in {".mp3", ".wav", ".m4a", ".flac", ".ogg"}:
            return "audio"
        return "text"

    def _read_csv(self, p: Path, max_rows: int) -> Dict[str, Any]:
        with open(p, newline="", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)
            rows = []
            for i, row in enumerate(reader):
                if i >= max_rows:
                    break
                rows.append(row)
        return {"success": True, "mode": "csv", "file_path": str(p),
                "row_count": len(rows), "rows": rows}

    def _read_json(self, p: Path, max_len: int) -> Dict[str, Any]:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        text = json.dumps(data, ensure_ascii=False)
        return {"success": True, "mode": "json", "file_path": str(p),
                "content": text[:max_len], "truncated": len(text) > max_len}

    def _read_text(self, p: Path, max_len: int) -> Dict[str, Any]:
        text = p.read_text(encoding="utf-8", errors="ignore")
        return {"success": True, "mode": "text", "file_path": str(p),
                "content": text[:max_len], "truncated": len(text) > max_len}

    def _delegate_pdf(self, p: Path) -> Dict[str, Any]:
        try:
            from MAS.epc_aw.tools.pdf_reader.tool import PDF_Reader_Tool
            return PDF_Reader_Tool().execute(file_path=str(p), mode="full_text")
        except Exception as e:
            return {"success": False, "error": f"PDF delegation failed: {e}"}

    def _delegate_image(self, p: Path) -> Dict[str, Any]:
        try:
            from MAS.epc_aw.tools.vision_ocr.tool import Vision_OCR_Tool
            return Vision_OCR_Tool().execute(image_input=str(p), ocr_prompt="general_ocr")
        except Exception as e:
            return {"success": False, "error": f"image delegation failed: {e}"}

    def _delegate_audio(self, p: Path) -> Dict[str, Any]:
        try:
            from MAS.epc_aw.tools.audio.tool import Audio_Tool
            return Audio_Tool().execute(audio_input=str(p), mode="stt")
        except Exception as e:
            return {"success": False, "error": f"audio delegation failed: {e}"}

    def get_metadata(self):
        return super().get_metadata()


if __name__ == "__main__":
    tool = File_Reader_Tool()
    print(json.dumps(tool.get_metadata(), indent=2, ensure_ascii=False))
    print("Done!")
