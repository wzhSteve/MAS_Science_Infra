from MAS.epc_aw.tools.base import BaseTool
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
import os
import json
import base64
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

TOOL_NAME = "Browser_Tool"

LIMITATION = f"""
The {TOOL_NAME} has several limitations:
1. Requires Playwright + Chromium installed; headless only.
2. Each execute() call runs one browser session; state does NOT persist
   across separate execute() calls. Pass a multi-step `actions` list to
   chain interactions inside one session.
3. No authenticated / paywalled / CAPTCHA-protected pages.
4. selectors must be CSS selectors; `text=` shorthand does text-based click.
5. 30s navigation timeout per goto; long pages should disable wait_until=networkidle.
6. Not for computation (use Python_Coder_Tool) or static URL RAG (use Web_Search_Tool).
"""

BEST_PRACTICE = f"""
For optimal results with the {TOOL_NAME}:
1. Pass `actions` as an ordered list to chain goto → click → fill → extract.
2. Prefer `click(text="Next")` for pagination; loop via Python_Coder_Tool if needed.
3. Use `ctrl_f(term=...)` to locate on-page text, then `extract_text` to pull context.
4. For figure/table extraction, follow with Screenshot_Tool + Vision_OCR_Tool.
5. For "View history" / date-dropdown / archived-snapshot tasks, drive via actions.
6. Return per-page counts to Python_Coder_Tool for summation.
"""

# Actions recognized by execute()
#   {"op": "goto", "url": "...", "wait_until": "networkidle"|"domcontentloaded"|"load"}
#   {"op": "click", "selector": "css"}  OR  {"op": "click", "text": "Next"}
#   {"op": "fill", "selector": "css", "value": "..."}
#   {"op": "select", "selector": "css", "value": "..."}
#   {"op": "wait_for", "selector": "css"}  OR  {"op": "wait_for", "timeout_ms": 2000}
#   {"op": "ctrl_f", "term": "..."}            → returns first match context
#   {"op": "extract_text", "selector": "css?"} → page text or element text
#   {"op": "extract_links", "selector": "a?"}  → list of {text, href}
#   {"op": "count", "selector": "css"}         → number of matches
#   {"op": "screenshot", "full_page": true}    → base64 png
#   {"op": "scroll", "y": 800}


class Browser_Tool(BaseTool):
    """Interactive Playwright browser for click/fill/paginate/extract workflows."""

    require_llm_engine = False

    def __init__(self, timeout_ms: int = 30000, output_dir: Optional[str] = None):
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description=(
                "An interactive Playwright browser tool that executes a sequence of "
                "actions (goto, click, fill, select, wait_for, ctrl_f, extract_text, "
                "extract_links, count, screenshot, scroll) in one headless session. "
                "Use this when the task requires real browser interaction: pagination, "
                "date dropdowns, View history, on-page Ctrl-F, JS-rendered listings. "
                "For static URL content retrieval prefer Web_Search_Tool; for open "
                "search prefer Google_Search_Tool."
            ),
            tool_version="1.0.0",
            input_types={
                "actions": "list[dict] - ordered action sequence (see BEST_PRACTICES)",
                "url": "str - optional shortcut for a single goto action",
                "query": "str - optional, used only to label/seed extraction intent",
            },
            output_type="dict - per-action results + final page_text/screenshots",
            demo_commands=[
                {
                    "command": 'execution = tool.execute(actions=[{"op":"goto","url":"https://arxiv.org/abs/2106.12345"},{"op":"extract_text"}])',
                    "description": "Navigate and extract full page text."
                },
                {
                    "command": 'execution = tool.execute(actions=[{"op":"goto","url":"https://example.com/list"},{"op":"count","selector":".item"},{"op":"click","text":"Next"},{"op":"count","selector":".item"}])',
                    "description": "Count items across two paginated pages."
                },
                {
                    "command": 'execution = tool.execute(actions=[{"op":"goto","url":"https://en.wikipedia.org/wiki/Moon"},{"op":"ctrl_f","term":"perigee"},{"op":"extract_text"}])',
                    "description": "On-page Ctrl-F then extract surrounding text."
                },
            ],
            user_metadata={"limitations": LIMITATION, "best_practices": BEST_PRACTICE},
        )
        self.timeout_ms = timeout_ms
        self.output_dir = output_dir

    def execute(self, actions: Optional[List[Dict[str, Any]]] = None,
                url: Optional[str] = None,
                query: Optional[str] = None) -> Dict[str, Any]:
        if actions is None:
            if not url:
                return {"success": False, "error": "Provide `actions` list or `url`.", "results": []}
            actions = [{"op": "goto", "url": url}, {"op": "extract_text"}]
        if isinstance(actions, dict):
            actions = [actions]

        results: List[Dict[str, Any]] = []
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                context = browser.new_context(
                    viewport={"width": 1920, "height": 1080},
                    user_agent=("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                "AppleWebKit/537.36 (KHTML, like Gecko) "
                                "Chrome/123.0.0.0 Safari/537.36"),
                )
                page = context.new_page()
                page.set_default_navigation_timeout(self.timeout_ms)
                page.set_default_timeout(self.timeout_ms)

                for i, act in enumerate(actions):
                    try:
                        results.append(self._apply_action(page, act, results))
                    except Exception as e:
                        results.append({
                            "index": i, "op": act.get("op", "?"),
                            "success": False, "error": f"{type(e).__name__}: {e}",
                        })
                        # Abort remaining actions on a hard failure
                        break

                final_text = ""
                try:
                    final_text = page.content() and page.inner_text("body") or ""
                except Exception:
                    pass
                context.close()
                browser.close()

            return {"success": True, "results": results, "final_page_text": final_text[:8000]}
        except Exception as e:
            return {"success": False, "results": results, "error": f"{type(e).__name__}: {e}"}

    # ----------------- action dispatch -----------------
    def _apply_action(self, page, act: Dict[str, Any], prior: List[Dict[str, Any]]) -> Dict[str, Any]:
        op = act.get("op", "")
        idx = len(prior)
        base = {"index": idx, "op": op, "success": True}

        if op == "goto":
            wait_until = act.get("wait_until", "domcontentloaded")
            page.goto(act["url"], wait_until=wait_until, timeout=act.get("timeout_ms", self.timeout_ms))
            base["url"] = page.url
            base["title"] = page.title()
            return base

        if op == "click":
            if "text" in act:
                page.get_by_text(act["text"]).first.click(timeout=act.get("timeout_ms", self.timeout_ms))
            else:
                page.click(act["selector"], timeout=act.get("timeout_ms", self.timeout_ms))
            base["clicked"] = act.get("text") or act.get("selector")
            return base

        if op == "fill":
            page.fill(act["selector"], act["value"])
            base["filled"] = act["selector"]
            return base

        if op == "select":
            page.select_option(act["selector"], act["value"])
            base["selected"] = act["selector"]
            return base

        if op == "wait_for":
            if "selector" in act:
                page.wait_for_selector(act["selector"], timeout=act.get("timeout_ms", self.timeout_ms))
                base["waited_for"] = act["selector"]
            else:
                page.wait_for_timeout(int(act.get("timeout_ms", 1000)))
            return base

        if op == "ctrl_f":
            term = act["term"]
            # Use Playwright's built-in text search via evaluate
            found = page.evaluate(
                """(term) => {
                    const body = document.body;
                    const re = new RegExp(term.replace(/[.*+?^${}()|[\\]\\\\]/g, '\\\\$&'), 'i');
                    const walker = document.createTreeWalker(body, NodeFilter.SHOW_TEXT, null);
                    while (walker.nextNode()) {
                        if (re.test(walker.currentNode.nodeValue)) {
                            const el = walker.currentNode.parentElement;
                            return el ? el.outerHTML.slice(0, 1000) : walker.currentNode.nodeValue.slice(0, 1000);
                        }
                    }
                    return null;
                }""",
                term,
            )
            base["term"] = term
            base["found"] = bool(found)
            base["context"] = found
            return base

        if op == "extract_text":
            if "selector" in act:
                text = page.inner_text(act["selector"])
            else:
                text = page.inner_text("body")
            base["text"] = (text or "")[:8000]
            return base

        if op == "extract_links":
            sel = act.get("selector", "a")
            links = page.eval_on_selector_all(
                sel,
                """els => els.map(e => ({text: (e.innerText||'').trim().slice(0,200), href: e.href}))""",
            )
            base["links"] = links[:200]
            return base

        if op == "count":
            sel = act["selector"]
            n = page.locator(sel).count()
            base["selector"] = sel
            base["count"] = n
            return base

        if op == "screenshot":
            full_page = act.get("full_page", True)
            png = page.screenshot(full_page=full_page)
            b64 = base64.b64encode(png).decode("utf-8")
            base["image_base64"] = b64 if act.get("return_base64", False) else f"<{len(png)} bytes png>"
            if self.output_dir:
                Path(self.output_dir).mkdir(parents=True, exist_ok=True)
                fp = Path(self.output_dir) / f"browser_{idx}.png"
                fp.write_bytes(png)
                base["image_path"] = str(fp)
            base["full_page"] = full_page
            return base

        if op == "scroll":
            page.evaluate(f"window.scrollBy(0, {int(act.get('y', 800))})")
            base["scrolled_y"] = int(act.get("y", 800))
            return base

        return {"index": idx, "op": op, "success": False, "error": f"unknown op: {op}"}

    def get_metadata(self):
        return super().get_metadata()


if __name__ == "__main__":
    tool = Browser_Tool()
    print(json.dumps(tool.get_metadata(), indent=2, ensure_ascii=False))
    # Minimal smoke test (requires network + playwright install)
    try:
        result = tool.execute(actions=[
            {"op": "goto", "url": "https://example.com", "wait_until": "domcontentloaded"},
            {"op": "extract_text"},
            {"op": "count", "selector": "a"},
        ])
        print("Smoke result success:", result.get("success"))
        for r in result.get("results", []):
            print(" ", r.get("op"), "ok=", r.get("success"), "count=", r.get("count"))
    except Exception as e:
        print(f"Smoke test skipped/failed: {e}")
    print("Done!")
