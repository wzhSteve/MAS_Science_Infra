from MAS.epc_aw.tools.base import BaseTool
from playwright.sync_api import sync_playwright
import os
import base64
import hashlib
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

TOOL_NAME = "Screenshot_Tool"


class Screenshot_Tool(BaseTool):
    """Capture complete webpage screenshots as PNG images for visual content analysis."""

    require_llm_engine = False

    def __init__(self, output_dir: Optional[str] = None, timeout: int = 30000,
                 memory_dir: str = "memory", task_id: Optional[str] = None):
        """
        Initialize Screenshot_Tool.

        Args:
            output_dir: Directory to save screenshots (optional, defaults to memory/screenshots/{task_id})
            timeout: Playwright navigation timeout in milliseconds (default 30000)
            memory_dir: Root memory directory (default "memory")
            task_id: Task ID for organizing screenshots by task (default "default")
        """
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description="Capture complete webpage screenshots as PNG images for visual content analysis.",
            tool_version="1.0.0",
            input_types={
                "url": "str - The website URL to capture",
                "viewport_width": "int - Browser viewport width (default 1920)",
                "viewport_height": "int - Browser viewport height (default 1080)",
                "full_page": "bool - Capture full page or viewport only (default True)"
            },
            output_type="dict - Screenshot with image_path, image_base64, and metadata",
            demo_commands=[
                {
                    "command": 'execution = tool.execute(url="https://example.com")',
                    "description": "Capture screenshot of example.com"
                },
                {
                    "command": 'execution = tool.execute(url="https://example.com", full_page=True)',
                    "description": "Full page screenshot"
                }
            ]
        )
        # 使用绝对路径
        self.memory_dir = Path(memory_dir).absolute()
        self.screenshot_dir = self.memory_dir / "screenshots"
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)

        # 任务专属目录
        self.task_id = task_id or "default"
        self.task_screenshot_dir = self.screenshot_dir / self.task_id
        self.task_screenshot_dir.mkdir(parents=True, exist_ok=True)

        self.output_dir = output_dir or str(self.task_screenshot_dir)
        self.timeout = timeout

    def execute(self, url: str, viewport_width: int = 1920,
                viewport_height: int = 1080, full_page: bool = True) -> dict:
        """
        Capture a screenshot of the given URL and automatically update index.

        Args:
            url: Target website URL
            viewport_width: Browser viewport width in pixels (default 1920)
            viewport_height: Browser viewport height in pixels (default 1080)
            full_page: Capture entire page or just viewport (default True)

        Returns:
            dict with keys:
                - success: bool indicating success/failure
                - image_path: str absolute path to saved PNG file
                - screenshot_id: str unique identifier for this screenshot
                - image_base64: str base64-encoded image (or None if failed)
                - screenshot_size: tuple (width, height)
                - page_title: str page title
                - page_url: str actual page URL after redirects
                - file_size: int file size in bytes
                - error: str error message (or None if successful)
        """
        try:
            # Validate URL
            if not url.startswith(('http://', 'https://')):
                return {
                    "success": False,
                    "image_path": None,
                    "screenshot_id": None,
                    "image_base64": None,
                    "error": "Invalid URL: must start with http:// or https://"
                }

            # Create output directory (ensure it exists)
            output_path = Path(self.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Generate filename: {timestamp}_{url_hash}.png
            url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{url_hash}.png"
            filepath = output_path / filename

            # Launch Playwright and capture screenshot
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                context = browser.new_context(
                    viewport={"width": viewport_width, "height": viewport_height}
                )
                page = context.new_page()

                try:
                    # Navigate to URL and wait for network to be idle
                    page.goto(url, wait_until="networkidle", timeout=self.timeout)

                    # Get page information before screenshot
                    page_title = page.title()
                    actual_url = page.url

                    # Take screenshot
                    screenshot_bytes = page.screenshot(full_page=full_page)

                    # Save to file
                    with open(filepath, "wb") as f:
                        f.write(screenshot_bytes)

                    # Encode as base64
                    with open(filepath, "rb") as f:
                        image_base64 = base64.b64encode(f.read()).decode('utf-8')

                    # Generate screenshot ID
                    screenshot_id = f"{timestamp}_{url_hash}"

                    # Update index file
                    self._update_screenshot_index(
                        filepath=filepath,
                        url=url,
                        page_title=page_title,
                        screenshot_size=(viewport_width, viewport_height),
                        file_size=len(screenshot_bytes),
                        screenshot_id=screenshot_id
                    )

                    return {
                        "success": True,
                        "image_path": str(filepath.absolute()),
                        "screenshot_id": screenshot_id,
                        "image_base64": image_base64,
                        "screenshot_size": (viewport_width, viewport_height),
                        "page_title": page_title,
                        "page_url": actual_url,
                        "file_size": len(screenshot_bytes),
                        "error": None
                    }

                finally:
                    context.close()
                    browser.close()

        except Exception as e:
            return {
                "success": False,
                "image_path": None,
                "screenshot_id": None,
                "image_base64": None,
                "error": f"{type(e).__name__}: {str(e)}"
            }

    def _update_screenshot_index(self, filepath: Path, url: str,
                                  page_title: str, screenshot_size: tuple,
                                  file_size: int, screenshot_id: str) -> None:
        """
        Update global screenshot index file.

        索引文件结构：
        {
            "last_updated": "ISO timestamp",
            "screenshots": [
                {
                    "screenshot_id": "20260625_134500_a1b2c3d4",
                    "task_id": "task_001",
                    "url": "https://example.com",
                    "file_path": "memory/screenshots/task_001/20260625_134500_a1b2c3d4.png",
                    "page_title": "Page Title",
                    "screenshot_size": [1920, 1080],
                    "file_size": 245782,
                    "created_at": "ISO timestamp"
                }
            ],
            "index_by_task": {
                "task_001": ["20260625_134500_a1b2c3d4", "..."]
            }
        }
        """
        try:
            # 索引条目
            index_entry = {
                "screenshot_id": screenshot_id,
                "task_id": self.task_id,
                "url": url,
                "file_path": str(filepath.relative_to(self.memory_dir)),
                "page_title": page_title,
                "screenshot_size": list(screenshot_size),
                "file_size": file_size,
                "created_at": datetime.now().isoformat()
            }

            # 更新全局索引
            global_index_file = self.screenshot_dir / "index_global.json"
            global_index = self._load_json(global_index_file)
            if not global_index:
                global_index = {
                    "last_updated": datetime.now().isoformat(),
                    "screenshots": [],
                    "index_by_task": {}
                }

            # 添加新条目
            global_index["screenshots"].append(index_entry)
            global_index["last_updated"] = datetime.now().isoformat()

            # 更新任务索引
            if self.task_id not in global_index["index_by_task"]:
                global_index["index_by_task"][self.task_id] = []
            global_index["index_by_task"][self.task_id].append(screenshot_id)

            # 写回全局索引
            with open(global_index_file, "w", encoding="utf-8") as f:
                json.dump(global_index, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"Warning: Failed to update screenshot index: {e}")

    def _load_json(self, filepath: Path) -> Optional[dict]:
        """安全加载JSON文件"""
        if not filepath or not filepath.exists():
            return None
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
