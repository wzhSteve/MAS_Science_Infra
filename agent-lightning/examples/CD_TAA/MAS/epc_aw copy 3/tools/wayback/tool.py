from MAS.epc_aw.tools.base import BaseTool
import os
import json
import requests
from datetime import datetime
from typing import Any, Dict, List, Optional

TOOL_NAME = "Wayback_Tool"

LIMITATION = f"""
The {TOOL_Name} has several limitations:
1. Depends on archive.org Availability API; not every URL/date has a snapshot.
2. Returned snapshot URLs must be fetched by Web_Search_Tool or Browser_Tool.
3. Diffing two snapshots is done by calling this tool twice then Python_Coder_Tool.
4. Rate-limited; avoid hammering the API.
""".replace("TOOL_Name", "TOOL_NAME")

BEST_PRACTICE = f"""
For optimal results with the {TOOL_NAME}:
1. mode="snapshot" with date=YYYYMMDD returns the closest archived snapshot URL.
2. mode="availability" lists all snapshots for a URL (for finding earliest/latest).
3. Pipe the returned snapshot URL to Web_Search_Tool(query, url=snapshot_url) to
   extract content as of that date.
4. For "what changed between date A and date B" tasks, fetch both snapshots then
   diff with Python_Coder_Tool.
"""


class Wayback_Tool(BaseTool):
    """Internet Archive Wayback Machine lookup: snapshot URL by date / availability list."""

    require_llm_engine = False

    def __init__(self):
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description=(
                "Query the Internet Archive Wayback Machine (web.archive.org) for "
                "archived snapshots of a URL. mode='snapshot' returns the closest "
                "archived snapshot URL for a given YYYYMMDD date; mode='availability' "
                "returns the list of archived snapshots. Use for GAIA tasks asking "
                "about historical page content, menu changes, or version diffs."
            ),
            tool_version="1.0.0",
            input_types={
                "url": "str - the original URL to look up in the archive",
                "mode": "str - snapshot | availability (default snapshot)",
                "date": "str - YYYYMMDD for mode='snapshot'",
            },
            output_type="dict - archived snapshot URL(s) with success status",
            demo_commands=[
                {
                    "command": 'execution = tool.execute(url="https://www.virtuerestaurant.com/menus/", mode="snapshot", date="20210322")',
                    "description": "Get the March 22 2021 archived snapshot of a menu page."
                },
                {
                    "command": 'execution = tool.execute(url="https://example.com", mode="availability")',
                    "description": "List all archived snapshots for a URL."
                },
            ],
            user_metadata={"limitations": LIMITATION, "best_practices": BEST_PRACTICE},
        )
        self.api = "https://archive.org/wayback/available"

    def execute(self, url: str, mode: str = "snapshot",
                date: Optional[str] = None) -> Dict[str, Any]:
        try:
            if mode == "snapshot":
                return self._snapshot(url, date)
            if mode == "availability":
                return self._availability(url)
            return {"success": False, "error": f"unknown mode: {mode}"}
        except Exception as e:
            return {"success": False, "error": f"{type(e).__name__}: {e}"}

    def _snapshot(self, url: str, date: Optional[str]) -> Dict[str, Any]:
        if not date:
            return {"success": False, "error": "mode='snapshot' requires `date` YYYYMMDD"}
        params = {"url": url, "timestamp": date}
        resp = requests.get(self.api, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        snap = (data.get("archived_snapshots") or {}).get("closest")
        if not snap:
            return {"success": True, "mode": "snapshot", "url": url, "date": date,
                    "snapshot_url": None, "available": False}
        return {"success": True, "mode": "snapshot", "url": url, "date": date,
                "snapshot_url": snap.get("url"), "timestamp": snap.get("timestamp"),
                "status": snap.get("status"), "available": True}

    def _availability(self, url: str) -> Dict[str, Any]:
        # CDX API for listing snapshots
        cdx = "http://web.archive.org/cdx/search/cdx"
        params = {"url": url, "output": "json", "limit": "50", "fl": "timestamp,statuscode,original"}
        resp = requests.get(cdx, params=params, timeout=30)
        resp.raise_for_status()
        rows = resp.json()
        if not rows or len(rows) < 2:
            return {"success": True, "mode": "availability", "url": url, "snapshots": []}
        header = rows[0]
        snapshots = [dict(zip(header, r)) for r in rows[1:]]
        return {"success": True, "mode": "availability", "url": url,
                "snapshot_count": len(snapshots), "snapshots": snapshots}

    def get_metadata(self):
        return super().get_metadata()


if __name__ == "__main__":
    tool = Wayback_Tool()
    print(json.dumps(tool.get_metadata(), indent=2, ensure_ascii=False))
    print("Done!")
