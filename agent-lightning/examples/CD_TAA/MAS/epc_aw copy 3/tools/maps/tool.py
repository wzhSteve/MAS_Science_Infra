from MAS.epc_aw.tools.base import BaseTool
import os
import json
import requests
from typing import Any, Dict, List, Optional

TOOL_NAME = "Maps_Tool"

LIMITATION = f"""
The {TOOL_NAME} has several limitations:
1. Uses OpenStreetMap Nominatim (no API key) for geocoding and place search;
   rate-limited (1 req/sec) — do not bulk-query.
2. Reverse geocoding returns the nearest address; may not match exact place.
3. No driving directions; for distance use coordinates + Python_Coder_Tool
   (haversine) or Browser_Tool on a maps site.
4. Not for street-view imagery.
"""

BEST_PRACTICE = f"""
For optimal results with the {TOOL_NAME}:
1. mode="geocode" with place="Fred Howard Park, Florida" → lat/lon + address parts.
2. mode="place_search" with query="clownfish sighting Fred Howard Park" for POI lookup.
3. mode="reverse" with lat/lon → address (incl. postcode for zip-code tasks).
4. For "zip code of <place>" tasks, geocode then reverse to read postcode field.
5. Combine with Python_Coder_Tool for haversine distance computation.
"""


class Maps_Tool(BaseTool):
    """Geocoding / place search / reverse geocoding via OpenStreetMap Nominatim."""

    require_llm_engine = False

    def __init__(self):
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description=(
                "Geographic lookup tool using OpenStreetMap Nominatim. Modes: "
                "geocode (place → lat/lon + address), place_search (query → POIs), "
                "reverse (lat/lon → address + postcode). Use for GAIA tasks asking "
                "for zip codes, coordinates, place identification, or distances. "
                "No API key required."
            ),
            tool_version="1.0.0",
            input_types={
                "place": "str - place name for mode='geocode'",
                "query": "str - free-text POI query for mode='place_search'",
                "lat": "float - latitude for mode='reverse'",
                "lon": "float - longitude for mode='reverse'",
                "mode": "str - geocode | place_search | reverse (default geocode)",
            },
            output_type="dict - geo results with success status",
            demo_commands=[
                {
                    "command": 'execution = tool.execute(mode="geocode", place="Fred Howard Park, Florida")',
                    "description": "Geocode a place to lat/lon and address."
                },
                {
                    "command": 'execution = tool.execute(mode="reverse", lat=28.21, lon=-82.79)',
                    "description": "Reverse geocode to get postcode/zip."
                },
            ],
            user_metadata={"limitations": LIMITATION, "best_practices": BEST_PRACTICE},
        )
        self.nominatim = "https://nominatim.openstreetmap.org"

    def execute(self, mode: str = "geocode", place: Optional[str] = None,
                query: Optional[str] = None, lat: Optional[float] = None,
                lon: Optional[float] = None) -> Dict[str, Any]:
        try:
            if mode == "geocode":
                return self._geocode(place or "")
            if mode == "place_search":
                return self._place_search(query or "")
            if mode == "reverse":
                return self._reverse(lat, lon)
            return {"success": False, "error": f"unknown mode: {mode}"}
        except Exception as e:
            return {"success": False, "error": f"{type(e).__name__}: {e}"}

    def _geocode(self, place: str) -> Dict[str, Any]:
        if not place:
            return {"success": False, "error": "mode='geocode' requires `place`"}
        resp = requests.get(
            f"{self.nominatim}/search",
            params={"q": place, "format": "json", "addressdetails": 1, "limit": 5},
            headers={"User-Agent": "EPC_AW-MAS/1.0 (gaia agent)"},
            timeout=20,
        )
        resp.raise_for_status()
        results = resp.json()
        return {"success": True, "mode": "geocode", "place": place, "results": results}

    def _place_search(self, query: str) -> Dict[str, Any]:
        resp = requests.get(
            f"{self.nominatim}/search",
            params={"q": query, "format": "json", "addressdetails": 1, "limit": 10},
            headers={"User-Agent": "EPC_AW-MAS/1.0 (gaia agent)"},
            timeout=20,
        )
        resp.raise_for_status()
        return {"success": True, "mode": "place_search", "query": query,
                "results": resp.json()}

    def _reverse(self, lat: Optional[float], lon: Optional[float]) -> Dict[str, Any]:
        if lat is None or lon is None:
            return {"success": False, "error": "mode='reverse' requires lat and lon"}
        resp = requests.get(
            f"{self.nominatim}/reverse",
            params={"lat": lat, "lon": lon, "format": "json", "addressdetails": 1},
            headers={"User-Agent": "EPC_AW-MAS/1.0 (gaia agent)"},
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
        address = data.get("address") or {}
        return {"success": True, "mode": "reverse", "lat": lat, "lon": lon,
                "address": address,
                "postcode": address.get("postcode"),
                "display_name": data.get("display_name")}

    def get_metadata(self):
        return super().get_metadata()


if __name__ == "__main__":
    tool = Maps_Tool()
    print(json.dumps(tool.get_metadata(), indent=2, ensure_ascii=False))
    print("Done!")
