from MAS.epc_aw.tools.base import BaseTool
import os
import json
import base64
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

TOOL_NAME = "YouTube_Tool"

LIMITATION = f"""
The {TOOL_NAME} has several limitations:
1. Requires `yt-dlp` installed for metadata/keyframes, and
   `youtube_transcript_api` for transcripts (network access needed).
2. Auto-generated captions may be missing or low-quality; fall back to
   keyframes + Vision_OCR_Tool or Audio_Tool in that case.
3. Keyframe extraction uses ffmpeg if available; otherwise returns metadata only.
4. Not for computation; pipe extracted numbers to Python_Coder_Tool.
"""

BEST_PRACTICE = f"""
For optimal results with the {TOOL_NAME}:
1. mode="transcript" first (cheapest); if empty, mode="keyframes" → Vision_OCR_Tool.
2. mode="keyframes" with n_frames=N samples N frames evenly → base64 PNGs.
3. mode="audio" extracts audio track → Audio_Tool for STT.
4. mode="metadata" returns title/duration/channel for identification tasks.
5. For "what color/text appears at timestamp T" tasks, use keyframes at T.
"""


class YouTube_Tool(BaseTool):
    """Extract transcript / keyframes / audio / metadata from a YouTube URL."""

    require_llm_engine = False

    def __init__(self, output_dir: Optional[str] = None):
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description=(
                "Extract content from YouTube videos: transcript (captions), keyframes "
                "(sampled PNG images for Vision_OCR_Tool), audio track (for Audio_Tool), "
                "or metadata (title/duration/channel). Use for GAIA tasks requiring video "
                "parsing, on-screen text, color recognition, or spoken content."
            ),
            tool_version="1.0.0",
            input_types={
                "url": "str - YouTube watch URL",
                "mode": "str - transcript | keyframes | audio | metadata (default transcript)",
                "n_frames": "int - frames to sample for keyframes (default 8)",
                "timestamps": "list[float] - specific seconds to capture (optional)",
            },
            output_type="dict - transcript text / base64 frames / audio path / metadata",
            demo_commands=[
                {
                    "command": 'execution = tool.execute(url="https://www.youtube.com/watch?v=xxxx", mode="transcript")',
                    "description": "Fetch the video transcript/captions."
                },
                {
                    "command": 'execution = tool.execute(url="https://www.youtube.com/watch?v=xxxx", mode="keyframes", n_frames=10)',
                    "description": "Sample 10 keyframes as base64 for Vision_OCR_Tool."
                },
            ],
            user_metadata={"limitations": LIMITATION, "best_practices": BEST_PRACTICE},
        )
        self.output_dir = output_dir

    def execute(self, url: str, mode: str = "transcript",
                n_frames: int = 8, timestamps: Optional[List[float]] = None) -> Dict[str, Any]:
        try:
            if mode == "transcript":
                return self._transcript(url)
            if mode == "metadata":
                return self._metadata(url)
            if mode == "keyframes":
                return self._keyframes(url, n_frames, timestamps)
            if mode == "audio":
                return self._audio(url)
            return {"success": False, "error": f"unknown mode: {mode}"}
        except Exception as e:
            return {"success": False, "error": f"{type(e).__name__}: {e}"}

    def _transcript(self, url: str) -> Dict[str, Any]:
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
        except ImportError:
            return {"success": False, "error": "youtube_transcript_api not installed"}
        video_id = self._video_id(url)
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            text = " ".join(seg["text"] for seg in transcript)
            return {"success": True, "mode": "transcript", "video_id": video_id,
                    "text": text, "segments": transcript[:50]}
        except Exception as e:
            return {"success": False, "error": f"transcript unavailable: {e}"}

    def _metadata(self, url: str) -> Dict[str, Any]:
        try:
            out = subprocess.run(
                ["yt-dlp", "--dump-json", "--no-playlist", url],
                capture_output=True, text=True, timeout=60,
            )
            if out.returncode != 0:
                return {"success": False, "error": f"yt-dlp failed: {out.stderr[:200]}"}
            data = json.loads(out.stdout)
            return {"success": True, "mode": "metadata",
                    "title": data.get("title"), "duration": data.get("duration"),
                    "channel": data.get("channel"), "upload_date": data.get("upload_date"),
                    "url": url}
        except FileNotFoundError:
            return {"success": False, "error": "yt-dlp not installed"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _keyframes(self, url: str, n_frames: int, timestamps: Optional[List[float]]) -> Dict[str, Any]:
        if not self._has_ffmpeg():
            return {"success": False, "error": "ffmpeg not available for keyframe extraction"}
        out_dir = Path(self.output_dir or "/tmp/youtube_frames")
        out_dir.mkdir(parents=True, exist_ok=True)
        # Download video (best mp4, limit to small)
        try:
            dl = subprocess.run(
                ["yt-dlp", "-f", "mp4", "-o", str(out_dir / "vid.mp4"),
                 "--no-playlist", url],
                capture_output=True, text=True, timeout=180,
            )
            if dl.returncode != 0:
                return {"success": False, "error": f"yt-dlp download failed: {dl.stderr[:200]}"}
        except FileNotFoundError:
            return {"success": False, "error": "yt-dlp not installed"}

        vid = out_dir / "vid.mp4"
        frames_b64: List[str] = []
        ts_list = timestamps or [i / max(1, n_frames - 1) for i in range(n_frames)]
        for i, t in enumerate(ts_list):
            frame_path = out_dir / f"frame_{i}.png"
            subprocess.run(
                ["ffmpeg", "-y", "-ss", str(t), "-i", str(vid),
                 "-frames:v", "1", str(frame_path)],
                capture_output=True, timeout=30,
            )
            if frame_path.exists():
                frames_b64.append(base64.b64encode(frame_path.read_bytes()).decode("utf-8"))
        return {"success": True, "mode": "keyframes", "frame_count": len(frames_b64),
                "frames_base64": frames_b64, "timestamps": ts_list}

    def _audio(self, url: str) -> Dict[str, Any]:
        out_dir = Path(self.output_dir or "/tmp/youtube_audio")
        out_dir.mkdir(parents=True, exist_ok=True)
        audio_path = out_dir / "audio.mp3"
        try:
            dl = subprocess.run(
                ["yt-dlp", "-x", "--audio-format", "mp3",
                 "-o", str(audio_path), "--no-playlist", url],
                capture_output=True, text=True, timeout=180,
            )
            if dl.returncode != 0:
                return {"success": False, "error": f"audio extract failed: {dl.stderr[:200]}"}
            return {"success": True, "mode": "audio", "audio_path": str(audio_path)}
        except FileNotFoundError:
            return {"success": False, "error": "yt-dlp not installed"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    def _video_id(url: str) -> str:
        m = re.search(r"(?:v=|youtu\.be/)([\w-]{11})", url)
        return m.group(1) if m else url

    @staticmethod
    def _has_ffmpeg() -> bool:
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5)
            return True
        except Exception:
            return False

    def get_metadata(self):
        return super().get_metadata()


if __name__ == "__main__":
    tool = YouTube_Tool()
    print(json.dumps(tool.get_metadata(), indent=2, ensure_ascii=False))
    print("Done!")
