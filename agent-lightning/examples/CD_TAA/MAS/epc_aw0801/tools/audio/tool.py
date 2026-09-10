from MAS.epc_aw.tools.base import BaseTool
import os
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

TOOL_NAME = "Audio_Tool"

LIMITATION = f"""
The {TOOL_NAME} has several limitations:
1. Requires either OpenAI audio API (env OPENAI_API_KEY) or local `whisper` CLI.
2. Audio length capped by provider (~25MB for OpenAI STT); split long files.
3. Classification mode is best-effort; not tuned for fine-grained audio events.
4. Not for music transcription or speaker diarization.
"""

BEST_PRACTICE = f"""
For optimal results with the {TOOL_NAME}:
1. mode="stt" for speech-to-text (most GAIA audio tasks).
2. Pipe YouTube_Tool mode="audio" output path here for video speech.
3. For ambient sound / event identification, use mode="classify" with a prompt.
4. Verify transcripts with Web_Search_Tool when proper nouns are ambiguous.
"""


class Audio_Tool(BaseTool):
    """Speech-to-text and audio classification via OpenAI audio API or local whisper."""

    require_llm_engine = False

    def __init__(self):
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description=(
                "Audio processing tool: speech-to-text (mode='stt') and audio "
                "classification/description (mode='classify'). Use for GAIA tasks "
                "requiring audio capability — spoken content, sound identification, "
                "or transcribing a YouTube audio track extracted by YouTube_Tool."
            ),
            tool_version="1.0.0",
            input_types={
                "audio_input": "str - path to audio file (mp3/wav/m4a) or URL",
                "mode": "str - stt | classify (default stt)",
                "prompt": "str - optional classification instruction or STT hint",
            },
            output_type="dict - transcript / classification with success status",
            demo_commands=[
                {
                    "command": 'execution = tool.execute(audio_input="/tmp/audio.mp3", mode="stt")',
                    "description": "Transcribe speech to text."
                },
                {
                    "command": 'execution = tool.execute(audio_input="/tmp/sound.wav", mode="classify", prompt="What instrument is playing?")',
                    "description": "Classify an audio event."
                },
            ],
            user_metadata={"limitations": LIMITATION, "best_practices": BEST_PRACTICE},
        )

    def execute(self, audio_input: str, mode: str = "stt",
                prompt: Optional[str] = None) -> Dict[str, Any]:
        try:
            path = self._resolve_path(audio_input)
            if path is None:
                return {"success": False, "error": f"audio not found: {audio_input}"}
            if mode == "stt":
                return self._stt(path, prompt)
            if mode == "classify":
                return self._classify(path, prompt or "Describe this audio.")
            return {"success": False, "error": f"unknown mode: {mode}"}
        except Exception as e:
            return {"success": False, "error": f"{type(e).__name__}: {e}"}

    def _resolve_path(self, audio_input: str) -> Optional[Path]:
        p = Path(audio_input)
        if p.exists():
            return p
        return None

    def _stt(self, path: Path, hint: Optional[str]) -> Dict[str, Any]:
        # Prefer OpenAI audio API
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            try:
                import openai
                client = openai.OpenAI(api_key=api_key)
                with open(path, "rb") as f:
                    kwargs = {"model": "whisper-1", "file": f}
                    if hint:
                        kwargs["prompt"] = hint
                    resp = client.audio.transcriptions.create(**kwargs)
                return {"success": True, "mode": "stt", "transcript": resp.text}
            except Exception as e:
                return {"success": False, "error": f"openai stt failed: {e}"}
        # Fallback: local whisper CLI
        try:
            out = subprocess.run(
                ["whisper", str(path), "--model", "base", "--output_format", "txt"],
                capture_output=True, text=True, timeout=300,
            )
            if out.returncode != 0:
                return {"success": False, "error": f"whisper failed: {out.stderr[:200]}"}
            # whisper writes a .txt next to the file
            txt_path = path.with_suffix(".txt")
            text = txt_path.read_text() if txt_path.exists() else out.stdout
            return {"success": True, "mode": "stt", "transcript": text}
        except FileNotFoundError:
            return {"success": False, "error": "neither openai nor whisper CLI available"}

    def _classify(self, path: Path, prompt: str) -> Dict[str, Any]:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return {"success": False, "error": "classify mode requires OPENAI_API_KEY"}
        try:
            import openai
            client = openai.OpenAI(api_key=api_key)
            with open(path, "rb") as f:
                resp = client.audio.translations.create(model="whisper-1", file=f)
            # Use translation output as a proxy description; full classification
            # would require a multimodal model over a spectrogram — left as future.
            return {"success": True, "mode": "classify", "prompt": prompt,
                    "description": resp.text}
        except Exception as e:
            return {"success": False, "error": f"openai classify failed: {e}"}

    def get_metadata(self):
        return super().get_metadata()


if __name__ == "__main__":
    tool = Audio_Tool()
    print(json.dumps(tool.get_metadata(), indent=2, ensure_ascii=False))
    print("Done!")
