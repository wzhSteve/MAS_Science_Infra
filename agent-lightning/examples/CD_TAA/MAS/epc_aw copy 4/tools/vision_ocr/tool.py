from MAS.epc_aw.tools.base import BaseTool
from MAS.epc_aw.engine.factory import create_llm_engine
import os
import base64
from pathlib import Path
from typing import Optional, Union
from .prompts import get_ocr_prompt

TOOL_NAME = "Vision_OCR_Tool"


class Vision_OCR_Tool(BaseTool):
    """Extract text and structured data from images using Vision LLM."""

    require_llm_engine = True

    def __init__(self, model_string: Optional[str] = None):
        """
        Initialize Vision_OCR_Tool with Vision LLM engine.

        Args:
            model_string: Model name for Vision LLM (default: VISUAL_MODEL_NAME from env)
        """
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description="Extract text and structured data from images using Vision LLM.",
            tool_version="1.0.0",
            input_types={
                "image_input": "str - File path or base64-encoded image",
                "ocr_prompt": "str - Optional OCR instruction (general_ocr, axis_labels, table_extraction, figure_description, text_extraction)",
                "extract_format": "str - Output format: text, structured, markdown (default text)"
            },
            output_type="dict - Extracted text with success status",
            demo_commands=[
                {
                    "command": 'execution = tool.execute(image_input="/path/to/image.png")',
                    "description": "Extract text from image"
                },
                {
                    "command": 'execution = tool.execute(image_input="/path/to/chart.png", ocr_prompt="axis_labels")',
                    "description": "Extract axis labels from a chart"
                }
            ]
        )

        self.model_string = model_string or os.getenv("VISUAL_MODEL_NAME", "qwen-vl-ocr-latest")

        # Initialize Vision LLM engine
        try:
            self.vision_engine = create_llm_engine(
                model_string=self.model_string,
                is_multimodal=True,
                temperature=0.0  # Deterministic OCR
            )
        except Exception as e:
            self.vision_engine = None
            self.initialization_error = str(e)

    def execute(self, image_input: str, ocr_prompt: Optional[str] = None,
                extract_format: str = "text") -> dict:
        """
        Extract text and data from an image using Vision LLM.

        Args:
            image_input: File path to image or base64-encoded image data
            ocr_prompt: Custom OCR instruction or preset type (general_ocr, axis_labels, etc.)
            extract_format: Output format (text, structured, markdown)

        Returns:
            dict with keys:
                - success: bool indicating success/failure
                - extracted_text: str with recognized content (or None if failed)
                - format: str output format used
                - model_used: str model name
                - error: str error message (or None if successful)
        """
        try:
            # Check if vision engine was initialized
            if self.vision_engine is None:
                return {
                    "success": False,
                    "extracted_text": None,
                    "format": extract_format,
                    "model_used": self.model_string,
                    "error": f"Vision engine initialization failed: {getattr(self, 'initialization_error', 'Unknown error')}"
                }

            # Load image and convert to base64 if needed
            image_base64 = self._load_image(image_input)
            if image_base64 is None:
                return {
                    "success": False,
                    "extracted_text": None,
                    "format": extract_format,
                    "model_used": self.model_string,
                    "error": f"Failed to load image from {image_input}"
                }

            # Select OCR prompt
            if ocr_prompt is None:
                ocr_prompt_text = get_ocr_prompt("general_ocr")
            elif ocr_prompt in get_ocr_prompt.get("__dict__", {}):
                ocr_prompt_text = get_ocr_prompt(ocr_prompt)
            else:
                # Try to get from prompts module
                try:
                    from .prompts import OCR_PROMPTS
                    ocr_prompt_text = OCR_PROMPTS.get(ocr_prompt, ocr_prompt)
                except:
                    ocr_prompt_text = ocr_prompt

            # Build multimodal prompt
            if extract_format == "structured":
                format_instruction = "Return the result in JSON format."
            elif extract_format == "markdown":
                format_instruction = "Return the result in structured Markdown format."
            else:
                format_instruction = "Return the result as plain text."

            full_prompt = f"""{ocr_prompt_text}

{format_instruction}"""

            # Call Vision LLM with multimodal input
            # The image_base64 needs to be converted to bytes for the engine
            try:
                image_bytes = base64.b64decode(image_base64)
            except Exception:
                image_bytes = image_base64.encode('utf-8')

            # The generate method automatically detects multimodal content (bytes)
            # and routes to _generate_multimodal internally
            response = self.vision_engine.generate(
                content=[full_prompt, image_bytes]
            )

            # Handle potential error responses from LLM
            if isinstance(response, dict) and "error" in response:
                return {
                    "success": False,
                    "extracted_text": None,
                    "format": extract_format,
                    "model_used": self.model_string,
                    "error": f"Vision LLM error: {response.get('message', 'Unknown error')}"
                }

            return {
                "success": True,
                "extracted_text": response,
                "format": extract_format,
                "model_used": self.model_string,
                "error": None
            }

        except Exception as e:
            return {
                "success": False,
                "extracted_text": None,
                "format": extract_format,
                "model_used": self.model_string,
                "error": f"{type(e).__name__}: {str(e)}"
            }

    def _load_image(self, image_input: str) -> Optional[str]:
        """
        Load image from file path or return base64 if already encoded.

        Args:
            image_input: File path or base64-encoded string

        Returns:
            Base64-encoded image string, or None if failed
        """
        try:
            # Check if it's a file path
            if image_input.startswith('/') or os.path.exists(image_input):
                with open(image_input, 'rb') as f:
                    image_data = f.read()
                return base64.b64encode(image_data).decode('utf-8')
            else:
                # Assume it's already base64
                # Validate by trying to decode
                base64.b64decode(image_input)
                return image_input
        except Exception as e:
            return None
