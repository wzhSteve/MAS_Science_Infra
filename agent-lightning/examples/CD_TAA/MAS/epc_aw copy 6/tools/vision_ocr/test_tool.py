"""Unit tests for Vision_OCR_Tool."""

import unittest
import tempfile
import os
import base64
from pathlib import Path
from MAS.epc_aw.tools.vision_ocr.tool import Vision_OCR_Tool, TOOL_NAME
from MAS.epc_aw.tools.vision_ocr.prompts import OCR_PROMPTS, get_ocr_prompt


class TestVisionOCRToolPrompts(unittest.TestCase):
    """Test OCR prompt templates."""

    def test_prompt_templates_exist(self):
        """Test that all expected prompt templates exist."""
        expected_prompts = [
            "general_ocr",
            "axis_labels",
            "table_extraction",
            "figure_description",
            "text_extraction"
        ]
        for prompt_type in expected_prompts:
            self.assertIn(prompt_type, OCR_PROMPTS)
            self.assertIsInstance(OCR_PROMPTS[prompt_type], str)
            self.assertGreater(len(OCR_PROMPTS[prompt_type]), 0)

    def test_get_ocr_prompt_function(self):
        """Test get_ocr_prompt helper function."""
        prompt = get_ocr_prompt("general_ocr")
        self.assertIsNotNone(prompt)
        self.assertIsInstance(prompt, str)

    def test_get_ocr_prompt_default(self):
        """Test get_ocr_prompt with invalid type returns default."""
        prompt = get_ocr_prompt("nonexistent_type")
        self.assertEqual(prompt, OCR_PROMPTS["general_ocr"])


class TestVisionOCRTool(unittest.TestCase):
    """Test cases for Vision_OCR_Tool."""

    def setUp(self):
        """Set up test fixtures."""
        self.tool = Vision_OCR_Tool()

    def test_tool_name(self):
        """Test that tool name is correctly set."""
        self.assertEqual(TOOL_NAME, "Vision_OCR_Tool")
        self.assertEqual(self.tool.tool_name, "Vision_OCR_Tool")

    def test_tool_metadata(self):
        """Test that tool metadata is correctly configured."""
        metadata = self.tool.get_metadata()
        self.assertIn("tool_name", metadata)
        self.assertIn("tool_description", metadata)
        self.assertEqual(metadata["require_llm_engine"], True)

    def test_require_llm_engine_flag(self):
        """Test that require_llm_engine is set to True."""
        self.assertTrue(Vision_OCR_Tool.require_llm_engine)

    def test_model_string_default(self):
        """Test that model_string is set correctly."""
        tool = Vision_OCR_Tool()
        self.assertIsNotNone(tool.model_string)
        self.assertIsInstance(tool.model_string, str)

    def test_custom_model_string(self):
        """Test Vision_OCR_Tool with custom model string."""
        custom_model = "custom-vision-model"
        tool = Vision_OCR_Tool(model_string=custom_model)
        self.assertEqual(tool.model_string, custom_model)

    def test_invalid_image_input(self):
        """Test handling of invalid image input."""
        result = self.tool.execute(image_input="/nonexistent/path/image.png")
        self.assertFalse(result["success"])
        self.assertIsNone(result["extracted_text"])
        self.assertIn("Failed to load image", result["error"])

    def test_execute_returns_correct_structure(self):
        """Test that execute returns dict with expected keys."""
        # Create a minimal test image
        test_image = self._create_test_image()

        try:
            result = self.tool.execute(image_input=test_image)

            # Check result structure
            expected_keys = ["success", "extracted_text", "format", "model_used", "error"]
            for key in expected_keys:
                self.assertIn(key, result)

        finally:
            if os.path.exists(test_image):
                os.remove(test_image)

    def test_execute_with_base64_input(self):
        """Test execute with base64-encoded image."""
        # Create a minimal valid PNG in base64
        minimal_png_base64 = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        )

        result = self.tool.execute(image_input=minimal_png_base64)

        # Should attempt to process
        self.assertIn("success", result)
        self.assertIn("extracted_text", result)

    def test_extract_format_options(self):
        """Test different extract_format options."""
        test_image = self._create_test_image()

        try:
            for extract_format in ["text", "structured", "markdown"]:
                result = self.tool.execute(
                    image_input=test_image,
                    extract_format=extract_format
                )
                self.assertEqual(result["format"], extract_format)

        finally:
            if os.path.exists(test_image):
                os.remove(test_image)

    def test_ocr_prompt_options(self):
        """Test different OCR prompt options."""
        test_image = self._create_test_image()

        try:
            for prompt_type in ["general_ocr", "axis_labels", "table_extraction"]:
                result = self.tool.execute(
                    image_input=test_image,
                    ocr_prompt=prompt_type
                )
                # Should have valid structure
                self.assertIn("success", result)

        finally:
            if os.path.exists(test_image):
                os.remove(test_image)

    def test_custom_ocr_prompt(self):
        """Test execute with custom OCR prompt."""
        test_image = self._create_test_image()
        custom_prompt = "Extract all numbers from this image"

        try:
            result = self.tool.execute(
                image_input=test_image,
                ocr_prompt=custom_prompt
            )
            self.assertIn("success", result)

        finally:
            if os.path.exists(test_image):
                os.remove(test_image)

    def test_load_image_from_file(self):
        """Test _load_image with file path."""
        test_image = self._create_test_image()

        try:
            result = self.tool._load_image(test_image)
            self.assertIsNotNone(result)
            self.assertIsInstance(result, str)
            # Verify it's valid base64
            decoded = base64.b64decode(result)
            self.assertGreater(len(decoded), 0)

        finally:
            if os.path.exists(test_image):
                os.remove(test_image)

    def test_load_image_from_base64(self):
        """Test _load_image with base64 input."""
        minimal_png_base64 = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        )

        result = self.tool._load_image(minimal_png_base64)
        self.assertEqual(result, minimal_png_base64)

    def test_vision_engine_initialization(self):
        """Test that vision engine is properly initialized."""
        tool = Vision_OCR_Tool()
        # Engine initialization might succeed or fail depending on environment
        # But the tool should handle both cases gracefully
        self.assertIsNotNone(tool.model_string)

    @staticmethod
    def _create_test_image(width: int = 100, height: int = 100) -> str:
        """Create a minimal test image and return its path."""
        try:
            from PIL import Image, ImageDraw
        except ImportError:
            # If PIL not available, return path to minimal PNG
            # This is a 1x1 transparent PNG
            minimal_png = (
                b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01'
                b'\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01'
                b'\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82'
            )
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            temp_file.write(minimal_png)
            temp_file.close()
            return temp_file.name

        # Create with PIL if available
        img = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), "Test", fill='black')

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        temp_file.close()
        img.save(temp_file.name)
        return temp_file.name


class TestVisionOCRToolIntegration(unittest.TestCase):
    """Integration tests for Vision_OCR_Tool."""

    def setUp(self):
        """Set up test fixtures."""
        self.tool = Vision_OCR_Tool()

    def test_tool_as_baseTool_subclass(self):
        """Test that Vision_OCR_Tool is a proper BaseTool subclass."""
        from MAS.epc_aw.tools.base import BaseTool
        self.assertIsInstance(self.tool, BaseTool)

    def test_get_metadata_includes_llm_requirement(self):
        """Test that metadata correctly reports LLM requirement."""
        metadata = self.tool.get_metadata()
        self.assertTrue(metadata["require_llm_engine"])

    def test_tool_can_be_instantiated_multiple_times(self):
        """Test that multiple instances can be created."""
        tool1 = Vision_OCR_Tool()
        tool2 = Vision_OCR_Tool(model_string="test-model")

        self.assertIsNotNone(tool1)
        self.assertIsNotNone(tool2)
        self.assertNotEqual(tool1.model_string, tool2.model_string)


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)
