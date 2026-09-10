"""Integration tests for Screenshot_Tool + Vision_OCR_Tool workflow."""

import unittest
import tempfile
import os
from pathlib import Path
from MAS.epc_aw.tools.screenshot.tool import Screenshot_Tool
from MAS.epc_aw.tools.vision_ocr.tool import Vision_OCR_Tool


class TestScreenshotVisionOCRIntegration(unittest.TestCase):
    """Integration tests for combined Screenshot + Vision_OCR workflow."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.screenshot_tool = Screenshot_Tool(output_dir=self.temp_dir)
        self.ocr_tool = Vision_OCR_Tool()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_screenshot_then_ocr_workflow(self):
        """Test complete workflow: Screenshot URL -> OCR image."""
        # Step 1: Take screenshot
        screenshot_result = self.screenshot_tool.execute(
            url="https://example.com",
            full_page=False
        )

        if screenshot_result["success"]:
            # Step 2: Extract text from screenshot
            image_path = screenshot_result["image_path"]
            ocr_result = self.ocr_tool.execute(
                image_input=image_path,
                ocr_prompt="general_ocr"
            )

            # Verify workflow structure
            self.assertIsNotNone(ocr_result)
            self.assertIn("success", ocr_result)
            self.assertIn("extracted_text", ocr_result)

    def test_base64_workflow(self):
        """Test workflow using base64 encoding."""
        # Step 1: Take screenshot and get base64
        screenshot_result = self.screenshot_tool.execute(
            url="https://example.com",
            full_page=False
        )

        if screenshot_result["success"]:
            # Step 2: Pass base64 directly to OCR
            ocr_result = self.ocr_tool.execute(
                image_input=screenshot_result["image_base64"],
                ocr_prompt="general_ocr"
            )

            self.assertIsNotNone(ocr_result)
            self.assertIn("success", ocr_result)

    def test_multiple_extraction_formats(self):
        """Test different extraction formats."""
        screenshot_result = self.screenshot_tool.execute(
            url="https://example.com",
            full_page=False
        )

        if screenshot_result["success"]:
            image_base64 = screenshot_result["image_base64"]

            # Test different formats
            for extract_format in ["text", "structured", "markdown"]:
                ocr_result = self.ocr_tool.execute(
                    image_input=image_base64,
                    extract_format=extract_format
                )
                self.assertEqual(ocr_result["format"], extract_format)

    def test_error_propagation(self):
        """Test that errors are properly propagated."""
        # Invalid image input
        ocr_result = self.ocr_tool.execute(
            image_input="/nonexistent/image.png"
        )
        self.assertFalse(ocr_result["success"])
        self.assertIsNone(ocr_result["extracted_text"])

    def test_tool_metadata_availability(self):
        """Test that tool metadata is available for solver."""
        screenshot_meta = self.screenshot_tool.get_metadata()
        ocr_meta = self.ocr_tool.get_metadata()

        # Both should have complete metadata
        self.assertEqual(screenshot_meta["tool_name"], "Screenshot_Tool")
        self.assertEqual(ocr_meta["tool_name"], "Vision_OCR_Tool")

        # Screenshot should not require LLM
        self.assertFalse(screenshot_meta["require_llm_engine"])

        # OCR should require LLM
        self.assertTrue(ocr_meta["require_llm_engine"])

    def test_tool_discovery_compatibility(self):
        """Test that tools are compatible with auto-discovery mechanism."""
        from MAS.epc_aw.tools.base import BaseTool

        # Both should be BaseTool subclasses
        self.assertIsInstance(self.screenshot_tool, BaseTool)
        self.assertIsInstance(self.ocr_tool, BaseTool)

        # Both should have TOOL_NAME defined
        from MAS.epc_aw.tools.screenshot.tool import TOOL_NAME as SCREENSHOT_TOOL_NAME
        from MAS.epc_aw.tools.vision_ocr.tool import TOOL_NAME as OCR_TOOL_NAME

        self.assertEqual(SCREENSHOT_TOOL_NAME, "Screenshot_Tool")
        self.assertEqual(OCR_TOOL_NAME, "Vision_OCR_Tool")


class TestFindingNemoWorkflow(unittest.TestCase):
    """Test Finding Nemo use case workflow."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.screenshot_tool = Screenshot_Tool(output_dir=self.temp_dir)
        self.ocr_tool = Vision_OCR_Tool()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_usgs_data_extraction_workflow(self):
        """
        Test workflow for Finding Nemo problem:
        1. Screenshot USGS database URL
        2. Extract table data using Vision_OCR
        """
        # In actual use, this would be the USGS database URL
        # For testing, we use example.com as a substitute
        test_url = "https://example.com"

        # Step 1: Take screenshot
        screenshot_result = self.screenshot_tool.execute(
            url=test_url,
            full_page=True,
            viewport_width=1920,
            viewport_height=1080
        )

        if screenshot_result["success"]:
            # Step 2: Extract table data
            ocr_result = self.ocr_tool.execute(
                image_input=screenshot_result["image_path"],
                ocr_prompt="table_extraction",
                extract_format="structured"
            )

            self.assertIsNotNone(ocr_result)
            self.assertEqual(ocr_result["format"], "structured")


class TestAIRegulationWorkflow(unittest.TestCase):
    """Test AI Regulation use case workflow."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.screenshot_tool = Screenshot_Tool(output_dir=self.temp_dir)
        self.ocr_tool = Vision_OCR_Tool()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_paper_figure_analysis_workflow(self):
        """
        Test workflow for AI Regulation problem:
        1. Screenshot paper figure URL
        2. Extract axis labels using Vision_OCR
        """
        test_url = "https://example.com"

        # Step 1: Take screenshot
        screenshot_result = self.screenshot_tool.execute(
            url=test_url,
            full_page=False
        )

        if screenshot_result["success"]:
            # Step 2: Extract axis labels
            ocr_result = self.ocr_tool.execute(
                image_input=screenshot_result["image_base64"],
                ocr_prompt="axis_labels",
                extract_format="text"
            )

            self.assertIsNotNone(ocr_result)
            self.assertIn("extracted_text", ocr_result)


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)
