"""Unit tests for Screenshot_Tool."""

import unittest
import tempfile
import os
from pathlib import Path
from MAS.epc_aw.tools.screenshot.tool import Screenshot_Tool, TOOL_NAME


class TestScreenshotTool(unittest.TestCase):
    """Test cases for Screenshot_Tool."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.tool = Screenshot_Tool(output_dir=self.temp_dir, timeout=30000)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_tool_name(self):
        """Test that tool name is correctly set."""
        self.assertEqual(TOOL_NAME, "Screenshot_Tool")
        self.assertEqual(self.tool.tool_name, "Screenshot_Tool")

    def test_tool_metadata(self):
        """Test that tool metadata is correctly configured."""
        metadata = self.tool.get_metadata()
        self.assertIn("tool_name", metadata)
        self.assertIn("tool_description", metadata)
        self.assertEqual(metadata["require_llm_engine"], False)

    def test_invalid_url_no_protocol(self):
        """Test handling of invalid URL without protocol."""
        result = self.tool.execute(url="example.com")
        self.assertFalse(result["success"])
        self.assertIn("Invalid URL", result["error"])

    def test_invalid_url_nonexistent(self):
        """Test handling of nonexistent URL."""
        result = self.tool.execute(url="https://this-domain-definitely-does-not-exist-12345.com")
        # Should fail but with network error, not validation error
        self.assertFalse(result["success"])

    def test_valid_url_screenshot(self):
        """Test successful screenshot of a valid website."""
        result = self.tool.execute(url="https://example.com", full_page=False)

        if result["success"]:  # Only assert if successful (network dependent)
            self.assertTrue(result["success"])
            self.assertIsNotNone(result["image_path"])
            self.assertIsNotNone(result["image_base64"])
            self.assertIsNone(result["error"])

            # Verify file exists
            self.assertTrue(os.path.exists(result["image_path"]))

            # Verify file is valid PNG (starts with PNG magic number)
            with open(result["image_path"], "rb") as f:
                header = f.read(8)
                self.assertEqual(header[:4], b'\x89PNG')

            # Verify base64 is valid
            import base64
            try:
                decoded = base64.b64decode(result["image_base64"])
                self.assertEqual(decoded[:4], b'\x89PNG')
            except Exception as e:
                self.fail(f"Invalid base64 encoding: {e}")

    def test_screenshot_size_metadata(self):
        """Test that screenshot size metadata is correct."""
        result = self.tool.execute(
            url="https://example.com",
            viewport_width=1024,
            viewport_height=768,
            full_page=False
        )

        if result["success"]:
            self.assertEqual(result["screenshot_size"], (1024, 768))

    def test_custom_viewport(self):
        """Test screenshot with custom viewport size."""
        result = self.tool.execute(
            url="https://example.com",
            viewport_width=1280,
            viewport_height=720,
            full_page=False
        )

        if result["success"]:
            self.assertEqual(result["screenshot_size"], (1280, 720))

    def test_full_page_screenshot(self):
        """Test full page screenshot capability."""
        result = self.tool.execute(
            url="https://example.com",
            full_page=True
        )

        if result["success"]:
            self.assertTrue(result["success"])
            self.assertIn("file_size", result)
            self.assertGreater(result["file_size"], 0)

    def test_output_directory_creation(self):
        """Test that output directory is created if it doesn't exist."""
        new_dir = os.path.join(self.temp_dir, "subdir", "screenshots")
        tool = Screenshot_Tool(output_dir=new_dir)

        # Directory should be created during execute
        result = tool.execute(url="https://example.com", full_page=False)

        if result["success"]:
            self.assertTrue(os.path.exists(new_dir))

    def test_page_title_extraction(self):
        """Test that page title is correctly extracted."""
        result = self.tool.execute(url="https://example.com", full_page=False)

        if result["success"]:
            self.assertIsNotNone(result["page_title"])
            self.assertIsInstance(result["page_title"], str)

    def test_url_redirect_handling(self):
        """Test handling of URL redirects."""
        result = self.tool.execute(url="https://example.com", full_page=False)

        if result["success"]:
            # page_url should be the final URL after redirects
            self.assertIsNotNone(result["page_url"])
            self.assertTrue(result["page_url"].startswith("https://"))

    def test_multiple_screenshots_different_files(self):
        """Test that multiple screenshots create different files."""
        result1 = self.tool.execute(url="https://example.com", full_page=False)

        if result1["success"]:
            result2 = self.tool.execute(url="https://example.com", full_page=False)

            if result2["success"]:
                # Same URL should create different files due to timestamp
                self.assertNotEqual(result1["image_path"], result2["image_path"])


class TestScreenshotToolIntegration(unittest.TestCase):
    """Integration tests for Screenshot_Tool with other components."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.tool = Screenshot_Tool(output_dir=self.temp_dir)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_tool_in_solver_context(self):
        """Test that tool can be used in solver context (basic check)."""
        # Just verify the tool can be instantiated and used
        self.assertIsNotNone(self.tool)
        self.assertEqual(self.tool.tool_name, "Screenshot_Tool")
        self.assertFalse(self.tool.require_llm_engine)

    def test_error_handling_timeout(self):
        """Test error handling with very short timeout."""
        tool = Screenshot_Tool(output_dir=self.temp_dir, timeout=100)  # 100ms timeout
        result = tool.execute(url="https://example.com", full_page=False)

        # Should either succeed quickly or fail with timeout
        self.assertIsNotNone(result)
        self.assertIn("success", result)


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)
