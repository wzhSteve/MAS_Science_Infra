"""OCR prompts for Vision_OCR_Tool."""

OCR_PROMPTS = {
    "general_ocr": """Please carefully identify all text, numbers, and symbols in the image.
Return the recognized content while maintaining the original layout and formatting.""",

    "axis_labels": """This is a chart or data visualization image.
Please identify and extract all axis labels.
Return in format: X-axis: [...], Y-axis: [...], Z-axis: [...]""",

    "table_extraction": """This is a data table.
Please extract the table content in a structured format (CSV, JSON, or table).
Preserve all data and structure.""",

    "figure_description": """Please describe all key information, titles, legends, and data in this figure.
Output in structured Markdown format.""",

    "text_extraction": """Extract all text from this image.
Preserve the original formatting and structure as much as possible.""",
}


def get_ocr_prompt(prompt_type: str) -> str:
    """Get an OCR prompt template by type."""
    return OCR_PROMPTS.get(prompt_type, OCR_PROMPTS["general_ocr"])
