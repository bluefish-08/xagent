"""Parser registry for managing file type to parser mappings."""

from typing import Dict, List, Set

# File extension to supported parser methods mapping
# This ensures type-based parse method consistency when allow_mixed_parse_methods=False
PARSER_COMPATIBILITY: Dict[str, List[str]] = {
    # Documents
    ".pdf": ["deepdoc", "pymupdf", "pdfplumber", "unstructured"],
    ".docx": ["docx", "unstructured"],
    ".doc": ["docx", "unstructured"],
    ".pptx": ["unstructured"],
    ".ppt": ["unstructured"],
    # Text/Markdown
    ".txt": ["text"],
    ".md": ["markdown", "commonmark"],
    ".rst": ["rst"],
    # Code files
    ".py": ["code", "python_ast"],
    ".js": ["code", "javascript"],
    ".ts": ["code", "typescript"],
    ".java": ["code", "java"],
    ".cpp": ["code", "cpp"],
    ".c": ["code", "c"],
    ".go": ["code", "go"],
    ".rs": ["code", "rust"],
    ".php": ["code", "php"],
    ".rb": ["code", "ruby"],
    ".sh": ["code", "bash"],
    ".sql": ["code", "sql"],
    # Web formats
    ".html": ["html", "beautifulsoup"],
    ".xml": ["xml"],
    ".json": ["json"],
    ".yaml": ["yaml"],
    ".yml": ["yaml"],
    # Data formats
    ".csv": ["csv"],
    ".xlsx": ["excel", "openpyxl"],
    ".xls": ["excel", "openpyxl"],
    # Images (for OCR or captioning)
    ".jpg": ["image", "image_caption"],
    ".jpeg": ["image", "image_caption"],
    ".png": ["image", "image_caption"],
    ".gif": ["image", "image_caption"],
    ".bmp": ["image", "image_caption"],
    ".tiff": ["image", "image_caption"],
    ".webp": ["image", "image_caption"],
}


def get_supported_parsers(file_extension: str) -> List[str]:
    """Get supported parser methods for a file extension.

    Args:
        file_extension: File extension (with or without leading dot)

    Returns:
        List of supported parser method names
    """
    # Normalize extension
    if not file_extension.startswith("."):
        file_extension = "." + file_extension

    file_extension = file_extension.lower()
    return PARSER_COMPATIBILITY.get(file_extension, [])


def validate_parser_compatibility(
    file_extension: str, parser_method: str, allow_mixed: bool = False
) -> bool:
    """Validate if a parser method is compatible with a file type.

    Args:
        file_extension: File extension to check
        parser_method: Parser method to validate
        allow_mixed: If True, allow any parser method

    Returns:
        True if compatible, False otherwise
    """
    if allow_mixed:
        return True

    supported_parsers = get_supported_parsers(file_extension)
    return parser_method in supported_parsers


def get_all_supported_extensions() -> Set[str]:
    """Get all supported file extensions."""
    return set(PARSER_COMPATIBILITY.keys())


def register_parser_support(file_extension: str, parser_method: str) -> None:
    """Register a new parser method for a file extension.

    This is used when adding new parsers to the system.

    Args:
        file_extension: File extension (with leading dot)
        parser_method: Parser method name to add
    """
    if not file_extension.startswith("."):
        file_extension = "." + file_extension

    file_extension = file_extension.lower()

    if file_extension not in PARSER_COMPATIBILITY:
        PARSER_COMPATIBILITY[file_extension] = []

    if parser_method not in PARSER_COMPATIBILITY[file_extension]:
        PARSER_COMPATIBILITY[file_extension].append(parser_method)
