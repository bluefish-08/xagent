"""
Pdf parser providers module.

This module provides various pdf parser backends
"""

from xagent.providers.pdf_parser.base import (
    DocumentParser,
    FigureParsing,
    FullTextResult,
    LocalParsing,
    ParsedFigures,
    ParsedTextSegment,
    ParseResult,
    RemoteParsing,
    SegmentedTextResult,
    TextParsing,
)

from .basic import PdfPlumberParser, PyMuPdfParser, PyPdfParser, UnstructuredParser
from .deepdoc import DeepDocParser

__all__ = [
    "ParseResult",
    "FigureParsing",
    "DeepDocParser",
    "PyPdfParser",
    "PdfPlumberParser",
    "UnstructuredParser",
    "PyMuPdfParser",
    "DocumentParser",
    "TextParsing",
    "FullTextResult",
    "SegmentedTextResult",
    "LocalParsing",
    "RemoteParsing",
    "ParsedTextSegment",
    "ParsedFigures",
]
