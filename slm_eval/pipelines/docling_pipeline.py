"""Docling integration pipeline for document parsing.

This module provides a specialized pipeline for processing document formats using docling,
which can handle PDF, DOCX, PPTX, XLSX, HTML, EPUB, images, and more.

The pipeline is designed to be optional and integrates with the unified dataset pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Import docling classes
try:
    from docling import DoclingProcessor
    from docling.document import Document
    from docling.model import DocumentType
    from docling.processing import ProcessorConfig
    from docling.utils import load_resources
except ImportError:
    # Mock for testing when docling is not installed
    class Document:
        def __init__(self, content: str):
            self.content = content
            self.text = content
            self.pages = []
            self.images = []

    class ProcessorConfig:
        """Mock processor config when docling is not installed."""
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class DoclingProcessor:
        def __init__(self, config: ProcessorConfig | None = None):
            self.config = config or ProcessorConfig()

        def process(self, source: str | Path) -> Document:
            # Simplified mock implementation
            if isinstance(source, str):
                # Assume it's a file path or URL
                if source.endswith('.pdf'):
                    # Mock PDF processing
                    return Document(f"Processed PDF content from {source}")
                elif source.endswith('.docx'):
                    # Mock DOCX processing
                    return Document(f"Processed DOCX content from {source}")
                elif source.endswith('.html'):
                    # Mock HTML processing
                    return Document(f"Processed HTML content from {source}")
                else:
                    # Generic text processing
                    return Document(str(source))
            else:
                # Local file
                return Document(f"Processed file {source}")

def load_docling_config() -> ProcessorConfig:
    """Load docling processor configuration."""
    # This could be extended to load from config files
    return ProcessorConfig()

class DoclingPipeline:
    """Docling-based document processing pipeline."""

    def __init__(self, config: ProcessorConfig | None = None):
        self.config = config or load_docling_config()
        self.processor = DoclingProcessor(self.config)

    def process_file(self, source: str | Path) -> dict[str, Any]:
        """Process a single file and return structured document data."""
        try:
            # Process the file
            doc = self.processor.process(source)

            # Extract text content
            text_content = doc.text

            # Extract metadata
            metadata = {
                "source": str(source),
                "file_type": self._get_file_type(source),
                "page_count": len(doc.pages) if hasattr(doc, 'pages') else 0,
                "image_count": len(doc.images) if hasattr(doc, 'images') else 0,
            }

            return {
                "text": text_content,
                "metadata": metadata,
                "success": True,
                "error": None
            }

        except Exception as e:
            return {
                "text": "",
                "metadata": {},
                "success": False,
                "error": str(e)
            }

    def _get_file_type(self, source: str | Path) -> str:
        """Determine file type from extension."""
        if isinstance(source, str) or hasattr(source, 'suffix'):
            ext = (str(source).lower() if isinstance(source, str) else source.suffix.lower())
            if ext == '.pdf':
                return 'pdf'
            elif ext == '.docx':
                return 'docx'
            elif ext == '.pptx':
                return 'pptx'
            elif ext == '.xlsx':
                return 'xlsx'
            elif ext == '.html':
                return 'html'
            elif ext == '.epub':
                return 'epub'
            elif ext in ('.jpg', '.jpeg', '.png', '.tiff'):
                return 'image'
            else:
                return 'text'
        return 'unknown'

def process_directory(
    directory: str | Path,
    output_dir: str | Path,
    format: str = "jsonl",
    include_subdirs: bool = True
) -> list[dict[str, Any]]:
    """Process all files in a directory and save results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    files_processed = 0
    files_failed = 0

    # Get all files
    if include_subdirs:
        files = list(Path(directory).rglob("*"))
    else:
        files = list(Path(directory).glob("*"))

    for file_path in files:
        if file_path.is_file():
            try:
                result = DoclingPipeline().process_file(file_path)
                if result["success"]:
                    results.append(result)
                    # Save to file
                    if format == "jsonl":
                        with open(output_dir / f"{file_path.stem}.jsonl", "w") as f:
                            f.write(json.dumps(result) + "\n")
                    else:
                        with open(output_dir / f"{file_path.stem}.json", "w") as f:
                            json.dump(result, f, indent=2)
                    files_processed += 1
                else:
                    files_failed += 1
            except Exception:
                files_failed += 1

    return results, files_processed, files_failed

def integrate_with_pipeline(
    cases: list[dict[str, Any]],
    docling_results: list[dict[str, Any]],
    text_field: str = "text",
    id_field: str = "case_id"
) -> list[dict[str, Any]]:
    """Integrate docling results with existing cases."""
    # Build a mapping from source to docling result
    result_by_source = {}
    for result in docling_results:
        if not result["success"]:
            continue
        source = result["metadata"].get("source")
        if source:
            result_by_source[source] = result

    # Update matching cases with docling-processed text
    integrated = [dict(case) for case in cases]
    for case in integrated:
        # Check strata dict for source
        strata = case.get("strata", {})
        case_source = strata.get("source")
        if case_source and case_source in result_by_source:
            case[text_field] = result_by_source[case_source]["text"]

    return integrated
