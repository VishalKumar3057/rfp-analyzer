"""Base document loader abstract class."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from src.models.documents import RFPDocument
from src.utils.logging import LoggerMixin


class BaseDocumentLoader(ABC, LoggerMixin):
    """Abstract base class for document loaders."""

    def __init__(self, file_path: Path):
        """Initialize the loader with a file path.

        Args:
            file_path: Path to the document file.
        """
        self.file_path = Path(file_path)
        self._validate_file()

    def _validate_file(self) -> None:
        """Validate that the file exists and is readable."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")
        if not self.file_path.is_file():
            raise ValueError(f"Path is not a file: {self.file_path}")

    @abstractmethod
    def load(self) -> RFPDocument:
        """Load and parse the document.

        Returns:
            RFPDocument: Parsed document with content and metadata.
        """
        pass

    @abstractmethod
    def extract_metadata(self) -> dict[str, Any]:
        """Extract metadata from the document.

        Returns:
            Dictionary containing document metadata.
        """
        pass

    @abstractmethod
    def extract_sections(self) -> list[dict[str, Any]]:
        """Extract document sections with their hierarchy.

        Returns:
            List of section dictionaries with title, content, and hierarchy info.
        """
        pass

    def get_project_name(self) -> str:
        """Extract or generate project name from document.

        Returns:
            Project name string.
        """
        # Default implementation uses filename
        return self.file_path.stem.replace("_", " ").replace("-", " ").title()

    @property
    def supported_extensions(self) -> list[str]:
        """Get list of supported file extensions."""
        return []

