import os
import shutil
from pathlib import Path
from typing import List
from loguru import logger


def ensure_directories():
    """Create required directories if they don't exist."""
    dirs = ["data", "vectorstore"]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    logger.info("Directories verified")


def save_uploaded_file(file_content: bytes, filename: str) -> str:
    """
    Save an uploaded file to the data/ directory.

    Args:
        file_content: Raw bytes of the uploaded file
        filename: Original filename

    Returns:
        Full path where file was saved
    """
    ensure_directories()

    # Sanitize filename — remove dangerous characters
    safe_filename = "".join(
        c for c in filename
        if c.isalnum() or c in "._- "
    ).strip()

    if not safe_filename:
        safe_filename = "uploaded_document.pdf"

    save_path = os.path.join("data", safe_filename)

    with open(save_path, "wb") as f:
        f.write(file_content)

    logger.info(f"Saved uploaded file: {save_path}")
    return save_path


def get_uploaded_files() -> List[str]:
    """Return list of all PDF files in data/ directory."""
    data_dir = Path("data")
    if not data_dir.exists():
        return []

    pdf_files = [
        str(f) for f in data_dir.iterdir()
        if f.suffix.lower() == ".pdf"
    ]
    return pdf_files


def clear_data_directory():
    """Remove all PDFs from data/ directory."""
    data_dir = Path("data")
    if data_dir.exists():
        for f in data_dir.iterdir():
            if f.suffix.lower() == ".pdf":
                f.unlink()
                logger.info(f"Deleted: {f}")


def clear_vectorstore():
    """Remove all files from vectorstore/ directory."""
    vs_dir = Path("vectorstore")
    if vs_dir.exists():
        shutil.rmtree(vs_dir)
        vs_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Vectorstore cleared")


def format_file_size(size_bytes: int) -> str:
    """Convert bytes to human readable size."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes/1024:.1f} KB"
    else:
        return f"{size_bytes/(1024*1024):.1f} MB"