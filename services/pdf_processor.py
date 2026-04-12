import os
from pathlib import Path
from typing import List
import fitz
import pdfplumber
from loguru import logger


class ProcessedDocument:
    def __init__(self, filename, pages, total_pages):
        self.filename = filename
        self.pages = pages
        self.total_pages = total_pages
        self.full_text = "\n\n".join(
            p["text"] for p in pages if p["text"].strip()
        )

    def __repr__(self):
        return (
            f"ProcessedDocument("
            f"filename={self.filename!r}, "
            f"pages={self.total_pages}, "
            f"chars={len(self.full_text)})"
        )


def clean_text(text):
    if not text:
        return ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


def process_pdf(pdf_path, use_fallback=False):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if not pdf_path.lower().endswith(".pdf"):
        raise ValueError(f"Not a PDF: {pdf_path}")

    filename = Path(pdf_path).name
    logger.info(f"Processing: {filename}")
    pages = []

    try:
        doc = fitz.open(pdf_path)
        for i in range(len(doc)):
            text = clean_text(doc[i].get_text("text"))
            pages.append({"page_num": i + 1, "text": text})
        doc.close()
    except Exception as e:
        logger.error(f"PyMuPDF failed: {e}")
        raise

    result = ProcessedDocument(filename, pages, len(pages))
    logger.success(
        f"Done: {filename} - {result.total_pages} pages, "
        f"{len(result.full_text):,} chars"
    )
    return result


def load_pdfs_from_folder(folder_path):
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    pdf_files = [
        str(f) for f in folder.iterdir()
        if f.suffix.lower() == ".pdf"
    ]

    if not pdf_files:
        logger.warning(f"No PDFs found in {folder_path}")
        return []

    logger.info(f"Found {len(pdf_files)} PDF(s)")
    return [process_pdf(p) for p in pdf_files]


def process_multiple_pdfs(pdf_paths):
    documents = []
    for pdf_path in pdf_paths:
        try:
            doc = process_pdf(pdf_path)
            documents.append(doc)
        except Exception as e:
            logger.error(f"Skipping {pdf_path}: {e}")
    return documents


if __name__ == "__main__":
    import sys
    docs = load_pdfs_from_folder("data")
    if not docs:
        print("No PDFs found. Add a PDF to the data/ folder!")
        sys.exit(1)
    for doc in docs:
        print("\n" + "=" * 50)
        print(f"File     : {doc.filename}")
        print(f"Pages    : {doc.total_pages}")
        print(f"Chars    : {len(doc.full_text):,}")
        print(f"Preview  :\n{doc.full_text[:400]}")
        print("=" * 50)
