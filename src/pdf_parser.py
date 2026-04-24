from __future__ import annotations

import argparse
from pathlib import Path

import fitz
from tqdm import tqdm

from src.text_cleaner import clean_pages
from src.utils import ensure_dir


def extract_pages_from_pdf(pdf_path: str | Path) -> list[str]:
    """Extract raw page text from a PDF with PyMuPDF."""
    pdf_path = Path(pdf_path)
    pages: list[str] = []

    with fitz.open(pdf_path) as document:
        for page in document:
            pages.append(page.get_text("text", sort=True))

    return pages


def parse_pdf_to_text(pdf_path: str | Path, output_dir: str | Path = "data/text") -> Path:
    """Parse one PDF, clean its text, and save it as a .txt file."""
    pdf_path = Path(pdf_path)
    output_dir = ensure_dir(output_dir)

    pages = extract_pages_from_pdf(pdf_path)
    cleaned_text = clean_pages(pages)

    output_path = output_dir / f"{pdf_path.stem}.txt"
    output_path.write_text(cleaned_text, encoding="utf-8")
    return output_path


def parse_pdf_directory(
    pdf_dir: str | Path = "data/raw_pdfs",
    output_dir: str | Path = "data/text",
    overwrite: bool = False,
) -> list[Path]:
    """Parse every PDF in a directory."""
    pdf_dir = Path(pdf_dir)
    output_dir = ensure_dir(output_dir)

    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    output_paths: list[Path] = []

    for pdf_path in tqdm(pdf_files, desc="Parsing PDFs"):
        output_path = output_dir / f"{pdf_path.stem}.txt"
        if output_path.exists() and not overwrite:
            output_paths.append(output_path)
            continue

        output_paths.append(parse_pdf_to_text(pdf_path, output_dir=output_dir))

    return output_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse downloaded PDFs into cleaned text files.")
    parser.add_argument("--pdf-dir", default="data/raw_pdfs", help="Directory containing PDF files.")
    parser.add_argument("--output-dir", default="data/text", help="Directory for cleaned .txt files.")
    parser.add_argument("--pdf-path", default=None, help="Optional path to a single PDF file.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing text files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.pdf_path:
        output_path = parse_pdf_to_text(args.pdf_path, output_dir=args.output_dir)
        print(f"Saved cleaned text to: {output_path}")
        return

    output_paths = parse_pdf_directory(
        pdf_dir=args.pdf_dir,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    print(f"Parsed {len(output_paths)} PDF files.")
    print(f"Text files saved to: {Path(args.output_dir)}")


if __name__ == "__main__":
    main()
