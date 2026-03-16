"""
PDF 텍스트 추출 모듈.

업로드된 PDF 파일에서 텍스트를 추출하여
에이전트 컨텍스트로 사용할 수 있는 문자열로 반환한다.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# pymupdf (fitz) import
try:
    import fitz  # pymupdf
    _HAS_FITZ = True
except ImportError:
    _HAS_FITZ = False


def extract_text_from_pdf(file_bytes: bytes, max_pages: int = 50) -> str:
    """
    PDF 바이트에서 텍스트 추출.

    Args:
        file_bytes: PDF 파일 바이트 데이터
        max_pages: 최대 처리 페이지 수

    Returns:
        추출된 텍스트 (페이지별 구분)
    """
    if not _HAS_FITZ:
        raise RuntimeError("pymupdf 패키지가 설치되지 않았습니다: pip install pymupdf")

    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages = []
    page_count = len(doc)
    total = min(page_count, max_pages)

    for i in range(total):
        page = doc[i]
        text = page.get_text("text")
        if text.strip():
            pages.append(f"--- 페이지 {i + 1}/{page_count} ---\n{text.strip()}")

    doc.close()

    if not pages:
        return "(PDF에서 텍스트를 추출할 수 없습니다. 이미지 기반 PDF일 수 있습니다.)"

    header = f"[PDF 문서 — 총 {page_count}페이지 중 {total}페이지 추출]\n\n"
    return header + "\n\n".join(pages)


def extract_text_from_file(file_bytes: bytes, filename: str) -> str:
    """
    파일 유형에 따라 텍스트 추출.

    지원 형식: PDF, TXT, CSV, MD
    """
    ext = Path(filename).suffix.lower()

    if ext == ".pdf":
        return extract_text_from_pdf(file_bytes)
    elif ext in (".txt", ".csv", ".md", ".json"):
        try:
            return file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            try:
                return file_bytes.decode("euc-kr")
            except UnicodeDecodeError:
                return file_bytes.decode("utf-8", errors="replace")
    else:
        raise ValueError(f"지원하지 않는 파일 형식입니다: {ext} (지원: PDF, TXT, CSV, MD, JSON)")
