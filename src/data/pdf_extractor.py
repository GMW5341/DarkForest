"""
PDF 텍스트 추출 모듈.

업로드된 PDF 파일에서 텍스트를 추출하여
에이전트 컨텍스트로 사용할 수 있는 문자열로 반환한다.
텍스트 레이어가 없는 이미지 기반(스캔) PDF는 OCR로 폴백한다.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

# pymupdf (fitz) import
try:
    import fitz  # pymupdf
    _HAS_FITZ = True
except ImportError:
    _HAS_FITZ = False

_HAS_TESSERACT = shutil.which("tesseract") is not None


def _extract_page_text(page: "fitz.Page", use_ocr: bool = False) -> str:
    """페이지에서 텍스트 추출. 빈 결과이고 OCR 가능하면 OCR 시도."""
    text = page.get_text("text")
    if text.strip():
        return text.strip()

    if not use_ocr or not _HAS_TESSERACT:
        return ""

    # 텍스트 레이어 없음 → OCR 폴백
    try:
        tp = page.get_textpage_ocr(full=True)
        ocr_text = page.get_text("text", textpage=tp)
        return ocr_text.strip()
    except Exception as exc:
        logger.warning(f"OCR 실패: {exc}")
        return ""


def extract_text_from_pdf(file_bytes: bytes, max_pages: int = 50) -> str:
    """
    PDF 바이트에서 텍스트 추출.

    1차: 텍스트 레이어에서 직접 추출
    2차: 텍스트가 없으면 Tesseract OCR로 폴백 (설치된 경우)

    Args:
        file_bytes: PDF 파일 바이트 데이터
        max_pages: 최대 처리 페이지 수

    Returns:
        추출된 텍스트 (페이지별 구분)
    """
    if not _HAS_FITZ:
        raise RuntimeError("pymupdf 패키지가 설치되지 않았습니다: pip install pymupdf")

    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as exc:
        raise RuntimeError(f"PDF 파일을 열 수 없습니다: {exc}") from exc

    pages = []
    ocr_pages = []
    page_count = len(doc)
    total = min(page_count, max_pages)

    # 1차: 텍스트 레이어 추출
    empty_page_indices = []
    for i in range(total):
        try:
            page = doc[i]
            text = page.get_text("text")
            if text.strip():
                pages.append((i, text.strip()))
            else:
                empty_page_indices.append(i)
        except Exception:
            logger.warning(f"페이지 {i + 1} 추출 실패, 건너뜀")
            continue

    # 2차: 빈 페이지에 OCR 시도
    if empty_page_indices and _HAS_TESSERACT:
        for i in empty_page_indices:
            try:
                page = doc[i]
                tp = page.get_textpage_ocr(full=True)
                ocr_text = page.get_text("text", textpage=tp)
                if ocr_text.strip():
                    ocr_pages.append((i, ocr_text.strip()))
            except Exception as exc:
                logger.warning(f"페이지 {i + 1} OCR 실패: {exc}")
                continue

    doc.close()

    # 결과 합치기 (페이지 순서대로)
    all_pages = sorted(pages + ocr_pages, key=lambda x: x[0])

    if not all_pages:
        if not _HAS_TESSERACT and empty_page_indices:
            return "(이미지 기반 PDF입니다. Tesseract가 설치되면 OCR로 텍스트를 추출할 수 있습니다.)"
        return "(PDF에서 텍스트를 추출할 수 없습니다.)"

    lines = []
    ocr_count = len(ocr_pages)
    for idx, text in all_pages:
        tag = " [OCR]" if idx in [p[0] for p in ocr_pages] else ""
        lines.append(f"--- 페이지 {idx + 1}/{page_count}{tag} ---\n{text}")

    method = f", OCR {ocr_count}페이지" if ocr_count else ""
    header = f"[PDF 문서 — 총 {page_count}페이지 중 {len(all_pages)}페이지 추출{method}]\n\n"
    return header + "\n\n".join(lines)


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
