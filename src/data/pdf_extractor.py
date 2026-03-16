"""
PDF 텍스트 추출 모듈.

업로드된 PDF 파일에서 텍스트를 추출하여
에이전트 컨텍스트로 사용할 수 있는 문자열로 반환한다.

추출 우선순위:
  1차: 텍스트 레이어 직접 추출 (가장 빠름)
  2차: Tesseract OCR 폴백 (스캔된 텍스트)
  3차: Claude Vision 해석 (그래프, 차트, 모식도 등 비주얼 컨텐츠)
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

# Vision 해석 프롬프트
_VISION_SYSTEM = (
    "당신은 투자 분석 보고서의 시각 자료(차트, 그래프, 모식도, 테이블 이미지 등)를 "
    "정확하게 텍스트로 변환하는 전문가입니다."
)

_VISION_PROMPT = """\
이 PDF 페이지 이미지를 분석하여 텍스트로 변환해주세요.

다음 규칙을 따르세요:
1. **그래프/차트**: 제목, 축 라벨, 범례를 기록하고, 데이터 추이와 핵심 수치를 서술형으로 설명하세요.
   예: "사모 대출 규모는 2020년 $800B에서 2024년 $1.7T로 약 2배 증가"
2. **모식도/다이어그램**: 구성 요소들과 그들 간의 관계(화살표, 연결선)를 구조화하여 설명하세요.
   예: "데이터센터 → 엣지서버 → 통신사 → 위성 순으로 연결, 양방향 통신"
3. **테이블 이미지**: 마크다운 테이블 형식으로 재구성하세요.
4. **일반 텍스트가 포함된 경우**: OCR처럼 텍스트를 그대로 옮기되, 시각 요소의 해석도 포함하세요.
5. **숫자/데이터**: 가능한 한 정확한 수치를 읽어내세요. 불확실하면 "약 ~" 표기를 사용하세요.

출력은 한국어로, 구조화된 형태로 작성하세요."""


def _page_has_visual_content(page: "fitz.Page") -> bool:
    """페이지에 이미지나 벡터 드로잉이 있는지 확인."""
    # 삽입된 이미지 확인
    if page.get_images(full=False):
        return True
    # 벡터 드로잉 확인 — 드로잉 객체 수 또는 총 아이템 수로 판단
    drawings = page.get_drawings()
    if len(drawings) > 3:
        return True
    total_items = sum(len(d.get("items", [])) for d in drawings)
    if total_items > 5:  # 선, 곡선, 사각형 등 총 요소가 5개 초과
        return True
    return False


def _render_page_to_png(page: "fitz.Page", dpi: int = 150) -> bytes:
    """페이지를 PNG 이미지 바이트로 렌더링."""
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    return pix.tobytes("png")


def extract_text_from_pdf(file_bytes: bytes, max_pages: int = 50) -> str:
    """
    PDF 바이트에서 텍스트 추출 (동기, Vision 없이).

    1차: 텍스트 레이어에서 직접 추출
    2차: 텍스트가 없으면 Tesseract OCR로 폴백 (설치된 경우)

    그래프/차트 등 비주얼 컨텐츠는 extract_text_from_pdf_with_vision()으로
    처리할 수 있다.
    """
    if not _HAS_FITZ:
        raise RuntimeError("pymupdf 패키지가 설치되지 않았습니다: pip install pymupdf")

    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as exc:
        raise RuntimeError(f"PDF 파일을 열 수 없습니다: {exc}") from exc

    pages = []
    ocr_pages = []
    visual_only_pages = []
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
    still_empty = []
    if empty_page_indices and _HAS_TESSERACT:
        for i in empty_page_indices:
            try:
                page = doc[i]
                tp = page.get_textpage_ocr(full=True)
                ocr_text = page.get_text("text", textpage=tp)
                if ocr_text.strip():
                    ocr_pages.append((i, ocr_text.strip()))
                else:
                    still_empty.append(i)
            except Exception as exc:
                logger.warning(f"페이지 {i + 1} OCR 실패: {exc}")
                still_empty.append(i)
                continue
    else:
        still_empty = empty_page_indices

    # 비주얼 컨텐츠가 있는 빈 페이지 식별
    for i in still_empty:
        try:
            page = doc[i]
            if _page_has_visual_content(page):
                visual_only_pages.append(i)
        except Exception:
            pass

    doc.close()

    # 결과 합치기 (페이지 순서대로)
    all_pages = sorted(pages + ocr_pages, key=lambda x: x[0])
    ocr_idx_set = {p[0] for p in ocr_pages}

    if not all_pages and not visual_only_pages:
        if not _HAS_TESSERACT and empty_page_indices:
            return "(이미지 기반 PDF입니다. Tesseract가 설치되면 OCR로 텍스트를 추출할 수 있습니다.)"
        return "(PDF에서 텍스트를 추출할 수 없습니다.)"

    lines = []
    ocr_count = len(ocr_pages)
    for idx, text in all_pages:
        tag = " [OCR]" if idx in ocr_idx_set else ""
        lines.append(f"--- 페이지 {idx + 1}/{page_count}{tag} ---\n{text}")

    # 비주얼 전용 페이지 안내
    if visual_only_pages:
        for i in visual_only_pages:
            lines.append(
                f"--- 페이지 {i + 1}/{page_count} [시각자료] ---\n"
                f"(이 페이지는 그래프/차트/모식도 등 시각 컨텐츠로 구성되어 있습니다. "
                f"Vision 분석이 필요합니다.)"
            )

    method_parts = []
    if ocr_count:
        method_parts.append(f"OCR {ocr_count}페이지")
    if visual_only_pages:
        method_parts.append(f"시각자료 {len(visual_only_pages)}페이지(Vision 대기)")
    method = f", {', '.join(method_parts)}" if method_parts else ""
    extracted = len(all_pages)
    header = f"[PDF 문서 — 총 {page_count}페이지 중 {extracted}페이지 추출{method}]\n\n"
    return header + "\n\n".join(lines)


async def extract_text_from_pdf_with_vision(
    file_bytes: bytes,
    claude_client: "ClaudeClient",
    max_pages: int = 50,
    vision_max_pages: int = 10,
) -> str:
    """
    PDF 바이트에서 텍스트 추출 — Vision 포함 비동기 버전.

    1차: 텍스트 레이어 직접 추출
    2차: OCR 폴백
    3차: 비주얼 페이지를 Claude Vision으로 해석

    Args:
        file_bytes: PDF 파일 바이트 데이터
        claude_client: Vision API 호출을 위한 ClaudeClient 인스턴스
        max_pages: 최대 처리 페이지 수
        vision_max_pages: Vision으로 처리할 최대 페이지 수 (API 비용 제어)
    """
    if not _HAS_FITZ:
        raise RuntimeError("pymupdf 패키지가 설치되지 않았습니다: pip install pymupdf")

    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as exc:
        raise RuntimeError(f"PDF 파일을 열 수 없습니다: {exc}") from exc

    pages = []
    ocr_pages = []
    vision_pages = []
    page_count = len(doc)
    total = min(page_count, max_pages)

    # 1차: 텍스트 레이어 추출
    empty_page_indices = []
    for i in range(total):
        try:
            page = doc[i]
            text = page.get_text("text")
            if text.strip():
                pages.append((i, text.strip(), "text"))
            else:
                empty_page_indices.append(i)
        except Exception:
            logger.warning(f"페이지 {i + 1} 추출 실패, 건너뜀")
            continue

    # 2차: 빈 페이지에 OCR 시도
    still_empty = []
    if empty_page_indices and _HAS_TESSERACT:
        for i in empty_page_indices:
            try:
                page = doc[i]
                tp = page.get_textpage_ocr(full=True)
                ocr_text = page.get_text("text", textpage=tp)
                if ocr_text.strip():
                    ocr_pages.append((i, ocr_text.strip(), "ocr"))
                else:
                    still_empty.append(i)
            except Exception as exc:
                logger.warning(f"페이지 {i + 1} OCR 실패: {exc}")
                still_empty.append(i)
    else:
        still_empty = empty_page_indices

    # 3차: 비주얼 페이지를 이미지로 렌더링 → Claude Vision 호출
    # 텍스트가 적지만 비주얼 콘텐츠가 많은 페이지도 포함
    vision_candidates = []
    for i in still_empty:
        try:
            page = doc[i]
            if _page_has_visual_content(page):
                vision_candidates.append(i)
        except Exception:
            pass

    # 텍스트가 있지만 비주얼 콘텐츠도 상당한 페이지 (차트+캡션 등)
    for i, text, _ in pages:
        try:
            page = doc[i]
            if _page_has_visual_content(page) and len(text) < 200:
                vision_candidates.append(i)
        except Exception:
            pass

    vision_candidates = sorted(set(vision_candidates))[:vision_max_pages]

    if vision_candidates:
        logger.info(f"Vision 분석 대상: {len(vision_candidates)}페이지")
        for i in vision_candidates:
            try:
                page = doc[i]
                png_bytes = _render_page_to_png(page, dpi=150)
                description = await claude_client.ask_with_images(
                    text_prompt=_VISION_PROMPT,
                    images=[(png_bytes, "image/png")],
                    system=_VISION_SYSTEM,
                )
                if description.strip():
                    vision_pages.append((i, description.strip(), "vision"))
            except Exception as exc:
                logger.warning(f"페이지 {i + 1} Vision 분석 실패: {exc}")

    doc.close()

    # 결과 합치기 (페이지 순서대로, 중복 제거)
    all_results: dict[int, tuple[str, str]] = {}
    for idx, text, method in pages:
        all_results[idx] = (text, method)
    for idx, text, method in ocr_pages:
        all_results[idx] = (text, method)
    # Vision 결과: 기존 텍스트에 추가하거나 대체
    for idx, text, method in vision_pages:
        if idx in all_results:
            existing_text, existing_method = all_results[idx]
            all_results[idx] = (
                f"{existing_text}\n\n[시각자료 해석]\n{text}",
                f"{existing_method}+vision",
            )
        else:
            all_results[idx] = (text, method)

    if not all_results:
        return "(PDF에서 텍스트를 추출할 수 없습니다.)"

    lines = []
    method_tags = {"text": "", "ocr": " [OCR]", "vision": " [Vision]"}
    ocr_count = len(ocr_pages)
    vision_count = len(vision_pages)

    for idx in sorted(all_results.keys()):
        text, method = all_results[idx]
        tag = method_tags.get(method, f" [{method}]")
        lines.append(f"--- 페이지 {idx + 1}/{page_count}{tag} ---\n{text}")

    method_parts = []
    if ocr_count:
        method_parts.append(f"OCR {ocr_count}페이지")
    if vision_count:
        method_parts.append(f"Vision {vision_count}페이지")
    method_str = f", {', '.join(method_parts)}" if method_parts else ""
    header = f"[PDF 문서 — 총 {page_count}페이지 중 {len(all_results)}페이지 추출{method_str}]\n\n"
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


async def extract_text_from_file_with_vision(
    file_bytes: bytes,
    filename: str,
    claude_client: "ClaudeClient",
) -> str:
    """
    파일 유형에 따라 텍스트 추출 — Vision 포함 비동기 버전.

    PDF의 경우 그래프/차트/모식도를 Claude Vision으로 해석한다.
    """
    ext = Path(filename).suffix.lower()

    if ext == ".pdf":
        return await extract_text_from_pdf_with_vision(file_bytes, claude_client)
    else:
        return extract_text_from_file(file_bytes, filename)
