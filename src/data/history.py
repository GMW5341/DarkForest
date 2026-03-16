"""
Persistent debate history, draft (mid-debate checkpoint) & document storage.

각 토론 결과와 업로드 문서를 JSON 파일로 영구 저장하고
피드백 루프를 통해 이후 토론에 반영한다.

토론이 최종 결론에 도달하기 전에 중단되어도 draft로 자동 저장되며,
이후 이어서 진행(resume)할 수 있다.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import DATA_DIR

logger = logging.getLogger(__name__)

# 기본 저장 경로 (DARKFOREST_DATA_DIR 환경변수 우선)
_DEFAULT_DIR = DATA_DIR / "history"


def _ensure_dir(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)


class DebateHistoryStore:
    """JSON 파일 기반 토론 히스토리 저장소."""

    def __init__(self, directory: str | Path | None = None):
        self.directory = Path(directory) if directory else _DEFAULT_DIR
        _ensure_dir(self.directory)

    def _path(self, session_id: str) -> Path:
        return self.directory / f"{session_id}.json"

    def save(self, session_id: str, record: dict[str, Any]) -> Path:
        """토론 결과 저장. 타임스탬프 자동 추가."""
        record.setdefault("session_id", session_id)
        record.setdefault("saved_at", datetime.now(timezone.utc).isoformat())
        path = self._path(session_id)
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"Debate history saved: {path}")
        return path

    def load(self, session_id: str) -> dict[str, Any] | None:
        """세션 ID로 토론 결과 조회."""
        path = self._path(session_id)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"Failed to load {path}: {e}")
            return None

    def list_all(self, limit: int = 50) -> list[dict[str, Any]]:
        """모든 토론 히스토리 목록 (최신순, 요약 정보만)."""
        records = []
        files = sorted(self.directory.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        for f in files[:limit]:
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                records.append({
                    "session_id": data.get("session_id", f.stem),
                    "mode": data.get("mode", ""),
                    "topic": data.get("topic", ""),
                    "overall_stance": data.get("overall_stance", ""),
                    "executive_summary": data.get("executive_summary", "")[:200],
                    "saved_at": data.get("saved_at", ""),
                    "round_count": data.get("round_count", 0),
                })
            except Exception:
                continue
        return records

    def delete(self, session_id: str) -> bool:
        """토론 기록 삭제."""
        path = self._path(session_id)
        if path.exists():
            path.unlink()
            return True
        return False

    def get_recent_insights(self, limit: int = 5) -> list[dict[str, Any]]:
        """최근 토론의 인사이트 요약 (에이전트 컨텍스트 주입용)."""
        records = self.list_all(limit=limit)
        return [
            {
                "mode": r["mode"],
                "topic": r["topic"],
                "overall_stance": r["overall_stance"],
                "executive_summary": r["executive_summary"],
            }
            for r in records
        ]


# ── 문서 저장소 (영구) ──

_DEFAULT_DOC_DIR = DATA_DIR / "documents"


class DocumentStore:
    """업로드 문서 영구 저장소. 누적되는 지식 베이스 역할."""

    def __init__(self, directory: str | Path | None = None):
        self.directory = Path(directory) if directory else _DEFAULT_DOC_DIR
        _ensure_dir(self.directory)
        self._index_path = self.directory / "_index.json"

    def _load_index(self) -> list[dict[str, Any]]:
        if self._index_path.exists():
            try:
                return json.loads(self._index_path.read_text(encoding="utf-8"))
            except Exception:
                return []
        return []

    def _save_index(self, index: list[dict[str, Any]]) -> None:
        self._index_path.write_text(
            json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def add(self, filename: str, text: str) -> dict[str, Any]:
        """문서 추가. 텍스트와 메타데이터를 영구 저장."""
        import hashlib
        name_hash = hashlib.sha256(filename.encode()).hexdigest()[:12]
        doc_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + name_hash
        doc_path = self.directory / f"{doc_id}.txt"
        doc_path.write_text(text, encoding="utf-8")

        entry = {
            "doc_id": doc_id,
            "filename": filename,
            "text_length": len(text),
            "added_at": datetime.now(timezone.utc).isoformat(),
        }

        index = self._load_index()
        index.append(entry)
        self._save_index(index)

        logger.info(f"Document saved: {doc_path} ({len(text)} chars)")
        return entry

    def list_all(self) -> list[dict[str, Any]]:
        """저장된 모든 문서 목록."""
        return self._load_index()

    def load_text(self, doc_id: str) -> str | None:
        """문서 텍스트 로드."""
        doc_path = self.directory / f"{doc_id}.txt"
        if doc_path.exists():
            return doc_path.read_text(encoding="utf-8")
        return None

    def delete(self, doc_id: str) -> bool:
        """문서 삭제."""
        doc_path = self.directory / f"{doc_id}.txt"
        if doc_path.exists():
            doc_path.unlink()
        index = self._load_index()
        new_index = [e for e in index if e["doc_id"] != doc_id]
        if len(new_index) < len(index):
            self._save_index(new_index)
            return True
        return False

    def get_all_context(
        self,
        max_chars_per_doc: int = 8000,
        max_total_chars: int = 40000,
    ) -> str:
        """
        모든 저장 문서의 텍스트를 에이전트 컨텍스트 문자열로 반환.
        누적된 전체 지식 베이스.

        Args:
            max_chars_per_doc: 문서당 최대 글자수
            max_total_chars: 전체 컨텍스트 총 글자수 예산 (~12,000 토큰)
        """
        index = self._load_index()
        if not index:
            return ""

        # 문서가 많으면 문서당 할당량을 줄여서 총량 내에 맞춤
        per_doc_budget = min(max_chars_per_doc, max_total_chars // max(len(index), 1))

        lines = ["\n\n## 참고 자료 (누적 문서 베이스)"]
        lines.append(f"(총 {len(index)}건의 문서가 등록되어 있습니다)\n")

        total_used = 0
        for i, entry in enumerate(index, 1):
            if total_used >= max_total_chars:
                lines.append(f"### 문서 {i}~{len(index)}: (토큰 예산 초과로 생략)")
                break
            text = self.load_text(entry["doc_id"])
            if not text:
                continue
            remaining = max_total_chars - total_used
            budget = min(per_doc_budget, remaining)
            lines.append(f"### 문서 {i}: {entry['filename']}")
            if len(text) > budget:
                text = text[:budget] + f"\n\n... (총 {len(text):,}자 중 {budget:,}자까지 포함)"
            lines.append(text)
            lines.append("")
            total_used += len(text)

        return "\n".join(lines) if len(lines) > 2 else ""


# ── 토론 중간 저장 (Draft) ──

_DEFAULT_DRAFT_DIR = DATA_DIR / "drafts"


class DraftStore:
    """토론 중간 저장소. 라운드 완료 시마다 자동 체크포인트."""

    def __init__(self, directory: str | Path | None = None):
        self.directory = Path(directory) if directory else _DEFAULT_DRAFT_DIR
        _ensure_dir(self.directory)

    def _path(self, session_id: str) -> Path:
        return self.directory / f"{session_id}.json"

    def save(self, session_id: str, snapshot: dict[str, Any]) -> Path:
        """세션 스냅샷 저장. 매 라운드 완료 시 호출."""
        snapshot.setdefault("session_id", session_id)
        snapshot["updated_at"] = datetime.now(timezone.utc).isoformat()
        snapshot.setdefault("created_at", snapshot["updated_at"])
        path = self._path(session_id)
        path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"Draft saved: {path} (phase={snapshot.get('phase', '?')})")
        return path

    def load(self, session_id: str) -> dict[str, Any] | None:
        """드래프트 로드."""
        path = self._path(session_id)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"Failed to load draft {path}: {e}")
            return None

    def delete(self, session_id: str) -> bool:
        """드래프트 삭제 (synthesize 완료 후 정리 용도)."""
        path = self._path(session_id)
        if path.exists():
            path.unlink()
            return True
        return False

    def list_all(self, limit: int = 50) -> list[dict[str, Any]]:
        """미완료 드래프트 목록 (최신순)."""
        records = []
        files = sorted(
            self.directory.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for f in files[:limit]:
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                records.append({
                    "session_id": data.get("session_id", f.stem),
                    "mode": data.get("mode", ""),
                    "topic": data.get("topic", ""),
                    "phase": data.get("phase", ""),
                    "round_count": data.get("round_count", 0),
                    "created_at": data.get("created_at", ""),
                    "updated_at": data.get("updated_at", ""),
                })
            except Exception:
                continue
        return records


# ── 싱글턴 인스턴스 ──

_store: DebateHistoryStore | None = None
_doc_store: DocumentStore | None = None
_draft_store: DraftStore | None = None


def get_history_store() -> DebateHistoryStore:
    """글로벌 히스토리 스토어 인스턴스."""
    global _store
    if _store is None:
        _store = DebateHistoryStore()
    return _store


def get_document_store() -> DocumentStore:
    """글로벌 문서 스토어 인스턴스."""
    global _doc_store
    if _doc_store is None:
        _doc_store = DocumentStore()
    return _doc_store


def get_draft_store() -> DraftStore:
    """글로벌 드래프트 스토어 인스턴스."""
    global _draft_store
    if _draft_store is None:
        _draft_store = DraftStore()
    return _draft_store
