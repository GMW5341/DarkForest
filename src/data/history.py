"""
Persistent debate history storage using JSON files.

각 토론 결과를 JSON 파일로 영구 저장하고 조회/검색 기능을 제공한다.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# 기본 저장 경로
_DEFAULT_DIR = Path("data/history")


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


# 싱글턴 인스턴스
_store: DebateHistoryStore | None = None


def get_history_store() -> DebateHistoryStore:
    """글로벌 히스토리 스토어 인스턴스."""
    global _store
    if _store is None:
        _store = DebateHistoryStore()
    return _store
