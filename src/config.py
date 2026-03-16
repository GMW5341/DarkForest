"""
DarkForest 설정 관리 모듈.

모델 선택, API 비용 추적, 토론 파라미터 등을 중앙 관리한다.
환경변수 또는 settings.json 파일로 오버라이드 가능.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# ── 영구 데이터 디렉토리 (Render Persistent Disk 등) ──
DATA_DIR = Path(os.environ.get("DARKFOREST_DATA_DIR", "data"))

_SETTINGS_PATH = Path(os.environ.get(
    "DARKFOREST_SETTINGS", str(DATA_DIR / "settings.json"),
))

# ── Anthropic 모델별 가격 (USD / 1M tokens, 2025-05 기준) ──
MODEL_PRICING: dict[str, dict[str, float]] = {
    "claude-opus-4-20250514": {"input": 15.0, "output": 75.0},
    "claude-sonnet-4-20250514": {"input": 3.0, "output": 15.0},
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.0},
}


@dataclass
class DarkForestSettings:
    """전역 설정."""

    # 모델
    default_model: str = "claude-opus-4-20250514"
    max_tokens: int = 4096

    # 컨텍스트 예산
    max_total_context_chars: int = 40000
    max_chars_per_doc: int = 8000

    # 비용 추적
    track_costs: bool = True

    # 외부 API 키 (환경변수 → settings.json → 빈 문자열 순으로 fallback)
    dart_api_key: str = ""  # OpenDART API 키 (https://opendart.fss.or.kr)

    def to_dict(self) -> dict:
        return {
            "default_model": self.default_model,
            "max_tokens": self.max_tokens,
            "max_total_context_chars": self.max_total_context_chars,
            "max_chars_per_doc": self.max_chars_per_doc,
            "track_costs": self.track_costs,
            "dart_api_key": self.dart_api_key,
        }

    def save(self) -> None:
        _SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        _SETTINGS_PATH.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls) -> DarkForestSettings:
        data: dict = {}
        if _SETTINGS_PATH.exists():
            try:
                data = json.loads(_SETTINGS_PATH.read_text(encoding="utf-8"))
            except Exception:
                logger.warning("설정 파일 로드 실패, 기본값 사용")

        # 환경변수가 있으면 settings.json보다 우선
        env_overrides = {
            "dart_api_key": os.environ.get("DART_API_KEY", ""),
            "default_model": os.environ.get("DARKFOREST_MODEL", ""),
            "max_tokens": os.environ.get("DARKFOREST_MAX_TOKENS", ""),
        }
        for key, env_val in env_overrides.items():
            if env_val:
                data[key] = int(env_val) if key == "max_tokens" else env_val

        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ── API 사용량 추적 ──

@dataclass
class UsageRecord:
    """단일 API 호출 사용량."""
    model: str
    input_tokens: int
    output_tokens: int

    @property
    def cost_usd(self) -> float:
        pricing = MODEL_PRICING.get(self.model, {"input": 3.0, "output": 15.0})
        return (
            self.input_tokens * pricing["input"] / 1_000_000
            + self.output_tokens * pricing["output"] / 1_000_000
        )


@dataclass
class UsageTracker:
    """세션 내 누적 API 사용량 추적."""
    records: list[UsageRecord] = field(default_factory=list)

    def add(self, model: str, input_tokens: int, output_tokens: int) -> UsageRecord:
        record = UsageRecord(model=model, input_tokens=input_tokens, output_tokens=output_tokens)
        self.records.append(record)
        return record

    @property
    def total_input_tokens(self) -> int:
        return sum(r.input_tokens for r in self.records)

    @property
    def total_output_tokens(self) -> int:
        return sum(r.output_tokens for r in self.records)

    @property
    def total_cost_usd(self) -> float:
        return sum(r.cost_usd for r in self.records)

    @property
    def call_count(self) -> int:
        return len(self.records)

    def summary(self) -> dict:
        return {
            "call_count": self.call_count,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "total_cost_usd": round(self.total_cost_usd, 4),
            "total_cost_krw": round(self.total_cost_usd * 1400, 0),
            "model": self.records[-1].model if self.records else "N/A",
        }

    def reset(self) -> None:
        self.records.clear()


def get_settings() -> DarkForestSettings:
    """전역 설정 싱글턴."""
    global _settings
    if _settings is None:
        _settings = DarkForestSettings.load()
    return _settings


def update_settings(**kwargs) -> DarkForestSettings:
    """설정 업데이트 후 저장."""
    settings = get_settings()
    for k, v in kwargs.items():
        if hasattr(settings, k):
            setattr(settings, k, v)
    settings.save()
    return settings


_settings: DarkForestSettings | None = None


# ── 누적 API 비용 추적 (영구 저장) ──

_USAGE_PATH = DATA_DIR / "usage.json"


@dataclass
class CumulativeUsageTracker:
    """재배포해도 유지되는 누적 API 비용 추적기.

    매 API 호출마다 usage.json에 즉시 기록한다.
    """

    total_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0

    def add(self, model: str, input_tokens: int, output_tokens: int) -> None:
        """API 호출 1건 누적 기록."""
        pricing = MODEL_PRICING.get(model, {"input": 3.0, "output": 15.0})
        cost = (
            input_tokens * pricing["input"] / 1_000_000
            + output_tokens * pricing["output"] / 1_000_000
        )
        self.total_calls += 1
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost_usd += cost
        self._save()

    def summary(self) -> dict:
        return {
            "total_calls": self.total_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "total_cost_krw": round(self.total_cost_usd * 1400, 0),
        }

    def _save(self) -> None:
        try:
            _USAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
            _USAGE_PATH.write_text(
                json.dumps(self.summary(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning(f"누적 사용량 저장 실패: {e}")

    @classmethod
    def load(cls) -> CumulativeUsageTracker:
        if _USAGE_PATH.exists():
            try:
                data = json.loads(_USAGE_PATH.read_text(encoding="utf-8"))
                return cls(
                    total_calls=data.get("total_calls", 0),
                    total_input_tokens=data.get("total_input_tokens", 0),
                    total_output_tokens=data.get("total_output_tokens", 0),
                    total_cost_usd=data.get("total_cost_usd", 0.0),
                )
            except Exception:
                logger.warning("누적 사용량 파일 로드 실패, 0부터 시작")
        return cls()


_cumulative_usage: CumulativeUsageTracker | None = None


def get_cumulative_usage() -> CumulativeUsageTracker:
    """글로벌 누적 사용량 트래커."""
    global _cumulative_usage
    if _cumulative_usage is None:
        _cumulative_usage = CumulativeUsageTracker.load()
    return _cumulative_usage
