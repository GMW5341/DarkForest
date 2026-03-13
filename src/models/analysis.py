"""Analysis result models."""

from __future__ import annotations

from enum import Enum
from pydantic import BaseModel, Field


class Verdict(str, Enum):
    """분석 결론 — 매매 신호가 아닌 논리적 판단."""
    STRONG_HOLD = "strong_hold"       # 유지 강화
    HOLD = "hold"                     # 유지
    REBALANCE = "rebalance"           # 리밸런싱 필요
    REDUCE = "reduce"                 # 비중 축소 검토
    EXIT = "exit"                     # 이탈 검토
    NEEDS_MORE_DATA = "needs_more_data"  # 판단 보류 (정보 부족)


class FrameAnalysis(BaseModel):
    """하나의 질문 프레임에 대한 분석 결과."""
    frame_name: str = Field(description="질문 프레임 이름")
    question: str = Field(description="핵심 질문")
    reasoning: str = Field(description="추론 과정")
    conclusion: str = Field(description="해당 프레임의 결론")
    confidence: float = Field(ge=0, le=1, description="논리 정합성 자신감 (0~1)")
    evidence: list[str] = Field(default_factory=list, description="근거 목록")


class HoldingAnalysis(BaseModel):
    """개별 종목 분석 결과."""
    ticker: str
    name: str
    frame_analyses: list[FrameAnalysis] = Field(default_factory=list)
    overall_verdict: Verdict = Verdict.NEEDS_MORE_DATA
    verdict_reasoning: str = Field(default="", description="종합 판단 근거")
    action_items: list[str] = Field(
        default_factory=list,
        description="구체적 행동 제안",
    )


class PortfolioAnalysis(BaseModel):
    """포트폴리오 전체 분석 결과."""
    summary: str = Field(default="", description="포트폴리오 종합 요약")
    holding_analyses: list[HoldingAnalysis] = Field(default_factory=list)
    structural_issues: list[str] = Field(
        default_factory=list,
        description="포트폴리오 구조적 문제점 (집중도, 상관관계 등)",
    )
    improvement_plan: list[str] = Field(
        default_factory=list,
        description="구체적 개선 방안",
    )
    risk_assessment: str = Field(default="", description="리스크 평가")
    frame_integrity_score: float = Field(
        default=0.0,
        ge=0,
        le=1,
        description="전체 분석의 논리 정합성 점수",
    )
