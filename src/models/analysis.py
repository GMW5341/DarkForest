"""Analysis result models — Multi-Agent Debate 기반 투자 판정."""

from __future__ import annotations

from enum import Enum
from pydantic import BaseModel, Field


class InvestmentDecision(str, Enum):
    """최종 투자 판정."""
    BUY = "buy"                     # 매수
    WAIT = "wait"                   # 대기 (좋지만 타이밍 아님)
    PASS = "pass"                   # 패스 (투자 부적합)
    NEEDS_MORE_DATA = "needs_more_data"  # 판단 보류


class AgentOpinion(BaseModel):
    """개별 에이전트의 의견서."""
    agent_name: str = Field(description="에이전트 이름")
    agent_role: str = Field(description="에이전트 역할 (예: 펀더멘털 애널리스트)")
    stance: InvestmentDecision = Field(description="이 에이전트의 입장")
    confidence: float = Field(ge=0, le=1, description="확신도 (0~1)")
    reasoning: str = Field(description="판단 근거")
    key_points: list[str] = Field(default_factory=list, description="핵심 포인트")
    red_flags: list[str] = Field(default_factory=list, description="경고 신호")
    evidence: list[str] = Field(default_factory=list, description="근거 데이터")


class DebateMessage(BaseModel):
    """토론 중 하나의 발언."""
    agent_name: str
    round_number: int
    message_type: str = Field(description="opening | rebuttal | final")
    stance: InvestmentDecision
    confidence: float = Field(ge=0, le=1)
    content: str = Field(description="발언 내용")
    agreements: list[str] = Field(default_factory=list, description="동의하는 포인트")
    disagreements: list[str] = Field(default_factory=list, description="반론 포인트")


class DebateRound(BaseModel):
    """토론 라운드."""
    round_number: int
    round_type: str = Field(description="opening | cross_examination | final")
    messages: list[DebateMessage] = Field(default_factory=list)


class DebateResult(BaseModel):
    """토론 전체 결과."""
    target_company: str = Field(description="분석 대상 기업명")
    target_ticker: str = Field(default="", description="티커")
    rounds: list[DebateRound] = Field(default_factory=list)
    consensus_points: list[str] = Field(
        default_factory=list,
        description="에이전트들이 합의한 포인트",
    )
    dissent_points: list[str] = Field(
        default_factory=list,
        description="의견이 갈린 포인트",
    )
    final_decision: InvestmentDecision = InvestmentDecision.NEEDS_MORE_DATA
    final_confidence: float = Field(
        default=0.0, ge=0, le=1,
        description="최종 확신도",
    )
    final_reasoning: str = Field(default="", description="최종 판단 근거")
    action_items: list[str] = Field(
        default_factory=list,
        description="구체적 행동 제안",
    )
    risk_summary: str = Field(default="", description="핵심 리스크 요약")
    price_assessment: str = Field(default="", description="적정가 평가")


# ── 기존 포트폴리오 분석과의 호환을 위한 모델 ──

class Verdict(str, Enum):
    """포트폴리오 레벨 판단 (기존 호환)."""
    STRONG_HOLD = "strong_hold"
    HOLD = "hold"
    REBALANCE = "rebalance"
    REDUCE = "reduce"
    EXIT = "exit"
    NEEDS_MORE_DATA = "needs_more_data"


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
    debate_result: DebateResult | None = Field(
        default=None, description="Multi-Agent Debate 결과",
    )
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
        description="포트폴리오 구조적 문제점",
    )
    improvement_plan: list[str] = Field(
        default_factory=list,
        description="구체적 개선 방안",
    )
    risk_assessment: str = Field(default="", description="리스크 평가")
    frame_integrity_score: float = Field(
        default=0.0, ge=0, le=1,
        description="전체 분석의 논리 정합성 점수",
    )
