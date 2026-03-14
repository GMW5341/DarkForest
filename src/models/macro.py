"""거시 경제 토론 모델 — 매크로 이슈/테마 기반 분석 결과."""

from __future__ import annotations

from pydantic import BaseModel, Field


class MacroTopic(BaseModel):
    """거시 경제 토론 주제."""
    title: str = Field(description="토론 주제 (예: '사모대출 사기 사태와 대규모 환매')")
    context: str = Field(
        default="",
        description="배경 정보, 참고 자료, 뉴스 등 (자유 형식 텍스트)",
    )


class InvestmentImplication(BaseModel):
    """투자 함의."""
    category: str = Field(description="자산 배분 | 섹터 | 방어 전략 | 기회 포착")
    recommendation: str = Field(description="구체적 권고")
    rationale: str = Field(default="", description="근거")


class RiskScenario(BaseModel):
    """리스크 시나리오."""
    scenario: str = Field(description="시나리오 설명")
    probability: str = Field(default="중간", description="높음/중간/낮음")
    impact: str = Field(default="중간", description="높음/중간/낮음")
    hedge: str = Field(default="", description="대응 방안")


class MacroMessageModel(BaseModel):
    """거시 토론 발언 (API 직렬화용)."""
    agent_name: str
    round_number: int
    message_type: str
    stance: str
    confidence: float = Field(ge=0, le=1)
    content: str
    agreements: list[str] = Field(default_factory=list)
    disagreements: list[str] = Field(default_factory=list)


class MacroDebateRoundModel(BaseModel):
    """거시 토론 라운드 (API 직렬화용)."""
    round_number: int
    round_type: str = Field(description="opening | cross_examination | final")
    messages: list[MacroMessageModel] = Field(default_factory=list)


class MacroDebateResult(BaseModel):
    """거시 경제 토론 최종 결과."""
    topic: str = Field(description="토론 주제")
    overall_stance: str = Field(default="중립", description="방어적 | 공격적 | 중립 | 관망")
    confidence: float = Field(default=0.5, ge=0, le=1)
    executive_summary: str = Field(default="", description="핵심 요약")
    consensus_points: list[str] = Field(default_factory=list)
    dissent_points: list[str] = Field(default_factory=list)
    investment_implications: list[InvestmentImplication] = Field(default_factory=list)
    risk_scenarios: list[RiskScenario] = Field(default_factory=list)
    action_items: list[str] = Field(default_factory=list)
    monitoring_points: list[str] = Field(default_factory=list)
    rounds: list[MacroDebateRoundModel] = Field(default_factory=list)
