"""
Portfolio Analyzer — 포트폴리오 분석기.

사용자의 포트폴리오를 입력받아 Multi-Agent Debate + Question Frame 기반
분석을 수행하고 투자 판정과 구체적인 개선 방안을 제시하는 최상위 인터페이스.
"""

from __future__ import annotations

from src.api.client import ClaudeClient
from src.config import UsageTracker
from src.engine.reasoning import ReasoningEngine
from src.engine.question_frame import QuestionFrame
from src.models.analysis import PortfolioAnalysis, HoldingAnalysis
from src.models.portfolio import Portfolio, Holding


class PortfolioAnalyzer:
    """포트폴리오 분석기 — 메인 인터페이스."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-20250514",
        custom_frames: list[QuestionFrame] | None = None,
        agents: list | None = None,
        usage_tracker: UsageTracker | None = None,
    ):
        self.client = ClaudeClient(api_key=api_key, model=model, usage_tracker=usage_tracker)
        self.engine = ReasoningEngine(
            client=self.client,
            frames=custom_frames,
            agents=agents,
        )

    async def analyze(self, portfolio: Portfolio) -> PortfolioAnalysis:
        """포트폴리오 전체 분석 실행 (토론 + 프레임 — 풀 모드)."""
        return await self.engine.analyze_portfolio(portfolio)

    async def debate(self, portfolio: Portfolio) -> PortfolioAnalysis:
        """포트폴리오 전체 분석 실행 (토론만 — 빠른 모드)."""
        return await self.engine.debate_portfolio(portfolio)

    async def analyze_single(
        self,
        holding: Holding,
        portfolio: Portfolio | None = None,
    ) -> HoldingAnalysis:
        """개별 종목 분석 (토론 기반)."""
        if portfolio is None:
            portfolio = Portfolio(
                name="Single Analysis",
                holdings=[holding],
            )
        return await self.engine.debate_only(holding, portfolio)
