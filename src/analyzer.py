"""
Portfolio Analyzer — 포트폴리오 분석기.

사용자의 포트폴리오를 입력받아 Question Frame 기반 분석을 수행하고
구체적인 개선 방안을 제시하는 최상위 인터페이스.
"""

from __future__ import annotations

from src.api.client import ClaudeClient
from src.engine.reasoning import ReasoningEngine
from src.engine.question_frame import QuestionFrame
from src.models.analysis import PortfolioAnalysis
from src.models.portfolio import Portfolio


class PortfolioAnalyzer:
    """포트폴리오 분석기 — 메인 인터페이스."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-20250514",
        custom_frames: list[QuestionFrame] | None = None,
    ):
        self.client = ClaudeClient(api_key=api_key, model=model)
        self.engine = ReasoningEngine(
            client=self.client,
            frames=custom_frames,
        )

    async def analyze(self, portfolio: Portfolio) -> PortfolioAnalysis:
        """포트폴리오 전체 분석 실행."""
        return await self.engine.analyze_portfolio(portfolio)
