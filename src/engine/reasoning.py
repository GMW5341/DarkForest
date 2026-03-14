"""
Reasoning Engine — 추론 엔진.

질문 프레임 기반 분석 + Multi-Agent Debate를 결합한 핵심 엔진.

원칙:
- 분석은 100% 논리적으로 정합해야 한다
- 결론은 분석의 자연스러운 귀결이어야 한다
- 에이전트 간 토론을 통해 편향을 줄이고 결론 품질을 높인다
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from src.agents.base import BaseAnalystAgent
from src.agents.debate import DebateOrchestrator
from src.agents.fundamental import FundamentalAgent
from src.agents.risk import RiskAgent
from src.agents.valuation import ValuationAgent
from src.engine.question_frame import DEFAULT_FRAMES, QuestionFrame
from src.models.analysis import (
    DebateResult,
    FrameAnalysis,
    HoldingAnalysis,
    InvestmentDecision,
    PortfolioAnalysis,
    Verdict,
)
from src.models.portfolio import Holding, Portfolio

if TYPE_CHECKING:
    from src.api.client import ClaudeClient


# Claude에게 보내는 시스템 프롬프트 (프레임 분석용)
SYSTEM_PROMPT = """\
당신은 투자 분석 전문가입니다. 다음 원칙을 반드시 따르세요:

## 핵심 원칙
1. **분석의 논리 정합성**: 결론이 맞고 틀리고는 결과의 문제이지만, \
분석은 판단 당시의 논리 정합성의 문제입니다. 분석은 100% 논리적이어야 합니다.
2. **확고한 관점(Frame)**: 특정 시점에 특정 논리와 사고를 근거로 판단을 \
내렸다는 확고한 관점을 제공하세요.
3. **사고 훈련**: 단순한 지식의 전달이 아닌, 사고 훈련의 장으로서 기능하세요.
4. **결론보다 과정**: '매수/매도' 결론 자체보다, 그 결론에 도달하기까지의 \
논리적 과정이 더 중요합니다.

## 응답 형식
반드시 다음 JSON 형식으로 응답하세요:
{
    "reasoning": "추론 과정 (step by step)",
    "conclusion": "이 프레임에서의 결론",
    "confidence": 0.0~1.0 (논리 정합성에 대한 자신감),
    "evidence": ["근거1", "근거2", ...],
    "red_flags": ["경고 신호1", ...] (없으면 빈 배열)
}
"""

PORTFOLIO_SYSTEM_PROMPT = """\
당신은 투자 포트폴리오 분석 전문가입니다. 개별 종목의 Multi-Agent Debate 결과를 \
종합하여 포트폴리오 전체에 대한 구조적 평가를 수행합니다.

## 핵심 원칙
1. 포트폴리오의 구조적 문제를 식별하세요 (집중도, 상관관계, 자산 배분).
2. 개별 종목의 토론 결과를 포트폴리오 맥락에서 재평가하세요.
3. 구체적이고 실행 가능한 개선 방안을 제시하세요.

## 응답 형식
반드시 다음 JSON 형식으로 응답하세요:
{
    "summary": "포트폴리오 종합 요약",
    "structural_issues": ["구조적 문제1", ...],
    "improvement_plan": ["구체적 개선 방안1", ...],
    "risk_assessment": "전체 리스크 평가",
    "holding_verdicts": {
        "TICKER": {
            "verdict": "hold/rebalance/reduce/exit/strong_hold/needs_more_data",
            "reasoning": "판단 근거",
            "action_items": ["구체적 행동1", ...]
        }
    }
}
"""


def _format_holding_info(holding: Holding, portfolio: Portfolio) -> str:
    """종목 정보를 분석용 텍스트로 포맷."""
    return (
        f"종목명: {holding.name}\n"
        f"티커: {holding.ticker}\n"
        f"자산군: {holding.asset_class.value}\n"
        f"섹터: {holding.sector or '미분류'}\n"
        f"평균 매수가: {holding.avg_price:,.0f} {holding.currency}\n"
        f"현재가: {holding.current_price:,.0f} {holding.currency}\n"
        f"수익률: {holding.return_pct:+.2f}%\n"
        f"포트폴리오 비중: {portfolio.weight_of(holding):.1f}%\n"
        f"매수 근거: {holding.memo or '없음'}"
    )


def _format_portfolio_summary(portfolio: Portfolio) -> str:
    """포트폴리오 요약 정보."""
    lines = [
        f"포트폴리오: {portfolio.name}",
        f"투자 목표: {portfolio.investment_goal or '미설정'}",
        f"위험 성향: {portfolio.risk_tolerance or '미설정'}",
        f"투자 기간: {portfolio.investment_horizon or '미설정'}",
        f"총 평가액: {portfolio.total_value:,.0f}",
        f"총 수익률: {portfolio.total_return_pct:+.2f}%",
        "",
        "자산군별 비중:",
    ]
    for asset_class, weight in portfolio.weights_by_asset_class().items():
        lines.append(f"  {asset_class.value}: {weight:.1f}%")

    lines.append("")
    lines.append("보유 종목:")
    for h in portfolio.holdings:
        lines.append(
            f"  {h.name} ({h.ticker}) - "
            f"비중 {portfolio.weight_of(h):.1f}%, "
            f"수익률 {h.return_pct:+.2f}%"
        )
    return "\n".join(lines)


class ReasoningEngine:
    """
    추론 엔진 — Multi-Agent Debate + 질문 프레임 기반 포트폴리오 분석.

    각 보유 종목에 대해 3명의 AI 애널리스트가 토론을 벌이고,
    질문 프레임 분석과 결합하여 최종 판정을 도출한다.
    """

    def __init__(
        self,
        client: ClaudeClient,
        frames: list[QuestionFrame] | None = None,
        agents: list[BaseAnalystAgent] | None = None,
    ):
        self.client = client
        self.frames = frames or DEFAULT_FRAMES

        # AI 애널리스트 에이전트 초기화 (외부 주입 or 기본 3인)
        self.agents: list[BaseAnalystAgent] = agents or [
            FundamentalAgent(client),
            ValuationAgent(client),
            RiskAgent(client),
        ]
        self.debate_orchestrator = DebateOrchestrator(
            agents=self.agents,
            synthesizer_client=client,
        )

    async def analyze_holding_with_frame(
        self,
        holding: Holding,
        portfolio: Portfolio,
        frame: QuestionFrame,
    ) -> FrameAnalysis:
        """단일 종목을 단일 프레임으로 분석."""
        holding_info = _format_holding_info(holding, portfolio)
        portfolio_summary = _format_portfolio_summary(portfolio)

        prompt = frame.prompt_template.format(
            holding_info=holding_info,
            portfolio_summary=portfolio_summary,
        )

        response = await self.client.ask(
            system=SYSTEM_PROMPT,
            user_message=prompt,
        )

        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            data = {
                "reasoning": response,
                "conclusion": "파싱 오류로 구조화 실패",
                "confidence": 0.3,
                "evidence": [],
            }

        return FrameAnalysis(
            frame_name=frame.name,
            question=frame.core_question,
            reasoning=data.get("reasoning", ""),
            conclusion=data.get("conclusion", ""),
            confidence=min(max(data.get("confidence", 0.5), 0.0), 1.0),
            evidence=data.get("evidence", []),
        )

    async def debate_holding(
        self,
        holding: Holding,
        portfolio: Portfolio,
    ) -> DebateResult:
        """단일 종목에 대해 Multi-Agent Debate 실행."""
        return await self.debate_orchestrator.run_debate(holding, portfolio)

    async def analyze_holding(
        self,
        holding: Holding,
        portfolio: Portfolio,
    ) -> HoldingAnalysis:
        """단일 종목을 프레임 + 토론으로 분석."""
        # Step 1: Multi-Agent Debate
        debate_result = await self.debate_holding(holding, portfolio)

        # Step 2: Question Frame 분석 (기존 시스템과 병행)
        frame_analyses: list[FrameAnalysis] = []
        for frame in self.frames:
            analysis = await self.analyze_holding_with_frame(
                holding, portfolio, frame,
            )
            frame_analyses.append(analysis)

        # Debate 결과를 Verdict로 변환
        verdict = self._decision_to_verdict(debate_result.final_decision)

        return HoldingAnalysis(
            ticker=holding.ticker,
            name=holding.name,
            frame_analyses=frame_analyses,
            debate_result=debate_result,
            overall_verdict=verdict,
            verdict_reasoning=debate_result.final_reasoning,
            action_items=debate_result.action_items,
        )

    async def debate_only(
        self,
        holding: Holding,
        portfolio: Portfolio,
    ) -> HoldingAnalysis:
        """토론만 실행 (프레임 분석 생략 — 빠른 분석)."""
        debate_result = await self.debate_holding(holding, portfolio)
        verdict = self._decision_to_verdict(debate_result.final_decision)

        return HoldingAnalysis(
            ticker=holding.ticker,
            name=holding.name,
            debate_result=debate_result,
            overall_verdict=verdict,
            verdict_reasoning=debate_result.final_reasoning,
            action_items=debate_result.action_items,
        )

    async def analyze_portfolio(self, portfolio: Portfolio) -> PortfolioAnalysis:
        """포트폴리오 전체 분석 (토론 + 프레임)."""
        holding_analyses: list[HoldingAnalysis] = []
        for holding in portfolio.holdings:
            analysis = await self.analyze_holding(holding, portfolio)
            holding_analyses.append(analysis)

        return await self._synthesize_portfolio(portfolio, holding_analyses)

    async def debate_portfolio(self, portfolio: Portfolio) -> PortfolioAnalysis:
        """포트폴리오 전체 분석 (토론만 — 빠른 모드)."""
        holding_analyses: list[HoldingAnalysis] = []
        for holding in portfolio.holdings:
            analysis = await self.debate_only(holding, portfolio)
            holding_analyses.append(analysis)

        return await self._synthesize_portfolio(portfolio, holding_analyses)

    async def _synthesize_portfolio(
        self,
        portfolio: Portfolio,
        holding_analyses: list[HoldingAnalysis],
    ) -> PortfolioAnalysis:
        """개별 분석을 종합하여 포트폴리오 레벨 판단."""
        individual_summaries = []
        for ha in holding_analyses:
            if ha.debate_result:
                dr = ha.debate_result
                summary = (
                    f"### {ha.name} ({ha.ticker})\n"
                    f"- 토론 결과: {dr.final_decision.value.upper()} "
                    f"(확신도: {dr.final_confidence:.0%})\n"
                    f"- 판단 근거: {dr.final_reasoning}\n"
                    f"- 합의 포인트: {', '.join(dr.consensus_points) if dr.consensus_points else '없음'}\n"
                    f"- 의견 불일치: {', '.join(dr.dissent_points) if dr.dissent_points else '없음'}"
                )
            else:
                frame_conclusions = "\n".join(
                    f"  - {fa.frame_name}: {fa.conclusion} "
                    f"(confidence: {fa.confidence:.2f})"
                    for fa in ha.frame_analyses
                )
                summary = f"### {ha.name} ({ha.ticker})\n{frame_conclusions}"
            individual_summaries.append(summary)

        synthesis_prompt = (
            f"다음은 포트폴리오의 각 종목에 대한 분석 결과입니다.\n\n"
            f"포트폴리오 정보:\n{_format_portfolio_summary(portfolio)}\n\n"
            f"개별 종목 분석:\n"
            + "\n\n".join(individual_summaries)
            + "\n\n"
            "위 분석을 종합하여 포트폴리오 전체에 대한 평가를 해주세요."
        )

        response = await self.client.ask(
            system=PORTFOLIO_SYSTEM_PROMPT,
            user_message=synthesis_prompt,
        )

        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            data = {
                "summary": response,
                "structural_issues": [],
                "improvement_plan": [],
                "risk_assessment": "파싱 오류로 구조화 실패",
                "holding_verdicts": {},
            }

        # 종목별 verdict 반영
        holding_verdicts = data.get("holding_verdicts", {})
        for ha in holding_analyses:
            if ha.ticker in holding_verdicts:
                v = holding_verdicts[ha.ticker]
                try:
                    ha.overall_verdict = Verdict(
                        v.get("verdict", "needs_more_data")
                    )
                except ValueError:
                    pass  # 토론 결과의 verdict 유지
                if v.get("reasoning"):
                    ha.verdict_reasoning = v["reasoning"]
                if v.get("action_items"):
                    ha.action_items = v["action_items"]

        # 정합성 점수 계산
        all_confidences = [
            fa.confidence
            for ha in holding_analyses
            for fa in ha.frame_analyses
        ]
        # 토론 결과의 확신도도 포함
        for ha in holding_analyses:
            if ha.debate_result:
                all_confidences.append(ha.debate_result.final_confidence)

        avg_confidence = (
            sum(all_confidences) / len(all_confidences)
            if all_confidences
            else 0.0
        )

        return PortfolioAnalysis(
            summary=data.get("summary", ""),
            holding_analyses=holding_analyses,
            structural_issues=data.get("structural_issues", []),
            improvement_plan=data.get("improvement_plan", []),
            risk_assessment=data.get("risk_assessment", ""),
            frame_integrity_score=avg_confidence,
        )

    @staticmethod
    def _decision_to_verdict(decision: InvestmentDecision) -> Verdict:
        """InvestmentDecision을 Verdict로 변환."""
        mapping = {
            InvestmentDecision.BUY: Verdict.STRONG_HOLD,
            InvestmentDecision.WAIT: Verdict.HOLD,
            InvestmentDecision.PASS: Verdict.EXIT,
            InvestmentDecision.NEEDS_MORE_DATA: Verdict.NEEDS_MORE_DATA,
        }
        return mapping.get(decision, Verdict.NEEDS_MORE_DATA)
