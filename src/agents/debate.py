"""
Debate Orchestrator — Multi-Agent Debate 오케스트레이터.

여러 AI 애널리스트 에이전트가 토론을 통해 최종 투자 판정에 도달하는 시스템.

토론 플로우:
  Round 1 (Opening): 각 에이전트가 독립적으로 분석 → 의견서 제출
  Round 2 (Cross-Examination): 서로의 의견서를 읽고 반론/동의
  Round 3 (Final Position): 토론을 종합하여 최종 입장 정리
  Synthesis: 전체 토론을 보고 최종 BUY/WAIT/PASS 판정
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from src.agents.base import BaseAnalystAgent
from src.models.analysis import (
    AgentOpinion,
    DebateMessage,
    DebateResult,
    DebateRound,
    InvestmentDecision,
)

if TYPE_CHECKING:
    from src.api.client import ClaudeClient
    from src.models.portfolio import Holding, Portfolio


SYNTHESIS_SYSTEM_PROMPT = """\
당신은 투자위원회(Investment Committee)의 의장입니다.

여러 애널리스트들의 토론을 종합하여 최종 투자 판정을 내립니다.

## 당신의 역할
- 각 애널리스트의 논리를 객관적으로 평가합니다
- 합의된 포인트와 갈린 포인트를 명확히 구분합니다
- 다수결이 아닌 논리의 질(quality)을 기준으로 판단합니다
- 최종 판정에는 반드시 구체적 행동 제안을 포함합니다

## 판정 기준
- **BUY**: 펀더멘털이 강하고, 밸류에이션이 합리적이며, 리스크가 관리 가능할 때
- **WAIT**: 기업은 좋지만 타이밍이 아직 아닐 때 (비쌈, 촉매 부족 등)
- **PASS**: 펀더멘털 문제, 과도한 리스크, 또는 더 나은 대안이 있을 때

## 응답 형식 (반드시 JSON)
{
    "final_decision": "buy" | "wait" | "pass" | "needs_more_data",
    "final_confidence": 0.0~1.0,
    "final_reasoning": "최종 판단 근거 (토론 전체를 종합한 논리)",
    "consensus_points": ["합의 포인트1", ...],
    "dissent_points": ["의견 불일치 포인트1", ...],
    "action_items": ["구체적 행동 제안1", ...],
    "risk_summary": "핵심 리스크 요약",
    "price_assessment": "적정가 평가 및 현재가 판단"
}"""


class DebateOrchestrator:
    """
    Multi-Agent Debate 오케스트레이터.

    여러 애널리스트 에이전트 간의 구조화된 토론을 진행하고
    최종 투자 판정을 도출한다.
    """

    def __init__(
        self,
        agents: list[BaseAnalystAgent],
        synthesizer_client: ClaudeClient,
    ):
        self.agents = agents
        self.synthesizer_client = synthesizer_client

    async def run_debate(
        self,
        holding: Holding,
        portfolio: Portfolio,
    ) -> DebateResult:
        """전체 토론 프로세스 실행."""
        rounds: list[DebateRound] = []

        # ── Round 1: Opening — 각 에이전트 독립 분석 ──
        opinions: list[AgentOpinion] = []
        round1_messages: list[DebateMessage] = []

        for agent in self.agents:
            opinion = await agent.analyze(holding, portfolio)
            opinions.append(opinion)
            round1_messages.append(DebateMessage(
                agent_name=opinion.agent_name,
                round_number=1,
                message_type="opening",
                stance=opinion.stance,
                confidence=opinion.confidence,
                content=opinion.reasoning,
                agreements=opinion.key_points,
                disagreements=opinion.red_flags,
            ))

        rounds.append(DebateRound(
            round_number=1,
            round_type="opening",
            messages=round1_messages,
        ))

        # ── Round 2: Cross-Examination — 상호 반론 ──
        round2_messages: list[DebateMessage] = []

        for i, agent in enumerate(self.agents):
            # 자기 자신을 제외한 다른 에이전트들의 의견
            other_opinions = [op for j, op in enumerate(opinions) if j != i]
            rebuttal = await agent.rebut(
                holding, portfolio, other_opinions, round_number=2,
            )
            round2_messages.append(rebuttal)

        rounds.append(DebateRound(
            round_number=2,
            round_type="cross_examination",
            messages=round2_messages,
        ))

        # ── Round 3: Final Position — 최종 입장 정리 ──
        debate_history = self._format_debate_history(rounds)
        round3_messages: list[DebateMessage] = []

        for agent in self.agents:
            final = await agent.final_position(
                holding, portfolio, debate_history,
            )
            round3_messages.append(final)

        rounds.append(DebateRound(
            round_number=3,
            round_type="final",
            messages=round3_messages,
        ))

        # ── Synthesis — 최종 판정 ──
        full_history = self._format_debate_history(rounds)
        synthesis = await self._synthesize(holding, full_history)

        return DebateResult(
            target_company=holding.name,
            target_ticker=holding.ticker,
            rounds=rounds,
            consensus_points=synthesis.get("consensus_points", []),
            dissent_points=synthesis.get("dissent_points", []),
            final_decision=self._parse_decision(
                synthesis.get("final_decision", "needs_more_data")
            ),
            final_confidence=min(
                max(synthesis.get("final_confidence", 0.5), 0.0), 1.0
            ),
            final_reasoning=synthesis.get("final_reasoning", ""),
            action_items=synthesis.get("action_items", []),
            risk_summary=synthesis.get("risk_summary", ""),
            price_assessment=synthesis.get("price_assessment", ""),
        )

    async def _synthesize(
        self,
        holding: Holding,
        debate_history: str,
    ) -> dict:
        """전체 토론을 종합하여 최종 판정."""
        prompt = (
            f"## 분석 대상\n"
            f"종목: {holding.name} ({holding.ticker})\n"
            f"현재가: {holding.current_price:,.0f} {holding.currency}\n"
            f"평균 매수가: {holding.avg_price:,.0f} {holding.currency}\n"
            f"수익률: {holding.return_pct:+.2f}%\n\n"
            f"## 애널리스트 토론 전체 기록\n{debate_history}\n\n"
            f"위 토론을 종합하여 최종 투자 판정을 내려주세요.\n"
            f"반드시 JSON 형식으로 응답하세요."
        )

        response = await self.synthesizer_client.ask(
            system=SYNTHESIS_SYSTEM_PROMPT,
            user_message=prompt,
        )

        return BaseAnalystAgent._safe_parse_json(response)

    @staticmethod
    def _format_debate_history(rounds: list[DebateRound]) -> str:
        """토론 기록을 텍스트로 포맷."""
        parts = []
        round_type_labels = {
            "opening": "1라운드: 독립 분석",
            "cross_examination": "2라운드: 상호 반론",
            "final": "3라운드: 최종 입장",
        }

        for rnd in rounds:
            label = round_type_labels.get(rnd.round_type, f"라운드 {rnd.round_number}")
            parts.append(f"\n{'='*60}\n## {label}\n{'='*60}")

            for msg in rnd.messages:
                stance_emoji = {
                    InvestmentDecision.BUY: "BUY",
                    InvestmentDecision.WAIT: "WAIT",
                    InvestmentDecision.PASS: "PASS",
                    InvestmentDecision.NEEDS_MORE_DATA: "HOLD",
                }.get(msg.stance, "?")

                parts.append(
                    f"\n### [{msg.agent_name}] — {stance_emoji} "
                    f"(확신도: {msg.confidence:.0%})\n"
                    f"{msg.content}"
                )

                if msg.agreements:
                    parts.append(
                        f"\n동의/핵심 포인트:\n"
                        + "\n".join(f"  + {a}" for a in msg.agreements)
                    )
                if msg.disagreements:
                    parts.append(
                        f"\n반론/경고:\n"
                        + "\n".join(f"  - {d}" for d in msg.disagreements)
                    )

        return "\n".join(parts)

    @staticmethod
    def _parse_decision(value: str) -> InvestmentDecision:
        """문자열을 InvestmentDecision으로 변환."""
        try:
            return InvestmentDecision(value.lower().strip())
        except ValueError:
            return InvestmentDecision.NEEDS_MORE_DATA
