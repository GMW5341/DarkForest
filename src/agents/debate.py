"""
Debate Orchestrator — Multi-Agent Debate 오케스트레이터 (Human-in-the-Loop).

여러 AI 애널리스트 에이전트가 토론을 통해 최종 투자 판정에 도달하되,
각 라운드 사이에 사용자가 피드백을 제공하여 토론을 조율할 수 있다.

토론 플로우:
  Round 1 (Opening): 각 에이전트 독립 분석 → 의견서 제출
    ⏸ 사용자 리뷰 & 피드백
  Round 2 (Cross-Examination): 피드백 반영 + 상호 반론
    ⏸ 사용자 리뷰 & 피드백
  Round 3 (Final Position): 피드백 반영 + 최종 입장
    ⏸ 사용자 최종 확인
  Synthesis: 전체 토론 + 사용자 의견 종합 → BUY/WAIT/PASS
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.agents.base import BaseAnalystAgent
from src.models.analysis import (
    AgentOpinion,
    DebateMessage,
    DebatePhase,
    DebateResult,
    DebateRound,
    InvestmentDecision,
    UserFeedback,
)

if TYPE_CHECKING:
    from src.api.client import ClaudeClient
    from src.models.portfolio import Holding, Portfolio


SYNTHESIS_SYSTEM_PROMPT = """\
당신은 투자위원회(Investment Committee)의 의장입니다.

여러 애널리스트들의 토론과 **투자자 본인의 피드백**을 종합하여 최종 투자 판정을 내립니다.

## 당신의 역할
- 각 애널리스트의 논리를 객관적으로 평가합니다
- **투자자의 피드백과 관점을 최우선으로 존중합니다**
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
    Multi-Agent Debate 오케스트레이터 (Human-in-the-Loop).

    각 라운드를 개별 step으로 실행하고, 라운드 사이에
    사용자가 피드백을 제공하여 토론 방향을 조율할 수 있다.
    """

    def __init__(
        self,
        agents: list[BaseAnalystAgent],
        synthesizer_client: ClaudeClient,
    ):
        self.agents = agents
        self.synthesizer_client = synthesizer_client

    # ── Step-by-Step API ──

    async def run_round1(
        self,
        holding: Holding,
        portfolio: Portfolio,
    ) -> tuple[DebateRound, list[AgentOpinion]]:
        """Round 1: 각 에이전트 독립 분석. 결과 반환 후 사용자 피드백 대기."""
        opinions: list[AgentOpinion] = []
        messages: list[DebateMessage] = []

        for agent in self.agents:
            opinion = await agent.analyze(holding, portfolio)
            opinions.append(opinion)
            messages.append(DebateMessage(
                agent_name=opinion.agent_name,
                round_number=1,
                message_type="opening",
                stance=opinion.stance,
                confidence=opinion.confidence,
                content=opinion.reasoning,
                agreements=opinion.key_points,
                disagreements=opinion.red_flags,
            ))

        round1 = DebateRound(
            round_number=1,
            round_type="opening",
            messages=messages,
        )
        return round1, opinions

    async def run_round2(
        self,
        holding: Holding,
        portfolio: Portfolio,
        opinions: list[AgentOpinion],
        user_feedback: UserFeedback | None = None,
    ) -> DebateRound:
        """Round 2: 사용자 피드백 반영 + 상호 반론."""
        messages: list[DebateMessage] = []

        for i, agent in enumerate(self.agents):
            other_opinions = [op for j, op in enumerate(opinions) if j != i]
            rebuttal = await agent.rebut(
                holding, portfolio, other_opinions,
                round_number=2,
                user_feedback=user_feedback,
            )
            messages.append(rebuttal)

        return DebateRound(
            round_number=2,
            round_type="cross_examination",
            messages=messages,
        )

    async def run_round3(
        self,
        holding: Holding,
        portfolio: Portfolio,
        rounds: list[DebateRound],
        user_feedback: UserFeedback | None = None,
    ) -> DebateRound:
        """Round 3: 사용자 피드백 반영 + 최종 입장 정리."""
        debate_history = self._format_debate_history(rounds)
        messages: list[DebateMessage] = []

        for agent in self.agents:
            final = await agent.final_position(
                holding, portfolio, debate_history,
                user_feedback=user_feedback,
            )
            messages.append(final)

        return DebateRound(
            round_number=3,
            round_type="final",
            messages=messages,
        )

    async def synthesize(
        self,
        holding: Holding,
        rounds: list[DebateRound],
        user_feedbacks: list[UserFeedback],
        user_final_comment: str = "",
    ) -> dict:
        """전체 토론 + 사용자 피드백을 종합하여 최종 판정."""
        full_history = self._format_debate_history(rounds)

        # 사용자 피드백 히스토리 포맷
        feedback_section = ""
        if user_feedbacks:
            feedback_parts = []
            round_names = ["Round 1 후", "Round 2 후", "Round 3 후"]
            for i, fb in enumerate(user_feedbacks):
                label = round_names[i] if i < len(round_names) else f"피드백 {i+1}"
                parts = [f"### {label}"]
                parts.append(f"- 의견: {fb.content}")
                if fb.focus_on:
                    parts.append(f"- 집중 요청: {', '.join(fb.focus_on)}")
                if fb.additional_context:
                    parts.append(f"- 추가 정보: {fb.additional_context}")
                if fb.override_stance:
                    parts.append(f"- 투자자 선호 방향: {fb.override_stance.upper()}")
                feedback_parts.append("\n".join(parts))
            feedback_section = (
                "\n\n## 투자자 본인의 피드백 (라운드별)\n"
                + "\n\n".join(feedback_parts)
            )

        if user_final_comment:
            feedback_section += f"\n\n## 투자자 최종 코멘트\n{user_final_comment}"

        prompt = (
            f"## 분석 대상\n"
            f"종목: {holding.name} ({holding.ticker})\n"
            f"현재가: {holding.current_price:,.0f} {holding.currency}\n"
            f"평균 매수가: {holding.avg_price:,.0f} {holding.currency}\n"
            f"수익률: {holding.return_pct:+.2f}%\n\n"
            f"## 애널리스트 토론 전체 기록\n{full_history}"
            f"{feedback_section}\n\n"
            f"위 토론과 투자자 피드백을 종합하여 최종 투자 판정을 내려주세요.\n"
            f"반드시 JSON 형식으로 응답하세요."
        )

        response = await self.synthesizer_client.ask(
            system=SYNTHESIS_SYSTEM_PROMPT,
            user_message=prompt,
        )

        return BaseAnalystAgent._safe_parse_json(response)

    # ── Convenience: 자동 모드 (기존 호환) ──

    async def run_debate(
        self,
        holding: Holding,
        portfolio: Portfolio,
    ) -> DebateResult:
        """전체 토론을 자동으로 실행 (피드백 없이)."""
        round1, opinions = await self.run_round1(holding, portfolio)
        rounds = [round1]

        round2 = await self.run_round2(holding, portfolio, opinions)
        rounds.append(round2)

        round3 = await self.run_round3(holding, portfolio, rounds)
        rounds.append(round3)

        synthesis = await self.synthesize(holding, rounds, [])

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
            phase=DebatePhase.SYNTHESIZED,
        )

    # ── Helpers ──

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
                stance_label = {
                    InvestmentDecision.BUY: "BUY",
                    InvestmentDecision.WAIT: "WAIT",
                    InvestmentDecision.PASS: "PASS",
                    InvestmentDecision.NEEDS_MORE_DATA: "HOLD",
                }.get(msg.stance, "?")

                parts.append(
                    f"\n### [{msg.agent_name}] — {stance_label} "
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
