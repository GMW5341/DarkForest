"""
Macro Debate Orchestrator — 거시 경제 토론 오케스트레이터 (Human-in-the-Loop).

개별 종목이 아닌 거시 경제 이슈/테마에 대해
3명의 매크로 전문가가 토론하고 사용자가 조율한다.

토론 플로우:
  Round 1 (Opening): 각 전문가 독립 분석 → 의견 제출
    ⏸ 사용자 리뷰 & 피드백
  Round 2 (Cross-Examination): 피드백 반영 + 상호 반론
    ⏸ 사용자 리뷰 & 피드백
  Round 3 (Final Position): 피드백 반영 + 최종 입장
    ⏸ 사용자 최종 확인
  Synthesis: 전체 종합 → 투자 전략 권고
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.agents.macro_base import (
    BaseMacroAgent,
    MacroAgentOpinion,
    MacroDebateMessage,
    _safe_parse_json,
)
from src.models.analysis import UserFeedback

if TYPE_CHECKING:
    from src.api.client import ClaudeClient


MACRO_SYNTHESIS_PROMPT = """\
당신은 투자 전략 회의의 의장(Chief Investment Strategist)입니다.

여러 매크로 전문가들의 토론과 **투자자 본인의 피드백**을 종합하여
현 거시 경제 상황에 대한 최종 투자 전략 권고를 내립니다.

## 당신의 역할
- 각 전문가의 논리를 객관적으로 평가합니다
- **투자자의 피드백과 관심사를 최우선으로 존중합니다**
- 합의된 포인트와 갈린 포인트를 명확히 구분합니다
- 실행 가능한 구체적 전략을 제시합니다

## 응답 형식 (반드시 JSON)
{
    "overall_stance": "방어적" | "공격적" | "중립" | "관망",
    "confidence": 0.0~1.0,
    "executive_summary": "핵심 요약 (3-5문장)",
    "consensus_points": ["합의 포인트1", ...],
    "dissent_points": ["의견 불일치 포인트1", ...],
    "investment_implications": [
        {"category": "자산 배분" | "섹터" | "방어 전략" | "기회 포착",
         "recommendation": "구체적 권고",
         "rationale": "근거"}
    ],
    "risk_scenarios": [
        {"scenario": "시나리오 설명",
         "probability": "높음/중간/낮음",
         "impact": "높음/중간/낮음",
         "hedge": "대응 방안"}
    ],
    "action_items": ["지금 당장 해야 할 것1", ...],
    "monitoring_points": ["앞으로 주시해야 할 것1", ...]
}"""


class MacroDebateRound:
    """거시 토론 라운드."""

    def __init__(
        self,
        round_number: int,
        round_type: str,
        messages: list[MacroDebateMessage] | None = None,
    ):
        self.round_number = round_number
        self.round_type = round_type
        self.messages = messages or []

    def to_dict(self) -> dict:
        return {
            "round_number": self.round_number,
            "round_type": self.round_type,
            "messages": [
                {
                    "agent_name": m.agent_name,
                    "round_number": m.round_number,
                    "message_type": m.message_type,
                    "stance": m.stance,
                    "confidence": m.confidence,
                    "content": m.content,
                    "agreements": m.agreements,
                    "disagreements": m.disagreements,
                }
                for m in self.messages
            ],
        }


class MacroDebateOrchestrator:
    """거시 경제 토론 오케스트레이터."""

    def __init__(
        self,
        agents: list[BaseMacroAgent],
        synthesizer_client: ClaudeClient,
    ):
        self.agents = agents
        self.synthesizer_client = synthesizer_client

    async def run_round1(
        self,
        topic: str,
        context: str = "",
    ) -> tuple[MacroDebateRound, list[MacroAgentOpinion]]:
        """Round 1: 각 전문가 독립 분석."""
        opinions: list[MacroAgentOpinion] = []
        messages: list[MacroDebateMessage] = []

        for agent in self.agents:
            opinion = await agent.analyze(topic, context)
            opinions.append(opinion)
            messages.append(MacroDebateMessage(
                agent_name=opinion.agent_name,
                round_number=1,
                message_type="opening",
                stance=opinion.stance,
                confidence=opinion.confidence,
                content=opinion.reasoning,
                agreements=opinion.key_points,
                disagreements=opinion.risks,
            ))

        round1 = MacroDebateRound(
            round_number=1,
            round_type="opening",
            messages=messages,
        )
        return round1, opinions

    async def run_round2(
        self,
        topic: str,
        opinions: list[MacroAgentOpinion],
        user_feedback: UserFeedback | None = None,
        context: str = "",
    ) -> MacroDebateRound:
        """Round 2: 피드백 반영 + 상호 반론."""
        messages: list[MacroDebateMessage] = []

        for i, agent in enumerate(self.agents):
            other_opinions = [op for j, op in enumerate(opinions) if j != i]
            rebuttal = await agent.rebut(
                topic, other_opinions,
                round_number=2,
                user_feedback=user_feedback,
                context=context,
            )
            messages.append(rebuttal)

        return MacroDebateRound(
            round_number=2,
            round_type="cross_examination",
            messages=messages,
        )

    async def run_round3(
        self,
        topic: str,
        rounds: list[MacroDebateRound],
        user_feedback: UserFeedback | None = None,
        context: str = "",
    ) -> MacroDebateRound:
        """Round 3: 최종 입장 정리."""
        debate_history = self._format_debate_history(rounds)
        messages: list[MacroDebateMessage] = []

        for agent in self.agents:
            final = await agent.final_position(
                topic, debate_history,
                user_feedback=user_feedback,
                context=context,
            )
            messages.append(final)

        return MacroDebateRound(
            round_number=3,
            round_type="final",
            messages=messages,
        )

    async def synthesize(
        self,
        topic: str,
        rounds: list[MacroDebateRound],
        user_feedbacks: list[UserFeedback],
        user_final_comment: str = "",
        context: str = "",
    ) -> dict:
        """전체 토론 + 사용자 피드백 종합 → 최종 전략 권고."""
        full_history = self._format_debate_history(rounds)

        feedback_section = ""
        if user_feedbacks:
            feedback_parts = []
            round_names = ["Round 1 후", "Round 2 후", "Round 3 후"]
            for i, fb in enumerate(user_feedbacks):
                label = round_names[i] if i < len(round_names) else f"피드백 {i+1}"
                parts = [f"### {label}"]
                parts.append(f"- 의견: {fb.content}")
                feedback_parts.append("\n".join(parts))
            feedback_section = (
                "\n\n## 투자자 본인의 피드백 (라운드별)\n"
                + "\n\n".join(feedback_parts)
            )

        if user_final_comment:
            feedback_section += f"\n\n## 투자자 최종 코멘트\n{user_final_comment}"

        prompt = (
            f"## 토론 주제\n{topic}\n\n"
        )
        if context:
            prompt += f"## 배경 정보\n{context}\n\n"
        prompt += (
            f"## 전문가 토론 전체 기록\n{full_history}"
            f"{feedback_section}\n\n"
            f"위 토론과 투자자 피드백을 종합하여 최종 투자 전략 권고를 내려주세요.\n"
            f"반드시 JSON 형식으로 응답하세요."
        )

        response = await self.synthesizer_client.ask(
            system=MACRO_SYNTHESIS_PROMPT,
            user_message=prompt,
        )

        return _safe_parse_json(response)

    @staticmethod
    def _format_debate_history(rounds: list[MacroDebateRound]) -> str:
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
                parts.append(
                    f"\n### [{msg.agent_name}] — {msg.stance} "
                    f"(확신도: {msg.confidence:.0%})\n"
                    f"{msg.content}"
                )

                if msg.agreements:
                    parts.append(
                        f"\n핵심 포인트:\n"
                        + "\n".join(f"  + {a}" for a in msg.agreements)
                    )
                if msg.disagreements:
                    parts.append(
                        f"\n리스크/반론:\n"
                        + "\n".join(f"  - {d}" for d in msg.disagreements)
                    )

        return "\n".join(parts)
