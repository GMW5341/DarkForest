"""
Macro Analyst Agent — 거시 경제 분석 에이전트 공통 인터페이스.

개별 종목이 아닌 거시 경제 이슈, 시장 테마, 정책 변화 등
넓은 관점의 주제에 대해 토론하는 에이전트 베이스 클래스.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from src.models.analysis import InvestmentDecision, UserFeedback

if TYPE_CHECKING:
    from src.api.client import ClaudeClient


class MacroAgentOpinion:
    """거시 분석 에이전트의 의견."""

    def __init__(
        self,
        agent_name: str,
        agent_role: str,
        stance: str,
        confidence: float,
        reasoning: str,
        key_points: list[str] | None = None,
        risks: list[str] | None = None,
        action_items: list[str] | None = None,
    ):
        self.agent_name = agent_name
        self.agent_role = agent_role
        self.stance = stance
        self.confidence = confidence
        self.reasoning = reasoning
        self.key_points = key_points or []
        self.risks = risks or []
        self.action_items = action_items or []


class MacroDebateMessage:
    """거시 토론 중 하나의 발언."""

    def __init__(
        self,
        agent_name: str,
        round_number: int,
        message_type: str,
        stance: str,
        confidence: float,
        content: str,
        agreements: list[str] | None = None,
        disagreements: list[str] | None = None,
    ):
        self.agent_name = agent_name
        self.round_number = round_number
        self.message_type = message_type
        self.stance = stance
        self.confidence = confidence
        self.content = content
        self.agreements = agreements or []
        self.disagreements = disagreements or []


class BaseMacroAgent(ABC):
    """거시 경제 분석 에이전트 베이스 클래스."""

    def __init__(self, client: ClaudeClient):
        self.client = client

    @property
    @abstractmethod
    def name(self) -> str:
        """에이전트 이름."""

    @property
    @abstractmethod
    def role(self) -> str:
        """에이전트 역할."""

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """페르소나 시스템 프롬프트."""

    async def analyze(self, topic: str, context: str = "") -> MacroAgentOpinion:
        """주제에 대한 독립 분석."""
        prompt = (
            f"## 분석 주제\n{topic}\n"
        )
        if context:
            prompt += f"\n## 배경 정보 / 참고 자료\n{context}\n"
        prompt += (
            f"\n## 지시사항\n"
            f"위 주제에 대해 당신의 관점({self.role})에서 심층 분석하세요.\n"
            f"- 현재 상황에 대한 진단\n"
            f"- 투자자에게 미치는 영향\n"
            f"- 구체적인 행동 제안\n"
            f"- **근거와 출처를 반드시 밝히세요**: '~에 따르면', '역사적으로 ~한 사례가 있었으며' 등 구체적 근거를 제시하세요. 통계, 역사적 사례, 학술 연구, 기관 보고서 등을 인용하세요.\n\n"
            f"반드시 다음 JSON으로 응답:\n"
            f'{{\n'
            f'  "stance": "방어적" | "공격적" | "중립" | "관망",\n'
            f'  "confidence": 0.0~1.0,\n'
            f'  "reasoning": "분석 내용",\n'
            f'  "key_points": ["핵심 포인트1", ...],\n'
            f'  "risks": ["리스크1", ...],\n'
            f'  "action_items": ["행동 제안1", ...]\n'
            f'}}'
        )

        response = await self.client.ask(
            system=self.system_prompt,
            user_message=prompt,
        )
        data = _safe_parse_json(response)

        return MacroAgentOpinion(
            agent_name=self.name,
            agent_role=self.role,
            stance=data.get("stance", "중립"),
            confidence=min(max(data.get("confidence", 0.5), 0.0), 1.0),
            reasoning=data.get("reasoning", response),
            key_points=data.get("key_points", []),
            risks=data.get("risks", []),
            action_items=data.get("action_items", []),
        )

    async def rebut(
        self,
        topic: str,
        other_opinions: list[MacroAgentOpinion],
        round_number: int,
        user_feedback: UserFeedback | None = None,
        context: str = "",
    ) -> MacroDebateMessage:
        """다른 에이전트 의견에 대한 반론/동의."""
        opinions_text = _format_macro_opinions(other_opinions)

        feedback_section = ""
        if user_feedback and user_feedback.content:
            feedback_section = (
                f"\n## 투자자가 직접 한 말 (반드시 반영하세요)\n"
                f'"{user_feedback.content}"\n'
            )

        prompt = (
            f"당신은 '{self.name}' ({self.role})입니다.\n\n"
            f"## 토론 주제\n{topic}\n\n"
        )
        if context:
            prompt += f"## 배경 정보\n{context}\n\n"
        prompt += (
            f"## 다른 전문가들의 의견\n{opinions_text}\n"
            f"{feedback_section}\n"
            f"## 지시사항\n"
            f"위 의견들을 읽고 반론 또는 동의를 표명하세요.\n"
            f"- 동의하는 포인트와 그 이유\n"
            f"- 반대하는 포인트와 그 근거\n"
            f"- 당신의 수정된 입장\n"
            f"- **반드시 근거를 밝히세요**: 데이터, 역사적 사례, 연구 결과 등 출처를 명시하세요\n"
        )
        if user_feedback and user_feedback.content:
            prompt += f"- 투자자 말에 대한 당신의 의견도 꼭 포함\n"
        prompt += (
            f"\n반드시 다음 JSON으로 응답:\n"
            f'{{\n'
            f'  "stance": "방어적" | "공격적" | "중립" | "관망",\n'
            f'  "confidence": 0.0~1.0,\n'
            f'  "content": "반론/동의 내용",\n'
            f'  "agreements": ["동의 포인트1", ...],\n'
            f'  "disagreements": ["반론 포인트1", ...]\n'
            f'}}'
        )

        response = await self.client.ask(
            system=self.system_prompt,
            user_message=prompt,
        )
        data = _safe_parse_json(response)

        return MacroDebateMessage(
            agent_name=self.name,
            round_number=round_number,
            message_type="rebuttal",
            stance=data.get("stance", "중립"),
            confidence=min(max(data.get("confidence", 0.5), 0.0), 1.0),
            content=data.get("content", response),
            agreements=data.get("agreements", []),
            disagreements=data.get("disagreements", []),
        )

    async def final_position(
        self,
        topic: str,
        debate_history: str,
        user_feedback: UserFeedback | None = None,
        context: str = "",
    ) -> MacroDebateMessage:
        """최종 입장 정리."""
        feedback_section = ""
        if user_feedback and user_feedback.content:
            feedback_section = (
                f"\n## 투자자가 직접 한 말 (반드시 반영하세요)\n"
                f'"{user_feedback.content}"\n'
            )

        prompt = (
            f"당신은 '{self.name}' ({self.role})입니다.\n\n"
            f"## 토론 주제\n{topic}\n\n"
        )
        if context:
            prompt += f"## 배경 정보\n{context}\n\n"
        prompt += (
            f"## 토론 경과\n{debate_history}\n"
            f"{feedback_section}\n"
            f"## 지시사항\n"
            f"토론 전체를 종합하여 최종 입장을 정리하세요.\n"
            f"- 토론을 통해 변경된 점\n"
            f"- 최종 판단과 확신도\n"
            f"- 투자자를 위한 구체적 행동 제안 3가지\n"
            f"- **각 주장의 근거를 명시하세요**: '~에 따르면', '~한 사례에서' 등 구체적 출처를 밝히세요\n"
        )
        if user_feedback and user_feedback.content:
            prompt += f"- 투자자 피드백을 어떻게 반영했는지 명시\n"
        prompt += (
            f"\n반드시 다음 JSON으로 응답:\n"
            f'{{\n'
            f'  "stance": "방어적" | "공격적" | "중립" | "관망",\n'
            f'  "confidence": 0.0~1.0,\n'
            f'  "content": "최종 입장 정리",\n'
            f'  "key_reasons": ["핵심 근거1", "핵심 근거2", "핵심 근거3"]\n'
            f'}}'
        )

        response = await self.client.ask(
            system=self.system_prompt,
            user_message=prompt,
        )
        data = _safe_parse_json(response)

        return MacroDebateMessage(
            agent_name=self.name,
            round_number=3,
            message_type="final",
            stance=data.get("stance", "중립"),
            confidence=min(max(data.get("confidence", 0.5), 0.0), 1.0),
            content=data.get("content", response),
            agreements=data.get("key_reasons", []),
            disagreements=[],
        )


# ── Helpers ──

def _safe_parse_json(text: str) -> dict:
    """JSON 파싱. 실패 시 빈 dict + 원본 보존."""
    clean = text.strip()
    try:
        if "```json" in clean:
            start = clean.index("```json") + 7
            end = clean.index("```", start)
            clean = clean[start:end].strip()
        elif "```" in clean:
            start = clean.index("```") + 3
            end = clean.index("```", start)
            clean = clean[start:end].strip()
    except ValueError:
        pass

    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        brace_start = text.find("{")
        brace_end = text.rfind("}")
        if brace_start != -1 and brace_end > brace_start:
            try:
                return json.loads(text[brace_start:brace_end + 1])
            except json.JSONDecodeError:
                pass
        return {"reasoning": text}


def _format_macro_opinions(opinions: list[MacroAgentOpinion]) -> str:
    """다른 에이전트 의견을 텍스트로 포맷."""
    parts = []
    for op in opinions:
        parts.append(
            f"### {op.agent_name} ({op.agent_role})\n"
            f"- 입장: {op.stance}\n"
            f"- 확신도: {op.confidence:.0%}\n"
            f"- 분석: {op.reasoning}\n"
            f"- 핵심 포인트: {', '.join(op.key_points) if op.key_points else '없음'}\n"
            f"- 리스크: {', '.join(op.risks) if op.risks else '없음'}"
        )
    return "\n\n".join(parts)
