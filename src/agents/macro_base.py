"""
Macro Analyst Agent — 거시 경제 분석 에이전트 공통 인터페이스.

개별 종목이 아닌 거시 경제 이슈, 시장 테마, 정책 변화 등
넓은 관점의 주제에 대해 토론하는 에이전트 베이스 클래스.

설계 원칙: "자연어 사고 우선, 구조화는 마지막에"
- Claude가 자유롭게 사고하고 글을 쓸 수 있도록 자연어 분석을 먼저 요청
- JSON은 메타데이터(입장, 확신도)만 담아 마지막에 추출
- 이렇게 하면 Claude가 깊이 있는 추론을 하면서도 구조화된 데이터를 얻을 수 있음
"""

from __future__ import annotations

import json
import re
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
            f"위 주제에 대해 당신의 관점({self.role})에서 **전문가로서 충분히 깊이 있게** 분석하세요.\n"
            f"마치 투자 위원회에서 발표하듯이 자연스러운 글로 작성하세요.\n\n"
            f"### 반드시 포함할 내용\n"
            f"1. **현재 상황 진단**: 참고 자료의 데이터를 인용하며 현황 분석\n"
            f"2. **인과관계 분석**: 왜 이런 상황이 발생했고, 앞으로 어떤 경로를 밟을지\n"
            f"   (예: 'A 요인 → B 영향 → C 결과' 형태의 논리 체인)\n"
            f"3. **투자 시사점**: 구체적 행동 제안과 **왜** 그 행동이 유효한지 근거\n"
            f"4. **리스크**: 본인 판단이 틀릴 수 있는 조건과 시나리오\n\n"
            f"### ⚠️ 근거 규칙 (최우선)\n"
            f"- 참고 자료에 있는 내용을 인용할 때: '(문서 N에 따르면, ~)'\n"
            f"- 일반적 경제 원리: '(일반적으로 ~한 경향)'\n"
            f"- 본인 추론: '(본인 판단)'\n"
            f"- **참고 자료에 없는 구체적 수치(%, bp, 금액)를 만들어내지 마세요.**\n"
            f"  확인할 수 없으면 '정확한 수치는 확인 필요'라고 쓰세요.\n\n"
            f"### 응답 형식\n"
            f"자연어로 충분히 분석한 후, 글 맨 마지막에 아래 JSON 블록을 추가하세요:\n"
            f"```json\n"
            f'{{"stance": "방어적|공격적|중립|관망", "confidence": 0.0~1.0}}\n'
            f"```"
        )

        response = await self.client.ask(
            system=self.system_prompt,
            user_message=prompt,
        )
        reasoning_text, meta = _split_response(response)

        return MacroAgentOpinion(
            agent_name=self.name,
            agent_role=self.role,
            stance=meta.get("stance", "중립"),
            confidence=min(max(meta.get("confidence", 0.5), 0.0), 1.0),
            reasoning=reasoning_text,
            key_points=meta.get("key_points", []),
            risks=meta.get("risks", []),
            action_items=meta.get("action_items", []),
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
            f"다른 전문가들의 분석을 읽고, 전문가로서 자연스러운 대화체로 반론/동의를 작성하세요.\n\n"
            f"### 반드시 포함할 내용\n"
            f"- 동의하는 포인트: **왜** 동의하는지 논리적 근거\n"
            f"- 반대하는 포인트: 상대의 **어떤 전제/가정/논리**가 약한지 구체적 지적\n"
            f"- 당신의 수정된 입장\n"
        )
        if user_feedback and user_feedback.content:
            prompt += f"- 투자자 의견에 대한 당신의 견해\n"
        prompt += (
            f"\n### ⚠️ 근거 규칙\n"
            f"- 참고 자료에 없는 수치를 만들어내지 마세요.\n"
            f"- 다른 전문가가 근거 없는 수치를 인용했으면 그 점을 지적하세요.\n"
            f"- 본인 추론은 '(본인 판단)' 으로 구분하세요.\n\n"
            f"### 응답 형식\n"
            f"자연어로 충분히 논의한 후, 글 맨 마지막에 아래 JSON 블록을 추가하세요:\n"
            f"```json\n"
            f'{{"stance": "방어적|공격적|중립|관망", "confidence": 0.0~1.0}}\n'
            f"```"
        )

        response = await self.client.ask(
            system=self.system_prompt,
            user_message=prompt,
        )
        content_text, meta = _split_response(response)

        return MacroDebateMessage(
            agent_name=self.name,
            round_number=round_number,
            message_type="rebuttal",
            stance=meta.get("stance", "중립"),
            confidence=min(max(meta.get("confidence", 0.5), 0.0), 1.0),
            content=content_text,
            agreements=meta.get("agreements", []),
            disagreements=meta.get("disagreements", []),
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
            f"토론 전체를 종합하여 최종 입장을 자연스러운 전문가 어조로 정리하세요.\n\n"
            f"### 반드시 포함할 내용\n"
            f"- 토론을 통해 변경된 점과 그 이유\n"
            f"- 최종 판단의 핵심 논거 3가지 (각각 인과 체인 포함)\n"
            f"  예: '금 비중 확대 → 이유: A 요인으로 달러 약세 → B 경로로 금 수요 증가'\n"
            f"- 투자자를 위한 구체적 행동 제안 3가지\n"
        )
        if user_feedback and user_feedback.content:
            prompt += f"- 투자자 피드백을 어떻게 반영했는지 명시\n"
        prompt += (
            f"\n### ⚠️ 근거 규칙\n"
            f"- 참고 자료에 없는 수치를 만들어내지 마세요.\n"
            f"- 본인 추론은 '(본인 판단)' 으로 구분하세요.\n\n"
            f"### 응답 형식\n"
            f"자연어로 충분히 정리한 후, 글 맨 마지막에 아래 JSON 블록을 추가하세요:\n"
            f"```json\n"
            f'{{"stance": "방어적|공격적|중립|관망", "confidence": 0.0~1.0}}\n'
            f"```"
        )

        response = await self.client.ask(
            system=self.system_prompt,
            user_message=prompt,
        )
        content_text, meta = _split_response(response)

        return MacroDebateMessage(
            agent_name=self.name,
            round_number=3,
            message_type="final",
            stance=meta.get("stance", "중립"),
            confidence=min(max(meta.get("confidence", 0.5), 0.0), 1.0),
            content=content_text,
            agreements=meta.get("key_reasons", []),
            disagreements=[],
        )


# ── Helpers ──

def _split_response(text: str) -> tuple[str, dict]:
    """
    응답에서 자연어 본문과 마지막 JSON 블록을 분리.

    Claude가 자연어로 충분히 사고한 뒤 마지막에 JSON 메타데이터를 붙이면,
    본문(reasoning/content)과 구조화 데이터(stance, confidence)를 각각 추출.
    """
    # ```json ... ``` 블록 찾기
    json_block_pattern = re.compile(r"```json\s*\n?(.*?)\n?\s*```", re.DOTALL)
    matches = list(json_block_pattern.finditer(text))

    if matches:
        last_match = matches[-1]
        # JSON 블록 이전까지가 본문
        body = text[:last_match.start()].strip()
        try:
            meta = json.loads(last_match.group(1).strip())
        except json.JSONDecodeError:
            meta = {}
        return body, meta

    # JSON 블록이 없으면 전체에서 JSON 추출 시도 (하위 호환)
    data = _safe_parse_json(text)
    if "reasoning" in data:
        return data.get("reasoning", text), data
    if "content" in data:
        return data.get("content", text), data
    return text, data


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
            f"- 입장: {op.stance} (확신도: {op.confidence:.0%})\n\n"
            f"{op.reasoning}"
        )
    return "\n\n---\n\n".join(parts)
