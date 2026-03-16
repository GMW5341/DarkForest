"""
Base Analyst Agent — AI 애널리스트 에이전트 공통 인터페이스.

각 에이전트는 고유한 관점(persona)과 분석 프레임을 가지고,
독립적으로 기업을 분석한 뒤 토론에 참여한다.

설계 원칙: "자연어 사고 우선, 구조화는 마지막에"
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from src.models.analysis import (
    AgentOpinion,
    DebateMessage,
    InvestmentDecision,
    UserFeedback,
)

if TYPE_CHECKING:
    from src.api.client import ClaudeClient
    from src.models.portfolio import Holding, Portfolio


class BaseAnalystAgent(ABC):
    """AI 애널리스트 에이전트 베이스 클래스."""

    def __init__(self, client: ClaudeClient):
        self.client = client

    @property
    @abstractmethod
    def name(self) -> str:
        """에이전트 이름."""

    @property
    @abstractmethod
    def role(self) -> str:
        """에이전트 역할 설명."""

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """에이전트의 페르소나를 정의하는 시스템 프롬프트."""

    @abstractmethod
    def build_analysis_prompt(
        self,
        holding: Holding,
        portfolio: Portfolio,
    ) -> str:
        """분석 프롬프트 생성."""

    async def analyze(
        self,
        holding: Holding,
        portfolio: Portfolio,
    ) -> AgentOpinion:
        """독립 분석 수행 → 의견서 생성."""
        prompt = self.build_analysis_prompt(holding, portfolio)
        response = await self.client.ask(
            system=self.system_prompt,
            user_message=prompt,
        )
        return self._parse_opinion(response)

    async def rebut(
        self,
        holding: Holding,
        portfolio: Portfolio,
        other_opinions: list[AgentOpinion],
        round_number: int,
        user_feedback: UserFeedback | None = None,
    ) -> DebateMessage:
        """다른 에이전트의 의견을 읽고 반론/동의. 사용자 피드백 반영."""
        opinions_text = self._format_other_opinions(other_opinions)

        feedback_section = ""
        if user_feedback and user_feedback.content:
            feedback_section = (
                f"\n## 투자자가 직접 한 말 (반드시 반영하세요)\n"
                f'"{user_feedback.content}"\n'
            )

        prompt = (
            f"당신은 '{self.name}' ({self.role})입니다.\n\n"
            f"## 분석 대상\n"
            f"종목: {holding.name} ({holding.ticker})\n\n"
            f"## 다른 애널리스트들의 의견\n{opinions_text}\n"
            f"{feedback_section}\n"
            f"## 지시사항\n"
            f"다른 애널리스트들의 분석을 읽고, 전문가로서 자연스러운 대화체로 반론/동의를 작성하세요.\n\n"
            f"### 반드시 포함할 내용\n"
            f"- 동의하는 포인트: **왜** 동의하는지 논리적 근거\n"
            f"- 반대하는 포인트: 상대의 **어떤 전제/가정/논리**가 약한지 구체적 지적\n"
            f"- 당신의 수정된 입장\n"
        )
        if user_feedback and user_feedback.content:
            prompt += f"- 투자자 말에 대한 당신의 의견도 꼭 포함\n"
        prompt += (
            f"\n### ⚠️ 근거 규칙\n"
            f"- 제공된 재무 데이터/참고 자료에 있는 정보만 인용하세요.\n"
            f"- 확인되지 않은 수치를 만들어내지 마세요.\n"
            f"- 본인 추론은 '(본인 판단)' 으로 구분하세요.\n\n"
            f"### 응답 형식\n"
            f"자연어로 충분히 논의한 후, 글 맨 마지막에 아래 JSON 블록을 추가하세요:\n"
            f"```json\n"
            f'{{"stance": "buy|wait|pass|needs_more_data", "confidence": 0.0~1.0}}\n'
            f"```"
        )

        response = await self.client.ask(
            system=self.system_prompt,
            user_message=prompt,
        )
        content_text, meta = _split_response(response)

        return DebateMessage(
            agent_name=self.name,
            round_number=round_number,
            message_type="rebuttal",
            stance=self._parse_stance(meta.get("stance", "needs_more_data")),
            confidence=min(max(meta.get("confidence", 0.5), 0.0), 1.0),
            content=content_text,
            agreements=meta.get("agreements", []),
            disagreements=meta.get("disagreements", []),
        )

    async def final_position(
        self,
        holding: Holding,
        portfolio: Portfolio,
        debate_history: str,
        user_feedback: UserFeedback | None = None,
    ) -> DebateMessage:
        """토론을 종합하고 최종 입장 정리. 사용자 피드백 반영."""
        feedback_section = ""
        if user_feedback and user_feedback.content:
            feedback_section = (
                f"\n## 투자자가 직접 한 말 (반드시 반영하세요)\n"
                f'"{user_feedback.content}"\n'
            )

        prompt = (
            f"당신은 '{self.name}' ({self.role})입니다.\n\n"
            f"## 분석 대상\n"
            f"종목: {holding.name} ({holding.ticker})\n\n"
            f"## 토론 경과\n{debate_history}\n"
            f"{feedback_section}\n"
            f"## 지시사항\n"
            f"토론 전체를 종합하여 최종 입장을 자연스러운 전문가 어조로 정리하세요.\n\n"
            f"### 반드시 포함할 내용\n"
            f"- 토론을 통해 변경된 점과 그 이유\n"
            f"- 최종 투자 판단의 핵심 논거 3가지 (각각 인과 체인 포함)\n"
            f"- 구체적 행동 제안\n"
        )
        if user_feedback and user_feedback.content:
            prompt += f"- 투자자 피드백을 어떻게 반영했는지 명시\n"
        prompt += (
            f"\n### ⚠️ 근거 규칙\n"
            f"- 제공된 재무 데이터/참고 자료에 있는 정보만 인용하세요.\n"
            f"- 확인되지 않은 수치를 만들어내지 마세요.\n"
            f"- 본인 추론은 '(본인 판단)' 으로 구분하세요.\n\n"
            f"### 응답 형식\n"
            f"자연어로 충분히 정리한 후, 글 맨 마지막에 아래 JSON 블록을 추가하세요:\n"
            f"```json\n"
            f'{{"stance": "buy|wait|pass|needs_more_data", "confidence": 0.0~1.0}}\n'
            f"```"
        )

        response = await self.client.ask(
            system=self.system_prompt,
            user_message=prompt,
        )
        content_text, meta = _split_response(response)

        return DebateMessage(
            agent_name=self.name,
            round_number=3,
            message_type="final",
            stance=self._parse_stance(meta.get("stance", "needs_more_data")),
            confidence=min(max(meta.get("confidence", 0.5), 0.0), 1.0),
            content=content_text,
            agreements=meta.get("key_reasons", []),
            disagreements=[],
        )

    # ── Helpers ──

    def _parse_opinion(self, response: str) -> AgentOpinion:
        """Claude 응답을 AgentOpinion으로 파싱."""
        reasoning_text, meta = _split_response(response)
        return AgentOpinion(
            agent_name=self.name,
            agent_role=self.role,
            stance=self._parse_stance(meta.get("stance", "needs_more_data")),
            confidence=min(max(meta.get("confidence", 0.5), 0.0), 1.0),
            reasoning=reasoning_text,
            key_points=meta.get("key_points", []),
            red_flags=meta.get("red_flags", []),
            evidence=meta.get("evidence", []),
        )

    @staticmethod
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

    @staticmethod
    def _parse_stance(value: str) -> InvestmentDecision:
        """문자열을 InvestmentDecision으로 변환."""
        try:
            return InvestmentDecision(value.lower().strip())
        except ValueError:
            return InvestmentDecision.NEEDS_MORE_DATA

    @staticmethod
    def _format_other_opinions(opinions: list[AgentOpinion]) -> str:
        """다른 에이전트들의 의견을 텍스트로 포맷."""
        parts = []
        for op in opinions:
            parts.append(
                f"### {op.agent_name} ({op.agent_role})\n"
                f"- 입장: {op.stance.value.upper()} (확신도: {op.confidence:.0%})\n\n"
                f"{op.reasoning}"
            )
        return "\n\n---\n\n".join(parts)

    @staticmethod
    def _format_holding_info(holding: Holding, portfolio: Portfolio) -> str:
        """종목 정보를 분석용 텍스트로 포맷."""
        info = (
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

        if holding.financial_data:
            info += f"\n\n## 재무 데이터\n{holding.financial_data}"

        if holding.reports:
            info += "\n\n## 참고 자료/리포트"
            for i, report in enumerate(holding.reports, 1):
                info += f"\n\n### 자료 {i}\n{report}"

        return info


# ── 응답 파서 ──

def _split_response(text: str) -> tuple[str, dict]:
    """응답에서 자연어 본문과 마지막 JSON 블록을 분리."""
    json_block_pattern = re.compile(r"```json\s*\n?(.*?)\n?\s*```", re.DOTALL)
    matches = list(json_block_pattern.finditer(text))

    if matches:
        last_match = matches[-1]
        body = text[:last_match.start()].strip()
        try:
            meta = json.loads(last_match.group(1).strip())
        except json.JSONDecodeError:
            meta = {}
        return body, meta

    # 하위 호환: JSON-only 응답
    data = BaseAnalystAgent._safe_parse_json(text)
    if "reasoning" in data:
        return data.get("reasoning", text), data
    if "content" in data:
        return data.get("content", text), data
    return text, data
