"""
Claude API Client — Anthropic API 통합 레이어.

추론 엔진이 Claude와 통신하기 위한 클라이언트.
"""

from __future__ import annotations

import os

import anthropic


class ClaudeClient:
    """Anthropic Claude API 클라이언트."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 4096,
    ):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY가 설정되지 않았습니다. "
                "환경변수를 설정하거나 api_key 파라미터를 전달하세요."
            )
        self.model = model
        self.max_tokens = max_tokens
        self._client = anthropic.AsyncAnthropic(api_key=self.api_key)

    async def ask(
        self,
        user_message: str,
        system: str = "",
    ) -> str:
        """Claude에게 질문하고 텍스트 응답을 반환."""
        message = await self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[
                {"role": "user", "content": user_message},
            ],
        )
        # 텍스트 블록만 추출
        text_parts = [
            block.text
            for block in message.content
            if block.type == "text"
        ]
        return "\n".join(text_parts)
