"""
Claude API Client — Anthropic API 통합 레이어.

추론 엔진이 Claude와 통신하기 위한 클라이언트.
텍스트 전용 ask()와 이미지 포함 ask_with_images()를 제공한다.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os

import anthropic

from src.config import UsageTracker

logger = logging.getLogger(__name__)

_MAX_RETRIES = 4
_BASE_DELAY = 2.0  # seconds


class ClaudeClient:
    """Anthropic Claude API 클라이언트."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-opus-4-20250514",
        max_tokens: int = 4096,
        usage_tracker: UsageTracker | None = None,
    ):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY가 설정되지 않았습니다. "
                "환경변수를 설정하거나 api_key 파라미터를 전달하세요."
            )
        self.model = model
        self.max_tokens = max_tokens
        self.usage_tracker = usage_tracker or UsageTracker()
        self._client = anthropic.AsyncAnthropic(api_key=self.api_key)

    def _track_usage(self, message) -> None:
        """API 응답에서 사용량을 추적."""
        if hasattr(message, "usage"):
            self.usage_tracker.add(
                model=self.model,
                input_tokens=message.usage.input_tokens,
                output_tokens=message.usage.output_tokens,
            )

    async def _call_with_retry(self, create_fn):
        """429 Rate Limit 에러 시 지수 백오프로 재시도."""
        for attempt in range(_MAX_RETRIES + 1):
            try:
                return await create_fn()
            except anthropic.RateLimitError as e:
                if attempt == _MAX_RETRIES:
                    raise
                # retry-after 헤더가 있으면 사용, 없으면 지수 백오프
                retry_after = getattr(e.response, "headers", {}).get("retry-after")
                if retry_after:
                    delay = float(retry_after)
                else:
                    delay = _BASE_DELAY * (2 ** attempt)
                logger.warning(
                    f"Rate limit (429). {delay:.1f}초 대기 후 재시도 ({attempt + 1}/{_MAX_RETRIES})..."
                )
                await asyncio.sleep(delay)

    async def ask(
        self,
        user_message: str,
        system: str = "",
    ) -> str:
        """Claude에게 질문하고 텍스트 응답을 반환."""
        async def _create():
            return await self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system,
                messages=[
                    {"role": "user", "content": user_message},
                ],
            )

        message = await self._call_with_retry(_create)
        self._track_usage(message)
        # 텍스트 블록만 추출
        text_parts = [
            block.text
            for block in message.content
            if block.type == "text"
        ]
        return "\n".join(text_parts)

    async def ask_with_images(
        self,
        text_prompt: str,
        images: list[tuple[bytes, str]],
        system: str = "",
    ) -> str:
        """
        이미지와 텍스트를 함께 보내고 응답을 받는다 (Vision).

        Args:
            text_prompt: 텍스트 프롬프트
            images: (이미지 바이트, media_type) 튜플 리스트
                    media_type 예: "image/png", "image/jpeg"
            system: 시스템 프롬프트

        Returns:
            Claude의 텍스트 응답
        """
        content: list[dict] = []

        for img_bytes, media_type in images:
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": base64.standard_b64encode(img_bytes).decode("ascii"),
                },
            })

        content.append({"type": "text", "text": text_prompt})

        async def _create():
            return await self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system,
                messages=[{"role": "user", "content": content}],
            )

        message = await self._call_with_retry(_create)
        self._track_usage(message)

        text_parts = [
            block.text
            for block in message.content
            if block.type == "text"
        ]
        return "\n".join(text_parts)
