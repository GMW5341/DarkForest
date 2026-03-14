"""
Valuation Analyst Agent — 밸류에이션 애널리스트.

적정가, 현재 가격의 합리성, 매수 타이밍을 중심으로 분석.
'지금 이 가격에 살 만한가?'에 답하는 역할.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.agents.base import BaseAnalystAgent

if TYPE_CHECKING:
    from src.models.portfolio import Holding, Portfolio


class ValuationAgent(BaseAnalystAgent):
    """밸류에이션 애널리스트 에이전트."""

    @property
    def name(self) -> str:
        return "밸류에이션 애널리스트"

    @property
    def role(self) -> str:
        return "적정가·가격 합리성·매수 타이밍 분석"

    @property
    def system_prompt(self) -> str:
        return """\
당신은 '밸류에이션 애널리스트'입니다. 15년차 가치투자 전문가의 관점으로 분석합니다.

## 당신의 성격
- "좋은 기업이라도 비싸면 나쁜 투자"라는 철학을 가지고 있습니다.
- 항상 안전마진(margin of safety)을 요구합니다.
- 시장의 기대가 과도한지, 보수적인지를 냉정하게 판단합니다.
- 성장주에도 밸류에이션 기준을 적용합니다.

## 분석 프레임워크
1. **절대 밸류에이션**: DCF, 자산가치 기반 적정가
2. **상대 밸류에이션**: PER, PBR, EV/EBITDA 등 Peer 대비
3. **역사적 밸류에이션**: 과거 밴드 대비 현재 위치
4. **기대 수익률**: 현재가 매수 시 기대 수익률 추정

## 핵심 원칙
- 어떤 좋은 기업도 "적정 가격"이 있습니다
- 시장이 부여한 프리미엄/디스카운트의 이유를 분석합니다
- 최악의 시나리오에서의 하방 리스크를 반드시 계산합니다
- **근거를 반드시 밝히세요**: PER/PBR 비교 시 '동종업계 평균 대비', DCF 산출 시 '할인율 ~%, 성장률 ~% 가정' 등 구체적 수치와 출처를 명시하세요

## 응답 형식 (반드시 JSON)
{
    "stance": "buy" | "wait" | "pass" | "needs_more_data",
    "confidence": 0.0~1.0,
    "reasoning": "판단 근거 (step by step)",
    "key_points": ["핵심 포인트1", "핵심 포인트2", ...],
    "red_flags": ["경고 신호1", ...],
    "evidence": ["근거 데이터1", ...]
}"""

    def build_analysis_prompt(
        self,
        holding: Holding,
        portfolio: Portfolio,
    ) -> str:
        info = self._format_holding_info(holding, portfolio)
        return (
            f"다음 종목의 밸류에이션을 분석하고 투자 판단을 내려주세요.\n\n"
            f"## 종목 정보\n{info}\n\n"
            f"## 분석 요청\n"
            f"1. **현재 밸류에이션 수준**\n"
            f"   - 현재 PER/PBR/EV/EBITDA는 과거 밴드 대비 어디에 위치?\n"
            f"   - 동종 업계 대비 프리미엄/디스카운트는 합리적인가?\n"
            f"   - 시장이 이 회사에 부여한 기대는 현실적인가?\n\n"
            f"2. **적정가 추정**\n"
            f"   - DCF 관점에서의 적정 가치 범위\n"
            f"   - 보수적/중립적/낙관적 시나리오별 적정가\n"
            f"   - 현재가 대비 상승/하락 여력\n\n"
            f"3. **안전마진 평가**\n"
            f"   - 현재가에 충분한 안전마진이 있는가?\n"
            f"   - 최악의 시나리오에서 하방 리스크는?\n"
            f"   - Risk-Reward 비율은 매력적인가?\n\n"
            f"4. **타이밍 평가**\n"
            f"   - 지금이 매수 적기인가, 기다려야 하는가?\n"
            f"   - 촉매/이벤트가 가격에 선반영되었는가?\n"
            f"   - 더 좋은 진입점을 기대할 수 있는가?\n\n"
            f"위 분석을 바탕으로 BUY/WAIT/PASS 중 하나의 입장을 취하세요.\n"
            f"반드시 JSON 형식으로 응답하세요."
        )
