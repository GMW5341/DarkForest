"""
Fundamental Analyst Agent — 펀더멘털 애널리스트.

사업 모델, 매출 구조, 경쟁력, 성장성을 중심으로 분석.
'이 회사는 진짜 좋은 회사인가?'에 답하는 역할.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.agents.base import BaseAnalystAgent

if TYPE_CHECKING:
    from src.models.portfolio import Holding, Portfolio


class FundamentalAgent(BaseAnalystAgent):
    """펀더멘털 애널리스트 에이전트."""

    @property
    def name(self) -> str:
        return "펀더멘털 애널리스트"

    @property
    def role(self) -> str:
        return "사업 모델·매출 구조·경쟁력·성장성 분석"

    @property
    def system_prompt(self) -> str:
        return """\
당신은 '펀더멘털 애널리스트'입니다. 10년차 바이사이드 리서치 애널리스트의 관점으로 분석합니다.

## 당신의 성격
- 숫자와 팩트에 집착합니다. "느낌"이나 "분위기"로 판단하지 않습니다.
- "이 회사가 10년 뒤에도 존재할 것인가?"를 항상 자문합니다.
- 경영진의 말보다 재무제표의 숫자를 믿습니다.
- 좋은 기업과 좋은 투자는 다르다는 것을 잘 알고 있습니다.

## 분석 프레임워크
1. **사업 정체성**: 이 회사는 본질적으로 무엇을 파는가?
2. **매출 구조**: 매출원이 건강하고 다변화되어 있는가?
3. **경쟁 해자**: 경쟁자가 모방하기 어려운 이유는?
4. **성장 동력**: 향후 3~5년 성장은 어디서 오는가?

## 핵심 원칙
- 분석의 논리 정합성이 결론보다 중요합니다
- 불확실한 부분은 솔직하게 '모른다'고 밝히세요
- 경고 신호(red flags)는 반드시 명시하세요
- **근거를 반드시 밝히세요**: 재무 데이터 인용 시 '~분기 실적 기준', 산업 데이터는 '~에 따르면' 등 출처를 명시하세요. 유사 기업 사례, 과거 실적 추이 등 구체적 근거를 제시하세요.

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
            f"다음 종목을 펀더멘털 관점에서 깊이 분석하고 투자 판단을 내려주세요.\n\n"
            f"## 종목 정보\n{info}\n\n"
            f"## 분석 요청\n"
            f"1. **사업 정체성**: 이 회사는 본질적으로 무엇을 파는 회사인가?\n"
            f"   - 핵심 제품/서비스를 한 문장으로 설명\n"
            f"   - 이 회사가 없어지면 고객은 어디로 가는가?\n"
            f"   - 5년 후에도 같은 사업을 하고 있을 것인가?\n\n"
            f"2. **매출 구조**: 매출은 어디서, 어떻게 발생하는가?\n"
            f"   - 매출의 주요 소스와 비중\n"
            f"   - 반복 매출(recurring) 비중\n"
            f"   - 매출 집중도 리스크\n\n"
            f"3. **경쟁 해자**: 경쟁자가 모방하기 어려운 이유는?\n"
            f"   - 진입 장벽의 종류와 강도\n"
            f"   - 전환 비용(switching cost)\n"
            f"   - 해자의 지속 가능성\n\n"
            f"4. **성장 동력**: 향후 성장은 어디서 오는가?\n"
            f"   - 유기적 성장 vs M&A\n"
            f"   - TAM(Total Addressable Market) 전망\n"
            f"   - 성장률 추이와 전망\n\n"
            f"위 분석을 바탕으로 BUY/WAIT/PASS 중 하나의 입장을 취하세요.\n"
            f"반드시 JSON 형식으로 응답하세요."
        )
