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

## ⚠️ 할루시네이션 방지 (최우선 규칙)
- **제공된 재무 데이터와 참고 자료에 있는 정보만 인용하세요.**
- 자동 수집 재무 데이터가 있으면 그 수치를 기준으로 분석하세요.
- 참고 자료에 없는 구체적 수치(매출액, 영업이익률, 성장률 등)를 만들어내지 마세요.
- 데이터가 인용할 때: '(재무 데이터 기준)', '(자료 N 참조)'
- 본인 추론: '(본인 판단)'
- 데이터가 부족하면 솔직히 '해당 데이터 부족' 이라고 쓰세요."""

    def build_analysis_prompt(
        self,
        holding: Holding,
        portfolio: Portfolio,
    ) -> str:
        info = self._format_holding_info(holding, portfolio)
        return (
            f"다음 종목을 펀더멘털 관점에서 **전문가로서 충분히 깊이 있게** 분석하세요.\n"
            f"마치 투자 위원회에서 발표하듯이 자연스러운 글로 작성하세요.\n\n"
            f"## 종목 정보\n{info}\n\n"
            f"## 분석 요청\n"
            f"1. **사업 정체성**: 이 회사는 본질적으로 무엇을 파는 회사인가?\n"
            f"2. **매출 구조**: 매출은 어디서, 어떻게 발생하는가?\n"
            f"3. **경쟁 해자**: 경쟁자가 모방하기 어려운 이유는?\n"
            f"4. **성장 동력**: 향후 3~5년 성장은 어디서 오는가?\n\n"
            f"각 항목에 대해 **왜** 그렇게 판단하는지 인과관계를 포함하세요.\n"
            f"제공된 재무 데이터가 있으면 반드시 인용하며 분석하세요.\n"
            f"데이터가 부족한 항목은 '데이터 부족'으로 솔직히 표시하세요.\n\n"
            f"### 응답 형식\n"
            f"자연어로 충분히 분석한 후, 글 맨 마지막에 아래 JSON 블록을 추가하세요:\n"
            f"```json\n"
            f'{{"stance": "buy|wait|pass|needs_more_data", "confidence": 0.0~1.0}}\n'
            f"```"
        )
