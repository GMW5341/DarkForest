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

## ⚠️ 할루시네이션 방지 (최우선 규칙)
- **제공된 재무 데이터(PER, PBR, ROE 등)를 기준으로 분석하세요.**
- 자동 수집된 밸류에이션 지표가 있으면 반드시 활용하세요.
- 참고 자료에 없는 동종업계 PER, 적정 주가 등의 수치를 만들어내지 마세요.
- 데이터 인용: '(재무 데이터 기준 PER X배)', '(자료 N 참조)'
- 본인 추론: '(본인 판단)'
- 데이터 부족 시: 'DCF 산출에 필요한 데이터 부족' 등 솔직히 표시하세요."""

    def build_analysis_prompt(
        self,
        holding: Holding,
        portfolio: Portfolio,
    ) -> str:
        info = self._format_holding_info(holding, portfolio)
        return (
            f"다음 종목의 밸류에이션을 **전문가로서 충분히 깊이 있게** 분석하세요.\n"
            f"마치 투자 위원회에서 발표하듯이 자연스러운 글로 작성하세요.\n\n"
            f"## 종목 정보\n{info}\n\n"
            f"## 분석 요청\n"
            f"1. **현재 밸류에이션 수준**: 제공된 PER/PBR 등 지표 기반 평가\n"
            f"2. **적정가 추정**: 가능한 범위에서 시나리오별 적정가\n"
            f"3. **안전마진 평가**: 현재가에 충분한 안전마진이 있는가?\n"
            f"4. **타이밍 평가**: 지금이 매수 적기인가?\n\n"
            f"제공된 재무 데이터(PER, PBR, ROE, 컨센서스 등)가 있으면 반드시 인용하세요.\n"
            f"데이터가 없는 지표는 분석을 생략하고 '데이터 부족'으로 표시하세요.\n"
            f"**수치를 만들어내지 마세요.**\n\n"
            f"### 응답 형식\n"
            f"자연어로 충분히 분석한 후, 글 맨 마지막에 아래 JSON 블록을 추가하세요:\n"
            f"```json\n"
            f'{{"stance": "buy|wait|pass|needs_more_data", "confidence": 0.0~1.0}}\n'
            f"```"
        )
