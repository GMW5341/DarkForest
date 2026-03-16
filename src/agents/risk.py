"""
Risk Analyst Agent — 리스크 애널리스트.

재무 건전성, 규제 리스크, 시장 리스크, 실패 시나리오를 중심으로 분석.
'이 투자가 망하는 시나리오는 무엇인가?'에 답하는 역할.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.agents.base import BaseAnalystAgent

if TYPE_CHECKING:
    from src.models.portfolio import Holding, Portfolio


class RiskAgent(BaseAnalystAgent):
    """리스크 애널리스트 에이전트."""

    @property
    def name(self) -> str:
        return "리스크 애널리스트"

    @property
    def role(self) -> str:
        return "재무 건전성·규제 리스크·실패 시나리오 분석"

    @property
    def system_prompt(self) -> str:
        return """\
당신은 '리스크 애널리스트'입니다. 대형 운용사의 리스크 관리 책임자(CRO)의 관점으로 분석합니다.

## 당신의 성격
- 본질적으로 회의적(skeptical)입니다. "무엇이 잘못될 수 있는가?"를 먼저 생각합니다.
- 낙관론에 항상 의문을 제기합니다. 최악의 시나리오를 반드시 고려합니다.
- "생존 편향"을 경계합니다 — 성공 사례만 보고 리스크를 과소평가하지 않습니다.
- 정량적 리스크 분석을 선호하지만, 정성적 리스크도 무시하지 않습니다.

## 분석 프레임워크
1. **재무 리스크**: 부채, 현금흐름, 유동성
2. **사업 리스크**: 경쟁 심화, 기술 변화, 고객 집중도
3. **규제/정책 리스크**: 규제 변화, 정책 영향
4. **시장 리스크**: 매크로 환경, 금리, 환율, 지정학

## 핵심 원칙
- 리스크 없는 투자는 없습니다. 핵심은 "감수할 만한 리스크인가?"입니다.
- 리스크를 정량화할 수 없으면 최소한 시나리오로 표현하세요
- "최악의 경우 얼마를 잃을 수 있는가?"에 반드시 답하세요
- 손절 기준을 명확히 제시하세요

## ⚠️ 할루시네이션 방지 (최우선 규칙)
- **제공된 재무 데이터(부채비율, 시총, 외국인 지분 등)를 기준으로 분석하세요.**
- 참고 자료에 없는 수치를 만들어내지 마세요.
- 데이터 인용: '(재무 데이터 기준)', '(자료 N 참조)'
- 본인 추론: '(본인 판단)'
- 데이터 부족 시: '해당 데이터 부족으로 정량 분석 불가' 등 솔직히 표시하세요."""

    def build_analysis_prompt(
        self,
        holding: Holding,
        portfolio: Portfolio,
    ) -> str:
        info = self._format_holding_info(holding, portfolio)

        portfolio_context = (
            f"포트폴리오 전체 평가액: {portfolio.total_value:,.0f}\n"
            f"포트폴리오 수익률: {portfolio.total_return_pct:+.2f}%\n"
            f"투자 목표: {portfolio.investment_goal or '미설정'}\n"
            f"위험 성향: {portfolio.risk_tolerance or '미설정'}\n"
            f"투자 기간: {portfolio.investment_horizon or '미설정'}"
        )

        return (
            f"다음 종목의 리스크를 **전문가로서 충분히 깊이 있게** 분석하세요.\n"
            f"마치 리스크 관리 위원회에서 보고하듯이 자연스러운 글로 작성하세요.\n\n"
            f"## 종목 정보\n{info}\n\n"
            f"## 포트폴리오 컨텍스트\n{portfolio_context}\n\n"
            f"## 분석 요청\n"
            f"1. **재무 건전성**: 부채비율, 현금흐름, 유동성 분석\n"
            f"2. **사업 리스크**: 경쟁 심화, 기술 변화, 고객 집중도\n"
            f"3. **외부 리스크**: 규제, 매크로, 지정학\n"
            f"4. **실패 시나리오**: 이 투자가 실패하는 Top 3 시나리오\n"
            f"5. **포트폴리오 적합성**: 전체 리스크에 미치는 영향\n\n"
            f"제공된 재무 데이터(부채비율, 외국인 지분, 52주 고저 등)가 있으면 반드시 인용하세요.\n"
            f"데이터가 없는 항목은 정성적으로 분석하되 '정량 데이터 부족'으로 표시하세요.\n\n"
            f"### 응답 형식\n"
            f"자연어로 충분히 분석한 후, 글 맨 마지막에 아래 JSON 블록을 추가하세요:\n"
            f"```json\n"
            f'{{"stance": "buy|wait|pass|needs_more_data", "confidence": 0.0~1.0}}\n'
            f"```"
        )
