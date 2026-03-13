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

        # 포트폴리오 컨텍스트 추가
        portfolio_context = (
            f"포트폴리오 전체 평가액: {portfolio.total_value:,.0f}\n"
            f"포트폴리오 수익률: {portfolio.total_return_pct:+.2f}%\n"
            f"투자 목표: {portfolio.investment_goal or '미설정'}\n"
            f"위험 성향: {portfolio.risk_tolerance or '미설정'}\n"
            f"투자 기간: {portfolio.investment_horizon or '미설정'}"
        )

        return (
            f"다음 종목의 리스크를 분석하고 투자 판단을 내려주세요.\n\n"
            f"## 종목 정보\n{info}\n\n"
            f"## 포트폴리오 컨텍스트\n{portfolio_context}\n\n"
            f"## 분석 요청\n"
            f"1. **재무 건전성**\n"
            f"   - 부채비율은 업종 평균 대비 적정한가?\n"
            f"   - 영업현금흐름은 꾸준히 플러스인가?\n"
            f"   - 이익의 질(quality of earnings)은 높은가?\n"
            f"   - 유동성 위기 가능성은?\n\n"
            f"2. **사업 리스크**\n"
            f"   - 핵심 사업의 구조적 위협은?\n"
            f"   - 기술 변화로 인한 파괴 가능성은?\n"
            f"   - 고객/공급사 집중도 리스크는?\n\n"
            f"3. **외부 리스크**\n"
            f"   - 규제 변화 리스크는?\n"
            f"   - 매크로 환경(금리, 환율) 민감도는?\n"
            f"   - 지정학적 리스크는?\n\n"
            f"4. **실패 시나리오**\n"
            f"   - 이 투자가 실패하는 Top 3 시나리오는?\n"
            f"   - 최악의 경우 최대 손실은 얼마인가?\n"
            f"   - 어떤 신호가 나타나면 손절해야 하는가?\n\n"
            f"5. **포트폴리오 적합성**\n"
            f"   - 이 종목이 포트폴리오 전체의 리스크를 높이는가?\n"
            f"   - 기존 보유 종목과 상관관계가 높은가?\n"
            f"   - 비중은 위험 성향에 부합하는가?\n\n"
            f"위 분석을 바탕으로 BUY/WAIT/PASS 중 하나의 입장을 취하세요.\n"
            f"리스크 관점에서 용인 가능한 수준인지를 기준으로 판단하세요.\n"
            f"반드시 JSON 형식으로 응답하세요."
        )
