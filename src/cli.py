"""
CLI Entry Point — 대화형 커맨드라인 인터페이스.

LLM과 대화하듯이 토론에 참여할 수 있는 인터랙티브 CLI.

Usage:
    darkforest portfolio.json          # 인터랙티브 대화 모드
    darkforest --example               # 예시 포트폴리오
    darkforest --example --auto        # 자동 모드 (대화 없이)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from src.analyzer import PortfolioAnalyzer
from src.agents.debate import DebateOrchestrator
from src.models.analysis import (
    DebatePhase,
    DebateResult,
    InvestmentDecision,
    UserFeedback,
)
from src.models.portfolio import AssetClass, Holding, Portfolio
from src.utils.display import display_analysis, display_round, display_final_verdict

console = Console()

DECISION_STYLE = {
    InvestmentDecision.BUY: ("BUY", "bold green"),
    InvestmentDecision.WAIT: ("WAIT", "yellow"),
    InvestmentDecision.PASS: ("PASS", "bold red"),
    InvestmentDecision.NEEDS_MORE_DATA: ("HOLD", "dim"),
}


def _example_portfolio() -> Portfolio:
    """예시 포트폴리오."""
    return Portfolio(
        name="예시 포트폴리오",
        investment_goal="장기 성장 + 배당 수익",
        risk_tolerance="moderate",
        investment_horizon="5년+",
        holdings=[
            Holding(
                ticker="005930",
                name="삼성전자",
                asset_class=AssetClass.STOCK_KR,
                quantity=100,
                avg_price=72000,
                current_price=68000,
                currency="KRW",
                sector="반도체",
                memo="AI 반도체 수요 증가 기대",
                financial_data=(
                    "2024 매출: 258조원 (+12% YoY)\n"
                    "영업이익: 32조원\n"
                    "PER: 12.5배 (5년 평균 14배)\n"
                    "PBR: 1.2배\n"
                    "ROE: 9.8%\n"
                    "부채비율: 35%\n"
                    "배당수익률: 2.1%"
                ),
            ),
            Holding(
                ticker="AAPL",
                name="Apple Inc.",
                asset_class=AssetClass.STOCK_US,
                quantity=10,
                avg_price=180,
                current_price=210,
                currency="USD",
                sector="테크",
                memo="생태계 락인 효과, 서비스 매출 성장",
            ),
            Holding(
                ticker="SCHD",
                name="Schwab US Dividend Equity ETF",
                asset_class=AssetClass.ETF,
                quantity=50,
                avg_price=75,
                current_price=82,
                currency="USD",
                sector="배당 ETF",
                memo="배당 성장 + 안정적 수익",
            ),
            Holding(
                ticker="373220",
                name="LG에너지솔루션",
                asset_class=AssetClass.STOCK_KR,
                quantity=5,
                avg_price=550000,
                current_price=380000,
                currency="KRW",
                sector="2차전지",
                memo="EV 시장 성장에 베팅",
            ),
        ],
    )


def _load_portfolio(path: str) -> Portfolio:
    """JSON 파일에서 포트폴리오 로드."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return Portfolio(**data)


def _chat_input(prompt_text: str = "you") -> str:
    """사용자로부터 자연어 입력을 받는다."""
    console.print()
    try:
        user_input = console.input(f"[bold yellow]{prompt_text} >[/] ")
    except (EOFError, KeyboardInterrupt):
        return ""
    return user_input.strip()


async def _run_interactive(
    portfolio: Portfolio,
    api_key: str | None,
    model: str,
) -> None:
    """인터랙티브 대화 모드: LLM과 대화하듯이 토론에 참여."""
    analyzer = PortfolioAnalyzer(api_key=api_key, model=model)
    orchestrator = analyzer.engine.debate_orchestrator

    for holding in portfolio.holdings:
        console.print()
        console.print(Panel(
            f"[bold]{holding.name}[/] ({holding.ticker}) 토론을 시작합니다.\n"
            f"매 라운드 후 대화하듯이 의견을 입력하세요.\n"
            f"[dim]엔터만 누르면 의견 없이 다음 라운드로 넘어갑니다.[/]",
            title="Debate Session",
            border_style="magenta",
        ))

        user_feedbacks: list[UserFeedback] = []

        # ── Round 1 ──
        console.print("\n[bold cyan]Round 1: 독립 분석 중...[/]")
        round1, opinions = await orchestrator.run_round1(holding, portfolio)
        display_round(round1)

        user_input = _chat_input()
        fb1 = UserFeedback(content=user_input) if user_input else None
        if fb1:
            user_feedbacks.append(fb1)

        # ── Round 2 ──
        console.print("\n[bold cyan]Round 2: 반론 진행 중...[/]")
        round2 = await orchestrator.run_round2(
            holding, portfolio, opinions, user_feedback=fb1,
        )
        rounds = [round1, round2]
        display_round(round2)

        user_input = _chat_input()
        fb2 = UserFeedback(content=user_input) if user_input else None
        if fb2:
            user_feedbacks.append(fb2)

        # ── Round 3 ──
        console.print("\n[bold cyan]Round 3: 최종 입장 정리 중...[/]")
        round3 = await orchestrator.run_round3(
            holding, portfolio, rounds, user_feedback=fb2,
        )
        rounds.append(round3)
        display_round(round3)

        user_input = _chat_input()
        fb3 = UserFeedback(content=user_input) if user_input else None
        if fb3:
            user_feedbacks.append(fb3)

        # ── Synthesis ──
        console.print("\n[bold cyan]투자위원회 최종 판정 중...[/]")
        synthesis = await orchestrator.synthesize(
            holding, rounds, user_feedbacks,
            user_final_comment=fb3.content if fb3 else "",
        )

        result = DebateResult(
            target_company=holding.name,
            target_ticker=holding.ticker,
            rounds=rounds,
            consensus_points=synthesis.get("consensus_points", []),
            dissent_points=synthesis.get("dissent_points", []),
            final_decision=orchestrator._parse_decision(
                synthesis.get("final_decision", "needs_more_data")
            ),
            final_confidence=min(
                max(synthesis.get("final_confidence", 0.5), 0.0), 1.0
            ),
            final_reasoning=synthesis.get("final_reasoning", ""),
            action_items=synthesis.get("action_items", []),
            risk_summary=synthesis.get("risk_summary", ""),
            price_assessment=synthesis.get("price_assessment", ""),
            user_feedbacks=user_feedbacks,
            phase=DebatePhase.SYNTHESIZED,
        )
        display_final_verdict(result)


async def _run_auto(
    portfolio: Portfolio,
    api_key: str | None,
    model: str,
    debate_only: bool,
) -> None:
    """자동 모드."""
    analyzer = PortfolioAnalyzer(api_key=api_key, model=model)
    if debate_only:
        result = await analyzer.debate(portfolio)
    else:
        result = await analyzer.analyze(portfolio)
    display_analysis(result)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DarkForest — Multi-Agent Debate 투자 분석기",
    )
    parser.add_argument("portfolio_file", nargs="?", help="포트폴리오 JSON 파일")
    parser.add_argument("--example", action="store_true", help="예시 포트폴리오")
    parser.add_argument("--auto", action="store_true", help="자동 모드 (대화 없이)")
    parser.add_argument("--debate-only", action="store_true", help="토론만 (프레임 생략)")
    parser.add_argument("--api-key", help="Anthropic API 키")
    parser.add_argument("--model", default="claude-sonnet-4-20250514")

    args = parser.parse_args()

    if args.example:
        portfolio = _example_portfolio()
    elif args.portfolio_file:
        portfolio = _load_portfolio(args.portfolio_file)
    else:
        parser.print_help()
        sys.exit(1)

    if args.auto:
        asyncio.run(_run_auto(portfolio, args.api_key, args.model, args.debate_only))
    else:
        asyncio.run(_run_interactive(portfolio, args.api_key, args.model))


if __name__ == "__main__":
    main()
