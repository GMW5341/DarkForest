"""
CLI Entry Point — 인터랙티브 커맨드라인 인터페이스.

포트폴리오 JSON 파일을 입력받아 Multi-Agent Debate 기반 분석을 실행한다.
각 라운드 사이에 사용자가 피드백을 제공하여 토론 방향을 조율할 수 있다.

Usage:
    darkforest portfolio.json                    # 인터랙티브 토론
    darkforest portfolio.json --auto             # 자동 모드 (피드백 없이)
    darkforest --example                         # 예시 포트폴리오
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from src.analyzer import PortfolioAnalyzer
from src.agents.debate import DebateOrchestrator
from src.models.analysis import (
    AgentOpinion,
    DebatePhase,
    DebateResult,
    DebateRound,
    InvestmentDecision,
    UserFeedback,
)
from src.models.portfolio import AssetClass, Holding, Portfolio
from src.utils.display import display_analysis, display_round, display_final_verdict

console = Console()


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


def _collect_feedback(round_name: str) -> UserFeedback | None:
    """사용자로부터 피드백을 수집."""
    console.print()
    console.print(Panel(
        f"[bold]{round_name}[/] 결과를 확인하셨습니다.\n"
        f"피드백을 입력하면 다음 라운드에 반영됩니다.\n"
        f"[dim]엔터만 누르면 피드백 없이 진행합니다.[/]",
        border_style="yellow",
    ))

    content = Prompt.ask(
        "\n[bold yellow]피드백[/] (의견/지시사항)",
        default="",
    )

    if not content.strip():
        return None

    focus = Prompt.ask(
        "[yellow]집중할 포인트[/] (쉼표로 구분, 선택)",
        default="",
    )

    additional = Prompt.ask(
        "[yellow]추가 정보[/] (추가 데이터/맥락, 선택)",
        default="",
    )

    override = Prompt.ask(
        "[yellow]선호 방향[/] (buy/wait/pass, 선택)",
        default="",
    )

    return UserFeedback(
        content=content.strip(),
        focus_on=[f.strip() for f in focus.split(",") if f.strip()] if focus else [],
        additional_context=additional.strip(),
        override_stance=override.strip().lower() if override.strip() else None,
    )


async def _run_interactive(
    portfolio: Portfolio,
    api_key: str | None,
    model: str,
) -> None:
    """인터랙티브 모드: 라운드별 피드백 수집."""
    analyzer = PortfolioAnalyzer(api_key=api_key, model=model)
    orchestrator = analyzer.engine.debate_orchestrator

    for holding in portfolio.holdings:
        console.print()
        console.print(Panel(
            f"[bold]{holding.name}[/] ({holding.ticker}) 분석 시작",
            title="종목 분석",
            border_style="magenta",
        ))

        user_feedbacks: list[UserFeedback] = []

        # ── Round 1: 독립 분석 ──
        console.print("\n[bold cyan]Round 1: 독립 분석 진행 중...[/]")
        round1, opinions = await orchestrator.run_round1(holding, portfolio)
        display_round(round1)

        fb1 = _collect_feedback("Round 1 (독립 분석)")
        if fb1:
            user_feedbacks.append(fb1)

        # ── Round 2: 상호 반론 ──
        console.print("\n[bold cyan]Round 2: 상호 반론 진행 중...[/]")
        round2 = await orchestrator.run_round2(
            holding, portfolio, opinions,
            user_feedback=fb1,
        )
        rounds = [round1, round2]
        display_round(round2)

        fb2 = _collect_feedback("Round 2 (상호 반론)")
        if fb2:
            user_feedbacks.append(fb2)

        # ── Round 3: 최종 입장 ──
        console.print("\n[bold cyan]Round 3: 최종 입장 정리 중...[/]")
        round3 = await orchestrator.run_round3(
            holding, portfolio, rounds,
            user_feedback=fb2,
        )
        rounds.append(round3)
        display_round(round3)

        fb3 = _collect_feedback("Round 3 (최종 입장)")
        if fb3:
            user_feedbacks.append(fb3)

        # ── Synthesis: 최종 판정 ──
        final_comment = ""
        if fb3:
            final_comment = fb3.content

        console.print("\n[bold cyan]투자위원회 최종 판정 중...[/]")
        synthesis = await orchestrator.synthesize(
            holding, rounds, user_feedbacks, final_comment,
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
    """자동 모드: 피드백 없이 실행."""
    analyzer = PortfolioAnalyzer(api_key=api_key, model=model)
    if debate_only:
        result = await analyzer.debate(portfolio)
    else:
        result = await analyzer.analyze(portfolio)
    display_analysis(result)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DarkForest — Multi-Agent Debate 투자 포트폴리오 분석기",
    )
    parser.add_argument(
        "portfolio_file",
        nargs="?",
        help="포트폴리오 JSON 파일 경로",
    )
    parser.add_argument(
        "--example",
        action="store_true",
        help="예시 포트폴리오로 분석 실행",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="자동 모드 (라운드 간 피드백 없이 실행)",
    )
    parser.add_argument(
        "--debate-only",
        action="store_true",
        help="토론만 실행 (Question Frame 분석 생략)",
    )
    parser.add_argument(
        "--api-key",
        help="Anthropic API 키 (미지정 시 ANTHROPIC_API_KEY 환경변수 사용)",
    )
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-20250514",
        help="사용할 Claude 모델 (기본: claude-sonnet-4-20250514)",
    )

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
