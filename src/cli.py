"""
CLI Entry Point — 커맨드라인 인터페이스.

포트폴리오 JSON 파일을 입력받아 분석을 실행한다.

Usage:
    python -m src.cli portfolio.json
    python -m src.cli --example  # 예시 포트폴리오로 실행
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from src.analyzer import PortfolioAnalyzer
from src.models.portfolio import AssetClass, Holding, Portfolio
from src.utils.display import display_analysis


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


async def _run(portfolio: Portfolio, api_key: str | None, model: str) -> None:
    analyzer = PortfolioAnalyzer(api_key=api_key, model=model)
    result = await analyzer.analyze(portfolio)
    display_analysis(result)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DarkForest — 투자 포트폴리오 분석기",
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

    asyncio.run(_run(portfolio, args.api_key, args.model))


if __name__ == "__main__":
    main()
