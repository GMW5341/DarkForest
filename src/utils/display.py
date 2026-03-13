"""분석 결과 출력 유틸리티."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.models.analysis import PortfolioAnalysis, Verdict

console = Console()

VERDICT_COLORS = {
    Verdict.STRONG_HOLD: "bold green",
    Verdict.HOLD: "green",
    Verdict.REBALANCE: "yellow",
    Verdict.REDUCE: "dark_orange",
    Verdict.EXIT: "bold red",
    Verdict.NEEDS_MORE_DATA: "dim",
}

VERDICT_LABELS = {
    Verdict.STRONG_HOLD: "유지 강화",
    Verdict.HOLD: "유지",
    Verdict.REBALANCE: "리밸런싱",
    Verdict.REDUCE: "비중 축소",
    Verdict.EXIT: "이탈 검토",
    Verdict.NEEDS_MORE_DATA: "판단 보류",
}


def display_analysis(result: PortfolioAnalysis) -> None:
    """분석 결과를 터미널에 출력."""
    # 종합 요약
    console.print()
    console.print(Panel(
        result.summary,
        title="포트폴리오 종합 분석",
        border_style="blue",
    ))

    # 논리 정합성 점수
    score = result.frame_integrity_score
    score_color = "green" if score >= 0.7 else "yellow" if score >= 0.5 else "red"
    console.print(
        f"\n분석 논리 정합성 점수: [{score_color}]{score:.0%}[/]"
    )

    # 종목별 분석
    for ha in result.holding_analyses:
        console.print()
        verdict_style = VERDICT_COLORS.get(ha.overall_verdict, "dim")
        verdict_label = VERDICT_LABELS.get(ha.overall_verdict, "?")

        console.print(Panel(
            f"[bold]{ha.name}[/] ({ha.ticker})\n"
            f"종합 판단: [{verdict_style}]{verdict_label}[/]\n"
            f"{ha.verdict_reasoning}",
            title=f"종목 분석: {ha.name}",
            border_style="cyan",
        ))

        # 프레임별 분석 테이블
        table = Table(title="프레임별 분석", show_lines=True)
        table.add_column("프레임", style="bold", width=20)
        table.add_column("결론", width=40)
        table.add_column("정합성", justify="center", width=8)

        for fa in ha.frame_analyses:
            conf_color = (
                "green" if fa.confidence >= 0.7
                else "yellow" if fa.confidence >= 0.5
                else "red"
            )
            table.add_row(
                fa.frame_name,
                fa.conclusion[:80] + ("..." if len(fa.conclusion) > 80 else ""),
                Text(f"{fa.confidence:.0%}", style=conf_color),
            )
        console.print(table)

        # 행동 제안
        if ha.action_items:
            console.print("\n[bold]구체적 행동 제안:[/]")
            for i, item in enumerate(ha.action_items, 1):
                console.print(f"  {i}. {item}")

    # 구조적 문제점
    if result.structural_issues:
        console.print()
        console.print(Panel(
            "\n".join(f"- {issue}" for issue in result.structural_issues),
            title="포트폴리오 구조적 문제점",
            border_style="red",
        ))

    # 개선 방안
    if result.improvement_plan:
        console.print()
        console.print(Panel(
            "\n".join(
                f"{i}. {plan}"
                for i, plan in enumerate(result.improvement_plan, 1)
            ),
            title="구체적 개선 방안",
            border_style="green",
        ))

    # 리스크 평가
    if result.risk_assessment:
        console.print()
        console.print(Panel(
            result.risk_assessment,
            title="리스크 평가",
            border_style="yellow",
        ))
