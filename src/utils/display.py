"""분석 결과 출력 유틸리티 — Multi-Agent Debate 포함."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.models.analysis import (
    DebateResult,
    InvestmentDecision,
    PortfolioAnalysis,
    Verdict,
)

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
    Verdict.STRONG_HOLD: "매수 (BUY)",
    Verdict.HOLD: "대기 (WAIT)",
    Verdict.REBALANCE: "리밸런싱",
    Verdict.REDUCE: "비중 축소",
    Verdict.EXIT: "패스 (PASS)",
    Verdict.NEEDS_MORE_DATA: "판단 보류",
}

DECISION_COLORS = {
    InvestmentDecision.BUY: "bold green",
    InvestmentDecision.WAIT: "yellow",
    InvestmentDecision.PASS: "bold red",
    InvestmentDecision.NEEDS_MORE_DATA: "dim",
}

DECISION_LABELS = {
    InvestmentDecision.BUY: "BUY",
    InvestmentDecision.WAIT: "WAIT",
    InvestmentDecision.PASS: "PASS",
    InvestmentDecision.NEEDS_MORE_DATA: "HOLD",
}


def display_round(rnd: "DebateRound") -> None:
    """단일 토론 라운드를 터미널에 출력."""
    from src.models.analysis import DebateRound

    round_labels = {
        "opening": "1R: 독립 분석",
        "cross_examination": "2R: 상호 반론",
        "final": "3R: 최종 입장",
    }

    label = round_labels.get(rnd.round_type, f"Round {rnd.round_number}")
    console.print(f"\n[bold cyan]{'='*50}[/]")
    console.print(f"[bold cyan]{label}[/]")
    console.print(f"[bold cyan]{'='*50}[/]")

    for msg in rnd.messages:
        stance_style = DECISION_COLORS.get(msg.stance, "dim")
        stance_label = DECISION_LABELS.get(msg.stance, "?")

        console.print(
            f"\n  [bold]{msg.agent_name}[/] → "
            f"[{stance_style}]{stance_label}[/] "
            f"(확신도: {msg.confidence:.0%})"
        )
        for line in msg.content.split("\n"):
            if line.strip():
                console.print(f"    {line.strip()}")

        if msg.agreements:
            console.print(f"    [green]핵심:[/] {', '.join(msg.agreements[:3])}")
        if msg.disagreements:
            console.print(f"    [red]반론:[/] {', '.join(msg.disagreements[:3])}")


def display_final_verdict(debate: DebateResult) -> None:
    """최종 판정만 출력 (토론 후)."""
    console.print()
    decision_style = DECISION_COLORS.get(debate.final_decision, "dim")
    decision_label = DECISION_LABELS.get(debate.final_decision, "?")

    console.print(Panel(
        f"[bold]{debate.target_company}[/] ({debate.target_ticker})\n\n"
        f"판정: [{decision_style}][bold]{decision_label}[/bold][/] "
        f"(확신도: {debate.final_confidence:.0%})\n\n"
        f"{debate.final_reasoning}",
        title="투자위원회 최종 판정",
        border_style=decision_style.replace("bold ", ""),
    ))

    if debate.consensus_points:
        console.print("\n[bold green]합의 포인트:[/]")
        for pt in debate.consensus_points:
            console.print(f"  + {pt}")

    if debate.dissent_points:
        console.print("\n[bold yellow]의견 불일치:[/]")
        for pt in debate.dissent_points:
            console.print(f"  ? {pt}")

    if debate.price_assessment:
        console.print(f"\n[bold]적정가 평가:[/] {debate.price_assessment}")
    if debate.risk_summary:
        console.print(f"[bold]핵심 리스크:[/] {debate.risk_summary}")

    if debate.action_items:
        console.print("\n[bold]행동 제안:[/]")
        for i, item in enumerate(debate.action_items, 1):
            console.print(f"  {i}. {item}")

    if debate.user_feedbacks:
        console.print(f"\n[dim]반영된 사용자 피드백: {len(debate.user_feedbacks)}건[/]")


def display_debate(debate: DebateResult) -> None:
    """토론 결과를 터미널에 출력."""
    console.print()
    console.print(Panel(
        f"[bold]{debate.target_company}[/] ({debate.target_ticker})",
        title="Multi-Agent Debate",
        border_style="magenta",
    ))

    # 라운드별 출력
    round_labels = {
        "opening": "1R: 독립 분석",
        "cross_examination": "2R: 상호 반론",
        "final": "3R: 최종 입장",
    }

    for rnd in debate.rounds:
        label = round_labels.get(rnd.round_type, f"Round {rnd.round_number}")
        console.print(f"\n[bold cyan]--- {label} ---[/]")

        for msg in rnd.messages:
            stance_style = DECISION_COLORS.get(msg.stance, "dim")
            stance_label = DECISION_LABELS.get(msg.stance, "?")

            console.print(
                f"\n  [bold]{msg.agent_name}[/] → "
                f"[{stance_style}]{stance_label}[/] "
                f"(확신도: {msg.confidence:.0%})"
            )
            # 내용은 들여쓰기로 출력
            for line in msg.content.split("\n"):
                if line.strip():
                    console.print(f"    {line.strip()}")

            if msg.agreements:
                console.print(f"    [green]동의:[/] {', '.join(msg.agreements[:3])}")
            if msg.disagreements:
                console.print(f"    [red]반론:[/] {', '.join(msg.disagreements[:3])}")

    # 최종 판정
    console.print()
    decision_style = DECISION_COLORS.get(debate.final_decision, "dim")
    decision_label = DECISION_LABELS.get(debate.final_decision, "?")

    console.print(Panel(
        f"판정: [{decision_style}][bold]{decision_label}[/bold][/] "
        f"(확신도: {debate.final_confidence:.0%})\n\n"
        f"{debate.final_reasoning}",
        title="투자위원회 최종 판정",
        border_style=decision_style.replace("bold ", ""),
    ))

    # 합의/불일치 포인트
    if debate.consensus_points:
        console.print("\n[bold green]합의 포인트:[/]")
        for pt in debate.consensus_points:
            console.print(f"  + {pt}")

    if debate.dissent_points:
        console.print("\n[bold yellow]의견 불일치:[/]")
        for pt in debate.dissent_points:
            console.print(f"  ? {pt}")

    # 적정가 & 리스크
    if debate.price_assessment:
        console.print(f"\n[bold]적정가 평가:[/] {debate.price_assessment}")
    if debate.risk_summary:
        console.print(f"[bold]핵심 리스크:[/] {debate.risk_summary}")

    # 행동 제안
    if debate.action_items:
        console.print("\n[bold]행동 제안:[/]")
        for i, item in enumerate(debate.action_items, 1):
            console.print(f"  {i}. {item}")


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

        # 토론 결과가 있으면 토론 내용 표시
        if ha.debate_result:
            display_debate(ha.debate_result)

        # 프레임별 분석 테이블 (있는 경우)
        if ha.frame_analyses:
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
