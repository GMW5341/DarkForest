"""
Scenario Engine — 시나리오 분석 + 베이지안 업데이트 + 스트레스 테스트.

거시 경제 이슈에 대해:
  1. 시나리오 분석: 3~4개 시나리오별 확률, 영향, 대응 전략
  2. 베이지안 업데이트: 새 정보로 시나리오 확률 갱신
  3. 스트레스 테스트: 자산군별 충격 시뮬레이션
  4. 자산 배분 권고: 시나리오별 최적 자산 배분 제안

MCDA와의 차이:
  - MCDA: 기준 간 독립 가정, 선형 합산 → 개별 종목 비교에 적합
  - 시나리오: 비선형, 연쇄 효과, 꼬리 리스크 포착 → 거시/시스템 이슈에 적합
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


# ── 자산군 정의 ──

ASSET_CLASSES = [
    "stock_kr",      # 한국 주식
    "stock_us",      # 미국 주식
    "bond_gov",      # 국채
    "bond_corp",     # 회사채
    "etf",           # ETF
    "fund",          # 펀드 (사모/공모)
    "commodity",     # 원자재 (금, 원유 등)
    "real_estate",   # 부동산/리츠
    "crypto",        # 암호화폐
    "cash",          # 현금/MMF
]

ASSET_LABELS = {
    "stock_kr": "한국 주식",
    "stock_us": "미국 주식",
    "bond_gov": "국채",
    "bond_corp": "회사채",
    "etf": "ETF",
    "fund": "펀드",
    "commodity": "원자재",
    "real_estate": "부동산/리츠",
    "crypto": "암호화폐",
    "cash": "현금/MMF",
}


# ── 데이터 구조 ──

@dataclass
class Scenario:
    """단일 시나리오."""
    name: str                              # 시나리오 이름
    description: str                       # 상세 설명
    probability: float                     # 발생 확률 (0~1)
    severity: str = "중간"                  # 심각도: 경미/중간/심각/극단
    asset_impacts: dict[str, float] = field(default_factory=dict)
    # 자산군별 예상 수익률 영향 (%, 예: {"stock_kr": -15, "bond_gov": +5, "cash": 0})
    recommended_allocation: dict[str, float] = field(default_factory=dict)
    # 이 시나리오 하에서의 권장 자산 배분 비중 (%, 합 100)
    key_signals: list[str] = field(default_factory=list)
    # 이 시나리오가 현실화되고 있다는 신호
    hedges: list[str] = field(default_factory=list)
    # 방어/헤지 방법

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "probability": round(self.probability, 3),
            "severity": self.severity,
            "asset_impacts": {k: round(v, 1) for k, v in self.asset_impacts.items()},
            "recommended_allocation": {k: round(v, 1) for k, v in self.recommended_allocation.items()},
            "key_signals": self.key_signals,
            "hedges": self.hedges,
        }


@dataclass
class BayesianUpdate:
    """베이지안 업데이트 기록."""
    evidence: str                          # 새로운 정보/뉴스
    prior: dict[str, float] = field(default_factory=dict)    # 이전 확률
    likelihood: dict[str, float] = field(default_factory=dict)  # 우도
    posterior: dict[str, float] = field(default_factory=dict)  # 사후 확률

    def to_dict(self) -> dict:
        return {
            "evidence": self.evidence,
            "prior": {k: round(v, 3) for k, v in self.prior.items()},
            "posterior": {k: round(v, 3) for k, v in self.posterior.items()},
        }


@dataclass
class StressTestResult:
    """스트레스 테스트 결과."""
    scenario_name: str
    portfolio_impact_pct: float            # 포트폴리오 전체 수익률 영향 (%)
    asset_impacts: dict[str, float] = field(default_factory=dict)
    # 자산군별 수익률 영향
    worst_asset: str = ""
    best_asset: str = ""
    max_drawdown_pct: float = 0.0          # 최대 낙폭

    def to_dict(self) -> dict:
        return {
            "scenario_name": self.scenario_name,
            "portfolio_impact_pct": round(self.portfolio_impact_pct, 2),
            "asset_impacts": {ASSET_LABELS.get(k, k): round(v, 1) for k, v in self.asset_impacts.items()},
            "worst_asset": ASSET_LABELS.get(self.worst_asset, self.worst_asset),
            "best_asset": ASSET_LABELS.get(self.best_asset, self.best_asset),
            "max_drawdown_pct": round(self.max_drawdown_pct, 2),
        }


@dataclass
class ScenarioAnalysisResult:
    """시나리오 분석 전체 결과."""
    scenarios: list[Scenario] = field(default_factory=list)
    bayesian_updates: list[BayesianUpdate] = field(default_factory=list)
    stress_tests: list[StressTestResult] = field(default_factory=list)
    weighted_allocation: dict[str, float] = field(default_factory=dict)
    # 확률 가중 최적 자산 배분
    expected_portfolio_return: float = 0.0
    # 확률 가중 기대 수익률

    def to_dict(self) -> dict:
        return {
            "scenarios": [s.to_dict() for s in self.scenarios],
            "bayesian_updates": [b.to_dict() for b in self.bayesian_updates],
            "stress_tests": [st.to_dict() for st in self.stress_tests],
            "weighted_allocation": {
                ASSET_LABELS.get(k, k): round(v, 1)
                for k, v in self.weighted_allocation.items()
            },
            "expected_portfolio_return": round(self.expected_portfolio_return, 2),
        }


# ── 시나리오 분석 ──

def parse_scenarios_from_dict(data: dict) -> list[Scenario]:
    """LLM 응답(dict)에서 시나리오 리스트 파싱."""
    scenarios = []
    for s in data.get("scenarios", []):
        if not isinstance(s, dict):
            continue
        scenarios.append(Scenario(
            name=s.get("name", ""),
            description=s.get("description", ""),
            probability=min(max(s.get("probability", 0.25), 0.0), 1.0),
            severity=s.get("severity", "중간"),
            asset_impacts=s.get("asset_impacts", {}),
            recommended_allocation=s.get("recommended_allocation", {}),
            key_signals=s.get("key_signals", []),
            hedges=s.get("hedges", []),
        ))

    # 확률 합이 1이 되도록 정규화
    if scenarios:
        total = sum(s.probability for s in scenarios)
        if total > 0:
            for s in scenarios:
                s.probability /= total

    return scenarios


# ── 베이지안 업데이트 ──

def bayesian_update(
    scenarios: list[Scenario],
    evidence: str,
    likelihoods: dict[str, float],
) -> BayesianUpdate:
    """
    베이지안 업데이트로 시나리오 확률 갱신.

    P(scenario|evidence) ∝ P(evidence|scenario) × P(scenario)

    Args:
        scenarios: 현재 시나리오 리스트
        evidence: 새로운 정보 (텍스트)
        likelihoods: {시나리오 이름: P(evidence|scenario)} (0~1)
            예: {"낙관": 0.2, "기본": 0.5, "비관": 0.9}
            → 이 증거가 비관 시나리오에서 나타날 확률이 90%라는 의미

    Returns:
        BayesianUpdate with updated scenario probabilities
    """
    priors = {s.name: s.probability for s in scenarios}

    # P(evidence|scenario) × P(scenario) 계산
    numerators = {}
    for s in scenarios:
        lk = likelihoods.get(s.name, 0.5)  # 기본 우도 0.5
        numerators[s.name] = lk * s.probability

    # 정규화 (합이 1이 되도록)
    total = sum(numerators.values())
    if total == 0:
        total = 1.0

    posteriors = {name: val / total for name, val in numerators.items()}

    # 시나리오 확률 업데이트
    for s in scenarios:
        s.probability = posteriors.get(s.name, s.probability)

    return BayesianUpdate(
        evidence=evidence,
        prior=priors,
        likelihood=likelihoods,
        posterior=posteriors,
    )


# ── 스트레스 테스트 ──

def stress_test(
    scenario: Scenario,
    portfolio_weights: dict[str, float] | None = None,
) -> StressTestResult:
    """
    특정 시나리오에서의 포트폴리오 스트레스 테스트.

    Args:
        scenario: 적용할 시나리오
        portfolio_weights: 현재 자산 배분 비중 (%, 합 100)
            없으면 기본 배분 사용

    Returns:
        StressTestResult with portfolio-level impact
    """
    # 기본 포트폴리오: 균형형
    if not portfolio_weights:
        portfolio_weights = {
            "stock_kr": 30,
            "stock_us": 20,
            "bond_gov": 15,
            "bond_corp": 10,
            "etf": 10,
            "commodity": 5,
            "cash": 10,
        }

    impacts = scenario.asset_impacts
    total_impact = 0.0
    asset_level = {}
    worst = ("", float("inf"))
    best = ("", float("-inf"))

    for asset, weight in portfolio_weights.items():
        impact = impacts.get(asset, 0.0)
        contribution = (weight / 100) * impact
        total_impact += contribution
        asset_level[asset] = impact

        if impact < worst[1]:
            worst = (asset, impact)
        if impact > best[1]:
            best = (asset, impact)

    # 최대 낙폭 (비관적 추정: 충격의 1.5배)
    max_dd = min(total_impact * 1.5, 0) if total_impact < 0 else 0

    return StressTestResult(
        scenario_name=scenario.name,
        portfolio_impact_pct=total_impact,
        asset_impacts=asset_level,
        worst_asset=worst[0],
        best_asset=best[0],
        max_drawdown_pct=max_dd,
    )


# ── 확률 가중 최적 자산 배분 ──

def weighted_optimal_allocation(scenarios: list[Scenario]) -> dict[str, float]:
    """
    시나리오 확률 가중 최적 자산 배분 계산.

    각 시나리오의 권장 배분을 확률로 가중 평균하여
    "모든 시나리오를 고려한" 최적 배분을 산출.
    """
    allocation: dict[str, float] = {}

    for scenario in scenarios:
        for asset, weight in scenario.recommended_allocation.items():
            allocation[asset] = allocation.get(asset, 0) + weight * scenario.probability

    # 합이 100이 되도록 정규화
    total = sum(allocation.values())
    if total > 0:
        allocation = {k: (v / total) * 100 for k, v in allocation.items()}

    return allocation


def expected_return(scenarios: list[Scenario], portfolio_weights: dict[str, float] | None = None) -> float:
    """확률 가중 기대 수익률."""
    if not portfolio_weights:
        portfolio_weights = weighted_optimal_allocation(scenarios)

    total = 0.0
    for scenario in scenarios:
        scenario_return = 0.0
        for asset, weight in portfolio_weights.items():
            impact = scenario.asset_impacts.get(asset, 0.0)
            scenario_return += (weight / 100) * impact
        total += scenario.probability * scenario_return

    return total


# ── 전체 분석 실행 ──

def run_scenario_analysis(
    scenarios_data: dict,
    portfolio_weights: dict[str, float] | None = None,
    evidence_updates: list[dict] | None = None,
) -> ScenarioAnalysisResult:
    """
    시나리오 분석 전체 파이프라인.

    1. 시나리오 파싱
    2. 베이지안 업데이트 (있으면)
    3. 각 시나리오별 스트레스 테스트
    4. 확률 가중 최적 배분 계산
    """
    scenarios = parse_scenarios_from_dict(scenarios_data)
    if not scenarios:
        return ScenarioAnalysisResult()

    # 베이지안 업데이트
    bayesian_updates = []
    if evidence_updates:
        for update in evidence_updates:
            evidence = update.get("evidence", "")
            likelihoods = update.get("likelihoods", {})
            if evidence and likelihoods:
                bu = bayesian_update(scenarios, evidence, likelihoods)
                bayesian_updates.append(bu)

    # 스트레스 테스트
    stress_tests = [stress_test(s, portfolio_weights) for s in scenarios]

    # 최적 배분
    allocation = weighted_optimal_allocation(scenarios)
    exp_ret = expected_return(scenarios, portfolio_weights)

    return ScenarioAnalysisResult(
        scenarios=scenarios,
        bayesian_updates=bayesian_updates,
        stress_tests=stress_tests,
        weighted_allocation=allocation,
        expected_portfolio_return=exp_ret,
    )


# ── 프롬프트 생성 ──

SCENARIO_SYSTEM_PROMPT = """\
당신은 시나리오 분석 전문가입니다. 거시 경제 이슈에 대해
3~4개의 구체적인 시나리오를 수립하고, 각 시나리오별로
자산군 영향과 최적 배분을 제시합니다.

## 응답 형식 (반드시 JSON)
{
    "scenarios": [
        {
            "name": "시나리오 이름 (예: 낙관, 기본, 비관, 극단)",
            "description": "시나리오 상세 설명",
            "probability": 0.0~1.0,
            "severity": "경미" | "중간" | "심각" | "극단",
            "asset_impacts": {
                "stock_kr": -10.0,
                "stock_us": -5.0,
                "bond_gov": 3.0,
                "bond_corp": -2.0,
                "etf": -7.0,
                "fund": -8.0,
                "commodity": 5.0,
                "real_estate": -3.0,
                "crypto": -15.0,
                "cash": 0.5
            },
            "recommended_allocation": {
                "stock_kr": 20,
                "stock_us": 15,
                "bond_gov": 25,
                "bond_corp": 5,
                "commodity": 10,
                "cash": 25
            },
            "key_signals": ["이 시나리오가 현실화되는 신호1", ...],
            "hedges": ["방어/헤지 방법1", ...]
        }
    ]
}

## 주의사항
- asset_impacts: 각 자산군의 예상 수익률 변동 (%, 양수=상승, 음수=하락)
- recommended_allocation: 해당 시나리오 하 권장 비중 (%, 합계 약 100)
- 시나리오 확률의 합이 1.0이 되도록 하세요
- 반드시 3~4개 시나리오를 제시하세요
- 극단 시나리오(tail risk)를 최소 1개 포함하세요
- 자산군: stock_kr, stock_us, bond_gov, bond_corp, etf, fund, commodity, real_estate, crypto, cash"""


BAYESIAN_UPDATE_PROMPT = """\
다음 새로운 정보가 나왔습니다. 각 시나리오에서 이 정보가 나타날 확률(우도)을 추정하세요.

## 새로운 정보
{evidence}

## 현재 시나리오들
{scenarios_text}

## 응답 형식 (반드시 JSON)
{{
    "likelihoods": {{
        "시나리오 이름1": 0.0~1.0,
        "시나리오 이름2": 0.0~1.0,
        ...
    }},
    "reasoning": "우도 추정 근거"
}}

P(evidence|scenario)를 추정하세요.
이 증거가 해당 시나리오가 맞을 때 관찰될 확률입니다."""


def build_scenario_prompt(topic: str, context: str, debate_summary: str) -> str:
    """시나리오 분석 프롬프트 생성."""
    prompt = f"## 분석 주제\n{topic}\n\n"
    if context:
        prompt += f"## 배경 정보\n{context}\n\n"
    if debate_summary:
        prompt += f"## 전문가 토론 요약\n{debate_summary}\n\n"
    prompt += (
        "위 주제와 토론 내용을 바탕으로 시나리오 분석을 수행하세요.\n"
        "각 시나리오에 대해 모든 자산군(주식, 채권, 원자재, 코인 등)의\n"
        "예상 영향과 최적 배분을 제시하세요.\n"
        "반드시 JSON 형식으로 응답하세요."
    )
    return prompt


def build_bayesian_prompt(evidence: str, scenarios: list[Scenario]) -> str:
    """베이지안 업데이트 프롬프트 생성."""
    scenarios_text = "\n".join(
        f"- {s.name} (현재 확률: {s.probability:.1%}): {s.description}"
        for s in scenarios
    )
    return BAYESIAN_UPDATE_PROMPT.format(
        evidence=evidence,
        scenarios_text=scenarios_text,
    )


def format_scenario_for_synthesis(result: ScenarioAnalysisResult) -> str:
    """시나리오 분석 결과를 합성 프롬프트에 삽입할 텍스트로 포맷."""
    lines = ["\n## 시나리오 분석 결과"]

    for s in result.scenarios:
        severity_emoji = {"경미": "▸", "중간": "▸▸", "심각": "▸▸▸", "극단": "▸▸▸▸"}.get(s.severity, "▸")
        lines.append(
            f"\n### {s.name} (확률: {s.probability:.0%}, 심각도: {severity_emoji} {s.severity})"
        )
        lines.append(s.description)

        if s.asset_impacts:
            lines.append("\n자산군별 예상 영향:")
            for asset, impact in sorted(s.asset_impacts.items(), key=lambda x: x[1]):
                label = ASSET_LABELS.get(asset, asset)
                sign = "+" if impact >= 0 else ""
                lines.append(f"  {label}: {sign}{impact:.1f}%")

        if s.key_signals:
            lines.append("\n주시 신호: " + " / ".join(s.key_signals))

    # 스트레스 테스트
    if result.stress_tests:
        lines.append("\n## 스트레스 테스트 결과")
        for st in result.stress_tests:
            lines.append(
                f"  {st.scenario_name}: 포트폴리오 영향 {st.portfolio_impact_pct:+.1f}%"
                f" (최대 낙폭 {st.max_drawdown_pct:.1f}%)"
            )

    # 최적 배분
    if result.weighted_allocation:
        lines.append("\n## 확률 가중 최적 자산 배분")
        for asset, weight in sorted(result.weighted_allocation.items(), key=lambda x: -x[1]):
            if weight >= 1:
                label = ASSET_LABELS.get(asset, asset)
                bar = "█" * int(weight / 5) + "░" * max(0, 20 - int(weight / 5))
                lines.append(f"  {label:10s} [{bar}] {weight:.0f}%")

    lines.append(
        f"\n확률 가중 기대 수익률: {result.expected_portfolio_return:+.1f}%"
    )

    # 베이지안 업데이트
    if result.bayesian_updates:
        lines.append("\n## 베이지안 확률 갱신 이력")
        for bu in result.bayesian_updates:
            lines.append(f"\n새 정보: \"{bu.evidence}\"")
            for name in bu.posterior:
                prior = bu.prior.get(name, 0)
                post = bu.posterior.get(name, 0)
                arrow = "↑" if post > prior else "↓" if post < prior else "→"
                lines.append(f"  {name}: {prior:.0%} {arrow} {post:.0%}")

    return "\n".join(lines)
