"""
MCDA Engine — Multi-Criteria Decision Analysis.

에이전트 토론 결과를 정량적 점수로 변환하여
수학적으로 투자 판정을 도출하는 엔진.

Claude는 "느낌"으로 판정하지 않는다.
에이전트들의 정성 분석 → 정량 점수 → 수학적 합산 → 판정.
Claude는 그 결과를 설명하는 역할만 한다.

지원 방법:
  - Weighted Sum: 가중치 × 점수 단순 합산
  - TOPSIS: 이상적 해(ideal solution)와의 거리 기반 순위
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

from src.models.analysis import AgentOpinion, InvestmentDecision


# ── 평가 기준 ──

class Criterion(str, Enum):
    """투자 평가 기준."""
    BUSINESS_QUALITY = "business_quality"    # 사업 품질 (해자, 경쟁력)
    FINANCIAL_HEALTH = "financial_health"    # 재무 건전성
    VALUATION = "valuation"                  # 밸류에이션 매력도
    GROWTH = "growth"                        # 성장 잠재력
    RISK = "risk"                            # 리스크 수준 (0=위험, 1=안전)
    CATALYST = "catalyst"                    # 촉매/모멘텀
    PORTFOLIO_FIT = "portfolio_fit"          # 포트폴리오 적합성


# 기본 가중치
DEFAULT_WEIGHTS: dict[Criterion, float] = {
    Criterion.BUSINESS_QUALITY: 0.20,
    Criterion.FINANCIAL_HEALTH: 0.15,
    Criterion.VALUATION: 0.20,
    Criterion.GROWTH: 0.15,
    Criterion.RISK: 0.15,
    Criterion.CATALYST: 0.05,
    Criterion.PORTFOLIO_FIT: 0.10,
}


@dataclass
class CriteriaScores:
    """종목의 기준별 점수 (0.0 ~ 1.0)."""
    business_quality: float = 0.5
    financial_health: float = 0.5
    valuation: float = 0.5
    growth: float = 0.5
    risk: float = 0.5   # 0=매우 위험, 1=매우 안전
    catalyst: float = 0.5
    portfolio_fit: float = 0.5

    def to_dict(self) -> dict[Criterion, float]:
        return {
            Criterion.BUSINESS_QUALITY: self.business_quality,
            Criterion.FINANCIAL_HEALTH: self.financial_health,
            Criterion.VALUATION: self.valuation,
            Criterion.GROWTH: self.growth,
            Criterion.RISK: self.risk,
            Criterion.CATALYST: self.catalyst,
            Criterion.PORTFOLIO_FIT: self.portfolio_fit,
        }

    def as_list(self) -> list[float]:
        return [
            self.business_quality,
            self.financial_health,
            self.valuation,
            self.growth,
            self.risk,
            self.catalyst,
            self.portfolio_fit,
        ]


@dataclass
class MCDAResult:
    """MCDA 분석 결과."""
    criteria_scores: CriteriaScores
    weighted_score: float              # 가중 합산 점수 (0~1)
    decision: InvestmentDecision       # 점수 기반 판정
    method: str = "weighted_sum"       # 사용된 방법
    score_breakdown: dict[str, float] = field(default_factory=dict)
    # TOPSIS용 (여러 종목 비교 시)
    topsis_rank: int | None = None
    topsis_closeness: float | None = None

    def to_dict(self) -> dict:
        d = {
            "method": self.method,
            "weighted_score": round(self.weighted_score, 4),
            "decision": self.decision.value,
            "criteria_scores": {
                k.value: round(v, 3)
                for k, v in self.criteria_scores.to_dict().items()
            },
            "score_breakdown": {
                k: round(v, 4) for k, v in self.score_breakdown.items()
            },
        }
        if self.topsis_rank is not None:
            d["topsis_rank"] = self.topsis_rank
            d["topsis_closeness"] = round(self.topsis_closeness, 4)
        return d


# ── 점수 추출 ──

# 에이전트 입장(stance) → 기본 점수 매핑
_STANCE_SCORE = {
    InvestmentDecision.BUY: 0.85,
    InvestmentDecision.WAIT: 0.55,
    InvestmentDecision.PASS: 0.20,
    InvestmentDecision.NEEDS_MORE_DATA: 0.40,
}

# 에이전트 역할 → 담당 기준 매핑
_AGENT_CRITERIA_MAP = {
    "펀더멘털 애널리스트": [
        Criterion.BUSINESS_QUALITY,
        Criterion.GROWTH,
    ],
    "밸류에이션 애널리스트": [
        Criterion.VALUATION,
        Criterion.CATALYST,
    ],
    "리스크 애널리스트": [
        Criterion.RISK,
        Criterion.FINANCIAL_HEALTH,
    ],
}


def extract_scores(opinions: list[AgentOpinion]) -> CriteriaScores:
    """
    에이전트 의견들에서 기준별 점수를 추출.

    각 에이전트의 stance와 confidence를 결합하여 담당 기준의 점수를 산출.
    점수 = base_score × (0.5 + 0.5 × confidence)

    red_flags 개수가 많으면 감점, key_points가 많으면 가점.
    """
    scores: dict[Criterion, list[float]] = {c: [] for c in Criterion}

    for opinion in opinions:
        base = _STANCE_SCORE.get(opinion.stance, 0.5)
        # confidence로 조정: confidence 1.0이면 그대로, 0.0이면 반감
        adjusted = base * (0.5 + 0.5 * opinion.confidence)

        # red_flags 감점 (개당 -0.03, 최대 -0.15)
        red_penalty = min(len(opinion.red_flags) * 0.03, 0.15)
        adjusted -= red_penalty

        # key_points 가점 (개당 +0.02, 최대 +0.10)
        key_bonus = min(len(opinion.key_points) * 0.02, 0.10)
        adjusted += key_bonus

        # 0~1 범위 클램프
        adjusted = max(0.0, min(1.0, adjusted))

        # 담당 기준에 할당
        mapped_criteria = _AGENT_CRITERIA_MAP.get(opinion.agent_name, [])
        for criterion in mapped_criteria:
            scores[criterion].append(adjusted)

        # 모든 에이전트가 portfolio_fit에 기여
        scores[Criterion.PORTFOLIO_FIT].append(adjusted)

    # 각 기준별 평균 (데이터가 없으면 0.5)
    return CriteriaScores(
        business_quality=_avg(scores[Criterion.BUSINESS_QUALITY]),
        financial_health=_avg(scores[Criterion.FINANCIAL_HEALTH]),
        valuation=_avg(scores[Criterion.VALUATION]),
        growth=_avg(scores[Criterion.GROWTH]),
        risk=_avg(scores[Criterion.RISK]),
        catalyst=_avg(scores[Criterion.CATALYST]),
        portfolio_fit=_avg(scores[Criterion.PORTFOLIO_FIT]),
    )


def _avg(values: list[float], default: float = 0.5) -> float:
    return sum(values) / len(values) if values else default


# ── Weighted Sum ──

def weighted_sum(
    scores: CriteriaScores,
    weights: dict[Criterion, float] | None = None,
) -> MCDAResult:
    """가중 합산법: Σ(weight_i × score_i)."""
    w = weights or DEFAULT_WEIGHTS

    # 가중치 정규화 (합이 1이 되도록)
    total_w = sum(w.values())
    norm_w = {k: v / total_w for k, v in w.items()}

    score_map = scores.to_dict()
    breakdown = {}
    total = 0.0

    for criterion in Criterion:
        s = score_map.get(criterion, 0.5)
        cw = norm_w.get(criterion, 0.0)
        contribution = s * cw
        breakdown[criterion.value] = contribution
        total += contribution

    return MCDAResult(
        criteria_scores=scores,
        weighted_score=total,
        decision=_score_to_decision(total),
        method="weighted_sum",
        score_breakdown=breakdown,
    )


# ── TOPSIS ──

def topsis(
    alternatives: list[tuple[str, CriteriaScores]],
    weights: dict[Criterion, float] | None = None,
) -> list[tuple[str, MCDAResult]]:
    """
    TOPSIS (Technique for Order Preference by Similarity to Ideal Solution).

    여러 종목을 비교하여 순위를 매긴다.
    이상적 해(ideal)에 가깝고 최악의 해(anti-ideal)에서 먼 대안이 높은 순위.

    Args:
        alternatives: [(ticker, CriteriaScores), ...]
        weights: 기준별 가중치

    Returns:
        [(ticker, MCDAResult), ...] 순위순 정렬
    """
    if not alternatives:
        return []

    w = weights or DEFAULT_WEIGHTS
    total_w = sum(w.values())
    norm_w = {k: v / total_w for k, v in w.items()}
    criteria_list = list(Criterion)
    n_criteria = len(criteria_list)

    # Step 1: 결정 행렬 구성
    matrix = []
    for _, scores in alternatives:
        matrix.append(scores.as_list())

    n_alt = len(matrix)

    # Step 2: 정규화 (벡터 정규화)
    norm_matrix = []
    for j in range(n_criteria):
        col = [matrix[i][j] for i in range(n_alt)]
        denom = math.sqrt(sum(x * x for x in col))
        if denom == 0:
            denom = 1.0
        norm_matrix.append([x / denom for x in col])

    # Step 3: 가중 정규화 행렬
    weighted_matrix = []
    for i in range(n_alt):
        row = []
        for j in range(n_criteria):
            cw = norm_w.get(criteria_list[j], 0.0)
            row.append(norm_matrix[j][i] * cw)
        weighted_matrix.append(row)

    # Step 4: 이상적 해 & 최악의 해
    ideal = [max(weighted_matrix[i][j] for i in range(n_alt)) for j in range(n_criteria)]
    anti_ideal = [min(weighted_matrix[i][j] for i in range(n_alt)) for j in range(n_criteria)]

    # Step 5: 거리 계산
    results = []
    for i in range(n_alt):
        d_ideal = math.sqrt(sum((weighted_matrix[i][j] - ideal[j]) ** 2 for j in range(n_criteria)))
        d_anti = math.sqrt(sum((weighted_matrix[i][j] - anti_ideal[j]) ** 2 for j in range(n_criteria)))

        closeness = d_anti / (d_ideal + d_anti) if (d_ideal + d_anti) > 0 else 0.5

        ticker, scores = alternatives[i]
        ws_result = weighted_sum(scores, weights)
        ws_result.method = "topsis"
        ws_result.topsis_closeness = closeness
        results.append((ticker, ws_result))

    # Step 6: 순위 매기기 (closeness 내림차순)
    results.sort(key=lambda x: x[1].topsis_closeness, reverse=True)
    for rank, (ticker, result) in enumerate(results, 1):
        result.topsis_rank = rank

    return results


# ── 판정 변환 ──

def _score_to_decision(score: float) -> InvestmentDecision:
    """정량 점수를 투자 판정으로 변환."""
    if score >= 0.70:
        return InvestmentDecision.BUY
    elif score >= 0.50:
        return InvestmentDecision.WAIT
    elif score >= 0.30:
        return InvestmentDecision.PASS
    else:
        return InvestmentDecision.PASS


def format_mcda_for_prompt(result: MCDAResult) -> str:
    """MCDA 결과를 합성 프롬프트에 삽입할 텍스트로 포맷."""
    lines = [
        f"\n## MCDA 정량 분석 결과 (방법: {result.method})",
        f"종합 점수: {result.weighted_score:.2f}/1.00",
        f"정량 판정: {result.decision.value.upper()}",
        "",
        "기준별 점수:",
    ]

    labels = {
        "business_quality": "사업 품질",
        "financial_health": "재무 건전성",
        "valuation": "밸류에이션",
        "growth": "성장성",
        "risk": "안전성",
        "catalyst": "촉매",
        "portfolio_fit": "포트폴리오 적합성",
    }

    for k, v in result.criteria_scores.to_dict().items():
        bar = "█" * int(v * 10) + "░" * (10 - int(v * 10))
        lines.append(f"  {labels.get(k.value, k.value):10s} [{bar}] {v:.2f}")

    if result.topsis_rank is not None:
        lines.append(f"\nTOPSIS 순위: {result.topsis_rank}위 (근접도: {result.topsis_closeness:.3f})")

    lines.append(
        "\n※ 이 정량 점수는 에이전트들의 정성 분석을 수학적으로 합산한 것입니다."
        "\n  최종 판정 시 이 점수를 기준으로 하되, 정성적 논의와 투자자 피드백도 종합하세요."
    )

    return "\n".join(lines)
