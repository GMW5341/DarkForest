"""Portfolio data models."""

from __future__ import annotations

from enum import Enum
from pydantic import BaseModel, Field


class AssetClass(str, Enum):
    STOCK_KR = "stock_kr"       # 한국 주식
    STOCK_US = "stock_us"       # 미국 주식
    BOND = "bond"               # 채권
    ETF = "etf"                 # ETF
    FUND = "fund"               # 펀드
    CASH = "cash"               # 현금성
    CRYPTO = "crypto"           # 암호화폐
    REAL_ESTATE = "real_estate" # 부동산/리츠
    COMMODITY = "commodity"     # 원자재


class Holding(BaseModel):
    """개별 보유 자산."""
    ticker: str = Field(description="종목코드 또는 티커")
    name: str = Field(description="종목명")
    asset_class: AssetClass
    quantity: float = Field(gt=0, description="보유 수량")
    avg_price: float = Field(gt=0, description="평균 매수가")
    current_price: float = Field(gt=0, description="현재가")
    currency: str = Field(default="KRW", description="통화 (KRW, USD 등)")
    sector: str | None = Field(default=None, description="섹터/업종")
    memo: str | None = Field(default=None, description="매수 근거 메모")
    financial_data: str = Field(
        default="",
        description=(
            "재무 데이터 (자유 형식). "
            "예: 매출, 영업이익, PER, PBR, ROE, 부채비율 등 "
            "텍스트로 붙여넣기"
        ),
    )
    reports: list[str] = Field(
        default_factory=list,
        description=(
            "애널리스트 리포트 또는 참고 자료 (텍스트). "
            "증권사 리포트, 뉴스, IR 자료 등을 텍스트로 붙여넣기"
        ),
    )

    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price

    @property
    def cost_basis(self) -> float:
        return self.quantity * self.avg_price

    @property
    def return_pct(self) -> float:
        return (self.current_price - self.avg_price) / self.avg_price * 100

    @property
    def weight_value(self) -> float:
        """포트폴리오 비중 계산 시 사용할 시장가치."""
        return self.market_value


class Portfolio(BaseModel):
    """투자 포트폴리오."""
    name: str = Field(default="My Portfolio", description="포트폴리오 이름")
    holdings: list[Holding] = Field(default_factory=list)
    investment_goal: str | None = Field(
        default=None,
        description="투자 목표 (예: 장기 성장, 배당 수익, 자산 보전 등)",
    )
    risk_tolerance: str | None = Field(
        default=None,
        description="위험 성향 (conservative, moderate, aggressive)",
    )
    investment_horizon: str | None = Field(
        default=None,
        description="투자 기간 (예: 1년, 3년, 5년+)",
    )

    @property
    def total_value(self) -> float:
        return sum(h.market_value for h in self.holdings)

    @property
    def total_cost(self) -> float:
        return sum(h.cost_basis for h in self.holdings)

    @property
    def total_return_pct(self) -> float:
        if self.total_cost == 0:
            return 0.0
        return (self.total_value - self.total_cost) / self.total_cost * 100

    def weight_of(self, holding: Holding) -> float:
        """특정 보유 자산의 포트폴리오 내 비중(%)."""
        if self.total_value == 0:
            return 0.0
        return holding.market_value / self.total_value * 100

    def weights_by_asset_class(self) -> dict[AssetClass, float]:
        """자산군별 비중."""
        result: dict[AssetClass, float] = {}
        for h in self.holdings:
            result[h.asset_class] = result.get(h.asset_class, 0.0) + self.weight_of(h)
        return result

    def weights_by_sector(self) -> dict[str, float]:
        """섹터별 비중."""
        result: dict[str, float] = {}
        for h in self.holdings:
            key = h.sector or "미분류"
            result[key] = result.get(key, 0.0) + self.weight_of(h)
        return result

    def top_holdings(self, n: int = 5) -> list[tuple[Holding, float]]:
        """비중 상위 N개 종목."""
        weighted = [(h, self.weight_of(h)) for h in self.holdings]
        weighted.sort(key=lambda x: x[1], reverse=True)
        return weighted[:n]
