"""
Market Data Fetcher — 실시간 시장 데이터 수집.

에이전트에게 최신 시장 정보를 제공하기 위해
여러 데이터 소스에서 병렬로 데이터를 수집한다.

데이터 소스 (fallback chain):
  1. Yahoo Finance v8 API
  2. 네이버 금융 웹 스크래핑
  3. Google Finance 웹 스크래핑
  4. MarketWatch 웹 스크래핑

수집 항목:
  - 주요 지수 (KOSPI, KOSDAQ, S&P500, NASDAQ 등)
  - 개별 종목 현재가/변동률
  - 환율 (USD/KRW)
  - 최근 주요 뉴스 헤드라인
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Attempt to import BeautifulSoup; degrade gracefully.
try:
    from bs4 import BeautifulSoup
    _HAS_BS4 = True
except ImportError:
    _HAS_BS4 = False

# ── 브라우저 헤더 (스크래핑 공통) ──

_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
}


# ── 데이터 구조 ──

@dataclass
class MarketIndex:
    """시장 지수."""
    name: str
    value: float
    change: float = 0.0
    change_pct: float = 0.0
    source: str = ""


@dataclass
class StockQuote:
    """개별 종목 시세."""
    ticker: str
    name: str
    price: float
    change: float = 0.0
    change_pct: float = 0.0
    volume: int = 0
    market_cap: str = ""
    source: str = ""


@dataclass
class NewsItem:
    """뉴스 헤드라인."""
    title: str
    source: str = ""
    link: str = ""
    published: str = ""


@dataclass
class MarketSnapshot:
    """시장 전체 스냅샷."""
    timestamp: str = ""
    indices: list[MarketIndex] = field(default_factory=list)
    quotes: list[StockQuote] = field(default_factory=list)
    exchange_rates: list[MarketIndex] = field(default_factory=list)
    news: list[NewsItem] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def has_data(self) -> bool:
        return bool(self.indices or self.quotes or self.exchange_rates)

    def to_context_text(self) -> str:
        """에이전트 프롬프트에 삽입할 텍스트로 변환."""
        if not self.has_data and not self.news:
            return ""

        lines = [f"\n## 실시간 시장 데이터 (조회 시각: {self.timestamp})"]
        lines.append("(아래 데이터는 실시간 조회 결과입니다. 분석 시 이 수치를 기준으로 하세요.)\n")

        if self.indices:
            lines.append("### 주요 지수")
            for idx in self.indices:
                sign = "+" if idx.change >= 0 else ""
                lines.append(
                    f"- {idx.name}: {idx.value:,.2f} "
                    f"({sign}{idx.change:,.2f}, {sign}{idx.change_pct:.2f}%)"
                )

        if self.exchange_rates:
            lines.append("\n### 환율")
            for fx in self.exchange_rates:
                sign = "+" if fx.change >= 0 else ""
                lines.append(
                    f"- {fx.name}: {fx.value:,.2f} "
                    f"({sign}{fx.change:,.2f}, {sign}{fx.change_pct:.2f}%)"
                )

        if self.quotes:
            lines.append("\n### 종목 시세")
            for q in self.quotes:
                sign = "+" if q.change >= 0 else ""
                lines.append(
                    f"- {q.name} ({q.ticker}): {q.price:,.0f} "
                    f"({sign}{q.change:,.0f}, {sign}{q.change_pct:.2f}%)"
                    + (f" | 시가총액: {q.market_cap}" if q.market_cap else "")
                )

        if self.news:
            lines.append("\n### 최근 주요 뉴스")
            for n in self.news:
                src = f" [{n.source}]" if n.source else ""
                lines.append(f"- {n.title}{src}")

        return "\n".join(lines)


# ── HTTP 헬퍼 ──

async def _get(client: httpx.AsyncClient, url: str, **kwargs) -> httpx.Response | None:
    """HTTP GET with error handling."""
    try:
        resp = await client.get(url, timeout=12, headers=_BROWSER_HEADERS, **kwargs)
        resp.raise_for_status()
        return resp
    except Exception as e:
        logger.debug(f"GET {url} failed: {e}")
        return None


# ═══════════════════════════════════════════════
# 데이터 소스 1: Yahoo Finance v8 API
# ═══════════════════════════════════════════════

YAHOO_CHART_URL = "https://query2.finance.yahoo.com/v8/finance/chart/{symbol}?range=1d&interval=1d"


async def _yahoo_quote(client: httpx.AsyncClient, symbol: str, name: str) -> MarketIndex | None:
    resp = await _get(client, YAHOO_CHART_URL.format(symbol=symbol))
    if not resp:
        return None
    try:
        data = resp.json()
        meta = data["chart"]["result"][0]["meta"]
        price = meta.get("regularMarketPrice", 0)
        prev = meta.get("chartPreviousClose", meta.get("previousClose", price))
        chg = price - prev
        pct = (chg / prev * 100) if prev else 0
        return MarketIndex(name=name, value=price, change=chg, change_pct=pct, source="Yahoo")
    except Exception:
        return None


# ═══════════════════════════════════════════════
# 데이터 소스 2: 네이버 금융 (KR 지수/종목/뉴스)
# ═══════════════════════════════════════════════

NAVER_SISE_URL = "https://finance.naver.com/sise/"
NAVER_WORLD_URL = "https://finance.naver.com/world/"
NAVER_STOCK_URL = "https://finance.naver.com/item/main.naver?code={code}"
NAVER_NEWS_URL = "https://finance.naver.com/news/mainnews.naver"
# Naver API endpoint (no login, returns JSON)
NAVER_API_INDEX_URL = "https://m.stock.naver.com/api/index/{code}/basic"
NAVER_API_STOCK_URL = "https://m.stock.naver.com/api/stock/{code}/basic"


async def _naver_api_index(client: httpx.AsyncClient, code: str, name: str) -> MarketIndex | None:
    """네이버 모바일 API로 지수 조회 (JSON, 로그인 불필요)."""
    resp = await _get(client, NAVER_API_INDEX_URL.format(code=code))
    if not resp:
        return None
    try:
        data = resp.json()
        price = float(data.get("closePrice", "0").replace(",", ""))
        change = float(data.get("compareToPreviousClosePrice", "0").replace(",", ""))
        pct = float(data.get("fluctuationsRatio", "0").replace(",", ""))
        return MarketIndex(name=name, value=price, change=change, change_pct=pct, source="네이버")
    except Exception:
        return None


async def _naver_api_stock(client: httpx.AsyncClient, code: str, name: str = "") -> StockQuote | None:
    """네이버 모바일 API로 개별 종목 조회."""
    resp = await _get(client, NAVER_API_STOCK_URL.format(code=code))
    if not resp:
        return None
    try:
        data = resp.json()
        price = float(data.get("closePrice", "0").replace(",", ""))
        change = float(data.get("compareToPreviousClosePrice", "0").replace(",", ""))
        pct = float(data.get("fluctuationsRatio", "0").replace(",", ""))
        stock_name = name or data.get("stockName", code)
        cap = data.get("marketValue", "")
        return StockQuote(
            ticker=code, name=stock_name, price=price,
            change=change, change_pct=pct,
            market_cap=cap, source="네이버",
        )
    except Exception:
        return None


async def _naver_news(client: httpx.AsyncClient, max_items: int = 8) -> list[NewsItem]:
    """네이버 금융 주요 뉴스."""
    if not _HAS_BS4:
        return []
    resp = await _get(client, NAVER_NEWS_URL)
    if not resp:
        return []
    news = []
    try:
        soup = BeautifulSoup(resp.text, "html.parser")
        for sel in ["dd.articleSubject a", ".articleSubject a", "a[href*='article_id']"]:
            items = soup.select(sel)
            if items:
                break
        else:
            items = []
        for item in items[:max_items]:
            title = item.get_text(strip=True)
            if title:
                news.append(NewsItem(title=title, source="네이버 금융"))
    except Exception:
        pass
    return news


# ═══════════════════════════════════════════════
# 데이터 소스 3: Google Finance 스크래핑
# ═══════════════════════════════════════════════

GOOGLE_FINANCE_URL = "https://www.google.com/finance/quote/{symbol}"


async def _google_finance_quote(
    client: httpx.AsyncClient, symbol: str, name: str
) -> MarketIndex | None:
    """Google Finance 페이지에서 가격 스크래핑."""
    if not _HAS_BS4:
        return None
    resp = await _get(client, GOOGLE_FINANCE_URL.format(symbol=symbol))
    if not resp:
        return None
    try:
        soup = BeautifulSoup(resp.text, "html.parser")
        # Google Finance uses data-last-price attribute
        price_el = soup.select_one("[data-last-price]")
        if price_el:
            price = float(price_el["data-last-price"])
            change_el = soup.select_one("[data-currency-code] [data-last-price]")
            return MarketIndex(name=name, value=price, source="Google Finance")
        # Fallback: look for the main price text
        price_el = soup.select_one(".YMlKec.fxKbKc")
        if price_el:
            price_text = price_el.get_text(strip=True).replace(",", "").replace("$", "").replace("₩", "")
            price = float(price_text)
            return MarketIndex(name=name, value=price, source="Google Finance")
    except Exception:
        pass
    return None


# ═══════════════════════════════════════════════
# 데이터 소스 4: MarketWatch API
# ═══════════════════════════════════════════════

MARKETWATCH_API_URL = "https://api.wsj.net/api/dylan/quotes/v2/comp/quote?id={id}&type={type}&country={country}"


async def _marketwatch_quote(
    client: httpx.AsyncClient, mw_id: str, mw_type: str,
    country: str, name: str
) -> MarketIndex | None:
    """MarketWatch/WSJ API에서 시세 조회."""
    url = MARKETWATCH_API_URL.format(id=mw_id, type=mw_type, country=country)
    resp = await _get(client, url)
    if not resp:
        return None
    try:
        data = resp.json()
        instrument = data.get("GetInstrumentResponse", {}).get("InstrumentResponses", [{}])[0]
        dynamics = instrument.get("Dynamics", {})
        price = dynamics.get("Last", {}).get("Value", 0)
        change = dynamics.get("Change", {}).get("Value", 0)
        pct = dynamics.get("ChangePercent", {}).get("Value", 0)
        if price:
            return MarketIndex(name=name, value=float(price), change=float(change),
                             change_pct=float(pct), source="MarketWatch")
    except Exception:
        pass
    return None


# ═══════════════════════════════════════════════
# Fallback chain: 여러 소스를 시도하고 첫 성공 반환
# ═══════════════════════════════════════════════

async def _fetch_with_fallback(
    client: httpx.AsyncClient,
    fetchers: list,
) -> MarketIndex | StockQuote | None:
    """여러 fetcher를 순서대로 시도, 첫 성공 반환."""
    for fetcher_coro in fetchers:
        result = await fetcher_coro
        if result is not None:
            return result
    return None


# ═══════════════════════════════════════════════
# 지수/환율/원자재 심볼 매핑
# ═══════════════════════════════════════════════

# (yahoo_symbol, naver_code, google_symbol, name)
INDEX_DEFS = [
    ("^KS11",   "KOSPI",  "KOSPI:KRX",      "KOSPI"),
    ("^KQ11",   "KOSDAQ", "KOSDAQ:KRX",      "KOSDAQ"),
    ("^GSPC",   None,     ".INX:INDEXSP",    "S&P 500"),
    ("^IXIC",   None,     ".IXIC:INDEXNASDAQ","NASDAQ"),
    ("^DJI",    None,     ".DJI:INDEXDJX",   "다우 존스"),
    ("^N225",   None,     "NI225:INDEXNIKKEI","닛케이 225"),
    ("^HSI",    None,     "HSI:INDEXHANGSENG","항셍"),
]

FX_DEFS = [
    ("KRW=X",     None, "USD-KRW",  "USD/KRW"),
    ("JPY=X",     None, "USD-JPY",  "USD/JPY"),
    ("EURUSD=X",  None, "EUR-USD",  "EUR/USD"),
]

COMMODITY_DEFS = [
    ("GC=F",    None, None, "금 (Gold)"),
    ("CL=F",    None, None, "원유 (WTI)"),
    ("BTC-USD", None, "BTC-USD", "비트코인"),
]


# ═══════════════════════════════════════════════
# 메인: 시장 스냅샷 수집
# ═══════════════════════════════════════════════

async def fetch_market_snapshot(
    ticker: str | None = None,
    ticker_name: str | None = None,
    include_news: bool = True,
    include_commodities: bool = False,
) -> MarketSnapshot:
    """
    시장 전체 스냅샷 수집 (여러 소스 fallback).

    Args:
        ticker: 분석 대상 종목 코드 (한국: 6자리 숫자, 미국: 심볼)
        ticker_name: 종목명
        include_news: 뉴스 포함 여부
        include_commodities: 원자재/코인 포함 여부
    """
    snapshot = MarketSnapshot(
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    )

    async with httpx.AsyncClient(follow_redirects=True) as client:

        # ── 지수 수집 (병렬) ──
        async def fetch_index(yahoo_sym, naver_code, google_sym, name):
            fetchers = [_yahoo_quote(client, yahoo_sym, name)]
            if naver_code:
                fetchers.append(_naver_api_index(client, naver_code, name))
            if google_sym:
                fetchers.append(_google_finance_quote(client, google_sym, name))
            return await _fetch_with_fallback(client, fetchers)

        index_tasks = [
            fetch_index(ys, nc, gs, nm) for ys, nc, gs, nm in INDEX_DEFS
        ]

        # ── 환율 수집 (병렬) ──
        async def fetch_fx(yahoo_sym, _, google_sym, name):
            fetchers = [_yahoo_quote(client, yahoo_sym, name)]
            if google_sym:
                fetchers.append(_google_finance_quote(client, google_sym, name))
            return await _fetch_with_fallback(client, fetchers)

        fx_tasks = [
            fetch_fx(ys, nc, gs, nm) for ys, nc, gs, nm in FX_DEFS
        ]

        # ── 원자재 수집 ──
        commodity_tasks = []
        if include_commodities:
            for ys, nc, gs, nm in COMMODITY_DEFS:
                async def fetch_commodity(yahoo_sym=ys, google_sym=gs, name=nm):
                    fetchers = [_yahoo_quote(client, yahoo_sym, name)]
                    if google_sym:
                        fetchers.append(_google_finance_quote(client, google_sym, name))
                    return await _fetch_with_fallback(client, fetchers)
                commodity_tasks.append(fetch_commodity())

        # ── 개별 종목 ──
        stock_task = None
        if ticker:
            is_kr = ticker.isdigit() and len(ticker) == 6
            if is_kr:
                stock_task = _naver_api_stock(client, ticker, ticker_name or "")
            else:
                stock_task = _yahoo_quote(client, ticker, ticker_name or ticker)

        # ── 뉴스 ──
        news_task = _naver_news(client) if include_news else None

        # ── 병렬 실행 ──
        all_tasks = []
        all_tasks.extend(index_tasks)
        all_tasks.extend(fx_tasks)
        all_tasks.extend(commodity_tasks)
        if stock_task:
            all_tasks.append(stock_task)
        if news_task:
            all_tasks.append(news_task)

        results = await asyncio.gather(*all_tasks, return_exceptions=True)

        idx = 0
        # 지수 결과
        for _ in INDEX_DEFS:
            r = results[idx]; idx += 1
            if isinstance(r, MarketIndex):
                snapshot.indices.append(r)

        # 환율 결과
        for _ in FX_DEFS:
            r = results[idx]; idx += 1
            if isinstance(r, MarketIndex):
                snapshot.exchange_rates.append(r)

        # 원자재 결과
        if include_commodities:
            for _ in COMMODITY_DEFS:
                r = results[idx]; idx += 1
                if isinstance(r, MarketIndex):
                    snapshot.indices.append(r)

        # 개별 종목
        if stock_task:
            r = results[idx]; idx += 1
            if isinstance(r, StockQuote):
                snapshot.quotes.append(r)
            elif isinstance(r, MarketIndex):
                snapshot.quotes.append(StockQuote(
                    ticker=ticker or "", name=r.name, price=r.value,
                    change=r.change, change_pct=r.change_pct, source=r.source,
                ))

        # 뉴스
        if news_task:
            r = results[idx]; idx += 1
            if isinstance(r, list):
                snapshot.news = r

    if not snapshot.has_data:
        snapshot.errors.append("외부 데이터 소스 접근 실패 — 네트워크 상태를 확인하세요")

    return snapshot


async def fetch_macro_market_data(include_commodities: bool = True) -> MarketSnapshot:
    """거시 경제 토론용 시장 데이터 (지수 + 환율 + 원자재 + 뉴스)."""
    return await fetch_market_snapshot(
        include_news=True,
        include_commodities=include_commodities,
    )
