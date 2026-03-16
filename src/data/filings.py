"""
Corporate Filings Fetcher — DART (한국) / SEC EDGAR (미국) 공시 자동 수집.

종목 분석 시 ticker와 asset_class만 주어지면:
  - 한국 종목: OpenDART API로 최신 사업보고서 재무제표 수집
  - 미국 종목: SEC EDGAR API로 최신 10-K 핵심 재무 데이터 수집

에이전트에게 실제 공시 기반 데이터를 제공하여 할루시네이션을 방지한다.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# ── HTTP 공통 ──

_TIMEOUT = 15

_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/html, */*",
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
}

# SEC EDGAR는 contact email을 User-Agent에 포함해야 함
_SEC_HEADERS = {
    "User-Agent": "DarkForest/1.0 (darkforest@example.com)",
    "Accept": "application/json",
}


# ═══════════════════════════════════════════════
# 결과 모델
# ═══════════════════════════════════════════════

@dataclass
class FilingData:
    """공시에서 추출한 재무 데이터."""
    ticker: str
    name: str = ""
    source: str = ""               # "DART" or "SEC EDGAR"
    filing_type: str = ""          # "사업보고서" or "10-K"
    filing_date: str = ""          # 공시 일자
    fiscal_year: str = ""          # 회계 연도

    # 손익계산서
    revenue: str = ""
    operating_profit: str = ""
    net_income: str = ""
    # 재무상태표
    total_assets: str = ""
    total_liabilities: str = ""
    total_equity: str = ""
    # 현금흐름표
    operating_cashflow: str = ""
    investing_cashflow: str = ""
    financing_cashflow: str = ""
    # 주요 비율
    debt_ratio: str = ""
    roe: str = ""
    operating_margin: str = ""

    # 연간 재무 추이 (최근 3~5년)
    annual_data: list[dict[str, str]] = field(default_factory=list)
    # 원본 텍스트 (사업 개요 등)
    business_summary: str = ""
    # 수집 중 에러
    errors: list[str] = field(default_factory=list)

    def to_context_text(self) -> str:
        """에이전트 프롬프트에 삽입할 공시 데이터 텍스트."""
        if not self.revenue and not self.total_assets and not self.annual_data:
            if self.errors:
                return f"\n## 공시 데이터 ({self.source})\n(수집 실패: {', '.join(self.errors)})\n"
            return ""

        lines = [f"\n## 공시 데이터 ({self.source} — {self.filing_type})"]
        lines.append(
            "(아래 데이터는 공시 원문에서 자동 추출한 실제 재무제표입니다. "
            "추측이 아닌 이 수치를 기준으로 분석하세요.)\n"
        )

        if self.filing_date:
            lines.append(f"공시일: {self.filing_date}")
        if self.fiscal_year:
            lines.append(f"회계연도: {self.fiscal_year}")

        # 손익계산서
        if self.revenue or self.operating_profit or self.net_income:
            lines.append("\n### 손익계산서")
            if self.revenue:
                lines.append(f"- 매출액: {self.revenue}")
            if self.operating_profit:
                lines.append(f"- 영업이익: {self.operating_profit}")
            if self.net_income:
                lines.append(f"- 당기순이익: {self.net_income}")
            if self.operating_margin:
                lines.append(f"- 영업이익률: {self.operating_margin}")

        # 재무상태표
        if self.total_assets or self.total_equity:
            lines.append("\n### 재무상태표")
            if self.total_assets:
                lines.append(f"- 자산총계: {self.total_assets}")
            if self.total_liabilities:
                lines.append(f"- 부채총계: {self.total_liabilities}")
            if self.total_equity:
                lines.append(f"- 자본총계: {self.total_equity}")
            if self.debt_ratio:
                lines.append(f"- 부채비율: {self.debt_ratio}")

        # 현금흐름표
        if self.operating_cashflow:
            lines.append("\n### 현금흐름표")
            lines.append(f"- 영업활동 현금흐름: {self.operating_cashflow}")
            if self.investing_cashflow:
                lines.append(f"- 투자활동 현금흐름: {self.investing_cashflow}")
            if self.financing_cashflow:
                lines.append(f"- 재무활동 현금흐름: {self.financing_cashflow}")

        # ROE
        if self.roe:
            lines.append(f"\n### 수익성")
            lines.append(f"- ROE: {self.roe}")

        # 연간 추이
        if self.annual_data:
            lines.append("\n### 연간 재무 추이")
            for yr in self.annual_data:
                period = yr.get("period", "?")
                rev = yr.get("revenue", "-")
                op = yr.get("operating_profit", "-")
                ni = yr.get("net_income", "-")
                lines.append(f"- {period}: 매출 {rev} / 영업이익 {op} / 순이익 {ni}")

        # 사업 개요
        if self.business_summary:
            lines.append(f"\n### 사업 개요 (공시 원문 발췌)")
            lines.append(self.business_summary[:3000])

        if self.errors:
            lines.append(f"\n(일부 항목 수집 실패: {', '.join(self.errors)})")

        return "\n".join(lines)


# ═══════════════════════════════════════════════
# 메인 진입점
# ═══════════════════════════════════════════════

async def fetch_filing_data(
    ticker: str,
    name: str = "",
    dart_api_key: str = "",
) -> FilingData:
    """
    종목 코드/티커로 공시 재무 데이터를 자동 수집.

    한국 종목(6자리 숫자): OpenDART API
    미국 종목: SEC EDGAR API (키 불필요)
    """
    is_kr = ticker.isdigit() and len(ticker) == 6
    filing = FilingData(ticker=ticker, name=name)

    async with httpx.AsyncClient(follow_redirects=True) as client:
        if is_kr:
            api_key = dart_api_key or os.environ.get("DART_API_KEY", "")
            if not api_key:
                filing.errors.append(
                    "DART API 키가 없습니다. 설정에서 dart_api_key를 등록하거나 "
                    "DART_API_KEY 환경변수를 설정하세요. "
                    "(https://opendart.fss.or.kr 에서 무료 발급)"
                )
                return filing
            await _fetch_dart(client, ticker, api_key, filing)
        else:
            await _fetch_sec_edgar(client, ticker, filing)

    return filing


# ═══════════════════════════════════════════════
# DART (한국 종목)
# ═══════════════════════════════════════════════

# OpenDART API endpoints
_DART_BASE = "https://opendart.fss.or.kr/api"
_DART_CORP_CODE = f"{_DART_BASE}/corpCode.xml"          # 고유번호 매핑
_DART_COMPANY = f"{_DART_BASE}/company.json"             # 기업 개황
_DART_LIST = f"{_DART_BASE}/list.json"                   # 공시 검색
_DART_FNLTT_SINGLE = f"{_DART_BASE}/fnlttSinglAcntAll.json"  # 단일회사 전체 재무제표
_DART_KEY_FINANCIAL = f"{_DART_BASE}/fnlttMultiAcnt.json"     # 다중회사 주요 재무


async def _fetch_dart(
    client: httpx.AsyncClient,
    stock_code: str,
    api_key: str,
    out: FilingData,
) -> None:
    """OpenDART API로 한국 종목 공시 재무 데이터 수집."""
    out.source = "DART"
    out.filing_type = "사업보고서"

    # Step 1: 종목코드 → DART 고유번호(corp_code) 매핑
    corp_code = await _dart_get_corp_code(client, stock_code, api_key, out)
    if not corp_code:
        return

    # Step 2: 기업 개황 조회
    await _dart_company_info(client, corp_code, api_key, out)

    # Step 3: 최신 사업보고서 재무제표 (최근 3년)
    import datetime
    current_year = datetime.date.today().year

    for year in range(current_year - 1, current_year - 4, -1):
        await _dart_financial_statements(client, corp_code, api_key, str(year), out)


async def _dart_get_corp_code(
    client: httpx.AsyncClient,
    stock_code: str,
    api_key: str,
    out: FilingData,
) -> str | None:
    """종목코드(6자리) → DART 고유번호 매핑.

    OpenDART의 corpCode.xml은 10MB+ ZIP이라 무겁다.
    대신 list.json API에 종목코드 직접 조회하여 corp_code를 역추출한다.
    """
    try:
        resp = await client.get(
            _DART_LIST,
            params={
                "crtfc_key": api_key,
                "corp_code": "",
                "bgn_de": "20240101",
                "end_de": "20261231",
                "pblntf_ty": "A",  # 정기공시
                "page_count": "5",
                "stock_code": stock_code,
            },
            timeout=_TIMEOUT,
            headers=_BROWSER_HEADERS,
        )
        resp.raise_for_status()
        data = resp.json()

        if data.get("status") != "000":
            # status != 000이면 데이터 없음 — company.json으로 시도
            return await _dart_corp_code_via_company(client, stock_code, api_key, out)

        items = data.get("list", [])
        if items:
            corp_code = items[0].get("corp_code", "")
            out.name = out.name or items[0].get("corp_name", "")
            if corp_code:
                return corp_code

    except Exception as e:
        logger.debug(f"DART corp_code lookup failed: {e}")

    return await _dart_corp_code_via_company(client, stock_code, api_key, out)


async def _dart_corp_code_via_company(
    client: httpx.AsyncClient,
    stock_code: str,
    api_key: str,
    out: FilingData,
) -> str | None:
    """company.json API로 종목코드 → corp_code 매핑 (fallback)."""
    try:
        resp = await client.get(
            _DART_COMPANY,
            params={"crtfc_key": api_key, "stock_code": stock_code},
            timeout=_TIMEOUT,
            headers=_BROWSER_HEADERS,
        )
        resp.raise_for_status()
        data = resp.json()

        if data.get("status") == "000":
            out.name = out.name or data.get("corp_name", "")
            return data.get("corp_code", "")

        out.errors.append(f"DART 기업 조회 실패: {data.get('message', 'unknown')}")
    except Exception as e:
        out.errors.append(f"DART 기업 조회 오류: {e}")

    return None


async def _dart_company_info(
    client: httpx.AsyncClient,
    corp_code: str,
    api_key: str,
    out: FilingData,
) -> None:
    """기업 개황 (업종, 대표자, 설립일 등)."""
    try:
        resp = await client.get(
            _DART_COMPANY,
            params={"crtfc_key": api_key, "corp_code": corp_code},
            timeout=_TIMEOUT,
            headers=_BROWSER_HEADERS,
        )
        resp.raise_for_status()
        data = resp.json()

        if data.get("status") == "000":
            out.name = out.name or data.get("corp_name", "")
            # 사업 개요에 기업 개황 추가
            parts = []
            if data.get("induty_code"):
                parts.append(f"업종코드: {data['induty_code']}")
            if data.get("est_dt"):
                parts.append(f"설립일: {data['est_dt']}")
            if data.get("ceo_nm"):
                parts.append(f"대표자: {data['ceo_nm']}")
            if data.get("hm_url"):
                parts.append(f"홈페이지: {data['hm_url']}")
            if data.get("acc_mt"):
                parts.append(f"결산월: {data['acc_mt']}월")
            if parts:
                out.business_summary = "기업 개황: " + " | ".join(parts)
    except Exception as e:
        logger.debug(f"DART company info failed: {e}")


async def _dart_financial_statements(
    client: httpx.AsyncClient,
    corp_code: str,
    api_key: str,
    year: str,
    out: FilingData,
) -> None:
    """특정 연도 사업보고서 재무제표 수집.

    reprt_code:
      11011 = 사업보고서 (연간)
      11012 = 반기보고서
      11013 = 1분기보고서
      11014 = 3분기보고서

    fs_div:
      OFS = 개별재무제표
      CFS = 연결재무제표 (우선 시도)
    """
    for fs_div in ("CFS", "OFS"):
        try:
            resp = await client.get(
                _DART_FNLTT_SINGLE,
                params={
                    "crtfc_key": api_key,
                    "corp_code": corp_code,
                    "bsns_year": year,
                    "reprt_code": "11011",  # 사업보고서
                    "fs_div": fs_div,
                },
                timeout=_TIMEOUT,
                headers=_BROWSER_HEADERS,
            )
            resp.raise_for_status()
            data = resp.json()

            if data.get("status") != "000":
                continue

            items = data.get("list", [])
            if not items:
                continue

            # 연결재무제표 데이터 있으면 파싱
            _parse_dart_financials(items, year, fs_div, out)
            return  # 성공하면 리턴

        except Exception as e:
            logger.debug(f"DART financials {year}/{fs_div} failed: {e}")

    # CFS/OFS 둘 다 실패
    out.errors.append(f"{year}년 사업보고서 재무제표 조회 실패")


def _parse_dart_financials(
    items: list[dict],
    year: str,
    fs_div: str,
    out: FilingData,
) -> None:
    """DART 재무제표 JSON → FilingData 필드 매핑."""
    fs_label = "연결" if fs_div == "CFS" else "개별"
    out.fiscal_year = out.fiscal_year or f"{year}년 ({fs_label}재무제표)"
    out.filing_date = out.filing_date or f"{year}년 사업보고서"

    # account_nm 기준으로 주요 항목 추출
    yearly = {"period": year}

    for item in items:
        acct = item.get("account_nm", "")
        amount_str = item.get("thstrm_amount", "") or item.get("thstrm_dt", "")

        # 당기 금액 파싱
        if not amount_str:
            continue

        # 손익계산서 항목
        if acct in ("매출액", "수익(매출액)", "영업수익"):
            if not out.revenue:
                out.revenue = _format_krw(amount_str)
            yearly["revenue"] = _format_krw(amount_str)

        elif acct in ("영업이익", "영업이익(손실)"):
            if not out.operating_profit:
                out.operating_profit = _format_krw(amount_str)
            yearly["operating_profit"] = _format_krw(amount_str)

        elif acct in ("당기순이익", "당기순이익(손실)", "당기순이익(손실)의 귀속"):
            if not out.net_income:
                out.net_income = _format_krw(amount_str)
            yearly["net_income"] = _format_krw(amount_str)

        # 재무상태표 항목
        elif acct in ("자산총계",):
            if not out.total_assets:
                out.total_assets = _format_krw(amount_str)

        elif acct in ("부채총계",):
            if not out.total_liabilities:
                out.total_liabilities = _format_krw(amount_str)

        elif acct in ("자본총계",):
            if not out.total_equity:
                out.total_equity = _format_krw(amount_str)

        # 현금흐름표 항목
        elif acct in ("영업활동현금흐름", "영업활동 현금흐름"):
            if not out.operating_cashflow:
                out.operating_cashflow = _format_krw(amount_str)

        elif acct in ("투자활동현금흐름", "투자활동 현금흐름"):
            if not out.investing_cashflow:
                out.investing_cashflow = _format_krw(amount_str)

        elif acct in ("재무활동현금흐름", "재무활동 현금흐름"):
            if not out.financing_cashflow:
                out.financing_cashflow = _format_krw(amount_str)

    # 부채비율 계산
    if out.total_liabilities and out.total_equity:
        try:
            liab = _parse_amount(out.total_liabilities)
            eq = _parse_amount(out.total_equity)
            if eq and eq != 0:
                out.debt_ratio = f"{liab / eq * 100:.1f}%"
        except (ValueError, ZeroDivisionError):
            pass

    # 영업이익률 계산
    if out.revenue and out.operating_profit:
        try:
            rev = _parse_amount(out.revenue)
            op = _parse_amount(out.operating_profit)
            if rev and rev != 0:
                out.operating_margin = f"{op / rev * 100:.1f}%"
        except (ValueError, ZeroDivisionError):
            pass

    # ROE 계산
    if out.net_income and out.total_equity:
        try:
            ni = _parse_amount(out.net_income)
            eq = _parse_amount(out.total_equity)
            if eq and eq != 0:
                out.roe = f"{ni / eq * 100:.1f}%"
        except (ValueError, ZeroDivisionError):
            pass

    # 연간 데이터에 추가
    if any(v for k, v in yearly.items() if k != "period"):
        out.annual_data.append(yearly)


def _format_krw(amount_str: str) -> str:
    """금액 문자열을 읽기 쉬운 형식으로 변환. DART는 원 단위."""
    try:
        amount_str = amount_str.replace(",", "").replace(" ", "").strip()
        if not amount_str or amount_str == "-":
            return ""
        val = int(amount_str)
        if abs(val) >= 1_000_000_000_000:  # 조
            return f"{val / 1_000_000_000_000:.2f}조원"
        if abs(val) >= 100_000_000:  # 억
            return f"{val / 100_000_000:.0f}억원"
        return f"{val:,}원"
    except (ValueError, TypeError):
        return amount_str


def _parse_amount(formatted: str) -> float:
    """포맷된 금액을 숫자로 역변환."""
    s = formatted.replace(",", "").replace(" ", "").strip()
    if s.endswith("조원"):
        return float(s.replace("조원", "")) * 1_000_000_000_000
    if s.endswith("억원"):
        return float(s.replace("억원", "")) * 100_000_000
    if s.endswith("원"):
        return float(s.replace("원", ""))
    return float(s)


# ═══════════════════════════════════════════════
# SEC EDGAR (미국 종목)
# ═══════════════════════════════════════════════

_SEC_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik}.json"
_SEC_COMPANY_FACTS = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
_SEC_TICKER_MAP = "https://www.sec.gov/files/company_tickers.json"


async def _fetch_sec_edgar(
    client: httpx.AsyncClient,
    ticker: str,
    out: FilingData,
) -> None:
    """SEC EDGAR API로 미국 종목 10-K 재무 데이터 수집."""
    out.source = "SEC EDGAR"
    out.filing_type = "10-K"

    # Step 1: Ticker → CIK 번호 매핑
    cik = await _sec_get_cik(client, ticker, out)
    if not cik:
        return

    # Step 2: Company submissions에서 10-K 공시일 확인
    await _sec_filing_info(client, cik, out)

    # Step 3: XBRL Company Facts로 재무 데이터 수집
    await _sec_company_facts(client, cik, out)


async def _sec_get_cik(
    client: httpx.AsyncClient,
    ticker: str,
    out: FilingData,
) -> str | None:
    """Ticker → CIK 번호 매핑."""
    try:
        resp = await client.get(
            _SEC_TICKER_MAP,
            timeout=_TIMEOUT,
            headers=_SEC_HEADERS,
        )
        resp.raise_for_status()
        data = resp.json()

        ticker_upper = ticker.upper()
        for entry in data.values():
            if entry.get("ticker", "").upper() == ticker_upper:
                cik = str(entry.get("cik_str", ""))
                out.name = out.name or entry.get("title", "")
                # CIK는 10자리 0-padding
                return cik.zfill(10)

        out.errors.append(f"SEC에서 티커 '{ticker}' 매핑 실패")

    except Exception as e:
        out.errors.append(f"SEC 티커 매핑 오류: {e}")

    return None


async def _sec_filing_info(
    client: httpx.AsyncClient,
    cik: str,
    out: FilingData,
) -> None:
    """Submissions API로 최신 10-K 공시 정보 확인."""
    try:
        url = _SEC_SUBMISSIONS.format(cik=cik)
        resp = await client.get(url, timeout=_TIMEOUT, headers=_SEC_HEADERS)
        resp.raise_for_status()
        data = resp.json()

        out.name = out.name or data.get("name", "")

        # SIC code → industry
        if data.get("sic"):
            out.business_summary = f"SIC: {data['sic']} ({data.get('sicDescription', '')})"
        if data.get("stateOfIncorporation"):
            out.business_summary += f" | 설립지: {data['stateOfIncorporation']}"

        # 최근 filing에서 10-K 찾기
        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        periods = recent.get("reportDate", [])

        for i, form in enumerate(forms):
            if form == "10-K":
                out.filing_date = dates[i] if i < len(dates) else ""
                out.fiscal_year = periods[i] if i < len(periods) else ""
                break

    except Exception as e:
        logger.debug(f"SEC filing info failed: {e}")


async def _sec_company_facts(
    client: httpx.AsyncClient,
    cik: str,
    out: FilingData,
) -> None:
    """XBRL Company Facts API로 핵심 재무 데이터 수집.

    이 API는 회사의 모든 XBRL 태그 데이터를 시계열로 제공한다.
    10-K에 해당하는 연간 데이터만 추출한다.
    """
    try:
        url = _SEC_COMPANY_FACTS.format(cik=cik)
        resp = await client.get(url, timeout=20, headers=_SEC_HEADERS)
        resp.raise_for_status()
        data = resp.json()

        facts = data.get("facts", {})
        us_gaap = facts.get("us-gaap", {})
        ifrs = facts.get("ifrs-full", {})

        # us-gaap 우선, 없으면 ifrs
        source = us_gaap if us_gaap else ifrs

        if not source:
            out.errors.append("XBRL 재무 데이터 없음")
            return

        # 핵심 항목 매핑 (us-gaap 태그명 → 필드)
        mapping = {
            # 손익계산서
            "Revenues": "revenue",
            "RevenueFromContractWithCustomerExcludingAssessedTax": "revenue",
            "SalesRevenueNet": "revenue",
            "OperatingIncomeLoss": "operating_profit",
            "NetIncomeLoss": "net_income",
            # 재무상태표
            "Assets": "total_assets",
            "Liabilities": "total_liabilities",
            "StockholdersEquity": "total_equity",
            "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest": "total_equity",
            # 현금흐름표
            "NetCashProvidedByUsedInOperatingActivities": "operating_cashflow",
            "NetCashProvidedByUsedInInvestingActivities": "investing_cashflow",
            "NetCashProvidedByUsedInFinancingActivities": "financing_cashflow",
        }

        # 연간 데이터 수집 (최근 4년)
        annual_map: dict[str, dict[str, str]] = {}  # year → {field: value}

        for xbrl_tag, field_name in mapping.items():
            tag_data = source.get(xbrl_tag, {})
            units = tag_data.get("units", {})
            # USD 단위 찾기
            values = units.get("USD", []) or units.get("USD/shares", [])
            if not values:
                continue

            # 10-K (연간) 데이터만 필터: form == "10-K"
            annual_entries = [
                v for v in values
                if v.get("form") == "10-K" and v.get("val") is not None
            ]

            if not annual_entries:
                continue

            # 최신 순 정렬
            annual_entries.sort(key=lambda x: x.get("end", ""), reverse=True)

            # 가장 최신 값을 메인 필드에 할당
            latest = annual_entries[0]
            if not getattr(out, field_name):
                setattr(out, field_name, _format_usd(latest["val"]))

            # 연간 추이 데이터 수집
            for entry in annual_entries[:4]:
                year = entry.get("end", "")[:4]
                if year:
                    if year not in annual_map:
                        annual_map[year] = {"period": year}
                    if field_name in ("revenue", "operating_profit", "net_income"):
                        annual_map[year][field_name] = _format_usd(entry["val"])

        # 연간 추이 정렬 (오래된 순)
        if annual_map:
            out.annual_data = sorted(
                annual_map.values(),
                key=lambda x: x.get("period", ""),
            )

        # 비율 계산
        _compute_sec_ratios(out)

    except Exception as e:
        out.errors.append(f"SEC XBRL 데이터 수집 오류: {e}")


def _compute_sec_ratios(out: FilingData) -> None:
    """SEC 데이터에서 비율 계산."""
    if out.total_liabilities and out.total_equity:
        try:
            liab = _parse_usd(out.total_liabilities)
            eq = _parse_usd(out.total_equity)
            if eq and eq != 0:
                out.debt_ratio = f"{liab / eq * 100:.1f}%"
        except (ValueError, ZeroDivisionError):
            pass

    if out.revenue and out.operating_profit:
        try:
            rev = _parse_usd(out.revenue)
            op = _parse_usd(out.operating_profit)
            if rev and rev != 0:
                out.operating_margin = f"{op / rev * 100:.1f}%"
        except (ValueError, ZeroDivisionError):
            pass

    if out.net_income and out.total_equity:
        try:
            ni = _parse_usd(out.net_income)
            eq = _parse_usd(out.total_equity)
            if eq and eq != 0:
                out.roe = f"{ni / eq * 100:.1f}%"
        except (ValueError, ZeroDivisionError):
            pass


def _format_usd(val: float | int) -> str:
    """USD 금액을 읽기 쉬운 형식으로 변환."""
    try:
        val = float(val)
        if abs(val) >= 1_000_000_000_000:
            return f"${val / 1_000_000_000_000:.2f}T"
        if abs(val) >= 1_000_000_000:
            return f"${val / 1_000_000_000:.2f}B"
        if abs(val) >= 1_000_000:
            return f"${val / 1_000_000:.1f}M"
        return f"${val:,.0f}"
    except (ValueError, TypeError):
        return str(val)


def _parse_usd(formatted: str) -> float:
    """포맷된 USD 금액을 숫자로 역변환."""
    s = formatted.replace("$", "").replace(",", "").strip()
    if s.endswith("T"):
        return float(s[:-1]) * 1_000_000_000_000
    if s.endswith("B"):
        return float(s[:-1]) * 1_000_000_000
    if s.endswith("M"):
        return float(s[:-1]) * 1_000_000
    return float(s)
