#!/usr/bin/env python3
"""Build a long-horizon, family-recurrent finance forecasting benchmark.

The benchmark is intentionally deterministic and does not use an LLM.  It
downloads public FRED series and SEC company facts, then creates 25 recurring
question families with 21 chronological observations each.  The first 14
observations in every family are memory and the last 7 are test, yielding a
500-question, category-balanced chronological split.

Only information known at the forecast cutoff is stored in ``metadata.finance``.
Resolution-only provenance lives in ``metadata.benchmark_private`` and must be
removed from every forecast-time proxy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import statistics
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, time as wall_time, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from src.domain.models.domain import Domain
from src.domain.models.question import Question, QuestionType


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_ROOT = PROJECT_ROOT.parent
DEFAULT_OUTPUT = (
    RESEARCH_ROOT / "data" / "worldreasoner" / "finance_questions_500.jsonl"
)
DEFAULT_MANIFEST = (
    RESEARCH_ROOT
    / "data"
    / "worldreasoner"
    / "finance_questions_500_manifest.json"
)
HGF_300_OUTPUT = (
    RESEARCH_ROOT / "data" / "worldreasoner" / "finance_questions_hgf_300.jsonl"
)
HGF_300_MANIFEST = (
    RESEARCH_ROOT
    / "data"
    / "worldreasoner"
    / "finance_questions_hgf_300_manifest.json"
)
DEFAULT_CACHE = RESEARCH_ROOT / "data" / "cache" / "finance_benchmark_v3"

DATASET_VERSION = "finance_benchmark_v3_500"
START_MONTH = date(2021, 1, 1)
END_MONTH = date(2026, 6, 30)
USER_AGENT = "WorldReasoner finance benchmark research contact: research@example.com"


@dataclass(frozen=True)
class FredFamily:
    family_id: str
    category: str
    subdomain: str
    entity: str
    target_metric: str
    series_id: str
    series_title: str
    change_mode: str
    change_unit: str
    timing: str
    cluster_group: str


@dataclass(frozen=True)
class CompanyFamily:
    family_id: str
    ticker: str
    entity: str
    subdomain: str
    cik: str
    revenue_tags: tuple[str, ...]


@dataclass(frozen=True)
class DatasetProfile:
    """Selection and split policy for one reproducible benchmark view."""

    name: str
    dataset_version: str
    start_month: date
    end_month: date
    large_family_ids: frozenset[str]
    large_family_size: int
    small_family_size: int
    memory_per_family: int
    expected_category_size: int
    expected_question_count: int
    expected_memory_count: int
    expected_test_count: int
    expected_mcq_per_category: int
    expected_binary_per_category: int
    minimum_mcq_distinct_labels: int
    minimum_binary_distinct_labels: int
    minimum_mcq_class_size: int
    minimum_binary_class_size: int
    selection_description: str


FRED_FAMILIES = (
    FredFamily("us_cpi_monthly_change", "macro", "inflation", "United States CPI", "monthly CPI change", "CPIAUCSL", "Consumer Price Index for All Urban Consumers", "pct", "percent", "release", "us_cpi"),
    FredFamily("us_core_cpi_monthly_change", "macro", "inflation", "United States core CPI", "monthly core CPI change", "CPILFESL", "Consumer Price Index excluding food and energy", "pct", "percent", "release", "us_cpi"),
    FredFamily("us_unemployment_rate_change", "macro", "labour", "United States unemployment rate", "monthly unemployment-rate change", "UNRATE", "Civilian Unemployment Rate", "absolute", "percentage points", "release", "us_jobs"),
    FredFamily("us_payroll_monthly_growth", "macro", "labour", "United States nonfarm payrolls", "monthly payroll growth", "PAYEMS", "All Employees, Total Nonfarm", "pct", "percent", "release", "us_jobs"),
    FredFamily("us_retail_sales_monthly_growth", "macro", "consumption", "United States retail sales", "monthly retail-sales growth", "RSAFS", "Advance Retail Sales", "pct", "percent", "release", "us_retail_sales"),
    FredFamily("us_pce_price_monthly_change", "macro", "consumer prices", "United States PCE price index", "monthly PCE-price change", "PCEPI", "Personal Consumption Expenditures: Chain-type Price Index", "pct", "percent", "release_slow", "us_pce_prices"),
    FredFamily("us_industrial_production_monthly_growth", "macro", "industrial activity", "United States industrial production", "monthly industrial-production growth", "INDPRO", "Industrial Production: Total Index", "pct", "percent", "release", "us_industrial_production"),
    FredFamily("us_housing_starts_monthly_growth", "macro", "housing", "United States housing starts", "monthly housing-starts growth", "HOUST", "Housing Starts: Total: New Privately Owned Housing Units Started", "pct", "percent", "release", "us_housing"),
    FredFamily("fed_funds_monthly_change", "monetary_policy", "policy rate", "Federal Reserve effective federal funds rate", "monthly federal-funds-rate change", "FEDFUNDS", "Effective Federal Funds Rate", "absolute", "percentage points", "period", "fed_funds"),
    FredFamily("fed_balance_sheet_monthly_growth", "monetary_policy", "balance sheet", "Federal Reserve total assets", "monthly Federal Reserve asset growth", "WALCL", "Federal Reserve Total Assets", "pct", "percent", "period", "fed_balance_sheet"),
    FredFamily("us_m2_monthly_growth", "monetary_policy", "money supply", "United States M2 money stock", "monthly M2 growth", "M2SL", "M2 Money Stock", "pct", "percent", "period", "us_money_supply"),
    FredFamily("us_2y_yield_monthly_change", "monetary_policy", "rate expectations", "United States 2-year Treasury yield", "monthly 2-year yield change", "DGS2", "2-Year Treasury Constant Maturity Rate", "absolute", "percentage points", "period", "treasury_2y"),
    FredFamily("us_5y_breakeven_monthly_change", "monetary_policy", "inflation expectations", "United States 5-year breakeven inflation rate", "monthly 5-year breakeven change", "T5YIE", "5-Year Breakeven Inflation Rate", "absolute", "percentage points", "period", "inflation_expectations"),
    FredFamily("us_3m_yield_monthly_change", "monetary_policy", "short-term rates", "United States 3-month Treasury yield", "monthly 3-month yield change", "DGS3MO", "3-Month Treasury Constant Maturity Rate", "absolute", "percentage points", "period", "treasury_3m"),
    FredFamily("us_30y_yield_monthly_change", "monetary_policy", "long-term rates", "United States 30-year Treasury yield", "monthly 30-year yield change", "DGS30", "30-Year Treasury Constant Maturity Rate", "absolute", "percentage points", "period", "treasury_30y"),
    FredFamily("us_10y_real_yield_monthly_change", "monetary_policy", "real rates", "United States 10-year real Treasury yield", "monthly 10-year real-yield change", "DFII10", "10-Year Treasury Inflation-Indexed Security, Constant Maturity", "absolute", "percentage points", "period", "treasury_real_10y"),
    FredFamily("sp500_monthly_return", "market_fx_credit", "equity market", "S&P 500 index", "monthly S&P 500 return", "SP500", "S&P 500", "pct", "percent", "market", "equity_market"),
    FredFamily("vix_monthly_change", "market_fx_credit", "volatility", "CBOE VIX index", "monthly VIX change", "VIXCLS", "CBOE Volatility Index", "pct", "percent", "market", "market_volatility"),
    FredFamily("broad_usd_monthly_return", "market_fx_credit", "foreign exchange", "Nominal broad U.S. dollar index", "monthly broad-dollar return", "DTWEXBGS", "Nominal Broad U.S. Dollar Index", "pct", "percent", "market", "foreign_exchange"),
    FredFamily("us_baa_credit_spread_change", "market_fx_credit", "credit spreads", "Moody's Baa corporate yield spread", "monthly Baa-minus-10-year-Treasury spread change", "BAA10Y", "Moody's Seasoned Baa Corporate Bond Yield Relative to 10-Year Treasury", "absolute", "percentage points", "market", "credit_spreads"),
    FredFamily("us_10y_yield_monthly_change", "market_fx_credit", "Treasury rates", "United States 10-year Treasury yield", "monthly 10-year yield change", "DGS10", "10-Year Treasury Constant Maturity Rate", "absolute", "percentage points", "market", "treasury_10y"),
    FredFamily("nasdaq_monthly_return", "market_fx_credit", "technology equities", "NASDAQ Composite index", "monthly NASDAQ Composite return", "NASDAQCOM", "NASDAQ Composite Index", "pct", "percent", "market", "nasdaq_market"),
    FredFamily("us_high_yield_spread_monthly_change", "market_fx_credit", "high-yield credit", "United States high-yield corporate option-adjusted spread", "monthly high-yield spread change", "BAMLH0A0HYM2", "ICE BofA US High Yield Index Option-Adjusted Spread", "absolute", "percentage points", "market", "high_yield_credit"),
    FredFamily("stl_financial_stress_monthly_change", "market_fx_credit", "financial conditions", "St. Louis Fed Financial Stress Index", "monthly financial-stress-index change", "STLFSI4", "St. Louis Fed Financial Stress Index", "absolute", "index points", "market", "financial_stress"),
    FredFamily("wti_monthly_return", "energy_commodities", "crude oil", "West Texas Intermediate crude oil", "monthly WTI price return", "DCOILWTICO", "WTI Crude Oil Price", "pct", "percent", "market", "crude_oil"),
    FredFamily("brent_monthly_return", "energy_commodities", "crude oil", "Brent crude oil", "monthly Brent price return", "DCOILBRENTEU", "Brent Crude Oil Price", "pct", "percent", "market", "crude_oil"),
    FredFamily("henry_hub_monthly_return", "energy_commodities", "natural gas", "Henry Hub natural gas", "monthly Henry Hub price return", "DHHNGSP", "Henry Hub Natural Gas Spot Price", "pct", "percent", "market", "natural_gas"),
    FredFamily("us_gasoline_monthly_change", "energy_commodities", "refined products", "United States regular gasoline price", "monthly retail gasoline-price change", "GASREGW", "US Regular All Formulations Gas Price", "pct", "percent", "market", "refined_fuels"),
    FredFamily("us_diesel_monthly_change", "energy_commodities", "refined products", "United States diesel price", "monthly retail diesel-price change", "GASDESW", "US Diesel Sales Price", "pct", "percent", "market", "refined_fuels"),
    FredFamily("global_copper_monthly_return", "energy_commodities", "industrial metals", "Global copper price", "monthly copper-price return", "PCOPPUSDM", "Global price of Copper", "pct", "percent", "period", "industrial_metals_copper"),
    FredFamily("global_aluminum_monthly_return", "energy_commodities", "industrial metals", "Global aluminum price", "monthly aluminum-price return", "PALUMUSDM", "Global price of Aluminum", "pct", "percent", "period", "industrial_metals_aluminum"),
    FredFamily("global_wheat_monthly_return", "energy_commodities", "agricultural commodities", "Global wheat price", "monthly wheat-price return", "PWHEAMTUSDM", "Global price of Wheat", "pct", "percent", "period", "agricultural_wheat"),
)

COMPANY_FAMILIES = (
    CompanyFamily("aapl_revenue_growth_acceleration", "AAPL", "Apple Inc.", "consumer technology", "0000320193", ("RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet", "Revenues")),
    CompanyFamily("msft_revenue_growth_acceleration", "MSFT", "Microsoft Corporation", "software and cloud", "0000789019", ("RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet", "Revenues")),
    CompanyFamily("amzn_revenue_growth_acceleration", "AMZN", "Amazon.com, Inc.", "e-commerce and cloud", "0001018724", ("RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet", "Revenues")),
    CompanyFamily("googl_revenue_growth_acceleration", "GOOGL", "Alphabet Inc.", "digital advertising", "0001652044", ("RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet")),
    CompanyFamily("nvda_revenue_growth_acceleration", "NVDA", "NVIDIA Corporation", "semiconductors", "0001045810", ("RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet")),
    CompanyFamily("meta_revenue_growth_acceleration", "META", "Meta Platforms, Inc.", "social media and advertising", "0001326801", ("RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet")),
    CompanyFamily("tsla_revenue_growth_acceleration", "TSLA", "Tesla, Inc.", "automotive", "0001318605", ("RevenueFromContractWithCustomerExcludingAssessedTax", "AutomotiveRevenues", "Revenues")),
    CompanyFamily("wmt_revenue_growth_acceleration", "WMT", "Walmart Inc.", "retail", "0000104169", ("RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet", "Revenues")),
)

# Four 13-observation and four 12-observation families in each category produce
# exactly 100 questions per category. Their chronological memory sizes are 9
# and 8 respectively; every family keeps its latest four observations for test.
FAMILY_SIZE_13_IDS = {
    "us_cpi_monthly_change", "us_core_cpi_monthly_change",
    "us_unemployment_rate_change", "us_payroll_monthly_growth",
    "fed_funds_monthly_change", "fed_balance_sheet_monthly_growth",
    "us_m2_monthly_growth", "us_2y_yield_monthly_change",
    "sp500_monthly_return", "vix_monthly_change", "broad_usd_monthly_return",
    "us_baa_credit_spread_change",
    "wti_monthly_return", "brent_monthly_return",
    "us_gasoline_monthly_change", "us_diesel_monthly_change",
    "aapl_revenue_growth_acceleration", "msft_revenue_growth_acceleration",
    "amzn_revenue_growth_acceleration", "googl_revenue_growth_acceleration",
}

# Five of eight families in every category use a genuine single-question MCQ.
# The options are stable across time; numeric bucket boundaries are registered
# from pre-cutoff history for each observation.
MCQ_FAMILY_IDS = {
    # macro
    "us_cpi_monthly_change",
    "us_unemployment_rate_change",
    "us_payroll_monthly_growth",
    "us_industrial_production_monthly_growth",
    "us_housing_starts_monthly_growth",
    # monetary policy
    "fed_funds_monthly_change",
    "us_2y_yield_monthly_change",
    "us_5y_breakeven_monthly_change",
    "us_m2_monthly_growth",
    "us_10y_real_yield_monthly_change",
    # market / FX / credit
    "sp500_monthly_return",
    "vix_monthly_change",
    "broad_usd_monthly_return",
    "nasdaq_monthly_return",
    "stl_financial_stress_monthly_change",
    # energy / commodities
    "wti_monthly_return",
    "us_gasoline_monthly_change",
    "brent_monthly_return",
    "henry_hub_monthly_return",
    "global_copper_monthly_return",
    # corporate earnings
    "aapl_revenue_growth_acceleration",
    "msft_revenue_growth_acceleration",
    "amzn_revenue_growth_acceleration",
    "tsla_revenue_growth_acceleration",
    "wmt_revenue_growth_acceleration",
}
MCQ_OPTIONS = ["below recent range", "within recent range", "above recent range"]

# The HGF paper view keeps every recurring family while shortening the horizon to
# 2023 onward.  In every category, three MCQ families and one binary family have
# eight observations; the other four families have seven.  This yields exactly
# 60 questions (38 MCQ, 22 binary) per category and five memory observations per
# family.
HGF_300_SIZE_8_IDS = {
    # corporate earnings
    "amzn_revenue_growth_acceleration",
    "meta_revenue_growth_acceleration",
    "tsla_revenue_growth_acceleration",
    "wmt_revenue_growth_acceleration",
    # energy / commodities
    "wti_monthly_return",
    "brent_monthly_return",
    "us_gasoline_monthly_change",
    "us_diesel_monthly_change",
    # macro
    "us_cpi_monthly_change",
    "us_unemployment_rate_change",
    "us_payroll_monthly_growth",
    "us_core_cpi_monthly_change",
    # market / FX / credit
    "sp500_monthly_return",
    "vix_monthly_change",
    "broad_usd_monthly_return",
    "us_baa_credit_spread_change",
    # monetary policy
    "fed_funds_monthly_change",
    "us_2y_yield_monthly_change",
    "us_5y_breakeven_monthly_change",
    "fed_balance_sheet_monthly_growth",
}

V3_500_PROFILE = DatasetProfile(
    name="v3_500",
    dataset_version=DATASET_VERSION,
    start_month=START_MONTH,
    end_month=END_MONTH,
    large_family_ids=frozenset(FAMILY_SIZE_13_IDS),
    large_family_size=13,
    small_family_size=12,
    memory_per_family=-1,
    expected_category_size=100,
    expected_question_count=500,
    expected_memory_count=340,
    expected_test_count=160,
    expected_mcq_per_category=63,
    expected_binary_per_category=37,
    minimum_mcq_distinct_labels=3,
    minimum_binary_distinct_labels=2,
    minimum_mcq_class_size=2,
    minimum_binary_class_size=3,
    selection_description="earliest_8_or_9_memory_latest_4_test",
)

HGF_300_PROFILE = DatasetProfile(
    name="hgf_300",
    dataset_version="finance_hgf_v1_300",
    start_month=date(2023, 1, 1),
    end_month=END_MONTH,
    large_family_ids=frozenset(HGF_300_SIZE_8_IDS),
    large_family_size=8,
    small_family_size=7,
    memory_per_family=5,
    expected_category_size=60,
    expected_question_count=300,
    expected_memory_count=200,
    expected_test_count=100,
    expected_mcq_per_category=38,
    expected_binary_per_category=22,
    minimum_mcq_distinct_labels=2,
    minimum_binary_distinct_labels=2,
    minimum_mcq_class_size=1,
    minimum_binary_class_size=1,
    selection_description="five_memory_per_family_then_latest_2_or_3_test",
)

DATASET_PROFILES = {
    V3_500_PROFILE.name: V3_500_PROFILE,
    HGF_300_PROFILE.name: HGF_300_PROFILE,
}


def _family_size(
    family_id: str, profile: DatasetProfile = V3_500_PROFILE
) -> int:
    return (
        profile.large_family_size
        if family_id in profile.large_family_ids
        else profile.small_family_size
    )


def _memory_size(
    family_id: str, profile: DatasetProfile = V3_500_PROFILE
) -> int:
    if profile.memory_per_family >= 0:
        return profile.memory_per_family
    return _family_size(family_id, profile) - 4


def _fetch(url: str, cache_path: Path, refresh: bool) -> tuple[bytes, str]:
    if cache_path.exists() and not refresh:
        payload = cache_path.read_bytes()
        return payload, hashlib.sha256(payload).hexdigest()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                payload = response.read()
            cache_path.write_bytes(payload)
            return payload, hashlib.sha256(payload).hexdigest()
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            time.sleep(1.0 + attempt)
    raise RuntimeError(f"Failed to fetch {url}: {last_error}")


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def _month_start(value: date) -> date:
    return value.replace(day=1)


def _month_end(value: date) -> date:
    next_month = (value.replace(day=28) + timedelta(days=4)).replace(day=1)
    return next_month - timedelta(days=1)


def _at_utc(value: date, *, end_of_day: bool = False) -> datetime:
    clock = wall_time(23, 59, 59) if end_of_day else wall_time.min
    return datetime.combine(value, clock, tzinfo=timezone.utc)


def _evenly_spaced(items: list[Any], count: int) -> list[Any]:
    if len(items) < count:
        raise ValueError(f"Need {count} observations, found {len(items)}")
    if count == 1:
        return [items[-1]]
    indexes = [round(index * (len(items) - 1) / (count - 1)) for index in range(count)]
    if len(set(indexes)) != count:
        raise ValueError("Even-spacing produced duplicate observation indexes")
    return [items[index] for index in indexes]


def _change(current: float, previous: float, mode: str) -> float:
    if mode == "pct":
        if previous == 0:
            raise ValueError("Cannot compute percentage change from zero")
        return (current / previous - 1.0) * 100.0
    if mode == "absolute":
        return current - previous
    raise ValueError(f"Unsupported change mode: {mode}")


def _format_number(value: float) -> str:
    precision = 3 if abs(value) < 10 else 2
    rendered = f"{value:.{precision}f}"
    return rendered.rstrip("0").rstrip(".") if "." in rendered else rendered


def _quartile_boundaries(values: list[float]) -> tuple[float, float]:
    quartiles = statistics.quantiles(values, n=4, method="inclusive")
    lower, upper = quartiles[0], quartiles[2]
    if math.isclose(lower, upper):
        spread = max(abs(lower) * 0.05, 0.01)
        lower -= spread
        upper += spread
    return lower, upper


def _range_class(value: float, lower: float, upper: float) -> str:
    if value < lower:
        return MCQ_OPTIONS[0]
    if value < upper:
        return MCQ_OPTIONS[1]
    return MCQ_OPTIONS[2]


def _fred_monthly_observations(payload: bytes, series_id: str) -> list[dict[str, Any]]:
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8")))
    by_month: dict[tuple[int, int], dict[str, Any]] = {}
    for row in reader:
        raw_value = str(row.get(series_id) or "").strip()
        if not raw_value or raw_value == ".":
            continue
        observed = _parse_date(row["observation_date"])
        by_month[(observed.year, observed.month)] = {
            "date": observed,
            "value": float(raw_value),
        }
    return [by_month[key] for key in sorted(by_month)]


def _fred_cutoff_and_resolution(
    observed: date, timing: str
) -> tuple[datetime, datetime]:
    if timing == "release":
        cutoff_day = _month_end(observed)
        return _at_utc(cutoff_day), _at_utc(cutoff_day + timedelta(days=21), end_of_day=True)
    if timing == "release_slow":
        cutoff_day = _month_end(observed)
        return _at_utc(cutoff_day), _at_utc(cutoff_day + timedelta(days=35), end_of_day=True)
    cutoff_day = _month_start(observed)
    if timing == "period":
        return _at_utc(cutoff_day), _at_utc(_month_end(observed) + timedelta(days=15), end_of_day=True)
    if timing == "market":
        return _at_utc(cutoff_day), _at_utc(observed, end_of_day=True)
    raise ValueError(f"Unsupported timing policy: {timing}")


def _question_payload(question: Question) -> dict[str, Any]:
    return question.model_dump(mode="json", exclude={"created_at", "updated_at"})


def _build_fred_family(
    spec: FredFamily,
    cache_dir: Path,
    refresh: bool,
    retrieved_at: str,
    profile: DatasetProfile = V3_500_PROFILE,
) -> list[Question]:
    url = (
        "https://fred.stlouisfed.org/graph/fredgraph.csv"
        f"?id={spec.series_id}&cosd=2019-01-01"
        f"&coed={profile.end_month.isoformat()}"
    )
    payload, source_hash = _fetch(url, cache_dir / f"fred_{spec.series_id}.csv", refresh)
    observations = _fred_monthly_observations(payload, spec.series_id)
    is_mcq = spec.family_id in MCQ_FAMILY_IDS
    history_change_count = 8 if is_mcq else 4
    eligible_indexes = [
        index
        for index, item in enumerate(observations)
        if profile.start_month <= item["date"] <= profile.end_month
        and index >= history_change_count + 1
    ]
    family_size = _family_size(spec.family_id, profile)
    memory_size = _memory_size(spec.family_id, profile)
    selected_indexes = _evenly_spaced(eligible_indexes, family_size)
    questions: list[Question] = []

    for family_index, observation_index in enumerate(selected_indexes):
        target = observations[observation_index]
        previous = observations[observation_index - 1]
        target_change = _change(target["value"], previous["value"], spec.change_mode)
        prior_changes = [
            _change(
                observations[index]["value"],
                observations[index - 1]["value"],
                spec.change_mode,
            )
            for index in range(
                observation_index - history_change_count, observation_index
            )
        ]
        threshold = statistics.median(prior_changes)
        lower, upper = _quartile_boundaries(prior_changes)
        if is_mcq:
            answer: Any = _range_class(target_change, lower, upper)
            question_type = QuestionType.MCQ
            options = list(MCQ_OPTIONS)
            resolved_class = answer
            comparison_rule = (
                "target change bucketed by the lower and upper quartiles of "
                f"the {history_change_count} prior observed changes"
            )
        else:
            answer = target_change >= threshold
            question_type = QuestionType.BINARY
            options = ["yes", "no"]
            resolved_class = "yes" if answer else "no"
            comparison_rule = (
                f"target change >= median of {history_change_count} prior observed changes"
            )
        cutoff, resolution = _fred_cutoff_and_resolution(target["date"], spec.timing)
        split = "memory" if family_index < memory_size else "test"
        target_period = target["date"].strftime("%Y-%m")
        threshold_text = _format_number(threshold)
        question_id = f"v3_{spec.family_id}_{target_period.replace('-', '_')}"
        source_page = f"https://fred.stlouisfed.org/series/{spec.series_id}"

        public_metadata = {
            "dataset_version": profile.dataset_version,
            "family_id": spec.family_id,
            "event_cluster_id": f"{spec.cluster_group}_{target_period}",
            "category": spec.category,
            "original_domain": spec.category,
            "subdomain": spec.subdomain,
            "entity": spec.entity,
            "region": "US",
            "target_metric": spec.target_metric,
            "target_period": target_period,
            "forecast_cutoff": cutoff.isoformat(),
            "forecast_date_options": [cutoff.date().isoformat()],
            "resolution_available_at": resolution.isoformat(),
            "horizon_days": (resolution - cutoff).days,
            "source_type": "fred_series_snapshot",
            "source_series_id": spec.series_id,
            "source_homepage": source_page,
            "comparison_rule": comparison_rule,
            "comparison_threshold": round(threshold, 6),
            "comparison_thresholds": (
                {"lower": round(lower, 6), "upper": round(upper, 6)}
                if is_mcq
                else None
            ),
            "change_unit": spec.change_unit,
            "threshold_observation_dates": [
                observations[index]["date"].isoformat()
                for index in range(
                    observation_index - history_change_count - 1,
                    observation_index,
                )
            ],
            "split": split,
            "family_position": family_index + 1,
            "family_size": family_size,
            "consensus_snapshot": None,
            "consensus_status": "no_exact_option_space_match",
        }
        private_metadata = {
            "resolved_outcome_label": answer if is_mcq else resolved_class,
            "resolved_class": resolved_class,
            "resolution_value": target["value"],
            "target_change": round(target_change, 8),
            "source_observation_date": target["date"].isoformat(),
            "resolution_document": {
                "publisher": "Federal Reserve Bank of St. Louis (FRED)",
                "title": spec.series_title,
                "url": source_page,
            },
            "source_download_url": url,
            "source_sha256": source_hash,
            "source_retrieved_at": retrieved_at,
        }
        boundary_context = (
            f"\nPre-registered recent-range boundaries: lower "
            f"{_format_number(lower)}, upper {_format_number(upper)} "
            f"{spec.change_unit}."
            if is_mcq
            else ""
        )
        question_text = (
            f"As of {cutoff.date().isoformat()}, which range will the "
            f"{spec.target_metric} for {spec.entity} in {target_period} fall into?"
            if is_mcq
            else (
                f"As of {cutoff.date().isoformat()}, will the {spec.target_metric} "
                f"for {spec.entity} in {target_period} be at least "
                f"{threshold_text} {spec.change_unit}?"
            )
        )
        resolution_criteria = (
            f"Use FRED series {spec.series_id}. Compute the change from the previous "
            f"monthly observation using mode '{spec.change_mode}'. Resolve 'below "
            f"recent range' below {_format_number(lower)}, 'within recent range' "
            f"from {_format_number(lower)} inclusive to {_format_number(upper)} "
            f"exclusive, and 'above recent range' at or above "
            f"{_format_number(upper)} {spec.change_unit}."
            if is_mcq
            else (
                f"Use FRED series {spec.series_id}. Compute the change from the "
                f"previous monthly observation using mode '{spec.change_mode}'. "
                f"Resolve yes if the change is at least {threshold_text} "
                f"{spec.change_unit}, otherwise no."
            )
        )
        questions.append(
            Question(
                id=question_id,
                question_text=question_text,
                question_type=question_type,
                domain=Domain.FINANCE,
                source=profile.dataset_version,
                difficulty=3,
                resolution_date=resolution,
                estimated_start_time=cutoff,
                ground_truth=answer,
                options=options,
                context=(
                    f"Entity: {spec.entity}\nTarget period: {target_period}\n"
                    f"Financial area: {spec.category} / {spec.subdomain}\nRegion: US"
                    f"{boundary_context}"
                ),
                resolution_criteria=resolution_criteria,
                resolution_reasoning=(
                    f"The snapshotted target change was {_format_number(target_change)} "
                    f"{spec.change_unit}; it resolved to {resolved_class!r}."
                ),
                is_synthetic=False,
                metadata={
                    "finance": public_metadata,
                    "benchmark_private": private_metadata,
                },
            )
        )
    return questions


def _duration_days(item: dict[str, Any]) -> int:
    return (_parse_date(item["end"]) - _parse_date(item["start"])).days + 1


def _earliest_by_period(
    facts: Iterable[dict[str, Any]], form: str, minimum_days: int, maximum_days: int
) -> dict[tuple[str, str], dict[str, Any]]:
    selected: dict[tuple[str, str], dict[str, Any]] = {}
    for item in facts:
        if item.get("form") != form or not all(
            item.get(key) for key in ("start", "end", "filed", "accn")
        ):
            continue
        duration = _duration_days(item)
        if not minimum_days <= duration <= maximum_days:
            continue
        if _parse_date(item["filed"]) > _parse_date(item["end"]) + timedelta(days=180):
            # Comparative values repeated in a later filing are not the original
            # resolution document for that historical quarter.
            continue
        key = (item["start"], item["end"])
        if key not in selected or item["filed"] < selected[key]["filed"]:
            selected[key] = item
    return selected


def _quarterly_revenue(
    company_facts: dict[str, Any], tags: tuple[str, ...]
) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    us_gaap = company_facts["facts"]["us-gaap"]
    for tag in tags:
        for item in us_gaap.get(tag, {}).get("units", {}).get("USD", []):
            facts.append({**item, "_source_tag": tag})
    quarters = _earliest_by_period(facts, "10-Q", 70, 120)
    annuals = _earliest_by_period(facts, "10-K", 300, 400)
    derived: list[dict[str, Any]] = []
    for (annual_start, annual_end), annual in sorted(annuals.items()):
        inside = sorted(
            (
                item
                for item in quarters.values()
                if _parse_date(annual_start)
                <= _parse_date(item["start"])
                <= _parse_date(item["end"])
                <= _parse_date(annual_end)
            ),
            key=lambda item: item["end"],
        )
        if len(inside) < 3:
            continue
        first_three = inside[:3]
        q4_value = int(annual["val"]) - sum(int(item["val"]) for item in first_three)
        if q4_value <= 0:
            continue
        derived.extend(first_three)
        derived.append(
            {
                "start": (
                    _parse_date(first_three[-1]["end"]) + timedelta(days=1)
                ).isoformat(),
                "end": annual_end,
                "filed": annual["filed"],
                "val": q4_value,
                "accn": annual["accn"],
                "form": "10-K",
                "_source_tag": annual.get("_source_tag"),
            }
        )
    by_end: dict[str, dict[str, Any]] = {}
    for item in derived:
        end = item["end"]
        if end not in by_end or item["filed"] < by_end[end]["filed"]:
            by_end[end] = item
    return [by_end[end] for end in sorted(by_end)]


def _sec_filing_url(cik: str, accession: str) -> str:
    cik_plain = str(int(cik))
    accession_plain = accession.replace("-", "")
    return (
        "https://www.sec.gov/Archives/edgar/data/"
        f"{cik_plain}/{accession_plain}/{accession}-index.html"
    )


def _build_company_family(
    spec: CompanyFamily,
    cache_dir: Path,
    refresh: bool,
    retrieved_at: str,
    profile: DatasetProfile = V3_500_PROFILE,
) -> list[Question]:
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{spec.cik}.json"
    payload, source_hash = _fetch(url, cache_dir / f"sec_{spec.ticker}.json", refresh)
    company_facts = json.loads(payload)
    quarters = _quarterly_revenue(company_facts, spec.revenue_tags)
    is_mcq = spec.family_id in MCQ_FAMILY_IDS
    minimum_history = 9 if is_mcq else 5
    eligible_indexes = [
        index
        for index, item in enumerate(quarters)
        if profile.start_month <= _parse_date(item["end"]) <= profile.end_month
        and index >= minimum_history
    ]
    family_size = _family_size(spec.family_id, profile)
    memory_size = _memory_size(spec.family_id, profile)
    selected_indexes = _evenly_spaced(eligible_indexes, family_size)
    questions: list[Question] = []

    for family_index, quarter_index in enumerate(selected_indexes):
        target = quarters[quarter_index]
        previous = quarters[quarter_index - 1]
        target_yoy = (int(target["val"]) / int(quarters[quarter_index - 4]["val"]) - 1) * 100
        previous_yoy = (int(previous["val"]) / int(quarters[quarter_index - 5]["val"]) - 1) * 100
        target_acceleration = target_yoy - previous_yoy
        if is_mcq:
            prior_accelerations = []
            for index in range(quarter_index - 4, quarter_index):
                growth = (
                    int(quarters[index]["val"])
                    / int(quarters[index - 4]["val"])
                    - 1
                ) * 100
                prior_growth = (
                    int(quarters[index - 1]["val"])
                    / int(quarters[index - 5]["val"])
                    - 1
                ) * 100
                prior_accelerations.append(growth - prior_growth)
            lower, upper = _quartile_boundaries(prior_accelerations)
            answer: Any = _range_class(target_acceleration, lower, upper)
            question_type = QuestionType.MCQ
            options = list(MCQ_OPTIONS)
            resolved_class = answer
            comparison_rule = (
                "target YoY revenue-growth acceleration bucketed by quartiles "
                "of the four prior quarterly accelerations"
            )
        else:
            lower = upper = previous_yoy
            answer = target_yoy >= previous_yoy
            question_type = QuestionType.BINARY
            options = ["yes", "no"]
            resolved_class = "yes" if answer else "no"
            comparison_rule = (
                "target YoY revenue growth >= prior-quarter YoY revenue growth"
            )
        cutoff = _at_utc(_parse_date(target["end"]))
        resolution = _at_utc(_parse_date(target["filed"]), end_of_day=True)
        if cutoff >= resolution:
            raise ValueError(f"SEC cutoff does not predate filing for {spec.ticker}: {target}")
        split = "memory" if family_index < memory_size else "test"
        period = target["end"]
        threshold_text = _format_number(previous_yoy)
        question_id = f"v3_{spec.family_id}_{period.replace('-', '_')}"
        filings_home = f"https://www.sec.gov/edgar/browse/?CIK={int(spec.cik)}"
        filing_url = _sec_filing_url(spec.cik, target["accn"])

        public_metadata = {
            "dataset_version": profile.dataset_version,
            "family_id": spec.family_id,
            "event_cluster_id": f"{spec.ticker.lower()}_earnings_{period}",
            "category": "corporate_earnings",
            "original_domain": "corporate_earnings",
            "subdomain": spec.subdomain,
            "entity": f"{spec.entity} ({spec.ticker})",
            "region": "US",
            "target_metric": "quarterly revenue year-over-year growth acceleration",
            "target_period": f"fiscal quarter ending {period}",
            "forecast_cutoff": cutoff.isoformat(),
            "forecast_date_options": [cutoff.date().isoformat()],
            "resolution_available_at": resolution.isoformat(),
            "horizon_days": (resolution - cutoff).days,
            "source_type": "sec_companyfacts_snapshot",
            "source_series_id": f"CIK{spec.cik}:revenue",
            "source_homepage": filings_home,
            "comparison_rule": comparison_rule,
            "comparison_threshold": round(previous_yoy, 6),
            "comparison_thresholds": (
                {"lower": round(lower, 6), "upper": round(upper, 6)}
                if is_mcq
                else None
            ),
            "change_unit": "percent",
            "threshold_observation_dates": [
                quarters[quarter_index - 5]["end"],
                quarters[quarter_index - 4]["end"],
                previous["end"],
            ],
            "split": split,
            "family_position": family_index + 1,
            "family_size": family_size,
            "consensus_snapshot": None,
            "consensus_status": "no_exact_option_space_match",
        }
        private_metadata = {
            "resolved_outcome_label": answer if is_mcq else resolved_class,
            "resolved_class": resolved_class,
            "resolution_value": int(target["val"]),
            "target_change": round(
                target_acceleration if is_mcq else target_yoy, 8
            ),
            "target_yoy_growth": round(target_yoy, 8),
            "target_growth_acceleration": round(target_acceleration, 8),
            "source_observation_date": period,
            "resolution_document": {
                "publisher": "U.S. Securities and Exchange Commission",
                "title": f"{spec.entity} {target['form']} filed {target['filed']}",
                "url": filing_url,
            },
            "source_download_url": url,
            "source_sha256": source_hash,
            "source_retrieved_at": retrieved_at,
            "accession_number": target["accn"],
            "xbrl_tag": target.get("_source_tag"),
        }
        boundary_context = (
            f"\nPre-registered growth-acceleration boundaries: lower "
            f"{_format_number(lower)}, upper {_format_number(upper)} percentage points."
            if is_mcq
            else ""
        )
        question_text = (
            f"As of {cutoff.date().isoformat()}, which range will {spec.entity}'s "
            f"quarterly year-over-year revenue-growth acceleration for the fiscal "
            f"quarter ending {period} fall into?"
            if is_mcq
            else (
                f"As of {cutoff.date().isoformat()}, will {spec.entity}'s quarterly "
                f"revenue year-over-year growth for the fiscal quarter ending {period} "
                f"be at least {threshold_text} percent?"
            )
        )
        resolution_criteria = (
            "Use the issuer's first SEC 10-Q or 10-K containing the target quarter. "
            "Compute the change in YoY revenue growth from the prior quarter. Resolve "
            f"'below recent range' below {_format_number(lower)}, 'within recent "
            f"range' from {_format_number(lower)} inclusive to {_format_number(upper)} "
            f"exclusive, and 'above recent range' at or above "
            f"{_format_number(upper)} percentage points."
            if is_mcq
            else (
                "Use the issuer's first SEC 10-Q or 10-K containing the target "
                f"quarter. Resolve yes if YoY revenue growth is at least "
                f"{threshold_text} percent, the prior quarter's pre-registered "
                "YoY growth rate; otherwise resolve no."
            )
        )
        questions.append(
            Question(
                id=question_id,
                question_text=question_text,
                question_type=question_type,
                domain=Domain.FINANCE,
                source=profile.dataset_version,
                difficulty=4,
                resolution_date=resolution,
                estimated_start_time=cutoff,
                ground_truth=answer,
                options=options,
                context=(
                    f"Entity: {spec.entity} ({spec.ticker})\n"
                    f"Target period: fiscal quarter ending {period}\n"
                    f"Financial area: corporate_earnings / {spec.subdomain}\nRegion: US"
                    f"{boundary_context}"
                ),
                resolution_criteria=resolution_criteria,
                resolution_reasoning=(
                    f"The snapshotted target-quarter YoY revenue growth was "
                    f"{_format_number(target_yoy)} percent and growth acceleration was "
                    f"{_format_number(target_acceleration)} percentage points; it "
                    f"resolved to {resolved_class!r}."
                ),
                is_synthetic=False,
                metadata={
                    "finance": public_metadata,
                    "benchmark_private": private_metadata,
                },
            )
        )
    return questions


def _validate(
    questions: list[Question],
    profile: DatasetProfile = V3_500_PROFILE,
) -> dict[str, Any]:
    family_ids = [spec.family_id for spec in (*FRED_FAMILIES, *COMPANY_FAMILIES)]
    expected = sum(_family_size(family_id, profile) for family_id in family_ids)
    if len(questions) != expected:
        raise ValueError(f"Expected {expected} questions, found {len(questions)}")
    if len({question.id for question in questions}) != len(questions):
        raise ValueError("Duplicate question IDs found")

    family_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    question_type_counts: Counter[str] = Counter()
    category_type_counts: dict[str, Counter[str]] = {}
    split_type_counts: dict[str, Counter[str]] = {}
    subcategory_counts: dict[str, Counter[str]] = {}
    source_type_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    overall_labels: Counter[str] = Counter()
    labels_by_split: dict[str, Counter[str]] = {}
    label_counts: dict[str, Counter[str]] = {}
    horizons_by_category: dict[str, list[int]] = {}
    dates: list[str] = []
    cluster_splits: dict[str, set[str]] = {}
    for question in questions:
        metadata = question.metadata["finance"]
        private = question.metadata["benchmark_private"]
        family_id = metadata["family_id"]
        split = metadata["split"]
        family_counts[family_id] += 1
        category_counts[metadata["category"]] += 1
        question_type_counts[question.question_type.value] += 1
        category_type_counts.setdefault(metadata["category"], Counter())[
            question.question_type.value
        ] += 1
        split_type_counts.setdefault(split, Counter())[
            question.question_type.value
        ] += 1
        subcategory_counts.setdefault(metadata["category"], Counter())[
            metadata["subdomain"]
        ] += 1
        source_type_counts[metadata["source_type"]] += 1
        split_counts[split] += 1
        label = private.get("resolved_class", private["resolved_outcome_label"])
        overall_labels[label] += 1
        labels_by_split.setdefault(split, Counter())[label] += 1
        label_counts.setdefault(family_id, Counter())[label] += 1
        horizons_by_category.setdefault(metadata["category"], []).append(
            metadata["horizon_days"]
        )
        dates.append(metadata["target_period"][-10:])
        cluster_splits.setdefault(metadata["event_cluster_id"], set()).add(split)
        cutoff = datetime.fromisoformat(metadata["forecast_cutoff"])
        if cutoff >= question.resolution_date:
            raise ValueError(f"Cutoff/resolution violation: {question.id}")
        if "resolution_value" in metadata or "resolved_outcome_label" in metadata:
            raise ValueError(f"Outcome leaked into public finance metadata: {question.id}")

    expected_family_counts = {
        family_id: _family_size(family_id, profile) for family_id in family_ids
    }
    if dict(family_counts) != expected_family_counts:
        raise ValueError(f"Family counts do not match the design: {family_counts}")
    if set(category_counts.values()) != {profile.expected_category_size}:
        raise ValueError(f"Category counts are not uniform: {category_counts}")
    expected_category_types = {
        "mcq": profile.expected_mcq_per_category,
        "binary": profile.expected_binary_per_category,
    }
    if any(
        dict(counts) != expected_category_types
        for counts in category_type_counts.values()
    ):
        raise ValueError(
            "Category question-type counts do not match the profile: "
            f"{category_type_counts}"
        )
    if split_counts != {
        "memory": profile.expected_memory_count,
        "test": profile.expected_test_count,
    }:
        raise ValueError(f"Split counts do not match the profile: {split_counts}")
    leaking_clusters = [cluster for cluster, splits in cluster_splits.items() if len(splits) > 1]
    if leaking_clusters:
        raise ValueError(f"Event clusters cross split boundaries: {leaking_clusters[:10]}")
    for family_id, counts in label_counts.items():
        minimum_distinct_labels = (
            profile.minimum_mcq_distinct_labels
            if family_id in MCQ_FAMILY_IDS
            else profile.minimum_binary_distinct_labels
        )
        minimum_class_size = (
            profile.minimum_mcq_class_size
            if family_id in MCQ_FAMILY_IDS
            else profile.minimum_binary_class_size
        )
        if (
            len(counts) < minimum_distinct_labels
            or min(counts.values()) < minimum_class_size
        ):
            raise ValueError(f"Severely imbalanced labels in {family_id}: {counts}")

    memory_by_family: dict[str, list[Question]] = {}
    for question in questions:
        metadata = question.metadata["finance"]
        if metadata["split"] == "memory":
            memory_by_family.setdefault(metadata["family_id"], []).append(question)
    same_family_eligible_counts = []
    for question in questions:
        metadata = question.metadata["finance"]
        if metadata["split"] != "test":
            continue
        cutoff = datetime.fromisoformat(metadata["forecast_cutoff"])
        same_family_eligible_counts.append(
            sum(
                memory.resolution_date < cutoff
                for memory in memory_by_family[metadata["family_id"]]
            )
        )
    if set(same_family_eligible_counts) != {profile.memory_per_family} and (
        profile.memory_per_family >= 0
    ):
        raise ValueError(
            "Not every test has all same-family memories cutoff-eligible: "
            f"{Counter(same_family_eligible_counts)}"
        )
    if profile.memory_per_family < 0 and (
        min(same_family_eligible_counts) < 8
        or max(same_family_eligible_counts) > 9
    ):
        raise ValueError(
            "Not every test has all same-family memories cutoff-eligible: "
            f"{Counter(same_family_eligible_counts)}"
        )

    return {
        "dataset_version": profile.dataset_version,
        "question_count": len(questions),
        "family_count": len(family_counts),
        "observations_per_family": {
            "min": profile.small_family_size,
            "max": profile.large_family_size,
        },
        "category_counts": dict(sorted(category_counts.items())),
        "question_type_counts": dict(question_type_counts),
        "category_type_counts": {
            category: dict(counts)
            for category, counts in sorted(category_type_counts.items())
        },
        "split_type_counts": {
            split: dict(counts) for split, counts in sorted(split_type_counts.items())
        },
        "subcategory_counts": {
            category: dict(sorted(counts.items()))
            for category, counts in sorted(subcategory_counts.items())
        },
        "subcategory_count_by_category": {
            category: len(counts)
            for category, counts in sorted(subcategory_counts.items())
        },
        "source_type_counts": dict(source_type_counts),
        "split_counts": dict(split_counts),
        "overall_label_counts": dict(overall_labels),
        "label_counts_by_split": {
            split: dict(counts) for split, counts in sorted(labels_by_split.items())
        },
        "family_counts": dict(sorted(family_counts.items())),
        "family_label_counts": {
            family: dict(counts) for family, counts in sorted(label_counts.items())
        },
        "target_period_min": min(dates),
        "target_period_max": max(dates),
        "horizon_days_by_category": {
            category: {
                "min": min(values),
                "median": statistics.median(values),
                "max": max(values),
            }
            for category, values in sorted(horizons_by_category.items())
        },
        "same_family_eligible_memory": {
            "min": min(same_family_eligible_counts),
            "max": max(same_family_eligible_counts),
        },
        "cluster_count": len(cluster_splits),
        "cluster_split_violations": 0,
    }


def _write_jsonl(path: Path, questions: Iterable[Question]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for question in questions:
            handle.write(json.dumps(_question_payload(question), ensure_ascii=False) + "\n")


def build_dataset(
    output: Path,
    manifest_path: Path,
    cache_dir: Path,
    refresh: bool = False,
    profile: DatasetProfile = V3_500_PROFILE,
) -> dict[str, Any]:
    retrieved_at = datetime.now(timezone.utc).isoformat()
    questions: list[Question] = []
    for spec in FRED_FAMILIES:
        questions.extend(
            _build_fred_family(
                spec,
                cache_dir,
                refresh,
                retrieved_at,
                profile=profile,
            )
        )
    for spec in COMPANY_FAMILIES:
        questions.extend(
            _build_company_family(
                spec,
                cache_dir,
                refresh,
                retrieved_at,
                profile=profile,
            )
        )
    questions.sort(
        key=lambda question: (
            question.metadata["finance"]["category"],
            question.metadata["finance"]["family_id"],
            question.metadata["finance"]["family_position"],
        )
    )
    summary = _validate(questions, profile)
    _write_jsonl(output, questions)
    manifest = {
        **summary,
        "output": str(output.resolve()),
        "construction": {
            "profile": profile.name,
            "randomized": False,
            "time_range": [
                profile.start_month.isoformat(),
                profile.end_month.isoformat(),
            ],
            "family_split": profile.selection_description,
            "split_ratio": (
                f"{profile.expected_memory_count}:"
                f"{profile.expected_test_count}"
            ),
            "resolution_availability_policy": (
                "SEC uses the first filing timestamp. FRED release series use "
                "conservative fixed lags, while period and market series use "
                "conservative post-period availability. This can delay memory "
                "eligibility but cannot expose unresolved outcomes."
            ),
            "outcome_privacy": (
                "Forecast-visible fields are under metadata.finance; resolved-only "
                "fields are under metadata.benchmark_private."
            ),
            "polymarket_policy": (
                "Attach a consensus snapshot only when market question and option "
                "space exactly match at the forecast cutoff."
            ),
        },
        "memory_question_ids": [
            question.id
            for question in questions
            if question.metadata["finance"]["split"] == "memory"
        ],
        "test_question_ids": [
            question.id
            for question in questions
            if question.metadata["finance"]["split"] == "test"
        ],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=tuple(DATASET_PROFILES),
        default=V3_500_PROFILE.name,
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--refresh", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = DATASET_PROFILES[args.profile]
    output = args.output or (
        HGF_300_OUTPUT if profile is HGF_300_PROFILE else DEFAULT_OUTPUT
    )
    manifest_path = args.manifest or (
        HGF_300_MANIFEST if profile is HGF_300_PROFILE else DEFAULT_MANIFEST
    )
    manifest = build_dataset(
        output,
        manifest_path,
        args.cache_dir,
        args.refresh,
        profile=profile,
    )
    print(
        json.dumps(
            {
                key: manifest[key]
                for key in (
                    "dataset_version",
                    "question_count",
                    "family_count",
                    "category_counts",
                    "split_counts",
                    "target_period_min",
                    "target_period_max",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
