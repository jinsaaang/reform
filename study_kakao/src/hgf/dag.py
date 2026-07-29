"""Compile resolved WorldReasoner DAG exports into bounded agent memories."""

from __future__ import annotations

import re
from typing import Any


_ANSWER_BUCKET_RE = re.compile(
    r"(?:below\s+\d+(?:\.\d+)?%|"
    r"\d+(?:\.\d+)?%\s+to\s+<\d+(?:\.\d+)?%|"
    r"\d+(?:\.\d+)?%\s+or\s+higher)",
    re.IGNORECASE,
)

_RETRIEVAL_STOPWORDS = {
    "about",
    "after",
    "before",
    "bucket",
    "during",
    "from",
    "into",
    "question",
    "that",
    "the",
    "this",
    "what",
    "which",
    "will",
    "with",
}

_SEARCH_FACTOR_FAMILIES = (
    (
        "energy_and_transport",
        r"energy|oil|gas|fuel|electric|utility|transport|shipping",
        "energy, fuel, electricity, transport costs, and commodity shocks",
    ),
    (
        "food_prices",
        r"food|grocery|restaurant|agricultur|crop|beef|coffee",
        "food-at-home, food-away-from-home, agriculture, and grocery prices",
    ),
    (
        "shelter_and_housing",
        r"shelter|housing|rent|mortgage|home price",
        "rent, shelter CPI, housing costs, and mortgage-sensitive demand",
    ),
    (
        "labour_and_wages",
        r"labou?r|wage|employment|unemployment|payroll|job",
        "wages, employment, unemployment, payrolls, and labour tightness",
    ),
    (
        "monetary_policy",
        r"central bank|interest rate|policy rate|repo rate|\bECB\b|\bFed\b|\bRBI\b|\bRBA\b|bank of",
        "central-bank decisions, rates, guidance, and inflation expectations",
    ),
    (
        "trade_fx_and_imports",
        r"tariff|trade|import|export|exchange rate|currency|dollar|euro|yen|rupee",
        "tariffs, trade, exchange rates, import prices, and currency pass-through",
    ),
    (
        "demand_and_growth",
        r"growth|demand|consum|retail|spending|output|GDP|recession|slack",
        "consumer demand, growth, output, spending, and economic slack",
    ),
    (
        "supply_and_geopolitics",
        r"supply|war|conflict|geopolit|disruption|shortage|sanction",
        "supply-chain disruptions, shortages, wars, sanctions, and geopolitical shocks",
    ),
    (
        "base_effects_and_methodology",
        r"base effect|base year|methodolog|tax holiday|tax change|classification|revision",
        "base effects, tax changes, index methodology, weights, and revisions",
    ),
    (
        "official_releases_and_price_momentum",
        r"inflation|consumer price|\bCPI\b|\bHICP\b|price index|official|statistics|release|report",
        "latest official headline/core release, monthly momentum, and component tables",
    ),
)


def _redact_answer_labels(value: Any, resolved_outcome: Any = None) -> Any:
    if not isinstance(value, str):
        return value
    redacted = _ANSWER_BUCKET_RE.sub("[PAST_OUTCOME_REDACTED]", value)
    outcome_text = str(resolved_outcome or "").strip()
    if len(outcome_text) >= 2:
        redacted = re.sub(
            re.escape(outcome_text),
            "[PAST_OUTCOME_REDACTED]",
            redacted,
            flags=re.IGNORECASE,
        )
    return redacted


def _finance_metadata(value: Any) -> dict[str, Any]:
    metadata = getattr(value, "metadata", None)
    if metadata is None and isinstance(value, dict):
        metadata = value.get("metadata", {})
    if not isinstance(metadata, dict):
        return {}
    for namespace in ("finance", "finfactorbench", "benchmark"):
        candidate = metadata.get(namespace)
        if isinstance(candidate, dict):
            return candidate
    return metadata















