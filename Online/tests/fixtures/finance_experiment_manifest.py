"""Locked question rows for the July 18 finance pilot manifest."""

EXPECTED_FINANCE_QUESTION_ROWS = (
    (
        "fin-fed-cut-2026",
        "Will the U.S. Federal Reserve lower its target federal funds range at least once between July 18 and December 31, 2026?",
        "Track the Federal Reserve target range after the cutoff through 2026-12-31.",
        "Yes iff either bound of the federal-funds target range becomes lower than at cutoff on an effective FOMC date after cutoff; Federal Reserve FOMC statements/rate history.",
    ),
    (
        "fin-spx-7000-2026",
        "Will the S&P 500 index close at or above 7,000 on December 31, 2026?",
        "Use the official S&P 500 close on the final regular trading session of 2026.",
        "Yes iff the S&P Dow Jones Indices official index close on the final regular trading session of 2026 is at least 7,000.",
    ),
    (
        "fin-btc-150k-2026",
        "Will Bitcoin close at or above 150,000 U.S. dollars on December 31, 2026?",
        "Use the Coinbase BTC-USD daily candle for 2026-12-31 UTC.",
        "Yes iff the Coinbase Exchange BTC-USD daily candle close for 2026-12-31 UTC is at least USD 150,000.",
    ),
    (
        "fin-gold-4000-2026",
        "Will spot gold close at or above 4,000 U.S. dollars per troy ounce on December 31, 2026?",
        "Use the final LBMA Gold Price PM spot benchmark published in 2026.",
        "Yes iff the final LBMA Gold Price PM benchmark published on or before 2026-12-31 is at least USD 4,000 per troy ounce.",
    ),
    (
        "fin-brent-100-2026",
        "Will Brent crude oil close at or above 100 U.S. dollars per barrel on December 31, 2026?",
        "Use ICE's official Brent front-month settlement on the final applicable 2026 trading day.",
        "Yes iff ICE's official Brent front-month settlement on the final applicable 2026 trading day is at least USD 100 per barrel.",
    ),
    (
        "fin-nvda-5t-2026",
        "Will NVIDIA market cap exceed 5.0 trillion dollars at the end of 2026?",
        "Use final-2026 Nasdaq NVDA close and the latest SEC-filed cover-page common-share count available by 2026-12-31.",
        "Yes iff final-2026 Nasdaq official NVDA close multiplied by common shares outstanding on the cover page of the latest NVIDIA Form 10-Q or 10-K filed with the SEC on or before 2026-12-31 exceeds USD 5T; retain filing accession and cover-page as-of date.",
    ),
    (
        "fin-us-cpi-below-3-2026",
        "Will U.S. CPI be below 3.0 percent in December 2026?",
        "Use the BLS December 2026 CPI-U all-items unadjusted 12-month change.",
        "Yes iff BLS CPI-U all-items, U.S. city average, not seasonally adjusted, December 2026 12-month percent change is below 3.0%.",
    ),
    (
        "fin-boe-cut-2026",
        "Will the Bank of England cut Bank Rate before the end of 2026?",
        "Track official Bank Rate effective dates after the cutoff through 2026-12-31.",
        "Yes iff the official Bank of England Bank Rate is reduced below its cutoff level on an effective date after cutoff through 2026-12-31.",
    ),
    (
        "fin-ecb-cut-2026",
        "Will the European Central Bank lower its deposit facility rate at least once between July 18 and December 31, 2026?",
        "Track official ECB deposit-facility rate effective dates after the cutoff through 2026-12-31.",
        "Yes iff the ECB official deposit-facility rate is reduced below its cutoff level on an effective date after cutoff through 2026-12-31.",
    ),
    (
        "fin-eth-10k-2026",
        "Will Ethereum close above 10,000 U.S. dollars on December 31, 2026?",
        "Use the Coinbase ETH-USD daily candle for 2026-12-31 UTC.",
        "Yes iff the Coinbase Exchange ETH-USD daily candle close for 2026-12-31 UTC is strictly above USD 10,000.",
    ),
)

__all__ = ["EXPECTED_FINANCE_QUESTION_ROWS"]
