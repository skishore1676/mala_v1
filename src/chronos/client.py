"""
Polygon.io REST API client for the Chronos data pipeline.

Handles:
  - Authentication via API key
  - Fetching 1-minute OHLCV aggregates with automatic pagination
  - Rate-limit-aware retries
"""

from __future__ import annotations

import time
from datetime import date, timedelta
from typing import List, Optional

import requests
from loguru import logger
from tqdm import tqdm

from src.config import settings

# ── Constants ────────────────────────────────────────────────────────────────
BASE_URL = "https://api.polygon.io"
AGGS_ENDPOINT = "/v2/aggs/ticker/{ticker}/range/1/minute/{start}/{end}"
MAX_RESULTS_PER_PAGE = 50_000  # Polygon max limit per request
REQUEST_DELAY_S = 0.25  # polite pause between paginated calls


class PolygonClient:
    """Thin wrapper around the Polygon.io Aggregates API."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key or settings.polygon_api_key
        if not self.api_key:
            raise ValueError(
                "POLYGON_API_KEY is not set. "
                "Provide it in .env or pass it explicitly."
            )
        self._session = requests.Session()
        self._session.params = {"apiKey": self.api_key}  # type: ignore[assignment]

    # ── Public API ───────────────────────────────────────────────────────
    def fetch_aggs(
        self,
        ticker: str,
        start: date,
        end: date,
        adjusted: bool = True,
    ) -> List[dict]:
        """
        Download 1-minute aggregates for *ticker* between *start* and *end*
        (inclusive). Automatically paginates if the window exceeds one page.

        Returns a list of bar dicts with keys:
            t, o, h, l, c, v, vw, n
        """
        all_results: List[dict] = []
        url = BASE_URL + AGGS_ENDPOINT.format(
            ticker=ticker,
            start=start.isoformat(),
            end=end.isoformat(),
        )
        params = {
            "adjusted": str(adjusted).lower(),
            "sort": "asc",
            "limit": MAX_RESULTS_PER_PAGE,
        }

        page = 0
        while url:
            page += 1
            logger.debug("Fetching page {} for {} ({} → {})", page, ticker, start, end)
            resp = self._get_with_retry(url, params=params if page == 1 else None)
            data = resp.json()

            results = data.get("results", [])
            all_results.extend(results)

            # Polygon returns "next_url" if there are more pages
            url = data.get("next_url")
            if url:
                time.sleep(REQUEST_DELAY_S)

        logger.info(
            "Downloaded {} bars for {} ({} → {})",
            len(all_results),
            ticker,
            start,
            end,
        )
        return all_results

    def fetch_aggs_chunked(
        self,
        ticker: str,
        start: date,
        end: date,
        chunk_days: int = 30,
        adjusted: bool = True,
    ) -> List[dict]:
        """
        Break a large date range into *chunk_days*-sized windows and fetch
        each one sequentially. This avoids hitting Polygon timeouts on
        multi-year queries.
        """
        all_results: List[dict] = []
        current = start
        total_days = (end - start).days
        pbar = tqdm(total=total_days, desc=f"Downloading {ticker}", unit="day")

        while current <= end:
            chunk_end = min(current + timedelta(days=chunk_days - 1), end)
            bars = self.fetch_aggs(ticker, current, chunk_end, adjusted=adjusted)
            all_results.extend(bars)
            advance = (chunk_end - current).days + 1
            pbar.update(advance)
            current = chunk_end + timedelta(days=1)

        pbar.close()
        return all_results

    # ── Private Helpers ──────────────────────────────────────────────────
    def _get_with_retry(
        self,
        url: str,
        params: Optional[dict] = None,
        max_retries: int = 3,
        backoff: float = 2.0,
    ) -> requests.Response:
        """GET with exponential back-off on transient errors."""
        for attempt in range(1, max_retries + 1):
            try:
                resp = self._session.get(url, params=params, timeout=30)
                resp.raise_for_status()
                return resp
            except requests.exceptions.RequestException as exc:
                if attempt == max_retries:
                    logger.error("Request failed after {} retries: {}", max_retries, exc)
                    raise
                wait = backoff ** attempt
                logger.warning(
                    "Attempt {}/{} failed ({}). Retrying in {:.1f}s …",
                    attempt,
                    max_retries,
                    exc,
                    wait,
                )
                time.sleep(wait)
        # unreachable, but keeps mypy happy
        raise RuntimeError("Unreachable")
