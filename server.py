from __future__ import annotations

import argparse
import json
import math
import mimetypes
import os
import threading
import time
import traceback
import webbrowser
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, time as clock_time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yfinance as yf


APP_ROOT = Path(__file__).resolve().parent
WEB_ROOT = APP_ROOT / "web"
CACHE_TTL_SECONDS = max(30, int(os.environ.get("CACHE_TTL_SECONDS", "300")))
MAX_CACHE_ENTRIES = max(4, int(os.environ.get("MAX_CACHE_ENTRIES", "64")))
RATE_LIMIT_REQUESTS = max(2, int(os.environ.get("RATE_LIMIT_REQUESTS", "20")))
RATE_LIMIT_WINDOW_SECONDS = max(60, int(os.environ.get("RATE_LIMIT_WINDOW_SECONDS", "600")))
SCREEN_CONCURRENCY = max(1, int(os.environ.get("SCREEN_CONCURRENCY", "2")))
SCREEN_CACHE: OrderedDict[str, tuple[float, dict[str, Any]]] = OrderedDict()
SCREEN_CACHE_LOCK = threading.Lock()
SCREEN_SEMAPHORE = threading.BoundedSemaphore(SCREEN_CONCURRENCY)
RATE_LIMIT_BUCKETS: dict[str, deque[float]] = {}
RATE_LIMIT_LOCK = threading.Lock()
TRADINGVIEW_COLUMNS = [
    "name",
    "description",
    "exchange",
    "close",
    "change",
    "volume",
    "relative_volume_10d_calc",
    "market_cap_basic",
    "float_shares_outstanding",
]

MARKETS = {
    "TW": {
        "label": "台股",
        "scanner": "taiwan",
        "benchmark": "^TWII",
        "currency": "TWD",
        "timezone": "Asia/Taipei",
        "market_open": clock_time(9, 0),
        "market_close": clock_time(13, 30),
        "surge_threshold": 9.5,
        "cap_min": 20_000_000_000,
        "cap_max": 100_000_000_000,
    },
    "US": {
        "label": "美股",
        "scanner": "america",
        "benchmark": "SPY",
        "currency": "USD",
        "timezone": "America/New_York",
        "market_open": clock_time(9, 30),
        "market_close": clock_time(16, 0),
        "surge_threshold": 10.0,
        "cap_min": 1_000_000_000,
        "cap_max": 10_000_000_000,
    },
}

DEFAULT_TICKERS = {
    "TW": [
        "2330.TW", "2317.TW", "2454.TW", "2382.TW", "2308.TW",
        "2881.TW", "2882.TW", "2891.TW", "3711.TW", "3231.TW",
        "6669.TW", "3037.TW", "3017.TW", "2603.TW", "2618.TW",
        "2408.TW", "2344.TW", "1605.TW", "2002.TW", "1301.TW",
    ],
    "US": [
        "AAPL", "MSFT", "NVDA", "AMZN", "META", "TSLA", "AMD", "PLTR",
        "COIN", "SOFI", "HOOD", "SMCI", "AVGO", "NFLX", "MU", "INTC",
        "ARM", "CRWD", "UBER", "SNOW",
    ],
}

RULE_LABELS = {
    "washout": "主力拉盤前期／洗盤",
    "overnight": "隔日沖（一夜持股法）",
    "pre_surge": "大漲前四特徵",
}


def number(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def clamp_int(value: Any, low: int, high: int, fallback: int) -> int:
    try:
        return max(low, min(high, int(value)))
    except (TypeError, ValueError):
        return fallback


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return number(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def screen_cache_key(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def cached_screen(payload: dict[str, Any]) -> dict[str, Any] | None:
    key = screen_cache_key(payload)
    now = time.time()
    with SCREEN_CACHE_LOCK:
        expired = [cache_key for cache_key, (created_at, _) in SCREEN_CACHE.items() if now - created_at > CACHE_TTL_SECONDS]
        for cache_key in expired:
            SCREEN_CACHE.pop(cache_key, None)
        cached = SCREEN_CACHE.get(key)
        if cached is None:
            return None
        SCREEN_CACHE.move_to_end(key)
        return {**cached[1], "cache_hit": True}


def store_screen_cache(payload: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    key = screen_cache_key(payload)
    stored = {**result, "cache_hit": False}
    with SCREEN_CACHE_LOCK:
        SCREEN_CACHE[key] = (time.time(), stored)
        SCREEN_CACHE.move_to_end(key)
        while len(SCREEN_CACHE) > MAX_CACHE_ENTRIES:
            SCREEN_CACHE.popitem(last=False)
    return stored


def rate_limit_status(client_key: str) -> tuple[bool, int]:
    now = time.time()
    cutoff = now - RATE_LIMIT_WINDOW_SECONDS
    with RATE_LIMIT_LOCK:
        bucket = RATE_LIMIT_BUCKETS.setdefault(client_key, deque())
        while bucket and bucket[0] < cutoff:
            bucket.popleft()
        if len(bucket) >= RATE_LIMIT_REQUESTS:
            retry_after = max(1, math.ceil(bucket[0] + RATE_LIMIT_WINDOW_SECONDS - now))
            return False, retry_after
        bucket.append(now)
        if len(RATE_LIMIT_BUCKETS) > 5000:
            stale = [key for key, values in RATE_LIMIT_BUCKETS.items() if not values or values[-1] < cutoff]
            for key in stale:
                RATE_LIMIT_BUCKETS.pop(key, None)
        return True, 0


def normalize_symbol(raw: str, market: str) -> str | None:
    symbol = raw.strip().upper().replace(" ", "")
    if not symbol:
        return None
    if ":" in symbol:
        symbol = symbol.split(":", 1)[1]
    if market == "TW" and not symbol.endswith((".TW", ".TWO")):
        symbol = f"{symbol}.TW"
    if market == "US":
        symbol = symbol.replace("/", "-")
    return symbol


def parse_tickers(text: str, market: str) -> list[str]:
    for delimiter in ["\n", ",", ";", "\t", "，", "；"]:
        text = text.replace(delimiter, " ")
    seen: set[str] = set()
    output: list[str] = []
    for part in text.split():
        symbol = normalize_symbol(part, market)
        if symbol and symbol not in seen:
            seen.add(symbol)
            output.append(symbol)
    return output


def market_session_is_partial(market: str) -> bool:
    settings = MARKETS[market]
    now = datetime.now(ZoneInfo(settings["timezone"]))
    return now.weekday() < 5 and settings["market_open"] <= now.time() < settings["market_close"]


def tradingview_scan(market: str, strategy: str, universe_limit: int, params: dict[str, Any]) -> list[dict[str, Any]]:
    settings = MARKETS[market]
    filters: list[dict[str, Any]] = [{"left": "type", "operation": "equal", "right": "stock"}]
    if market == "US":
        filters.append({"left": "subtype", "operation": "equal", "right": "common"})
        filters.append({"left": "exchange", "operation": "in_range", "right": ["NASDAQ", "NYSE", "AMEX"]})
    else:
        filters.append({"left": "exchange", "operation": "in_range", "right": ["TWSE", "TPEX"]})
    if strategy == "overnight":
        filters.extend([
            {"left": "change", "operation": "in_range", "right": [params["change_min"], params["change_max"]]},
            {"left": "relative_volume_10d_calc", "operation": "egreater", "right": params["volume_ratio_min"]},
            {"left": "market_cap_basic", "operation": "in_range", "right": [params["cap_min"], params["cap_max"]]},
        ])
        fetch_limit = min(5000, max(universe_limit, 1000))
    else:
        fetch_limit = universe_limit

    payload = {
        "filter": filters,
        "options": {"lang": "zh_TW" if market == "TW" else "en"},
        "markets": [settings["scanner"]],
        "symbols": {"query": {"types": []}, "tickers": []},
        "columns": TRADINGVIEW_COLUMNS,
        "sort": {"sortBy": "volume", "sortOrder": "desc"},
        "range": [0, max(0, fetch_limit - 1)],
    }
    url = f"https://scanner.tradingview.com/{settings['scanner']}/scan"
    response = requests.post(url, json=payload, timeout=30)
    response.raise_for_status()
    raw = response.json()
    rows = []
    for item in raw.get("data", []):
        values = item.get("d", [])
        if len(values) != len(TRADINGVIEW_COLUMNS):
            continue
        row = dict(zip(TRADINGVIEW_COLUMNS, values))
        allowed_exchanges = {"TWSE", "TPEX"} if market == "TW" else {"NASDAQ", "NYSE", "AMEX"}
        if str(row.get("exchange", "")).upper() not in allowed_exchanges:
            continue
        provider_symbol = item.get("s", "")
        raw_symbol = str(row["name"])
        if market == "TW" and str(row.get("exchange", "")).upper() == "TPEX":
            symbol = f"{raw_symbol}.TWO"
        else:
            symbol = normalize_symbol(raw_symbol, market)
        if not symbol:
            continue
        float_shares = number(row.get("float_shares_outstanding"))
        volume = number(row.get("volume"))
        turnover = volume / float_shares * 100 if volume is not None and float_shares else None
        row.update({
            "symbol": symbol,
            "provider_symbol": provider_symbol,
            "float_shares": float_shares,
            "shares_basis": "流通股本",
            "turnover_pct_snapshot": turnover,
        })
        if strategy == "overnight" and not (
            turnover is not None and params["turnover_min"] <= turnover <= params["turnover_max"]
        ):
            continue
        rows.append(row)
    return rows[:universe_limit]


def extract_frame(download: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if download.empty:
        return pd.DataFrame()
    if isinstance(download.columns, pd.MultiIndex):
        first = download.columns.get_level_values(0)
        second = download.columns.get_level_values(1)
        if symbol in first:
            return download[symbol].dropna(how="all")
        if symbol in second:
            return download.xs(symbol, axis=1, level=1).dropna(how="all")
    return download.dropna(how="all")


def download_prices(tickers: list[str], period: str, interval: str) -> tuple[pd.DataFrame, list[str]]:
    collected: list[pd.DataFrame] = []
    failed: list[str] = []
    chunk_size = 80 if interval == "1d" else 25
    for index in range(0, len(tickers), chunk_size):
        chunk = tickers[index:index + chunk_size]
        try:
            data = yf.download(
                chunk,
                period=period,
                interval=interval,
                auto_adjust=False,
                progress=False,
                group_by="ticker",
                threads=True,
                timeout=20,
            )
            if data.empty:
                failed.extend(chunk)
            else:
                collected.append(data)
        except Exception:
            failed.extend(chunk)
    if not collected:
        return pd.DataFrame(), failed
    if len(collected) == 1:
        return collected[0], failed
    return pd.concat(collected, axis=1), failed


def fetch_shares_for_custom(tickers: list[str]) -> dict[str, float | None]:
    shares: dict[str, float | None] = {}

    def one(symbol: str) -> tuple[str, float | None]:
        try:
            return symbol, number(yf.Ticker(symbol).fast_info.get("shares"))
        except Exception:
            return symbol, None

    with ThreadPoolExecutor(max_workers=min(8, max(1, len(tickers)))) as pool:
        futures = [pool.submit(one, symbol) for symbol in tickers]
        for future in as_completed(futures):
            symbol, value = future.result()
            shares[symbol] = value
    return shares


def max_up_streak(close: pd.Series, lookback: int = 20) -> int:
    best = current = 0
    for is_up in (close.diff().iloc[-lookback:] > 0).tolist():
        current = current + 1 if is_up else 0
        best = max(best, current)
    return best


def linear_slope(values: pd.Series) -> float | None:
    clean = values.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 2 or clean.mean() == 0:
        return None
    return float(np.polyfit(np.arange(len(clean)), clean / clean.mean(), 1)[0])


def common_metrics(
    symbol: str,
    frame: pd.DataFrame,
    market: str,
    meta: dict[str, Any],
    benchmark_return_20d: float | None,
) -> dict[str, Any] | None:
    required = ["Open", "High", "Low", "Close", "Volume"]
    if frame.empty or any(column not in frame for column in required):
        return None
    frame = frame.dropna(subset=required).copy()
    if len(frame) < 70:
        return None
    open_ = frame["Open"].astype(float)
    high = frame["High"].astype(float)
    low = frame["Low"].astype(float)
    close = frame["Close"].astype(float)
    volume = frame["Volume"].astype(float)
    ret = close.pct_change()
    ma5_series = close.rolling(5).mean()
    ma10_series = close.rolling(10).mean()
    ma20_series = close.rolling(20).mean()
    ma30_series = close.rolling(30).mean()
    avg20_prev = volume.shift(1).rolling(20).mean().iloc[-1]
    volume_ratio = volume.iloc[-1] / avg20_prev if avg20_prev and avg20_prev > 0 else np.nan
    prior20_volume = volume.iloc[-25:-5].mean()
    recent5_vs_prior20 = volume.iloc[-5:].mean() / prior20_volume if prior20_volume > 0 else np.nan
    float_shares = number(meta.get("float_shares"))
    shares_basis = meta.get("shares_basis") if float_shares else None
    turnover = volume.iloc[-1] / float_shares * 100 if float_shares else None
    cumulative_turnover_20 = volume.iloc[-20:].sum() / float_shares * 100 if float_shares else None
    market_cap = number(meta.get("market_cap_basic"))
    if market_cap is None and float_shares:
        market_cap = close.iloc[-1] * float_shares
    strict_gaps = (low > high.shift(1)).iloc[-20:]
    recent5_volume = volume.iloc[-5:].mean()
    volume_cv_5 = volume.iloc[-5:].std(ddof=0) / recent5_volume if recent5_volume > 0 else np.nan
    volume_slope_3 = linear_slope(volume.iloc[-3:])
    return {
        "symbol": symbol,
        "name": meta.get("description") or symbol,
        "exchange": meta.get("exchange") or ("TW" if market == "TW" else "US"),
        "as_of": frame.index[-1].strftime("%Y-%m-%d"),
        "close": float(close.iloc[-1]),
        "day_change_pct": float(ret.iloc[-1] * 100),
        "volume": float(volume.iloc[-1]),
        "volume_ratio": float(volume_ratio),
        "recent5_vs_prior20_volume": float(recent5_vs_prior20),
        "volume_cv_5": float(volume_cv_5),
        "volume_slope_3": volume_slope_3,
        "ma5": float(ma5_series.iloc[-1]),
        "ma10": float(ma10_series.iloc[-1]),
        "ma20": float(ma20_series.iloc[-1]),
        "ma30": float(ma30_series.iloc[-1]),
        "bullish_ma": bool(close.iloc[-1] > ma5_series.iloc[-1] > ma10_series.iloc[-1] > ma20_series.iloc[-1] > ma30_series.iloc[-1]),
        "near_60d_high_pct": float((close.iloc[-1] / high.iloc[-60:].max() - 1) * 100),
        "turnover_pct": turnover,
        "cumulative_turnover_20d": cumulative_turnover_20,
        "shares_basis": shares_basis,
        "market_cap": market_cap,
        "benchmark_return_20d": benchmark_return_20d,
        "max_up_streak_20d": max_up_streak(close),
        "strict_gap_count_20d": int(strict_gaps.fillna(False).sum()),
        "max_gain_20d": float(ret.iloc[-20:].max() * 100),
        "sparkline": [round(float(value), 4) for value in close.iloc[-30:]],
        "_frame": frame,
    }


def intraday_metrics(symbol: str, frame: pd.DataFrame, benchmark: pd.DataFrame, market: str) -> dict[str, Any]:
    if frame.empty or benchmark.empty:
        return {"available": False}
    required = ["Open", "High", "Low", "Close", "Volume"]
    if any(column not in frame for column in required) or any(column not in benchmark for column in ["Open", "Close"]):
        return {"available": False}
    frame = frame.dropna(subset=required).copy()
    benchmark = benchmark.dropna(subset=["Open", "Close"]).copy()
    if frame.empty or benchmark.empty:
        return {"available": False}
    frame = frame[frame.index.date == frame.index[-1].date()]
    benchmark = benchmark[benchmark.index.date == benchmark.index[-1].date()]
    typical = (frame["High"] + frame["Low"] + frame["Close"]) / 3
    denominator = frame["Volume"].cumsum().replace(0, np.nan)
    vwap = (typical * frame["Volume"]).cumsum() / denominator
    valid = vwap.notna()
    above_ratio = float((frame.loc[valid, "Close"] >= vwap.loc[valid]).mean()) if valid.any() else None
    stock_return = float((frame["Close"].iloc[-1] / frame["Open"].iloc[0] - 1) * 100)
    benchmark_return = float((benchmark["Close"].iloc[-1] / benchmark["Open"].iloc[0] - 1) * 100)
    pullback = float((frame["Close"].iloc[-1] / frame["High"].max() - 1) * 100)
    minimum_bars = 45 if market == "TW" else 70
    return {
        "available": True,
        "bars": len(frame),
        "session_complete_proxy": len(frame) >= minimum_bars,
        "latest_bar": frame.index[-1].isoformat(),
        "above_vwap_ratio": above_ratio,
        "close_above_vwap": bool(frame["Close"].iloc[-1] >= vwap.iloc[-1]),
        "session_return_pct": stock_return,
        "benchmark_return_pct": benchmark_return,
        "relative_strength_pct": stock_return - benchmark_return,
        "close_from_high_pct": pullback,
    }


def strategy_washout(row: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    frame: pd.DataFrame = row["_frame"]
    close = frame["Close"].astype(float)
    volume = frame["Volume"].astype(float)
    ma = close.rolling(params["ma_days"]).mean()
    holds_ma = bool(close.iloc[-1] >= ma.iloc[-1] * (1 - params["ma_tolerance_pct"] / 100))
    volume_contraction = bool(row["recent5_vs_prior20_volume"] <= params["contraction_ratio"])
    market_ok = row["benchmark_return_20d"] is None or row["benchmark_return_20d"] >= params["benchmark_floor_pct"]
    turnover_ok = row["cumulative_turnover_20d"] is None or row["cumulative_turnover_20d"] <= params["cumulative_turnover_max"]
    distribution = bool(close.iloc[-1] < row["ma30"] * (1 - params["ma_tolerance_pct"] / 100) and row["volume_ratio"] >= params["distribution_volume_ratio"])

    rolling_base = volume.shift(5).rolling(20).mean()
    rolling_recent = volume.rolling(5).mean()
    prior_wash = ((close >= ma * (1 - params["ma_tolerance_pct"] / 100)) & (rolling_recent / rolling_base <= params["contraction_ratio"]))
    wash_lookback = max(10, min(63, int(params["wash_lookback_days"])))
    wash_seen = bool(prior_wash.iloc[-wash_lookback:-1].fillna(False).any())
    re_expanding = bool(row["volume_ratio"] >= params["reexpansion_ratio"] and close.iloc[-1] >= row["ma20"] and close.diff().iloc[-1] > 0)

    if distribution:
        classification = "疑似出貨風險"
    elif wash_seen and re_expanding and market_ok:
        classification = "拉盤前期代理訊號"
    elif holds_ma and volume_contraction and market_ok and turnover_ok and close.iloc[-1] >= row["ma20"] and close.diff().iloc[-1] > 0:
        classification = "洗盤可能接近結束"
    elif holds_ma and volume_contraction and market_ok and turnover_ok:
        classification = "疑似洗盤中"
    else:
        classification = "未形成明確型態"

    checks = [
        ("守住關鍵均線", holds_ma),
        ("近 5 日量能收斂", volume_contraction),
        ("20 日累積換手未超標", turnover_ok),
        ("市場環境未惡化", market_ok),
        (f"{wash_lookback} 交易日內曾出現洗盤型態", wash_seen),
        ("量價重新擴張", re_expanding),
    ]
    score = round(sum(passed for _, passed in checks) / len(checks) * 100)
    return {**row, "classification": classification, "score": score, "matched": classification != "未形成明確型態", "checks": checks}


def strategy_overnight(row: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    checks = [
        (f"漲幅 {params['change_min']:.1f}–{params['change_max']:.1f}%", params["change_min"] <= row["day_change_pct"] <= params["change_max"]),
        (f"量比 ≥ {params['volume_ratio_min']:.1f}", row["volume_ratio"] >= params["volume_ratio_min"]),
        (f"換手率 {params['turnover_min']:.1f}–{params['turnover_max']:.1f}%", row["turnover_pct"] is not None and params["turnover_min"] <= row["turnover_pct"] <= params["turnover_max"]),
        ("流通市值在範圍內", row["market_cap"] is not None and params["cap_min"] <= row["market_cap"] <= params["cap_max"]),
        ("成交量趨勢規律", row["volume_cv_5"] <= params["volume_cv_max"] and row["volume_slope_3"] is not None and row["volume_slope_3"] >= params["volume_slope_min"]),
        ("均線多頭且接近 60 日高點", row["bullish_ma"] and row["near_60d_high_pct"] >= -params["near_high_pct"]),
    ]
    score = round(sum(passed for _, passed in checks) / 8 * 100)
    return {**row, "score": score, "matched": False, "checks": checks, "classification": "待盤中條件"}


def apply_overnight_intraday(row: dict[str, Any], params: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    if metrics.get("available"):
        g_pass = bool(metrics["above_vwap_ratio"] >= params["above_vwap_ratio_min"] and metrics["relative_strength_pct"] > 0)
        h_pass = bool(metrics["close_above_vwap"] and -params["tail_pullback_max_pct"] <= metrics["close_from_high_pct"] <= 0)
    else:
        g_pass = h_pass = False
    checks = list(row["checks"]) + [
        ("大部分時間在 VWAP 上且強於大盤", g_pass),
        ("尾盤回踩仍守住 VWAP", h_pass),
    ]
    score = round(sum(passed for _, passed in checks) / len(checks) * 100)
    matched = all(passed for _, passed in checks)
    if matched:
        classification = "完整符合"
    elif score >= 75:
        classification = "接近符合"
    else:
        classification = "條件不足"
    return {**row, "intraday": metrics, "checks": checks, "score": score, "matched": matched, "classification": classification}


def strategy_pre_surge(row: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    frame: pd.DataFrame = row["_frame"]
    volume = frame["Volume"].astype(float)
    expanded_days = 0
    ratios: list[float] = []
    for index in range(max(20, len(volume) - params["volume_recent_days"]), len(volume)):
        baseline = volume.iloc[max(0, index - 20):index].mean()
        ratio = volume.iloc[index] / baseline if baseline else np.nan
        ratios.append(float(ratio))
        if params["volume_expand_min"] <= ratio <= params["volume_expand_max"]:
            expanded_days += 1
    checks = [
        (f"20 日內單日漲幅 ≥ {params['surge_threshold']:.1f}%", row["max_gain_20d"] >= params["surge_threshold"]),
        (f"20 日內至少連漲 {params['up_streak_min']} 天", row["max_up_streak_20d"] >= params["up_streak_min"]),
        ("20 日內出現向上跳空缺口", row["strict_gap_count_20d"] >= 1),
        (f"近 {params['volume_recent_days']} 日至少 {params['volume_expand_days']} 日量能為基準的 {params['volume_expand_min']:.1f}–{params['volume_expand_max']:.1f} 倍", expanded_days >= params["volume_expand_days"]),
    ]
    feature_count = sum(passed for _, passed in checks)
    matched = feature_count >= params["features_required"]
    long_streak_bonus = row["max_up_streak_20d"] >= 8
    weak_market_bonus = row["max_up_streak_20d"] >= params["up_streak_min"] and row["benchmark_return_20d"] is not None and row["benchmark_return_20d"] <= 0
    bonus_labels = []
    if long_streak_bonus:
        bonus_labels.append("8 日以上連漲")
    if weak_market_bonus:
        bonus_labels.append("弱市連漲")
    classification = f"符合 {feature_count}/4 特徵" if matched else f"僅符合 {feature_count}/4"
    if bonus_labels:
        classification += "・" + "／".join(bonus_labels)
    return {
        **row,
        "expanded_volume_days": expanded_days,
        "recent_volume_ratios": ratios,
        "feature_count": feature_count,
        "score": feature_count * 25,
        "priority_score": feature_count * 100 + (20 if long_streak_bonus else 0) + (10 if weak_market_bonus else 0),
        "bonus_labels": bonus_labels,
        "matched": matched,
        "classification": classification,
        "checks": checks,
    }


def strip_private(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def default_params(market: str, strategy: str, supplied: dict[str, Any]) -> dict[str, Any]:
    settings = MARKETS[market]
    base: dict[str, Any]
    if strategy == "washout":
        base = {
            "ma_days": 30,
            "wash_lookback_days": 63,
            "ma_tolerance_pct": 1.0,
            "contraction_ratio": 0.65,
            "cumulative_turnover_max": 30.0,
            "benchmark_floor_pct": -5.0,
            "distribution_volume_ratio": 1.3,
            "reexpansion_ratio": 1.0,
        }
    elif strategy == "overnight":
        base = {
            "change_min": 3.0,
            "change_max": 5.0,
            "volume_ratio_min": 1.0,
            "turnover_min": 5.0,
            "turnover_max": 10.0,
            "cap_min": settings["cap_min"],
            "cap_max": settings["cap_max"],
            "volume_cv_max": 0.8,
            "volume_slope_min": 0.0,
            "near_high_pct": 5.0,
            "above_vwap_ratio_min": 0.8,
            "tail_pullback_max_pct": 2.5,
        }
    else:
        base = {
            "surge_threshold": settings["surge_threshold"],
            "up_streak_min": 4,
            "volume_recent_days": 5,
            "volume_expand_days": 2,
            "volume_expand_min": 1.0,
            "volume_expand_max": 2.0,
            "features_required": 3,
        }
    for key in base:
        if key in supplied:
            try:
                base[key] = int(supplied[key]) if isinstance(base[key], int) else float(supplied[key])
            except (TypeError, ValueError):
                pass
    return base


def run_screen(payload: dict[str, Any]) -> dict[str, Any]:
    started = time.time()
    market = str(payload.get("market", "TW")).upper()
    strategy = str(payload.get("strategy", "washout"))
    universe_mode = str(payload.get("universe_mode", "market"))
    if market not in MARKETS:
        raise ValueError("market 必須是 TW 或 US")
    if strategy not in RULE_LABELS:
        raise ValueError("未知的策略")
    params = default_params(market, strategy, payload.get("params") or {})
    universe_limit = clamp_int(payload.get("universe_limit"), 20, 1000, 200)
    detail_limit = clamp_int(payload.get("detail_limit"), 10, 200, 60)
    use_intraday = bool(payload.get("use_intraday", True))
    scanner_rows: list[dict[str, Any]] = []
    warnings: list[str] = []

    if universe_mode == "custom":
        tickers = parse_tickers(str(payload.get("tickers", "")), market)
        if not tickers:
            tickers = DEFAULT_TICKERS[market]
            warnings.append("未輸入代碼，已改用內建示範清單。")
        tickers = tickers[:200]
        shares = fetch_shares_for_custom(tickers)
        meta_map = {
            symbol: {
                "description": symbol,
                "exchange": "TW" if market == "TW" else "US",
                "float_shares": shares.get(symbol),
                "shares_basis": "總股本代理",
                "market_cap_basic": None,
            }
            for symbol in tickers
        }
        if any(shares.get(symbol) for symbol in tickers):
            warnings.append("自訂清單的換手率以總股本代理；全市場模式優先使用流通股本。")
    else:
        scanner_rows = tradingview_scan(market, strategy, universe_limit, params)
        tickers = [row["symbol"] for row in scanner_rows]
        meta_map = {row["symbol"]: row for row in scanner_rows}
        if not tickers:
            return empty_result(market, strategy, params, universe_mode, started, "廣域初篩沒有候選股。")

    benchmark = MARKETS[market]["benchmark"]
    download, failed_chunks = download_prices(tickers + [benchmark], period="6mo", interval="1d")
    benchmark_frame = extract_frame(download, benchmark)
    benchmark_return_20d = None
    if len(benchmark_frame) >= 21:
        benchmark_return_20d = float((benchmark_frame["Close"].iloc[-1] / benchmark_frame["Close"].iloc[-21] - 1) * 100)

    rows: list[dict[str, Any]] = []
    unavailable: list[str] = []
    for symbol in tickers:
        metrics = common_metrics(symbol, extract_frame(download, symbol), market, meta_map.get(symbol, {}), benchmark_return_20d)
        if metrics is None:
            unavailable.append(symbol)
            continue
        if strategy == "washout":
            rows.append(strategy_washout(metrics, params))
        elif strategy == "overnight":
            rows.append(strategy_overnight(metrics, params))
        else:
            rows.append(strategy_pre_surge(metrics, params))

    if strategy == "overnight" and rows:
        rows.sort(key=lambda item: item["score"], reverse=True)
        intraday_targets = [row["symbol"] for row in rows[: min(20, detail_limit)]] if use_intraday else []
        intraday_download = pd.DataFrame()
        if intraday_targets:
            intraday_download, _ = download_prices(intraday_targets + [benchmark], period="5d", interval="5m")
        benchmark_intraday = extract_frame(intraday_download, benchmark)
        completed = []
        for row in rows:
            if row["symbol"] in intraday_targets:
                metrics = intraday_metrics(row["symbol"], extract_frame(intraday_download, row["symbol"]), benchmark_intraday, market)
            else:
                metrics = {"available": False}
            completed.append(apply_overnight_intraday(row, params, metrics))
        rows = completed

    rows.sort(key=lambda item: (bool(item["matched"]), item.get("priority_score", item["score"]), item["volume_ratio"]), reverse=True)
    displayed = rows[:detail_limit]
    matched_count = sum(bool(row["matched"]) for row in rows)
    if unavailable:
        warnings.append(f"{len(unavailable)} 檔因歷史資料不足或代碼無效而跳過。")
    if failed_chunks:
        warnings.append("部分批次下載曾失敗；已保留成功取得的股票。")
    if market_session_is_partial(market):
        warnings.append("目前仍在盤中，當日成交量、漲幅與換手率會隨行情變動。")
    if strategy == "washout":
        warnings.append("『主力持股 30–50%』與真實大單托底需要籌碼／Level 2 資料，本工具不會用價格型態冒充這兩項證據。")
    if strategy == "pre_surge" and market == "US":
        warnings.append("美股沒有台股式每日漲停；此處以可調的單日大漲門檻取代『20 日內漲停』。")

    return json_safe({
        "ok": True,
        "market": market,
        "market_label": MARKETS[market]["label"],
        "currency": MARKETS[market]["currency"],
        "strategy": strategy,
        "strategy_label": RULE_LABELS[strategy],
        "generated_at": datetime.now().astimezone().isoformat(),
        "elapsed_seconds": round(time.time() - started, 2),
        "universe_mode": universe_mode,
        "universe_count": len(tickers),
        "analyzed_count": len(rows),
        "matched_count": matched_count,
        "displayed_count": len(displayed),
        "benchmark": benchmark,
        "benchmark_return_20d": benchmark_return_20d,
        "daily_bar_partial": market_session_is_partial(market),
        "params": params,
        "warnings": warnings,
        "rows": [strip_private(row) for row in displayed],
        "sources": [
            {
                "name": "TradingView Scanner",
                "role": "全市場即時／延遲快照與流通股本初篩" if universe_mode == "market" else "自訂清單模式未使用",
                "url": f"https://www.tradingview.com/markets/stocks-{MARKETS[market]['scanner']}/market-movers-active/",
                "used": universe_mode == "market",
            },
            {
                "name": "Yahoo Finance（透過 yfinance）",
                "role": "6 個月日線、5 分鐘線與大盤基準",
                "url": "https://finance.yahoo.com/",
                "used": True,
            },
        ],
    })


def empty_result(market: str, strategy: str, params: dict[str, Any], universe_mode: str, started: float, message: str) -> dict[str, Any]:
    return {
        "ok": True,
        "market": market,
        "market_label": MARKETS[market]["label"],
        "currency": MARKETS[market]["currency"],
        "strategy": strategy,
        "strategy_label": RULE_LABELS[strategy],
        "generated_at": datetime.now().astimezone().isoformat(),
        "elapsed_seconds": round(time.time() - started, 2),
        "universe_mode": universe_mode,
        "universe_count": 0,
        "analyzed_count": 0,
        "matched_count": 0,
        "displayed_count": 0,
        "params": params,
        "warnings": [message],
        "rows": [],
        "sources": [],
    }


class ScreenerHandler(SimpleHTTPRequestHandler):
    server_version = "TWUSStockScreener/1.0"

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {format % args}")

    def send_json(self, data: Any, status: int = 200) -> None:
        body = json.dumps(json_safe(data), ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_security_headers()
        self.end_headers()
        self.wfile.write(body)

    def send_security_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "SAMEORIGIN")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; connect-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; base-uri 'none'; form-action 'self'; frame-ancestors 'self'",
        )

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/health":
            self.send_json({"ok": True, "service": "台美股策略篩選器", "time": datetime.now().astimezone().isoformat()})
            return
        if parsed.path == "/api/config":
            public_markets = {
                key: {field: value for field, value in settings.items() if field not in {"market_open", "market_close"}}
                for key, settings in MARKETS.items()
            }
            self.send_json({"markets": public_markets, "strategies": RULE_LABELS, "default_tickers": DEFAULT_TICKERS})
            return
        path = unquote(parsed.path.lstrip("/")) or "index.html"
        target = (WEB_ROOT / path).resolve()
        if WEB_ROOT.resolve() not in target.parents and target != WEB_ROOT.resolve():
            self.send_error(403)
            return
        if not target.is_file():
            target = WEB_ROOT / "index.html"
        try:
            content = target.read_bytes()
        except OSError:
            self.send_error(404)
            return
        mime = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", f"{mime}; charset=utf-8" if mime.startswith(("text/", "application/javascript")) else mime)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-cache")
        self.send_security_headers()
        self.end_headers()
        self.wfile.write(content)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/screen":
            self.send_error(404)
            return
        try:
            length = clamp_int(self.headers.get("Content-Length"), 0, 1_000_000, 0)
            payload = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(payload, dict):
                raise ValueError("請求格式必須是 JSON 物件")
            cached = cached_screen(payload)
            if cached is not None:
                self.send_json(cached)
                return
            forwarded_for = self.headers.get("X-Forwarded-For", "").split(",", 1)[0].strip()
            client_key = forwarded_for or self.client_address[0]
            allowed, retry_after = rate_limit_status(client_key)
            if not allowed:
                self.send_response(429)
                self.send_header("Retry-After", str(retry_after))
                body = json.dumps({"ok": False, "error": "操作過於頻繁，請稍後再試。"}, ensure_ascii=False).encode("utf-8")
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_security_headers()
                self.end_headers()
                self.wfile.write(body)
                return
            if not SCREEN_SEMAPHORE.acquire(timeout=2):
                self.send_json({"ok": False, "error": "目前同時使用人數較多，請稍後再試。"}, 503)
                return
            try:
                result = store_screen_cache(payload, run_screen(payload))
            finally:
                SCREEN_SEMAPHORE.release()
            self.send_json(result)
        except ValueError as exc:
            self.send_json({"ok": False, "error": str(exc)}, 400)
        except Exception as exc:
            traceback.print_exc()
            self.send_json({"ok": False, "error": f"篩選失敗：{exc}"}, 500)


def main() -> None:
    parser = argparse.ArgumentParser(description="台美股三策略篩選器")
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", os.environ.get("STOCK_SCREENER_PORT", "8765"))))
    parser.add_argument("--open", action="store_true", help="啟動後用預設瀏覽器開啟")
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), ScreenerHandler)
    server.daemon_threads = True
    display_host = "127.0.0.1" if args.host == "0.0.0.0" else args.host
    url = f"http://{display_host}:{args.port}/"
    print(f"台美股策略篩選器已啟動：{url}")
    print("關閉視窗或按 Control-C 即可停止。")
    if args.open:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
