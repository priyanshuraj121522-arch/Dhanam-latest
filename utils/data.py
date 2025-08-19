import time
from typing import List, Dict
import pandas as pd
import requests

BENCHMARKS = {"IN": "^NSEI", "US": "^GSPC"}

def is_india_ticker(t: str) -> bool:
    t = (t or "").upper()
    return t.endswith(".NS") or t.endswith(".BO")

def infer_market(tickers: List[str]) -> str:
    tickers = [str(t).strip() for t in tickers if str(t).strip()]
    has_in = any(is_india_ticker(t) for t in tickers)
    has_us = any(not is_india_ticker(t) for t in tickers)
    if has_in and has_us: return "MIX"
    return "IN" if has_in else "US"

def _fetch_yahoo_chart(symbol: str, period: str = "1y", interval: str = "1d") -> pd.Series:
    headers = {"User-Agent": "Mozilla/5.0"}
    params = {"range": period, "interval": interval, "includeAdjustedClose": "true"}
    for host in ("query1", "query2"):
        url = f"https://{host}.finance.yahoo.com/v7/finance/chart/{symbol}"
        try:
            r = requests.get(url, params=params, headers=headers, timeout=12)
            if r.status_code != 200:
                continue
            j = r.json()
            result = j.get("chart", {}).get("result", [])
            if not result: 
                continue
            res0 = result[0]
            ts = res0.get("timestamp", [])
            ind = res0.get("indicators", {})
            adj = None
            if "adjclose" in ind and ind["adjclose"] and "adjclose" in ind["adjclose"][0]:
                adj = ind["adjclose"][0]["adjclose"]
            elif "quote" in ind and ind["quote"] and "close" in ind["quote"][0]:
                adj = ind["quote"][0]["close"]
            if not ts or not adj:
                continue
            idx = pd.to_datetime(ts, unit="s", utc=True).tz_convert(None)
            s = pd.Series(adj, index=idx, name=symbol).dropna()
            return s
        except Exception:
            continue
    return pd.Series(dtype="float64", name=symbol)

def download_prices(tickers: List[str], period: str = "1y") -> pd.DataFrame:
    tickers = list(dict.fromkeys([str(t).strip() for t in tickers if str(t).strip()]))
    if not tickers: return pd.DataFrame()
    frames = []
    for t in tickers:
        s = _fetch_yahoo_chart(t, period=period, interval="1d")
        if not s.empty: frames.append(s.rename(t))
        time.sleep(0.2)
    if not frames: return pd.DataFrame()
    return pd.concat(frames, axis=1).dropna(how="all")

def latest_prices(prices: pd.DataFrame) -> pd.Series:
    return prices.ffill().iloc[-1]

def fetch_benchmark(market: str, period: str = "1y") -> pd.Series:
    sym = BENCHMARKS.get(market, "^GSPC")
    s = _fetch_yahoo_chart(sym, period=period, interval="1d")
    if s.empty and sym != "^GSPC":
        s = _fetch_yahoo_chart("^GSPC", period=period, interval="1d")
    return s

def get_fx_series(period: str = "1y") -> pd.Series:
    return _fetch_yahoo_chart("USDINR=X", period=period, interval="1d")

def fetch_series_for(symbols: List[str], period: str = "1y") -> pd.DataFrame:
    frames = []
    for s in symbols:
        ss = _fetch_yahoo_chart(s, period=period, interval="1d")
        if not ss.empty: frames.append(ss.rename(s))
        time.sleep(0.2)
    return pd.concat(frames, axis=1).dropna(how="all") if frames else pd.DataFrame()

def fetch_sector_for_tickers(tickers: List[str]) -> Dict[str, str]:
    return {t: "Unknown" for t in tickers}
