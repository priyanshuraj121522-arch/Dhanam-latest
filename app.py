import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from utils.data import (
    infer_market, download_prices, latest_prices, fetch_benchmark, get_fx_series,
    fetch_sector_for_tickers, is_india_ticker, fetch_series_for
)
from utils.risk import (
    value_from_positions, weights, daily_returns, portfolio_series, drawdown,
    herfindahl_hirschman_index, compute_correlation, risk_score, scenario_impact,
    variance_contributions, sharpe_ratio, tracking_error
)
from utils import visuals
from utils.report import build_pdf

BASE_DIR = Path(__file__).parent

st.set_page_config(page_title="DHANAM — Bloomberg Dark", page_icon="💹", layout="wide")
with open(BASE_DIR / "static" / "styles.css") as f: st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Header
left, mid, right = st.columns([0.12,0.58,0.30])
with left: st.image(str((BASE_DIR / "assets" / "logo.svg").resolve()))
with mid: st.markdown("### **DHANAM — Smart Portfolio Risk Suite**")
with right: base_ccy = st.selectbox("Base Currency", ["INR","USD"], index=0)

st.caption("India + USA portfolio analytics • Bloomberg-style dark theme")
st.markdown("---")

# Sidebar
st.sidebar.markdown("## Portfolio")
up = st.sidebar.file_uploader("Upload CSV", type=["csv"])
if up is not None: positions = pd.read_csv(up)
else:
    if st.sidebar.button("Load sample"): positions = pd.read_csv((BASE_DIR / "sample_portfolio.csv").resolve())
    else: positions = pd.DataFrame(columns=["exchange","symbol","quantity","avg_price","sector"])

if st.sidebar.checkbox("Edit in app", value=False): positions = st.sidebar.data_editor(positions, num_rows="dynamic")
if positions.empty or positions["symbol"].dropna().empty:
    st.info("Upload a CSV with `symbol, quantity, [avg_price], [sector]`. Use .NS/.BO for India tickers."); st.stop()

period = st.sidebar.selectbox("History", ["6mo","1y","2y","5y"], index=1)

# Data
prices = download_prices(positions["symbol"].tolist(), period=period)
if prices.empty: st.error("Could not fetch prices; check symbols or try again."); st.stop()
bench_market = infer_market(positions["symbol"].tolist())
bench = fetch_benchmark("IN" if bench_market in ("IN","MIX") and base_ccy=="INR" else "US", period=period)
bench_r = bench.pct_change().dropna()
fx = get_fx_series(period=period); fx_latest = float(fx.ffill().iloc[-1]) if not fx.empty else None
latest = latest_prices(prices); values = value_from_positions(latest, positions, fx_latest, base_ccy); w = weights(values)

# Tabs
tab_dash, tab_scen, tab_factors, tab_reb, tab_watch, tab_report = st.tabs(
    ["Dashboard","Scenario Lab","Factor Exposures","Rebalance Ideas","Watchlist","Reports"]
)

with tab_dash:
    rets = daily_returns(prices); port_r = portfolio_series(rets, w)
    vol_d = float(port_r.std()); idx = port_r.index.intersection(bench_r.index)
    beta = float(np.cov(port_r.loc[idx], bench_r.loc[idx])[0,1] / np.var(bench_r.loc[idx])) if len(idx)>5 and np.var(bench_r.loc[idx])>0 else 1.0
    dd = float(drawdown(port_r).min()); hhi = float(herfindahl_hirschman_index(w))
    score = risk_score({"volatility": vol_d, "beta": beta, "max_drawdown": dd, "hhi": hhi})
    sharpe = sharpe_ratio(port_r); te = tracking_error(port_r, bench_r)

    c1,c2,c3,c4 = st.columns(4)
    with c1: st.markdown('<div class="d-card">', unsafe_allow_html=True); st.metric("Portfolio Value", f"{float(values.sum()):,.2f} {base_ccy}"); st.markdown('</div>', unsafe_allow_html=True)
    with c2: st.markdown('<div class="d-card">', unsafe_allow_html=True); st.metric("Risk Score (0–100)", score); st.pyplot(visuals.risk_gauge(score), use_container_width=True); st.markdown('</div>', unsafe_allow_html=True)
    with c3: st.markdown('<div class="d-card">', unsafe_allow_html=True); st.metric("Sharpe (ann.)", f"{sharpe:.2f}"); st.markdown('</div>', unsafe_allow_html=True)
    with c4: st.markdown('<div class="d-card">', unsafe_allow_html=True); st.metric("Tracking Error (ann.)", f"{te:.2f}%"); st.markdown('</div>', unsafe_allow_html=True)

    a,b = st.columns([0.5,0.5])
    with a:
        pos = positions.copy(); pos["value"] = values
        by_sector = pos.groupby(pos["sector"].replace({np.nan:"Unknown"}))["value"].sum(); w_sector = (by_sector/by_sector.sum()).fillna(0.0)
        st.markdown('<div class="d-card">', unsafe_allow_html=True); st.pyplot(visuals.donut_series(w_sector, "Exposure by Sector"), use_container_width=True); st.markdown('</div>', unsafe_allow_html=True)
    with b:
        corr = compute_correlation(rets); st.markdown('<div class="d-card">', unsafe_allow_html=True); st.pyplot(visuals.heatmap_corr(corr, "Correlation Heatmap"), use_container_width=True); st.markdown('</div>', unsafe_allow_html=True)

    p1,p2 = st.columns([0.64,0.36])
    with p1:
        perf = pd.DataFrame({"Portfolio":(1+port_r).cumprod(),"Benchmark":(1+bench_r.reindex_like(port_r).fillna(0)).cumprod()}).dropna()
        st.markdown('<div class="d-card">', unsafe_allow_html=True); st.pyplot(visuals.line_series(perf, "Performance vs Benchmark"), use_container_width=True); st.markdown('</div>', unsafe_allow_html=True)
    with p2:
        tbl = positions.copy(); last_conv = []
        for s in positions["symbol"]:
            px = latest.get(s, np.nan)
            if base_ccy=="INR" and not is_india_ticker(s): px*=fx_latest
            if base_ccy=="USD" and is_india_ticker(s): px/=fx_latest if fx_latest else np.nan
            last_conv.append(px)
        tbl["last_price_base"] = last_conv; tbl["value"] = values; tbl["weight_%"] = (w*100).round(2)
        if "avg_price" in tbl.columns and tbl["avg_price"].notna().any():
            ap = []
            for s,avg in zip(positions["symbol"], positions.get("avg_price", [])):
                avgp = avg
                if base_ccy=="INR" and not is_india_ticker(s): avgp = (avg or np.nan) * fx_latest if fx_latest else np.nan
                if base_ccy=="USD" and is_india_ticker(s): avgp = (avg or np.nan) / fx_latest if fx_latest else np.nan
                ap.append(avgp)
            tbl["avg_price_base"] = ap; tbl["pnl_%"] = ((tbl["last_price_base"]-tbl["avg_price_base"])/tbl["avg_price_base"]*100).round(2)
        st.markdown('<div class="d-card">', unsafe_allow_html=True); st.markdown("**Holdings**"); st.dataframe(tbl, use_container_width=True); st.markdown('</div>', unsafe_allow_html=True)

with tab_scen:
    st.markdown('<div class="d-card">', unsafe_allow_html=True)
    st.markdown("### Scenario Lab")
    pos = positions.copy(); pos["value"] = values
    by_sector = pos.groupby(pos["sector"].replace({np.nan:"Unknown"}))["value"].sum(); w_sector = (by_sector/by_sector.sum()).fillna(0)
    c1,c2 = st.columns(2)
    with c1: scen = st.selectbox("Preset", ["Oil +10%","Rate +25 bps","Rate -25 bps","USDINR +2%"], index=0)
    with c2: mag = st.slider("Magnitude (x)", 0.5, 2.0, 1.0, 0.1)
    if scen=="USDINR +2%":
        us_weight = float(pos[~pos['symbol'].str.upper().str.endswith(('.NS','.BO'))]["value"].sum() / pos["value"].sum())
        in_weight = 1 - us_weight; impact = (us_weight if base_ccy=="INR" else -in_weight) * 2.0 * mag
    else: impact = scenario_impact(w_sector, scen, magnitude=mag)
    st.metric("Estimated Portfolio Impact", f"{impact:+.2f}%")
    st.markdown('</div>', unsafe_allow_html=True)

with tab_factors:
    st.markdown('<div class="d-card">', unsafe_allow_html=True)
    st.markdown("### Factor Exposures")
    factor_symbols = ["SPY","QQQ","XLF","XLE","XLK","XLV","IWM"]
    factor_prices = fetch_series_for(factor_symbols, period=period)
    rets = daily_returns(prices); port_r = portfolio_series(rets, w)
    if not factor_prices.empty:
        X = factor_prices.pct_change().dropna(); y = port_r.reindex(X.index).dropna(); X = X.reindex(y.index)
        if len(X) >= 25 and y.std(ddof=0)>0:
            Xn = (X - X.mean())/X.std(ddof=0); yn = (y - y.mean())/y.std(ddof=0)
            import numpy as np
            beta = np.linalg.lstsq(np.c_[np.ones(len(Xn)), Xn.values], yn.values, rcond=None)[0]
            data = {"alpha": float(beta[0])}; 
            for i, c in enumerate(Xn.columns, start=1): data[c] = float(beta[i])
            st.dataframe(pd.DataFrame([data]).T.rename(columns={0:"beta"}), use_container_width=True)
        else: st.info("Not enough data to compute betas.")
    else: st.info("Could not load factor data right now.")
    st.markdown('</div>', unsafe_allow_html=True)

with tab_reb:
    st.markdown('<div class="d-card">', unsafe_allow_html=True)
    st.markdown("### Rebalance Ideas")
    rets = daily_returns(prices); port_r = portfolio_series(rets, w)
    contrib = variance_contributions(rets, w).sort_values(ascending=False)
    st.markdown("**Top variance contributors (positions):**")
    st.dataframe(contrib.to_frame("risk_contribution").head(10), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with tab_watch:
    st.markdown('<div class="d-card">', unsafe_allow_html=True)
    st.markdown("### Watchlist")
    syms = st.text_input("Enter tickers (.NS/.BO for India):", "RELIANCE.NS, TCS.NS, AAPL, MSFT")
    if st.button("Fetch Watchlist"):
        lst = [s.strip() for s in syms.split(",") if s.strip()]
        wp = download_prices(lst, period="6mo")
        if wp.empty: st.warning("No data for watchlist.")
        else:
            last = wp.ffill().iloc[-1]; chg = wp.ffill().pct_change().iloc[-1]*100
            st.dataframe(pd.DataFrame({"Last":last,"1D %":chg.round(2)}), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with tab_report:
    st.markdown('<div class="d-card">', unsafe_allow_html=True)
    st.markdown("### Reports & Exports")
    rets = daily_returns(prices); port_r = portfolio_series(rets, w)
    pos = positions.copy(); pos["value"] = values
    by_sector = pos.groupby(pos["sector"].replace({np.nan:"Unknown"}))["value"].sum(); w_sector = (by_sector/by_sector.sum()).fillna(0.0)
    fig1 = visuals.donut_series(w_sector, "Exposure by Sector"); corr = compute_correlation(rets)
    fig2 = visuals.heatmap_corr(corr, "Correlation Heatmap")
    perf = pd.DataFrame({"Portfolio":(1+port_r).cumprod(),"Benchmark":(1+bench_r.reindex_like(port_r).fillna(0)).cumprod()}).dropna()
    fig3 = visuals.line_series(perf, "Performance vs Benchmark")
    img1, img2, img3 = "sector.png","corr.png","perf.png"
    fig1.savefig(img1, bbox_inches="tight"); fig2.savefig(img2, bbox_inches="tight"); fig3.savefig(img3, bbox_inches="tight")
    metrics = {
        "Portfolio Value": f"{float(values.sum()):,.2f} {base_ccy}",
        "Risk Score": f"{score:.1f}",
        "Sharpe (ann.)": f"{sharpe:.2f}",
        "Tracking Error (ann.)": f"{te:.2f}%",
    }
    try:
        pdf_path = "dhanam_report.pdf"
        build_pdf(pdf_path, "DHANAM Risk Report", metrics, [("Exposure by Sector", img1),("Correlation Heatmap", img2),("Performance vs Benchmark", img3)])
        with open(pdf_path,"rb") as f: st.download_button("Download PDF Risk Report", f, file_name="dhanam_report.pdf")
    except Exception:
        st.info("PDF generation requires reportlab. If it failed to install, remove it from requirements.")
    st.markdown('</div>', unsafe_allow_html=True)
