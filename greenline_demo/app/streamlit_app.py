
import json, math
from pathlib import Path
from typing import Tuple, List, Dict

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from graphviz import Digraph

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
ASSETS = HERE / "assets"

ABS_LOGO = Path("/Users/apepper23/greenline_hr_data_compass_demo/greenline_demo/app/assets/Greenline_logo.png")
REL_LOGO = ASSETS / "Greenline_logo.png"
LOGO_PATH = ABS_LOGO if ABS_LOGO.exists() else REL_LOGO

COKE_RED = "#F40009"
COKE_BLACK = "#1E1E1E"

st.set_page_config(
    page_title="GreenLine HR Data Compass — Azure Edition",
    page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else "🧭",
    layout="wide",
)

def to_jsonable(obj):
    """Recursively convert pandas/NumPy objects into JSON-serializable types."""
    import numpy as _np
    import pandas as _pd
    if isinstance(obj, _pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, (_np.integer, )):
        return int(obj)
    if isinstance(obj, (_np.floating, )):
        return float(obj)
    if isinstance(obj, (_np.ndarray, )):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj

def altair_theme(dark: bool):
    return {
        "config": {
            "background": "transparent",
            "axis": {
                "labelFontSize": 12,
                "titleFontSize": 12,
                "labelColor": "#EAEAEA" if dark else "#1E1E1E",
                "titleColor": "#EAEAEA" if dark else "#1E1E1E",
                "gridColor": "#333333"  if dark else "#E6E6E6",
                "domainColor": "#666666" if dark else "#BDBDBD",
            },
            "legend": {
                "labelColor": "#EAEAEA" if dark else "#1E1E1E",
                "titleColor": "#EAEAEA" if dark else "#1E1E1E",
            },
            "title": {"color": "#EAEAEA" if dark else "#1E1E1E", "fontSize": 14},
            "view": {"stroke": "transparent"},
        }
    }

def inject_brand_css(dark: bool) -> None:
    if dark:
        bg, panel, text, subtext, border, metric_label = "#0E0E0E", "#151515", "#EAEAEA", "#C9C9C9", "#2A2A2A", "#BDBDBD"
    else:
        bg, panel, text, subtext, border, metric_label = "#FFFFFF", "#FFFFFF", "#1E1E1E", "#464646", "#EAEAEA", "#555555"
    st.markdown(
        f"""
        <style>
          :root {{
            --coke-red:{COKE_RED}; --text:{text}; --subtext:{subtext};
            --bg:{bg}; --panel:{panel}; --border:{border};
          }}
          html, body, [data-testid="stAppViewContainer"] {{
            color: var(--text); background: var(--bg);
            font-family: "Segoe UI", system-ui, -apple-system, Arial, sans-serif;
          }}
          [data-testid="stSidebar"] {{
            background: var(--panel); color: var(--text); border-right: 1px solid var(--border);
          }}
          .cc-banner {{
            background: var(--coke-red); color: #fff; border-radius: 10px; padding: 10px 14px;
            display:flex; gap:10px; align-items:center; box-shadow: 0 2px 10px rgba(0,0,0,0.25); margin-bottom: 10px;
          }}
          .cc-title {{ font-size: 1.25rem; font-weight: 700; letter-spacing: .2px; line-height: 1.1; }}
          .cc-pill {{ background: rgba(255,255,255,.18); padding: 3px 8px; border-radius: 999px; font-weight: 700; font-size: .75rem; }}
          .stButton>button {{ background: var(--coke-red); color: #fff; border: 0; border-radius: 10px; font-weight: 700; }}
          [data-testid="stMetric"] {{ background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 6px 10px; }}
          [data-testid="stMetric"] label {{ color: {metric_label} !important; font-size: 0.80rem !important; }}
          [data-testid="stMetricValue"] {{ font-size: 1.4rem !important; color: var(--text) !important; }}
          [data-testid="stMetricDelta"] {{ font-size: 0.85rem !important; }}
          h1,.stMarkdown h1{{font-size:1.5rem}} h2,.stMarkdown h2{{font-size:1.2rem}} h3,.stMarkdown h3{{font-size:1.05rem}}
          p,.stMarkdown p,.stCaption{{font-size:.95rem}}
        </style>
        """,
        unsafe_allow_html=True,
    )

dark_mode = st.session_state.get("dark_mode_toggle", True)
inject_brand_css(dark_mode)

col_logo, col_banner = st.columns([1, 6], gap="small")
with col_logo:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), use_column_width=True)
with col_banner:
    st.markdown(
        """
        <div class="cc-banner">
          <div class="cc-title">GreenLine HR Data Compass</div>
          <span class="cc-pill">Azure Edition</span>
          <span class="cc-pill">Workday → Synapse → Power BI</span>
          <span class="cc-pill">Global BUs • Compliance • Assurance</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
st.caption("Global HR sustainability reporting (SEC / EU CSRD / GRI / CbCR) with governance, trends, sandbox targets, Monte Carlo forecasts, and third-party assurance.")

@st.cache_data
def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ts = pd.read_csv(DATA / "metrics_timeseries.csv", parse_dates=["date"])
    mapping = pd.read_csv(DATA / "standards_mapping.csv")
    comp = pd.read_csv(DATA / "compliance_readiness.csv")
    mat = pd.read_csv(DATA / "materiality.csv")
    lin = pd.read_csv(DATA / "lineage_edges.csv")
    return ts, mapping, comp, mat, lin

ts, mapping, comp, mat, lin = load_data()

if "DEI_Representation" in mapping["metric_id"].tolist():
    mapping = mapping[mapping["metric_id"] != "DEI_Representation"].reset_index(drop=True)

BUS = ["APAC", "EMEA", "LATAM", "NORTH AMERICA"]
REG_WEIGHTS = {"APAC": 1.00, "EMEA": 1.02, "LATAM": 0.98, "NORTH AMERICA": 1.01}

def regionize(df: pd.DataFrame) -> pd.DataFrame:
    """If no region column, synthesize per-region series with small perturbations."""
    if "region" in df.columns:
        out = df.copy()
        out["date"] = pd.to_datetime(out["date"])
        return out
    rows = []
    value_cols = [c for c in df.columns if c not in ("date","region")]
    for _, r in df.iterrows():
        for region, w in REG_WEIGHTS.items():
            nr = r.copy(); nr["region"] = region
            for col in value_cols:
                if pd.api.types.is_numeric_dtype(df[col]):
                    nr[col] = float(r[col]) * w * (1 + np.random.normal(0, 0.01))
            rows.append(nr)
    out = pd.DataFrame(rows)
    out["date"] = pd.to_datetime(out["date"])
    return out

ts_reg = regionize(ts)

@st.cache_data
def build_mock_headcount(tsr: pd.DataFrame) -> pd.DataFrame:
    """Synthetic headcount by date/region until Workday feed is wired."""
    dates = sorted(tsr["date"].unique())
    rows = []
    base = {"APAC": 12000, "EMEA": 20000, "LATAM": 8000, "NORTH AMERICA": 18000}
    for d in dates:
        for r in BUS:
            factor = 1 + 0.03*np.sin(len(rows)/12.0) + np.random.normal(0, 0.005)
            rows.append({"date": pd.to_datetime(d), "region": r, "headcount": int(base[r] * factor)})
    return pd.DataFrame(rows)

hc = build_mock_headcount(ts_reg)

with st.sidebar:
    st.header("🌍 Global Filters")
    regions = st.multiselect("Business Units", BUS, default=BUS)
    if not regions:
        st.warning("Select at least one region to view the data.")
        st.stop()

    frameworks = st.multiselect(
        "Reporting Frameworks", ["SEC", "CSRD", "GRI", "CbCR"],
        default=["GRI","CSRD","SEC","CbCR"]
    )

    st.subheader("Aggregation")
    agg_mode = st.radio(
        "How to aggregate regions?",
        ["Simple average", "Headcount-weighted"],
        index=1,
        help="Headcount-weighted = more realistic global rollup"
    )

    st.subheader("Appearance")
    dark_mode = st.toggle("Dark / High-contrast", value=st.session_state.get("dark_mode_toggle", True), key="dark_mode_toggle")

inject_brand_css(dark_mode)

ALL_METRICS = mapping["metric_id"].tolist()

@st.cache_data
def aggregate_selected(tsr: pd.DataFrame, sel_regions: List[str], hc_df: pd.DataFrame, mode: str) -> pd.DataFrame:
    df = tsr[tsr["region"].isin(sel_regions)].copy()
    d = df.merge(hc_df, on=["date","region"], how="left")
    value_cols = [c for c in df.columns if c not in ("date","region")]
    out = []
    for dte, g in d.groupby("date"):
        rec = {"date": pd.to_datetime(dte)}
        if mode == "Headcount-weighted":
            weights = g["headcount"].fillna(0).values
            wsum = weights.sum() if weights.sum() > 0 else 1.0
            for col in value_cols:
                vals = g[col].astype(float).values
                rec[col] = float(np.sum(vals * weights) / wsum)
        else:
            for col in value_cols:
                rec[col] = float(g[col].mean())
        out.append(rec)
    return pd.DataFrame(out).sort_values("date")

ts_global = aggregate_selected(ts_reg, regions, hc, agg_mode)

def linear_trend_stats(df: pd.DataFrame, metric: str) -> Dict[str, float]:
    """Slope per month, annualized slope, R², last value, residual std. Safe month index."""
    d = df.dropna(subset=[metric]).copy().sort_values("date")
    if d.empty:
        return {"slope_m": 0.0, "slope_y": 0.0, "r2": 0.0, "last": float("nan"), "resid_std": 0.0}
    s = pd.to_datetime(d["date"])
    y0, m0 = s.iloc[0].year, s.iloc[0].month
    x = ((s.dt.year - y0) * 12 + (s.dt.month - m0)).to_numpy(dtype=float)
    y = d[metric].to_numpy(dtype=float)
    if x.size < 2:
        return {"slope_m": 0.0, "slope_y": 0.0, "r2": 0.0, "last": (float(y[-1]) if y.size else float("nan")), "resid_std": 0.0}
    a, b = np.polyfit(x, y, 1)
    y_hat = a * x + b
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2)) if y.size > 1 else 0.0
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    resid_std = float(np.std(y - y_hat, ddof=1)) if y.size > 2 else 0.0
    return {"slope_m": float(a), "slope_y": float(a * 12), "r2": float(r2), "last": float(y[-1]), "resid_std": resid_std}

def months_until(deadline: pd.Timestamp, last_date: pd.Timestamp) -> int:
    dl = pd.to_datetime(deadline); ld = pd.to_datetime(last_date)
    return max(0, (dl.year - ld.year) * 12 + (dl.month - ld.month))

def mc_forecast(last_value: float, slope_m: float, resid_std: float, months_ahead: int, n_sims: int = 5000) -> Dict[str, float]:
    sims = np.repeat(last_value, n_sims).astype(float)
    for _ in range(months_ahead):
        sims = sims + slope_m + np.random.normal(0, max(resid_std, 1e-9), size=n_sims)
    return {
        "p10": float(np.percentile(sims, 10)),
        "p50": float(np.percentile(sims, 50)),
        "p90": float(np.percentile(sims, 90)),
        "mean": float(np.mean(sims)),
        "std": float(np.std(sims)),
        "dist": sims,
    }

GOALBOOK = {
    "Pay_Equity":    {"direction": "up",   "SEC": 0.95, "CSRD": 0.95, "GRI": 0.95, "CbCR": 0.95},
    "Training_Hours":{"direction": "up",   "SEC": 12.0, "CSRD": 14.0, "GRI": 12.0, "CbCR": 12.0},
    "Safety_IR":     {"direction": "down", "SEC": 1.3,  "CSRD": 1.2,  "GRI": 1.3,  "CbCR": 1.3},
    "Carbon_per_FTE":{"direction": "down", "SEC": 160,  "CSRD": 150,  "GRI": 160,  "CbCR": 155},
}
def pass_fail(value: float, metric: str, fw: str) -> bool:
    t = GOALBOOK.get(metric, {}).get(fw)
    if t is None: return True
    return value >= t if GOALBOOK[metric]["direction"] == "up" else value <= t

tabs = st.tabs([
    "📊 Dashboard", "🗺️ Compliance", "🎯 Materiality", "🔗 Lineage",
    "🛡️ Assurance Tracker", "📈 Trend Analytics", "🧪 Sandbox & Targets",
    "🗓️ Reporting Calendar", "📦 Export"
])

with tabs[0]:
    st.subheader("HR Sustainability Metrics")
    metric = st.selectbox("Metric", ALL_METRICS, index=0)
    codes = mapping[mapping["metric_id"] == metric].iloc[0].to_dict()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("SEC", codes.get("SEC", "—"))
    c2.metric("CSRD", codes.get("CSRD", "—"))
    c3.metric("GRI", codes.get("GRI", "—"))
    c4.metric("Definition", codes.get("description", "—"))

    line = (
        alt.Chart(ts_global)
        .mark_line(point=True, color=COKE_RED)
        .encode(x="date:T", y=alt.Y(f"{metric}:Q", title=metric.replace("_"," ")), tooltip=["date:T", f"{metric}:Q"])
        .properties(height=360)
        .configure(**altair_theme(dark_mode)["config"])
    )
    st.altair_chart(line, use_container_width=True)

    st.subheader("What-If Simulation (Global Aggregate)")
    a, b, c = st.columns(3)
    remote = a.slider("Remote eligibility Δ%", -20, 20, 5, key="remote_dash")
    training = b.slider("Training hrs/FTE Δ%", -20, 20, 10, key="training_dash")
    safety = c.slider("Safety intensity Δ%", -20, 20, 0, key="safety_dash")

    last_row = ts_global.iloc[-1].copy()
    proj = last_row.copy()
    proj["Carbon_per_FTE"] *= (1 - remote*0.006)
    proj["Training_Hours"] *= (1 + training*0.01)
    proj["Safety_IR"] *= (1 - max(safety,0)*0.005 + min(safety,0)*0.003)

    before = float(last_row.get(metric))
    after = float(proj.get(metric))
    st.metric(f"{metric} (projected)", value=round(after,3), delta=round(after-before,3))

    df = pd.DataFrame([{"scenario":"Current","value":before},{"scenario":"Projected","value":after}])
    bar = (
        alt.Chart(df).mark_bar(color=COKE_RED).encode(x="scenario:N", y="value:Q")
        .properties(height=220).configure(**altair_theme(dark_mode)["config"])
    )
    st.altair_chart(bar, use_container_width=True)

with tabs[1]:
    st.subheader("Compliance Heatmap (Filtered)")
    def comp_filter(r):
        if r["framework"] == "SEC":  return "SEC"  in frameworks and "NORTH AMERICA" in regions
        if r["framework"] == "CSRD": return "CSRD" in frameworks
        if r["framework"] == "GRI":  return "GRI"  in frameworks
        if r["framework"] == "CbCR": return "CbCR" in frameworks
        return False
    comp_view = comp[comp.apply(comp_filter, axis=1)]
    if comp_view.empty:
        st.info("No applicable controls for these filters.")
    else:
        heat = (
            alt.Chart(comp_view)
            .mark_rect()
            .encode(
                x=alt.X("framework:N", title="Framework"),
                y=alt.Y("control:N", title="Control"),
                color=alt.Color("readiness:N", scale=alt.Scale(domain=["Red","Yellow","Green"], range=["#d73027","#fee08b","#1a9850"])),
                tooltip=["framework","control","readiness","owner"]
            ).properties(height=360).configure(**altair_theme(dark_mode)["config"])
        )
        st.altair_chart(heat, use_container_width=True)
        st.dataframe(comp_view, use_container_width=True)

with tabs[2]:
    st.subheader("Materiality Matrix")
    scatter = (
        alt.Chart(mat)
        .mark_circle(size=200)
        .encode(
            x=alt.X("stakeholder_importance:Q", title="Stakeholder importance (1–7)"),
            y=alt.Y("business_impact:Q", title="Business impact (1–7)"),
            color=alt.Color("stakeholder_group:N", legend=alt.Legend(title="Stakeholders")),
            size=alt.Size("weight:Q", title="Weight"),
            tooltip=["topic","stakeholder_group","stakeholder_importance","business_impact","weight"],
        ).properties(height=380).configure(**altair_theme(dark_mode)["config"])
    )
    st.altair_chart(scatter, use_container_width=True)
    st.dataframe(mat, use_container_width=True)

with tabs[3]:
    st.subheader("End-to-End Lineage (Mock)")
    try:
        if dark_mode:
            bg_graph = "transparent"
            fontcolor = "#FFFFFF"
            edge_color = "#AAAAAA"
            border = "#8A8A8A"
            # node fills
            fill_workday = "#2F3A4A"   # deep slate
            fill_synapse = "#234A2F"   # deep green
            fill_powerbi = "#4A2F2F"   # deep maroon
            fill_other   = "#2A2A2A"   # dark gray
        else:
            bg_graph = "transparent"
            fontcolor = "#1E1E1E"
            edge_color = "#6F6F6F"
            border = "#9A9A9A"
            fill_workday = "#E7F0FF"   # light blue
            fill_synapse = "#E8F7EA"   # light green
            fill_powerbi = "#FFEAEA"   # light red
            fill_other   = "#F2F2F2"   # light gray

        nodes = sorted(set(lin["source"]).union(lin["target"]))
        dg = Digraph(graph_attr={
            "rankdir":"LR",
            "bgcolor": bg_graph,
            "splines":"spline",
            "pad":"0.2",
            "nodesep":"0.35",
            "ranksep":"0.6",
            "fontcolor": fontcolor,
            "color": edge_color,
        })
        # default node style
        dg.attr("node", shape="box", style="filled,rounded", color=border, fontcolor=fontcolor, penwidth="1.2")

        for n in nodes:
            if n.startswith("Workday"):
                dg.node(n, n, fillcolor=fill_workday)
            elif n.startswith("Synapse"):
                dg.node(n, n, fillcolor=fill_synapse)
            elif n.startswith("PowerBI"):
                dg.node(n, n, fillcolor=fill_powerbi)
            else:
                dg.node(n, n, fillcolor=fill_other)

        # Edges
        for _, r in lin.iterrows():
            dg.edge(str(r["source"]), str(r["target"]), color=edge_color, penwidth="1.1")

        st.graphviz_chart(dg, use_container_width=True)
        st.caption("Readable lineage colors automatically adapt to dark/light mode.")
    except Exception as e:
        st.warning(f"Lineage unavailable: {e}")

with tabs[4]:
    st.subheader("Third-Party Assurance Tracker")
    rng = pd.date_range(ts_reg["date"].min(), periods=4, freq="Q")
    auditor = {"APAC":"KPMG","EMEA":"PwC","LATAM":"EY","NORTH AMERICA":"Deloitte"}
    metrics_env = ["Pay_Equity","Training_Hours","Safety_IR","Carbon_per_FTE"]
    rows = []
    for r in regions:
        for m in metrics_env:
            rows.append({
                "region": r, "metric": m, "auditor": auditor[r],
                "status": np.random.choice(["Unassessed","Fieldwork","Assured"], p=[0.3,0.4,0.3]),
                "window_start": rng[2].date().isoformat(), "window_end": rng[3].date().isoformat(),
                "owner": np.random.choice(["People Insights","Comp & Ben","EHS","Sustainability"])
            })
    adf = pd.DataFrame(rows)
    st.dataframe(adf, use_container_width=True, height=380)
    pbc = adf.rename(columns={"owner":"pbc_owner"})
    st.download_button("⬇️ Download PBC CSV", data=pbc.to_csv(index=False), file_name="pbc.csv", mime="text/csv")

with tabs[5]:
    st.subheader("Trend Analytics (Global Aggregate)")
    m = st.selectbox("Metric for analysis", ALL_METRICS, key="trend_metric")
    stats = linear_trend_stats(ts_global, m)
    r1, r2, r3, r4, r5 = st.columns(5)
    r1.metric("Last value", round(stats["last"], 3))
    r2.metric("Slope / month", round(stats["slope_m"], 4))
    r3.metric("Annualized change", round(stats["slope_y"], 3))
    r4.metric("R² (fit)", round(stats["r2"], 3))
    r5.metric("Residual σ", round(stats["resid_std"], 3))

    roll = ts_global[["date", m]].copy().sort_values("date")
    roll["roll_3m"] = roll[m].rolling(3, min_periods=1).mean()

    months_ahead = st.slider("Forecast horizon (months)", 3, 24, 12, key="horizon_trend")
    sims = st.slider("Monte Carlo simulations", 1000, 20000, 5000, step=1000, key="mc_sims_trend")

    mc = mc_forecast(stats["last"], stats["slope_m"], stats["resid_std"], months_ahead, n_sims=sims)
    st.caption(f"Monte Carlo @ +{months_ahead}m → p50={mc['p50']:.3f}, p10={mc['p10']:.3f}, p90={mc['p90']:.3f}")

    last_date = roll["date"].max()
    future_date = last_date + pd.DateOffset(months=months_ahead)
    fan_df = pd.DataFrame({"date":[last_date, future_date], "p10":[stats["last"], mc["p10"]], "p50":[stats["last"], mc["p50"]], "p90":[stats["last"], mc["p90"]]})

    hist = alt.Chart(roll).mark_line(point=True).encode(x="date:T", y=alt.Y(f"{m}:Q", title=m.replace("_"," ")))
    roll_line = alt.Chart(roll).mark_line(strokeDash=[4,3]).encode(x="date:T", y="roll_3m:Q")
    band = alt.Chart(fan_df).mark_area(opacity=0.2).encode(x="date:T", y="p10:Q", y2="p90:Q")
    median = alt.Chart(fan_df).mark_line(color=COKE_RED).encode(x="date:T", y="p50:Q")

    st.altair_chart((hist + roll_line + band + median).properties(height=360).configure(**altair_theme(dark_mode)["config"]), use_container_width=True)

with tabs[6]:
    st.subheader("Sandbox: Set Targets & See Gap to Goal")
    m = st.selectbox("Metric for target", ALL_METRICS, key="target_metric")
    region_choice = st.selectbox("Region (BU)", regions, key="target_region")
    series = ts_reg[ts_reg["region"] == region_choice].sort_values("date").reset_index(drop=True)
    last_date = series["date"].max()
    last_val = float(series[m].iloc[-1])

    direction = GOALBOOK.get(m, {}).get("direction", "up")
    default_target = last_val * (1.05 if direction=="up" else 0.95)
    target_value = st.number_input("Target value", value=float(round(default_target, 3)))
    deadline = st.date_input("Deadline", (last_date + pd.DateOffset(months=12)).date())
    months = months_until(pd.Timestamp(deadline), last_date)

    stats_r = linear_trend_stats(series[["date", m]], m)
    sims_region = st.slider("Monte Carlo simulations (region)", 1000, 20000, 5000, step=1000, key="sims_region")
    mc_r = mc_forecast(stats_r["last"], stats_r["slope_m"], stats_r["resid_std"], months, n_sims=sims_region)

    if direction == "up":
        prob = float((mc_r["dist"] >= target_value).mean()); gap = target_value - last_val
    else:
        prob = float((mc_r["dist"] <= target_value).mean()); gap = last_val - target_value
    monthly_delta_needed = (target_value - last_val) / max(months,1)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current", round(last_val,3))
    c2.metric("Gap to goal", round(gap,3))
    c3.metric("Req. Δ / month", round(monthly_delta_needed,4))
    c4.metric("P(hit by deadline)", f"{prob*100:.1f}%")

    st.subheader("Policy Levers (what-if)")
    a, b, c = st.columns(3)
    remote = a.slider("Remote eligibility Δ%", -20, 20, 0, key="remote_sandbox")
    training = b.slider("Training hrs/FTE Δ%", -20, 20, 0, key="training_sandbox")
    safety = c.slider("Safety intensity Δ%", -20, 20, 0, key="safety_sandbox")

    proj = last_val
    if m == "Carbon_per_FTE": proj *= (1 - remote*0.006)
    if m == "Training_Hours": proj *= (1 + training*0.01)
    if m == "Safety_IR":      proj *= (1 - max(safety,0)*0.005 + min(safety,0)*0.003)
    if m == "Pay_Equity":     proj *= (1 + training*0.002)  # illustrative proxy

    hist = alt.Chart(series[["date", m]]).mark_line(point=True).encode(x="date:T", y=alt.Y(f"{m}:Q", title=m))
    tunnel = pd.DataFrame({"date":[last_date, pd.Timestamp(deadline)], "target":[last_val, target_value]})
    tun_line = alt.Chart(tunnel).mark_line(strokeDash=[4,3]).encode(x="date:T", y="target:Q")
    lever_df = pd.DataFrame({"date":[last_date], "lever":[proj]})
    lever_point = alt.Chart(lever_df).mark_point(size=100, color=COKE_RED).encode(x="date:T", y="lever:Q")

    st.altair_chart((hist + tun_line + lever_point).properties(height=360).configure(**altair_theme(dark_mode)["config"]), use_container_width=True)

with tabs[7]:
    st.subheader("Reporting Calendar (Gantt)")
    cal_rows = [
        {"event":"SEC HCM (10-K)",         "framework":"SEC",  "region_scope":"NORTH AMERICA",              "start":"2025-12-01", "end":"2026-03-15", "owner":"Legal / People Insights"},
        {"event":"EU CSRD ESRS S1/S2",     "framework":"CSRD", "region_scope":"EMEA (global consolidation)","start":"2025-10-01", "end":"2026-04-30", "owner":"Sustainability / Finance / HR"},
        {"event":"GRI Refresh",            "framework":"GRI",  "region_scope":"Global",                     "start":"2025-09-01", "end":"2025-12-15", "owner":"Sustainability"},
        {"event":"CbCR Workforce/Tax",     "framework":"CbCR", "region_scope":"Global",                     "start":"2025-11-01", "end":"2026-01-31", "owner":"Finance / Tax"},
        {"event":"Assurance Fieldwork",    "framework":"Multi","region_scope":"APAC/EMEA/LATAM/NA",         "start":"2026-02-01", "end":"2026-03-31", "owner":"External Auditors"},
    ]
    cal = pd.DataFrame(cal_rows)
    cal["start"] = pd.to_datetime(cal["start"], errors="coerce")
    cal["end"]   = pd.to_datetime(cal["end"],   errors="coerce")

    gantt = (
        alt.Chart(cal).mark_bar().encode(
            y=alt.Y("event:N", title="Milestone"),
            x=alt.X("start:T", title="Timeline"),
            x2="end:T",
            color=alt.Color("framework:N"),
            tooltip=["event","framework","region_scope","start:T","end:T","owner"]
        ).properties(height=340).configure(**altair_theme(dark_mode)["config"])
    )
    st.altair_chart(gantt, use_container_width=True)

    def _fmt_date(x):
        try: return pd.to_datetime(x).date().isoformat()
        except Exception: return str(x)

    cal_view = cal.copy()
    cal_view["start"] = cal_view["start"].apply(_fmt_date)
    cal_view["end"]   = cal_view["end"].apply(_fmt_date)

    st.download_button(
        "⬇️ Download Calendar CSV",
        data=cal_view.to_csv(index=False),
        file_name="reporting_calendar.csv",
        mime="text/csv",
    )

with tabs[8]:
    st.subheader("Export Assurance + Targets Bundle")

    latest_safe = {k: (v.isoformat() if isinstance(v, pd.Timestamp) else v) for k, v in ts_global.iloc[-1].to_dict().items()}

    status_rows = []
    for r in regions:
        row = ts_reg[ts_reg["region"] == r].sort_values("date").iloc[-1]
        for m in ALL_METRICS:
            val = float(row[m])
            for fw in frameworks:
                if m in GOALBOOK:
                    status_rows.append({"region": r, "framework": fw, "metric": m, "value": val, "pass": pass_fail(val, m, fw)})

    package = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "regions": regions,
        "frameworks": frameworks,
        "aggregation": agg_mode,
        "metrics": ALL_METRICS,
        "latest_snapshot": latest_safe,
        "standards_mapping": mapping.to_dict("records"),
        "compliance_readiness": comp.to_dict("records"),
        "compliance_status": status_rows,
        "lineage": lin.to_dict("records"),
        "assurance": adf.to_dict("records") if "adf" in locals() else [],
        "reporting_calendar": cal_view.to_dict("records"),
        "brand": {"primary": COKE_RED, "text": COKE_BLACK, "logo_exists": LOGO_PATH.exists()},
    }

    payload = json.dumps(to_jsonable(package), indent=2)
    st.download_button("⬇️ Download JSON", data=payload, file_name="assurance_targets_bundle.json", mime="application/json")
    st.code(payload[:1200] + ("...\n" if len(payload) > 1200 else ""), language="json")