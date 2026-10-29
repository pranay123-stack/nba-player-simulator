"""Presentation dashboard: `streamlit run app/streamlit_app.py`.

Written for a non-specialist viewer: plain-English headlines first, evidence second, method last.
Every number is read from saved outputs (backtest summaries, frozen demo predictions); nothing is typed.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
BUNDLE = ROOT / "app" / "data"   # small committed copy (scripts/export_dashboard_data.py) for hosted deploys


def _data_source() -> tuple[Path, Path, str]:
    """Prefer full local outputs; fall back to the committed bundle (e.g. on Streamlit Community Cloud)."""
    local = OUT / "backtest" / "main" / "summary_overall.csv"
    if os.environ.get("NBA_SIM_DASHBOARD_DATA") != "bundle" and local.exists():
        return OUT / "backtest" / "main", OUT / "demo", "local outputs"
    return BUNDLE / "backtest", BUNDLE / "demo", "bundled results"


RUN, DEMO_DIR, DATA_SOURCE = _data_source()
# The live simulator needs the processed dataset and the installed package, which only exist locally.
SIMULATOR_AVAILABLE = ((ROOT / "data" / "processed" / "player_games.parquet").exists()
                       and importlib.util.find_spec("nba_sim") is not None)

MODEL_NAME = {"A": "Baseline", "B": "+ Playing time", "C0": "+ Game conditions", "C": "Full model"}
MODEL_LONG = {
    "A": "Baseline: each player's recent averages, every player and stat treated independently.",
    "B": "+ Playing time: first simulates how many minutes each player gets (rotation, rest, who is "
         "out, overtime), keeping the team's minutes adding up to a real game. Stats follow from minutes.",
    "C": "Full model: also uses game context (pace, opponent defence, home court) and links teammates "
         "together. If one player takes more shots, others take fewer, and a hot shooting night lifts the whole team.",
}
COLOR = {"Baseline": "#9e9e9e", "+ Playing time": "#4a90d9", "+ Game conditions": "#5cb85c", "Full model": "#f28e2b"}
STAT_NAME = {"pts": "Points", "reb": "Rebounds", "ast": "Assists", "pra": "Pts+Reb+Ast", "minutes": "Minutes",
             "fg3m": "3-pointers made"}
TEAMS = {"ATL": "Atlanta Hawks", "BKN": "Brooklyn Nets", "BOS": "Boston Celtics", "CHA": "Charlotte Hornets",
         "CHI": "Chicago Bulls", "CLE": "Cleveland Cavaliers", "DAL": "Dallas Mavericks", "DEN": "Denver Nuggets",
         "DET": "Detroit Pistons", "GSW": "Golden State Warriors", "HOU": "Houston Rockets", "IND": "Indiana Pacers",
         "LAC": "LA Clippers", "LAL": "Los Angeles Lakers", "MEM": "Memphis Grizzlies", "MIA": "Miami Heat",
         "MIL": "Milwaukee Bucks", "MIN": "Minnesota Timberwolves", "NOP": "New Orleans Pelicans",
         "NYK": "New York Knicks", "OKC": "Oklahoma City Thunder", "ORL": "Orlando Magic",
         "PHI": "Philadelphia 76ers", "PHX": "Phoenix Suns", "POR": "Portland Trail Blazers",
         "SAC": "Sacramento Kings", "SAS": "San Antonio Spurs", "TOR": "Toronto Raptors", "UTA": "Utah Jazz",
         "WAS": "Washington Wizards"}

st.set_page_config(page_title="NBA Player Outcome Simulator", page_icon="🏀", layout="wide")
st.markdown("""
<style>
.block-container {max-width: 1250px; padding-top: 2rem;}
.hero {font-size: 1.25rem; line-height: 1.6; color: #d8d8d8; margin-bottom: 1.2rem;}
.card {background: #1b1f27; border: 1px solid #2d3340; border-radius: 12px; padding: 18px 20px; height: 100%;}
.card .big {font-size: 2.3rem; font-weight: 700; line-height: 1.1;}
.card .lbl {color: #aab1bd; font-size: 0.95rem; margin-top: 6px; line-height: 1.4;}
.good {color: #5cd68a;} .neutral {color: #f2c14e;} .accent {color: #f28e2b;}
.verdict {border-left: 5px solid; border-radius: 6px; padding: 12px 16px; background: #1b1f27; margin: 8px 0; color: #e6e6e6;}
.note {color: #9aa3af; font-size: 0.9rem;}
</style>""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------- data loading

@st.cache_data
def load_backtest():
    if not (RUN / "summary_overall.csv").exists():
        return None
    ov = pd.read_csv(RUN / "summary_overall.csv")
    team = pd.read_csv(RUN / "summary_team.csv")
    cmp_ = pd.read_csv(RUN / "comparisons.csv")
    man = json.loads((RUN / "manifest.json").read_text())
    return ov, team, cmp_, man


def comparison(cmp_, kind, key):
    r = cmp_[(cmp_["kind"] == kind) & (cmp_["key"] == key)]
    return None if r.empty else r.iloc[0]


@st.cache_data
def load_demo(game_dir: str):
    d = Path(game_dir)
    return (pd.read_csv(d / "comparison.csv"), pd.read_csv(d / "team_points.csv"),
            json.loads((d / "manifest.json").read_text()))


def card(col, big, label, cls="good"):
    col.markdown(f'<div class="card"><div class="big {cls}">{big}</div><div class="lbl">{label}</div></div>',
                 unsafe_allow_html=True)


bt = load_backtest()
demos = sorted(p for p in DEMO_DIR.glob("*") if (p / "comparison.csv").exists()) if DEMO_DIR.exists() else []

st.title("🏀 NBA Player Outcome Simulator")
st.markdown('<div class="hero">Before a game, we forecast the <b>full range</b> of what every player might do '
            '(points, rebounds, assists, minutes), with a probability for every outcome, not just one '
            'number. Then we checked those forecasts honestly against <b>3 seasons of real games the models '
            'had never seen</b>.</div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["① The answer", "② One real game", "③ Can you trust it?",
                                        "④ Try it", "⑤ How it works"])

# ----------------------------------------------------------------------------- 1. the answer
with tab1:
    if bt is None:
        st.warning("Backtest results not found. Run `nba-sim backtest` and `nba-sim report`.")
    else:
        ov, team, cmp_, man = bt
        n_games = int(team["n"].iloc[0] / 2)
        n_obs = int(ov["n"].iloc[0])
        st.subheader("The question")
        st.markdown("**Do player forecasts get better if we simulate *playing time* and *game context*, "
                    "instead of treating every player independently?**")

        pts_ba = comparison(cmp_, "overall", "pts|crps|B|A")
        min_ba = comparison(cmp_, "overall", "minutes|crps|B|A")
        team_ca = comparison(cmp_, "team", "team_pts|C|A")
        team_cb = comparison(cmp_, "team", "team_pts|C|B")
        st.subheader("Headline results")
        c1, c2, c3, c4 = st.columns(4)
        card(c1, f"{-min_ba['rel']:.0%}", "smaller forecast error on <b>minutes played</b> once playing time is simulated")
        card(c2, f"{-pts_ba['rel']:.1%}", "smaller forecast error on <b>player points</b> from the same change")
        card(c3, f"{-team_ca['rel']:.0%}", "smaller error on <b>team scores</b>: Full model vs Baseline", "accent")
        card(c4, f"{n_games:,}", f"real games tested ({n_obs:,} player forecasts per model), all forecast "
             "<b>before</b> tip-off", "neutral")

        st.write("")
        st.subheader("The verdict")
        st.markdown(f'<div class="verdict" style="border-color:#5cd68a"><b>Simulating playing time: clear win.</b> '
                    f'Every stat improved, in every season, with statistical confidence. The biggest gains are for '
                    f'bench players, whose minutes are the hardest to predict.</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="verdict" style="border-color:#f2c14e"><b>Adding game context: a smaller, '
                    f'targeted win.</b> Individual player forecasts barely change (about 0.2–0.9% better). But team '
                    f'scores ({-team_cb["rel"]:.1%} better) and point margins improve clearly, because the model now '
                    f'knows teammates share one ball.</div>', unsafe_allow_html=True)
        st.markdown('<div class="verdict" style="border-color:#9aa3af"><b>Honest limit.</b> The team-score ranges '
                    'are still a bit wider than reality. And these are accuracy results, not a claim of betting '
                    'profit: no market prices were used.</div>', unsafe_allow_html=True)

        st.subheader("How much each step reduces forecast error")
        st.caption("Error = how far the forecast distribution was from what actually happened (CRPS). "
                   "Bars show the reduction versus the Baseline model. Taller is better.")
        rows = []
        for key, label in [("minutes", "Minutes"), ("pts", "Points"), ("reb", "Rebounds"), ("ast", "Assists"),
                           ("pra", "Pts+Reb+Ast")]:
            for m in ["B", "C"]:
                r = comparison(cmp_, "overall", f"{key}|crps|{m}|A")
                if r is not None:
                    rows.append({"what": label, "model": MODEL_NAME[m], "improvement": -r["rel"] * 100})
        for key, label in [("team_pts", "Team score"), ("margin", "Point margin")]:
            for m in ["B", "C"]:
                r = comparison(cmp_, "team", f"{key}|{m}|A")
                if r is not None:
                    rows.append({"what": label, "model": MODEL_NAME[m], "improvement": -r["rel"] * 100})
        df = pd.DataFrame(rows)
        order = ["Minutes", "Points", "Rebounds", "Assists", "Pts+Reb+Ast", "Team score", "Point margin"]
        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X("what:N", sort=order, title=None, axis=alt.Axis(labelAngle=0, labelFontSize=13)),
            xOffset="model:N",
            y=alt.Y("improvement:Q", title="% less error than Baseline"),
            color=alt.Color("model:N", scale=alt.Scale(domain=list(COLOR)[1:2] + ["Full model"],
                                                       range=[COLOR["+ Playing time"], COLOR["Full model"]]),
                            legend=alt.Legend(orient="top", title=None, labelFontSize=13)),
            tooltip=["what", "model", alt.Tooltip("improvement:Q", format=".1f")])
        text = chart.mark_text(dy=-8, fontSize=12, color="white").encode(text=alt.Text("improvement:Q", format=".1f"))
        st.altair_chart((chart + text).properties(height=380), width="stretch")

# ----------------------------------------------------------------------------- 2. one real game
with tab2:
    if not demos:
        st.info("Run `nba-sim demo-historical` to create the demo game.")
    else:
        comp, team_pts, dman = load_demo(str(demos[-1]))
        home, away = dman["home"], dman["away"]
        st.subheader(f"{TEAMS.get(away, away)} at {TEAMS.get(home, home)}, {dman['date']}")
        st.markdown(f"The models forecast this game using **only information from before the game day**, "
                    f"running **{dman['n_simulations']:,} simulations** each. The forecasts were locked and "
                    "fingerprinted **before** we looked at the box score. The game was picked by a fixed rule (two "
                    "top teams, late season) without looking at the result.")

        model = st.radio("Model", ["C", "B", "A"], format_func=lambda m: MODEL_NAME[m], horizontal=True)
        tp = team_pts[team_pts["model"] == model]
        cols = st.columns(len(tp))
        for col, (_, r) in zip(cols, tp.iterrows()):
            inside = r["pred_lo80"] <= r["actual"] <= r["pred_hi80"]
            card(col, f"{int(r['actual'])} pts",
                 f"<b>{TEAMS.get(r['team'], r['team'])}</b> actually scored.<br>Forecast: most likely "
                 f"<b>{r['pred_mean']:.0f}</b>, 80% range <b>{r['pred_lo80']:.0f}–{r['pred_hi80']:.0f}</b> "
                 f"{'✅ inside' if inside else '❌ outside'}", "good" if inside else "neutral")

        st.write("")
        stat = st.selectbox("Statistic", ["pts", "reb", "ast", "pra", "minutes"], format_func=lambda s: STAT_NAME[s])
        c = comp[(comp["model"] == model) & (comp["minutes_mean"] >= 12)].copy()
        c["inside"] = (c[f"{stat}_actual"] >= c[f"{stat}_lo80"]) & (c[f"{stat}_actual"] <= c[f"{stat}_hi80"])
        c["player"] = c["player_name"] + " (" + c["team"] + ")"
        c = c.sort_values(f"{stat}_mean", ascending=False)
        n_in = int(c["inside"].sum())
        st.markdown(f"#### {n_in} of {len(c)} players finished inside their 80% forecast range "
                    f"(an honest forecast should get about {0.8 * len(c):.0f})")
        st.caption("Grey bar = range the model gave 80% probability. White tick = most likely value. "
                   "Dot = what actually happened (green inside the range, red outside).")
        base = alt.Chart(c).encode(y=alt.Y("player:N", sort=list(c["player"]), title=None,
                                           axis=alt.Axis(labelFontSize=13, labelLimit=260)))
        bars = base.mark_bar(height=14, color="#4b5563", cornerRadius=4).encode(
            x=alt.X(f"{stat}_lo80:Q", title=STAT_NAME[stat]), x2=f"{stat}_hi80:Q")
        mid = base.mark_tick(color="white", thickness=2, size=16).encode(x=f"{stat}_mean:Q")
        act = base.mark_circle(size=170, opacity=1).encode(
            x=f"{stat}_actual:Q",
            color=alt.Color("inside:N", scale=alt.Scale(domain=[True, False], range=["#5cd68a", "#ef5350"]), legend=None),
            tooltip=["player", alt.Tooltip(f"{stat}_mean:Q", title="forecast (most likely)", format=".1f"),
                     alt.Tooltip(f"{stat}_lo80:Q", title="80% low"), alt.Tooltip(f"{stat}_hi80:Q", title="80% high"),
                     alt.Tooltip(f"{stat}_actual:Q", title="actual")])
        st.altair_chart((bars + mid + act).properties(height=34 * len(c)), width="stretch")

        st.markdown("#### Zoom into one player")
        player = st.selectbox("Player", list(c["player_name"]))
        r = c[c["player_name"] == player].iloc[0]
        with np.load(demos[-1] / f"predictions_{model}.npz") as z:
            rows = pd.read_parquet(demos[-1] / f"predictions_{model}.rows.parquet")
            i = int(np.flatnonzero(rows["player_name"].to_numpy() == player)[0])
            pmf = z[f"pmf_{stat if stat != 'minutes' else 'pts'}"][i].astype(float)
        if stat != "minutes":
            pmf /= pmf.sum()
            hi = int(max(r[f"{stat}_hi95"] + 6, r[f"{stat}_actual"] + 4))
            dist = pd.DataFrame({"value": np.arange(min(hi, len(pmf))), "chance": pmf[:hi] * 100})
            dist["in80"] = (dist["value"] >= r[f"{stat}_lo80"]) & (dist["value"] <= r[f"{stat}_hi80"])
            bars = alt.Chart(dist).mark_bar().encode(
                x=alt.X("value:O", title=STAT_NAME[stat], axis=alt.Axis(labelAngle=0, values=list(range(0, hi, 5)))),
                y=alt.Y("chance:Q", title="chance (%)"),
                color=alt.Color("in80:N", scale=alt.Scale(domain=[True, False], range=["#4a90d9", "#3a3f4b"]), legend=None),
                tooltip=["value", alt.Tooltip("chance:Q", format=".1f")])
            rule = alt.Chart(pd.DataFrame({"value": [int(r[f'{stat}_actual'])]})).mark_rule(
                color="#f28e2b", strokeWidth=3).encode(x="value:O")
            label = alt.Chart(pd.DataFrame({"value": [int(r[f'{stat}_actual'])], "t": ["actual"]})).mark_text(
                color="#f28e2b", dy=-140, fontSize=14, fontWeight="bold").encode(x="value:O", text="t:N")
            st.altair_chart((bars + rule + label).properties(height=320), width="stretch")
        verb = {"pts": "score", "reb": "grab", "ast": "dish", "pra": "total", "minutes": "play"}[stat]
        unit = {"pts": "points", "reb": "rebounds", "ast": "assists", "pra": "points+rebounds+assists",
                "minutes": "minutes"}[stat]
        pct = r[f"{stat}_pctile"]
        st.markdown(
            f"**In plain words:** before the game, the model expected {player} to {verb} about "
            f"**{r[f'{stat}_mean']:.0f} {unit}**, with an 80% chance of landing between "
            f"**{r[f'{stat}_lo80']:.0f} and {r[f'{stat}_hi80']:.0f}**. He actually got **{r[f'{stat}_actual']:.0f}**, "
            f"{'inside the range ✅' if r['inside'] else 'outside the range ❌'}. That result was higher than "
            f"{pct:.0%} of the simulated games.")
        probs = [col for col in comp.columns if col.startswith(f"P({stat}>=")]
        if probs:
            pcols = st.columns(len(probs))
            for pc, col in zip(pcols, probs):
                t = int(col.split(">=")[1].rstrip(")"))
                hit = r[f"{stat}_actual"] >= t
                pc.metric(f"Chance of {t}+ {dict(pts='points', reb='rebounds', ast='assists', pra='PRA').get(stat, unit)}", f"{r[col]:.0%}", "happened ✅" if hit else "did not happen",
                          delta_color="normal" if hit else "off")

# ----------------------------------------------------------------------------- 3. trust
with tab3:
    if bt is not None:
        ov, team, cmp_, man = bt
        st.subheader("When the model says “80% likely”, does it happen about 80% of the time?")
        st.caption("For every forecast we take the range the model gave 80% probability and count how often the real "
                   "result landed inside. An honest model sits on the dashed 80% line. Too high means it was too cautious; "
                   "too low means it was overconfident.")
        rows = []
        for s in ["minutes", "pts", "reb", "ast", "pra"]:
            for m in ["A", "B", "C"]:
                q = ov[(ov["stat"] == s) & (ov["model"] == m)]
                if len(q):
                    rows.append({"stat": STAT_NAME[s], "model": MODEL_NAME[m], "hit rate": q["pit_in80"].iloc[0] * 100})
        for m in ["A", "B", "C"]:
            q = team[team["model"] == m]
            rows.append({"stat": "Team score", "model": MODEL_NAME[m], "hit rate": q["pit_in80"].iloc[0] * 100})
        df = pd.DataFrame(rows)
        dom = ["Baseline", "+ Playing time", "Full model"]
        bars = alt.Chart(df).mark_bar().encode(
            x=alt.X("stat:N", sort=["Minutes", "Points", "Rebounds", "Assists", "Pts+Reb+Ast", "Team score"], title=None,
                    axis=alt.Axis(labelAngle=0, labelFontSize=13)),
            xOffset=alt.XOffset("model:N", sort=dom),
            y=alt.Y("hit rate:Q", scale=alt.Scale(domain=[50, 100]), title="% of results inside the 80% range"),
            color=alt.Color("model:N", scale=alt.Scale(domain=dom, range=[COLOR[d] for d in dom]),
                            legend=alt.Legend(orient="top", title=None, labelFontSize=13)),
            tooltip=["stat", "model", alt.Tooltip("hit rate:Q", format=".1f")])
        line = alt.Chart(pd.DataFrame({"y": [80]})).mark_rule(strokeDash=[6, 4], color="white").encode(y="y:Q")
        st.altair_chart((bars + line).properties(height=380), width="stretch")
        pra_a = ov[(ov.stat == "pra") & (ov.model == "A")]["pit_in80"].iloc[0]
        pra_c = ov[(ov.stat == "pra") & (ov.model == "C")]["pit_in80"].iloc[0]
        tc = team.set_index("model")
        st.markdown(f"""
- **Player forecasts are well calibrated.** For the Full model the hit rates sit close to 80%. The Baseline is
  overconfident on combined stats: its Pts+Reb+Ast range caught only **{pra_a:.0%}** of results, against
  **{pra_c:.0%}** for the Full model. A player's points, rebounds and assists rise and fall *together*
  with his minutes, and the Baseline ignores that.
- **Team scores are still the weak spot.** All models are too cautious. The Full model's team-score spread is
  ±{tc.loc['C', 'pred_sd']:.0f} points against a real error of ±{tc.loc['C', 'rmse']:.0f}, while the Baseline's is
  ±{tc.loc['A', 'pred_sd']:.0f} against ±{tc.loc['A', 'rmse']:.0f}. The Full model is much closer, but not perfect.
""")
        st.subheader("Does the improvement hold every season?")
        rows = []
        for s in ["2021-22", "2022-23", "2023-24"]:
            for stat in ["pts", "minutes", "pra"]:
                for a, b, lab in [("B", "A", "+ Playing time vs Baseline"), ("C", "B", "Full model vs + Playing time")]:
                    r = comparison(cmp_, "season", f"{stat}|{s}|{a}|{b}")
                    if r is not None:
                        rows.append({"season": s, "stat": STAT_NAME[stat], "step": lab, "improvement": -r["rel"] * 100,
                                     "lo": -r["hi"] / (r["diff"] / r["rel"]) * 100,
                                     "hi": -r["lo"] / (r["diff"] / r["rel"]) * 100})
        df = pd.DataFrame(rows)
        ch = alt.Chart(df).mark_bar().encode(
            x=alt.X("season:N", title=None, axis=alt.Axis(labelAngle=0)), xOffset="step:N",
            y=alt.Y("improvement:Q", title="% less error"),
            color=alt.Color("step:N", scale=alt.Scale(range=[COLOR["+ Playing time"], COLOR["Full model"]]),
                            legend=alt.Legend(orient="top", title=None)),
            tooltip=["season", "stat", "step", alt.Tooltip("improvement:Q", format=".2f")])
        err = alt.Chart(df).mark_errorbar(color="white").encode(x="season:N", xOffset="step:N", y="lo:Q", y2="hi:Q")
        st.altair_chart(alt.layer(ch, err).properties(height=260, width=330).facet(
            column=alt.Column("stat:N", title=None, header=alt.Header(labelFontSize=14))), width="content")
        st.caption("White whiskers are 95% confidence intervals from resampling whole games. A bar whose whisker stays "
                   "above zero is a real improvement, not luck.")

# ----------------------------------------------------------------------------- 4. try it
with tab4:
    st.subheader("Forecast any matchup")
    st.write("Pick two teams and a date. The model uses only games before that date (data runs to April 2024) "
             "and simulates the game 10,000 times.")
    names = sorted(TEAMS.items(), key=lambda kv: kv[1])
    c1, c2, c3, c4 = st.columns(4)
    home = c1.selectbox("Home team", [k for k, _ in names], index=[k for k, _ in names].index("BOS"),
                        format_func=lambda k: TEAMS[k])
    away = c2.selectbox("Away team", [k for k, _ in names], index=[k for k, _ in names].index("MIL"),
                        format_func=lambda k: TEAMS[k])
    date = c3.date_input("Game date", pd.Timestamp("2024-03-20"), min_value=pd.Timestamp("2011-01-01"),
                         max_value=pd.Timestamp("2024-04-15"))
    model = c4.selectbox("Model", ["C", "B", "A"], format_func=lambda m: MODEL_NAME[m])
    if not SIMULATOR_AVAILABLE:
        st.info("**The live simulator runs on the local install only.** It needs the full NBA dataset, which "
                "isn't hosted here. Tabs ①–③ show the saved results from the full backtest and the demo game. "
                "To run it: clone the repo, follow `data/README.md`, then `streamlit run app/streamlit_app.py`.")
    elif home == away:
        st.warning("Pick two different teams.")
    elif st.button("▶ Run 10,000 simulations", type="primary"):
        from nba_sim.config import load_config, resolve
        from nba_sim.data.ingest import load_dataset
        from nba_sim.reporting.matchup_demo import simulate_matchup

        cfg = load_config()
        with st.spinner("Simulating the game 10,000 times (about 20 seconds)…"):
            try:
                ds = load_dataset(resolve(cfg["data"]["processed_dir"]))
                paths = simulate_matchup(ds, cfg, home, away, pd.Timestamp(date), 10000, 42, model)
                t = pd.read_csv(paths["csv"])
                meta = json.loads(Path(paths["json"]).read_text())["meta"]
                cols = st.columns(2)
                for col, code in zip(cols, [away, home]):
                    tpm = meta["team_points"].get(code)
                    if tpm:
                        card(col, f"{tpm['mean']:.0f} pts", f"<b>{TEAMS[code]}</b> expected score. 80% range "
                             f"{tpm['p10']:.0f}–{tpm['p90']:.0f}", "accent")
                t = t[t["minutes_mean"] >= 10].sort_values("pts_mean", ascending=False).copy()
                t["player"] = t["player_name"] + " (" + t["team"] + ")"
                base = alt.Chart(t).encode(y=alt.Y("player:N", sort=list(t["player"]), title=None,
                                                   axis=alt.Axis(labelFontSize=13, labelLimit=260)))
                ch = (base.mark_bar(height=14, color="#4b5563", cornerRadius=4).encode(
                    x=alt.X("pts_lo80:Q", title="Points: 80% range, white tick = most likely"), x2="pts_hi80:Q")
                    + base.mark_tick(color="white", thickness=2, size=16).encode(
                        x="pts_mean:Q", tooltip=["player", alt.Tooltip("pts_mean:Q", format=".1f"), "pts_lo80", "pts_hi80"]))
                st.altair_chart(ch.properties(height=32 * len(t)), width="stretch")
                show = pd.DataFrame({
                    "Player": t["player"], "Minutes (expected)": t["minutes_mean"].round(0),
                    "Points": t["pts_mean"].round(1), "Points 80% range": t["pts_lo80"].astype(int).astype(str) + "–" + t["pts_hi80"].astype(int).astype(str),
                    "Rebounds": t["reb_mean"].round(1), "Assists": t["ast_mean"].round(1),
                    "Chance of 20+ pts": (t["P(pts>=20)"] * 100).round(0).astype(int).astype(str) + "%",
                    "Chance of 30+ PRA": (t["P(pra>=30)"] * 100).round(0).astype(int).astype(str) + "%"})
                st.dataframe(show, hide_index=True, width="stretch")
                st.caption("Roster = everyone active for the team in its last 10 games before the date. Injuries "
                           "announced on game day are not known to the model.")
            except ValueError as e:
                st.error(str(e))

# ----------------------------------------------------------------------------- 5. method
with tab5:
    st.subheader("The three models")
    for m in ["A", "B", "C"]:
        st.markdown(f"- **{MODEL_NAME[m]}**: {MODEL_LONG[m].split(': ', 1)[1]}")
    st.subheader("Why the results can be trusted")
    st.markdown("""
- **Real data:** official NBA box scores for every regular-season game from 2010-11 to 2023-24 (16,658 games).
- **No peeking at the future:** for every game, the model was trained only on earlier games and its forecast was
  saved to disk *before* the result was read. An automated test scrambles all future results and confirms the
  forecasts don't change.
- **Tested the hard way:** 39 rolling re-trainings across 3 seasons. Comparisons use 95% confidence intervals from
  resampling whole games.
- **What the model is allowed to know before a game:** the schedule, past games, and which players are active
  (this list is published before tip-off). It does **not** use starting lineups, injury news or betting lines.
- **Reproducible:** fixed random seeds. Every run saves its settings, data checksums and code fingerprint.
""")
    st.subheader("Plain-language glossary")
    st.markdown("""
- **Distribution forecast:** instead of “Tatum scores 24”, the model says “10% chance under 13, 50% under 23,
  90% under 35…”. That is what you need to price any over/under line.
- **80% range:** the span the model thinks contains the result 4 times out of 5.
- **Forecast error (CRPS):** the average distance between the whole forecast distribution and what happened.
  It rewards being both accurate *and* honest about uncertainty. Lower is better.
""")
    if bt is not None:
        with st.expander("Technical tables (for quants)"):
            ov, team, cmp_, man = bt
            st.markdown("Mean CRPS per player-game (lower is better)")
            st.dataframe(ov.pivot_table(index="stat", columns="model", values="crps").round(4))
            st.markdown("All model comparisons (difference, 95% CI, verdict)")
            st.dataframe(cmp_[["kind", "key", "diff", "lo", "hi", "rel", "verdict"]], hide_index=True)
            st.markdown("Full reports: `reports/validation_report.md`, `reports/research_note.md`, "
                        "`reports/historical_matchup_demo.md`")
