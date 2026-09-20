"""
classic_bull_flag_v1.py — DailyBreakouts Classic Bull Flag V1
================================================================

Standalone Classic Bull Flag scanner + 27-combination portfolio backtest.

WHY THIS IS GENUINELY DIFFERENT FROM tight_flag_v1.py
tight_flag_v1.py checked "flag depth <= 10%" but never required the flag
to actually SLOPE DOWNWARD, and treated volume as informational only.
The canonical Bull Flag definition (checked against multiple technical-
analysis references) requires BOTH of these as hard structural gates:
  - The flag itself is a PARALLEL, DOWNWARD-SLOPING channel (not a flat
    box, and not converging like a Pennant/Wedge).
  - Volume DECREASES during the flag and SPIKES on the breakout --
    confirming quiet profit-taking during the pause, then real buying
    interest at the break.
This makes the flag's geometry AND its volume signature both hard gates
here, not just structural depth -- a real, canonical pattern this time,
not one built from scratch with invented thresholds.

CLASSIC BULL FLAG V1 — FROZEN SIGNAL RULES
1) Flagpole: a strong, causal rally of >= 20% over a 15-40 session
   lookback ending at the flag's start (same impulse-search style as
   tight_flag_v1.py).
2) Flag duration: 5-15 completed sessions before the breakout day.
3) Flag slope: linear regression of daily closes across the flag must
   have a NEGATIVE slope (the channel genuinely drifts down) -- not
   flat, not up. A flat or upward-drifting consolidation is a
   different pattern (a base, not a flag) and is rejected here.
4) Flag depth / retracement cap: the flag's low must not retrace more
   than 50% of the flagpole's own gain (a standard Fibonacci-style
   cap from the canonical definition) -- a flag that gives back more
   than half the pole's move is a failing/invalidated setup, not a
   healthy pause.
5) Volume during the flag must be DECREASING on average (the second
   half of the flag's average volume must be lower than the first
   half's) -- confirms quiet profit-taking, not panic selling.
6) Breakout: Close > the flag's own high (the upper channel boundary
   projected to the breakout day), on volume >= 1.5x the flag's own
   average volume -- confirms real buying interest at the break, not
   a low-conviction drift above the line. Enter at that Close.
7) Trend/quality filters, identical to VCM/Tight Flag/Pullback: Close
   > 50DMA and > 200DMA on breakout day.
8) Historical cross-sectional RS percentile >= 70 (same bar as VCM/
   Tight Flag -- this is a strength-continuation pattern, matching
   their selection criterion, unlike Pullback/Low-RS).
9) 20-day median traded value >= Rs 5 crore.
10) Point-in-time market cap >= Rs 500 crore, same annual snapshot
    framework as the other three patterns.
11) One signal per symbol/date; one simultaneous position per symbol.

EXIT RULES -- SAME CONVENTION AS THE OTHER THREE PATTERNS, DELIBERATELY
Reused unchanged so all four patterns plug into the same analysis
tooling (switching simulations, walk-forward, oos tests, correlation
checks against VCM) for a fair, apples-to-apples comparison.

PORTFOLIO GRID
- Trail: 25 / 50 / 100 DMA
- Stop: 4% / 8% / 12%
- Max positions: 5 / 10 / 15
- Fixed starting/sizing base: Rs 10 lakh; no compounding.
- Whole shares.
- Same exchange-charge model as the other three patterns.
- No slippage.

IMPORTANT LIMITATION
The universe directory contains currently available symbols, so
survivorship bias remains unless a historical delisted/renamed-stock
universe is supplied.
"""

from pathlib import Path
from collections import deque
import numpy as np, pandas as pd, time, warnings
warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
DATA = (HERE / ".." / "data").resolve()
if not DATA.exists(): DATA = Path(r"D:\projects\dailybreakouts\data")
UDIR = DATA / "universe"
START = pd.Timestamp("2020-01-01"); END = pd.Timestamp("2026-09-15")
START_CAP = 1_000_000.

# ------------------------------------------------------------------
# POINT-IN-TIME MARKET CAP FILTER -- identical to the other 3 patterns.
# ------------------------------------------------------------------
MARKETCAP_MIN_CR = 500.0
MCAP_SNAPSHOT_FILES = {
    2020: DATA/"mcap_2019-12-31.xlsx",
    2021: DATA/"mcap_2020-12-31.xlsx",
    2022: DATA/"mcap_2021-12-31.xlsx",
    2023: DATA/"mcap_2022-12-31.xlsx",
    2024: DATA/"mcap_2023-12-31.xlsx",
    2025: DATA/"mcap_2024-12-31.xlsx",
    2026: DATA/"market_caps.csv",
}

def _load_one_mcap_file(path):
    if not path.exists():
        return None
    df = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_excel(path)
    if "symbol" in df.columns and "market_cap_cr" in df.columns:
        vals = pd.to_numeric(df["market_cap_cr"], errors="coerce")
        return dict(zip(df["symbol"].astype(str).str.upper(), vals))
    mcap_col = next((c for c in df.columns if "market capitalisation" in c.lower()
                      or "market capitalization" in c.lower()), None)
    if mcap_col is None or "Symbol" not in df.columns:
        print(f"WARNING: couldn't parse {path.name} -- this snapshot will be skipped.")
        return None
    col_lower = mcap_col.lower()
    divisor = 100 if "lakh" in col_lower else (1 if "crore" in col_lower else 1e7)
    out = {}
    for _, row in df.iterrows():
        sym = str(row["Symbol"]).strip().upper()
        raw = row[mcap_col]
        if pd.isna(raw) or sym in ("", "NAN"):
            continue
        try:
            out[sym] = float(raw) / divisor
        except (ValueError, TypeError):
            continue
    return out

def load_marketcap_by_year():
    by_year = {}
    for year, path in MCAP_SNAPSHOT_FILES.items():
        d = _load_one_mcap_file(path)
        if d is not None:
            by_year[year] = d
            print(f"Market cap snapshot for {year} (from {path.name}): {len(d)} symbols.")
        else:
            print(f"Market cap snapshot for {year} MISSING ({path.name} not found/unparseable) "
                  f"-- symbols will fall back to the nearest earlier snapshot for this year.")
    if not by_year:
        print("WARNING: no market cap snapshots loaded at all -- mcap_pass will be False for everything.")
    return by_year

MCAP_BY_YEAR = load_marketcap_by_year()

def mcap_for(symbol, year):
    sym = symbol.upper()
    for y in range(max(year, 2020), 2019, -1):
        if y in MCAP_BY_YEAR and sym in MCAP_BY_YEAR[y]:
            return MCAP_BY_YEAR[y][sym]
    return float("nan")

# ------------------------------------------------------------------
# EXCHANGE CHARGES -- identical to the other 3 patterns.
# ------------------------------------------------------------------
STT_PCT = 0.001
NSE_TXN_PCT = 0.0000297
SEBI_FEE_PCT = 0.000001
STAMP_DUTY_PCT = 0.00015

def buy_charges(value):
    return value * (STT_PCT + NSE_TXN_PCT + SEBI_FEE_PCT + STAMP_DUTY_PCT)

def sell_charges(value):
    return value * (STT_PCT + NSE_TXN_PCT + SEBI_FEE_PCT)


def load(p):
    try: d = pd.read_csv(p)
    except: return None
    d.columns = [str(x).lower().strip() for x in d.columns]
    if "date" not in d: return None
    d["date"] = pd.to_datetime(d.date, errors="coerce")
    for c in ["open","high","low","close","volume"]:
        if c in d: d[c] = pd.to_numeric(d[c], errors="coerce")
    req = ["date","open","high","low","close","volume"]
    if any(c not in d for c in req): return None
    d = d.dropna(subset=req).sort_values("date").drop_duplicates("date")
    d = d[(d.close>0)&(d.high>0)&(d.low>0)&(d.volume>=0)].reset_index(drop=True)
    return d if len(d)>=260 else None


def flag_slope(y):
    """Linear regression slope of an array of prices against session
    index (0,1,2,...). Negative = genuinely downward-sloping channel."""
    x = np.arange(len(y))
    if len(y) < 2 or np.std(y) == 0:
        return 0.0
    slope, _ = np.polyfit(x, y, 1)
    return slope


def features_and_candidates(p):
    sym = p.stem
    d = load(p)
    if d is None:
        return [], None, None

    c=d.close.to_numpy(float); h=d.high.to_numpy(float); l=d.low.to_numpy(float); v=d.volume.to_numpy(float)
    n=len(d); dates=d.date

    ma50=pd.Series(c).rolling(50).mean().to_numpy()
    ma200=pd.Series(c).rolling(200).mean().to_numpy()
    turn20=pd.Series(c*v/1e7).rolling(20).median().to_numpy()

    s=pd.Series(c)
    r63=s/s.shift(63)-1
    q2=s.shift(63)/s.shift(126)-1
    q3=s.shift(126)/s.shift(189)-1
    q4=s.shift(189)/s.shift(252)-1
    rsraw=.4*r63+.2*q2+.2*q3+.2*q4
    rsdf=pd.DataFrame({"date":dates,"symbol":sym,"rsraw":rsraw})

    out=[]

    for i in range(220, n):
        dt=dates.iloc[i]
        if not (START<=dt<=END):
            continue

        trend = (np.isfinite(ma50[i]) and np.isfinite(ma200[i])
                 and c[i]>ma50[i] and c[i]>ma200[i])
        turnover = np.isfinite(turn20[i]) and turn20[i]>=5
        if not (trend and turnover):
            continue

        found=None
        for flag_len in range(5,16):
            fs=i-flag_len   # flag start
            fe=i-1          # flag end (day before breakout)
            if fs<41:
                continue

            flag_closes = c[fs:i]
            flag_vol = v[fs:i]
            flag_high = float(np.max(h[fs:i]))
            flag_low = float(np.min(l[fs:i]))
            if flag_high<=0:
                continue

            # Gate: the flag must genuinely SLOPE DOWNWARD.
            slope = flag_slope(flag_closes)
            if slope>=0:
                continue

            # Breakout must close above the flag's own high.
            if not (c[i]>flag_high):
                continue

            # Flagpole search: strongest causal 15-40 session lookback
            # ending at the flag's start.
            best_gain=-np.inf; best_start=None
            for k in range(15,41):
                j=fs-k
                if j<0 or c[j]<=0:
                    continue
                gain=(flag_high/c[j]-1)*100
                if gain>best_gain:
                    best_gain=gain; best_start=j
            if best_start is None or best_gain<20:
                continue

            # Retracement cap: flag_low must not give back more than
            # 50% of the flagpole's own gain.
            pole_gain_abs = flag_high - c[best_start]
            if pole_gain_abs<=0:
                continue
            retracement_pct = (flag_high-flag_low)/pole_gain_abs*100
            if retracement_pct>50:
                continue

            # Volume must be DECREASING during the flag (second half
            # average lower than first half average).
            half = max(1, len(flag_vol)//2)
            first_half_vol = float(np.mean(flag_vol[:half]))
            second_half_vol = float(np.mean(flag_vol[half:])) if len(flag_vol)>half else first_half_vol
            if not (second_half_vol < first_half_vol):
                continue

            # Breakout volume must SPIKE relative to the flag's own average.
            flag_avg_vol = float(np.mean(flag_vol))
            if flag_avg_vol<=0:
                continue
            breakout_vol_ratio = v[i]/flag_avg_vol
            if breakout_vol_ratio<1.5:
                continue

            found = dict(
                flag_start=dates.iloc[fs], flag_end=dates.iloc[fe],
                flag_sessions=flag_len, flag_high=flag_high, flag_low=flag_low,
                flag_slope=slope, impulse_start=dates.iloc[best_start],
                impulse_gain_pct=best_gain, retracement_pct=retracement_pct,
                breakout_vol_ratio=breakout_vol_ratio,
                pivot=flag_high,
            )
            break  # first (shortest) valid flag wins -- deterministic

        if found is None:
            continue

        mcap = mcap_for(sym, dt.year)
        out.append(dict(symbol=sym, date=dt, entry_price=float(c[i]), **found,
                         trend_pass=trend, turnover_pass=turnover, market_cap_cr=mcap))

    px=d.set_index("date")[["open","high","low","close"]].copy()
    px["ma25"]=pd.Series(c).rolling(25).mean().to_numpy()
    px["ma50"]=ma50
    px["ma100"]=pd.Series(c).rolling(100).mean().to_numpy()
    return out, px, rsdf


files=sorted(UDIR.glob("*.csv"))

print(f"CLASSIC BULL FLAG V1 scanning {len(files)} files once...")
print("Signal: flagpole >=20% | flag 5-15 sessions, SLOPES DOWNWARD | retracement")
print("<=50% of pole | volume DECREASES in flag, SPIKES >=1.5x on breakout.")
print("Grid: trail 25/50/100 DMA x stop 4/8/12% x max positions 5/10/15 = 27 tests")
print("Fixed Rs 10L, no compounding.")

cand=[]; prices={}; rs_parts=[]; t0=time.time()
for k,p in enumerate(files,1):
    a,px,r = features_and_candidates(p)
    if a: cand.extend(a)
    if px is not None: prices[p.stem]=px
    if r is not None: rs_parts.append(r[(r.date>=START)&(r.date<=END)])
    if k%100==0: print(f"{k}/{len(files)} | bull flag candidates {len(cand)} | {(time.time()-t0)/60:.1f} min")

C=pd.DataFrame(cand)
if C.empty:
    raise RuntimeError("Classic Bull Flag V1 produced zero structural candidates.")

print("Calculating historical cross-sectional RS percentiles...")
R=pd.concat(rs_parts,ignore_index=True).dropna(subset=["rsraw"])
R["rs_pct"]=R.groupby("date")["rsraw"].rank(pct=True)*99
R=R[["date","symbol","rs_pct"]]

C=C.merge(R,on=["date","symbol"],how="left")
C["rs_pass"]=C.rs_pct>=70
C["mcap_pass"]=C.market_cap_cr>=MARKETCAP_MIN_CR
C["eligible"]=C.trend_pass & C.turnover_pass & C.rs_pass & C.mcap_pass

S=C[C.eligible].copy()
S=S.sort_values(["date","symbol","retracement_pct"]).drop_duplicates(
    subset=["date","symbol"],keep="first"
).copy()

print(f"Eligible Classic Bull Flag V1 signals after RS70 + mcap500cr: {len(S)}")

funnel=[
    ("structural_candidates",len(C)),
    ("trend",int(C.trend_pass.sum())),
    ("turnover_5cr",int((C.trend_pass&C.turnover_pass).sum())),
    ("mcap_500cr_pit",int((C.trend_pass&C.turnover_pass&C.mcap_pass).sum())),
    ("rs70",len(S)),
]
C.round(6).to_csv(DATA/"classic_bull_flag_v1_candidates.csv",index=False)
S.round(6).to_csv(DATA/"classic_bull_flag_v1_signals.csv",index=False)
pd.DataFrame(funnel,columns=["stage","count"]).to_csv(
    DATA/"classic_bull_flag_v1_funnel.csv",index=False
)

# ------------------------------------------------------------------
# PORTFOLIO GRID -- identical mechanics to VCM/Tight Flag/Pullback.
# ------------------------------------------------------------------
all_dates=sorted({x for px in prices.values() for x in px.index if START<=x<=END})
bydate={x:g.sort_values("rs_pct", ascending=False) for x,g in S.groupby("date")}

TRAILS=[25,50,100]
STOPS=[4,8,12]
MAX_POSITIONS=[5,10,15]
summary_rows=[]
yearly_rows=[]
halfyearly_rows=[]
quarterly_rows=[]


def run_variant(trail_days, stop_pct, max_pos):
    cash=START_CAP
    pos=[]
    trades=[]
    eq=[]
    book_full=0
    insufficient_cash=0
    already_holding=0
    alloc=START_CAP/max_pos
    ma_col=f"ma{trail_days}"
    stop_frac=stop_pct/100.0

    def equity(day):
        z=cash
        for p in pos:
            px=prices[p["symbol"]]
            if day in px.index:
                p["last"]=float(px.loc[day,"close"])
            z+=p["shares"]*p["last"]
        return z

    for day in all_dates:
        keep=[]
        for p in pos:
            px=prices[p["symbol"]]
            if day not in px.index:
                keep.append(p); continue
            row=px.loc[day]
            cl=float(row["close"])
            ma=float(row[ma_col]) if pd.notna(row[ma_col]) else np.nan
            reason=None
            if cl<=p["stop"]:
                reason=f"{stop_pct}% close stop"
            elif not p["trail"] and np.isfinite(ma) and cl>ma:
                p["trail"]=True
            elif p["trail"] and np.isfinite(ma) and cl<ma:
                reason=f"closed below {trail_days}DMA"
            if reason:
                gross_proceeds=p["shares"]*cl
                proceeds=gross_proceeds-sell_charges(gross_proceeds)
                cash+=proceeds
                trades.append({**p,"exit_date":day,"exit_price":cl,"exit_reason":reason,
                               "return_pct":(proceeds/p["cost"]-1)*100,
                               "pnl":proceeds-p["cost"]})
            else:
                p["last"]=cl; keep.append(p)
        pos=keep

        if day in bydate:
            for _,r in bydate[day].iterrows():
                if len(pos)>=max_pos:
                    book_full+=1; continue
                if any(p["symbol"]==r.symbol for p in pos):
                    already_holding+=1; continue
                px=prices[r.symbol]
                cl=float(px.loc[day,"close"])
                ma=float(px.loc[day,ma_col]) if pd.notna(px.loc[day,ma_col]) else np.nan
                entry_price=cl
                sh=int(np.floor(alloc/entry_price))
                gross_cost=sh*entry_price
                cost=gross_cost+buy_charges(gross_cost)
                if cost>cash:
                    charge_rate=STT_PCT+NSE_TXN_PCT+SEBI_FEE_PCT+STAMP_DUTY_PCT
                    sh=int(np.floor(max(cash,0)/(entry_price*(1+charge_rate))))
                    gross_cost=sh*entry_price
                    cost=gross_cost+buy_charges(gross_cost)
                if sh<=0:
                    insufficient_cash+=1; continue
                cash-=cost
                pos.append(dict(symbol=r.symbol,entry_date=day,entry_price=entry_price,
                                shares=sh,cost=cost,stop=entry_price*(1-stop_frac),
                                trail=bool(np.isfinite(ma) and cl>ma),last=cl))
        eq.append((day,equity(day),cash,len(pos)))

    E=pd.DataFrame(eq,columns=["date","equity","cash","open_positions"])
    E["peak"]=E.equity.cummax()
    E["drawdown_pct"]=(E.equity/E.peak-1)*100
    T=pd.DataFrame(trades)

    if max_pos == 10:
        daily_dir = DATA/"daily_equity_curves_bull_flag"
        daily_dir.mkdir(exist_ok=True)
        E[["date","equity"]].to_csv(daily_dir/f"equity_{trail_days}_{stop_pct}.csv", index=False)

    if len(T):
        trades_dir = DATA/"classic_bull_flag_v1_27combos_trades"
        trades_dir.mkdir(exist_ok=True)
        T_export = T[["symbol","entry_date","entry_price","exit_date","exit_price",
                      "return_pct","pnl","exit_reason"]].sort_values("entry_date")
        T_export.to_csv(trades_dir/f"trades_{trail_days}_{stop_pct}_{max_pos}.csv", index=False)
    final=float(E.equity.iloc[-1])
    wins=T[T.return_pct>0] if len(T) else T
    losses=T[T.return_pct<=0] if len(T) else T
    gross_profit=float(wins.pnl.sum()) if len(wins) else 0.0
    gross_loss=float(-losses.pnl.sum()) if len(losses) else 0.0
    profit_factor=(gross_profit/gross_loss) if gross_loss>0 else np.nan

    year_end=E.set_index("date").equity.resample("YE").last()
    year_start=year_end.shift(1)
    if len(year_start): year_start.iloc[0]=START_CAP
    year_pnl=year_end-year_start
    E_indexed = E.set_index("date")
    for dt,pnl in year_pnl.items():
        year_slice = E_indexed[E_indexed.index.year == dt.year]
        year_max_dd = float(year_slice["drawdown_pct"].min()) if len(year_slice) else np.nan
        yearly_rows.append({
            "trail_dma":trail_days,"stop_pct":stop_pct,"max_positions":max_pos,
            "capital_per_position":alloc,"year":int(dt.year),"year_pnl":float(pnl),
            "return_on_fixed_10l_pct":float(pnl/START_CAP*100),
            "year_max_drawdown_pct":year_max_dd,
        })

    e_dates = E.set_index("date").equity
    half_label_series = e_dates.index.year.astype(str) + "-H" + np.where(e_dates.index.month <= 6, "1", "2")
    half_end = e_dates.groupby(half_label_series).last()
    half_end = half_end.reindex(sorted(half_end.index, key=lambda s: (s[:4], s[-1])))
    half_start = half_end.shift(1)
    if len(half_start): half_start.iloc[0] = START_CAP
    half_pnl = half_end - half_start
    period_end_dates = e_dates.groupby(half_label_series).apply(lambda s: s.index.max())
    for label, pnl in half_pnl.items():
        halfyearly_rows.append({
            "trail_dma":trail_days,"stop_pct":stop_pct,"max_positions":max_pos,
            "period":label,"period_end":str(period_end_dates[label].date()),
            "period_pnl":float(pnl),"return_on_fixed_10l_pct":float(pnl/START_CAP*100),
        })

    q_label_series = e_dates.index.year.astype(str) + "-Q" + ((e_dates.index.month - 1) // 3 + 1).astype(str)
    q_end = e_dates.groupby(q_label_series).last()
    q_end = q_end.reindex(sorted(q_end.index, key=lambda s: (s[:4], s[-1])))
    q_start = q_end.shift(1)
    if len(q_start): q_start.iloc[0] = START_CAP
    q_pnl = q_end - q_start
    q_end_dates = e_dates.groupby(q_label_series).apply(lambda s: s.index.max())
    for label, pnl in q_pnl.items():
        quarterly_rows.append({
            "trail_dma":trail_days,"stop_pct":stop_pct,"max_positions":max_pos,
            "period":label,"period_end":str(q_end_dates[label].date()),
            "period_pnl":float(pnl),"return_on_fixed_10l_pct":float(pnl/START_CAP*100),
        })

    summary_rows.append({
        "trail_dma":trail_days,"stop_pct":stop_pct,"max_positions":max_pos,
        "capital_per_position":alloc,"ending_equity":final,
        "total_pnl":final-START_CAP,"return_on_fixed_10l_total_pct":(final-START_CAP)/START_CAP*100,
        "max_drawdown_pct":float(E.drawdown_pct.min()),"closed_trades":int(len(T)),
        "open_at_end":int(len(pos)),"book_full":int(book_full),
        "insufficient_cash":int(insufficient_cash),
        "already_holding":int(already_holding),
        "win_rate_pct":float((T.return_pct>0).mean()*100) if len(T) else np.nan,
        "avg_winner_pct":float(wins.return_pct.mean()) if len(wins) else np.nan,
        "avg_loser_pct":float(losses.return_pct.mean()) if len(losses) else np.nan,
        "median_trade_pct":float(T.return_pct.median()) if len(T) else np.nan,
        "profit_factor":profit_factor
    })

for trail in TRAILS:
    for stop in STOPS:
        for mp in MAX_POSITIONS:
            print(f"Running trail={trail}DMA | stop={stop}% | max positions={mp} | capital/position=Rs {START_CAP/mp:,.2f}")
            run_variant(trail,stop,mp)

SUM=pd.DataFrame(summary_rows)
YR=pd.DataFrame(yearly_rows)
HALF=pd.DataFrame(halfyearly_rows)
QTR=pd.DataFrame(quarterly_rows)

summary_path=DATA/"classic_bull_flag_v1_27combos_summary.csv"
yearly_path=DATA/"classic_bull_flag_v1_27combos_yearly.csv"
halfyearly_path=DATA/"classic_bull_flag_v1_27combos_halfyearly.csv"
quarterly_path=DATA/"classic_bull_flag_v1_27combos_quarterly.csv"

for df in (SUM,YR,HALF,QTR):
    num=df.select_dtypes(include=[np.number]).columns
    df[num]=df[num].round(2)

SUM.to_csv(summary_path,index=False)
YR.to_csv(yearly_path,index=False)
HALF.to_csv(halfyearly_path,index=False)
QTR.to_csv(quarterly_path,index=False)

print("\n=== CLASSIC BULL FLAG V1 27-COMBINATION SUMMARY ===")
print(SUM.to_string(index=False))
print("\nSaved:")
print(summary_path)
print(yearly_path)
print(halfyearly_path)
print(quarterly_path)
print(f"\nClassic Bull Flag trade-by-trade CSVs saved to: {DATA/'classic_bull_flag_v1_27combos_trades'}")
print("Once this has run, compute correlation against VCM's own yearly returns")
print("(dailybreakouts_v29_27combo_yearly.csv) for the SAME combo, exactly as done")
print("for the other patterns -- check honestly rather than assume this diversifies.")
