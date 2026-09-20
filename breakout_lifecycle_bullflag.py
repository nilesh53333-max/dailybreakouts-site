"""
BULL FLAG LIFECYCLE SCANNER
==========================================================
Classifies every qualifying Classic Bull Flag setup into one of four
stages, matching the same 4-stage lifecycle as the VCM scanner:

    Forming     -- a flagpole has occurred and the stock is currently
                   inside a valid, still-forming downward-sloping flag
                   channel that hasn't broken out above its pivot yet.
    Fresh       -- closed above the flag's pivot TODAY specifically.
    Open        -- broke out earlier this calendar year, still open.
    Closed      -- broke out this calendar year and has already exited.

Uses the SAME detection logic as classic_bull_flag_v1.py (flagpole
>=20%, flag slopes downward, retracement <=50% of pole, volume
decreases in flag/spikes >=1.5x on breakout, RS>=70, mcap>=500cr) so
"what this screen shows" and "what the backtest would have flagged"
are always the same signal computed the same way.

Runs the same REAL capacity-constrained portfolio simulation across
the full 27-combination grid as the VCM scanner, with RS-ranked slot
allocation -- every signal carries its own outcome for all 27 combos.

Output: ../data/breakout_lifecycle_bullflag.json -- consumed by the
website's Classic Bull Flag strategy tab.

Usage:
    python update_universe_daily.py
    python breakout_lifecycle_bullflag.py
"""

from pathlib import Path
from collections import deque
import json
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")

HERE=Path(__file__).resolve().parent
DATA=(HERE/".."/"data").resolve()
if not DATA.exists(): DATA=Path(r"D:\projects\dailybreakouts\data")
UDIR=DATA/"universe"
OUTPUT_PATH=DATA/"breakout_lifecycle_bullflag.json"

START=pd.Timestamp("2020-01-01")
MARKETCAP_MIN_CR = 500.0

# ASSUMPTION -- change once your final config is locked.
TRAILS = [25, 50, 100]
STOPS = [4, 8, 12]

MCAP_CURRENT_FILE = DATA/"market_caps.csv"

def load_current_marketcap():
    if not MCAP_CURRENT_FILE.exists():
        print(f"WARNING: {MCAP_CURRENT_FILE} not found -- mcap filter will fail-safe.")
        return {}
    df = pd.read_csv(MCAP_CURRENT_FILE)
    vals = pd.to_numeric(df["market_cap_cr"], errors="coerce")
    return dict(zip(df["symbol"].astype(str).str.upper(), vals))

MCAP_CURRENT = load_current_marketcap()
def mcap_for(symbol): return MCAP_CURRENT.get(symbol.upper(), float("nan"))


def load(p):
    try:d=pd.read_csv(p)
    except:return None
    d.columns=[str(x).lower().strip() for x in d.columns]
    if "date" not in d:return None
    d["date"]=pd.to_datetime(d.date,errors="coerce")
    for c in ["open","high","low","close","volume"]:
        if c in d:d[c]=pd.to_numeric(d[c],errors="coerce")
    req=["date","open","high","low","close","volume"]
    if any(c not in d for c in req):return None
    d=d.dropna(subset=req).sort_values("date").drop_duplicates("date")
    d=d[(d.close>0)&(d.high>0)&(d.low>0)&(d.volume>=0)].reset_index(drop=True)
    return d if len(d)>=260 else None


def flag_slope(y):
    """Linear regression slope of daily closes against session index.
    Negative = genuinely downward-sloping channel."""
    x = np.arange(len(y))
    if len(y) < 2 or np.std(y) == 0:
        return 0.0
    slope, _ = np.polyfit(x, y, 1)
    return slope


def compute_all(p, end_date):
    """Returns (breakout_events, still_forming_flags, px_frame, rs_series)
    for one symbol, using Classic Bull Flag V1's detection rules
    (identical to classic_bull_flag_v1.py): flagpole >=20%, flag slopes
    downward, retracement <=50% of pole, volume decreases during the
    flag and spikes >=1.5x on breakout, trend/turnover/RS/mcap gates."""
    sym=p.stem; d=load(p)
    if d is None:return [],[],None,None
    c=d.close.to_numpy(float);h=d.high.to_numpy(float);l=d.low.to_numpy(float);v=d.volume.to_numpy(float)
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
    mcap=mcap_for(sym)

    out=[]
    for i in range(220,n):
        dt=dates.iloc[i]
        if not (START<=dt<=end_date):
            continue

        trend = (np.isfinite(ma50[i]) and np.isfinite(ma200[i])
                 and c[i]>ma50[i] and c[i]>ma200[i])
        turnover = np.isfinite(turn20[i]) and turn20[i]>=5
        if not (trend and turnover):
            continue

        found=None
        for flag_len in range(5,16):
            fs=i-flag_len; fe=i-1
            if fs<41:
                continue
            flag_closes = c[fs:i]
            flag_vol = v[fs:i]
            flag_high = float(np.max(h[fs:i]))
            flag_low = float(np.min(l[fs:i]))
            if flag_high<=0:
                continue
            slope = flag_slope(flag_closes)
            if slope>=0:
                continue
            if not (c[i]>flag_high):
                continue
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
            pole_gain_abs = flag_high - c[best_start]
            if pole_gain_abs<=0:
                continue
            retracement_pct = (flag_high-flag_low)/pole_gain_abs*100
            if retracement_pct>50:
                continue
            half = max(1, len(flag_vol)//2)
            first_half_vol = float(np.mean(flag_vol[:half]))
            second_half_vol = float(np.mean(flag_vol[half:])) if len(flag_vol)>half else first_half_vol
            if not (second_half_vol < first_half_vol):
                continue
            flag_avg_vol = float(np.mean(flag_vol))
            if flag_avg_vol<=0:
                continue
            breakout_vol_ratio = v[i]/flag_avg_vol
            if breakout_vol_ratio<1.5:
                continue

            found = dict(fs=fs, fe=fe, flag_high=flag_high, flag_low=flag_low,
                         pole_start=best_start, retracement_pct=retracement_pct)
            break

        if found is None:
            continue

        out.append(dict(symbol=sym, date=dt, pivot=found["flag_high"],
            base_start=dates.iloc[found["pole_start"]],
            base_high=found["flag_high"], base_low=found["flag_low"],
            close=c[i], age_sessions=i-found["pole_start"],
            structure_pass=True, trend_pass=trend, position_pass=True,
            turnover_pass=turnover, market_cap_cr=mcap, row_idx=i))

    # Still-forming flags: a flagpole has occurred and the stock is
    # currently drifting down inside a valid, not-yet-broken-out flag
    # channel as of end_date -- the Bull Flag equivalent of VCM's
    # "still-forming base".
    forming=[]
    last_i = n-1
    if dates.iloc[last_i] == end_date:
        i = last_i
        trend_now = (np.isfinite(ma50[i]) and np.isfinite(ma200[i])
                     and c[i]>ma50[i] and c[i]>ma200[i])
        turnover_now = np.isfinite(turn20[i]) and turn20[i]>=5
        if trend_now and turnover_now and np.isfinite(mcap) and mcap>=MARKETCAP_MIN_CR:
            for flag_len in range(5,16):
                fs=i-flag_len+1; fe=i  # flag still ongoing, includes today
                if fs<41:
                    continue
                flag_closes = c[fs:i+1]
                flag_vol = v[fs:i+1]
                flag_high = float(np.max(h[fs:i+1]))
                flag_low = float(np.min(l[fs:i+1]))
                if flag_high<=0 or c[i]>flag_high:
                    continue  # already broke out -- not "still forming"
                slope = flag_slope(flag_closes)
                if slope>=0:
                    continue
                best_gain=-np.inf; best_start=None
                for k in range(15,41):
                    j=fs-k
                    if j<0 or c[j]<=0: continue
                    gain=(flag_high/c[j]-1)*100
                    if gain>best_gain: best_gain=gain; best_start=j
                if best_start is None or best_gain<20:
                    continue
                pole_gain_abs = flag_high - c[best_start]
                if pole_gain_abs<=0: continue
                retracement_pct = (flag_high-flag_low)/pole_gain_abs*100
                if retracement_pct>50: continue
                dist_to_pivot = (flag_high/c[i]-1)*100
                forming.append(dict(symbol=sym, pivot=flag_high, close=float(c[i]),
                                     base_high=flag_high, base_low=flag_low,
                                     base_end_date=None, base_number=None,
                                     dist_to_pivot_pct=round(dist_to_pivot,2),
                                     base_start=str(dates.iloc[best_start].date()),
                                     age_sessions=i-best_start))
                break  # shortest valid in-progress flag wins, same as breakout search

    px=d.set_index("date")[["open","high","low","close","volume"]].copy()
    px["ma25"]=pd.Series(c).rolling(25).mean().to_numpy()
    px["ma50"]=ma50
    px["ma100"]=pd.Series(c).rolling(100).mean().to_numpy()
    return out, forming, px, rsdf


def main():
    files=sorted(UDIR.glob("*.csv"))
    if not files:
        print(f"No universe files found in {UDIR}."); return

    latest_dates=[]
    for p in files[:50]:
        d=load(p)
        if d is not None and len(d): latest_dates.append(d["date"].max())
    if not latest_dates:
        print("Couldn't determine latest date."); return
    end_date = max(latest_dates)
    current_year = end_date.year
    print(f"Scanning lifecycle as of: {end_date.date()} (year {current_year})")

    all_events=[]; all_forming=[]; rs_parts=[]; px_store={}
    for k,p in enumerate(files,1):
        events, forming, px, r = compute_all(p, end_date)
        if events: all_events.extend(events)
        if forming: all_forming.extend(forming)
        if px is not None: px_store[p.stem]=px
        if r is not None: rs_parts.append(r)
        if k%200==0: print(f"  {k}/{len(files)} scanned...")

    C=pd.DataFrame(all_events)
    R=pd.concat(rs_parts,ignore_index=True).dropna(subset=["rsraw"])
    R["rs_pct"]=R.groupby("date")["rsraw"].rank(pct=True)*99
    C=C.merge(R[["date","symbol","rs_pct"]],on=["date","symbol"],how="left")
    C["rs_pass"]=C.rs_pct>=70
    C["mcap_pass"]=C.market_cap_cr>=MARKETCAP_MIN_CR
    C["eligible"]=C.structure_pass&C.trend_pass&C.position_pass&C.turnover_pass&C.rs_pass&C.mcap_pass
    S=C[C.eligible].sort_values(["date","symbol","pivot"]).copy()
    S=S.drop_duplicates(subset=["date","symbol"], keep="first")

    # This calendar year's breakouts only.
    S_year = S[S.date.dt.year==current_year].copy()

    # ------------------------------------------------------------------
    # CONSTRAINED PORTFOLIO SIMULATION (change: Climbing/Played-out ->
    # Open Trades/Closed Trades). Previously, "Climbing" meant "every
    # signal that hasn't stopped out yet" -- completely ignoring position
    # sizing, so it counted signals as if the account had unlimited slots.
    # This runs a REAL day-by-day simulation with the same RS-ranked
    # slot allocation as vcm_engine.py's backtest, for each candidate
    # MAX_POSITIONS option, so "Open Trades" reflects what would actually
    # be held right now with that many slots -- and Fresh signals get
    # tagged with whether they actually got a slot ("Traded") or not
    # ("No trade -- book full") for the currently selected combo.
    # ------------------------------------------------------------------
    MAX_POSITIONS_OPTIONS = [5, 10, 15]

    def simulate_constrained_portfolio(signals_records, trail_dma, stop_pct, max_positions):
        """Walks all of this year's Base-1 signals in date order, applying
        the close-based stop / trail-activates-above exit rule for the
        given (trail_dma, stop_pct), admitting new entries RS-highest-first
        whenever slots are scarce -- identical logic to vcm_engine.py's
        run_variant(), just for one combo of the same 27-combination grid.
        Returns {(symbol, entry_date_str): outcome_dict}."""
        by_date = {}
        for s in signals_records:
            by_date.setdefault(s["date"], []).append(s)
        if not by_date:
            return {}
        for d in by_date:
            by_date[d].sort(key=lambda s: -(s["rs_pct"] if s["rs_pct"] is not None else 0))

        any_px = next(iter(px_store.values()))
        calendar = any_px.index[(any_px.index >= min(by_date.keys())) & (any_px.index <= end_date)]

        open_positions = []
        outcomes = {}
        ma_col = f"ma{trail_dma}"
        stop_frac = stop_pct / 100.0

        for day in calendar:
            still_open = []
            for p in open_positions:
                sym = p["symbol"]; px = px_store.get(sym)
                if px is None or day not in px.index:
                    still_open.append(p); continue
                row = px.loc[day]
                cl = float(row["close"])
                ma = float(row[ma_col]) if ma_col in row and pd.notna(row[ma_col]) else np.nan
                reason = None
                if cl <= p["stop"]:
                    reason = f"{stop_pct:.0f}% close stop"
                elif not p["trail"] and np.isfinite(ma) and cl > ma:
                    p["trail"] = True
                elif p["trail"] and np.isfinite(ma) and cl < ma:
                    reason = f"closed below {trail_dma}DMA"
                if reason:
                    outcomes[(sym, p["entry_date"])] = dict(
                        status="closed", exit_date=str(day.date()), exit_price=round(cl,2),
                        exit_reason=reason, return_pct=round((cl/p["entry_price"]-1)*100,2))
                else:
                    p["last"] = cl
                    still_open.append(p)
            open_positions = still_open

            for s in by_date.get(day, []):
                key = (s["symbol"], str(day.date()))
                if len(open_positions) >= max_positions:
                    outcomes[key] = dict(status="skipped")
                    continue
                # Same rule as vcm_engine.py's change #9: never hold two
                # simultaneous positions in one symbol, even if it's a
                # genuinely separate Base-1 signal after a 200DMA reset.
                if any(p["symbol"]==s["symbol"] for p in open_positions):
                    outcomes[key] = dict(status="skipped")
                    continue
                open_positions.append(dict(symbol=s["symbol"], entry_date=str(day.date()),
                                            entry_price=s["close"], stop=s["close"]*(1-stop_frac),
                                            trail=False, last=s["close"]))

        for p in open_positions:
            key = (p["symbol"], p["entry_date"])
            outcomes[key] = dict(status="open", current_price=round(p["last"],2),
                                  gain_pct=round((p["last"]/p["entry_price"]-1)*100,2))
        return outcomes

    signal_records = [
        dict(symbol=r.symbol, date=r.date, close=float(r.close), pivot=float(r.pivot),
             rs_pct=float(r.rs_pct) if pd.notna(r.rs_pct) else None, age_sessions=r.age_sessions,
             base_start=r.base_start, market_cap_cr=r.market_cap_cr)
        for _, r in S_year.iterrows() if px_store.get(r.symbol) is not None
    ]

    print("Simulating full 27-combination grid (trail x stop x max_positions)...")
    outcomes_by_combo = {}
    for trail in TRAILS:
        for stop in STOPS:
            for mp in MAX_POSITIONS_OPTIONS:
                outcomes_by_combo[(trail, stop, mp)] = simulate_constrained_portfolio(signal_records, trail, stop, mp)

    # COMBINED MIX (25/4/5 + 50/4/5, 10 positions total): the site's own
    # default view for Bull Flag. Each sleeve is genuinely independent
    # (its own 5-slot book, own exit rule) -- this does NOT force the
    # count to always show 10; it reports however many DISTINCT symbols
    # are actually open across both sleeves right now, which can be
    # fewer than 10 if the two sleeves happen to hold overlapping
    # symbols. "Open" wins if open in EITHER sleeve; otherwise "closed"
    # wins if closed in either.
    print("Simulating combined 25/4/5 + 50/4/5 mix (10 positions total)...")
    sleeve_a_outcomes = outcomes_by_combo[(25, 4, 5)]
    sleeve_b_outcomes = outcomes_by_combo[(50, 4, 5)]
    mix_outcomes = {}
    all_keys = set(sleeve_a_outcomes.keys()) | set(sleeve_b_outcomes.keys())
    for key in all_keys:
        oa = sleeve_a_outcomes.get(key, {"status":"skipped"})
        ob = sleeve_b_outcomes.get(key, {"status":"skipped"})
        if oa.get("status")=="open":
            mix_outcomes[key] = oa
        elif ob.get("status")=="open":
            mix_outcomes[key] = ob
        elif oa.get("status")=="closed":
            mix_outcomes[key] = oa
        elif ob.get("status")=="closed":
            mix_outcomes[key] = ob
        else:
            mix_outcomes[key] = oa
    outcomes_by_combo["mix_25_50"] = mix_outcomes

    signals_2026 = []
    for rec in signal_records:
        entry_date_str = str(rec["date"].date())
        base_entry = dict(
            symbol=rec["symbol"], entry_date=entry_date_str, entry_price=round(rec["close"],2),
            pivot=round(rec["pivot"],2), rs_pct=round(rec["rs_pct"],1) if rec["rs_pct"] is not None else None,
            base_weeks=round(rec["age_sessions"]/5,1), base_start=str(pd.Timestamp(rec["base_start"]).date()),
            base_end_date=entry_date_str, base_number=1,
            market_cap_cr=round(float(rec["market_cap_cr"]),1) if pd.notna(rec["market_cap_cr"]) else None,
        )
        outcomes = {}
        for trail in TRAILS:
            for stop in STOPS:
                for mp in MAX_POSITIONS_OPTIONS:
                    o = outcomes_by_combo[(trail, stop, mp)].get((rec["symbol"], entry_date_str), {"status": "unknown"})
                    outcomes[f"{trail}_{stop}_{mp}"] = o
        outcomes["mix_25_50"] = outcomes_by_combo["mix_25_50"].get((rec["symbol"], entry_date_str), {"status": "unknown"})
        # Base high/low weren't carried through signal_records above --
        # pull them back from S_year for this exact row.
        match = S_year[(S_year.symbol==rec["symbol"]) & (S_year.date==rec["date"])]
        if len(match):
            mrow = match.iloc[0]
            base_entry["base_high"] = round(float(mrow.base_high),2) if pd.notna(mrow.base_high) else None
            base_entry["base_low"] = round(float(mrow.base_low),2) if pd.notna(mrow.base_low) else None
        is_today = entry_date_str == str(end_date.date())
        signals_2026.append({**base_entry, "is_fresh_today": is_today, "outcomes": outcomes})

    forming_sorted = sorted(all_forming, key=lambda x: x["dist_to_pivot_pct"])[:60]
    true_forming_count = len(all_forming)

    # Attach chart history for whatever actually gets displayed --
    # Forming (top 60 shown) and all Fresh/Climbing (small lists).
    # Skipped for Played-out (1000+ entries -- would bloat the JSON for
    # a stage that's mostly for reference, not action).
    PRE_BASE_CONTEXT = 50  # sessions of context to guarantee before base_start_date
    HISTORY_DIR = DATA / "history"
    HISTORY_DIR.mkdir(exist_ok=True)
    written_symbols = set()  # avoid re-writing the same symbol's file multiple times

    def write_symbol_history_file(symbol, full):
        """Writes ONE file per symbol, containing its FULL available price
        history (not a per-signal lookback window) -- deduplicated, so a
        symbol with multiple signals only gets written once. Individual
        files stay small (one stock's several years of daily bars is a
        few hundred KB at most); the point is removing this from the
        main JSON entirely, not trimming it further."""
        if symbol in written_symbols:
            return
        written_symbols.add(symbol)
        hist = [
            {"date": str(idx.date()), "o": round(float(row["open"]),2),
             "h": round(float(row["high"]),2), "l": round(float(row["low"]),2),
             "c": round(float(row["close"]),2), "v": int(row["volume"])}
            for idx, row in full.iterrows()
        ]
        with open(HISTORY_DIR / f"{symbol}.json", "w") as f:
            json.dump(hist, f)

    def attach_history(entries, min_lookback=280):
        for e in entries:
            px = px_store.get(e["symbol"])
            if px is None:
                e["history_url"] = None
                e["blue_sky"] = None
                e["median_turnover_cr"] = None
                continue
            full = px[px.index <= end_date]

            write_symbol_history_file(e["symbol"], full)
            # Reference, not embedded data -- the frontend fetches this
            # file only when the person actually opens this stock's
            # chart, instead of every signal's full history being
            # downloaded upfront on page load.
            e["history_url"] = f"history/{e['symbol']}.json"
            # "Blue sky": current close at or within 1% of its ALL-TIME high

            # across the full available history (not just the embedded
            # chart window) -- no overhead resistance above current price.
            all_time_high = float(full["high"].max())
            latest_close = float(full["close"].iloc[-1])
            e["blue_sky"] = bool(all_time_high > 0 and latest_close >= all_time_high * 0.99)
            # Median rupee turnover over the last 20 sessions, in crore --
            # same definition vcm_engine.py uses for the liquidity filter.
            turnover_series = (full["close"] * full["volume"] / 1e7).tail(20)
            e["median_turnover_cr"] = round(float(turnover_series.median()),2) if len(turnover_series) else None

    attach_history(forming_sorted)

    # Only embed OHLC history for signals that can actually be selected
    # in a chart -- Fresh (today), or Open/Closed under ANY of the
    # user-selectable Max Positions options. A signal skipped for
    # book-full at every option never appears in any visible bucket and
    # would just bloat the JSON for nothing (this was the cause of the
    # file exceeding GitHub's 25MB web-upload limit -- with 1000+
    # signals/year, attaching history to all of them was wasteful).
    all_combo_keys = [f"{t}_{s}_{m}" for t in TRAILS for s in STOPS for m in MAX_POSITIONS_OPTIONS]
    chartable_signals = [
        s for s in signals_2026
        if s["is_fresh_today"] or any(
            s["outcomes"][k]["status"] in ("open", "closed") for k in all_combo_keys
        )
    ]
    # 5Y-capable history for all chartable signals -- Open/Fresh AND
    # Closed, per explicit request that the 5Y option be available for
    # closed trades too, not just active positions.
    attach_history(chartable_signals, min_lookback=1260)
    print(f"History attached to {len(chartable_signals)} of {len(signals_2026)} signals "
          f"(all get 5Y-capable history) "
          f"(rest were skipped for book-full at every combo and are never charted).")

    # Default combo for the initial page load matches vcm_engine.py's
    # default (25 DMA / 8% stop / 10 positions). The frontend recomputes
    # these instantly client-side when any of the three selectors change,
    # using the outcomes already embedded per signal -- no re-fetch needed.
    # Bull Flag's default view is the COMBINED 25/4/5 + 50/4/5 mix (10
    # positions total across two independent 5-slot sleeves) -- chosen
    # for lowest correlation with VCM while still tracking two combos
    # each already verified to perform well standalone. See
    # bull_flag_25_50_mix.py for the standalone version of this exact
    # simulation and its own backtest numbers.
    default_key = "mix_25_50"
    open_count = sum(1 for s in signals_2026 if s["outcomes"][default_key]["status"] == "open")
    closed_count = sum(1 for s in signals_2026 if s["outcomes"][default_key]["status"] == "closed")
    fresh_count = sum(1 for s in signals_2026 if s["is_fresh_today"])

    result = {
        "date": str(end_date.date()),
        "year": current_year,
        "trail_options": TRAILS,
        "stop_options": STOPS,
        "max_positions_options": MAX_POSITIONS_OPTIONS,
        "default_combo": {"trail_dma": None, "stop_pct": None, "max_positions": 10,
                           "note": "Combined 25 DMA/4%/5 + 50 DMA/4%/5 mix -- two independent sleeves, 10 positions total"},
        "default_combo_method": "Fixed -- chosen for lowest correlation with VCM (0.214), not a rolling pick",
        "default_combo_active_period": "N/A (fixed combo)",
        "counts": {
            "forming": true_forming_count, "fresh": fresh_count,
            "open": open_count, "closed": closed_count,
        },
        "forming_list_truncated": true_forming_count > len(forming_sorted),
        "stages": {
            "forming": forming_sorted,
        },
        # Every this-year signal, each carrying its own outcome for all 27
        # (trail, stop, max_positions) combos -- keyed "trail_stop_maxpos",
        # e.g. "25_8_10". The frontend buckets these into Fresh/Open/Closed
        # based on the three selected dropdowns, instantly, no re-fetch.
        "signals_2026": sorted(signals_2026, key=lambda x: -(x["rs_pct"] or 0)),
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\nForming: {true_forming_count} | Fresh today: {fresh_count} | "
          f"Open (max_pos=10): {open_count} | Closed 2026 (max_pos=10): {closed_count} | "
          f"Total signals this year: {len(signals_2026)}")
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
