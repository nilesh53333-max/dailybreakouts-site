"""
BREAKOUT LIFECYCLE SCANNER
==========================================================
Classifies every qualifying setup into one of four stages, matching
the confirmed BananaPatterns 4-stage lifecycle:

    Forming     -- an active base, meeting all quality gates, that
                   hasn't cleared its pivot yet.
    Fresh       -- cleared its pivot within the last 5 trading sessions
                   and hasn't been stopped out or trail-exited since.
    Climbing    -- broke out earlier this calendar year, still open
                   (hasn't hit the stop or trailing-MA exit yet).
    Played out  -- broke out this calendar year and has already exited
                   (stop or trailing-MA), win or loss.

Uses the SAME detection logic as vcm_engine.py (mcap point-in-time,
RS>=70, trend/position/turnover/depth, Base-1 restriction, same-day
dedup) so "what this screen shows" and "what the backtest would have
flagged" are always the same signal computed the same way.

ASSUMPTION (states clearly, change EXIT_TRAIL_DMA / EXIT_STOP_PCT
below once you lock a final config): Climbing vs Played-out uses a
single exit rule to simulate forward from each breakout to today.
Currently set to your leading candidate: 25-day trailing MA, 8% stop.

Output: ../data/breakout_lifecycle.json -- consumed by the website.

Usage:
    python update_universe_daily.py
    python breakout_lifecycle.py
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
OUTPUT_PATH=DATA/"breakout_lifecycle.json"

START=pd.Timestamp("2020-01-01")
MARKETCAP_MIN_CR = 500.0
FRESH_SESSION_WINDOW = 5   # "cleared it in the last N sessions"

# ASSUMPTION -- change once your final config is locked.
EXIT_TRAIL_DMA = 25
EXIT_STOP_PCT = 8.0

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


def compute_all(p, end_date):
    """Returns (breakout_events, still_forming_bases, px_frame, rs_series)
    for one symbol. breakout_events includes every Base-1 candidate
    breakout up through end_date (not just today) so we can classify
    older ones as Climbing/Played-out. still_forming_bases lists any
    currently-active, quality-passing base that hasn't broken out yet."""
    sym=p.stem; d=load(p)
    if d is None:return [],[],None,None
    c=d.close.to_numpy(float);h=d.high.to_numpy(float);l=d.low.to_numpy(float);v=d.volume.to_numpy(float)
    n=len(d); dates=d.date
    ma25=pd.Series(c).rolling(25).mean().to_numpy()
    ma50=pd.Series(c).rolling(50).mean().to_numpy()
    ma100=pd.Series(c).rolling(100).mean().to_numpy()
    ma200=pd.Series(c).rolling(200).mean().to_numpy()
    h52=pd.Series(h).rolling(252,min_periods=100).max().to_numpy()
    l52=pd.Series(l).rolling(252,min_periods=100).min().to_numpy()
    turn20=pd.Series(c*v/1e7).rolling(20).median().to_numpy()
    vol20=pd.Series(v).rolling(20,min_periods=5).mean().to_numpy()
    pc=np.r_[np.nan,c[:-1]]
    tr=np.nanmax(np.vstack([h-l,np.abs(h-pc),np.abs(l-pc)]),axis=0)
    atrp=tr/c*100
    rh20=pd.Series(h).shift(1).rolling(20,min_periods=1).max().to_numpy()
    rmax={z:pd.Series(c).shift(1).rolling(z,min_periods=1).max().to_numpy() for z in (5,10,20,40,60)}
    peak=np.zeros(n,bool);peak[1:-1]=(c[1:-1]>=c[:-2])&(c[1:-1]>=c[2:])
    recent=deque(); active=[]; out=[]

    s=pd.Series(c)
    r63=s/s.shift(63)-1
    q2=s.shift(63)/s.shift(126)-1
    q3=s.shift(126)/s.shift(189)-1
    q4=s.shift(189)/s.shift(252)-1
    rsraw=.4*r63+.2*q2+.2*q3+.2*q4
    rsdf=pd.DataFrame({"date":dates,"symbol":sym,"rsraw":rsraw})
    mcap=mcap_for(sym)

    for i in range(6,n):
        j=i-2
        if j>=1 and peak[j]:recent.append(j)
        while recent and recent[0]<i-60:recent.popleft()

        keep=[]
        for a in active:
            pv=a["pivot"]
            if c[i]>pv:
                age=i-a["s"]
                if age>=15 and START<=dates.iloc[i]<=end_date:
                    depth=(pv-np.min(l[a["s"]:i+1]))/pv*100
                    L=i-a["s"]+1;m=max(1,L//2)
                    a1=np.nanmean(atrp[a["s"]:a["s"]+m]);a2=np.nanmean(atrp[a["s"]+m:i+1])
                    coil=a2/a1 if a1>0 else np.nan
                    structure=depth<=35
                    trend=i>=200 and c[i]>=ma50[i] and c[i]>=ma200[i]
                    position=np.isfinite(h52[i]) and np.isfinite(l52[i]) and (h52[i]-c[i])/h52[i]*100<=30 and (c[i]-l52[i])/l52[i]*100>=15
                    turnover=np.isfinite(turn20[i]) and turn20[i]>=5
                    vol_x = v[i]/vol20[i] if np.isfinite(vol20[i]) and vol20[i]>0 else np.nan
                    out.append(dict(symbol=sym,date=dates.iloc[i],pivot=pv,base_start=dates.iloc[a["s"]],
                        close=c[i],age_sessions=age,depth=depth,coil=coil,vol_x=vol_x,
                        structure_pass=structure,trend_pass=trend,position_pass=position,
                        turnover_pass=turnover,market_cap_cr=mcap,row_idx=i))
                continue
            keep.append(a)
        active=keep

        if i<1:continue
        if not np.isfinite(rh20[i]) or c[i-1]<rh20[i]*.92:continue
        raw=[(c[i-1],1)]
        for z in (5,10,20,40,60):
            if np.isfinite(rmax[z][i]):raw.append((float(rmax[z][i]),z))
        for pj in recent:raw.append((float(c[pj]),i-pj))
        raw.sort(); seeds=[]
        for z in raw:
            if seeds and abs(z[0]/seeds[-1][0]-1)*100<=.25:continue
            seeds.append(z)
        prev=c[i-1]
        for pv,age in seeds:
            dist=(pv/prev-1)*100
            if -1<=dist<=8 and not any(abs(a["pivot"]/pv-1)*100<=.25 and abs(a["s"]-i)<=3 for a in active):
                active.append({"pivot":pv,"s":i})

    # Still-forming bases: whatever's left in `active` at end_date that
    # already qualifies on age + quality gates (everything except the
    # close>pivot trigger itself, since it hasn't happened yet).
    forming=[]
    last_i = n-1
    if dates.iloc[last_i] == end_date:
        for a in active:
            age = last_i - a["s"]
            if age < 15: continue
            depth=(a["pivot"]-np.min(l[a["s"]:last_i+1]))/a["pivot"]*100
            if depth>35: continue
            if not (last_i>=200 and c[last_i]>=ma50[last_i] and c[last_i]>=ma200[last_i]): continue
            if not (np.isfinite(h52[last_i]) and np.isfinite(l52[last_i]) and
                    (h52[last_i]-c[last_i])/h52[last_i]*100<=30 and
                    (c[last_i]-l52[last_i])/l52[last_i]*100>=15): continue
            if not (np.isfinite(turn20[last_i]) and turn20[last_i]>=5): continue
            if not (np.isfinite(mcap) and mcap>=MARKETCAP_MIN_CR): continue
            dist_to_pivot = (a["pivot"]/c[last_i]-1)*100
            forming.append(dict(symbol=sym, pivot=a["pivot"], close=float(c[last_i]),
                                 dist_to_pivot_pct=round(dist_to_pivot,2),
                                 base_start=dates.iloc[a["s"]], age_sessions=age))

    px=d.set_index("date")[["open","high","low","close"]].copy()
    px[f"ma{EXIT_TRAIL_DMA}"]=pd.Series(c).rolling(EXIT_TRAIL_DMA).mean().to_numpy()
    return out, forming, px, rsdf


def simulate_exit(px, entry_date, entry_price, end_date):
    """Forward-simulates ONE position using the same close-based stop +
    trailing-MA-activates-above logic as vcm_engine.py's run_variant(),
    from entry_date through end_date. Returns (still_open, exit_date,
    exit_price, exit_reason, return_pct_now_or_at_exit)."""
    ma_col = f"ma{EXIT_TRAIL_DMA}"
    stop = entry_price * (1 - EXIT_STOP_PCT/100.0)
    trail = False
    window = px.loc[(px.index>entry_date)&(px.index<=end_date)]
    last_close = entry_price
    for day, row in window.iterrows():
        cl = float(row["close"])
        ma = float(row[ma_col]) if pd.notna(row[ma_col]) else np.nan
        last_close = cl
        if cl <= stop:
            return False, day, cl, f"{EXIT_STOP_PCT:.0f}% close stop", (cl/entry_price-1)*100
        if not trail and np.isfinite(ma) and cl > ma:
            trail = True
        elif trail and np.isfinite(ma) and cl < ma:
            return False, day, cl, f"closed below {EXIT_TRAIL_DMA}DMA", (cl/entry_price-1)*100
    return True, None, last_close, None, (last_close/entry_price-1)*100


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
    print(f"Exit rule for Climbing/Played-out: {EXIT_TRAIL_DMA}DMA trail / {EXIT_STOP_PCT}% stop")

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

    # Base-1 restriction, same logic as vcm_engine.py / todays_breakouts.py
    S["base_number"]=np.nan
    for sym, idxs in S.groupby("symbol").groups.items():
        g=S.loc[list(idxs)].sort_values(["base_start","date","pivot"]).copy()
        px=px_store.get(sym)
        if px is None: continue
        ma200_full=px["close"].rolling(200).mean()
        stage=0; last_breakout=None
        for ridx,r in g.iterrows():
            bs=pd.Timestamp(r.base_start); bd=pd.Timestamp(r.date)
            same_base=(last_breakout is not None and bs<=last_breakout)
            if stage==0:
                stage=1; last_breakout=bd
            elif not same_base:
                reset=False
                if last_breakout is not None:
                    window=px.loc[(px.index>last_breakout)&(px.index<=bs)]
                    if len(window):
                        wma=ma200_full.reindex(window.index)
                        reset=bool(((window["close"]<wma)&wma.notna()).any())
                stage = 1 if reset else stage+1
                last_breakout=bd
            else:
                if bd<last_breakout: last_breakout=bd
            S.at[ridx,"base_number"]=stage
    S=S[S.base_number==1].copy()

    # This calendar year's breakouts only.
    S_year = S[S.date.dt.year==current_year].copy()

    fresh=[]; climbing=[]; played_out=[]

    for _,r in S_year.iterrows():
        sym=r.symbol; px=px_store.get(sym)
        if px is None: continue
        still_open, exit_date, exit_price, exit_reason, ret_pct = simulate_exit(px, r.date, r.close, end_date)

        sessions_since = (px.index>r.date) & (px.index<=end_date)
        sessions_since_count = int(sessions_since.sum())

        base_entry = dict(
            symbol=sym, entry_date=str(r.date.date()), entry_price=round(float(r.close),2),
            pivot=round(float(r.pivot),2), rs_pct=round(float(r.rs_pct),1) if pd.notna(r.rs_pct) else None,
            base_weeks=round(r.age_sessions/5,1),
        )

        if not still_open:
            played_out.append({**base_entry, "exit_date": str(exit_date.date()),
                                "exit_price": round(exit_price,2), "exit_reason": exit_reason,
                                "return_pct": round(ret_pct,2)})
        elif sessions_since_count <= FRESH_SESSION_WINDOW:
            fresh.append({**base_entry, "current_price": round(exit_price,2),
                          "gain_pct": round(ret_pct,2), "sessions_since_entry": sessions_since_count})
        else:
            climbing.append({**base_entry, "current_price": round(exit_price,2),
                             "gain_pct": round(ret_pct,2), "sessions_since_entry": sessions_since_count})

    forming_sorted = sorted(all_forming, key=lambda x: x["dist_to_pivot_pct"])[:60]
    true_forming_count = len(all_forming)

    result = {
        "date": str(end_date.date()),
        "year": current_year,
        "exit_rule": {"trail_dma": EXIT_TRAIL_DMA, "stop_pct": EXIT_STOP_PCT},
        "counts": {
            "forming": true_forming_count, "fresh": len(fresh),
            "climbing": len(climbing), "played_out": len(played_out),
        },
        "forming_list_truncated": true_forming_count > len(forming_sorted),
        "stages": {
            "forming": forming_sorted,
            "fresh": sorted(fresh, key=lambda x: -(x["rs_pct"] or 0)),
            "climbing": sorted(climbing, key=lambda x: -(x["rs_pct"] or 0)),
            "played_out": sorted(played_out, key=lambda x: x["exit_date"], reverse=True),
        }
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\nForming: {len(forming_sorted)} | Fresh: {len(fresh)} | "
          f"Climbing: {len(climbing)} | Played out: {len(played_out)}")
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
