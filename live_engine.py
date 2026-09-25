"""
live_engine.py -- ONE code path for backtest, website and live trading
======================================================================
Frozen live rules (Sep 2026):
  - Decide, enter and exit at 15:15 (uses the 3:15-world data).
  - Each strategy has its own book: starts at Rs 10L, 15 slots, COMPOUNDING
    (each new position = current book value / 15), same-day signals ranked
    by RS, one trade per stock at a time.
  - VCM, Bull Flag, Cup & Handle: ALL9 -- exit set (trail 25/50/100 DMA x
    close stop 4/8/12%) fixed by a hash of symbol + entry date.
  - Double Bottom: 100 DMA trail / 8% close stop.
  - Exit: close <= stop, or (trail active and close < trail DMA). Trail
    becomes active once close > trail DMA.

Everything else (website JSON, live algo) must import from here, so the
rules can never drift apart.

Usage (backend folder):
    py live_engine.py                 # use saved 3:15 signals (exp1515_signals_*.pkl)
    py live_engine.py --rebuild       # rebuild universe_1515_stitched from universe_1515 first
    py live_engine.py --rescan        # rescan signals from the stitched data (VCM/BF ~10 min, CH ~80 min)
    py live_engine.py --only VCM,DB
Output: ../data/live_state.json  (summary, yearly, equity, open positions,
        today's entries/exits, 2026 closed trades -- per strategy + portfolio)
"""
import argparse
import hashlib
import json
import time
import warnings
from datetime import date

import numpy as np
import pandas as pd

import unlimited_positions_experiment as base
import backtest_1515_experiment as b15

warnings.filterwarnings("ignore")
DATA = base.DATA
START_CAPITAL = 1_000_000.0
MAX_POS = 15
BUY, SELL = base.BUY_RATE, base.SELL_RATE
ALL9 = [(t, s) for t in (25, 50, 100) for s in (4, 8, 12)]
RULES = {"VCM": "ALL9", "BF": "ALL9", "CH": "ALL9", "DB": (100, 8)}
NAMES = {"VCM": "Volatility Contraction", "BF": "Classic Bull Flag", "CH": "Cup and Handle", "DB": "Double Bottom"}


# ------------------------------------------------------------------ rules
def all9_set(symbol, day):
    """Same (symbol, entry date) -> same exit set, on every machine and run."""
    key = f"{symbol}|{pd.Timestamp(day):%Y-%m-%d}".encode()
    return ALL9[int(hashlib.md5(key).hexdigest(), 16) % 9]


def exit_set(strategy, symbol, day):
    rule = RULES[strategy]
    return all9_set(symbol, day) if rule == "ALL9" else rule


def rule_text(strategy):
    r = RULES[strategy]
    return "ALL9: trail 25/50/100 DMA x stop 4/8/12%, set fixed by stock + date" if r == "ALL9" \
        else f"{r[0]} DMA trail / {r[1]}% close stop"


# ------------------------------------------------------------------ book
def simulate_book(strategy, S, P, end, start=base.SIM_START, capital=START_CAPITAL, compound=True):
    """Runs the book day by day. Returns (trades DataFrame, equity Series,
    open positions list). Identical logic is used for the live decision."""
    S = S[(S.date >= start) & (S.date <= end) & S.symbol.isin(P.keys())]
    by_date = {d: list(g.symbol) for d, g in S.sort_values("rs_pct", ascending=False).groupby("date")}
    cal = sorted({d for p in P.values() for d in p["dates"] if start <= d <= end})
    cash = capital
    pos, trades, eq = [], [], []
    for day in cal:
        keep = []
        for p in pos:
            sp = P[p["symbol"]]
            i = sp["pos"].get(day)
            if i is None:
                keep.append(p); continue
            cl = sp["close"][i]
            ma = sp[f"ma{p['trail']}"][i]
            p["trail_dma_value"] = float(ma) if np.isfinite(ma) else None
            reason = None
            if cl <= p["stop_px"]:
                reason = f"{p['stop']}% stop"
            elif not p["trail_on"] and np.isfinite(ma) and cl > ma:
                p["trail_on"] = True
            elif p["trail_on"] and np.isfinite(ma) and cl < ma:
                reason = f"closed below {p['trail']} DMA"
            if reason:
                proceeds = p["shares"] * cl * (1 - SELL)
                cash += proceeds
                pnl = proceeds - p["cost"]
                trades.append(dict(symbol=p["symbol"], set=f"{p['trail']}/{p['stop']}",
                                   entry_date=p["entry_date"], entry_price=p["entry"], shares=p["shares"],
                                   exit_date=day, exit_price=cl, exit_reason=reason,
                                   pnl=pnl, return_pct=pnl / p["cost"] * 100, status="closed"))
            else:
                p["last"] = cl
                keep.append(p)
        pos = keep

        value = cash + sum(p["shares"] * p["last"] * (1 - SELL) for p in pos)
        alloc = (value if compound else capital) / MAX_POS
        held = {p["symbol"] for p in pos}
        for sym in by_date.get(day, []):
            if sym in held or len(pos) >= MAX_POS:
                continue
            sp = P[sym]
            i = sp["pos"].get(day)
            if i is None:
                continue
            cl = sp["close"][i]
            trail, stop = exit_set(strategy, sym, day)
            sh = int(alloc // cl)
            cost = sh * cl * (1 + BUY)
            if cost > cash:
                sh = int(max(cash, 0) // (cl * (1 + BUY)))
                cost = sh * cl * (1 + BUY)
            if sh <= 0:
                continue
            cash -= cost
            ma = sp[f"ma{trail}"][i]
            pos.append(dict(symbol=sym, entry_date=day, entry=cl, shares=sh, cost=cost, trail=trail, stop=stop,
                            stop_px=cl * (1 - stop / 100), trail_on=bool(np.isfinite(ma) and cl > ma),
                            trail_dma_value=float(ma) if np.isfinite(ma) else None, last=cl))
            held.add(sym)
        eq.append((day, cash + sum(p["shares"] * p["last"] * (1 - SELL) for p in pos)))
    return pd.DataFrame(trades), pd.Series(dict(eq)).sort_index(), pos


# ------------------------------------------------------------------ metrics
def yearly_returns(E, capital, compound=True):
    ye = E.groupby(E.index.year).last()
    start = pd.Series([capital] + list(ye.values[:-1]), index=ye.index)
    return ((ye - start) / (start if compound else capital) * 100).round(2)


def cagr(E, capital):
    yrs = (E.index[-1] - E.index[0]).days / 365.25
    return round(((E.iloc[-1] / capital) ** (1 / yrs) - 1) * 100, 2)


def max_dd(E):
    return round(float((E / E.cummax() - 1).min() * 100), 2)


def summarize(strategy, T, E, capital=START_CAPITAL):
    closed = T[T.status == "closed"] if len(T) else T
    wins, losses = closed[closed.pnl > 0], closed[closed.pnl <= 0]
    return dict(strategy=strategy, name=NAMES.get(strategy, strategy), rule=rule_text(strategy) if strategy in RULES else "",
                start_capital=capital, end_value=round(float(E.iloc[-1]), 2),
                total_return_pct=round((E.iloc[-1] / capital - 1) * 100, 2), cagr_pct=cagr(E, capital),
                max_dd_pct=max_dd(E), closed_trades=int(len(closed)),
                win_rate_pct=round((closed.pnl > 0).mean() * 100, 2) if len(closed) else None,
                profit_factor=round(wins.pnl.sum() / -losses.pnl.sum(), 2) if len(losses) and losses.pnl.sum() else None)


# ------------------------------------------------------------------ state for website / live
def _d(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else (x.strftime("%Y-%m-%d") if hasattr(x, "strftime") else x)


def book_state(strategy, T, E, pos, as_of, S=None):
    value = float(E.iloc[-1])
    open_rows = []
    for p in sorted(pos, key=lambda q: q["entry_date"]):
        open_rows.append(dict(symbol=p["symbol"], set=f"{p['trail']}/{p['stop']}", trail_dma=p["trail"], stop_pct=p["stop"],
                              entry_date=_d(p["entry_date"]), entry_price=round(p["entry"], 2), shares=int(p["shares"]),
                              last_price=round(p["last"], 2), stop_price=round(p["stop_px"], 2),
                              trail_active=bool(p["trail_on"]),
                              trail_dma_value=round(p["trail_dma_value"], 2) if p["trail_dma_value"] else None,
                              gain_pct=round((p["last"] / p["entry"] - 1) * 100, 2),
                              weight_pct=round(p["shares"] * p["last"] / value * 100, 2)))
    closed = T[T.status == "closed"] if len(T) else T
    yr = pd.Timestamp(as_of).year
    closed_year = closed[pd.to_datetime(closed.exit_date).dt.year == yr] if len(closed) else closed
    rows = lambda df: [dict(symbol=r.symbol, set=r.set, entry_date=_d(r.entry_date), entry_price=round(r.entry_price, 2),
                            exit_date=_d(r.exit_date), exit_price=round(r.exit_price, 2), exit_reason=r.exit_reason,
                            return_pct=round(r.return_pct, 2)) for r in df.itertuples()]
    exits_today = closed[pd.to_datetime(closed.exit_date) == pd.Timestamp(as_of)] if len(closed) else closed
    entries_today = [r for r in open_rows if r["entry_date"] == _d(pd.Timestamp(as_of))]
    # Every signal of the day, and what the book did with it (website "Fresh").
    signals_today = []
    if S is not None:
        taken = {r["symbol"] for r in entries_today}
        held_before = {r["symbol"] for r in open_rows if r["entry_date"] != _d(pd.Timestamp(as_of))}
        day = S[S.date == pd.Timestamp(as_of)].sort_values("rs_pct", ascending=False)
        for r in day.itertuples():
            status = "taken" if r.symbol in taken else ("already held" if r.symbol in held_before else "book full")
            signals_today.append(dict(symbol=r.symbol, rs_pct=round(float(r.rs_pct), 2) if pd.notna(r.rs_pct) else None,
                                      status=status))
    return dict(value=round(value, 2), open_positions=open_rows, entries_today=entries_today,
                signals_today=signals_today,
                exits_today=rows(exits_today), closed_this_year=rows(closed_year.sort_values("exit_date", ascending=False)))


def equity_points(E, capital, every="W-FRI"):
    w = E.resample(every).last().dropna()
    return [dict(date=_d(d), value=round(float(v), 2), cum_pct=round((v / capital - 1) * 100, 2)) for d, v in w.items()]


def run(keys, rebuild=False, rescan=False):
    b15.configure("15:15")
    if rebuild:
        b15.build_stitched()
    out = dict(as_of=None, generated=date.today().isoformat(),
               rules=dict(time="15:15", max_positions=MAX_POS, start_capital_per_strategy=START_CAPITAL,
                          sizing="compounding: new position = current book value / 15",
                          ranking="RS percentile, highest first", one_trade_per_stock=True,
                          per_strategy={k: rule_text(k) for k in keys}),
               strategies={})
    curves = {}
    for key in keys:
        t0 = time.time()
        S, P, end = b15.load_1515(key, reuse=not rescan)
        T, E, pos = simulate_book(key, S, P, end)
        curves[key] = E
        as_of = E.index[-1]
        out["as_of"] = max(out["as_of"] or _d(as_of), _d(as_of))
        st = book_state(key, T, E, pos, as_of, S)
        out["strategies"][key] = dict(summary=summarize(key, T, E),
                                      yearly={int(k): v for k, v in yearly_returns(E, START_CAPITAL).items()},
                                      equity=equity_points(E, START_CAPITAL), as_of=_d(as_of), **st)
        s = out["strategies"][key]["summary"]
        print(f"{key}: CAGR {s['cagr_pct']}% | DD {s['max_dd_pct']}% | total {s['total_return_pct']}% | "
              f"open {len(st['open_positions'])} | entries today {len(st['entries_today'])} | "
              f"exits today {len(st['exits_today'])} | as of {_d(as_of)} ({(time.time()-t0)/60:.1f} min)")
    allE = pd.DataFrame(curves).sort_index().ffill().fillna(START_CAPITAL)
    port = allE.sum(axis=1)
    cap = START_CAPITAL * len(keys)
    out["portfolio"] = dict(summary=dict(start_capital=cap, end_value=round(float(port.iloc[-1]), 2),
                                         total_return_pct=round((port.iloc[-1] / cap - 1) * 100, 2),
                                         cagr_pct=cagr(port, cap), max_dd_pct=max_dd(port)),
                            yearly={int(k): v for k, v in yearly_returns(port, cap).items()},
                            equity=equity_points(port, cap))
    p = out["portfolio"]["summary"]
    print(f"PORTFOLIO: CAGR {p['cagr_pct']}% | DD {p['max_dd_pct']}% | total {p['total_return_pct']}%")
    with open(DATA / "live_state.json", "w", encoding="utf-8") as f:
        json.dump(out, f, separators=(",", ":"), default=str)
    print(f"Saved: {DATA / 'live_state.json'}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="VCM,BF,CH,DB")
    ap.add_argument("--rebuild", action="store_true", help="rebuild universe_1515_stitched first")
    ap.add_argument("--rescan", action="store_true", help="rescan signals instead of using saved ones")
    a = ap.parse_args()
    run([k.strip().upper() for k in a.only.split(",") if k.strip()], a.rebuild, a.rescan)
