"""
3:15 PM BACKTEST -- decide, enter and exit at 15:15 (all 4 strategies)
======================================================================
ONE-OFF RESEARCH SCRIPT. Reads data/universe_1515 (from
fetch_intraday_snapshots.py) and data/universe (read-only). Writes only
exp1515_* files and the data/universe_1515_stitched folder.

Needs in the same folder: unlimited_positions_experiment.py,
next_day_open_experiment.py (for the CLOSE baseline loaders), and the
engines (vcm_engine.py, bull_flag_alternating_trail_v1.py,
cup_and_handle_scan.py, double_bottom_v3_truew_scan.py).

The "3:15 world": every day's candle is the day as seen at 15:15
(open, high/low/volume up to 15:15, close = 15:15 price). The production
engines run on it UNCHANGED, so:
  - signals are decided on the 15:15 price (false breakouts that fade
    into the close are included; late breakouts are missed),
  - entry is at the 15:15 price, and stop / DMA exits use 15:15 prices,
  - volume rules compare 15:15 volume with other days' 15:15 volume.
Warm-up: rows BEFORE the first 15:15 date come from the daily files, so
200-DMA / 252-day RS are valid from Jan 2020. Data is cut at the last
daily date (drops a possibly partial "today").

Sanity check: a symbol whose day-to-day moves in the 15:15 file differ
wildly from the daily file (e.g. a split adjusted in one source only) is
excluded and listed.

Also reports the SIGNAL MATCH RATE: how many 15:15 signals are the same
(symbol, date) as the close-based signals, and vice versa.

Usage (backend folder):
    py backtest_1515_experiment.py                  # all 4 (C&H scan ~2 hours)
    py backtest_1515_experiment.py --only VCM,BF,DB # skip the long C&H scan
    py backtest_1515_experiment.py --reuse          # reuse saved 15:15 signals
    py backtest_1515_experiment.py --at 15:25       # test 15:25 (needs data/universe_1525)
    py backtest_1515_experiment.py --grid --reuse --no-rebuild   # all 27 combos, close vs 15:15
Outputs (../data): exp1515_summary.csv, exp1515_yearly.csv,
    exp1515_signal_match.csv, exp1515_excluded_symbols.csv
"""
import argparse
import os
import re
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import unlimited_positions_experiment as base
import next_day_open_experiment as nxt

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
DATA = base.DATA
# Set by configure(--at): which snapshot time is tested. 15:15 by default.
AT_TAG = "1515"
SNAP_DIR = DATA / "universe_1515"
STITCH_DIR = DATA / "universe_1515_stitched"
PREFIX = "exp1515"


def configure(at):
    global AT_TAG, SNAP_DIR, STITCH_DIR, PREFIX
    h, m = at.split(":")
    AT_TAG = f"{int(h):02d}{int(m):02d}"
    SNAP_DIR = DATA / f"universe_{AT_TAG}"
    STITCH_DIR = DATA / f"universe_{AT_TAG}_stitched"
    PREFIX = f"exp{AT_TAG}"
DAILY_DIR = DATA / "universe"
PERSIST_DIFF_PCT = 5.0  # after adjustment, a 10-day median gap this big -> genuine data problem
BIG_DAY_GAP_PCT = 10.0  # single days with a bigger 15:15-vs-close gap are KEPT (real moves), just counted
JUMP = np.log(1.08)    # a persistent >8% step in (daily close / 15:15 close) = corporate action


# ---------------------------------------------------------------- data
def _read(path):
    d = pd.read_csv(path)
    d.columns = [c.lower() for c in d.columns]
    d["date"] = pd.to_datetime(d["date"].astype(str).str[:10])
    return d.sort_values("date").drop_duplicates("date")


def adjust_to_daily(s, d):
    """Kite's daily files are split/bonus adjusted; the intraday files may not be.
    Find persistent level steps in (daily close / 15:15 close) and rescale each
    segment of the 15:15 data onto the daily price basis. Normal days (ratio ~1)
    are left untouched, so the genuine 15:15-vs-close difference is kept."""
    m = s[["date", "close"]].merge(d[["date", "close"]], on="date", suffixes=("_s", "_d"))
    if len(m) < 20:
        return s, 0
    lr = np.log(m["close_d"].to_numpy() / m["close_s"].to_numpy())
    n = len(lr)
    breaks = [0]
    for i in range(1, n):
        if abs(lr[i] - lr[i - 1]) > JUMP:
            before = np.median(lr[max(breaks[-1], i - 5):i])
            after = np.median(lr[i:min(n, i + 5)])
            if abs(after - before) > JUMP:
                breaks.append(i)
    breaks.append(n)
    factor = np.ones(n)
    for a, b in zip(breaks[:-1], breaks[1:]):
        med = np.median(lr[a:b])
        if abs(med) > np.log(1.02):          # only real level shifts, not daily noise
            factor[a:b] = np.exp(med)
    if np.all(factor == 1.0):
        return s, 0
    f = pd.Series(factor, index=m["date"]).reindex(s["date"]).ffill().bfill().to_numpy()
    s = s.copy()
    for c in ("open", "high", "low", "close"):
        s[c] = s[c] * f
    s["volume"] = s["volume"] / f
    return s, len(breaks) - 2


def build_stitched():
    STITCH_DIR.mkdir(exist_ok=True)
    snaps = sorted(SNAP_DIR.glob("*.csv"))
    if not snaps:
        raise SystemExit(f"No files in {SNAP_DIR} -- run fetch_intraday_snapshots.py first.")
    excluded = []
    written = adjusted = big_days = 0
    for p in snaps:
        sym = p.stem
        dpath = DAILY_DIR / f"{sym}.csv"
        if not dpath.exists():
            excluded.append((sym, "no daily file")); continue
        s, d = _read(p), _read(dpath)
        last_daily = d["date"].max()
        s = s[s["date"] <= last_daily]
        if len(s) < 50:
            excluded.append((sym, "too few 15:15 days")); continue
        s, n_breaks = adjust_to_daily(s, d)
        adjusted += bool(n_breaks)
        # Residual check (after adjustment): only a PERSISTENT gap between the
        # 15:15 and daily prices means a data problem. A single day where the
        # stock moved a lot after 15:15 (circuit, late news) is real and kept.
        m = s[["date", "close"]].merge(d[["date", "close"]], on="date", suffixes=("_s", "_d"))
        gap = (m["close_s"] / m["close_d"] - 1) * 100
        persistent = gap.rolling(10, min_periods=10).median().abs().max()
        if np.isfinite(persistent) and persistent > PERSIST_DIFF_PCT:
            day = m.loc[gap.rolling(10, min_periods=10).median().abs().idxmax(), "date"].date()
            excluded.append((sym, f"persistent price gap {persistent:.1f}% around {day}")); continue
        big_days += int((gap.abs() > BIG_DAY_GAP_PCT).sum())
        warm = d[d["date"] < s["date"].min()][["date", "open", "high", "low", "close", "volume"]]
        out = pd.concat([warm, s[["date", "open", "high", "low", "close", "volume"]]], ignore_index=True)
        out["date"] = out["date"].dt.strftime("%Y-%m-%d")
        out.to_csv(STITCH_DIR / f"{sym}.csv", index=False)
        written += 1
    for f in STITCH_DIR.glob("*.csv"):  # remove stale files from earlier runs
        if f.stem in {e[0] for e in excluded}:
            f.unlink()
    pd.DataFrame(excluded, columns=["symbol", "reason"]).to_csv(DATA / f"{PREFIX}_excluded_symbols.csv", index=False)
    print(f"Stitched {AT_TAG} universe: {written} symbols written ({adjusted} split/bonus-adjusted), "
          f"{len(excluded)} excluded (see {PREFIX}_excluded_symbols.csv) | "
          f"{big_days} single days with a >{BIG_DAY_GAP_PCT:g}% 15:15-vs-close gap kept as real")


def latest_stitched_date():
    """Newest trading date in the stitched folder (read from each file's last line)."""
    latest = None
    for f in STITCH_DIR.glob("*.csv"):
        with open(f, "rb") as fh:
            fh.seek(0, 2)
            fh.seek(max(0, fh.tell() - 200))
            tail = fh.read().decode(errors="ignore").strip().splitlines()
        if tail:
            d = tail[-1].split(",")[0][:10]
            if len(d) == 10 and (latest is None or d > latest):
                latest = d
    return latest


def _exec_until(filename, marker, exact=False):
    """Run an engine's scan code on the stitched 3:15 folder, stopping before
    its simulation / file-writing section."""
    path = HERE / filename
    src = path.read_text(encoding="utf-8")
    cut = src.find(marker)
    if cut < 0:
        raise RuntimeError(f"marker not found in {filename}: {marker!r}")
    if not exact:
        cut = src.rfind("\n# ---", 0, cut)
    code = src[:cut]
    for a in ('UDIR=DATA/"universe"', 'UDIR = DATA / "universe"'):
        code = code.replace(a, f'UDIR=DATA/"{STITCH_DIR.name}"')
    if STITCH_DIR.name not in code:
        raise RuntimeError(f"could not redirect UDIR in {filename}")
    # The engines hard-code END (e.g. 2026-09-15). Use the newest date in the
    # data instead, so daily runs always include the latest signals.
    latest = latest_stitched_date()
    if latest:
        code, n = re.subn(r'END\s*=\s*pd\.Timestamp\(\s*["\']\d{4}-\d{2}-\d{2}["\']\s*\)',
                          f'END = pd.Timestamp("{latest}")', code, count=1)
        print(f"  {filename}: END set to {latest}" if n else f"  WARNING: END not found in {filename}")
    ns = {"__file__": str(path), "__name__": f"engine_scan_{AT_TAG}"}
    exec(compile(code, str(path), "exec"), ns)
    return ns


def load_1515(key, reuse):
    cache = DATA / f"{PREFIX}_signals_{key}.pkl"
    if reuse and cache.exists():
        with open(cache, "rb") as f:
            return pickle.load(f)
    if key == "VCM":
        ns = _exec_until("vcm_engine.py", "V29 PARAMETER GRID")
        S, prices, end = ns["S"], ns["prices"], pd.Timestamp(ns["END"])
    elif key == "BF":
        ns = _exec_until("bull_flag_alternating_trail_v1.py", "ALTERNATING TRAIL EXPERIMENT")
        S, prices, end = ns["S"], ns["prices"], pd.Timestamp(ns["END"])
    elif key == "CH":
        ns = _exec_until("cup_and_handle_scan.py", 'C.round(6).to_csv(DATA/"cup_and_handle_v2_candidates.csv"', exact=True)
        S, prices, end = ns["S"], ns["prices"], pd.Timestamp(ns["END"])
    elif key == "DB":
        import double_bottom_v3_truew_scan as core
        core.UNIVERSE = STITCH_DIR
        sig, pr = core.scan_universe()
        S = sig.rename(columns={"rs_percentile": "rs_pct"})
        prices = {k: v[["Close", "MA25", "MA50", "MA100"]].rename(
            columns={"Close": "close", "MA25": "ma25", "MA50": "ma50", "MA100": "ma100"}) for k, v in pr.items()}
        end = max(p.index.max() for p in prices.values())
    S = S[["date", "symbol", "rs_pct"]].copy()
    S["date"] = pd.to_datetime(S["date"])
    out = (S, base._norm_prices(prices, {}), end)
    with open(cache, "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    return out


# ---------------------------------------------------------------- compare
def signal_match(key, S_close, S_1515, start, end):
    a = S_close[(S_close.date >= start) & (S_close.date <= end)][["symbol", "date"]].drop_duplicates()
    b = S_1515[(S_1515.date >= start) & (S_1515.date <= end)][["symbol", "date"]].drop_duplicates()
    both = a.merge(b, on=["symbol", "date"])
    # within one day either way (a breakout confirmed a day earlier/later)
    b1 = pd.concat([b.assign(date=b.date + pd.tseries.offsets.BDay(k)) for k in (-1, 0, 1)])
    near = a.merge(b1.drop_duplicates(), on=["symbol", "date"]).drop_duplicates()
    return dict(strategy=key, close_signals=len(a), signals_1515=len(b), same_day_match=len(both),
                pct_close_signals_seen_at_1515=round(len(both) / max(len(a), 1) * 100, 1),
                pct_1515_signals_confirmed_at_close=round(len(both) / max(len(b), 1) * 100, 1),
                pct_close_signals_within_1_day=round(len(near) / max(len(a), 1) * 100, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="VCM,BF,CH,DB")
    ap.add_argument("--reuse", action="store_true", help="reuse saved 15:15 signals and baseline caches")
    ap.add_argument("--no-rebuild", action="store_true", help="don't rebuild the stitched folder")
    ap.add_argument("--grid", action="store_true",
                    help="run ALL 27 combos (trail 25/50/100 x stop 4/8/12 x max 5/10/15), capped book, close vs snapshot")
    ap.add_argument("--at", default="15:15", help="snapshot time to test, e.g. 15:25 (needs data/universe_1525)")
    args = ap.parse_args()
    configure(args.at)
    keys = [k.strip().upper() for k in args.only.split(",") if k.strip()]

    if not (args.no_rebuild and STITCH_DIR.exists()):
        build_stitched()

    summaries, yearlies, matches = [], [], []
    grid_rows, grid_yearly = [], []
    print(f"Testing snapshot time {args.at}: {SNAP_DIR.name} -> outputs {PREFIX}_*")
    for key in keys:
        cfg = base.STRATEGIES[key]
        t0 = time.time()
        print(f"\n===== {key}: {cfg['name']} =====")
        S0, P0, end0 = nxt.load(key, reuse=True)          # CLOSE baseline (cached)
        S1, P1, end1 = load_1515(key, args.reuse)          # snapshot-time world
        end = min(end0, end1)
        # Apples-to-apples: the CLOSE baseline uses only symbols that exist in
        # the 3:15 world (a few stocks have no intraday data / were excluded).
        common = set(P1.keys())
        dropped = int((~S0.symbol.isin(common) & (S0.date >= base.SIM_START) & (S0.date <= end)).sum())
        S0 = S0[S0.symbol.isin(common)]
        print(f"  CLOSE baseline restricted to the same {len(common)} symbols "
              f"({dropped} close signals from other symbols dropped)")
        print(f"signals close {len(S0)} | {args.at} {len(S1)} | window to {end.date()} "
              f"({(time.time()-t0)/60:.1f} min)")
        matches.append(signal_match(key, S0, S1, base.SIM_START, end))
        m = matches[-1]
        print(f"  match: {m['pct_close_signals_seen_at_1515']}% of close signals also fire at {args.at} | "
              f"{m['pct_1515_signals_confirmed_at_close']}% of {args.at} signals are confirmed at close")
        if args.grid:
            combos = [(t, st, mp) for t in (25, 50, 100) for st in (4, 8, 12) for mp in (5, 10, 15)]
            if key == "BF":
                combos.append(("alt25_50", 4, 10))
            for label, S, P in (("CLOSE", S0, P0), (AT_TAG, S1, P1)):
                for t, st, mp in combos:
                    T, D, skf, skh, extra = nxt.simulate(S, P, end, t, st, mp, "close")
                    closed = T[T.status == "closed"] if len(T) else T
                    wins, losses = closed[closed.pnl > 0], closed[closed.pnl <= 0]
                    combo = f"{'alt25/50' if t == 'alt25_50' else t}/{st}/{mp}"
                    total = float(D.total_pnl.iloc[-1])
                    grid_rows.append(dict(
                        strategy=key, entry=label, combo=combo,
                        is_website_default=(combo == ("alt25/50/4/10" if key == "BF" else
                                            f"{cfg['trail']}/{cfg['stop']}/{cfg['maxpos']}")),
                        total_return_pct=round(total / base.CAPPED_CAPITAL * 100, 2),
                        max_dd_pct=extra["max_dd_equity_pct"], trades=len(T),
                        win_rate_pct=round((closed.pnl > 0).mean() * 100, 2) if len(closed) else np.nan,
                        profit_factor=round(wins.pnl.sum() / -losses.pnl.sum(), 2) if len(losses) and losses.pnl.sum() else np.nan))
                    yr = base.yearly(key, f"{label}/{combo}", D)
                    yr["entry"], yr["combo"] = label, combo
                    grid_yearly.append(yr)
                print(f"  grid {label}: {len(combos)} combos done ({(time.time()-t0)/60:.1f} min)")
            continue
        for label, S, P in (("CLOSE", S0, P0), (AT_TAG, S1, P1)):
            for book, mp in (("CAPPED", cfg["maxpos"]), ("UNLIMITED", None)):
                T, D, skf, skh, extra = nxt.simulate(S, P, end, cfg["trail"], cfg["stop"], mp, "close")
                mode = f"{label}/{book}"
                s = base.summarize(key, book, T, D, skf, skh, int(((S.date >= base.SIM_START) & (S.date <= end)).sum()))
                s.pop("max_dd_pct_of_basis", None)
                s.update(mode=mode, max_dd_equity_pct=extra["max_dd_equity_pct"])
                summaries.append(s)
                yearlies.append(base.yearly(key, mode, D))
                print(f"  {mode:16s} trades {s['trades_taken']:4d} | win {s['win_rate_pct']}% | "
                      f"avg {s['avg_trade_pct']}% | PF {s['profit_factor']} | "
                      f"return {s['return_pct']}% on {s['return_basis']}"
                      + (f" | DD {extra['max_dd_equity_pct']}%" if book == "CAPPED" else ""))

    if args.grid:
        G = pd.DataFrame(grid_rows)
        GY = pd.concat(grid_yearly, ignore_index=True)
        GY = GY[["strategy", "entry", "combo", "year", "pnl_rs", "return_on_10l_pct"]]
        G.to_csv(DATA / f"{PREFIX}_grid_summary.csv", index=False)
        GY.to_csv(DATA / f"{PREFIX}_grid_yearly.csv", index=False)
        pd.DataFrame(matches).to_csv(DATA / f"{PREFIX}_grid_signal_match.csv", index=False)
        piv = GY.pivot_table(index=["strategy", "combo", "entry"], columns="year",
                             values="return_on_10l_pct").round(1)
        tot = G.set_index(["strategy", "combo", "entry"])[["total_return_pct", "max_dd_pct"]]
        out = piv.join(tot).reset_index()
        pd.set_option("display.width", 250)
        for key in keys:
            print(f"\n=== {key}: yearly % on Rs 10L, all combos, CLOSE vs {args.at} (capped) ===")
            print(out[out.strategy == key].drop(columns="strategy").to_string(index=False))
        print(f"\nSaved: {PREFIX}_grid_summary.csv, {PREFIX}_grid_yearly.csv in {DATA}")
        return

    SUM, YR, MT = pd.DataFrame(summaries), pd.concat(yearlies, ignore_index=True), pd.DataFrame(matches)
    SUM.to_csv(DATA / f"{PREFIX}_summary.csv", index=False)
    YR.to_csv(DATA / f"{PREFIX}_yearly.csv", index=False)
    MT.to_csv(DATA / f"{PREFIX}_signal_match.csv", index=False)

    print("\n=== Yearly, CAPPED website book (% on Rs 10L) ===")
    cap = YR[YR["mode"].str.endswith("/CAPPED")]
    print(cap.pivot_table(index=["strategy", "mode"], columns="year", values="return_on_10l_pct").round(1).to_string())
    print("\n=== Yearly, UNLIMITED (% on that year's peak deployed capital) ===")
    unl = YR[YR["mode"].str.endswith("/UNLIMITED")]
    print(unl.pivot_table(index=["strategy", "mode"], columns="year",
                          values="return_on_year_peak_capital_pct").round(1).to_string())
    print(f"\n=== Signal match (close vs {args.at}) ===")
    print(MT.to_string(index=False))
    print("\n=== Summary ===")
    print(SUM[["strategy", "mode", "trades_taken", "win_rate_pct", "avg_trade_pct", "profit_factor",
               "return_pct", "max_dd_equity_pct"]].to_string(index=False))
    print(f"\nSaved: {PREFIX}_summary.csv, {PREFIX}_yearly.csv, {PREFIX}_signal_match.csv in {DATA}")


if __name__ == "__main__":
    main()
