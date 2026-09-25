"""
fetch_intraday_snapshots.py -- build a "3:15 PM world" copy of the universe
======================================================================
For every stock in data/universe/, downloads Kite intraday candles (60-minute
for 15:15 -- the largest size that ends exactly at the snapshot time) and
collapses each day into ONE candle as it looked at 15:15 IST
(default; change with --at):

    open   = day's first candle open (09:15)
    high   = highest high up to 15:15
    low    = lowest low up to 15:15
    close  = close of the 14:15-15:15 hourly candle = price at 15:15
    volume = total volume up to 15:15

Output: data/universe_1515/<SYMBOL>.csv  (same columns as data/universe/),
so the engines can be pointed at it unchanged.

It never touches data/universe/.

Kite limits: 60-minute data comes in ~400-day chunks, so ~7 requests per
stock for 2019-07 .. today (5-minute would need ~28). ~1636 x 7 = ~11k
requests = roughly 1 hour with 3 parallel workers (the default). The script is RESUMABLE: re-run it and it continues
where it stopped (finished symbols are skipped).

STEP 1 -- probe first (1 minute): checks how far back Kite's intraday
history goes and whether the 15:15 snapshot lines up with the daily close.
    py fetch_intraday_snapshots.py --probe

STEP 2 -- full download (leave it running, e.g. overnight):
    py fetch_intraday_snapshots.py

DAILY -- after 15:20 IST, append the latest days to every existing file
(fetches only the last --days calendar days per stock, ~10 minutes):
    py fetch_intraday_snapshots.py --update

STEP 0 (optional) -- which time is best? Uses 1-minute data (last ~60 days)
for a few stocks and shows how close the 15:15 / 15:20 / 15:25 / 15:28
price is to the official close:
    py fetch_intraday_snapshots.py --probe-times

Options:
    --at 15:28            snapshot time (default 15:15). The largest candle
                          size that ends exactly at that time is used
                          (15:15 -> 60-min, 15:25 -> 10-min, 15:28 -> 1-min). Output folder
                          is data/universe_HHMM (e.g. universe_1515).
    --start 2019-07-01    first date (default; engines need ~200 days of
                          history before 2020 for the 200 DMA)
    --symbols A,B,C       only these symbols
    --redo                re-download symbols that already have a file
"""
import argparse
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from kiteconnect import KiteConnect

import config

kite = KiteConnect(api_key=config.API_KEY)
kite.set_access_token(config.ACCESS_TOKEN)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data")
UNIVERSE_DIR = os.path.join(DATA_DIR, "universe")
OUT_DIR = os.path.join(DATA_DIR, "universe_1515")
IST = timezone(timedelta(hours=5, minutes=30))

PACE_SECONDS = 0.36   # minimum gap between ANY two requests (all threads) -> < 3 req/s


class RateLimiter:
    """Shared across threads: request STARTS are spaced >= PACE_SECONDS apart,
    so several threads can wait on the network at once without breaking
    Kite's 3 requests/second limit."""
    def __init__(self, gap):
        self.gap = gap
        self.lock = threading.Lock()
        self.next_at = 0.0

    def wait(self):
        with self.lock:
            now = time.monotonic()
            t = max(now, self.next_at)
            self.next_at = t + self.gap
        if t > now:
            time.sleep(t - now)


LIMITER = RateLimiter(PACE_SECONDS)
# Set in main() from --at: interval, chunk size, last candle start, output dir.
INTERVAL = "60minute"
CHUNK_DAYS = 395
SNAPSHOT_LAST_CANDLE = "14:15"  # the 60-min candle starting 14:15 ends at 15:15


# Kite max days per historical request, by interval.
CHUNK_LIMITS = {60: 395, 30: 195, 15: 195, 10: 95, 5: 95, 3: 95, 1: 58}
INTERVAL_NAMES = {60: "60minute", 30: "30minute", 15: "15minute", 10: "10minute",
                  5: "5minute", 3: "3minute", 1: "minute"}


def configure(at: str):
    """--at HH:MM -> pick the LARGEST candle size whose boundaries (counted from
    09:15) land exactly on that time; fewer requests, same snapshot.
    15:15 -> 60-minute candles (last one 14:15-15:15), ~7 requests per stock."""
    global INTERVAL, CHUNK_DAYS, SNAPSHOT_LAST_CANDLE, OUT_DIR
    h, m = map(int, at.split(":"))
    since_open = (h * 60 + m) - (9 * 60 + 15)
    step = next(st for st in (60, 30, 15, 10, 5, 3, 1) if since_open % st == 0)
    INTERVAL = INTERVAL_NAMES[step]
    CHUNK_DAYS = CHUNK_LIMITS[step]
    last = datetime(2000, 1, 1, h, m) - timedelta(minutes=step)
    SNAPSHOT_LAST_CANDLE = last.strftime("%H:%M")
    OUT_DIR = os.path.join(DATA_DIR, f"universe_{h:02d}{m:02d}")


def token_map():
    inst = pd.DataFrame(kite.instruments("NSE"))
    inst = inst[inst["instrument_type"] == "EQ"]
    return dict(zip(inst["tradingsymbol"], inst["instrument_token"]))


def fetch_5min(token, start, end, interval=None, chunk_days=None):
    interval = interval or INTERVAL
    chunk_days = chunk_days or CHUNK_DAYS
    frames = []
    cur = start
    while cur <= end:
        stop = min(cur + timedelta(days=chunk_days - 1), end)
        for attempt in range(5):
            try:
                LIMITER.wait()
                rows = kite.historical_data(token, cur.strftime("%Y-%m-%d"),
                                            stop.strftime("%Y-%m-%d"), interval)
                break
            except Exception as e:
                msg = str(e).lower()
                if "too many" in msg or "rate" in msg or "timed out" in msg:
                    time.sleep(2 + attempt * 3)
                    continue
                if "token" in msg or "api_key" in msg:
                    raise RuntimeError("TOKEN EXPIRED -- run generate_token.py, then re-run this script")
                raise
        else:
            raise RuntimeError(f"gave up on chunk {cur.date()}..{stop.date()}")
        if rows:
            frames.append(pd.DataFrame(rows))
        cur = stop + timedelta(days=1)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    if df["date"].dt.tz is not None:
        df["date"] = df["date"].dt.tz_convert(IST).dt.tz_localize(None)
    return df


def to_snapshots(df5, last_candle=None):
    """Collapse intraday candles into one snapshot candle per day (up to --at)."""
    if df5.empty:
        return df5
    df5 = df5.sort_values("date")
    df5["day"] = df5["date"].dt.normalize()
    df5["hm"] = df5["date"].dt.strftime("%H:%M")
    df5 = df5[df5["hm"] <= (last_candle or SNAPSHOT_LAST_CANDLE)]
    g = df5.groupby("day")
    snap = pd.DataFrame({
        "open": g["open"].first(),
        "high": g["high"].max(),
        "low": g["low"].min(),
        "close": g["close"].last(),       # close of the last candle before --at (or last traded before it)
        "volume": g["volume"].sum(),
        "last_candle": g["hm"].last(),
    }).reset_index().rename(columns={"day": "date"})
    return snap


def compare_with_daily(sym, snap):
    path = os.path.join(UNIVERSE_DIR, f"{sym}.csv")
    if not os.path.exists(path) or snap.empty:
        return None
    d = pd.read_csv(path)
    d["date"] = pd.to_datetime(d["date"].astype(str).str[:10])
    m = snap.merge(d[["date", "open", "close", "volume"]], on="date", suffixes=("_snap", "_day"))
    if m.empty:
        return None
    diff = (m["close_snap"] / m["close_day"] - 1) * 100
    return dict(days=len(m), first=str(m["date"].min().date()),
                open_match_pct=round((np.isclose(m["open_snap"], m["open_day"], rtol=1e-3)).mean() * 100, 1),
                median_abs_close_diff_pct=round(diff.abs().median(), 3),
                p95_abs_close_diff_pct=round(diff.abs().quantile(0.95), 3),
                vol_snap_share_pct=round((m["volume_snap"] / m["volume_day"]).median() * 100, 1))


def daily_update(args, tmap, today):
    """Append the latest snapshot days to existing files. New symbols (in the
    daily universe but without a snapshot file) get a full history download."""
    now = datetime.now(IST)
    cutoff_h, cutoff_m = map(int, args.at.split(":"))
    ready = (now.hour, now.minute) >= (cutoff_h, cutoff_m + 5) if cutoff_m + 5 < 60 else now.hour > cutoff_h
    end = pd.Timestamp(today) if ready else pd.Timestamp(today) - pd.Timedelta(days=1)
    if not ready:
        print(f"NOTE: it is {now:%H:%M} IST -- today's {args.at} snapshot is not final yet, so today is skipped.")
    start = end - pd.Timedelta(days=args.days)
    full_start = pd.Timestamp(args.start)

    symbols = sorted(f[:-4] for f in os.listdir(UNIVERSE_DIR) if f.endswith(".csv"))
    print(f"DAILY UPDATE ({args.at}): {len(symbols)} symbols, window {start.date()} .. {end.date()} -> {OUT_DIR}")
    t0 = time.time()
    failed, updated, new = [], 0, 0
    token_dead = threading.Event()

    def one(sym):
        if token_dead.is_set():
            return sym, "skipped (token expired)", None
        tok = tmap.get(sym)
        if tok is None:
            return sym, "not in NSE instrument list", None
        path = os.path.join(OUT_DIR, f"{sym}.csv")
        is_new = not os.path.exists(path)
        try:
            snap = to_snapshots(fetch_5min(tok, full_start if is_new else start, end))
            if snap.empty:
                return sym, None, "nodata"
            snap = snap[["date", "open", "high", "low", "close", "volume"]].copy()
            if not is_new:
                old = pd.read_csv(path)
                old["date"] = pd.to_datetime(old["date"].astype(str).str[:10])
                snap = pd.concat([old, snap], ignore_index=True)
            snap = snap.drop_duplicates("date", keep="last").sort_values("date")
            snap["date"] = pd.to_datetime(snap["date"]).dt.strftime("%Y-%m-%d")
            tmp = path + ".tmp"
            snap.to_csv(tmp, index=False)
            os.replace(tmp, path)
            return sym, None, "new" if is_new else "updated"
        except Exception as e:
            if "TOKEN EXPIRED" in str(e):
                token_dead.set()
            return sym, str(e)[:120], None

    done = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for fut in as_completed([pool.submit(one, s) for s in symbols]):
            sym, err, kind = fut.result()
            done += 1
            if err:
                failed.append((sym, err))
            elif kind == "updated":
                updated += 1
            elif kind == "new":
                new += 1
            if done % 200 == 0 or done == len(symbols):
                print(f"  {done}/{len(symbols)} | {(time.time()-t0)/60:.1f} min | updated {updated} new {new} failed {len(failed)}")
    if token_dead.is_set():
        print("STOPPED: Kite token expired. Run generate_token.py and re-run --update.")
    print(f"Done. {updated} updated, {new} new symbols (full history), {len(failed)} failed.")
    if failed:
        print(f"Failed (first 10): {failed[:10]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--probe-times", action="store_true")
    ap.add_argument("--at", default="15:15")
    ap.add_argument("--start", default="2019-07-01")
    ap.add_argument("--symbols", default="")
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--update", action="store_true",
                    help="DAILY MODE: fetch only the last --days and merge into existing files")
    ap.add_argument("--days", type=int, default=10, help="--update window in calendar days (default 10)")
    ap.add_argument("--workers", type=int, default=3,
                    help="parallel downloads (default 3). Requests stay under Kite's 3/sec either way")
    args = ap.parse_args()

    configure(args.at)
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(datetime.now(IST).date())
    os.makedirs(OUT_DIR, exist_ok=True)
    tmap = token_map()

    if args.probe_times:
        syms = ["RELIANCE", "TCS", "JINDALSTEL", "NOCIL", "DIVISLAB", "TEXRAIL", "OLAELEC", "GIPCL"]
        times = {"15:15": "15:14", "15:20": "15:19", "15:25": "15:24", "15:28": "15:27"}
        pstart = end - pd.Timedelta(days=55)
        print(f"PROBE-TIMES: 1-minute data {pstart.date()}..{end.date()}, {len(syms)} stocks")
        rows = []
        for sym in syms:
            tok = tmap.get(sym)
            if tok is None:
                continue
            df1 = fetch_5min(tok, pstart, end, interval="minute", chunk_days=58)
            if df1.empty:
                print(f"  {sym}: no minute data"); continue
            path = os.path.join(UNIVERSE_DIR, f"{sym}.csv")
            if not os.path.exists(path):
                continue
            d = pd.read_csv(path)
            d["date"] = pd.to_datetime(d["date"].astype(str).str[:10])
            daily = dict(zip(d["date"], d["close"]))
            for label, last in times.items():
                snap = to_snapshots(df1.copy(), last_candle=last)
                for _, r in snap.iterrows():
                    if r["date"] in daily:
                        rows.append(dict(symbol=sym, time=label,
                                         diff=abs(r["close"] / daily[r["date"]] - 1) * 100))
        R = pd.DataFrame(rows)
        if R.empty:
            print("No data."); return
        out = R.groupby("time")["diff"].agg(
            median_abs_diff_pct="median", p90_abs_diff_pct=lambda x: x.quantile(0.9),
            within_0_25pct=lambda x: (x <= 0.25).mean() * 100).round(3)
        print("\nPrice at time T vs official close (smaller = closer to the backtest):")
        print(out.to_string())
        return

    if args.probe:
        probe = ["RELIANCE", "TCS", "JINDALSTEL", "NOCIL", "DIVISLAB"]
        print(f"PROBE ({args.at}): fetching {probe} from {start.date()} (a few minutes)...")
        for sym in probe:
            tok = tmap.get(sym)
            if tok is None:
                print(f"  {sym}: not found"); continue
            snap = to_snapshots(fetch_5min(tok, start, end))
            if snap.empty:
                print(f"  {sym}: NO intraday data returned"); continue
            cmp_ = compare_with_daily(sym, snap)
            print(f"  {sym}: {len(snap)} days, first {snap['date'].min().date()}, "
                  f"last {snap['date'].max().date()} | vs daily: {cmp_}")
        print("\nRead this before the full run:")
        print("  first date     -> how far back Kite intraday goes (need 2019-07 for a full 2020+ test)")
        print("  open_match_pct -> should be ~100 (same price basis as the daily files)")
        print("  median_abs_close_diff_pct -> typical 3:25 vs close gap (expect ~0.2-0.5)")
        print("  vol_snap_share_pct -> share of the day's volume done by the snapshot time (15:15: expect ~80-88)")
        return

    if args.update:
        return daily_update(args, tmap, end)

    files = sorted(f[:-4] for f in os.listdir(UNIVERSE_DIR) if f.endswith(".csv"))
    if args.symbols:
        want = {s.strip().upper() for s in args.symbols.split(",")}
        files = [s for s in files if s in want]
    todo = [s for s in files if args.redo or not os.path.exists(os.path.join(OUT_DIR, f"{s}.csv"))]
    print(f"{len(files)} symbols, {len(files) - len(todo)} already done, {len(todo)} to fetch "
          f"({start.date()} .. {end.date()})")

    t0 = time.time()
    failed = []
    token_dead = threading.Event()

    def one(sym):
        if token_dead.is_set():
            return sym, "skipped (token expired)"
        tok = tmap.get(sym)
        if tok is None:
            return sym, "not in NSE instrument list"
        try:
            snap = to_snapshots(fetch_5min(tok, start, end))
            if snap.empty:
                return sym, "no intraday data"
            out = snap[["date", "open", "high", "low", "close", "volume"]].copy()
            out["date"] = out["date"].dt.strftime("%Y-%m-%d")
            tmp = os.path.join(OUT_DIR, f"{sym}.csv.tmp")
            out.to_csv(tmp, index=False)
            os.replace(tmp, os.path.join(OUT_DIR, f"{sym}.csv"))  # only complete files count as done
            return sym, None
        except Exception as e:
            if "TOKEN EXPIRED" in str(e):
                token_dead.set()
            return sym, str(e)[:120]

    print(f"Downloading with {args.workers} parallel workers (Ctrl+C is safe: finished symbols are kept)")
    done = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(one, s) for s in todo]
        for fut in as_completed(futures):
            sym, err = fut.result()
            done += 1
            if err:
                failed.append((sym, err))
                if not err.startswith("skipped"):
                    print(f"  FAILED {sym}: {err}")
            if done % 10 == 0 or done == len(todo):
                el = (time.time() - t0) / 60
                eta = el / done * (len(todo) - done)
                print(f"  {done}/{len(todo)} | {el:.0f} min elapsed | ETA {eta:.0f} min | failed {len(failed)}")
    if token_dead.is_set():
        print("\nSTOPPED: Kite token expired. Run generate_token.py, then run this script again -- "
              "it continues from where it stopped.")

    print(f"\nDone. Output: {OUT_DIR}")
    if failed:
        pd.DataFrame(failed, columns=["symbol", "reason"]).to_csv(
            OUT_DIR + "_failed.csv", index=False)
        print(f"{len(failed)} failed -> {os.path.basename(OUT_DIR)}_failed.csv (re-run the script to retry)")


if __name__ == "__main__":
    main()
