@echo off
REM ================================================================
REM DailyBreakouts -- daily update (run after 4 PM, fresh Kite token)
REM ================================================================
cd /d D:\projects\dailybreakouts\backend

echo [1/6] Daily candles...
py update_universe_daily.py || goto :fail
py fix_partial_last_bar.py

echo [2/6] 3:15 PM snapshots...
py fetch_intraday_snapshots.py --update || goto :fail

echo [3/6] Forming lists + chart data (lifecycle scripts)...
py breakout_lifecycle.py || goto :fail
py breakout_lifecycle_bullflag.py || goto :fail
py breakout_lifecycle_double_bottom.py || goto :fail
py breakout_lifecycle_cup_and_handle.py || goto :fail

echo [4/6] Live-rule book (3:15, ALL9 / DB 100-8, compounding)...
py live_engine.py --rebuild --rescan || goto :fail

echo [5/6] Website JSONs...
py build_live_website_json.py || goto :fail

echo [6/6] DONE. Upload from D:\projects\dailybreakouts\data :
echo    breakout_lifecycle.json, breakout_lifecycle_bullflag.json,
echo    breakout_lifecycle_cup_and_handle.json, breakout_lifecycle_double_bottom.json,
echo    live_state.json, history\  (NOT the *_full.json files)
goto :eof

:fail
echo.
echo *** STOPPED: the step above failed. Fix it and run this file again. ***
exit /b 1
