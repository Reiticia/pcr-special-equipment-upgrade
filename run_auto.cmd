@echo off
setlocal
pushd "%~dp0"
".venv\Scripts\python.exe" -X utf8 -u upgrade_equipment.py --fully-auto %*
set "rc=%errorlevel%"
echo.
echo Exit code: %rc%
pause
popd
exit /b %rc%
