@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo ========================================
echo  ComVision Module 2 - setup + run test
echo ========================================
echo.

where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] หา python ไม่เจอใน PATH
    echo ติดตั้ง Python แล้วลองใหม่ หรือเปิดจาก Developer Command Prompt
    goto :fail
)

if not exist ".venv\Scripts\python.exe" (
    echo [1/3] สร้าง .venv ...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] สร้าง venv ไม่สำเร็จ
        goto :fail
    )
) else (
    echo [1/3] พบ .venv แล้ว — ข้ามการสร้างใหม่
)

echo [2/3] ติดตั้ง dependencies จาก requirements.txt ...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :fail
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] pip install ไม่สำเร็จ
    goto :fail
)

echo.
echo [3/3] รัน module2.run_test ...
".venv\Scripts\python.exe" -m module2.run_test
if errorlevel 1 (
    echo [ERROR] รันเทสไม่สำเร็จ
    goto :fail
)

echo.
echo ========================================
echo  เสร็จแล้ว — ดูผลที่โฟลเดอร์ results_py\
echo ========================================
goto :end

:fail
echo.
echo ล้มเหลว — ดูข้อความด้านบน
exit /b 1

:end
if /i "%~1"=="nopause" exit /b 0
echo.
pause
exit /b 0
