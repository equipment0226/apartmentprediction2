@echo off
REM ======================================================
REM  TFT.py 실행용 배치파일
REM  - 이 파일을 더블클릭하거나 cmd에서 실행하면 됩니다.
REM  - 오류 발생 시 step2\_error_log.txt 에 저장됩니다.
REM ======================================================
SET KMP_DUPLICATE_LIB_OK=TRUE
SET PYTHONUTF8=1
SET PYTHONIOENCODING=utf-8:replace
SET PYTHONWARNINGS=ignore

cd /d "%~dp0"
python TFT.py
echo.
echo [완료] 아무 키나 누르면 창이 닫힙니다.
pause
