@echo off
REM ======================================================
REM  PITFT.py 실행용 배치파일 (Physics-Informed TFT)
REM  - 거시경제 부호 제약 손실항 결합 TFT 학습/예측
REM ======================================================
SET KMP_DUPLICATE_LIB_OK=TRUE
SET PYTHONUTF8=1
SET PYTHONIOENCODING=utf-8:replace
SET PYTHONWARNINGS=ignore

cd /d "%~dp0"
python PITFT.py
echo.
echo [완료] 아무 키나 누르면 창이 닫힙니다.
pause
