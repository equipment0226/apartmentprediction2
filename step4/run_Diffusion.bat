@echo off
REM ============================================================
REM  Diffusion(SSSD) 파이프라인 Windows 실행 런처
REM   - KMP_DUPLICATE_LIB_OK : torch + numpy OpenMP 충돌 방지
REM   - PYTHONUTF8           : Lightning/이모지 출력 인코딩 안전
REM   - PYTHONIOENCODING     : 콘솔 인코딩 fallback
REM ============================================================
SET KMP_DUPLICATE_LIB_OK=TRUE
SET PYTHONUTF8=1
SET PYTHONIOENCODING=utf-8:replace

cd /d "%~dp0"
python Diffusion.py
pause
