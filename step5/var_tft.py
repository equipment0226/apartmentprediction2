"""
VAR-TFT Hybrid : Bayesian VAR 시나리오 생성 + TFT 잔차 보정 파이프라인
===========================================================================

데이터 : data/dong_level_meta_table.csv  (2010-01 ~ 2025-12, 월 단위,
         243개 동(洞) × 25개 컬럼)
타겟   : reb__apt_sale_avg_price  (서울 아파트 매매 평균가, 동 단위 패널)

설계 철학 (TalkFile_M1-M5_파이프라인_쉬운설명_코드.md 준수)
---------------------------------------------------------------------------
  "사람이 직관으로 박는 숫자를 최대한 없앤다. 모든 기준을 과거 데이터와
   추정 결과에서 나오게 한다."

파이프라인 흐름:
  ┌─────────────────────────────────────────────────────────────────┐
  │  raw panel  (dong × month)                                        │
  └───────────┬─────────────────────────────────────────────────────┘
              │
       (서울 평균 집계)              (panel 그대로)
              │                              │
              ▼                              ▼
   ┌────────────────────────┐      ┌────────────────────────┐
   │  M1  ADF→diff (정상화) │      │  TFT 패널 학습용        │
   │  M2  Bayesian VAR 추정 │      │   - log-return 타겟     │
   │  M3  block bootstrap   │      │   - panel(동×시점)      │
   │      N 시나리오 생성   │      │   - GroupNormalizer     │
   │  M3+ 잔차주머니 보강   │      │       (center=True)     │
   │      N 하드조건 역산   │      │   - VAR 시나리오 변수    │
   │  M4  현실성 필터(리포트)│      │       을 decoder 에      │
   │  M5  원단위 복원·검증  │      │       VAR 시나리오로     │
   │      → VAR baseline    │ ───► │       주입(잔차보정)     │
   │         target 경로    │      │                          │
   └────────────────────────┘      └────────────┬─────────────┘
                                                 │
                                                 ▼
                                  ┌──────────────────────────────┐
                                  │   최종 예측 (Hybrid Ensemble)│
                                  │   pred = VAR_baseline +       │
                                  │          TFT_residual         │
                                  │   분포 = N개 VAR 시나리오     │
                                  │          × TFT 분위 예측       │
                                  └──────────────────────────────┘

핵심 차별 포인트
---------------------------------------------------------------------------
  1. VAR : 거시·수급 변수의 *방향성 있는 미래 경로* 를 데이터에서 추정해
           생성. 사람이 시나리오별 drift % 를 박지 않는다.
  2. TFT : VAR baseline 이 못 잡는 비선형/이질성(동별 특성·정책 충격)을
           잔차로 학습 → "VAR + TFT 잔차" 의 boosted ensemble.
  3. 분포: M3 가 만든 N개 covariate 경로를 TFT decoder 에 일일이 주입해
           N개 forecast 분포 생성 → quantile 산출 (단일 점추정 회피).
  4. 모든 하이퍼파라미터/시나리오 노브는 본 파일 상단 CONFIG 블록에
           단일 출처(single-source-of-truth) 로 두어 직접 tune 가능.

저장 폴더 : step5/  (step1~4 와 같은 형식으로 정리)
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# [Windows 필수 설정 - 반드시 다른 import 보다 먼저]
#   - KMP_DUPLICATE_LIB_OK : torch + numpy 가 각자 OpenMP 런타임을 로드할 때
#     충돌 방지. Windows Anaconda 환경에서 필수.
#   - PYTHONUTF8=1 : PyTorch Lightning 의 "💡 Tip:" 이모지가 cp949 콘솔에서
#     UnicodeEncodeError 를 일으켜 cmd 창이 바로 닫히는 증상 방지.
# ---------------------------------------------------------------------------
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8:replace")
os.environ.setdefault("PYTHONWARNINGS", "ignore")

import sys, warnings
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
warnings.filterwarnings("ignore")

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # GUI 없이 PNG 저장 전용 백엔드
import matplotlib.pyplot as plt

from scipy import stats
from scipy.stats import binom

from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import CSVLogger

from pytorch_forecasting import (
    TimeSeriesDataSet,
    TemporalFusionTransformer,
    GroupNormalizer,
)
from pytorch_forecasting.data.encoders import NaNLabelEncoder
from pytorch_forecasting.metrics import QuantileLoss


# ===========================================================================
# 0. CONFIG — 모든 tunable parameter 의 단일 출처
#
#   본 파이프라인에 들어가는 모든 "사람이 정하는 수치" 는 이 블록에 모은다.
#   타 함수에서는 절대 magic-number 를 쓰지 않는다.
#   각 값 옆 주석 = 그 값을 왜 그렇게 골랐는지에 대한 근거.
# ===========================================================================

# --- 경로 ---
BASE_DIR = Path(__file__).resolve().parent          # .../step5
ROOT_DIR = BASE_DIR.parent                          # .../newtrial
DATA_CSV = ROOT_DIR / "data" / "dong_level_meta_table.csv"
OUT_DIR  = BASE_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- 컬럼 ---
TARGET_LEVEL_COL = "reb__apt_sale_avg_price"   # 원본 가격 컬럼 (복원/차트용)
TARGET_COL       = "target_logret"             # TFT 학습 타겟 = 월간 로그수익률
TIME_COL         = "timestamp"
ID_COLS          = ["si", "gu", "dong"]

# --- 분석 동 선택 (1개 동 단위 VAR + TFT 구조) ---
#   이전에는 245개 동 평균(=서울시)으로 VAR 시나리오를 생성했으나,
#   설계 의도(시/구/동 batch)에 맞게 이제 '특정 1개 동'만 선택하여
#   VAR 시나리오 + TFT 학습/예측을 수행한다. 다른 동으로 전환하려면
#   이 상수만 변경. 추후 시/구 batch 로 확장할 때는 이 값을 loop 로 순회.
TARGET_DONG      = "개포동"

# --- 시나리오 분할 구간 ---
TRAIN_END   = "2023-12-01"   # 백테스트 학습 종료
VALID_START = "2024-01-01"
VALID_END   = "2025-12-01"   # (백테스트 기본 horizon = 24개월)
FUTURE_END  = "2045-12-01"   # 미래 예측 종료 (10년)


# ===========================================================================
# 0-1. M1~M5 (Bayesian VAR) 시나리오 생성 노브
# ===========================================================================
VAR_CFG = dict(
    # M1) 결측 처리 / 이상치 윈저라이징
    winsor_lower=0.01,        # 하위 1% 클리핑 — 과거 데이터의 통계적 극단을
    winsor_upper=0.99,        # 자르되, 그 자체가 '경제 가정'이 아닌 보수성 노브.
    adf_alpha=0.05,           # ADF p-value 임계 (≥ 0.05 → 비정상 → 차분).

    # M2) Bayesian VAR
    max_lag=12,               # BIC 자동탐색 상한. 월데이터 + 계절성 흡수 여유.
    minnesota_lambda=0.2,     # Minnesota shrinkage 강도 λ ∈ (0, 1].
                              # 0.2 = 중간 강도 (BVAR 문헌의 표준 권장).
                              # 0.0 = 일반 VAR (과적합), 1.0 = AR(1) 강제(과소적합).

    # M3) Block bootstrap
    horizon=240,              # 미래 시나리오 길이 (개월). FUTURE_END 와 정합.
    n_scenarios=100,          # ★ 사용자 지정 하한. 실제 생성 수는 M3+ 하드조건으로
                              #   자동 상향되므로 이 값이 그대로 쓰이는 경우는 드물다.
                              #   현재 설정(extreme_quantile=0.98, block_size=3,
                              #   extreme_k_min=30, confidence=0.99) 기준으로
                              #   실제 N ≈ 700~800 수준으로 산출됨.
                              #   → N 을 줄이고 싶으면 extreme_k_min 을 낮추거나
                              #     extreme_quantile 을 올릴 것 (꼬리 안정성 감소).
    block_size=3,             # block bootstrap 의 연속 충격 길이.
                              # 3 = 분기. 한 분기 동안의 충격 시간상관 보존.

    # M3+) 잔차주머니 보강 + 하드조건 N
    #   실제 생성 N 결정 흐름:
    #     p_rare_step  = (1 - extreme_quantile) ≈ 2%      # 극단 잔차 비율
    #     p_rare_block = 1 - (1-p_rare_step)^block_size   # block 당 확률 ≈ 5.9%
    #     N_hard = 최소 N s.t. P(Binomial(N, p_rare_block) ≥ k_min) ≥ confidence
    #            ≈ 700~800  (위 기본값 조합)
    #     최종 N = max(N_hard, n_scenarios, hard_floor_N)
    fat_tail=True,            # True → Student-t 꼬리 모델로 극단충격 보강.
    t_df_init=5.0,            # t분포 자유도 초기값. 5 = 두꺼운 꼬리 (Korean 부동산
                              # 잔차의 kurtosis 가 3 보다 큼 → Gaussian 부적합).
    tail_prob=0.05,           # 충격 block 추출시 t-tail 에서 뽑을 확률 (=5%).
                              # 95% 는 실측 잔차, 5% 는 가상 극단으로 분포 꼬리 채움.
    extreme_quantile=0.98,    # "극단" 정의 = 절대잔차 상위 2%.
                              # 이 값을 올릴수록 p_rare 감소 → N 증가.
    extreme_k_min=30,         # 극단 충격이 최소 k_min 번 등장하도록 N 역산.
                              # 낮출수록 N 감소 (꼬리 안정성 약화). 최솟값 권장 ≈ 20.
    confidence=0.99,          # 위 k_min 보장을 위한 신뢰수준.
    hard_floor_N=100,         # N_hard / n_scenarios 어느 쪽으로도 못 채울 때의
                              # 절대 하한. 사실상 현재 설정에서는 미발동.

    # M5) 정책 변수 양자화 (policy__* 컬럼 한정)
    #   VAR 는 연속값을 출력 → 정책 변수는 본래 {-1, 0, 1} 의 이산 regime.
    #   M5 복원 직후 임계값 기반 후처리로 깔끔한 형태 강제.
    #     v >=  +policy_quant_threshold  →  +1  (강화)
    #     v <=  -policy_quant_threshold  →  -1  (완화)
    #     그 외                          →   0  (유지)
    policy_quantize=True,     # False 면 연속값 그대로 유지 (ablation).
    policy_quant_threshold=0.5,

    # M4) 현실성 필터 임계 (대부분 데이터에서 자동산출 — 여기는 '보수성 노브'만)
    realism_extreme_q=0.99,   # T2 비현실적 급변 기준 = 과거 월변화 상위 1%.
                              # 더 엄격하게 하려면 0.995 등으로 올림.
    realism_corr_threshold=0.2,   # T3 변수쌍 부호 검사를 켤 최소 |corr|.
    realism_pair_min_corr=0.3,    # T3 시나리오에서 부호 위반으로 볼 최소 |corr|.
    realism_mode="report",    # "report"=리포트만, "drop"=경로 제거.
                              # 기본 report — 확률 비중 왜곡 방지 (M4 설계 원칙).

    # M5 군집 임계값 — 예측 기간 누적수익률(final/initial - 1) 기준
    #   「상승」: final_ret >  cluster_rise_thr
    #   「보합」: cluster_fall_thr ≤ final_ret ≤ cluster_rise_thr
    #   「하락」: final_ret <  cluster_fall_thr
    #
    #   기본값 ±5% 는 한국 아파트 단기 조정 범위 관행 (KB/한국감정원 기준
    #   보합 정의 ≒ 전고점 대비 ±3~5%)에서 가져왔으며, 나중에 기준금리
    #   (ecos__base_rate) 연 평균값으로 자동 산출하고 싶다면
    #   cluster_use_base_rate=True 로 전환.
    cluster_rise_thr=0.05,     # +5% 초과 → 상승. 변경 가능.
    cluster_fall_thr=-0.05,    # -5% 미만 → 하락. 변경 가능.
    cluster_use_base_rate=False,  # True 시: ecos__base_rate 예측 평균을
                                   # rise_thr 로 사용, fall_thr = -rise_thr.

    # M3 block bootstrap 재현성 시드
    seed=42,
)


# ===========================================================================
# 0-2. TFT 노브  (step2/TFT.py 와 동일 키 + 본 파일 고유 키 일부 추가)
# ===========================================================================
TFT_CFG = dict(
    max_encoder_length=72,   # encoder(과거) 길이 = 5년. 2020 boom + 2022 crash
                             # + 2023 회복 regime 전환을 한 윈도우에 담음.

    backtest_pred_len= 24,    # 백테스트 horizon (개월) = VALID 길이. // BVAR공통
    future_pred_len=240,     # 미래 horizon (개월) = 10년. VAR_CFG.horizon 과 정합.

    hidden_size=96,          # TFT 핵심 hidden dim. encoder 5년에 맞춰 32.→ 8 조정
    attention_head_size=4,   # multi-head attn. hidden 32 → 2 head 적정. → 1 조정 
    dropout=0.2,             # 과적합 방지. 원논문 디폴트 0.1 → 0.25 조정
    hidden_continuous_size=24,  # 연속변수 임베딩 dim = hidden_size//4.
    weight_decay=3e-4,        # AdamW L2 weight decay. 비현실적인 발산 규제

    learning_rate=1e-3,      # AdamW. encoder 5년 + log-return 타겟 → 작은 LR 안정.
    batch_size=64,           # CPU 메모리 + Lightning subprocess overhead 절충.
    max_epochs=30,           # 충분 + EarlyStopping 으로 자동 종료.
    gradient_clip_val=0.035,   # 원논문 표 4 권장값.
    early_stop_patience=3,   # val_loss 8 epoch 정체시 중단. (3 → 8 완화)
    recency_halflife_months=36,  # weight(t) = 0.5^((t_max-t)/halflife).
                                  # 10년 반감기 — 2010~2018 다른 체제 표본도
                                  # regime 전환 학습에 필요. 72 → 120 완화.
    num_workers=0,           # Windows: spawn freeze 회피.
    quantiles=[0.1, 0.5, 0.9],   # 중앙값 + 80% 예측구간.

    # --- 본 파일 고유 ---
    residual_mode=True,      # True : VAR baseline 을 빼고 TFT 가 잔차만 학습.
                             # False: TFT 단독 학습 (비교/ablation 용).
    # --- [Patch A] baseline 모드 ---
    #   "deterministic" : 추론 시 모든 시나리오에 VAR conditional mean(충격 0)을
    #     baseline 으로 주입 → 학습 시(in-sample fitted = 조건부 평균)와
    #     통계적 의미 일치. 시나리오 간 차이는 known_reals(feature 경로)로만 표현.
    #     → 잔차의 본래 의미("VAR 평균이 못 잡는 비선형/급변") 보존.
    #   "per_scenario" : 기존 동작(시나리오별 충격포함 경로를 baseline 으로 사용).
    #     학습/추론 분포 불일치 → 발산. ablation 비교용으로만 유지.
    baseline_mode="deterministic",
    # --- hybrid 잔차 발산(fan-out) 제어 ---
    #   level_p10/p90 = last_price * exp(cumsum(hybrid_p10/p90)) 구조에서
    #   quantile 별로 24~120 step 을 그대로 누적하면 P10/P90 spread 가
    #   기하급수적으로 폭주(상관 1.0 worst-case). 아래 3 knob 으로 통제.
    residual_decay=0.99,      # [Patch C] baseline 분포 불일치를 Patch A 로 해결했으므로
                             # 출력 압축 노브 모두 OFF. 1.0 = decay 끔.
                             # (구) 0.88 — 잔차 영향력의 월별 fade. rp_t *= decay**t.
                              # 1.0 = decay 끔, 0.90 ≈ 1년 후 영향 ~28%.
                              # 부동산은 mean-reverting → 0.85~0.95 권장.
                              # 근거 : Mean-reversion / shrinkage to prior 아이디어 — Hyndman 계열 forecast 교과서, ETS의 damped trend(Gardner & McKenzie 1985, forecast::holt(damped=TRUE) 의 φ 파라미터)와 동일한 형식. φ∈[0.8, 0.98] 권장 범위에서 중앙값 채택.
    residual_clip_abs=None, # 월간 잔차 절대값 cap (log-return 단위).
                              # 0.025 ≈ ±2.5%/월 → 단일 step 으로는 충분히 크고,
                              # 누적 발산은 차단. None 이면 cap 끔.
                              # 근거 : Gradient/output clipping 일반론. 월간 ±2.5% logret은 한국 아파트 월변동성 σ≈2.0%(2010-2025 KB지수 기준)의 ~1.25σ. "이상치는 자르되 정상 변동은 살린다"는 winsorize 정신.
    hybrid_alpha=1.0,         # [Patch C] 1.0 = 잔차 전량 반영.
                             # (구) 0.25 — boosting shrinkage. baseline 분포
                             # 불일치 가리기용. Patch A 로 더 이상 불필요.
                             # hybrid = alpha * residual + baseline.
                              # 1.0 = 잔차 전량 반영, 0.0 = VAR 단독으로 회귀.
                              # 0.5~0.7 이 보통 sharpness-coverage trade-off 최적.
                              # 근거 : Boosting의 learning rate (shrinkage) — Friedman 2001 "Greedy Function Approximation" §4. GBM의 ν 파라미터(0.050.3)와 같은 역할이되, 여기는 1-step 잔차 추가라 더 큰 값(0.7)이 합리적.
    hybrid_quantile_shrink=1.0,   # [Patch C] 1.0 = TFT quantile 원본 유지.
                                  # (구) 0.20 — p10/p90 → p50 압축. 누적 발산
                                  # 가리기용. Patch A 로 더 이상 불필요.
                              # p10/p90 의 중앙 회귀 정도(p50 쪽으로).
                              # 1.0 = TFT quantile 원본, 0.0 = 전부 p50 으로 붕괴.
                              # 24~120 step cumsum 발산을 추가 압축.
                              # 근거 : Quantile crossing 방지 & 분포 보수화 — Chernozhukov et al. 2010 "Quantile and Probability Curves Without Crossing"에서 분위 회귀 후처리의 한 형태. 단순히 p10/p90을 p50 쪽으로 끌어당기는 것은 quantile rearrangement의 단순화 버전.
)

ADF_ALPHA = VAR_CFG["adf_alpha"]    # alias (run_adf_report 용)


# ===========================================================================
# 1. 데이터 로딩
# ===========================================================================
def load_panel(csv_path: Path) -> pd.DataFrame:
    """원본 동 단위 패널을 그대로 로드. TFT 가 group_ids 로 다중 시계열 학습."""
    df = pd.read_csv(csv_path, parse_dates=[TIME_COL])
    df = df.sort_values(["dong", TIME_COL]).reset_index(drop=True)
    return df


def select_dong(panel: pd.DataFrame, dong_name: str) -> pd.DataFrame:
    """특정 동 단수를 VAR 입력용 단일 다변량 시계열로 추출.

    [근거]
      - 설계 의도(시/구/동 batch)에 맞게 광역 평균 대신 동별 분석을 수행.
      - 하나의 동이 이미 단일 다변량 시계열(VAR 입력조건) 이므로 추가 집계 불요.
      - policy__ 는 광역 단위이므로 동별 선택해도 동일한 정수값 유지.
    """
    sub = panel[panel["dong"] == dong_name].copy()
    if sub.empty:
        avail = sorted(panel["dong"].unique())[:10]
        raise ValueError(f"동 '{dong_name}' 미존재. 예시: {avail}")
    sub = sub.drop(columns=[c for c in ID_COLS if c in sub.columns])
    out = sub.set_index(TIME_COL).sort_index().asfreq("MS")
    policy_cols = [c for c in out.columns if c.startswith("policy__")]
    if policy_cols:
        out[policy_cols] = out[policy_cols].round().clip(-1, 1).astype(int)
    return out


# ===========================================================================
# 2. 패널 전처리 + log-return 타겟 생성  (TFT 학습용)
# ===========================================================================
def clean_missing_and_policy(panel: pd.DataFrame) -> pd.DataFrame:
    """동별 시간축 선형보간 + policy 정수화 + log-return 타겟 추가.

    [정책]
      - groupby("dong").interpolate(method="time") : 동 내부에서만 보간
        (cross-dong 누설 방지).
      - policy round/clip : 사용자 요구 (강화+1/유지0/완화-1) 강제.
      - target_logret(t) = log(price(t) / price(t-1))  → 정상성 확보.
        첫 월(t=0) NaN → 0.0 (min_encoder_length 로 실제 학습 sample 에서 제외됨).
    """
    out = panel.copy().set_index(TIME_COL)
    feature_cols = [c for c in out.columns if c not in ID_COLS]
    numeric_cols = [c for c in feature_cols
                    if pd.api.types.is_numeric_dtype(out[c])]
    out[numeric_cols] = (
        out.groupby("dong", group_keys=False)[numeric_cols]
           .apply(lambda g: g.interpolate(method="time",
                                          limit_direction="both"))
    )
    out = out.reset_index()
    policy_cols = [c for c in out.columns if c.startswith("policy__")]
    out[policy_cols] = out[policy_cols].round().clip(-1, 1).astype(int)
    out[numeric_cols] = out[numeric_cols].fillna(0.0)

    out = out.sort_values(["dong", TIME_COL]).reset_index(drop=True)
    out[TARGET_COL] = (
        out.groupby("dong")[TARGET_LEVEL_COL]
           .transform(lambda s: np.log(s).diff())
    ).fillna(0.0).astype(np.float32)
    return out


# ===========================================================================
# 3. ADF 리포트 (진단)
# ===========================================================================
def run_adf_report(panel: pd.DataFrame, save_path: Path) -> pd.DataFrame:
    """동별 TARGET_LEVEL_COL 시리즈에 ADF 검정 → 정상성 진단 csv."""
    rows = []
    for dong, g in panel.groupby("dong"):
        series = g[TARGET_LEVEL_COL].dropna().values
        if len(series) < 20:
            continue
        try:
            stat, pval, lags, nobs, _, _ = adfuller(
                series, regression="ct", autolag="AIC"
            )
            rows.append({"dong": dong, "adf_stat": stat, "p_value": pval,
                         "used_lag": lags, "n_obs": nobs,
                         "is_stationary": pval < ADF_ALPHA})
        except Exception as e:
            rows.append({"dong": dong, "adf_stat": np.nan, "p_value": np.nan,
                         "used_lag": np.nan, "n_obs": len(series),
                         "is_stationary": False, "error": str(e)})
    rep = pd.DataFrame(rows)
    rep.to_csv(save_path, index=False)
    n_stat = int(rep["is_stationary"].sum()) if "is_stationary" in rep else 0
    print(f"[ADF] stationary({ADF_ALPHA}) = {n_stat}/{len(rep)} dong")
    return rep


# ===========================================================================
# ===========  M1 ~ M5 : Bayesian VAR 시나리오 생성 파이프라인  =============
# ===========================================================================

# ---------------------------------------------------------------------------
# M1 — 정상화 (ADF 기반 자동 차분)
# ---------------------------------------------------------------------------
def m1_stationarize(seoul_df: pd.DataFrame,
                    feature_cols: List[str],
                    policy_cols: List[str]) -> Tuple[pd.DataFrame, Dict]:
    """raw 서울 시계열 → VAR 이 먹을 수 있는 정상 시계열.

    [동작]
      1) policy_cols 는 -1/0/1 이산값 → 차분/정규화 대상 아님 (스킵).
      2) 나머지 feature 는 ADF p-value 가 alpha 보다 크면 (=비정상) 1차 차분.
      3) 결측 보간 + winsor 윈저라이징 (상하위 q% 클리핑).
      4) "무엇을 어떻게 바꿨는지" transform_log 에 기록 → M5 에서 역변환.

    [근거]
      - ADF 검정은 단위근 존재 여부 검정. p > alpha → 단위근 있음 → 차분 필요.
      - 차분은 "이번달 값" → "이번달 변화량" 으로 바꿔 우상향 trend 제거.
      - winsor : 극단 outlier (입력 데이터의 이상치 / 잘못된 관측) 가 VAR 계수
        추정을 망치는 것을 방지. 단 winsor 폭은 보수성 노브일 뿐 경제가정 아님.
    """
    df = seoul_df.copy()
    transform_log: Dict[str, Dict] = {}

    for col in feature_cols:
        if col in policy_cols:
            transform_log[col] = {"diff": False}
            continue
        series = df[col].dropna()
        if len(series) < 12:
            transform_log[col] = {"diff": False}
            continue
        try:
            pval = adfuller(series.values, autolag="AIC")[1]
        except Exception:
            pval = 1.0   # 검정 실패 → 보수적으로 차분 수행
        if pval > VAR_CFG["adf_alpha"]:
            # 비정상 → 차분. 마지막 level 값을 기록해 둠 (M5 역변환용).
            transform_log[col] = {"diff": True,
                                  "last_level": float(df[col].iloc[-1])}
            df[col] = df[col].diff()
        else:
            transform_log[col] = {"diff": False}

    # 결측 보간 + winsor
    df[feature_cols] = df[feature_cols].interpolate(method="linear",
                                                     limit_direction="both")
    lo = df[feature_cols].quantile(VAR_CFG["winsor_lower"])
    hi = df[feature_cols].quantile(VAR_CFG["winsor_upper"])
    df[feature_cols] = df[feature_cols].clip(lower=lo, upper=hi, axis=1)

    stationary_df = df.dropna().reset_index(drop=True)
    return stationary_df, transform_log


# ---------------------------------------------------------------------------
# M2 — Bayesian VAR (Minnesota shrinkage)
# ---------------------------------------------------------------------------
def m2_bayesian_var(stationary_df: pd.DataFrame,
                    feature_cols: List[str]) -> Dict:
    """VAR 적합 → 계수 + 잔차 + 잔차공분산 + Minnesota shrinkage 적용.

    [동작]
      1) BIC 기준으로 lag 자동 선택 (max_lag 까지 탐색).
      2) VAR 적합 → fitted.coefs : (lag, n_feat, n_feat) 관계표.
      3) Minnesota shrinkage : 자기 자신 시차는 살리고, 타 변수·먼 시차는
         0 쪽으로 당김. 적은 데이터에서 계수 안정화 (BVAR 표준).
         shrink(L) = λ / (L+1)  → L=0 이면 λ, L=11 이면 λ/12.
      4) 잔차 = 모델이 설명 못한 충격 → M3 block bootstrap 의 재료.
    """
    data = stationary_df[feature_cols].values.astype(np.float64)
    model = VAR(data)
    try:
        order_sel = model.select_order(VAR_CFG["max_lag"])
        lag = max(1, int(order_sel.bic))
    except Exception:
        lag = 1
    fitted = model.fit(lag)

    coef = fitted.coefs.copy()              # (lag, n_feat, n_feat)
    resid = fitted.resid                    # (T-lag, n_feat)
    resid_cov = np.cov(resid.T)             # (n_feat, n_feat)

    # Minnesota shrinkage : 비대각(타변수) + 먼 시차일수록 더 0 쪽으로
    lam = VAR_CFG["minnesota_lambda"]
    n_feat = coef.shape[1]
    off_diag = ~np.eye(n_feat, dtype=bool)
    for L in range(coef.shape[0]):
        shrink = lam / (L + 1)
        coef[L][off_diag] *= (1 - shrink)

    return {"coef": coef, "resid": resid, "resid_cov": resid_cov,
            "lag": lag, "intercept": fitted.intercept,
            "feature_cols": feature_cols}


# ---------------------------------------------------------------------------
# M3+ — 잔차주머니 보강 (Student-t 꼬리)
# ---------------------------------------------------------------------------
def build_residual_pool(resid: np.ndarray) -> Dict:
    """잔차 주머니를 만들고, 옵션으로 변수별 Student-t 꼬리 적합.

    [근거]
      - 동 1개의 과거 데이터로는 극단 충격 표본이 부족 (2~3건).
      - Student-t (df=t_df_init) 적합 → "관측엔 없지만 그 분포라면 가능한"
        극단 충격을 생성 가능. df 가 작을수록 꼬리 두꺼움 (5~7 이 표준).
      - pool 에는 empirical(실측) 과 t_models(파라미터) 를 함께 보관.
        sample_shock_block 이 tail_prob 확률로 둘을 섞어 추출.
    """
    pool: Dict = {"empirical": resid}
    if VAR_CFG["fat_tail"]:
        t_models = []
        for j in range(resid.shape[1]):
            try:
                # scipy.stats.t.fit 는 (df, loc, scale) 반환.
                # f0=t_df_init → 초기 추정값. 자유롭게 최적화됨.
                df, loc, scale = stats.t.fit(resid[:, j],
                                             f0=VAR_CFG["t_df_init"])
            except Exception:
                # 적합 실패 → Gaussian fallback (df=∞ ≈ 1e6 으로 표현)
                df, loc, scale = 1e6, float(np.mean(resid[:, j])), \
                                 float(np.std(resid[:, j]) + 1e-9)
            t_models.append((float(df), float(loc), float(scale)))
        pool["t_models"] = t_models
    return pool


def sample_shock_block(pool: Dict, block_size: int,
                       rng: np.random.Generator) -> np.ndarray:
    """충격 block 한 덩어리(block_size, n_feat) 를 뽑는다.

    - 1 - tail_prob 확률: 실측 잔차에서 연속 block 리샘플 (현실 빈도 보존).
    - tail_prob 확률    : Student-t 꼬리에서 i.i.d. 추출 (극단 충격 보강).
    """
    emp = pool["empirical"]
    if ("t_models" in pool) and (rng.random() < VAR_CFG["tail_prob"]):
        n_feat = emp.shape[1]
        block = np.zeros((block_size, n_feat), dtype=np.float64)
        for j, (df, loc, scale) in enumerate(pool["t_models"]):
            block[:, j] = stats.t.rvs(df, loc=loc, scale=scale,
                                      size=block_size, random_state=rng)
        return block
    start = int(rng.integers(0, len(emp) - block_size + 1))
    return emp[start:start + block_size]


def required_N_hard(p_rare: float, k_min: int, confidence: float,
                    hard_floor: int) -> int:
    """N 하드조건 역산: P(X ≥ k_min) ≥ confidence  in  B(N, p_rare).

    [수식]
      X ~ Binomial(N, p_rare) ; 1 - CDF(k_min-1; N, p_rare) ≥ confidence
      를 만족하는 최소 N 을 찾는다.

    [근거]
      - "최소 1번"(기하분포) 으로 보장하면 외톨이 1개 경로뿐 → 꼬리 분포 불안정.
      - k_min ≈ 30 이면 P10/P90 같은 분위 안정성이 확보됨.
    """
    if p_rare <= 0:
        # 잔차주머니에 극단이 0 개 → N 으로 못 푼다. 보강(A) 단계 필요.
        # 여기서는 fallback : hard_floor 만 사용.
        return hard_floor
    N = k_min
    while binom.cdf(k_min - 1, N, p_rare) > (1 - confidence):
        N += 100
        if N > 10_000_000:    # 무한 루프 안전벨트
            break
    return max(N, hard_floor)


# ---------------------------------------------------------------------------
# M3 — block bootstrap 시나리오 생성
# ---------------------------------------------------------------------------
def m3_block_bootstrap(stationary_df: pd.DataFrame,
                       var_result: Dict,
                       feature_cols: List[str],
                       target_col: str,
                       horizon: int) -> Tuple[np.ndarray, int, Dict]:
    """VAR 엔진을 horizon 개월 굴려 N 시나리오 생성 (DIFF/정상화 공간).

    Returns
    -------
    scenarios : (N, horizon, n_feat)  — DIFF 공간 (M5 에서 역변환).
    N         : 최종 사용된 시나리오 수 (M3+ 하드조건 결과).
    pool      : 잔차주머니 (디버깅·재현용).

    [절차]
      1) build_residual_pool : 잔차주머니 + (옵션) t-tail.
      2) p_rare 산출 : 타깃 잔차 절대값 상위 (1 - extreme_quantile) 비율.
         block 단위로 보정 (block 당 추출확률 ≈ p_rare * block_size).
      3) required_N_hard : 위 조건으로 N 역산. n_scenarios 와 max() 로 floor.
      4) 시나리오 생성 :
         init  = stationary_df 의 마지막 lag 개월 실제값
         loop : next = intercept + Σ_L coef[L] @ recent[-L-1] + shock
                recent ← recent + next  (rolling window 갱신)
    """
    rng = np.random.default_rng(VAR_CFG["seed"])
    coef = var_result["coef"]
    resid = var_result["resid"]
    lag = var_result["lag"]
    intercept = var_result["intercept"]
    n_feat = len(feature_cols)
    tgt_idx = feature_cols.index(target_col)

    pool = build_residual_pool(resid)

    # --- M3+ (B) N 하드조건 ---
    abs_r_tgt = np.abs(pool["empirical"][:, tgt_idx])
    q_thresh = np.quantile(abs_r_tgt, VAR_CFG["extreme_quantile"])
    p_rare_per_step = float((abs_r_tgt > q_thresh).mean())   # ≈ 1 - q
    # block 단위 추출이므로 한 block 안에 극단이 들어갈 확률 근사
    p_rare_per_block = 1 - (1 - p_rare_per_step) ** VAR_CFG["block_size"]
    N = required_N_hard(p_rare_per_block,
                        k_min=VAR_CFG["extreme_k_min"],
                        confidence=VAR_CFG["confidence"],
                        hard_floor=max(VAR_CFG["n_scenarios"],
                                       VAR_CFG["hard_floor_N"]))
    print(f"[M3+] p_rare(block)={p_rare_per_block:.4f}  → N_hard={N}")

    init = stationary_df[feature_cols].values[-lag:].astype(np.float64)
    scenarios = np.zeros((N, horizon, n_feat), dtype=np.float64)

    for s in range(N):
        recent = init.copy()
        path = []
        t = 0
        while t < horizon:
            block = sample_shock_block(pool, VAR_CFG["block_size"], rng)
            for k in range(VAR_CFG["block_size"]):
                if t >= horizon:
                    break
                # ① VAR 평균 (intercept + Σ coef[L] @ recent[-L-1])
                mean_next = intercept.copy() if intercept is not None \
                            else np.zeros(n_feat)
                for L in range(lag):
                    mean_next = mean_next + coef[L] @ recent[-(L + 1)]
                # ② 충격 더하기
                next_val = mean_next + block[k]
                path.append(next_val)
                # ③ rolling window 갱신
                recent = np.vstack([recent, next_val])[-lag:]
                t += 1
        scenarios[s] = np.array(path)
    return scenarios, N, pool


# ---------------------------------------------------------------------------
# M4 — 현실성 필터 (리포트 모드 기본)
# ---------------------------------------------------------------------------
# [T1] 정의역 제약 — 변수 정의 자체이므로 상수로 둠 (경제 가정값 아님).
HARD_DOMAIN = {
    "ecos__base_rate":             (0.0, None),     # 금리는 음수 불가 (한국)
    "ecos__mortgage_rate_new":     (0.0, None),
    "ecos__cd_91d_rate":           (0.0, None),
    "ecos__cpi_housing":           (0.0, None),
    "ecos__unemployment_rate":     (0.0, 100.0),
    "reb__apt_sale_avg_price":     (0.0, None),
    "reb__apt_jeonse_avg_price":   (0.0, None),
    "reb__apt_sale_supply_demand": (0.0, 200.0),   # 매매수급지수 정의역
    "reb__apt_jeonse_supply_demand": (0.0, 200.0),
    "reb__apt_monthly_rent_supply_demand": (0.0, 200.0),
    "reb__apt_sale_index":         (0.0, None),
}


def _cov_to_corr(cov: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.diag(cov))
    return cov / np.outer(d, d + 1e-12)


def _rowwise_corr(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    Az = A - A.mean(axis=1, keepdims=True)
    Bz = B - B.mean(axis=1, keepdims=True)
    num = (Az * Bz).sum(axis=1)
    den = np.sqrt((Az ** 2).sum(axis=1) * (Bz ** 2).sum(axis=1)) + 1e-12
    return num / den


def m4_realism_check(paths_level: np.ndarray,
                     observed_seoul: pd.DataFrame,
                     var_result: Dict,
                     feature_cols: List[str],
                     target_col: str) -> Tuple[np.ndarray, Dict]:
    """경로별 3가지 현실성 검사. 기본 mode="report" (경로 안 버림).

    [T1] 정의역 이탈 (HARD_DOMAIN)
    [T2] 과거 월변화 상위 (1 - realism_extreme_q) 를 넘는 급변
    [T3] 변수쌍 부호 어긋남 — M2 잔차공분산에서 추정한 부호 vs 시나리오 부호.
    """
    th_chg = (observed_seoul[feature_cols].diff().abs()
                            .quantile(VAR_CFG["realism_extreme_q"]).to_dict())
    corr = _cov_to_corr(var_result["resid_cov"])
    pair_sign: Dict[Tuple[str, str], float] = {}
    cand_pairs = [(target_col, "reb__apt_sale_supply_demand"),
                  (target_col, "ecos__mortgage_rate_new"),
                  (target_col, "reb__apt_jeonse_avg_price")]
    for a, b in cand_pairs:
        if a in feature_cols and b in feature_cols:
            ia, ib = feature_cols.index(a), feature_cols.index(b)
            if abs(corr[ia, ib]) > VAR_CFG["realism_corr_threshold"]:
                pair_sign[(a, b)] = float(np.sign(corr[ia, ib]))

    N = paths_level.shape[0]
    v_t1 = np.zeros(N, dtype=bool)
    v_t2 = np.zeros(N, dtype=bool)
    v_t3 = np.zeros(N, dtype=bool)

    # T1
    for col, (lo, hi) in HARD_DOMAIN.items():
        if col not in feature_cols:
            continue
        j = feature_cols.index(col)
        v = paths_level[:, :, j]
        if lo is not None:
            v_t1 |= (v < lo).any(axis=1)
        if hi is not None:
            v_t1 |= (v > hi).any(axis=1)
    # T2
    for col, mx in th_chg.items():
        if col not in feature_cols or not np.isfinite(mx) or mx <= 0:
            continue
        j = feature_cols.index(col)
        step = np.abs(np.diff(paths_level[:, :, j], axis=1))
        v_t2 |= (step > mx).any(axis=1)
    # T3
    for (a, b), sign in pair_sign.items():
        ia, ib = feature_cols.index(a), feature_cols.index(b)
        da = np.diff(paths_level[:, :, ia], axis=1)
        db = np.diff(paths_level[:, :, ib], axis=1)
        pc = _rowwise_corr(da, db)
        v_t3 |= (np.sign(pc) != sign) & (np.abs(pc) > VAR_CFG["realism_pair_min_corr"])

    any_v = v_t1 | v_t2 | v_t3
    report = {
        "violation_rate": {
            "T1_domain":      float(v_t1.mean()),
            "T2_extreme":     float(v_t2.mean()),
            "T3_consistency": float(v_t3.mean()),
            "ANY":            float(any_v.mean()),
        },
        "pair_sign": {f"{a}|{b}": s for (a, b), s in pair_sign.items()},
    }
    if VAR_CFG["realism_mode"] == "drop":
        return paths_level[~any_v], report
    return paths_level, report


# ---------------------------------------------------------------------------
# M5 — 원단위 복원 + 검증 + 군집
# ---------------------------------------------------------------------------
def m5_restore_and_validate(scenarios_diff: np.ndarray,
                            transform_log: Dict,
                            feature_cols: List[str],
                            target_col: str,
                            actual_future_target: np.ndarray | None = None
                            ) -> Dict:
    """DIFF 공간 시나리오 → 원단위 복원 + 검증 점수 + 상승/보합/하락 군집.

    [복원]
      차분했던 변수: level[t] = last_level + Σ_{i<=t} diff[i]
      안 했던 변수: 그대로 사용.
    [검증] (actual_future_target 가 주어진 경우)
      coverage = P(p10 ≤ actual ≤ p90)
      width    = mean(p90 - p10)
      rank     = mean( P(scenario < actual) ) — 0.5 근처가 이상적.
    [군집] 최종월 수익률 기준 하위/중간/상위 1/3.
    """
    restored = scenarios_diff.copy()
    for col, info in transform_log.items():
        if col not in feature_cols or not info["diff"]:
            continue
        j = feature_cols.index(col)
        restored[:, :, j] = info["last_level"] + np.cumsum(
            scenarios_diff[:, :, j], axis=1
        )

    # --- 정책 변수 양자화 : 연속값 → {-1, 0, 1} ---
    # VAR 가 산출한 연속값을 임계값 기반으로 본래의 regime 코드로 강제.
    # 적용 시점: cumsum 복원 직후, target_paths 추출 전.
    if VAR_CFG.get("policy_quantize", True):
        thr = float(VAR_CFG.get("policy_quant_threshold", 0.5))
        policy_idx = [i for i, c in enumerate(feature_cols)
                      if c.startswith("policy__")]
        for j in policy_idx:
            v = restored[:, :, j]
            restored[:, :, j] = np.where(
                v >=  thr,  1.0,
                np.where(v <= -thr, -1.0, 0.0)
            )
        if policy_idx:
            print(f"[M5] policy 양자화: {len(policy_idx)} 변수, "
                  f"임계값=±{thr} → {{-1, 0, 1}}")

    tgt = feature_cols.index(target_col)
    tgt_paths = restored[:, :, tgt]            # (N, horizon)

    report: Dict = {}
    if actual_future_target is not None:
        L = min(len(actual_future_target), tgt_paths.shape[1])
        a = np.asarray(actual_future_target[:L], dtype=float)
        p = tgt_paths[:, :L]
        p10 = np.percentile(p, 10, axis=0)
        p90 = np.percentile(p, 90, axis=0)
        inside = (a >= p10) & (a <= p90)
        report["coverage"] = float(inside.mean())
        report["width"]    = float((p90 - p10).mean())
        report["rank_mean"] = float(np.mean(
            [(p[:, t] < a[t]).mean() for t in range(L)]
        ))

    final_ret = tgt_paths[:, -1] / np.clip(tgt_paths[:, 0], 1e-6, None) - 1

    # --- 군집 임계값 결정 ---
    # cluster_use_base_rate=True 면 복원된 ecos__base_rate 예측 평균(연율)을
    # rise_thr 로 사용. 기준금리만큼 오르지 않으면 실질적으로 보합/하락이라는 논리.
    rise_thr = float(VAR_CFG.get("cluster_rise_thr", 0.05))
    fall_thr = float(VAR_CFG.get("cluster_fall_thr", -0.05))
    if VAR_CFG.get("cluster_use_base_rate", False):
        br_col = "ecos__base_rate"
        if br_col in feature_cols:
            br_idx = feature_cols.index(br_col)
            # restored shape: (N, horizon, n_features)
            br_mean = float(np.nanmean(restored[:, :, br_idx]))
            # ecos__base_rate 는 % 단위 (e.g. 3.5) → 소수로 환산 후 누적
            # 예측 horizon 동안의 복리 환산: (1 + r/100)^H - 1
            H = restored.shape[1]
            rise_thr = float((1 + br_mean / 100) ** H - 1)
            fall_thr = -rise_thr
            print(f"[M5] 기준금리 평균={br_mean:.2f}% → "
                  f"rise_thr={rise_thr:+.4f} / fall_thr={fall_thr:+.4f}")
        else:
            print(f"[M5] cluster_use_base_rate=True 이나 '{br_col}' 컬럼 없음 "
                  f"→ 기본값 ±{rise_thr:.0%} 사용")

    # 임계값 기반 군집 분류
    clusters = {
        "하락": np.where(final_ret <  fall_thr)[0],
        "보합": np.where((final_ret >= fall_thr) & (final_ret <= rise_thr))[0],
        "상승": np.where(final_ret >  rise_thr)[0],
    }
    print(f"[M5] 군집 분류 (rise>{rise_thr:+.1%} / flat / fall<{fall_thr:+.1%}): "
          + " | ".join(f"{k}={len(v)}" for k, v in clusters.items()))

    bands = {}
    for name, idx in clusters.items():
        if len(idx) == 0:
            # 해당 군집에 속하는 시나리오 없음 → 빈 band 표시
            zero = np.full(tgt_paths.shape[1], np.nan)
            bands[name] = {"p10": zero, "p50": zero, "p90": zero}
            continue
        g = tgt_paths[idx]
        bands[name] = {
            "p10": np.percentile(g, 10, axis=0),
            "p50": np.percentile(g, 50, axis=0),
            "p90": np.percentile(g, 90, axis=0),
        }
    return {"restored": restored, "target_paths": tgt_paths,
            "validation": report, "clusters": bands}


# ===========================================================================
# 4. VAR baseline + TFT 학습 데이터 결합기
# ===========================================================================
def var_baseline_logret(restored_target_paths: np.ndarray,
                        last_train_price: float) -> np.ndarray:
    """N × horizon LEVEL 경로 → N × horizon log-return 경로.

    [공식]
      logret[s, t] = log(level[s, t] / level[s, t-1])  (s = 시나리오, t ≥ 1)
      logret[s, 0] = log(level[s, 0] / last_train_price)
    """
    N, H = restored_target_paths.shape
    out = np.zeros_like(restored_target_paths)
    out[:, 0] = np.log(np.clip(restored_target_paths[:, 0], 1e-9, None) /
                       max(last_train_price, 1e-9))
    out[:, 1:] = np.log(np.clip(restored_target_paths[:, 1:], 1e-9, None) /
                        np.clip(restored_target_paths[:, :-1], 1e-9, None))
    return out.astype(np.float32)


def var_deterministic_future_logret(var_result: Dict,
                                     stat_df: pd.DataFrame,
                                     tlog: Dict,
                                     last_price: float,
                                     horizon: int,
                                     feature_cols: List[str],
                                     target_col: str) -> np.ndarray:
    """[Patch A] VAR conditional-mean future path → target log-return (horizon,).

    [목적]
      추론 단계 TFT baseline 의 통계적 정의를 학습 단계와 일치시킴.
      - 학습 baseline = VAR in-sample fitted log-return (= 조건부 평균)
      - 기존 추론 baseline = M3 bootstrap 시나리오별 충격 포함 경로
        → 충격이 baseline 에 섞여 들어가 TFT 잔차가 본래 의미(잔차)에서 이탈.
      - 본 함수는 충격을 0 으로 두고 VAR 자체의 평균 경로만 산출.

    [동작]
      1) recent ← stat_df 의 마지막 lag 행 (정상화 공간)
      2) for t in horizon:
           mean_next = intercept + Σ coef[L] @ recent[-L-1]   (충격 없음)
           recent ← (recent + mean_next) [-lag:]
      3) tlog 의 diff 여부에 따라 target 채널만 level 복원
         (diff: last_level + cumsum 으로 절대수준; non-diff: 그대로)
      4) level → 1 단계 log-return 으로 변환 (last_price 가 t=0 직전 시점)
    """
    coef       = var_result["coef"]                # (lag, n_feat, n_feat)
    lag        = int(var_result["lag"])
    intercept  = var_result["intercept"]
    n_feat     = len(feature_cols)
    tgt_idx    = feature_cols.index(target_col)

    init   = stat_df[feature_cols].values[-lag:].astype(np.float64)
    recent = init.copy()
    mean_path = np.zeros((horizon, n_feat), dtype=np.float64)
    for t in range(horizon):
        mean_next = (intercept.copy()
                     if intercept is not None else np.zeros(n_feat))
        for L in range(lag):
            mean_next = mean_next + coef[L] @ recent[-(L + 1)]
        mean_path[t] = mean_next
        recent = np.vstack([recent, mean_next])[-lag:]

    # diff 였다면 cumsum 복원, 아니면 그대로
    info = tlog.get(target_col, {"diff": False})
    if info.get("diff", False):
        level = float(info["last_level"]) + np.cumsum(mean_path[:, tgt_idx])
    else:
        level = mean_path[:, tgt_idx]

    # level → log-return (이전월 대비)
    logret = np.zeros(horizon, dtype=np.float32)
    prev = max(float(last_price), 1e-9)
    for t in range(horizon):
        cur = max(float(level[t]), 1e-9)
        logret[t] = float(np.log(cur / prev))
        prev = cur
    return logret


# ===========================================================================
# ============================  TFT 파이프라인  ============================
# ===========================================================================
def prepare_for_tft(df: pd.DataFrame) -> pd.DataFrame:
    """TimeSeriesDataSet 스키마: int time_idx, str categorical, recency weight."""
    out = df.copy()
    base = out[TIME_COL].min()
    out["time_idx"] = ((out[TIME_COL].dt.year - base.year) * 12 +
                       (out[TIME_COL].dt.month - base.month)).astype(int)
    out["si"]   = out["si"].astype(str)
    out["gu"]   = out["gu"].astype(str)
    out["dong"] = out["dong"].astype(str)
    out["month"] = out[TIME_COL].dt.month.astype(str)

    # 정책 변수: {-1, 0, 1} 이산값을 그대로 numeric 으로 유지.
    #   → column_roles 에서 known_reals 로 속하며 (loose < keep < tight) 순서가
    #     보존되고, VAR 시나리오 주입(이미 numeric)과도 일관.
    policy_cols = [c for c in out.columns if c.startswith("policy__")]
    for c in policy_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0).astype(np.float32)
        # 정책 효과 3개월 지연 반영: lag3 컬럼 추가 (dong별 groupby)
        lag_col = f"{c}_lag3"
        out[lag_col] = (
            out.groupby("dong", group_keys=False)[c]
               .shift(3).fillna(0.0).astype(np.float32)
        )

    # Recency weighted loss : 2020+ regime 주세 학습.
    t_max = int(out["time_idx"].max())
    halflife = float(TFT_CFG["recency_halflife_months"])
    out["loss_weight"] = (0.5 ** ((t_max - out["time_idx"]) / halflife)
                          ).astype(np.float32)
    return out


def column_roles(df: pd.DataFrame, target: str) -> Dict[str, List[str]]:
    """TFT 컬럼 역할 매핑 (재설계).

    설계 원칙 (VAR-잔차 보정 구조):
      - target (residual_logret) : time_varying_unknown_reals 의 **유일한** 변수.
        encoder 구간까지만 값이 들어가고 decoder 구간은 모델이 스스로 예측.
      - 모든 VAR 시나리오 경로 (ecos__/reb__/policy__/var_baseline_logret) : known_reals.
        encoder + decoder 풀 길이로 주입되어 조건부 예측의 뼈대를 제공.
        (정책은 M5 양자화로 {-1, 0, 1} 이산값이지만 순서형이라 known_reals 로 처리.)
      - 정적 미시 변수 : gu, dong (행정구역 + 동 ID).
    """
    policy_cols = [c for c in df.columns if c.startswith("policy__")]
    # 모든 VAR-projected reals → known (target/원본 타겟은 제외)
    var_projected_reals = [
        c for c in df.columns
        if (c.startswith("ecos__") or c.startswith("reb__"))
        and c != target and c != TARGET_COL and c != TARGET_LEVEL_COL
    ]
    known_reals = ["time_idx"]
    if "var_baseline_logret" in df.columns:
        known_reals.append("var_baseline_logret")
    known_reals.extend(var_projected_reals)
    known_reals.extend(policy_cols)   # 정책 변수도 known_reals 로
    # unknown = target 잔차 단 1개
    unknown_reals = [target]
    return {
        "static_categoricals": ["gu", "dong"],
        "static_reals": [],
        "time_varying_known_categoricals": ["month"],
        "time_varying_known_reals": known_reals,
        "time_varying_unknown_reals": unknown_reals,
    }


def build_training_dataset(df: pd.DataFrame, training_cutoff: int,
                            max_pred_len: int,
                            target: str) -> TimeSeriesDataSet:
    """TimeSeriesDataSet 빌더. target = TARGET_COL 또는 'residual_logret'.

    GroupNormalizer(transformation=None, center=True):
      - log-return / residual 모두 0 중심 표준화가 정석.
      - center=True → 동별 평균을 빼서 NN 학습 안정화.
    """
    roles = column_roles(df, target=target)
    cat_encoders = {c: NaNLabelEncoder(add_nan=True)
                    for c in roles["time_varying_known_categoricals"]}
    return TimeSeriesDataSet(
        df[df["time_idx"] <= training_cutoff],
        time_idx="time_idx",
        target=target,
        group_ids=["dong"],
        weight="loss_weight",
        categorical_encoders=cat_encoders,
        max_encoder_length=TFT_CFG["max_encoder_length"],
        max_prediction_length=max_pred_len,
        min_encoder_length=TFT_CFG["max_encoder_length"] // 2,
        min_prediction_length=1,
        static_categoricals=roles["static_categoricals"],
        static_reals=roles["static_reals"],
        time_varying_known_categoricals=roles["time_varying_known_categoricals"],
        time_varying_known_reals=roles["time_varying_known_reals"],
        time_varying_unknown_reals=roles["time_varying_unknown_reals"],
        target_normalizer=GroupNormalizer(
            groups=["dong"], transformation=None, center=True
        ),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )


def train_tft(training: TimeSeriesDataSet, validation: TimeSeriesDataSet,
              log_subdir: str
              ) -> Tuple[TemporalFusionTransformer, pl.Trainer]:
    """QuantileLoss(10/50/90%) 기반 TFT 학습."""
    train_loader = training.to_dataloader(
        train=True, batch_size=TFT_CFG["batch_size"],
        num_workers=TFT_CFG["num_workers"],
    )
    val_loader = validation.to_dataloader(
        train=False, batch_size=TFT_CFG["batch_size"] * 2,
        num_workers=TFT_CFG["num_workers"],
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=TFT_CFG["early_stop_patience"],
        mode="min",
    )
    logger = CSVLogger(save_dir=str(OUT_DIR / "lightning_logs"),
                       name=log_subdir)
    trainer = pl.Trainer(
        max_epochs=TFT_CFG["max_epochs"],
        accelerator="cpu",
        gradient_clip_val=TFT_CFG["gradient_clip_val"],
        callbacks=[early_stop],
        logger=logger,
        enable_progress_bar=True,
        enable_model_summary=False,
        log_every_n_steps=10,
    )
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=TFT_CFG["learning_rate"],
        hidden_size=TFT_CFG["hidden_size"],
        attention_head_size=TFT_CFG["attention_head_size"],
        dropout=TFT_CFG["dropout"],
        hidden_continuous_size=TFT_CFG["hidden_continuous_size"],
        loss=QuantileLoss(quantiles=TFT_CFG["quantiles"]),
        log_interval=0,
        optimizer="adamw",
        reduce_on_plateau_patience=2,
    )
    trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)
    best = trainer.checkpoint_callback.best_model_path
    if best:
        tft = TemporalFusionTransformer.load_from_checkpoint(best)
    return tft, trainer


def _decode_predictions(pred_out, idx_to_ts: Dict[int, pd.Timestamp],
                        group_map=None,
                        dataset: Optional[TimeSeriesDataSet] = None) -> pd.DataFrame:
    """raw prediction → long-form DataFrame (dong, timestamp, p10/p50/p90).

    group_map(dict[int -> dong명]) 가 주어지면 그것을 우선 사용한다.
    그렇지 않고 dataset 이 주어지면 dataset.x_to_index(x) 로 dong 컬럼을
    안전하게 디코딩한다 (group_ids encoder 이름이 버전마다 다른 문제 회피).
    """
    raw, x = pred_out.output, pred_out.x
    pred_q = raw["prediction"].detach().cpu().numpy()
    q = TFT_CFG["quantiles"]
    qi_med = q.index(0.5)
    decoder_time_idx = x["decoder_time_idx"].detach().cpu().numpy()

    if group_map is not None:
        group_ids = x["groups"].detach().cpu().numpy()
        dong_names = [group_map[int(g[0])] for g in group_ids]
    elif dataset is not None:
        idx_df = dataset.x_to_index(x)
        # x_to_index 는 batch 행마다 (dong, time_idx) 를 디코딩해 줌.
        # 행 순서는 pred_q 의 0차원 순서와 동일.
        dong_names = idx_df["dong"].astype(str).tolist()
    else:
        raise ValueError("_decode_predictions: group_map 또는 dataset 중 하나는 필요")

    records = []
    for i, dong in enumerate(dong_names):
        for k in range(pred_q.shape[1]):
            ti = int(decoder_time_idx[i, k])
            records.append({
                "dong": dong,
                "timestamp": idx_to_ts.get(ti),
                "pred_p10": float(pred_q[i, k, 0]),
                "pred_p50": float(pred_q[i, k, qi_med]),
                "pred_p90": float(pred_q[i, k, -1]),
            })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# TFT 보조 저장 1 — VSN (Variable Selection Network) 가중치
# ---------------------------------------------------------------------------
def _save_vsn_weights(model: TemporalFusionTransformer,
                      pred_out,
                      training_ds: TimeSeriesDataSet,
                      save_path: Path) -> None:
    """TFT VSN 가중치를 CSV 로 저장.

    TFT 는 encoder / decoder / static 세 그룹에서 각 변수의 '기여 강도'를
    Gated Residual Network(GRN) → softmax 로 산출 (Variable Selection Network).
    높을수록 해당 구간에서 더 많이 참조된 변수임을 의미.

    저장 컬럼:
      scope       : encoder | decoder | static
      variable    : 변수명
      vsn_weight  : 정규화된 중요도 (배치 평균)
    """
    try:
        # reduction="mean" : 배치 전체 샘플을 평균하여 대표 중요도 반환
        interp = model.interpret_output(pred_out.output, reduction="mean")

        # TimeSeriesDataSet 속성에서 변수 이름 순서를 읽어 가중치와 매핑
        enc_vars = (list(training_ds.time_varying_unknown_reals) +
                    [v for v in training_ds.time_varying_known_reals
                     if v != "time_idx"])
        dec_vars  = list(training_ds.time_varying_known_reals)
        sta_vars  = (list(training_ds.static_categoricals) +
                     list(training_ds.static_reals))

        rows = []
        for scope, key, names in [
            ("encoder", "encoder_variables", enc_vars),
            ("decoder", "decoder_variables", dec_vars),
            ("static",  "static_variables",  sta_vars),
        ]:
            if key not in interp:
                continue
            weights = interp[key].detach().cpu().numpy().flatten()
            for i, w in enumerate(weights):
                var_name = names[i] if i < len(names) else f"{scope}_var_{i}"
                rows.append({"scope": scope, "variable": var_name,
                             "vsn_weight": float(w)})

        if rows:
            out_df = (pd.DataFrame(rows)
                      .sort_values(["scope", "vsn_weight"],
                                   ascending=[True, False])
                      .reset_index(drop=True))
            out_df.to_csv(save_path, index=False)
            print(f"[VSN] 저장 완료: {save_path}")
    except Exception as e:
        print(f"[VSN] 추출 실패 (스킵): {e}")


# ---------------------------------------------------------------------------
# TFT 보조 저장 2 — VAR M5 시나리오 분위 밴드
# ---------------------------------------------------------------------------
def _save_var_scenario_bands(target_paths: np.ndarray,
                              dates: pd.DatetimeIndex,
                              save_path: Path,
                              scenario_tag: str = "") -> None:
    """VAR M5 target_paths (N, horizon) → 분위 밴드 CSV.

    저장 컬럼: step, timestamp, p5, p10, p25, p50, p75, p90, p95, N_scenarios
    p5~p95 분위 7개를 저장해 M3 시나리오 분포 형태를 시각화·분석에 활용.
    """
    qs   = [5, 10, 25, 50, 75, 90, 95]
    pcts = np.percentile(target_paths, qs, axis=0)   # (7, horizon)
    rows = []
    for t, ts in enumerate(dates):
        row = {"step": t + 1, "timestamp": ts,
               "N_scenarios": target_paths.shape[0]}
        for q, val in zip(qs, pcts[:, t]):
            row[f"p{q}"] = float(val)
        rows.append(row)
    df_band = pd.DataFrame(rows)
    df_band.to_csv(save_path, index=False)
    print(f"[M5 bands] 저장 ({scenario_tag}): {save_path}")


# ---------------------------------------------------------------------------
# M0~M5 단계별 로깅 헬퍼 (중간 산출물 검증용)
# ---------------------------------------------------------------------------
def _save_transform_log(tlog: Dict, save_path: Path) -> None:
    """M1 transform_log dict → CSV (feature, diff 여부, last_level)."""
    rows = []
    for col, info in tlog.items():
        rows.append({
            "feature": col,
            "diff_applied": bool(info.get("diff", False)),
            "last_level": info.get("last_level", np.nan),
        })
    pd.DataFrame(rows).to_csv(save_path, index=False)
    print(f"[M1 log] transform_log 저장: {save_path}")


def _save_var_result_csvs(vr: Dict, feature_cols: List[str],
                           out_dir: Path, prefix: str) -> None:
    """M2 VAR 적합 결과 → 4개 CSV (계수 / 잔차 / 잔차공분산 / 메타).

    {prefix}_var_coefficients.csv : (lag, dependent, regressor, coef) — Minnesota shrinkage 적용 후
    {prefix}_var_residuals.csv    : (T-lag) x n_feat — block bootstrap 의 재료
    {prefix}_var_residual_cov.csv : n_feat x n_feat — M4 부호 일관성 검사 입력
    {prefix}_var_meta.csv         : selected_lag, n_obs, intercept 등 요약
    """
    coef = vr["coef"]
    lag = coef.shape[0]
    rows = []
    for L in range(lag):
        for i, dep in enumerate(feature_cols):
            for j, indep in enumerate(feature_cols):
                rows.append({"lag": L + 1,
                             "dependent": dep,
                             "regressor": indep,
                             "coef": float(coef[L, i, j])})
    pd.DataFrame(rows).to_csv(out_dir / f"{prefix}_var_coefficients.csv",
                              index=False)

    resid = vr["resid"]
    pd.DataFrame(resid, columns=feature_cols).to_csv(
        out_dir / f"{prefix}_var_residuals.csv", index=False
    )
    pd.DataFrame(vr["resid_cov"],
                 index=feature_cols, columns=feature_cols).to_csv(
        out_dir / f"{prefix}_var_residual_cov.csv"
    )

    intercept = vr["intercept"]
    int_str = (",".join(f"{x:.6g}" for x in intercept.tolist())
               if intercept is not None else "")
    pd.DataFrame([{
        "selected_lag": int(vr["lag"]),
        "n_features": len(feature_cols),
        "n_residual_obs": int(resid.shape[0]),
        "minnesota_lambda": VAR_CFG["minnesota_lambda"],
        "intercept_vec": int_str,
    }]).to_csv(out_dir / f"{prefix}_var_meta.csv", index=False)
    print(f"[M2] VAR result 4종 저장: {prefix}_var_*")


def _save_residual_pool(pool: Dict, feature_cols: List[str],
                         save_path: Path) -> None:
    """M3+ 잔차주머니 → feature별 Student-t MLE 파라미터 CSV.

    컬럼: feature, t_df, t_loc, t_scale, empirical_n_obs, empirical_mean, empirical_std
    """
    emp = pool["empirical"]
    rows = []
    for j, col in enumerate(feature_cols):
        row = {
            "feature": col,
            "empirical_n_obs": int(emp.shape[0]),
            "empirical_mean": float(emp[:, j].mean()),
            "empirical_std":  float(emp[:, j].std()),
        }
        if "t_models" in pool:
            df_t, loc, scale = pool["t_models"][j]
            row["t_df"] = df_t
            row["t_loc"] = loc
            row["t_scale"] = scale
        rows.append(row)
    pd.DataFrame(rows).to_csv(save_path, index=False)
    print(f"[M3+] residual pool 저장: {save_path}")


def _save_scenarios_long(scenarios: np.ndarray,
                          dates,
                          feature_cols: List[str],
                          save_path: Path,
                          space_tag: str = "") -> None:
    """M3 / M5 (N, horizon, n_feat) 시나리오 텐서 → long-form CSV.

    컬럼: scenario_id, step, timestamp, feature, value, space
      - scenario_id : 0 ~ N-1  (모든 개별 시나리오 보존)
      - step        : 1 ~ horizon
      - space       : 'diff' (M3) | 'level' (M5 복원 후) 등 태그

    vectorized 구현 → 수백만 행도 빠르게 처리.
    """
    N, H, F = scenarios.shape
    sid  = np.repeat(np.arange(N), H * F)
    step = np.tile(np.repeat(np.arange(H), F), N) + 1
    ts   = np.tile(np.repeat(np.asarray(dates), F), N)
    feat = np.tile(np.asarray(feature_cols), N * H)
    val  = scenarios.reshape(-1)
    df = pd.DataFrame({
        "scenario_id": sid,
        "step":        step,
        "timestamp":   ts,
        "feature":     feat,
        "value":       val,
    })
    if space_tag:
        df["space"] = space_tag
    df.to_csv(save_path, index=False)
    print(f"[scenarios] {space_tag} N={N} x H={H} x F={F} → {save_path}")


def _save_target_paths_long(target_paths: np.ndarray,
                             dates,
                             save_path: Path,
                             restored: np.ndarray | None = None,
                             feature_cols: List[str] | None = None,
                             target_col: str | None = None) -> None:
    """M5 target_paths (N, horizon) → long-form CSV (개별 시나리오 모두 보존).

    기본 컬럼: scenario_id, step, timestamp, target_level
      - TFT 투입 전 BVAR 최종 산출물 (원단위 복원된 N개 target 경로).

    [확장 옵션]
      restored      : M5 (N, horizon, n_feat) 전체 복원 시나리오 텐서
      feature_cols  : restored 의 마지막 축 변수명 (순서 일치)
      target_col    : restored 에서 target 구분용 (출력 컬럼명은 'target_level' 고정)

    세 파라미터가 모두 주어지면 target_level 우측에 각 feature 컬럼을
    붙여 wide-form 으로 저장 → "target 생성에 쓰인 모든 변수의
    동일 시나리오/시점 값"을 한 번에 검증 가능.
    """
    N, H = target_paths.shape
    sid  = np.repeat(np.arange(N), H)
    step = np.tile(np.arange(H), N) + 1
    ts   = np.tile(np.asarray(dates), N)
    val  = target_paths.reshape(-1)
    out = pd.DataFrame({
        "scenario_id":  sid,
        "step":         step,
        "timestamp":    ts,
        "target_level": val,
    })

    # —— 확장: feature 컬럼 붙이기 ——
    if restored is not None and feature_cols is not None:
        if restored.shape[:2] != (N, H):
            print(f"[M5 target_paths] restored shape 불일치 → feature 저장 스킵: "
                  f"restored={restored.shape}, target={target_paths.shape}")
        else:
            for j, col in enumerate(feature_cols):
                # target_col 과 중복되는 컬럼은 건너뜀 (target_level 이 이미 있음)
                if target_col is not None and col == target_col:
                    continue
                out[col] = restored[:, :, j].reshape(-1)

    out.to_csv(save_path, index=False)
    extra = (len(out.columns) - 4) if (restored is not None
                                       and feature_cols is not None) else 0
    print(f"[M5] target_paths N={N} x H={H} "
          f"(개별 시나리오 전체, +{extra} feature 컬럼) → {save_path}")


def _save_realism_report(realism: Dict, save_path: Path) -> None:
    """M4 위반율 + pair_sign → 단일 CSV."""
    rows = []
    for k, v in realism["violation_rate"].items():
        rows.append({"item": f"violation_rate__{k}",
                     "value": float(v),
                     "note": ""})
    for pair, s in realism.get("pair_sign", {}).items():
        rows.append({"item": f"pair_sign__{pair}",
                     "value": float(s),
                     "note": "M2 resid corr 부호 기준"})
    rows.append({"item": "realism_mode",
                 "value": np.nan,
                 "note": VAR_CFG["realism_mode"]})
    pd.DataFrame(rows).to_csv(save_path, index=False)
    print(f"[M4] realism report 저장: {save_path}")


# ===========================================================================
# 5. VAR baseline 을 panel 에 결합 (residual mode)
# ===========================================================================
def attach_var_baseline(panel_with_logret: pd.DataFrame,
                        var_baseline_per_month: Dict[pd.Timestamp, float]
                        ) -> pd.DataFrame:
    """panel 의 각 행에 VAR baseline log-return 컬럼을 부착하고,
    residual_logret = target_logret - var_baseline_logret 를 만든다.

    VAR baseline 은 *서울 광역* 1 series 이므로 모든 dong 에 동일 broadcast.
    (동별 이질성은 TFT 잔차가 학습 — 의도된 분업.)
    """
    out = panel_with_logret.copy()
    out["var_baseline_logret"] = out[TIME_COL].map(var_baseline_per_month
                                                   ).astype(np.float32)
    out["var_baseline_logret"] = out["var_baseline_logret"].fillna(0.0)
    out["residual_logret"] = (out[TARGET_COL] - out["var_baseline_logret"]
                              ).astype(np.float32)
    return out


def var_in_sample_baseline_map(var_result: Dict,
                                stat_df: pd.DataFrame,
                                tlog: Dict,
                                level_panel: pd.DataFrame,
                                feature_cols: List[str]) -> Dict[pd.Timestamp, float]:
    """과거 학습 구간의 VAR fitted log-return baseline 을 dict 로 반환.

    [필요성]
      TFT 가 'residual = target - VAR baseline' 을 학습하려면
      과거 학습 구간에도 실제 VAR fitted 값을 baseline 으로 부여해야 한다.
      과거 baseline 을 0 으로 두면 residual = target 이 되어
      TFT 는 잔차가 아닌 원본 수익률을 학습 → 추론 시 baseline 합산하면
      double counting (집값 폭주/폭락 궤적).

    [계산]
      stat_df 의 TARGET_LEVEL_COL 가 DIFF 공간이라 가정 (M1 에서 차분된 경우).
      - actual_diff_t = stat_df[target].iloc[lag + k]   (실제 변화량)
      - fitted_diff_t = actual_diff_t - resid_t          (VAR 가 맞춘 변화량)
      - level_{t-1}   = level_panel[target] @ (date_t - 1month)
      - fitted_level_t = level_{t-1} + fitted_diff_t
      - in_sample_logret_t = log(fitted_level_t / level_{t-1})

      target 이 diff 안 된 경우 (희박) → 안전하게 0 으로 두고 경고.

    Returns: {timestamp: in_sample_logret}.  학습 구간 전체에 대한 default 0,
             VAR 적합 가능한 시점만 실제 baseline 부여.
    """
    tgt_diffed = bool(tlog.get(TARGET_LEVEL_COL, {}).get("diff", False))
    all_dates = pd.to_datetime(level_panel[TIME_COL].unique())
    baseline_map = {pd.Timestamp(d): 0.0 for d in all_dates}

    if not tgt_diffed:
        print(f"[WARN] TARGET '{TARGET_LEVEL_COL}' was not differenced in M1 → "
              f"in-sample baseline=0 (residual=target).")
        return baseline_map

    lag = int(var_result["lag"])
    tgt_idx = feature_cols.index(TARGET_LEVEL_COL)
    resid_past = var_result["resid"][:, tgt_idx]
    n_resid = len(resid_past)

    # stat_df rows that the VAR resid corresponds to: stat_df.iloc[lag : lag + n_resid]
    if len(stat_df) < lag + n_resid:
        print(f"[WARN] stat_df({len(stat_df)}) < lag({lag})+n_resid({n_resid}); "
              f"skip in-sample baseline.")
        return baseline_map

    actual_diff_past = stat_df[TARGET_LEVEL_COL].values[lag : lag + n_resid]
    fitted_diff_past = actual_diff_past - resid_past

    today_dates = pd.to_datetime(stat_df[TIME_COL].values[lag : lag + n_resid])
    yesterday_dates = today_dates - pd.DateOffset(months=1)

    # level_panel : TIME_COL 컬럼 + TARGET_LEVEL_COL 컬럼이 있는 dong 단위 DataFrame
    level_ser = (level_panel.drop_duplicates(subset=[TIME_COL])
                            .set_index(TIME_COL)[TARGET_LEVEL_COL])
    yest_level = level_ser.reindex(yesterday_dates).values.astype(float)

    valid = ~np.isnan(yest_level)
    fitted_level_today = yest_level + fitted_diff_past
    safe_yest = np.clip(yest_level, 1e-9, None)
    safe_today = np.clip(fitted_level_today, 1e-9, None)
    logret = np.log(safe_today / safe_yest)

    for d, v, ok in zip(today_dates, logret, valid):
        if ok and np.isfinite(v):
            baseline_map[pd.Timestamp(d)] = float(v)

    print(f"[VAR baseline] in-sample fitted log-return 계산: "
          f"n={int(valid.sum())} / {n_resid} (lag={lag})")
    return baseline_map


# ===========================================================================
# 6. 백테스트 시나리오 (학습 ~ TRAIN_END / 검증 VALID_START~VALID_END)
# ===========================================================================
# 6-0. 시나리오별 TFT forward pass 헬퍼 (N개 시나리오 → N개 hybrid 경로)
# ===========================================================================
def _tft_predict_per_scenario(model,
                               training,
                               panel_dong_logret: pd.DataFrame,
                               restored: np.ndarray,
                               target_paths: np.ndarray,
                               feature_cols: List[str],
                               fut_dates,
                               last_obs_idx: pd.Timestamp,
                               last_price: float,
                               horizon: int,
                               baseline_past_map: Dict[pd.Timestamp, float] = None,
                               deterministic_baseline_seq: np.ndarray = None,
                               target_dong: str = None) -> Dict[str, np.ndarray]:
    """학습된 TFT 모델 + N개 VAR 시나리오 → N개 hybrid 경로.

    각 시나리오 s 에 대해:
      1) future_rows : ecos__/reb__/policy__ 컬럼을 scenario_s 의 경로로 주입
      2) baseline_logret_s : scenario_s 의 target_path → log-return
      3) TimeSeriesDataSet.from_dataset(training, df_s, predict=True)
         → model.predict 로 잔차(또는 logret) 예측
      4) hybrid = TFT residual + baseline_logret_s  (residual_mode 일 때)
      5) level 복원 : last_price * exp(cumsum(hybrid))

    Returns: dict of (N, horizon) arrays —
      tft_raw_p10/p50/p90, hybrid_p10/p50/p90, level_p10/p50/p90, baseline.
    """
    N_scen = restored.shape[0]
    tft_target = "residual_logret" if TFT_CFG["residual_mode"] else TARGET_COL
    baseline_mode = TFT_CFG.get("baseline_mode", "deterministic")

    # [Patch B] 추론용 panel 은 target_dong 만 사용 — full panel 이 들어와도 필터.
    if target_dong is not None and "dong" in panel_dong_logret.columns:
        panel_dong_logret = panel_dong_logret[
            panel_dong_logret["dong"] == target_dong
        ].copy()

    raw_p10  = np.zeros((N_scen, horizon))
    raw_p50  = np.zeros((N_scen, horizon))
    raw_p90  = np.zeros((N_scen, horizon))
    hyb_p10  = np.zeros((N_scen, horizon))
    hyb_p50  = np.zeros((N_scen, horizon))
    hyb_p90  = np.zeros((N_scen, horizon))
    baseline = np.zeros((N_scen, horizon))

    last_snap = panel_dong_logret[panel_dong_logret[TIME_COL] == last_obs_idx]\
                    .copy()
    if last_snap.empty:
        # last_obs_idx 가 train_end 이후일 수 있으므로 가장 가까운 과거 사용
        last_snap = (panel_dong_logret[panel_dong_logret[TIME_COL] <= last_obs_idx]
                     .sort_values(TIME_COL).tail(1).copy())
        last_obs_idx = pd.Timestamp(last_snap[TIME_COL].iloc[0])

    # 학습 구간(과거) baseline : 학습 시 사용한 값과 동일해야 인코더 일관성 유지
    past_dates = panel_dong_logret[panel_dong_logret[TIME_COL] <= last_obs_idx][TIME_COL].unique()
    if baseline_past_map is None:
        baseline_past = {d: 0.0 for d in past_dates}
    else:
        baseline_past = {pd.Timestamp(d): float(baseline_past_map.get(pd.Timestamp(d), 0.0))
                          for d in past_dates}

    progress_every = max(1, N_scen // 10)
    print(f"[TFT N-pass] N={N_scen} 시나리오 forward pass 시작 "
          f"(baseline_mode={baseline_mode}) ...")

    # [Patch A] deterministic 모드 : N 번 동일한 baseline_seq 사용.
    det_seq = None
    if baseline_mode == "deterministic":
        if deterministic_baseline_seq is None:
            raise ValueError(
                "baseline_mode='deterministic' 이지만 deterministic_baseline_seq 가 없음. "
                "호출자에서 var_deterministic_future_logret(...) 을 먼저 산출해 주입하세요."
            )
        det_seq = np.asarray(deterministic_baseline_seq, dtype=np.float32)[:horizon]
        if len(det_seq) < horizon:
            det_seq = np.concatenate(
                [det_seq, np.zeros(horizon - len(det_seq), np.float32)]
            )

    for s in range(N_scen):
        # 1) baseline logret —— [Patch A2] TFT 입력용 / 출력 합산용 분리
        #
        #   TFT 입력 (var_baseline_logret known_real) :
        #     baseline_mode='deterministic' → 모든 s 동일 (det_seq)
        #       → 학습 시 본 in-sample fitted 와 통계 의미 일치(조건부 평균).
        #       → TFT 잔차는 "VAR 평균이 못 잡는 비선형 성분" 으로 해석 보존.
        #     baseline_mode='per_scenario' → 시나리오별 경로(ablation 비교용).
        #
        #   최종 hybrid 합산 baseline (output) :
        #     항상 per-scenario VAR 경로의 log-return 을 사용.
        #     → M3 bootstrap 이 만든 N개 충격 다양성이 출력에 그대로 살아남.
        #     → 최종식 : hybrid[s,t] = var_path_s_logret[t] + TFT_residual[s,t]
        #               = (det_mean[t] + shock_s[t]) + TFT_residual[s,t]
        per_scen_baseline_seq = var_baseline_logret(
            target_paths[s][None, :], last_price
        )[0]
        if baseline_mode == "deterministic":
            tft_input_baseline_seq = det_seq
        else:   # per_scenario — 입력/출력 모두 시나리오 경로 (legacy)
            tft_input_baseline_seq = per_scen_baseline_seq

        baseline_map_s = {d: float(v) for d, v in zip(fut_dates, tft_input_baseline_seq)}
        baseline[s] = per_scen_baseline_seq   # 출력 합산용을 기록(시나리오 다양성 보존)

        # 2) future_rows : 단일 동 + scenario s 의 feature 경로
        future_rows = []
        for offset in range(1, horizon + 1):
            chunk = last_snap.copy()
            new_ts = last_obs_idx + pd.DateOffset(months=offset)
            chunk[TIME_COL] = new_ts
            step_idx = offset - 1
            for j, col in enumerate(feature_cols):
                if col in chunk.columns and col != TARGET_LEVEL_COL:
                    chunk[col] = float(restored[s, step_idx, j])
            chunk[TARGET_COL] = 0.0
            future_rows.append(chunk)
        panel_ext_s = pd.concat([panel_dong_logret] + future_rows,
                                ignore_index=True)

        # 3) baseline_full_s : 과거 0, 미래 scenario s
        baseline_full_s = {**baseline_past, **baseline_map_s}
        panel_aug_s = attach_var_baseline(panel_ext_s, baseline_full_s)
        df_s = prepare_for_tft(panel_aug_s)

        predict_ds_s = TimeSeriesDataSet.from_dataset(
            training, df_s, predict=True, stop_randomization=True
        )
        pred_loader = predict_ds_s.to_dataloader(
            train=False, batch_size=8, num_workers=0
        )
        pred_out = model.predict(pred_loader, mode="raw", return_x=True)
        idx_to_ts = (df_s[["time_idx", TIME_COL]].drop_duplicates()
                                                  .set_index("time_idx")[TIME_COL]
                                                  .to_dict())
        # [Fix] group_ids 의 정수 label 은 학습 시 부여된 값이라
        # df_s 만으로 매핑하면 어긋남. dataset.x_to_index 로 안전 디코딩.
        pred_df_s = _decode_predictions(pred_out, idx_to_ts,
                                        dataset=predict_ds_s)
        pred_df_s = pred_df_s.sort_values("timestamp").head(horizon)

        rp10 = pred_df_s["pred_p10"].values[:horizon]
        rp50 = pred_df_s["pred_p50"].values[:horizon]
        rp90 = pred_df_s["pred_p90"].values[:horizon]
        # 길이 보정 (혹시 짧으면 0 padding)
        if len(rp50) < horizon:
            pad = horizon - len(rp50)
            rp10 = np.concatenate([rp10, np.zeros(pad)])
            rp50 = np.concatenate([rp50, np.zeros(pad)])
            rp90 = np.concatenate([rp90, np.zeros(pad)])

        raw_p10[s] = rp10
        raw_p50[s] = rp50
        raw_p90[s] = rp90

        # --- hybrid 잔차 발산 제어 (knob: shrink / clip / decay / alpha) ---
        # 1) quantile shrink : p10/p90 을 p50 쪽으로 끌어당김
        sh = float(TFT_CFG.get("hybrid_quantile_shrink", 1.0))
        rp10_s = rp50 + sh * (rp10 - rp50)
        rp90_s = rp50 + sh * (rp90 - rp50)
        rp50_s = rp50.copy()

        # 2) residual clip : 월간 잔차 절대값 cap (log-return)
        cap = TFT_CFG.get("residual_clip_abs", None)
        if cap is not None:
            cap = float(cap)
            rp10_s = np.clip(rp10_s, -cap, cap)
            rp50_s = np.clip(rp50_s, -cap, cap)
            rp90_s = np.clip(rp90_s, -cap, cap)

        # 3) residual decay : 시간이 갈수록 잔차 영향 fade (mean-reversion)
        decay = float(TFT_CFG.get("residual_decay", 1.0))
        if decay != 1.0:
            t_axis = np.arange(len(rp50_s))
            dvec   = decay ** t_axis
            rp10_s = rp10_s * dvec
            rp50_s = rp50_s * dvec
            rp90_s = rp90_s * dvec

        # 4) hybrid alpha : 잔차 전체 magnitude 스케일
        a = float(TFT_CFG.get("hybrid_alpha", 1.0))
        rp10_s = a * rp10_s
        rp50_s = a * rp50_s
        rp90_s = a * rp90_s

        if TFT_CFG["residual_mode"]:
            # [Patch A2] 출력 합산은 항상 per-scenario VAR 경로 사용
            # → N개 시나리오 다양성(M3 shock) 출력에 반영.
            hyb_p10[s] = rp10_s + per_scen_baseline_seq
            hyb_p50[s] = rp50_s + per_scen_baseline_seq
            hyb_p90[s] = rp90_s + per_scen_baseline_seq
        else:
            hyb_p10[s] = rp10_s
            hyb_p50[s] = rp50_s
            hyb_p90[s] = rp90_s

        if (s + 1) % progress_every == 0 or s == N_scen - 1:
            print(f"  scenario {s+1}/{N_scen} 완료")

    # level 복원
    level_p10 = last_price * np.exp(np.cumsum(hyb_p10, axis=1))
    level_p50 = last_price * np.exp(np.cumsum(hyb_p50, axis=1))
    level_p90 = last_price * np.exp(np.cumsum(hyb_p90, axis=1))

    return {
        "tft_raw_p10": raw_p10, "tft_raw_p50": raw_p50, "tft_raw_p90": raw_p90,
        "hybrid_p10":  hyb_p10, "hybrid_p50":  hyb_p50, "hybrid_p90":  hyb_p90,
        "level_p10":   level_p10, "level_p50":  level_p50, "level_p90":  level_p90,
        "baseline":    baseline,
        "features":    restored,           # (N, H, F) 시나리오별 feature 경로
        "feature_cols": list(feature_cols), # F 개 컴럼명
    }


def _save_n_hybrid_paths(pump_out: Dict[str, np.ndarray],
                          fut_dates,
                          save_path: Path,
                          dong_name: str) -> None:
    """N개 시나리오의 hybrid logret + level + feature 경로를 long-form CSV 로 저장.

    명세 컬럼:
      타겟 관련 : tft_raw_p50, var_baseline, hybrid_logret_p10/50/90, level_p10/50/90
      feature 경로 : feature_cols 의 각 컬럼 그대로 (scenario별 VAR 복원 값)
    이로써 “TFT 잔차보정된 시나리오별 target+feature 최종 결과물”을 단일 CSV로 제공.
    """
    N, H = pump_out["level_p50"].shape
    features = pump_out.get("features")        # (N, H, F) or None
    feat_cols = pump_out.get("feature_cols", [])
    rows = []
    dates_arr = np.asarray(fut_dates)
    for s in range(N):
        for t in range(H):
            row = {
                "scenario_id": s,
                "step":        t + 1,
                "timestamp":   dates_arr[t],
                "dong":        dong_name,
                "tft_raw_p50":     pump_out["tft_raw_p50"][s, t],
                "var_baseline":    pump_out["baseline"][s, t],
                "hybrid_logret_p10": pump_out["hybrid_p10"][s, t],
                "hybrid_logret_p50": pump_out["hybrid_p50"][s, t],
                "hybrid_logret_p90": pump_out["hybrid_p90"][s, t],
                "level_p10":   pump_out["level_p10"][s, t],
                "level_p50":   pump_out["level_p50"][s, t],
                "level_p90":   pump_out["level_p90"][s, t],
            }
            if features is not None and feat_cols:
                for j, col in enumerate(feat_cols):
                    row[col] = float(features[s, t, j])
            rows.append(row)
    pd.DataFrame(rows).to_csv(save_path, index=False)
    print(f"[N-hybrid] N={N} x H={H} (+{len(feat_cols)} features) → {save_path}")


# ===========================================================================
# 6. 백테스트 시나리오 (학습 ~ TRAIN_END / 검증 VALID_START ~ VALID_END)
# ===========================================================================
def scenario_backtest(panel_logret: pd.DataFrame) -> Dict:
    """학습: ~TRAIN_END / 검증: VALID_START ~ VALID_END  (Hybrid VAR+TFT).

    [신규 구조 — 단일 동 + N-시나리오 TFT pump]
      1) TARGET_DONG 만 추출 → M1~M5 (BVAR N=VAR_CFG.n_scenarios 시나리오)
      2) TFT 1회 학습 (단일 동 history, baseline=0)
      3) N 개 시나리오 각각에 대해 TFT forward pass
         → N 개 hybrid (TFT 잔차 + VAR baseline) 경로 산출
      4) median across scenarios 로 metric 계산, N 경로 전부 CSV 저장
    """
    # --- 0) panel → 단일 동 ---
    panel_dong = panel_logret[panel_logret["dong"] == TARGET_DONG].copy()
    if panel_dong.empty:
        raise ValueError(f"TARGET_DONG '{TARGET_DONG}' 데이터 없음")
    dong_panel = select_dong(panel_logret.drop(columns=[TARGET_COL]),
                              TARGET_DONG)
    train_end = pd.Timestamp(TRAIN_END)
    train_dong = dong_panel.loc[:train_end]
    horizon = TFT_CFG["backtest_pred_len"]

    feature_cols = [c for c in train_dong.columns]
    policy_cols  = [c for c in feature_cols if c.startswith("policy__")]

    # --- M0) 단일 동 패널 저장 (VAR 입력 raw) ---
    train_dong.to_csv(OUT_DIR / "backtest_M0_dong_panel.csv")
    print(f"[Backtest][M0] dong='{TARGET_DONG}' 패널 저장: shape={train_dong.shape}")

    # --- 1) M1 정상화 ---
    stat_df, tlog = m1_stationarize(train_dong.reset_index(),
                                    feature_cols, policy_cols)
    stat_df.to_csv(OUT_DIR / "backtest_M1_stationary.csv", index=False)
    _save_transform_log(tlog, OUT_DIR / "backtest_M1_transform_log.csv")

    # --- 2) M2 BVAR ---
    var_result = m2_bayesian_var(stat_df, feature_cols)
    print(f"[Backtest][M2] lag={var_result['lag']}  "
          f"n_feat={len(feature_cols)}  resid_shape={var_result['resid'].shape}")
    _save_var_result_csvs(var_result, feature_cols, OUT_DIR, prefix="backtest_M2")

    # --- 3) M3 시나리오 + M3+ 잔차주머니 ---
    scen_diff, N_used, _pool = m3_block_bootstrap(
        stat_df, var_result, feature_cols, TARGET_LEVEL_COL, horizon
    )
    _save_residual_pool(_pool, feature_cols,
                        OUT_DIR / "backtest_M3plus_residual_pool.csv")
    fut_dates_bk = pd.date_range(VALID_START, periods=horizon, freq="MS")
    _save_scenarios_long(scen_diff, fut_dates_bk, feature_cols,
                          OUT_DIR / "backtest_M3_scenarios_diff.csv",
                          space_tag="diff")

    # --- 4) M5 복원 + M4 현실성 ---
    m5_out = m5_restore_and_validate(scen_diff, tlog, feature_cols,
                                     TARGET_LEVEL_COL)
    _save_scenarios_long(m5_out["restored"], fut_dates_bk, feature_cols,
                          OUT_DIR / "backtest_M5_scenarios_level.csv",
                          space_tag="level")
    _save_target_paths_long(m5_out["target_paths"], fut_dates_bk,
                             OUT_DIR / "backtest_M5_target_paths.csv",
                             restored=m5_out["restored"],
                             feature_cols=feature_cols,
                             target_col=TARGET_LEVEL_COL)

    _, realism = m4_realism_check(m5_out["restored"], train_dong.reset_index(),
                                  var_result, feature_cols, TARGET_LEVEL_COL)
    print("[Backtest][M4] violation_rate=", realism["violation_rate"])
    _save_realism_report(realism, OUT_DIR / "backtest_M4_realism.csv")

    # VAR 시나리오 밴드 (참고용)
    last_price = float(train_dong[TARGET_LEVEL_COL].iloc[-1])
    _save_var_scenario_bands(m5_out["target_paths"], fut_dates_bk,
                             OUT_DIR / "backtest_var_scenario_bands.csv",
                             scenario_tag="backtest")

    # --- 5) TFT 1회 학습 ([Patch B] full panel pooling + In-Sample VAR baseline) ---
    #   과거 학습 구간에 VAR fitted log-return 을 baseline 으로 부여 →
    #   residual_logret = target_logret - fitted_logret 가 진정한 잔차.
    #   (baseline=0 로 두면 추론 시 시나리오 baseline 합산으로 double counting)
    #   [Patch B] 학습은 panel_logret 전체(243개 동)로 → static 임베딩(gu,dong)
    #            이 비로소 의미 가짐. VAR baseline 은 timestamp 키로 모든 동에
    #            broadcast 되어 cross-section 잔차 학습.
    baseline_full_train = var_in_sample_baseline_map(
        var_result, stat_df, tlog, panel_dong, feature_cols
    )
    panel_aug_train = attach_var_baseline(panel_logret, baseline_full_train)
    tft_target = "residual_logret" if TFT_CFG["residual_mode"] else TARGET_COL

    df_train = prepare_for_tft(panel_aug_train)
    train_end_idx = int(df_train.loc[df_train[TIME_COL] == train_end,
                                      "time_idx"].iloc[0])
    training = build_training_dataset(df_train, training_cutoff=train_end_idx,
                                      max_pred_len=horizon, target=tft_target)
    validation = TimeSeriesDataSet.from_dataset(
        training, df_train, predict=True, stop_randomization=True
    )
    n_dongs_train = df_train["dong"].nunique()
    print(f"\n[Backtest] training TFT  (target={tft_target}, "
          f"panel pooling: {n_dongs_train} dongs, inference dong={TARGET_DONG}) ...")
    model, _ = train_tft(training, validation, log_subdir="backtest")

    # VSN 가중치 저장 (학습 모델 기준 — predict 1회 후 추출)
    val_loader = validation.to_dataloader(
        train=False, batch_size=8, num_workers=TFT_CFG["num_workers"]
    )
    _vsn_pred = model.predict(val_loader, mode="raw", return_x=True)
    _save_vsn_weights(model, _vsn_pred, training,
                      OUT_DIR / "backtest_vsn_weights.csv")
    del _vsn_pred

    # --- 6) N 개 시나리오 forward pass ---
    #   [Patch A] deterministic baseline : 학습 시 사용한 in-sample fitted 와
    #            동일한 통계적 의미(VAR 조건부 평균)를 갖는 미래 경로를 산출.
    #            모든 시나리오에 동일 baseline 을 주입 → TFT 잔차의 본래 의미
    #            ("VAR 평균이 못 잡는 비선형/급변") 유지.
    det_baseline_seq = var_deterministic_future_logret(
        var_result=var_result, stat_df=stat_df, tlog=tlog,
        last_price=last_price, horizon=horizon,
        feature_cols=feature_cols, target_col=TARGET_LEVEL_COL,
    )
    pump_out = _tft_predict_per_scenario(
        model=model, training=training,
        panel_dong_logret=panel_logret,
        restored=m5_out["restored"],
        target_paths=m5_out["target_paths"],
        feature_cols=feature_cols,
        fut_dates=fut_dates_bk,
        last_obs_idx=train_end,
        last_price=last_price,
        horizon=horizon,
        baseline_past_map=baseline_full_train,
        deterministic_baseline_seq=det_baseline_seq,
        target_dong=TARGET_DONG,
    )
    _save_n_hybrid_paths(pump_out, fut_dates_bk,
                          OUT_DIR / "backtest_hybrid_scenarios.csv",
                          TARGET_DONG)

    # --- 7) median across scenarios → 비교/metric ---
    level_p10_med = np.median(pump_out["level_p10"], axis=0)
    level_p50_med = np.median(pump_out["level_p50"], axis=0)
    level_p90_med = np.median(pump_out["level_p90"], axis=0)

    actual_dong = (panel_dong[[TIME_COL, TARGET_LEVEL_COL]]
                   .rename(columns={TIME_COL: "timestamp",
                                    TARGET_LEVEL_COL: "actual"}))
    timeline = pd.DataFrame({
        "timestamp":    fut_dates_bk,
        "forecast":     level_p50_med,
        "forecast_p10": level_p10_med,
        "forecast_p90": level_p90_med,
    }).merge(actual_dong, on="timestamp", how="left")
    timeline["dong"] = TARGET_DONG
    timeline.to_csv(OUT_DIR / "backtest_dong_timeline.csv", index=False)

    tl_v = timeline.dropna(subset=["actual"])
    if len(tl_v) > 0:
        mae  = mean_absolute_error(tl_v["actual"], tl_v["forecast"])
        rmse = float(np.sqrt(mean_squared_error(tl_v["actual"],
                                                tl_v["forecast"])))
        mape = float(np.mean(np.abs(
            (tl_v["actual"] - tl_v["forecast"]) / tl_v["actual"]
        )) * 100)
        r2   = r2_score(tl_v["actual"], tl_v["forecast"])
    else:
        mae = rmse = mape = r2 = float("nan")

    metrics = pd.DataFrame([{
        "MAE": mae, "RMSE": rmse, "MAPE(%)": mape, "R2": r2,
        "n_valid_months":   len(tl_v),
        "dong":             TARGET_DONG,
        "VAR_N_scenarios":  N_used,
        "residual_mode":    TFT_CFG["residual_mode"],
        "M4_violation_ANY": realism["violation_rate"]["ANY"],
    }])
    metrics.to_csv(OUT_DIR / "backtest_metrics.csv", index=False)
    print("[Backtest metrics]\n", metrics.to_string(index=False))

    # --- 8) 차트 ---
    full_hist = (panel_dong.sort_values(TIME_COL)
                            .set_index(TIME_COL)[TARGET_LEVEL_COL])
    plt.figure(figsize=(12, 5))
    plt.plot(full_hist.index, full_hist.values,
             label="Actual", color="#1f77b4")
    # 모든 시나리오 경로 (옅게)
    for s in range(min(50, pump_out["level_p50"].shape[0])):
        plt.plot(fut_dates_bk, pump_out["level_p50"][s],
                 color="#d62728", alpha=0.05, linewidth=0.6)
    plt.plot(timeline["timestamp"], timeline["forecast"],
             label="Hybrid median", color="#d62728", linewidth=1.6)
    plt.fill_between(timeline["timestamp"],
                     timeline["forecast_p10"], timeline["forecast_p90"],
                     color="#d62728", alpha=0.15, label="per-scenario 80%")
    plt.axvline(train_end, color="gray", linestyle=":", alpha=0.7,
                label="Train End")
    plt.title(f"VAR-TFT Backtest: {TARGET_DONG} "
              f"(N={pump_out['level_p50'].shape[0]} scenarios)")
    plt.xlabel("Date"); plt.ylabel(TARGET_LEVEL_COL)
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(OUT_DIR / "backtest_chart.png", dpi=130)
    plt.close()

    return {"metrics": metrics, "var_N": N_used, "realism": realism}


# ===========================================================================
# 7. 미래 시나리오 (전체 학습 → 2026-01 ~ FUTURE_END)
# ===========================================================================
def scenario_future(panel_logret: pd.DataFrame) -> Dict:
    """전체 데이터 재학습 → 미래 10년 (FUTURE_PRED_LEN) Hybrid 예측.

    [신규 구조 — 단일 동 + N-시나리오 TFT pump]
      - 백테스트와 동일한 구조이나 horizon=future_pred_len, last_obs=panel max.
      - metric 대신 N 경로의 p10/p50/p90 분포만 저장.
    """
    panel_dong = panel_logret[panel_logret["dong"] == TARGET_DONG].copy()
    if panel_dong.empty:
        raise ValueError(f"TARGET_DONG '{TARGET_DONG}' 데이터 없음")
    dong_panel = select_dong(panel_logret.drop(columns=[TARGET_COL]),
                              TARGET_DONG)
    horizon = TFT_CFG["future_pred_len"]

    feature_cols = [c for c in dong_panel.columns]
    policy_cols  = [c for c in feature_cols if c.startswith("policy__")]

    # --- M0) 단일 동 패널 저장 ---
    dong_panel.to_csv(OUT_DIR / "future_M0_dong_panel.csv")
    print(f"[Future][M0] dong='{TARGET_DONG}' 패널 저장: shape={dong_panel.shape}")

    # M1 → M2 → M3
    stat_df, tlog = m1_stationarize(dong_panel.reset_index(),
                                    feature_cols, policy_cols)
    stat_df.to_csv(OUT_DIR / "future_M1_stationary.csv", index=False)
    _save_transform_log(tlog, OUT_DIR / "future_M1_transform_log.csv")

    var_result = m2_bayesian_var(stat_df, feature_cols)
    print(f"[Future][M2] lag={var_result['lag']}  "
          f"n_feat={len(feature_cols)}")
    _save_var_result_csvs(var_result, feature_cols, OUT_DIR, prefix="future_M2")

    scen_diff, N_used, _pool = m3_block_bootstrap(
        stat_df, var_result, feature_cols, TARGET_LEVEL_COL, horizon
    )
    _save_residual_pool(_pool, feature_cols,
                        OUT_DIR / "future_M3plus_residual_pool.csv")
    last_obs_date_f = dong_panel.index.max()
    fut_dates_f = pd.date_range(last_obs_date_f + pd.DateOffset(months=1),
                                 periods=horizon, freq="MS")
    _save_scenarios_long(scen_diff, fut_dates_f, feature_cols,
                          OUT_DIR / "future_M3_scenarios_diff.csv",
                          space_tag="diff")

    # M5 복원 + M4
    m5_out = m5_restore_and_validate(scen_diff, tlog, feature_cols,
                                     TARGET_LEVEL_COL)
    _save_scenarios_long(m5_out["restored"], fut_dates_f, feature_cols,
                          OUT_DIR / "future_M5_scenarios_level.csv",
                          space_tag="level")
    _save_target_paths_long(m5_out["target_paths"], fut_dates_f,
                             OUT_DIR / "future_M5_target_paths.csv",
                             restored=m5_out["restored"],
                             feature_cols=feature_cols,
                             target_col=TARGET_LEVEL_COL)

    _, realism = m4_realism_check(m5_out["restored"], dong_panel.reset_index(),
                                  var_result, feature_cols, TARGET_LEVEL_COL)
    print("[Future][M4] violation_rate=", realism["violation_rate"])
    _save_realism_report(realism, OUT_DIR / "future_M4_realism.csv")

    last_price = float(dong_panel[TARGET_LEVEL_COL].iloc[-1])
    _save_var_scenario_bands(m5_out["target_paths"], fut_dates_f,
                             OUT_DIR / "future_var_scenario_bands.csv",
                             scenario_tag="future")

    last_obs_idx = panel_dong[TIME_COL].max()

    # --- TFT 1회 학습 ([Patch B] full panel pooling + In-Sample VAR baseline) ---
    #   double counting 방지: 과거 구간 baseline = VAR fitted log-return.
    #   [Patch B] 학습은 panel_logret 전체로 → 동별 cross-section pooling.
    baseline_full_train = var_in_sample_baseline_map(
        var_result, stat_df, tlog, panel_dong, feature_cols
    )
    panel_aug_train = attach_var_baseline(panel_logret, baseline_full_train)
    tft_target = "residual_logret" if TFT_CFG["residual_mode"] else TARGET_COL

    df_train = prepare_for_tft(panel_aug_train)
    cutoff = int(df_train.loc[df_train[TIME_COL] == last_obs_idx,
                               "time_idx"].iloc[0])
    training = build_training_dataset(df_train, training_cutoff=cutoff,
                                      max_pred_len=horizon, target=tft_target)
    # [Fix] future 모드는 cutoff 이후 실데이터가 없어 predict=True 로 horizon(240)
    # decoder window 를 자를 수 없음 → AssertionError("filters should not remove
    # all entries"). validation 은 VSN 추출/EarlyStopping 용도이므로 짧은
    # in-sample window 로 슬라이싱해도 무방. 마지막 12 개월을 decoder 로 사용.
    _val_pred_len = min(TFT_CFG["backtest_pred_len"], 12)
    _val_cutoff = cutoff - _val_pred_len
    validation = TimeSeriesDataSet.from_dataset(
        training, df_train, predict=True, stop_randomization=True,
        max_prediction_length=_val_pred_len,
        min_prediction_length=1,
        min_prediction_idx=_val_cutoff + 1,
    )
    n_dongs_train = df_train["dong"].nunique()
    print(f"\n[Future] training TFT (target={tft_target}, horizon={horizon}, "
          f"panel pooling: {n_dongs_train} dongs, "
          f"inference dong={TARGET_DONG}) ...")
    model, _ = train_tft(training, validation, log_subdir="future")

    # VSN 가중치 저장
    val_loader = validation.to_dataloader(
        train=False, batch_size=8, num_workers=TFT_CFG["num_workers"]
    )
    _vsn_pred = model.predict(val_loader, mode="raw", return_x=True)
    _save_vsn_weights(model, _vsn_pred, training,
                      OUT_DIR / "future_vsn_weights.csv")
    del _vsn_pred

    # --- N 개 시나리오 forward pass ---
    #   [Patch A] deterministic baseline 주입 → 학습/추론 baseline 통계 의미 일치.
    det_baseline_seq = var_deterministic_future_logret(
        var_result=var_result, stat_df=stat_df, tlog=tlog,
        last_price=last_price, horizon=horizon,
        feature_cols=feature_cols, target_col=TARGET_LEVEL_COL,
    )
    pump_out = _tft_predict_per_scenario(
        model=model, training=training,
        panel_dong_logret=panel_logret,
        restored=m5_out["restored"],
        target_paths=m5_out["target_paths"],
        feature_cols=feature_cols,
        fut_dates=fut_dates_f,
        last_obs_idx=last_obs_idx,
        last_price=last_price,
        horizon=horizon,
        baseline_past_map=baseline_full_train,
        deterministic_baseline_seq=det_baseline_seq,
        target_dong=TARGET_DONG,
    )
    _save_n_hybrid_paths(pump_out, fut_dates_f,
                          OUT_DIR / "future_hybrid_scenarios.csv",
                          TARGET_DONG)

    # --- median across scenarios → dong_forecast ---
    level_p10_med = np.median(pump_out["level_p10"], axis=0)
    level_p50_med = np.median(pump_out["level_p50"], axis=0)
    level_p90_med = np.median(pump_out["level_p90"], axis=0)
    future_forecast = pd.DataFrame({
        "timestamp":  fut_dates_f,
        "dong":       TARGET_DONG,
        "pred_p10":   level_p10_med,
        "pred_p50":   level_p50_med,
        "pred_p90":   level_p90_med,
    })
    future_forecast.to_csv(OUT_DIR / "future_dong_forecast.csv", index=False)

    # --- 동 timeline (actual + forecast) ---
    full_hist = (panel_dong.sort_values(TIME_COL)
                 [[TIME_COL, TARGET_LEVEL_COL]]
                 .rename(columns={TIME_COL: "timestamp",
                                  TARGET_LEVEL_COL: "actual"}))
    timeline = pd.concat([
        full_hist.assign(type="actual", forecast=np.nan,
                         forecast_p10=np.nan, forecast_p90=np.nan),
        future_forecast.rename(columns={"pred_p50": "forecast",
                                         "pred_p10": "forecast_p10",
                                         "pred_p90": "forecast_p90"})
                        .assign(type="forecast", actual=np.nan)
                        .drop(columns=["dong"]),
    ], ignore_index=True).sort_values("timestamp")
    timeline.to_csv(OUT_DIR / "future_dong_timeline.csv", index=False)

    # M5 군집 (VAR 시나리오 기준) 저장
    cluster_rows = []
    for name, band in m5_out["clusters"].items():
        for t, (p10, p50, p90) in enumerate(zip(band["p10"], band["p50"],
                                                band["p90"])):
            cluster_rows.append({
                "cluster":   name,
                "timestamp": fut_dates_f[t],
                "var_p10":   p10, "var_p50": p50, "var_p90": p90,
            })
    pd.DataFrame(cluster_rows).to_csv(OUT_DIR / "future_var_clusters.csv",
                                       index=False)

    plt.figure(figsize=(13, 5))
    plt.plot(full_hist["timestamp"], full_hist["actual"],
             label="Actual", color="#1f77b4")
    for s in range(min(50, pump_out["level_p50"].shape[0])):
        plt.plot(fut_dates_f, pump_out["level_p50"][s],
                 color="#2ca02c", alpha=0.05, linewidth=0.6)
    plt.plot(future_forecast["timestamp"], future_forecast["pred_p50"],
             label="Hybrid median", color="#2ca02c", linewidth=1.6)
    plt.fill_between(future_forecast["timestamp"],
                     future_forecast["pred_p10"], future_forecast["pred_p90"],
                     color="#2ca02c", alpha=0.15, label="per-scenario 80%")
    seoul_var_med = (m5_out["target_paths"].mean(axis=0))
    plt.plot(fut_dates_f, seoul_var_med, color="#9467bd", linestyle=":",
             alpha=0.7, label="VAR baseline (mean)")
    plt.axvline(pd.Timestamp(last_obs_idx), color="gray", linestyle=":",
                alpha=0.7, label="Forecast Start")
    plt.title(f"VAR-TFT 10-Year Forecast: {TARGET_DONG} "
              f"(N={pump_out['level_p50'].shape[0]} scenarios)")
    plt.xlabel("Date"); plt.ylabel(TARGET_LEVEL_COL)
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(OUT_DIR / "future_chart.png", dpi=130)
    plt.close()

    return {"var_N": N_used, "realism": realism,
            "clusters_csv": OUT_DIR / "future_var_clusters.csv"}


# ===========================================================================
# 8. hyperparameters dump (재현성)
# ===========================================================================
def dump_hyperparameters() -> None:
    """현재 사용된 모든 CONFIG 값을 csv 로 저장."""
    rows = []
    for k, v in {**VAR_CFG, **{f"TFT__{kk}": vv for kk, vv in TFT_CFG.items()}}.items():
        rows.append({"param": k, "value": str(v)})
    pd.DataFrame(rows).to_csv(OUT_DIR / "hyperparameters.csv", index=False)


# ===========================================================================
# 8-1. 모듈별 기법 레지스트리 로그
# ===========================================================================
def log_technique_registry(save_path: Path) -> None:
    """M1~M5 + TFT 각 단계에서 사용한 기법을 CSV 로 기록.

    컬럼 설명:
      module     : 파이프라인 단계 (M1 / M2 / M3 / M3+ / M4 / M5 / TFT)
      sub_step   : 해당 단계 내 세부 동작
      technique  : 기법 공식 명칭 (영문)
      key_params : 현재 CONFIG 에서 사용 중인 파라미터값
      formula    : 수식 / 알고리즘 개요
      rationale  : 이 기법을 선택한 이유 (Korean)
    """
    lam  = VAR_CFG["minnesota_lambda"]
    bs   = VAR_CFG["block_size"]
    tdf  = VAR_CFG["t_df_init"]
    tp   = VAR_CFG["tail_prob"]
    km   = VAR_CFG["extreme_k_min"]
    conf = VAR_CFG["confidence"]
    eq   = VAR_CFG["extreme_quantile"]
    rq   = VAR_CFG["realism_extreme_q"]
    hl   = TFT_CFG["recency_halflife_months"]
    enc  = TFT_CFG["max_encoder_length"]
    qs   = TFT_CFG["quantiles"]
    hs   = TFT_CFG["hidden_size"]
    hcs  = TFT_CFG["hidden_continuous_size"]
    ah   = TFT_CFG["attention_head_size"]

    rows = [
        # ===== M1 =====
        dict(module="M1", sub_step="단위근 검정",
             technique="Augmented Dickey-Fuller (ADF)",
             key_params=f"alpha={VAR_CFG['adf_alpha']} | autolag=AIC",
             formula="H0: unit root; p >= alpha → 비정상 → 차분",
             rationale="비정상 시계열은 VAR 계수가 spurious(허구)해짐; 차분 트리거를 자동으로 결정"),
        dict(module="M1", sub_step="1차 차분",
             technique="First-order differencing",
             key_params="(ADF p >= alpha 인 변수만 자동 적용)",
             formula="diff(x)[t] = x[t] - x[t-1]",
             rationale="우상향 trend 제거 → VAR 정상성 가정 충족; transform_log 에 기록해 M5 역변환"),
        dict(module="M1", sub_step="이상치 클리핑",
             technique="Winsorization",
             key_params=f"lower={VAR_CFG['winsor_lower']} | upper={VAR_CFG['winsor_upper']}",
             formula="x_clip = clip(x, quantile(x, lower), quantile(x, upper))",
             rationale="극단 outlier 가 VAR 계수 추정을 오염시키는 것 방지; 경제 가정 아닌 보수성 노브"),
        dict(module="M1", sub_step="결측 보간",
             technique="Time-axis linear interpolation",
             key_params="method=time | limit_direction=both",
             formula="x_t = x_a + (t-a)/(b-a)*(x_b - x_a)",
             rationale="월 단위 등간격 패널; 앞뒤 관측값 선형 연결로 결측 채움, cross-dong 누설 방지"),
        # ===== M2 =====
        dict(module="M2", sub_step="차수 선택",
             technique="BIC-based lag selection",
             key_params=f"max_lag={VAR_CFG['max_lag']}",
             formula="BIC = -2*logL + k*log(n); argmin over lag in [1, max_lag]",
             rationale="과적합 방지; 탐색 상한을 12로 넓혀 계절성 패턴 흡수 여유 확보"),
        dict(module="M2", sub_step="VAR 계수 추정",
             technique="OLS per-equation VAR (statsmodels)",
             key_params="(statsmodels VAR.fit, lag=BIC 선택값)",
             formula="Y_t = intercept + A_1*Y_{t-1} + ... + A_p*Y_{t-p} + u_t",
             rationale="다변량 시계열 선형 영향 관계를 방정식 단위 OLS로 추정; 잔차 u_t → M3 재료"),
        dict(module="M2", sub_step="사전분포 압축",
             technique=f"Minnesota prior shrinkage (lambda={lam})",
             key_params=f"minnesota_lambda={lam}",
             formula=f"coef_off_diag(L) *= (1 - {lam}/(L+1))  [L=0..lag-1]",
             rationale=f"비대각(타변수) + 먼 시차일수록 0 쪽으로 당김; lambda={lam}은 BVAR 문헌 표준 중간값"),
        # ===== M3 =====
        dict(module="M3", sub_step="충격 샘플링",
             technique=f"Block bootstrap (block_size={bs})",
             key_params=f"block_size={bs} | seed={VAR_CFG['seed']}",
             formula="shock_block = resid[rand_start : rand_start + block_size]",
             rationale=f"블록 크기 {bs}=분기; 한 분기 동안 충격의 시간상관 보존, i.i.d. 가정 완화"),
        dict(module="M3", sub_step="시나리오 전개",
             technique="VAR multi-step simulation",
             key_params=f"horizon={VAR_CFG['horizon']} | N >= hard_floor_N",
             formula="Y_{t+1} = intercept + sum_L A_L*Y_{t-L+1} + shock_block[k]",
             rationale="VAR 계수를 horizon 개월 반복 적용해 N개 경로 생성; M3+ 가 N을 하드조건으로 역산"),
        # ===== M3+ =====
        dict(module="M3+", sub_step="꼬리 보강",
             technique=f"Student-t tail augmentation (df_init={tdf})",
             key_params=f"t_df_init={tdf} | tail_prob={tp} | fat_tail={VAR_CFG['fat_tail']}",
             formula=f"P(t-tail block) = {tp}; resid_j ~ t(df_j, loc_j, scale_j) MLE 적합",
             rationale=f"부동산 잔차 kurtosis>3 → Gaussian 꼬리 과소; df={tdf} t분포로 극단 충격 보강"),
        dict(module="M3+", sub_step="N 하드조건 역산",
             technique="Binomial CDF inversion for N",
             key_params=f"k_min={km} | confidence={conf} | extreme_quantile={eq}",
             formula=f"min N s.t. P(X >= {km}) >= {conf}  where X ~ Binomial(N, p_rare)",
             rationale=f"직관 N 대신 '극단 충격을 {conf} 확률로 {km}번 이상 뽑는 N'을 이항분포로 역산"),
        # ===== M4 =====
        dict(module="M4", sub_step="정의역 검사 [T1]",
             technique="Hard domain constraint",
             key_params="HARD_DOMAIN dict (변수 정의 상수)",
             formula="lo <= x_t <= hi  (예: 금리 >= 0, 수급지수 in [0,200])",
             rationale="물리적·정의상 불가한 값 검출; '경제 가정'이 아닌 '변수 정의' → 상수로 두어도 정당"),
        dict(module="M4", sub_step="급변 검사 [T2]",
             technique="Extreme change rate filter",
             key_params=f"realism_extreme_q={rq}",
             formula=f"|delta_x_t| > quantile(|delta_x_obs|, {rq}) → 위반 플래그",
             rationale=f"과거 월변화 상위 {round((1-rq)*100,1)}% 초과 급변을 비현실 판정; 기준을 데이터에서 자동산출"),
        dict(module="M4", sub_step="부호 일관성 [T3]",
             technique="VAR residual sign consistency",
             key_params=f"corr_threshold={VAR_CFG['realism_corr_threshold']} | pair_min_corr={VAR_CFG['realism_pair_min_corr']}",
             formula="sign(rowcorr(delta_a, delta_b)) != sign(M2 resid corr) & |corr|>thresh → 위반",
             rationale="M2가 추정한 변수쌍 인과 방향이 시나리오에서 어긋남 검출; 기준이 추정 결과 → 하드코딩 아님"),
        dict(module="M4", sub_step="리포트 모드",
             technique="Non-destructive flagging (report-only default)",
             key_params=f"realism_mode={VAR_CFG['realism_mode']}",
             formula="(기본) 경로 제거 없이 위반율 dict 만 반환",
             rationale="경로를 과도하게 버리면 시나리오 확률 비중 왜곡 (급락만 제거 → '급락없음' 편향)"),
        # ===== M5 =====
        dict(module="M5", sub_step="원단위 역변환",
             technique="Cumulative sum inversion",
             key_params="(transform_log 의 last_level + diff 여부)",
             formula="level[t] = last_level + sum_{i=1}^{t} diff[i]",
             rationale="M1 차분의 역연산; 마지막 실제값 기준으로 누적합해 실제 집값 단위 복원"),
        dict(module="M5", sub_step="검증 채점",
             technique="Coverage / Width / Rank scoring",
             key_params="(실제 미래값 존재 시 자동 계산)",
             formula="coverage=P(p10<=actual<=p90); width=mean(p90-p10); rank=mean(P(scen<actual))",
             rationale="포괄력(coverage) · 정보력(width) · 확률 정확도(rank) 3종 채점; 0.5 근처 rank가 이상적"),
        dict(module="M5", sub_step="군집화",
             technique="Tercile clustering (post-hoc)",
             key_params="하락/보합/상승 = 최종수익률 하위/중간/상위 1/3",
             formula="final_ret = level[-1]/level[0]-1; argsort → split into thirds",
             rationale="시나리오를 먼저 자유 생성 후 사후 분류 → 기존 사전정의 방식의 편향 제거"),
        # ===== TFT =====
        dict(module="TFT", sub_step="인코더",
             technique=f"LSTM encoder (length={enc})",
             key_params=f"max_encoder_length={enc} | hidden_size={hs}",
             formula="h_t = LSTM(embed(x_t), h_{t-1})",
             rationale=f"과거 {enc}개월을 hidden state로 압축; 2020 boom + 2022 crash + 2023 회복 전체 포함"),
        dict(module="TFT", sub_step="변수 선택 (VSN)",
             technique="Variable Selection Network (VSN)",
             key_params=f"hidden_continuous_size={hcs}",
             formula="alpha_j = softmax(GRN(x_j, c)); x_tilde = sum_j alpha_j * GRN_j(x_j)",
             rationale="시점마다 변수별 중요도를 soft-attention으로 동적 산출; backtest/future_vsn_weights.csv에 저장"),
        dict(module="TFT", sub_step="어텐션",
             technique=f"Multi-head self-attention ({ah} heads)",
             key_params=f"attention_head_size={ah} | hidden_size={hs}",
             formula="Attn(Q,K,V) = softmax(QK^T/sqrt(d_k))*V; 헤드 {ah}개 병렬 적용",
             rationale="decoder가 encoder의 관련 시점에 집중; head 수가 적을수록 해석 가능성 높음"),
        dict(module="TFT", sub_step="손실함수",
             technique="Quantile Loss (Pinball Loss)",
             key_params=f"quantiles={qs}",
             formula="QL(q,y,yhat) = q*max(y-yhat,0) + (1-q)*max(yhat-y,0)",
             rationale="점추정 대신 예측 구간(p10/p50/p90)을 직접 학습; 불확실성 정량화"),
        dict(module="TFT", sub_step="정규화",
             technique="GroupNormalizer (center=True)",
             key_params="groups=[dong] | transformation=None | center=True",
             formula="x_norm = (x - mean_dong) / scale_dong",
             rationale="동별 평균 제거로 NN 수렴 안정화; log-return은 scale 이미 작으므로 transformation=None"),
        dict(module="TFT", sub_step="학습 가중치",
             technique="Recency exponential weighting",
             key_params=f"recency_halflife_months={hl}",
             formula=f"w(t) = 0.5^((t_max - t) / {hl})",
             rationale=f"반감기 {hl}개월; 최근 regime 주세 학습 + 과거 데이터도 일부 반영"),
        dict(module="TFT", sub_step="잔차 보정",
             technique="Hybrid VAR+TFT residual boosting",
             key_params=f"residual_mode={TFT_CFG['residual_mode']}",
             formula="pred_final = VAR_baseline_logret + TFT(target_logret - VAR_baseline_logret)",
             rationale="VAR 이 못 잡는 동별 이질성·비선형·정책 충격을 TFT가 잔차로 학습; boosted ensemble"),
        dict(module="TFT", sub_step="옵티마이저",
             technique="AdamW + EarlyStopping",
             key_params=(f"lr={TFT_CFG['learning_rate']} | gradient_clip={TFT_CFG['gradient_clip_val']}"
                         f" | early_stop_patience={TFT_CFG['early_stop_patience']}"),
             formula="theta_{t+1} = theta_t - lr*(m_t/(sqrt(v_t)+eps) + lambda*theta_t)",
             rationale="원논문 TFT 표 4 기준; gradient clip=0.1으로 exploding gradient 방지"),
    ]

    cols = ["module", "sub_step", "technique", "key_params", "formula", "rationale"]
    pd.DataFrame(rows, columns=cols).to_csv(
        save_path, index=False, encoding="utf-8-sig"   # BOM 포함 → Excel 한글 깨짐 방지
    )
    print(f"[Technique log] 저장: {save_path}  ({len(rows)} entries)")


# ===========================================================================
# 9. main
# ===========================================================================
def main() -> None:
    pl.seed_everything(42, workers=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[Load] panel ...")
    raw = load_panel(DATA_CSV)
    raw.to_csv(OUT_DIR / "00_panel_raw.csv", index=False)

    print("[Clean] panel + log-return target ...")
    clean = clean_missing_and_policy(raw)
    clean.to_csv(OUT_DIR / "01_panel_clean.csv", index=False)

    print("[ADF] target stationarity report ...")
    run_adf_report(clean, OUT_DIR / "02_adf_report.csv")

    dump_hyperparameters()
    log_technique_registry(OUT_DIR / "technique_log.csv")

    print("\n=== Scenario 1+2 : Backtest  "
          f"(~{TRAIN_END} 학습, {VALID_START} ~ {VALID_END} 검증) ===")
    scenario_backtest(clean)

    print(f"\n=== Scenario 3 : Future  "
          f"(2026-01 ~ {FUTURE_END}) ===")
    scenario_future(clean)

    print(f"\n[Done] 산출물 저장 위치: {OUT_DIR}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[interrupted]")
    except Exception:
        import traceback
        tb = traceback.format_exc()
        print("\n[ERROR]\n", tb)
        try:
            (OUT_DIR / "_error_log.txt").write_text(tb, encoding="utf-8")
        except Exception:
            pass
        sys.exit(1)
