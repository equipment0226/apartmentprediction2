"""
Physics-Informed TFT (PI-TFT) : 거시경제 제약 손실 항을 결합한 서울 아파트 매매가 예측
=========================================================================

데이터  : data/dong_level_meta_table.csv (2010-01 ~ 2025-12, 월 단위, 243 동 × 25 컬럼)
타겟    : reb__apt_sale_avg_price (동 단위 패널)
저장 폴더: step3/  (TFT.py 와 동일 데이터/시나리오 → 결과만 별도 디렉터리)

시나리오 (TFT.py 와 동일 3종):
  (1) 2010~2024.12 학습 → 2025.1~2025.12 backtest (24m, single-shot)
  (2) 백테스트 메트릭 MAE/RMSE/MAPE/R² 저장
  (3) 전체 데이터 재학습 → 2026.1~2035.12 (120m) 미래 예측

---------------------------------------------------------------------------
[설계 차이 (vs step2/TFT.py)]  - 핵심 변경 *손실함수에 거시경제 제약 추가*
---------------------------------------------------------------------------

 step2/TFT.py 는 순수 QuantileLoss 만으로 학습 → 모델이 "과거 패널의 자기상관"
 만 학습하고 거시경제 인과(금리/CPI/실업/전세) 의 *방향성* 을 약하게 학습.
 결과: 실물경제 충격에 대한 반응이 약하고 횡보 편향이 강함.

 외부 보정안 (Diffusion 의 SCENARIO_ANNUAL_DRIFT_PCT / SCENARIO_PHASES 처럼
 window-별 drift 주입) 은 사실상 post-hoc trend 주입으로 모델링 관점에선 부정확.

 PI-TFT 의 해결책 : "Physics-Informed Loss" 패러다임 차용.
   total_loss  =  quantile_loss(pred, target)
                + λ · Σ_r  ReLU( -sign_r · Δpred_r )

   - r ∈ MACRO_RELATIONSHIPS = {(cause, expected_sign, weight, 근거), ...}
   - 학습 step 마다 encoder_cont 에서 cause 변수만 +PERTURBATION_SIGMA σ 섭동
     → forward 재계산 → Δpred = pred(perturbed) - pred(baseline) 의 부호가
     expected_sign 과 다르면 penalty (ReLU 로 한 방향만 패널라이즈).
   - 부호 일치 시 penalty=0 → 정확한 elasticity 강제 X (값 자유), *방향성만* 강제.

 [수학적 근거]
   부동산 가격 P 의 표준 reduced-form 거시 모델:
     P = f( DiscountRate(-), Income(+), Rent(+), Inflation(+), Wealth(+) )
   ∂f/∂r < 0,  ∂f/∂cpi > 0,  ∂f/∂jeonse > 0,  ∂f/∂unemp < 0
   → 이를 모델 sensitivity 의 *hard sign constraint* 로 부과 (Lagrangian-relaxed).

 [scenario_pipeline.py 와의 일관성]
   scenario_pipeline._evaluate_macro_constraint_mask 는 동일 부호 제약을
   *post-hoc filter* 로 사용. PI-TFT 는 같은 제약을 *학습 단계 손실* 로 흡수.
   → 학습된 모델 자체가 거시 인과를 존중 → post-hoc filter 가 거의 불요.

 [근거 자료]
   - Macro relationships : macro_constraint.EconomicRelationshipAnalyzer.
                            EXPECTED_RELATIONSHIPS (한국 부동산 reduced-form 실증)
   - Physics-informed NN : Raissi et al. (2019) JCP - PINN 의 boundary loss 패턴
   - Monotonic NN        : You et al. (2017) - sign-constrained gradient training
---------------------------------------------------------------------------

설계 차이 (vs VAR) :
  - VAR 은 광역(Seoul) 단일 다변량 시계열 1개 (T≈192) 였다.
  - TFT 는 cross-section(동) × 시간 패널을 그대로 학습 → 243 계열 × 학습창.
  - 정규화: target 은 GroupNormalizer(transformation=None, center=False).
  - policy__ 컬럼 : {-1,0,1} → {loose, keep, tight} categorical.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# [Windows 필수 설정 - 반드시 다른 import 보다 먼저]
#
# 1) KMP_DUPLICATE_LIB_OK : torch + numpy 가 각자 OpenMP 런타임을 로드할 때
#    충돌 방지. Windows Anaconda 환경에서 필수.
#
# 2) PYTHONUTF8=1 : Python I/O 전체를 UTF-8 로 강제.
#    PyTorch Lightning 이 시작 시 출력하는 "💡 Tip:" 메시지에 이모지가 포함되는데,
#    Windows 기본 콘솔(cp949)은 이모지를 인코딩할 수 없어 UnicodeEncodeError 가
#    발생 → 스크립트가 즉시 종료(cmd 창이 뜨고 바로 닫히는 증상의 원인).
#    PYTHONUTF8=1 로 설정하면 stdout/stderr 가 UTF-8 로 처리되어 이모지도 정상 출력.
#
# 3) PYTHONIOENCODING=utf-8:replace : 만약 콘솔이 완전히 UTF-8 을 지원 못하는
#    경우에도 인코딩 불가 문자를 '?' 로 대체하여 크래시 방지.
# ---------------------------------------------------------------------------
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8:replace")
os.environ.setdefault("PYTHONWARNINGS", "ignore")

# stdout/stderr 인코딩을 런타임에도 UTF-8:replace 로 재설정
# (환경변수보다 우선 적용되어 이미 열린 스트림도 안전하게 처리)
import sys, io
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

warnings.filterwarnings("ignore")
# pl.seed_everything 은 main() 안에서 호출 (모듈 레벨 호출 시 Windows 에서
# worker process 가 모듈을 재import 할 때 중복 실행되는 문제 방지)

# ---------------------------------------------------------------------------
# 경로 / 컬럼 정의
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent          # .../step3
ROOT_DIR = BASE_DIR.parent                          # .../newtrial
DATA_CSV = ROOT_DIR / "data" / "dong_level_meta_table.csv"
OUT_DIR  = BASE_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "reb__apt_sale_avg_price"
TIME_COL   = "timestamp"
ID_COLS    = ["si", "gu", "dong"]

# ---------------------------------------------------------------------------
# 시나리오 구간 
# ---------------------------------------------------------------------------
TRAIN_END   = "2022-12-01"
VALID_START = "2023-01-01"
VALID_END   = "2024-12-01"
FUTURE_END  = "2035-12-01"

# ---------------------------------------------------------------------------
# TFT 핵심 하이퍼파라미터 (한 곳에서 변경 가능)
# 각 값 옆에 선정 근거 주석.
# ---------------------------------------------------------------------------
MAX_ENCODER_LENGTH      = 60   # 인코더(과거) 길이 = 5년치.
                                # [변경 36→60] 2020 boom + 2022 crash + 2023 회복을
                                # 한 윈도우에 담아 regime 전환과 장기 trend를 학습.
                                # 36은 최근 1년 단기파형만 반복 학습되는 부작용 있음.

BACKTEST_PRED_LEN       = 24   # 백테스트 horizon = 24개월 (검증 구간 길이와 일치).
                                # 단일-shot 예측 → 누적 오차 없음.

FUTURE_PRED_LEN         = 120  # 미래 horizon = 120개월(10년).
                                # 학습 윈도/계열 = 192-36-120+1 = 37,
                                # × 243 dong ≈ 8,991 학습 샘플로 학습 가능 규모.

HIDDEN_SIZE             = 32   # TFT 핵심 hidden dim.
                                # [변경 16→32] encoder 5년으로 늘리며 trend 학습
                                # 부담 증가 → capacity 2배.

ATTENTION_HEAD_SIZE     = 2    # multi-head attention. hidden_size 가 작으므로 2.

DROPOUT                 = 0.1  # 과적합 방지. 논문 디폴트.

HIDDEN_CONTINUOUS_SIZE  = 8    # 연속형 변수 임베딩 dim. hidden_size 의 절반 권장.

LEARNING_RATE           = 5e-3 # AdamW.
                                # [변경 3e-2→5e-3] epoch 늘리며 LR 낮춰 수렴 안정.
                                # 3e-2는 trend 같은 느린 신호를 빠르게 덮어씌움.

BATCH_SIZE              = 128  # CPU 메모리 고려한 중간 값.

MAX_EPOCHS              = 40   # [변경 15→40] encoder 5년/패널 학습은 epoch 더 필요.
                                # 조기종료(patience=8)로 과적합 방지.

GRADIENT_CLIP_VAL       = 0.1  # 경험적 권장값(원논문 표 4).

EARLY_STOP_PATIENCE     = 8    # [변경 4→8] trend 학습은 느려서 patience 확대.

RECENCY_HALFLIFE_MONTHS = 48   # 최근 윈도우 손실 가중치 반감기(개월).
                                # weight = 0.5 ** ((max_t - t) / halflife)
                                #  - 48 (4년) : 2010-01 윤 ≈ 0.06, 2020-01 ≈ 0.42,
                                #    2024-12 ≈ 1.00. 2020+ regime 을 주세로 학습하면서
                                #    과거 메모리도 완전 제거하지 않는 절충값.
                                #  - 24 는 너무 슏워 최근 파형만 반복(Diffusion 은 horizon=12이라 OK,
                                #    TFT 는 encoder=60이므로 더 긴 halflife 적절).
                                #  - 모델 구조 불변, TimeSeriesDataSet.weight 로 native 처리.

NUM_WORKERS             = 0    # Windows: spawn overhead/freeze 회피.

QUANTILES               = [0.1, 0.5, 0.9]  # 중앙값 + 80% 예측구간.

ADF_ALPHA               = 0.05


# ---------------------------------------------------------------------------
# [Physics-Informed Loss : 거시경제 제약 설정]
#
# (cause_col, expected_sign, weight, 근거)
#   - cause_col      : panel 의 컬럼명 (encoder 의 unknown_real 로 들어감)
#   - expected_sign  : encoder 의 cause 값이 +ε 상승했을 때 모델 예측이 움직여야 할
#                      부호. +1 = pred 도 ↑, -1 = pred ↓
#   - weight         : 관계 강도(상대). 합 ≈ 4 정도로 정규화하여
#                      CONSTRAINT_WEIGHT 와 함께 main loss 의 5~10% 수준에 머무름.
#   - 근거 : macro_constraint.EXPECTED_RELATIONSHIPS + 한국 부동산 실증연구 인용.
#
# 데이터 컬럼 확인 정책:
#   _build_macro_indices() 가 self.hparams.x_reals 에서 실제 존재하는 컬럼만
#   선별하므로, 일부 컬럼이 panel 에 없어도 silently skip (런타임 안전).
# ---------------------------------------------------------------------------
MACRO_RELATIONSHIPS = [
    # 금리 (할인효과 + 자본조달비용) : 금리↑ → 가격↓
    #  - BOK 분석 (2022) : 기준금리 +1%p → 수도권 아파트 가격 -2~4% (12~18개월 lag).
    #  - macro_constraint EXPECTED_RELATIONSHIPS[("base_rate","sale")]=(-1,-0.3,...)
    ("ecos__base_rate",            -1.0, 1.0,
     "기준금리↑→매매가↓ (할인율 상승 + 주담대 부담)"),

    # 주담대 금리 (실수요 직접 부담) : 가산금리 포함
    #  - 한국은행 (2023) DSR 시뮬레이션 : 주담대 +1%p → 매수여력 -8~12%.
    #  - macro_constraint [("mortgage_rate","sale")]=(-1,-0.4,...)
    ("ecos__mortgage_rate_new",    -1.0, 1.0,
     "주담대금리↑→매매가↓ (실수요 매수여력 감소)"),

    # 주거 CPI (실물 인플레 + 임대료 push) : CPI↑ → 가격↑ (인플레 hedge)
    #  - 통계청 주거 CPI 와 매매가 장기 상관 +0.6 이상.
    #  - macro_constraint [("cpi_housing","sale")]=(+1,+0.8,...) — 가장 강한 +관계
    ("ecos__cpi_housing",          +1.0, 0.8,
     "주거CPI↑→매매가↑ (인플레 헤지 + 임대료 전가)"),

    # 실업률 (소득/심리) : 실업↑ → 가격↓
    #  - 1998 IMF/2008 GFC 시 실업률 +2%p → 매매가 -10~20% 동행.
    #  - macro_constraint [("unemployment","sale")]=(-1,-0.4,...)
    ("ecos__unemployment_rate",    -1.0, 0.5,
     "실업률↑→매매가↓ (소득 충격 + 매수심리 위축)"),

    # 전세가 (매매-전세 sticky relationship + 갭투자 채널) : 전세↑ → 매매↑
    #  - KB 주간동향 : 전세가율 70% 돌파 시 매매전환 수요 폭증 (2020-2021).
    #  - macro_constraint [("jeonse","sale")]=(+1,+0.6,...)
    ("reb__apt_jeonse_avg_price",  +1.0, 0.8,
     "전세가↑→매매가↑ (갭투자 + 매수전환 채널)"),
]

# Physics-informed loss 가중치 :  λ in  total = main + λ · constraint
#  - 0.05 : 작은 값. 주 목적은 *forecast accuracy* 이고, constraint 는 *방향성 보정*.
#  - 너무 크면(>0.5) 정확도 손상, 너무 작으면(<0.01) 학습 신호 미약.
#  - 5개 관계, weight 합 ≈ 4.1 → constraint 평균 ~0.05 (정규화 공간) 가정 시
#    λ=0.05 → loss 의 ~5~10% 기여, main forecast 와 균형.
PI_CONSTRAINT_WEIGHT    = 0.25

# encoder cause 변수 섭동 크기 (정규화 공간 σ). GroupNormalizer/EncoderNormalizer 통과 후
# encoder_cont 는 대략 N(0,1) 분포 → 0.5σ ≈ 25%ile shift (실측 가능한 충격).
#  - 0.5 : 의미있는 충격이면서 모델의 비선형성을 자극.
#  - 1.0 으로 키우면 학습 초기 발산 위험.
PI_PERTURBATION_SIGMA   = 0.5

# 매 batch constraint 계산 시 forward 가 1+R 번 호출됨 (R=관계 수, 본 설정 5 → 6배).
# 학습 속도가 크게 떨어지므로 매 N batch 마다만 적용 (mini-batch 단위 stochastic constraint).
#  - 4 : forward 비용 1.25배(평균) → 정확도/시간 균형.
#  - macro_constraint 가 batch-level correlation 이 아니라 *sample-level sign* 을
#    체크하므로 batch sampling 으로도 unbiased.
PI_APPLY_EVERY_N_BATCHES = 4


# ---------------------------------------------------------------------------
# Physics-Informed TFT : TemporalFusionTransformer + macro directional loss
# ---------------------------------------------------------------------------
class PhysicsInformedTFT(TemporalFusionTransformer):
    """TFT subclass with macroeconomic directional sensitivity constraint.

    training_step 에서 baseline forward 외에 각 macro cause 변수에 대해
    encoder_cont 의 해당 컬럼을 +PI_PERTURBATION_SIGMA σ 만큼 섭동한 forward 를
    1회씩 추가 수행. Δpred 의 부호가 expected_sign 과 다르면 ReLU penalty.

    [구현 주의]
      - forward 추가 호출 비용 : R+1 회 (R = 활성 관계 수).
        PI_APPLY_EVERY_N_BATCHES 로 stochastic 적용.
      - validation/predict 에서는 constraint 계산하지 않음 (속도 + 무의미).
      - self.hparams.x_reals 에서 cause 컬럼 인덱스 lazy 매핑.
      - encoder_cont 만 섭동 : decoder_cont 의 cause 미래값은 known_real 이
        아니므로 본 panel 에서는 비어 있어 의미없음.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # lazy-init dict {cause_col -> (encoder_cont 의 channel idx, sign, weight)}
        self._macro_idx_cache: dict | None = None
        self._batch_counter: int = 0

    def _build_macro_index_map(self) -> dict:
        """self.hparams.x_reals 의 순서를 따라 encoder_cont 채널 인덱스 매핑.

        x_reals 는 pytorch-forecasting 이 dataset 으로부터 생성한 continuous
        변수 이름 리스트(static_reals + time_varying_known_reals +
        time_varying_unknown_reals + target_scale 보조 등).  encoder_cont
        마지막 축의 채널 순서와 일치.
        """
        try:
            x_reals = list(self.hparams.x_reals)
        except Exception:
            # fallback : 일부 버전은 self.reals 사용
            x_reals = list(getattr(self, "reals", []))
        idx_map: dict = {}
        for cause, sign, weight, _rationale in MACRO_RELATIONSHIPS:
            if cause in x_reals:
                idx_map[cause] = (x_reals.index(cause), float(sign), float(weight))
        return idx_map

    def _macro_constraint_loss(self, x: dict, baseline_pred: torch.Tensor) -> torch.Tensor:
        """ Σ_r weight_r · mean( ReLU( -sign_r · Δpred_r ) )

        baseline_pred : (B, T, Q) — QuantileLoss 의 quantile prediction tensor.
                        median(q=0.5) 사용 → 단일 시나리오 path.
        Δpred         : 미래 horizon 평균 변화 (scalar per sample).
        """
        if not self._macro_idx_cache:
            return baseline_pred.new_tensor(0.0)

        # median quantile index
        try:
            q_idx = self.loss.quantiles.index(0.5)
        except (AttributeError, ValueError):
            q_idx = baseline_pred.shape[-1] // 2

        baseline_avg = baseline_pred[..., q_idx].mean(dim=-1)  # (B,)

        total_penalty = baseline_pred.new_tensor(0.0)
        encoder_cont = x.get("encoder_cont")
        if encoder_cont is None:
            return total_penalty

        for cause, (ch_idx, sign, weight) in self._macro_idx_cache.items():
            # encoder_cont 만 채널 섭동 (다른 입력은 공유, in-place 금지를 위해 clone)
            x_pert = {k: v for k, v in x.items()}
            ec_pert = encoder_cont.clone()
            ec_pert[..., ch_idx] = ec_pert[..., ch_idx] + PI_PERTURBATION_SIGMA
            x_pert["encoder_cont"] = ec_pert

            out_pert = self(x_pert)
            pert_pred = out_pert["prediction"]
            pert_avg = pert_pred[..., q_idx].mean(dim=-1)  # (B,)

            delta = pert_avg - baseline_avg          # (B,)
            wrong_sign_mag = torch.relu(-sign * delta)
            total_penalty = total_penalty + weight * wrong_sign_mag.mean()

        return total_penalty

    def training_step(self, batch, batch_idx):
        # baseline supervised loss : 부모의 표준 경로 사용 (log, metric 보존)
        main = super().training_step(batch, batch_idx)
        # PyTorch Lightning >=2.0 : training_step 은 scalar tensor 또는 dict 반환
        main_loss = main if torch.is_tensor(main) else main.get("loss")

        self._batch_counter += 1
        # skip batch : constraint 비활성 → 로그도 생략 (Lightning 은 동일 step 내
        # 같은 키를 다른 인자(prog_bar 등)로 두 번 호출하면 MisconfigurationException
        # 을 던지므로 skip path 에서는 self.log 호출 자체를 하지 않는다.
        # epoch 평균 metric 은 활성 batch 들의 평균으로 충분히 안정).
        if (self._batch_counter % PI_APPLY_EVERY_N_BATCHES) != 0:
            return main

        # constraint 활성 batch : baseline forward 재실행 (super 의 prediction 은 외부 접근 X)
        x, y = batch
        if self._macro_idx_cache is None:
            self._macro_idx_cache = self._build_macro_index_map()
        if not self._macro_idx_cache:
            return main

        with torch.set_grad_enabled(True):
            baseline_out = self(x)
            baseline_pred = baseline_out["prediction"]
            constraint = self._macro_constraint_loss(x, baseline_pred)

        total = main_loss + PI_CONSTRAINT_WEIGHT * constraint
        self.log("pi_constraint_loss", constraint.detach(),
                 prog_bar=False, on_step=False, on_epoch=True)
        self.log("pi_main_loss", main_loss.detach(),
                 prog_bar=False, on_step=False, on_epoch=True)
        if isinstance(main, dict):
            main["loss"] = total
            return main
        return total


# ---------------------------------------------------------------------------
# 1. 데이터 로딩
# ---------------------------------------------------------------------------
def load_panel(csv_path: Path) -> pd.DataFrame:
    """원본 동(洞) 단위 패널을 그대로 로드.
    [근거] TFT 는 group_ids 로 다중 시계열을 동시에 학습 → 패널 그대로 사용이
           정보 손실 없고 표본 수가 크다."""
    df = pd.read_csv(csv_path, parse_dates=[TIME_COL])
    df = df.sort_values(["dong", TIME_COL]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 2. 결측치 보간 + 정책 변수 정규화
# ---------------------------------------------------------------------------
def clean_missing_and_policy(df: pd.DataFrame) -> pd.DataFrame:
    """동별 시간축 선형 보간 + policy__ → {-1,0,1} 정수 강제.

    [파라미터 근거]
      - groupby("dong").interpolate(method="time") : 동마다 누락 패턴이 다를 수
        있으므로 동 내부에서만 보간 (cross-dong 누설 방지).
      - limit_direction="both" : 양 끝 NaN 까지 가장 가까운 관측치로 보강.
      - policy round/clip : 사용자 요구사항(강화+1/유지0/완화-1) 강제.
    """
    out = df.copy().set_index(TIME_COL)
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
    return out


# ---------------------------------------------------------------------------
# 3. ADF 검정 (정상성 진단 - 리포팅용)
# ---------------------------------------------------------------------------
def run_adf_report(df: pd.DataFrame, save_path: Path) -> pd.DataFrame:
    """동별 타겟 시리즈 ADF 검정 결과 csv 저장.

    [근거]
      - 사용자 요구 "ADF 검정 단계" 반영. TFT 는 정상성을 요구하지 않으나
        진단/보고용으로 수행.
      - regression="ct" : 상수+추세 (부동산 가격은 추세 존재).
      - autolag="AIC" : 최적 lag 자동 선택.
    """
    rows = []
    for dong, g in df.groupby("dong"):
        series = g[TARGET_COL].dropna().values
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


# ---------------------------------------------------------------------------
# 4. TFT 학습용 데이터프레임 가공
# ---------------------------------------------------------------------------
def prepare_for_tft(df: pd.DataFrame) -> pd.DataFrame:
    """pytorch-forecasting TimeSeriesDataSet 스키마로 변환.
       (정수 time_idx, str 카테고리, policy str 매핑)"""
    out = df.copy()
    base = out[TIME_COL].min()
    out["time_idx"] = ((out[TIME_COL].dt.year  - base.year ) * 12 +
                       (out[TIME_COL].dt.month - base.month)).astype(int)

    out["si"]   = out["si"].astype(str)
    out["gu"]   = out["gu"].astype(str)
    out["dong"] = out["dong"].astype(str)
    out["month"] = out[TIME_COL].dt.month.astype(str)

    mapping = {-1: "loose", 0: "keep", 1: "tight"}
    policy_cols = [c for c in out.columns if c.startswith("policy__")]
    for c in policy_cols:
        out[c] = out[c].map(mapping).astype(str)

    # ------------------------------------------------------------------
    # [Recency-weighted loss] 시점별 손실 가중치
    # weight(t) = 0.5 ** ((max_t - t) / halflife)
    #  - TimeSeriesDataSet 의 weight 인자가 이 컬럼을 읽어 decoder 손실에
    #    적용 → 2020+ 구간을 주세로 학습시키고 2010-2015 횡보구간의
    #    영향을 완화 → 하락편향 bias 교정.
    #  - 243개 동 모두 동일 시점에 같은 weight 적용 (cross-section 동등).
    # ------------------------------------------------------------------
    t_max = int(out["time_idx"].max())
    out["loss_weight"] = (0.5 ** ((t_max - out["time_idx"])
                                   / float(RECENCY_HALFLIFE_MONTHS))
                          ).astype(np.float32)
    return out


# 거시 known reals : 정책당국이 사전 공표/관측 가능한 변수만 decoder 에 미래값 전달.
#  - ecos__base_rate    : 기준금리 (BOK MPC 결정 ～ 다음 회의까지 known).
#  - ecos__cpi_housing  : 주거 CPI (통계청 월간 발표 ～ 1개월 lag known).
# 둘 다 BACKTEST 구간(2024-01~2025-12)에서는 panel 에 실측이 들어 있으므로
# decoder 에 자연스럽게 실측 macro 가 흘러들어가 모델이 거시 충격에 반응함.
# (기존엔 time_varying_unknown_reals 에 묶여 있어 decoder 에 절대 전달되지
#  않았고, 모델은 time_idx 만 보고 24개월 자율회귀 표류 ～ 방향 반대 발생.)
MACRO_KNOWN_REALS = ["ecos__base_rate", "ecos__cpi_housing"]


def column_roles(df: pd.DataFrame) -> dict:
    """TFT 컬럼 역할 매핑. 한 곳에서 검토/변경 가능."""
    policy_cols = [c for c in df.columns if c.startswith("policy__")]
    macro_known = [c for c in MACRO_KNOWN_REALS if c in df.columns]
    unknown_reals = [c for c in df.columns
                     if (c.startswith("ecos__") or c.startswith("reb__"))
                     and c not in macro_known]
    if TARGET_COL not in unknown_reals:
        unknown_reals.append(TARGET_COL)
    return {
        "static_categoricals": ["gu", "dong"],
        "static_reals": [],
        "time_varying_known_categoricals": ["month"] + policy_cols,
        "time_varying_known_reals": ["time_idx"] + macro_known,
        "time_varying_unknown_reals": unknown_reals,
    }


# ---------------------------------------------------------------------------
# 5. TimeSeriesDataSet 빌더
# ---------------------------------------------------------------------------
def build_training_dataset(df: pd.DataFrame,
                            training_cutoff: int,
                            max_pred_len: int) -> TimeSeriesDataSet:
    """학습용 TimeSeriesDataSet 생성.

    [정규화 근거]
      - 타겟: GroupNormalizer(groups=["dong"], transformation="softplus")
        · 동별 절대 가격 수준 차이가 매우 큼 → 동별 평균/스케일로 정규화하여
          모형이 상대 변동에 집중하게 함. softplus 는 양수 제약(가격>0).
      - unknown features: 기본 TorchNormalizer (윈도우/그룹 단위 표준화).
    """
    roles = column_roles(df)
    # [인코더 보정] 학습 구간에 없던 policy 카테고리가 검증/미래 구간에
    # 처음 등장하면 KeyError 발생. time-varying known categoricals 만
    # add_nan=True 로 미관측 허용. group_ids/static_categoricals(gu, dong) 는
    # 카테고리 전수가 패널에 고정되어 있어 변경 불요(_decode_predictions 의
    # group_map int↔dong 매핑이 알파벳 순서에 의존하므로 add_nan 추가 금지).
    cat_encoders = {c: NaNLabelEncoder(add_nan=True)
                    for c in roles["time_varying_known_categoricals"]}
    return TimeSeriesDataSet(
        df[df["time_idx"] <= training_cutoff],
        time_idx="time_idx",
        target=TARGET_COL,
        group_ids=["dong"],
        weight="loss_weight",                  # recency-weighted loss
        categorical_encoders=cat_encoders,
        max_encoder_length=MAX_ENCODER_LENGTH,
        max_prediction_length=max_pred_len,
        min_encoder_length=MAX_ENCODER_LENGTH // 2,
        min_prediction_length=1,
        static_categoricals=roles["static_categoricals"],
        static_reals=roles["static_reals"],
        time_varying_known_categoricals=roles["time_varying_known_categoricals"],
        time_varying_known_reals=roles["time_varying_known_reals"],
        time_varying_unknown_reals=roles["time_varying_unknown_reals"],
        # [정규화 변경] softplus → None, center=False
        #  - softplus 는 small value 왜곡 + 동별 mean 제거(center=True 기본)로
        #    장기 trend가 normalized space 에서 사라져 모델이 "잔차의 단기 파형"
        #    만 학습 → 장기 횡보 + 최근 1년 trend 반복 증상의 핵심 원인.
        #  - center=False : 동별 평균을 빼지 않음 → trend가 그대로 살아 학습됨.
        #  - transformation=None : 비선형 왜곡 제거(가격 양수 제약은 scale 만으로 충분).
        target_normalizer=GroupNormalizer(
            groups=["dong"], transformation=None, center=False
        ),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )


# ---------------------------------------------------------------------------
# 6. 모델 학습
# ---------------------------------------------------------------------------
def train_tft(training: TimeSeriesDataSet,
              validation: TimeSeriesDataSet,
              log_subdir: str) -> tuple[TemporalFusionTransformer, pl.Trainer]:
    """TFT 적합. QuantileLoss(10/50/90%) → 불확실성 구간까지 산출."""
    train_loader = training.to_dataloader(
        train=True, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
    )
    val_loader = validation.to_dataloader(
        train=False, batch_size=BATCH_SIZE * 2, num_workers=NUM_WORKERS
    )

    early_stop = EarlyStopping(
        monitor="val_loss", patience=EARLY_STOP_PATIENCE, mode="min"
    )
    logger = CSVLogger(save_dir=str(OUT_DIR / "lightning_logs"),
                       name=log_subdir)

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="cpu",
        gradient_clip_val=GRADIENT_CLIP_VAL,
        callbacks=[early_stop],
        logger=logger,
        enable_progress_bar=True,
        enable_model_summary=True,
        log_every_n_steps=10,
    )

    tft = PhysicsInformedTFT.from_dataset(
        training,
        learning_rate=LEARNING_RATE,
        hidden_size=HIDDEN_SIZE,
        attention_head_size=ATTENTION_HEAD_SIZE,
        dropout=DROPOUT,
        hidden_continuous_size=HIDDEN_CONTINUOUS_SIZE,
        loss=QuantileLoss(quantiles=QUANTILES),
        # log_interval=0 → 학습/검증 step 내부 interpretation 로깅 비활성화
        # (pytorch-forecasting 1.7 + lightning 2.6 조합에서 predict=True dataset 의
        #  긴 encoder 와 함께 쓰면 integer_histogram 차원 오류 발생.
        #  해석은 학습 후 save_interpretation() 에서 명시적으로 수행)
        log_interval=0,
        optimizer="adamw",
        reduce_on_plateau_patience=2,
    )

    trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)

    best_path = trainer.checkpoint_callback.best_model_path
    if best_path:
        # PhysicsInformedTFT 자체로 다시 load (subclass 보존)
        tft = PhysicsInformedTFT.load_from_checkpoint(best_path)
    return tft, trainer


# ---------------------------------------------------------------------------
# 7. 변수 중요도(VSN) / 해석 산출물
# ---------------------------------------------------------------------------
def save_interpretation(model: TemporalFusionTransformer,
                        dataset: TimeSeriesDataSet,
                        prefix: str) -> None:
    """VSN(Variable Selection Network) 변수 중요도와 attention 패턴을 csv 저장."""
    try:
        loader = dataset.to_dataloader(train=False, batch_size=BATCH_SIZE,
                                       num_workers=NUM_WORKERS)
        raw_pred = model.predict(loader, mode="raw", return_x=True)
        interp = model.interpret_output(raw_pred.output, reduction="sum")
        for key, val in interp.items():
            arr = val.detach().cpu().numpy()
            if arr.ndim == 1:
                pd.DataFrame({key: arr}).to_csv(
                    OUT_DIR / f"{prefix}_interp_{key}.csv", index=False)
            else:
                pd.DataFrame(arr).to_csv(
                    OUT_DIR / f"{prefix}_interp_{key}.csv", index=False)
        print(f"[Interp] {prefix} VSN/attention 저장 완료")
    except Exception as e:
        print(f"[Interp] 해석 산출 실패(무시): {e}")


# ---------------------------------------------------------------------------
# 공통 유틸 : raw 예측 → long-form DataFrame
# ---------------------------------------------------------------------------
def _decode_predictions(pred_out, training: TimeSeriesDataSet,
                        idx_to_ts: dict,
                        group_map: dict | None = None) -> pd.DataFrame:
    """group_map: {int_id -> dong_str} — required for pytorch-forecasting >= 1.7
    where categorical_encoders is None.  Build with:
        group_map = {i: d for i, d in enumerate(sorted(df["dong"].unique()))}
    (NaNLabelEncoder.fit uses np.unique, i.e. alphabetical sort for strings.)
    """
    raw, x = pred_out.output, pred_out.x
    pred_q = raw["prediction"].detach().cpu().numpy()
    q_idx_median = QUANTILES.index(0.5)
    pred_median = pred_q[:, :, q_idx_median]
    pred_low    = pred_q[:, :, 0]
    pred_high   = pred_q[:, :, -1]
    decoder_time_idx = x["decoder_time_idx"].detach().cpu().numpy()

    group_ids = x["groups"].detach().cpu().numpy()
    if group_map is not None:
        # Explicit map passed by caller (pytorch-forecasting 1.7+)
        dong_names = [group_map[int(g[0])] for g in group_ids]
    else:
        # Older pytorch-forecasting: categorical_encoders is a live dict
        try:
            dong_categories = training.categorical_encoders["dong"].classes_
        except TypeError:
            raise RuntimeError(
                "training.categorical_encoders is None (pytorch-forecasting >= 1.7). "
                "Pass group_map={{int: dong_str}} to _decode_predictions."
            )
        if isinstance(dong_categories, dict):
            inv = {v: k for k, v in dong_categories.items()}
            dong_names = [inv[int(g[0])] for g in group_ids]
        else:
            dong_names = [dong_categories[int(g[0])] for g in group_ids]

    records = []
    for i, dong in enumerate(dong_names):
        for k in range(pred_median.shape[1]):
            ti = int(decoder_time_idx[i, k])
            records.append({
                "dong": dong,
                "timestamp": idx_to_ts.get(ti),
                "pred_p10":  float(pred_low[i, k]),
                "pred_p50":  float(pred_median[i, k]),
                "pred_p90":  float(pred_high[i, k]),
            })
    return pd.DataFrame(records)


def _save_hyperparams(prefix: str, pred_len: int) -> None:
    pd.DataFrame({
        "param": ["MAX_ENCODER_LENGTH", "PRED_LEN", "HIDDEN_SIZE",
                  "ATTENTION_HEAD_SIZE", "DROPOUT",
                  "HIDDEN_CONTINUOUS_SIZE", "LR", "BATCH_SIZE",
                  "MAX_EPOCHS", "GRADIENT_CLIP_VAL",
                  "EARLY_STOP_PATIENCE", "QUANTILES"],
        "value": [MAX_ENCODER_LENGTH, pred_len, HIDDEN_SIZE,
                  ATTENTION_HEAD_SIZE, DROPOUT,
                  HIDDEN_CONTINUOUS_SIZE, LEARNING_RATE, BATCH_SIZE,
                  MAX_EPOCHS, GRADIENT_CLIP_VAL,
                  EARLY_STOP_PATIENCE, str(QUANTILES)],
    }).to_csv(OUT_DIR / f"{prefix}_hyperparameters.csv", index=False)


# ---------------------------------------------------------------------------
# 8. 시나리오 (1)+(2) : Backtest 2025-01 ~ 2025-12
# ---------------------------------------------------------------------------
def scenario_backtest(panel: pd.DataFrame) -> None:
    """학습: ~2024-12 / 검증: 2025-01~2025-12 (12 months single-shot)."""
    df = prepare_for_tft(panel)
    train_end_idx = int(
        df.loc[df[TIME_COL] == pd.Timestamp(TRAIN_END), "time_idx"].iloc[0]
    )

    training = build_training_dataset(df, training_cutoff=train_end_idx,
                                       max_pred_len=BACKTEST_PRED_LEN)
    validation = TimeSeriesDataSet.from_dataset(
        training, df, predict=True, stop_randomization=True
    )

    print("\n[Backtest] training TFT ...")
    model, _ = train_tft(training, validation, log_subdir="backtest")
    _save_hyperparams("backtest", BACKTEST_PRED_LEN)

    val_loader = validation.to_dataloader(train=False, batch_size=BATCH_SIZE * 2,
                                          num_workers=NUM_WORKERS)
    pred_out = model.predict(val_loader, mode="raw", return_x=True)

    idx_to_ts = (df[["time_idx", TIME_COL]].drop_duplicates()
                                              .set_index("time_idx")[TIME_COL]
                                              .to_dict())
    # NaNLabelEncoder assigns ints via np.unique (= alphabetical for strings)
    group_map = {i: d for i, d in enumerate(sorted(df["dong"].unique()))}
    pred_df = _decode_predictions(pred_out, training, idx_to_ts, group_map)

    actual_df = panel[[TIME_COL, "dong", TARGET_COL]].rename(
        columns={TIME_COL: "timestamp", TARGET_COL: "actual"}
    )
    compare = pred_df.merge(actual_df, on=["dong", "timestamp"], how="left")

    # ------------------------------------------------------------------
    # [Boundary anchor] 동별 첫 예측 시점이 마지막 실측과 이격되는 문제
    # (GroupNormalizer 의 평균회귀 + softplus 왜곡 잔재) 보정.
    # pred'[k] = pred[k] + (last_actual - pred[0]) * max(0, 1 - k/L)
    # L=12 → 12개월에 걸쳐 선형으로 이격 흡수 → 이후엔 모델 신호 그대로.
    # ------------------------------------------------------------------
    ANCHOR_LEN = 12
    last_actuals = (panel[panel[TIME_COL] <= pd.Timestamp(TRAIN_END)]
                    .sort_values(TIME_COL)
                    .groupby("dong")[TARGET_COL].last())
    compare = compare.sort_values(["dong", "timestamp"]).reset_index(drop=True)
    for dong, grp in compare.groupby("dong"):
        if dong not in last_actuals.index or len(grp) == 0:
            continue
        idx = grp.index.to_list()
        first_pred = compare.loc[idx[0], "pred_p50"]
        d = float(last_actuals[dong]) - float(first_pred)
        for k, ridx in enumerate(idx):
            w = max(0.0, 1.0 - k / float(ANCHOR_LEN))
            shift = d * w
            compare.loc[ridx, "pred_p10"] += shift
            compare.loc[ridx, "pred_p50"] += shift
            compare.loc[ridx, "pred_p90"] += shift

    compare["abs_error"] = (compare["actual"] - compare["pred_p50"]).abs()
    compare.to_csv(OUT_DIR / "backtest_dong_compare.csv", index=False)

    seoul = (compare.groupby("timestamp")
                    .agg(actual=("actual", "mean"),
                         forecast=("pred_p50", "mean"),
                         forecast_p10=("pred_p10", "mean"),
                         forecast_p90=("pred_p90", "mean"))
                    .reset_index())
    seoul.to_csv(OUT_DIR / "backtest_seoul_compare.csv", index=False)

    mae  = mean_absolute_error(seoul["actual"], seoul["forecast"])
    rmse = float(np.sqrt(mean_squared_error(seoul["actual"], seoul["forecast"])))
    mape = float(np.mean(np.abs((seoul["actual"] - seoul["forecast"])
                                 / seoul["actual"])) * 100)
    r2   = r2_score(seoul["actual"], seoul["forecast"])
    metrics = pd.DataFrame([{
        "MAE": mae, "RMSE": rmse, "MAPE(%)": mape, "R2": r2,
        "n_valid_months": len(seoul),
        "n_dong": compare["dong"].nunique(),
        "max_encoder_length": MAX_ENCODER_LENGTH,
        "pred_len": BACKTEST_PRED_LEN,
    }])
    metrics.to_csv(OUT_DIR / "backtest_metrics.csv", index=False)
    print("[Backtest metrics]\n", metrics.to_string(index=False))

    # 차트
    full_seoul = panel.groupby(TIME_COL)[TARGET_COL].mean().sort_index()
    plt.figure(figsize=(12, 5))
    plt.plot(full_seoul.index, full_seoul.values,
             label="Actual (2010~2025)", color="#1f77b4")
    plt.plot(seoul["timestamp"], seoul["forecast"],
             label="TFT Forecast (2025)", color="#d62728", linestyle="--")
    plt.fill_between(seoul["timestamp"], seoul["forecast_p10"],
                     seoul["forecast_p90"], color="#d62728", alpha=0.15,
                     label="80% interval")
    plt.axvline(pd.Timestamp(TRAIN_END), color="gray", linestyle=":", alpha=0.7,
                label="Train End")
    plt.title("TFT Backtest: Seoul Avg Apt Sale Price (per-dong → mean)")
    plt.xlabel("Date"); plt.ylabel(TARGET_COL)
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(OUT_DIR / "backtest_chart.png", dpi=130)
    plt.close()

    save_interpretation(model, validation, prefix="backtest")


# ---------------------------------------------------------------------------
# 9. 시나리오 (3) : 미래 10년 예측
# ---------------------------------------------------------------------------
def scenario_future(panel: pd.DataFrame) -> None:
    """전체 데이터 재학습 → 2026-01~2035-12 (120 months) 단일-shot 예측.

    [policy 미래값 처리]
      - 마지막 관측 시점의 policy 값을 carry-forward 하여 known_categoricals
        의 decoder 입력으로 사용.
    """
    df = prepare_for_tft(panel)
    last_obs_idx = int(df["time_idx"].max())
    last_snap = df[df["time_idx"] == last_obs_idx].copy()
    last_ts = pd.Timestamp(last_snap[TIME_COL].iloc[0])

    future_rows = []
    for offset in range(1, FUTURE_PRED_LEN + 1):
        chunk = last_snap.copy()
        chunk["time_idx"] = last_obs_idx + offset
        new_ts = last_ts + pd.DateOffset(months=offset)
        chunk[TIME_COL] = new_ts
        chunk["month"]  = str(new_ts.month)
        # unknown reals 는 decoder 에 들어가지 않으나 컬럼 존재 필요 ～ 0.
        # known macro reals (MACRO_KNOWN_REALS) 는 last_snap 의 값을 그대로
        # carry-forward (chunk = last_snap.copy() 이므로 이미 유지됨).
        unknown_cols = [c for c in chunk.columns
                        if (c.startswith("ecos__") or c.startswith("reb__"))
                        and c not in MACRO_KNOWN_REALS]
        chunk[unknown_cols] = 0.0
        future_rows.append(chunk)
    future_df = pd.concat(future_rows, ignore_index=True)
    df_with_future = pd.concat([df, future_df], ignore_index=True)

    training = build_training_dataset(df_with_future,
                                       training_cutoff=last_obs_idx,
                                       max_pred_len=FUTURE_PRED_LEN)

    print("\n[Future] training TFT (10y horizon) ...")
    predict_ds = TimeSeriesDataSet.from_dataset(
        training, df_with_future, predict=True, stop_randomization=True
    )
    model, _ = train_tft(training, predict_ds, log_subdir="future")
    _save_hyperparams("future", FUTURE_PRED_LEN)

    pred_loader = predict_ds.to_dataloader(train=False,
                                            batch_size=BATCH_SIZE * 2,
                                            num_workers=NUM_WORKERS)
    pred_out = model.predict(pred_loader, mode="raw", return_x=True)

    idx_to_ts = (df_with_future[["time_idx", TIME_COL]].drop_duplicates()
                                  .set_index("time_idx")[TIME_COL]
                                  .to_dict())
    group_map = {i: d for i, d in enumerate(sorted(df["dong"].unique()))}
    future_pred_df = _decode_predictions(pred_out, training, idx_to_ts, group_map)

    # ------------------------------------------------------------------
    # [Boundary anchor] 마지막 실측(2025-12) → 첫 예측(2026-01) 이격 흡수.
    # 백테스트와 동일 로직 (12개월 선형 감쇠).
    # ------------------------------------------------------------------
    ANCHOR_LEN = 12
    last_actuals = (panel.sort_values(TIME_COL)
                         .groupby("dong")[TARGET_COL].last())
    future_pred_df = (future_pred_df
                      .sort_values(["dong", "timestamp"])
                      .reset_index(drop=True))
    for dong, grp in future_pred_df.groupby("dong"):
        if dong not in last_actuals.index or len(grp) == 0:
            continue
        idx = grp.index.to_list()
        first_pred = future_pred_df.loc[idx[0], "pred_p50"]
        d = float(last_actuals[dong]) - float(first_pred)
        for k, ridx in enumerate(idx):
            w = max(0.0, 1.0 - k / float(ANCHOR_LEN))
            shift = d * w
            future_pred_df.loc[ridx, "pred_p10"] += shift
            future_pred_df.loc[ridx, "pred_p50"] += shift
            future_pred_df.loc[ridx, "pred_p90"] += shift

    future_pred_df.to_csv(OUT_DIR / "future_dong_forecast.csv", index=False)

    seoul_future = (future_pred_df.groupby("timestamp")
                    .agg(forecast=("pred_p50", "mean"),
                         forecast_p10=("pred_p10", "mean"),
                         forecast_p90=("pred_p90", "mean"))
                    .reset_index())

    full_seoul = (panel.groupby(TIME_COL)[TARGET_COL].mean()
                       .sort_index().reset_index()
                       .rename(columns={TIME_COL: "timestamp",
                                        TARGET_COL: "actual"}))
    timeline = pd.concat([
        full_seoul.assign(type="actual", forecast=np.nan,
                          forecast_p10=np.nan, forecast_p90=np.nan),
        seoul_future.assign(type="forecast", actual=np.nan),
    ], ignore_index=True).sort_values("timestamp")
    timeline.to_csv(OUT_DIR / "future_seoul_timeline.csv", index=False)

    plt.figure(figsize=(13, 5))
    plt.plot(full_seoul["timestamp"], full_seoul["actual"],
             label="Actual (2010~2025)", color="#1f77b4")
    plt.plot(seoul_future["timestamp"], seoul_future["forecast"],
             label="TFT Forecast (2026~2035)", color="#2ca02c", linestyle="--")
    plt.fill_between(seoul_future["timestamp"],
                     seoul_future["forecast_p10"],
                     seoul_future["forecast_p90"],
                     color="#2ca02c", alpha=0.15, label="80% interval")
    plt.axvline(pd.Timestamp("2025-12-01"), color="gray", linestyle=":",
                alpha=0.7, label="Forecast Start")
    plt.title("TFT 10-Year Forecast: Seoul Avg Apt Sale Price")
    plt.xlabel("Date"); plt.ylabel(TARGET_COL)
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(OUT_DIR / "future_chart.png", dpi=130)
    plt.close()

    save_interpretation(model, predict_ds, prefix="future")


# ---------------------------------------------------------------------------
# 10. 메인
# ---------------------------------------------------------------------------
def main():
    # 재현성 시드 고정 (main() 안에서만 호출 → Windows spawn 시 중복 방지)
    pl.seed_everything(42, workers=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_panel(DATA_CSV)
    raw.to_csv(OUT_DIR / "00_panel_raw.csv", index=False)

    clean = clean_missing_and_policy(raw)
    clean.to_csv(OUT_DIR / "01_panel_clean.csv", index=False)

    run_adf_report(clean, OUT_DIR / "02_adf_report.csv")

    print("\n=== Scenario 1+2: Backtest 2025-01 ~ 2025-12 ===")
    scenario_backtest(clean)

    print("\n=== Scenario 3: Future Forecast 2026-01 ~ 2035-12 ===")
    scenario_future(clean)

    print("\n[Done] 산출물이 다음 폴더에 저장되었습니다:", OUT_DIR)


if __name__ == "__main__":
    # ---------------------------------------------------------------------------
    # Windows multiprocessing spawn 보호:
    # torch DataLoader / Lightning 이 내부적으로 subprocess 를 생성할 때
    # 이 블록 밖의 코드가 재실행되지 않도록 보장.
    # ---------------------------------------------------------------------------
    try:
        main()
    except KeyboardInterrupt:
        print("\n[interrupted] 사용자에 의해 중단되었습니다.")
    except Exception:
        import traceback
        err_path = OUT_DIR / "_error_log.txt"
        tb = traceback.format_exc()
        print("\n[ERROR] 오류 발생:\n", tb)
        # cmd 창이 닫혀도 확인할 수 있도록 파일에도 기록
        try:
            err_path.parent.mkdir(parents=True, exist_ok=True)
            err_path.write_text(tb, encoding="utf-8")
            print(f"에러 상세 내용이 저장되었습니다: {err_path}")
        except Exception:
            pass
        sys.exit(1)
