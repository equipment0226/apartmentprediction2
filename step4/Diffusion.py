"""
SSSD(Structured State-Space Diffusion) 기반 서울시 아파트 매매가
다중 시나리오(What-if) 생성 파이프라인
=================================================================

데이터 : data/dong_level_meta_table.csv  (2010-01 ~ 2025-12, 월 단위)
타겟   : reb__apt_sale_avg_price (서울 평균으로 광역 집계)
모델   : Denoising Diffusion Probabilistic Model (DDPM)
         + DiffWave/SSSD 스타일 1D dilated-convolution backbone
         + past window + exogenous covariate condition
샘플   : 동일 조건에서 N_SCENARIOS(=100) 회 sampling → 다양한 미래 경로

----------------------------------------------------------------------------
[ 왜 Diffusion(SSSD) 인가? ]
  - VAR/TFT 는 "기대값(point) + 예측구간(quantile)" 한 가지 분포만 제공한다.
  - Diffusion 은 단순 학습된 데이터 분포 p(y | past, exog) 에서 IID 샘플을 무한히
    뽑을 수 있다 → "다양하면서도 historically grounded 한 100개 분기" 자연 생성.
  - SSSD 논문(Alcaraz & Strodthoff, 2022)은 backbone 으로 S4(state-space) layer 를
    쓰지만 CPU 환경에서 학습 가능한 경량화를 위해 본 구현은 DiffWave 스타일
    dilated-conv 블록(=causal long-receptive-field, S4 의 핵심 효과를 단순화한
    형태)으로 대체. 동일 "diffusion + structured long-context" 아이디어.

[ 시나리오 구성 ]
  (Step A) 데이터 로딩 / 결측 보간 / policy 정규화 / ADF 진단
  (Step B) RobustScaler + 슬라이딩-윈도우 학습셋 생성
  (Step C) Diffusion 모델 학습
  (Step D) 백테스트 : 2025-01 ~ 2025-12 (12개월) × 100 path → 실측 비교/지표
  (Step E) Future  : 2026-01 ~ 2035-12 (120개월) × 100 path
           ※ What-if : 정책(policy__) + 거시(ecos__) + 시장수급(reb__)
             feature 그룹별 다중 시나리오 (8가지) × 각 100 path

[ What-if 시나리오 매트릭스 ]
  baseline           : 모든 exog = 마지막 관측치 carry-forward
  policy_tight       : policy__ 전 항목 +1 (강화)
  policy_loose       : policy__ 전 항목 -1 (완화)
  rate_hike          : ecos__base_rate / mortgage_rate_new +1.5%p
  rate_cut           : ecos__base_rate / mortgage_rate_new -1.5%p
  demand_boom        : reb__*_supply_demand +20%, reb__apt_sale_index +5%
  demand_bust        : reb__*_supply_demand -20%, reb__apt_sale_index -5%
  stagflation        : rate_hike + cpi_housing +10%

중간 산출물은 모두 step4/ 폴더에 csv/png 로 저장.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# [Windows 필수 환경설정 - 반드시 다른 import 보다 먼저]
# (TFT.py 와 동일 사유 : OpenMP 충돌 / cp949 콘솔 unicode 크래시 방지)
# ---------------------------------------------------------------------------
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8:replace")
os.environ.setdefault("PYTHONWARNINGS", "ignore")

import sys
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import math
import json
import warnings
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.stattools import adfuller

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

warnings.filterwarnings("ignore")


# ===========================================================================
# 0. 경로 / 컬럼 정의
# ===========================================================================
BASE_DIR = Path(__file__).resolve().parent          # .../step4
ROOT_DIR = BASE_DIR.parent                          # .../newtrial
DATA_CSV = ROOT_DIR / "data" / "dong_level_meta_table.csv"
OUT_DIR  = BASE_DIR

TARGET_COL = "reb__apt_sale_avg_price"
TIME_COL   = "timestamp"
ID_COLS    = ["si", "gu", "dong"]


# ===========================================================================
# 1. 시나리오 구간 / 하이퍼파라미터 ( 한 곳에서 변경 )
# ===========================================================================
TRAIN_END    = "2023-12-01"   # 백테스트용 학습 종료
VALID_START  = "2024-01-01"
VALID_END    = "2025-12-01"
FUTURE_START = "2026-01-01"
FUTURE_END   = "2035-12-01"

# --- 윈도우 / horizon ---
PAST_LEN          = 48   # 인코더 과거 길이 = 4년치.
                          # 36→48 : 부동산 전체 사이클(급등→조정→회복) 컨텍스트 확보.
BACKTEST_HORIZON  = 12   # 백테스트 horizon = 2025년 12개월.
FUTURE_HORIZON    = 120  # 미래 10년 = 120개월.

# --- Diffusion 스케줄 ---
DIFF_TIMESTEPS    = 100  # T : diffusion timestep 수.
                          # 200→100 : denoising 스텝 감소 → 샘플이 덜 매끄러워짐
                          # (급등락 같은 고주파 변동 보존).
BETA_START        = 1e-4 # linear β schedule 시작값 (DDPM 기본).
BETA_END          = 0.04 # linear β schedule 종료값.
                          # 0.02→0.04 : 상단 노이즈 레벨 확대 → 큰 진폭 복원 학습.

# --- 모델 아키텍처 ---
RES_CHANNELS      = 96   # residual block 채널수.
                          # 64→96 : 비선형 패턴(V자·급락 후 반등) 표현력 증가.
SKIP_CHANNELS     = 96   # skip connection 채널수 (RES_CHANNELS 와 동일 유지).
N_RES_LAYERS      = 12   # residual block 개수.
                          # 8→12 : dilation 최대 2^11=2048, receptive field 대폭 확장.
COND_EMB_DIM      = 64   # exogenous covariate 임베딩 차원.
DIFF_STEP_EMB_DIM = 128  # diffusion step t 의 sinusoidal+MLP 임베딩 차원.

# --- 학습 ---
BATCH_SIZE        = 16   # 64→16 : 에폭당 그래디언트 업데이트 횟수 증가
                          # → 희귀 급등락 이벤트 윈도우 과소평가 완화.
N_EPOCHS          = 200  # 120→200 : 모델 용량 증가(레이어·채널) 보상.
                          # (diffusion loss 는 분산이 커 plateau 판단 어려움).
LEARNING_RATE     = 1e-3 # Adam. DiffWave/DDPM 표준값.
GRAD_CLIP         = 2.0  # 1.0→2.0 : 급격한 변화 학습 시 gradient 덜 clip.
LOG_EVERY         = 10   # epoch 단위 손실 출력 주기.
SEED              = 42

# --- 샘플링 ---
N_SCENARIOS       = 100  # 사용자 요구 : 100개 분기.
SAMPLE_BATCH      = 25   # CPU 메모리 보호. 4회 누적으로 100 path 생성.

# Sampling 하이퍼파라미터
ANCHOR_LEN        = 12   # boundary anchor decay 길이(개월).
                          # 과거 마지막 시점과 첫 예측의 level 이격을 12개월에
                          # 걸쳐 선형으로 흡수. 모든 시나리오에 동일 적용.
NOISE_TEMP        = 1.5  # reverse-process posterior noise 온도 배율.
                          # 1.0 = 이론값. >1 은 후반 step 에서도 stochasticity
                          # 유지 → 초반에만 변동성이 몰리는 현상 완화.

# --- 진단 ---
ADF_ALPHA         = 0.05

DEVICE = torch.device("cpu")
torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))


# ===========================================================================
# 2. 데이터 로딩 & 전처리 ( VAR/TFT 와 동일 정신 )
# ===========================================================================
def load_and_aggregate(csv_path: Path) -> pd.DataFrame:
    """동(洞) 단위 패널을 서울 평균(광역) 월별 단변량 다특성 시계열로 집계.

    [근거] Diffusion 학습은 단일 시퀀스만으로도 풍부한 표현이 가능하고,
           '서울 전체 평균 가격' 이라는 macro 시나리오 해석이 사용자 의도에
           맞다. (TFT 처럼 panel 사용도 가능하나, 시나리오 분기 해석은
           광역 한 시계열이 더 명확.)
    """
    df = pd.read_csv(csv_path, parse_dates=[TIME_COL])
    df = df.sort_values([TIME_COL]).reset_index(drop=True)

    feature_cols = [c for c in df.columns if c not in ID_COLS + [TIME_COL]]
    numeric_cols = [c for c in feature_cols
                    if pd.api.types.is_numeric_dtype(df[c])]

    monthly = df.groupby(TIME_COL)[numeric_cols].mean().sort_index()
    return monthly


def clean_missing_and_policy(df: pd.DataFrame) -> pd.DataFrame:
    """시간축 보간 + policy__ → {-1,0,1} 정수 강제.

    [파라미터 근거]
      - interpolate(method="time", limit_direction="both") : 시간 가중 보간.
        양 끝 NaN 도 가까운 관측치로 채워 학습 윈도우 손실 방지.
      - policy round/clip : 사용자 요구 (강화+1/유지0/완화-1) 의미 보존.
    """
    out = df.copy().interpolate(method="time", limit_direction="both")
    out = out.fillna(0.0)
    policy_cols = [c for c in out.columns if c.startswith("policy__")]
    out[policy_cols] = out[policy_cols].round().clip(-1, 1).astype(int)
    return out


def run_adf_report(df: pd.DataFrame, save_path: Path) -> pd.DataFrame:
    """주요 수치형 컬럼 각각에 ADF 정상성 검정 → csv 저장 (진단/보고용).

    [근거] regression='ct' (상수+추세), autolag='AIC' → VAR/TFT 와 동일 설정.
           Diffusion 자체는 정상성 가정 불필요하나 사용자 요구 충족.
    """
    rows = []
    for col in df.columns:
        series = df[col].dropna()
        if series.nunique() < 5 or len(series) < 24:
            continue
        try:
            stat, pval, *_ = adfuller(series.values,
                                      regression="ct", autolag="AIC")
            rows.append({"column": col, "adf_stat": stat, "p_value": pval,
                         "stationary_at_5pct": int(pval < ADF_ALPHA)})
        except Exception as e:
            rows.append({"column": col, "adf_stat": np.nan, "p_value": np.nan,
                         "stationary_at_5pct": 0})
    report = pd.DataFrame(rows)
    report.to_csv(save_path, index=False)
    n_stat = int(report["stationary_at_5pct"].sum())
    print(f"[ADF] stationary(0.05) = {n_stat}/{len(report)} columns")
    return report


# ===========================================================================
# 3. Robust 스케일링 + 슬라이딩 윈도우 데이터셋
# ===========================================================================
def fit_scalers(train_df: pd.DataFrame):
    """타겟과 covariate 를 각각 RobustScaler 로 적합.

    [근거] target 과 exog 의 스케일 차이가 매우 큼 (가격 ~1e8, 금리 ~1e0).
           각각 별도 scaler 로 fit → inverse_transform 시 누설 없음.
    """
    target_scaler = RobustScaler(quantile_range=(1, 99)).fit(train_df[[TARGET_COL]].values)
    cov_cols = [c for c in train_df.columns if c != TARGET_COL]
    cov_scaler = RobustScaler(quantile_range=(1, 99)).fit(train_df[cov_cols].values)
    # quantile_range=(1,99) : (5,95) 대비 극단값까지 완전 보존.
    # 2020-21 고부양 +93% / 2022 급락 -30% 를 outlier 취급 안 하고
    # 스케일 내에 보존 → 모델이 장기 큰 진폭 변동도 자연스레운 그랄로 학습.
    return target_scaler, cov_scaler, cov_cols


def transform_df(df: pd.DataFrame, target_scaler: RobustScaler,
                 cov_scaler: RobustScaler, cov_cols: list[str]):
    """전체 df → (target_array, covariate_matrix) 정규화 결과 반환."""
    y = target_scaler.transform(df[[TARGET_COL]].values).astype(np.float32).flatten()
    X = cov_scaler.transform(df[cov_cols].values).astype(np.float32)
    return y, X


def build_sliding_windows(y: np.ndarray, X: np.ndarray,
                          past_len: int, horizon: int):
    """슬라이딩 윈도우 (past_y, past_X, future_X) → future_y 학습쌍 생성.

    [근거]
      - 학습 1 sample = (인코더 36 + 디코더 12) 의 multivariate window.
      - exogenous (covariate) 는 학습 시점에서 미래 구간이 "known input" 으로
        주어진다고 가정 (sklearn-style covariate / TFT 의 time_varying_known
        대응). 이를 통해 What-if 시나리오를 covariate 만 바꿔 생성 가능.
    """
    T = len(y)
    end = T - past_len - horizon + 1
    past_y_list, past_X_list, fut_X_list, fut_y_list = [], [], [], []
    for s in range(end):
        past_y_list.append(y[s : s + past_len])
        past_X_list.append(X[s : s + past_len])
        fut_X_list .append(X[s + past_len : s + past_len + horizon])
        fut_y_list .append(y[s + past_len : s + past_len + horizon])
    return (np.stack(past_y_list),
            np.stack(past_X_list),
            np.stack(fut_X_list),
            np.stack(fut_y_list))


def compute_window_weights(fut_y: np.ndarray,
                           recency_halflife: float = 12.0,
                           vol_alpha: float = 1.0,
                           vol_floor: float = 0.25) -> np.ndarray:
    """윤도우별 학습 가중치 계산 = recency × volatility.

    [문제]
      2010-2024 학습 윈도우 중 2010-19 flat 구간이 2/3 다수파 → 모델이
      'flat default' 을 학습해 2020+ 급등 구간 반영이 약하다.
      backtest 에서 실측 대비 band 이탈 · 높은 RMSE 로 나타남.

    [해결]
            (a) recency weight : w_r = 0.5 ^ ((N-1-i) / halflife)
                    - 이론적 근거 : 시계열 regime change (2020 저금리/유동성 쇼크)
                        이후 구조는 이전과 다른 분포 → 최근 관측치의 정보량 더 큼.
                    - halflife=12 (1년) → 2020+ 구간이 2010-15 구간보다 약 128배 가중.
      (b) volatility weight : w_v = (vol_floor + |Δy|.mean()) ** vol_alpha
          - 이론적 근거 : 변동성 큰 윈도우(급등·급락)이 극한적 분포의
            tail 을 제공 → mean-MSE 학습에서 제대로 반영되도록 up-weight.
          - vol_floor=0.25 로 flat 윈도우가 0으로 소멸되지 않도록 방지.
      최종 가중치 w = w_r * w_v, 평균=1로 정규화 (원 손실 스케일 유지).
    """
    N = len(fut_y)
    # (a) recency : 을 마지막으로 갈수로 1에 근접
    idx = np.arange(N, dtype=float)
    w_r = np.power(0.5, (N - 1 - idx) / float(recency_halflife))
    # (b) volatility : future horizon 내 월변화율 평균 |Δy|/y
    # fut_y shape: (N, horizon) - already scaled
    diff = np.abs(np.diff(fut_y, axis=1)).mean(axis=1)  # (N,)
    diff = diff / (np.median(diff) + 1e-8)  # 중앙값으로 정규화 → 스케일 무관
    w_v = np.power(vol_floor + diff, vol_alpha)
    w = w_r * w_v
    w = w / w.mean()  # 평균 1로 정규화 → 전체 loss 스케일 보존
    return w.astype(np.float32)


# ===========================================================================
# 4. Diffusion 스케줄 (DDPM)
# ===========================================================================
class DiffusionSchedule:
    """DDPM linear β schedule + 사전계산 α / sqrt 항.

    [수식]   q(x_t | x_0)   = N( √ᾱ_t x_0 , (1-ᾱ_t) I )
             p_θ(x_{t-1}|x_t,c) = N( μ_θ(x_t,t,c) , σ_t² I )
             ε-prediction : 모델은 x_t 에서 noise ε 를 예측.
    """
    def __init__(self, T: int = DIFF_TIMESTEPS,
                 beta_start: float = BETA_START,
                 beta_end:   float = BETA_END,
                 device: torch.device = DEVICE):
        self.T = T
        betas        = torch.linspace(beta_start, beta_end, T, device=device)
        alphas       = 1.0 - betas
        alpha_bar    = torch.cumprod(alphas, dim=0)
        alpha_bar_prev = torch.cat([torch.ones(1, device=device), alpha_bar[:-1]])

        self.betas             = betas
        self.alphas            = alphas
        self.alpha_bar         = alpha_bar
        self.alpha_bar_prev    = alpha_bar_prev
        self.sqrt_alpha_bar    = torch.sqrt(alpha_bar)
        self.sqrt_one_minus_ab = torch.sqrt(1.0 - alpha_bar)
        # posterior variance (DDPM Eq.7)
        self.posterior_var     = betas * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor,
                 noise: torch.Tensor) -> torch.Tensor:
        """forward diffusion : x_0 → x_t """
        s1 = self.sqrt_alpha_bar[t].view(-1, 1, 1)
        s2 = self.sqrt_one_minus_ab[t].view(-1, 1, 1)
        return s1 * x0 + s2 * noise


# ===========================================================================
# 5. DiffWave / SSSD 스타일 1D dilated-conv backbone
# ===========================================================================
class SinusoidalStepEmbedding(nn.Module):
    """Diffusion step t 의 sinusoidal positional embedding + MLP.
    [근거] DDPM/DiffWave 표준. t 에 대해 매끄러운 임베딩 제공.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim * 4),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device) / (half - 1)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.mlp(emb)


class ResidualBlock(nn.Module):
    """DiffWave residual block : dilated conv + gated activation + step/cond bias.

    [구조]
      h = conv_dilated(x)
      h = h + step_proj(step_emb) + cond_proj(cond_seq)
      h = tanh(h[:C]) * sigmoid(h[C:])      # gated
      residual = res_proj(h) + x
      skip     = skip_proj(h)
    """
    def __init__(self, res_channels: int, skip_channels: int,
                 dilation: int, step_emb_dim: int, cond_channels: int):
        super().__init__()
        self.dilated = nn.Conv1d(res_channels, 2 * res_channels, kernel_size=3,
                                 padding=dilation, dilation=dilation)
        self.step_proj = nn.Linear(step_emb_dim, 2 * res_channels)
        self.cond_proj = nn.Conv1d(cond_channels, 2 * res_channels, kernel_size=1)
        self.res_proj  = nn.Conv1d(res_channels, res_channels, kernel_size=1)
        self.skip_proj = nn.Conv1d(res_channels, skip_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, step_emb: torch.Tensor,
                cond_seq: torch.Tensor):
        # x        : (B, C, L_fut)
        # step_emb : (B, step_emb_dim)
        # cond_seq : (B, cond_channels, L_fut)
        h = self.dilated(x)
        h = h + self.step_proj(step_emb).unsqueeze(-1)
        h = h + self.cond_proj(cond_seq)
        gate, filt = h.chunk(2, dim=1)
        h = torch.sigmoid(gate) * torch.tanh(filt)
        return (self.res_proj(h) + x) / math.sqrt(2.0), self.skip_proj(h)


class SSSDDiffusionNet(nn.Module):
    """ε-prediction network for 1-D target diffusion conditioned on
       (past_target, past_covariates, future_covariates).

    [입력 텐서]
      noisy_y_future : (B, 1, L_fut)       ← x_t
      step           : (B,)
      cond           : (B, cond_dim, L_fut) ← past+future 요약을 L_fut 길이에 정렬
    """
    def __init__(self, cov_dim: int):
        super().__init__()
        self.cov_dim = cov_dim
        cond_channels = COND_EMB_DIM

        # ---- conditioning encoders ----
        # past : (past_y, past_X) 를 시계열 channel 로 concat → 1D conv 압축
        self.past_encoder = nn.Sequential(
            nn.Conv1d(1 + cov_dim, cond_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(cond_channels, cond_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            # global pool → broadcast 용도. shape : (B, cond_channels, 1)
            nn.AdaptiveAvgPool1d(1),
        )
        # future covariates : (B, cov_dim, L_fut) → cond_channels per-step
        self.future_encoder = nn.Sequential(
            nn.Conv1d(cov_dim, cond_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(cond_channels, cond_channels, kernel_size=3, padding=1),
        )

        # ---- diffusion step embedding ----
        self.step_emb = SinusoidalStepEmbedding(DIFF_STEP_EMB_DIM // 4)
        # SinusoidalStepEmbedding output dim = dim*4 = DIFF_STEP_EMB_DIM

        # ---- input projection ----
        self.input_proj = nn.Conv1d(1, RES_CHANNELS, kernel_size=1)

        # ---- residual stack (exponentially increasing dilation) ----
        self.res_layers = nn.ModuleList([
            ResidualBlock(
                res_channels  = RES_CHANNELS,
                skip_channels = SKIP_CHANNELS,
                dilation      = 2 ** (i % 9),   # 1,2,4,...,256 cyclic
                step_emb_dim  = DIFF_STEP_EMB_DIM,
                cond_channels = cond_channels,
            )
            for i in range(N_RES_LAYERS)
        ])

        # ---- output projection ----
        self.out = nn.Sequential(
            nn.SiLU(),
            nn.Conv1d(SKIP_CHANNELS, SKIP_CHANNELS, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(SKIP_CHANNELS, 1, kernel_size=1),
        )

    def forward(self, noisy_y: torch.Tensor, step: torch.Tensor,
                past_y: torch.Tensor, past_X: torch.Tensor,
                fut_X: torch.Tensor) -> torch.Tensor:
        # past_y : (B, L_past)        past_X : (B, L_past, cov_dim)
        # fut_X  : (B, L_fut,  cov_dim) noisy_y : (B, 1, L_fut)
        B, _, L_fut = noisy_y.shape

        past_in = torch.cat([past_y.unsqueeze(1),
                             past_X.transpose(1, 2)], dim=1)   # (B,1+C,L_past)
        past_vec = self.past_encoder(past_in)                  # (B, cond, 1)
        past_broad = past_vec.expand(-1, -1, L_fut)            # broadcast

        fut_in = fut_X.transpose(1, 2)                         # (B, C, L_fut)
        fut_enc = self.future_encoder(fut_in)                  # (B, cond, L_fut)

        cond_seq = past_broad + fut_enc                        # additive fusion

        step_emb = self.step_emb(step)                         # (B, step_emb_dim)

        x = self.input_proj(noisy_y)                           # (B, C_res, L_fut)
        skip_sum = 0.0
        for layer in self.res_layers:
            x, skip = layer(x, step_emb, cond_seq)
            skip_sum = skip_sum + skip
        out = self.out(skip_sum / math.sqrt(len(self.res_layers)))
        return out                                              # (B, 1, L_fut)


# ===========================================================================
# 6. 학습 루프
# ===========================================================================
def train_diffusion(model: SSSDDiffusionNet,
                    schedule: DiffusionSchedule,
                    loader: DataLoader,
                    n_epochs: int = N_EPOCHS,
                    lr: float = LEARNING_RATE,
                    log_every: int = LOG_EVERY,
                    log_csv: Path | None = None) -> pd.DataFrame:
    """ε-prediction MSE 손실로 학습. epoch 별 평균 loss 기록.

    [수식] L_simple = E_t [ || ε - ε_θ(x_t, t, c) ||² ]   (DDPM Eq.14)
    """
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = []

    for epoch in range(1, n_epochs + 1):
        losses = []
        for batch in loader:
            # 가중치 원활 지원 : (past_y, past_X, fut_X, fut_y, w) 또는 (..., fut_y)
            if len(batch) == 5:
                past_y, past_X, fut_X, fut_y, w = batch
                w = w.to(DEVICE)
            else:
                past_y, past_X, fut_X, fut_y = batch
                w = None
            past_y = past_y.to(DEVICE)
            past_X = past_X.to(DEVICE)
            fut_X  = fut_X .to(DEVICE)
            fut_y  = fut_y .to(DEVICE).unsqueeze(1)  # (B,1,L_fut)

            B = fut_y.size(0)
            t = torch.randint(0, schedule.T, (B,), device=DEVICE)
            noise = torch.randn_like(fut_y)
            noisy = schedule.q_sample(fut_y, t, noise)

            pred_noise = model(noisy, t, past_y, past_X, fut_X)
            if w is None:
                loss = F.mse_loss(pred_noise, noise)
            else:
                # per-sample weighted MSE : (B,1,L) → sample-mean → w 가중 평균
                per = (pred_noise - noise).pow(2).mean(dim=(1, 2))  # (B,)
                loss = (per * w).sum() / w.sum()

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            losses.append(loss.item())

        ep_loss = float(np.mean(losses))
        history.append({"epoch": epoch, "loss": ep_loss})
        if epoch == 1 or epoch % log_every == 0 or epoch == n_epochs:
            print(f"  [epoch {epoch:>3}/{n_epochs}] loss = {ep_loss:.4f}")

    hist_df = pd.DataFrame(history)
    if log_csv is not None:
        hist_df.to_csv(log_csv, index=False)
    return hist_df


# ===========================================================================
# 7. 샘플링 (Ancestral DDPM)
# ===========================================================================
@torch.no_grad()
def sample_scenarios(model: SSSDDiffusionNet,
                     schedule: DiffusionSchedule,
                     past_y_np: np.ndarray,
                     past_X_np: np.ndarray,
                     fut_X_np: np.ndarray,
                     n_scenarios: int = N_SCENARIOS,
                     batch: int = SAMPLE_BATCH,
                     anchor_len: int = ANCHOR_LEN,
                     noise_temp: float = NOISE_TEMP) -> np.ndarray:
    """동일한 (past, fut_X) 조건에서 n_scenarios 개 분기 샘플링.

    [근거] 표본은 reverse process p_θ(x_{t-1}|x_t) 의 stochastic 샘플링에서
           근본적으로 무작위성 → 동일 조건 N 회 호출만으로 다양 경로 확보.

    [보정] 
      1) noise_temp : reverse step 의 posterior noise 를 noise_temp 배 해
         후반 step 에서도 충분한 변동성 유지 (초반 급등락만 남는 현상 방지).
      2) boundary anchor : paths[:, k] += (past_y[-1] - paths[:, 0]) *
         max(0, 1 - k/anchor_len). 시나리오 무관 첨 시점 level 이격을
         12개월 선형 감쇠로 흡수. scaled space 에서 적용.

    Returns
    -------
    paths : (n_scenarios, horizon) np.ndarray  in scaled space
    """
    model.eval()
    L_fut = fut_X_np.shape[0]
    paths = []

    # 조건 텐서 (B=batch 반복)
    past_y_t = torch.from_numpy(past_y_np ).float().to(DEVICE).unsqueeze(0)
    past_X_t = torch.from_numpy(past_X_np ).float().to(DEVICE).unsqueeze(0)
    fut_X_t  = torch.from_numpy(fut_X_np  ).float().to(DEVICE).unsqueeze(0)

    # boundary anchor 기준값 : 과거 마지막 시점의 scaled target
    past_y_last_scaled = float(past_y_np[-1, 0]) if past_y_np.ndim == 2 \
                                                  else float(past_y_np[-1])

    remaining = n_scenarios
    while remaining > 0:
        b = min(batch, remaining)
        py = past_y_t.expand(b, -1).contiguous() if past_y_t.dim() == 2 \
             else past_y_t.expand(b, -1, -1).contiguous()
        pX = past_X_t.expand(b, -1, -1).contiguous()
        fX = fut_X_t .expand(b, -1, -1).contiguous()

        x = torch.randn(b, 1, L_fut, device=DEVICE)          # x_T ~ N(0,I)
        for t in reversed(range(schedule.T)):
            t_batch = torch.full((b,), t, device=DEVICE, dtype=torch.long)
            eps = model(x, t_batch, py, pX, fX)

            a_t   = schedule.alphas[t]
            ab_t  = schedule.alpha_bar[t]
            mean  = (1.0 / torch.sqrt(a_t)) * (
                x - (schedule.betas[t] / torch.sqrt(1.0 - ab_t)) * eps
            )
            if t > 0:
                var = schedule.posterior_var[t]
                # noise_temp > 1 → 후반 step에서도 충분한 stochasticity 유지
                x = mean + torch.sqrt(var) * noise_temp * torch.randn_like(x)
            else:
                x = mean

        paths.append(x.squeeze(1).cpu().numpy())
        remaining -= b

    paths_arr = np.concatenate(paths, axis=0)   # (n_scenarios, L_fut)

    # ------------------------------------------------------------------
    # Boundary anchor : (past_y_last - paths[:, 0]) 을 12개월 선형 감쇠로 흡수.
    # 모든 시나리오 · 모든 path 에 동일 적용 → t/t+1 이격 제거.
    # ------------------------------------------------------------------
    if anchor_len > 0 and L_fut > 0:
        delta = past_y_last_scaled - paths_arr[:, 0:1]            # (N, 1)
        k = np.arange(L_fut, dtype=float)
        w = np.maximum(0.0, 1.0 - k / float(anchor_len))[None, :]  # (1, L)
        paths_arr = paths_arr + delta * w

    return paths_arr


# ===========================================================================
# 8. What-if 시나리오 정의 (정책 + 금리 + 거시 + 수급 + 역사 재현)
# ===========================================================================
#
# [설계 원칙]
#   1) 모든 perturbation 값은 본 데이터(2010-01~2025-12)의 실제 관측 분포 또는
#      한국 부동산 역사적 사건에 anchor 한다. (출처를 주석으로 명시)
#   2) "급변" 시나리오는 데이터 내 가장 빠른 실제 변동률을 상한으로 사용.
#   3) ramp 구간을 두어 즉각 충격이 아닌 현실적 시간 경로(점진 변화)를 부여.
#   4) 카테고리:
#        A. 금리      : 4개  (코로나/긴축/완화/스태그)
#        B. 정책      : 5개  (문재인式/현행유지/세제만강화/대출만완화/혼합)
#        C. 거시      : 4개  (스태그플레이션/IMF형/V자회복/골디락스)
#        D. 수급      : 4개  (공급부족/공급과잉/전세난/거래절벽)
#        E. 역사재현  : 5개  (2010-14침체/2017-19규제/2020-21폭등/2022충격/2023회복)
#      + baseline = 총 23개
#
# [데이터 관측 사실 (시나리오 설계 근거)]
#   - base_rate         : 0.50(2020.5) ~ 3.50(2023.1) - 33개월 만에 +3.0%p 급변
#   - mortgage_rate_new : 2.39(2020.7) ~ 5.88(2010.1) - long-term 범위 3.5%p
#   - cd_91d_rate       : 0.63(2020.6) ~ 4.02(2022.12)
#   - cpi_housing       : 83.0(2010) ~ 117.1(2025) - 15년 누적 +41%
#   - unemployment      : 2.25(2022) ~ 5.15(2020.코로나) - 사이클 폭 ~3%p
#   - sale_supply_demand: 64(2022.12 거래절벽) ~ 127 (2017 과열) - 평균 97
#   - sale_index        : 72 ~ 192 (2010=83 → 2021=185)
#   - sale_avg_price    : 508(2010) → 1848(2021 peak) → 1267(2022) → 1506(2025)
#   - policy 컬럼       : 양극(-1, +1) 카테고리 분포 (0은 1기 신도시 등 일부)
# ===========================================================================

# 시나리오에서 자주 참조하는 컬럼 그룹 (한 곳에서 관리)
RATE_COLS_KR = ("ecos__base_rate", "ecos__mortgage_rate_new", "ecos__cd_91d_rate")
POLICY_DEMAND_COLS = (
    "policy__ltv_regime", "policy__dti_regime", "policy__dsr_regime",
)
POLICY_TAX_COLS = (
    "policy__capital_gains_tax_regime",
    "policy__comprehensive_real_estate_tax_regime",
    "policy__acquisition_tax_regime",
)
POLICY_ZONING_COLS = (
    "policy__adjustment_target_area_regime",
    "policy__speculative_overheated_district_regime",
    "policy__speculative_zone_regime",
    "policy__real_estate_policy_regime",
)
ALL_POLICY_COLS = POLICY_DEMAND_COLS + POLICY_TAX_COLS + POLICY_ZONING_COLS

SD_COLS = (
    "reb__apt_sale_supply_demand",
    "reb__apt_jeonse_supply_demand",
    "reb__apt_monthly_rent_supply_demand",
)


def _ramp(start: float, end: float, L: int, ramp_months: int) -> np.ndarray:
    """start 값에서 end 값으로 ramp_months 동안 선형 변화 후 end 유지.

    [근거] 통화정책/규제는 즉시가 아닌 수개월에 걸쳐 누적 효과 → 점진 ramp 가
           '한국은행 점도표 발표 → 시장 반영' 같은 실제 메커니즘과 부합.
    """
    out = np.full(L, end, dtype=float)
    n = max(0, min(ramp_months, L))
    if n > 0:
        out[:n] = np.linspace(start, end, n)
    return out


def _trend(start: float, annual_pct: float, L: int,
           floor: float | None = None, cap: float | None = None) -> np.ndarray:
    """start 에서 연 annual_pct(%) 복리로 장기 추세 (10년+ 지속).

    [근거] CPI / 인구구조 / 추세적 규제환경 같은 구조적 힘은
           ramp처럼 '수절 다다르면 멈추는' 게 아니라 매 월 누적된다.
           예: 일본 1990-2010 CPI 연 +0.3%, 한국 1970년대 연 +15%,
               2010-2025 한국 주거 CPI 연 +2.3% 실측.
           - ramp 와 달리 종료 시점이 없으므로 장기 평균회귀를 자연스럽게
             이겼다(보하/상방/하방 추세 유지).
    """
    r = annual_pct / 100.0 / 12.0  # 월복리
    out = start * np.power(1.0 + r, np.arange(L, dtype=float))
    if floor is not None:
        out = np.maximum(out, floor)
    if cap is not None:
        out = np.minimum(out, cap)
    return out


def _set_cols(x: np.ndarray, cov_cols: list[str], col_idx: dict,
              names: tuple, value) -> None:
    """names 에 해당하는 컬럼들을 value 로 덮어쓴다 (in-place).
    value 는 scalar 또는 (L,) ndarray."""
    for c in names:
        if c in col_idx:
            x[:, col_idx[c]] = value


def _add_cols(x: np.ndarray, cov_cols: list[str], col_idx: dict,
              names: tuple, delta) -> None:
    """names 컬럼들에 delta (scalar 또는 (L,) array) 를 더한다 (in-place)."""
    for c in names:
        if c in col_idx:
            x[:, col_idx[c]] = x[:, col_idx[c]] + delta


def build_whatif_covariates(base_future_X: np.ndarray,
                            cov_cols: list[str],
                            cov_scaler: RobustScaler) -> dict:
    """미래 covariate 행렬을 23개 What-if 시나리오(원 단위 기준)로 변형 후
       정규화 공간 ndarray 로 반환.

    Returns
    -------
    dict[scenario_name] = (L_fut, cov_dim) ndarray  (정규화 공간)
    """
    L = base_future_X.shape[0]
    base_orig = cov_scaler.inverse_transform(base_future_X)  # (L, C) 원 단위
    col_idx = {c: i for i, c in enumerate(cov_cols)}

    # 마지막 관측치 (2025-12) - 현재 시점 기준
    cur = base_orig[0].copy()
    def cur_v(c: str) -> float:
        return float(cur[col_idx[c]]) if c in col_idx else 0.0

    cur_base_rate     = cur_v("ecos__base_rate")          # ≈ 2.50
    cur_mortgage_rate = cur_v("ecos__mortgage_rate_new")  # ≈ 4.23
    cur_cd91          = cur_v("ecos__cd_91d_rate")        # ≈ 2.84
    cur_cpi           = cur_v("ecos__cpi_housing")        # ≈ 117.1
    cur_unemp         = cur_v("ecos__unemployment_rate")  # ≈ 3.70
    cur_sale_idx      = cur_v("reb__apt_sale_index")      # ≈ 192
    cur_sale_sd       = cur_v("reb__apt_sale_supply_demand")    # ≈ 104
    cur_jeon_sd       = cur_v("reb__apt_jeonse_supply_demand")  # ≈ 105
    cur_rent_sd       = cur_v("reb__apt_monthly_rent_supply_demand")

    def _make(modifier: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        """modifier 가 적용된 원 단위 행렬을 정규화 공간으로 변환."""
        x = base_orig.copy()
        x = modifier(x)
        return cov_scaler.transform(x).astype(np.float32)

    S: dict[str, np.ndarray] = {}

    # ----------------------------------------------------------------------
    # (0) baseline : 모든 exog = 2025-12 값 carry-forward (변화 없음)
    # ----------------------------------------------------------------------
    S["baseline"] = base_future_X.copy()

    # ======================================================================
    # A. 금리 시나리오 (4개)
    # ======================================================================

    # A1) rate_shock_hike_2022 : 2021.8 → 2023.1 의 한국은행 21개월 +3.0%p
    #     역사적으로 가장 가파른 인상 사이클을 그대로 재현.
    #     base : 2.50 → 5.50  (현재 + 3.0%p)
    #     mortgage : 4.23 → 7.0   (peak 대출금리 = base+ ~1.5%p 통상 spread,
    #                              2010.1 = 5.88 이 상한이지만 base+3.0 시 시장
    #                              spread 확장으로 7.0 추정)
    #     cd91 : 2.84 → 5.84
    def f_a1(x):
        base_path = _ramp(cur_base_rate,     cur_base_rate     + 3.0, L, 21)
        mort_path = _ramp(cur_mortgage_rate, cur_mortgage_rate + 2.8, L, 21)
        cd_path   = _ramp(cur_cd91,          cur_cd91          + 3.0, L, 21)
        x[:, col_idx["ecos__base_rate"]]          = base_path
        x[:, col_idx["ecos__mortgage_rate_new"]]  = mort_path
        x[:, col_idx["ecos__cd_91d_rate"]]        = cd_path
        return x
    S["A1_rate_shock_hike_2022"] = _make(f_a1)

    # A2) rate_gradual_hike : 24개월간 +1.5%p 점진 인상
    #     근거 : 일반적 긴축 사이클 평균 속도(2018-2019 0.75%p/yr).
    def f_a2(x):
        x[:, col_idx["ecos__base_rate"]]         = _ramp(cur_base_rate,
                                                          cur_base_rate + 1.5, L, 24)
        x[:, col_idx["ecos__mortgage_rate_new"]] = _ramp(cur_mortgage_rate,
                                                          cur_mortgage_rate + 1.4, L, 24)
        x[:, col_idx["ecos__cd_91d_rate"]]       = _ramp(cur_cd91,
                                                          cur_cd91 + 1.5, L, 24)
        return x
    S["A2_rate_gradual_hike"] = _make(f_a2)

    # A3) rate_gradual_cut : 24개월간 -1.5%p 점진 인하 (현 2.5 → 1.0)
    #     근거 : 2024~2025 한미 동시 인하 사이클 연장. 하한은 mortgage 2.5
    #            (코로나 저점 = 2.39).
    def f_a3(x):
        end_base = max(0.5, cur_base_rate - 1.5)
        end_mort = max(2.5, cur_mortgage_rate - 1.5)
        end_cd   = max(0.7, cur_cd91 - 1.5)
        x[:, col_idx["ecos__base_rate"]]         = _ramp(cur_base_rate,     end_base, L, 24)
        x[:, col_idx["ecos__mortgage_rate_new"]] = _ramp(cur_mortgage_rate, end_mort, L, 24)
        x[:, col_idx["ecos__cd_91d_rate"]]       = _ramp(cur_cd91,          end_cd,   L, 24)
        return x
    S["A3_rate_gradual_cut"] = _make(f_a3)

    # A4) rate_zero_lower_covid : 코로나급 제로금리 (base→0.5, mortgage→2.5)
    #     근거 : 2020.5 ~ 2021.7 실측값 그대로. ramp 12개월(빠른 양적완화).
    def f_a4(x):
        x[:, col_idx["ecos__base_rate"]]         = _ramp(cur_base_rate,     0.50, L, 12)
        x[:, col_idx["ecos__mortgage_rate_new"]] = _ramp(cur_mortgage_rate, 2.50, L, 12)
        x[:, col_idx["ecos__cd_91d_rate"]]       = _ramp(cur_cd91,          0.70, L, 12)
        return x
    S["A4_rate_zero_lower_covid"] = _make(f_a4)

    # ======================================================================
    # B. 정책 시나리오 (5개)
    # ======================================================================

    # B1) policy_full_tight_moonjaein : 2017-2019 패턴 = 모든 규제 +1
    #     근거 : 데이터상 2017.8 ~ 2020.6 구간 거의 모든 policy__ 컬럼 = +1.
    def f_b1(x):
        _set_cols(x, cov_cols, col_idx, ALL_POLICY_COLS, +1)
        return x
    S["B1_policy_full_tight_moonjaein"] = _make(f_b1)

    # B2) policy_full_loose_current : 2024~2025 현행 = 모든 규제 -1
    #     근거 : 2025-12 실측 - LTV=-1, 종부세=-1, 양도세=-1 등 대부분 완화.
    def f_b2(x):
        _set_cols(x, cov_cols, col_idx, ALL_POLICY_COLS, -1)
        return x
    S["B2_policy_full_loose_current"] = _make(f_b2)

    # B3) policy_demand_loose_tax_tight : 대출 완화 + 세제 강화
    #     근거 : 정책 mix 사례 = 2014-2016 (LTV/DTI 완화하되 보유세는 점진 강화).
    def f_b3(x):
        _set_cols(x, cov_cols, col_idx, POLICY_DEMAND_COLS, -1)
        _set_cols(x, cov_cols, col_idx, POLICY_TAX_COLS, +1)
        return x
    S["B3_demand_loose_tax_tight"] = _make(f_b3)

    # B4) policy_demand_tight_tax_loose : 대출 강화 + 세제 완화 (DSR 도입형)
    #     근거 : 2021 DSR 도입 + 세제는 다주택자 한정 강화 시나리오 변형.
    def f_b4(x):
        _set_cols(x, cov_cols, col_idx, POLICY_DEMAND_COLS, +1)
        _set_cols(x, cov_cols, col_idx, POLICY_TAX_COLS, -1)
        return x
    S["B4_demand_tight_tax_loose"] = _make(f_b4)

    # B5) policy_neutral_supply_focus : 모든 규제 0 + 공급정책 → 수급 +10
    #     근거 : 3기 신도시·재건축 활성화로 공급 우위 (sale_sd +10 가정).
    def f_b5(x):
        _set_cols(x, cov_cols, col_idx, ALL_POLICY_COLS, 0)
        # 공급정책 효과 = supply_demand 지수 +10 (2014-2016 평균 변화폭 ≈ 8)
        for c in SD_COLS:
            x[:, col_idx[c]] = x[:, col_idx[c]] + 10
        return x
    S["B5_neutral_supply_focus"] = _make(f_b5)

    # ======================================================================
    # C. 거시환경 시나리오 (4개)
    # ======================================================================

    # C1) stagflation_70s : CPI 장기 누적 +4%/yr (실측 2.3% 의 약 1.7배),
    #     unemp +1.5%p, base +2%p. CPI 는 ramp 가 아닌 월복리 trend 로
    #     10년 내내 상승 → 단기 ramp 이후도 필드에 지속적 자극 제공.
    def f_c1(x):
        x[:, col_idx["ecos__cpi_housing"]]       = _trend(cur_cpi, 4.0, L)  # 장기 누적
        x[:, col_idx["ecos__unemployment_rate"]] = _ramp(cur_unemp, cur_unemp + 1.5,  L, 24)
        x[:, col_idx["ecos__base_rate"]]         = _ramp(cur_base_rate,     cur_base_rate     + 2.0, L, 18)
        x[:, col_idx["ecos__mortgage_rate_new"]] = _ramp(cur_mortgage_rate, cur_mortgage_rate + 1.8, L, 18)
        return x
    S["C1_stagflation_70s"] = _make(f_c1)

    # C2) recession_imf_type : 실업 5.5 (1998 = 7.0, 2020 = 5.15 / 보수적),
    #     base +1.5 (외환위기 대응), 거래절벽 sale_sd → 64 (2022.12 실측 최저).
    def f_c2(x):
        x[:, col_idx["ecos__unemployment_rate"]] = _ramp(cur_unemp, 5.5, L, 12)
        x[:, col_idx["ecos__base_rate"]]         = _ramp(cur_base_rate,     cur_base_rate + 1.5, L, 12)
        x[:, col_idx["ecos__mortgage_rate_new"]] = _ramp(cur_mortgage_rate, cur_mortgage_rate + 1.5, L, 12)
        x[:, col_idx["reb__apt_sale_supply_demand"]]   = _ramp(cur_sale_sd, 64.0, L, 12)
        x[:, col_idx["reb__apt_jeonse_supply_demand"]] = _ramp(cur_jeon_sd, 70.0, L, 12)
        x[:, col_idx["reb__apt_sale_index"]]           = _ramp(cur_sale_idx, cur_sale_idx * 0.85, L, 18)
        return x
    S["C2_recession_imf_type"] = _make(f_c2)

    # C3) recovery_v_shape : 코로나형 V자 - 실업 단기 spike 후 정상화 + 금리 급락
    #     근거 : 2020.4 unemp 4.5 → 2022.10 2.9 (V자), base 0.5 유지 후 인상.
    def f_c3(x):
        # 실업 0~12개월 spike, 12~24 정상화
        unemp_path = np.full(L, cur_unemp, dtype=float)
        unemp_path[:12] = np.linspace(cur_unemp, 5.0, 12)
        unemp_path[12:24] = np.linspace(5.0, cur_unemp - 0.5, 12)
        unemp_path[24:] = cur_unemp - 0.5
        x[:, col_idx["ecos__unemployment_rate"]] = unemp_path

        # 금리 : 0~6개월 급락 → 24개월 후부터 점진 인상
        base_path = np.full(L, cur_base_rate, dtype=float)
        base_path[:6] = np.linspace(cur_base_rate, 0.5, 6)
        base_path[6:24] = 0.5
        if L > 24:
            base_path[24:] = np.linspace(0.5, 2.0, L - 24)
        x[:, col_idx["ecos__base_rate"]] = base_path
        x[:, col_idx["ecos__mortgage_rate_new"]] = base_path + 2.0  # spread 2%p
        return x
    S["C3_recovery_v_shape"] = _make(f_c3)

    # C4) goldilocks : 안정성장 - cpi 연 +1.5%p 누적 (일본형 디스인플레 X,
    #     한국 2014-2016 안정기 연 +1–2% 수준), unemp -0.3, base 점진 -0.5
    def f_c4(x):
        x[:, col_idx["ecos__cpi_housing"]]       = _trend(cur_cpi, 1.5, L)  # 장기 완만 상승
        x[:, col_idx["ecos__unemployment_rate"]] = _ramp(cur_unemp, cur_unemp - 0.3, L, 24)
        x[:, col_idx["ecos__base_rate"]]         = _ramp(cur_base_rate,     cur_base_rate - 0.5, L, 18)
        x[:, col_idx["ecos__mortgage_rate_new"]] = _ramp(cur_mortgage_rate, cur_mortgage_rate - 0.5, L, 18)
        return x
    S["C4_goldilocks"] = _make(f_c4)

    # ======================================================================
    # D. 수급 충격 시나리오 (4개)
    # ======================================================================

    # D1) supply_shortage : 3기신도시 지연 - sale_sd→120 (2017 과열 수준),
    #     jeonse_sd→125. 근거 : 2017-2018 평균 supply_demand ≈ 110~120 실측.
    def f_d1(x):
        x[:, col_idx["reb__apt_sale_supply_demand"]]   = _ramp(cur_sale_sd, 120.0, L, 18)
        x[:, col_idx["reb__apt_jeonse_supply_demand"]] = _ramp(cur_jeon_sd, 125.0, L, 18)
        x[:, col_idx["reb__apt_sale_index"]]           = _ramp(cur_sale_idx, cur_sale_idx * 1.10, L, 24)
        return x
    S["D1_supply_shortage"] = _make(f_d1)

    # D2) supply_glut : 입주 폭탄 - sale_sd→80, sale_index -8%
    #     근거 : 2010-2014 침체기 sale_sd 평균 85, sale_idx 75~80.
    def f_d2(x):
        x[:, col_idx["reb__apt_sale_supply_demand"]]   = _ramp(cur_sale_sd,  80.0, L, 18)
        x[:, col_idx["reb__apt_jeonse_supply_demand"]] = _ramp(cur_jeon_sd,  85.0, L, 18)
        x[:, col_idx["reb__apt_sale_index"]]           = _ramp(cur_sale_idx, cur_sale_idx * 0.92, L, 24)
        return x
    S["D2_supply_glut"] = _make(f_d2)

    # D3) jeonse_crisis_2022 : 역전세난 - jeonse_sd 60 (2022.12 실측 최저),
    #     monthly_rent_sd → 115 (월세 수요 폭증).
    def f_d3(x):
        x[:, col_idx["reb__apt_jeonse_supply_demand"]]         = _ramp(cur_jeon_sd, 60.0, L, 12)
        x[:, col_idx["reb__apt_monthly_rent_supply_demand"]]   = _ramp(cur_rent_sd, 115.0, L, 12)
        # 전세가 자체도 약 -10% (2022 사례)
        if "reb__apt_jeonse_avg_price" in col_idx:
            cur_jp = cur_v("reb__apt_jeonse_avg_price")
            x[:, col_idx["reb__apt_jeonse_avg_price"]] = _ramp(cur_jp, cur_jp * 0.90, L, 12)
        return x
    S["D3_jeonse_crisis"] = _make(f_d3)

    # D4) transaction_cliff : 거래절벽 - sale_sd 64 (2022.12 실측 최저)
    def f_d4(x):
        x[:, col_idx["reb__apt_sale_supply_demand"]] = _ramp(cur_sale_sd, 64.0, L, 9)
        return x
    S["D4_transaction_cliff"] = _make(f_d4)

    # ======================================================================
    # E. 역사 재현 시나리오 (5개) - 실제 관측 시점 값 그대로 복원
    # ======================================================================

    # E1) replay_2010_2014_stagnation : 글로벌 금융위기 후 침체기
    #     실측 평균값 (2011-01 ~ 2014-12 평균) 으로 12개월 ramp 후 유지.
    #     base=2.5, mortgage=4.5, cpi=92, unemp=3.4, sale_sd=88, sale_idx=78
    def f_e1(x):
        x[:, col_idx["ecos__base_rate"]]         = _ramp(cur_base_rate,     2.5,  L, 12)
        x[:, col_idx["ecos__mortgage_rate_new"]] = _ramp(cur_mortgage_rate, 4.5,  L, 12)
        x[:, col_idx["ecos__cd_91d_rate"]]       = _ramp(cur_cd91,          2.7,  L, 12)
        x[:, col_idx["ecos__cpi_housing"]]       = _ramp(cur_cpi,           92.0, L, 24)
        x[:, col_idx["ecos__unemployment_rate"]] = _ramp(cur_unemp,         3.4,  L, 12)
        x[:, col_idx["reb__apt_sale_supply_demand"]]   = _ramp(cur_sale_sd,  88.0, L, 12)
        x[:, col_idx["reb__apt_sale_index"]]           = _ramp(cur_sale_idx, 78.0, L, 24)
        _set_cols(x, cov_cols, col_idx, ALL_POLICY_COLS, -1)  # 이 시기 대부분 완화
        return x
    S["E1_replay_2010_2014_stagnation"] = _make(f_e1)

    # E2) replay_2017_2019_regulate : 문재인 규제기 (2017.12 ~ 2019.12 실측 평균)
    #     base=1.5, mortgage=3.4, sale_idx=110, sale_sd=100, 모든 정책 +1
    def f_e2(x):
        x[:, col_idx["ecos__base_rate"]]         = _ramp(cur_base_rate,     1.5,  L, 12)
        x[:, col_idx["ecos__mortgage_rate_new"]] = _ramp(cur_mortgage_rate, 3.4,  L, 12)
        x[:, col_idx["ecos__cd_91d_rate"]]       = _ramp(cur_cd91,          1.7,  L, 12)
        x[:, col_idx["ecos__unemployment_rate"]] = _ramp(cur_unemp,         3.5,  L, 12)
        x[:, col_idx["reb__apt_sale_supply_demand"]]   = _ramp(cur_sale_sd,  100.0, L, 12)
        x[:, col_idx["reb__apt_sale_index"]]           = _ramp(cur_sale_idx, 110.0, L, 18)
        _set_cols(x, cov_cols, col_idx, ALL_POLICY_COLS, +1)
        return x
    S["E2_replay_2017_2019_regulate"] = _make(f_e2)

    # E3) replay_2020_2021_boom : 코로나 폭등기 (2020.6 ~ 2021.12 실측 평균)
    #     base=0.5, mortgage=2.7, sale_idx=160, sale_sd=105, 정책 강화 (+1)
    #     - 정책은 강화였으나 가격 폭등 → 유동성이 규제를 압도한 사례
    def f_e3(x):
        x[:, col_idx["ecos__base_rate"]]         = _ramp(cur_base_rate,     0.5,  L, 9)
        x[:, col_idx["ecos__mortgage_rate_new"]] = _ramp(cur_mortgage_rate, 2.7,  L, 9)
        x[:, col_idx["ecos__cd_91d_rate"]]       = _ramp(cur_cd91,          0.8,  L, 9)
        x[:, col_idx["ecos__unemployment_rate"]] = _ramp(cur_unemp,         4.0,  L, 6)   # 코로나 spike
        x[:, col_idx["reb__apt_sale_supply_demand"]]   = _ramp(cur_sale_sd, 105.0, L, 9)
        x[:, col_idx["reb__apt_jeonse_supply_demand"]] = _ramp(cur_jeon_sd, 110.0, L, 9)
        x[:, col_idx["reb__apt_sale_index"]]           = _ramp(cur_sale_idx, 175.0, L, 18)
        _set_cols(x, cov_cols, col_idx, ALL_POLICY_COLS, +1)
        return x
    S["E3_replay_2020_2021_boom"] = _make(f_e3)

    # E4) replay_2022_shock : 긴축 충격 (2022.1 ~ 2023.6 실측 평균)
    #     base 0.5→3.5, mortgage 2.5→4.6 (21개월), sale_sd 64, sale_idx 145
    def f_e4(x):
        x[:, col_idx["ecos__base_rate"]]         = _ramp(cur_base_rate,     3.5,  L, 21)
        x[:, col_idx["ecos__mortgage_rate_new"]] = _ramp(cur_mortgage_rate, 4.7,  L, 21)
        x[:, col_idx["ecos__cd_91d_rate"]]       = _ramp(cur_cd91,          4.0,  L, 21)
        x[:, col_idx["reb__apt_sale_supply_demand"]]   = _ramp(cur_sale_sd, 64.0, L, 12)
        x[:, col_idx["reb__apt_jeonse_supply_demand"]] = _ramp(cur_jeon_sd, 60.0, L, 12)
        x[:, col_idx["reb__apt_sale_index"]]           = _ramp(cur_sale_idx, 145.0, L, 18)
        _set_cols(x, cov_cols, col_idx, ALL_POLICY_COLS, +1)
        return x
    S["E4_replay_2022_shock"] = _make(f_e4)

    # E5) replay_2023_2025_recovery : 회복기 (2023.7 ~ 2025.12 실측 평균)
    #     base 3.5→2.5, mortgage 4.6→4.2, sale_sd 100, sale_idx 175, 정책 완화
    def f_e5(x):
        x[:, col_idx["ecos__base_rate"]]         = _ramp(cur_base_rate,     2.5,  L, 24)
        x[:, col_idx["ecos__mortgage_rate_new"]] = _ramp(cur_mortgage_rate, 4.2,  L, 24)
        x[:, col_idx["ecos__cd_91d_rate"]]       = _ramp(cur_cd91,          3.0,  L, 24)
        x[:, col_idx["reb__apt_sale_supply_demand"]]   = _ramp(cur_sale_sd,  100.0, L, 12)
        x[:, col_idx["reb__apt_sale_index"]]           = _ramp(cur_sale_idx, 180.0, L, 18)
        _set_cols(x, cov_cols, col_idx, ALL_POLICY_COLS, -1)
        return x
    S["E5_replay_2023_2025_recovery"] = _make(f_e5)

    # =====================================================================
    # F. 장기 구조적 시나리오 (Long-term Structural)
    #   - ramp 가 아닌 _trend (월복리) 사용 → 120개월 내내 자극 지속
    #   - 기존 A~E 가 '한번 변화 후 유지' 인 반면, F 는 '계속 굴러가는' 힘
    #   - 경험적 근거:
    #       * 일본 1990-2010 : CPI 연 +0.3%, 주택가 누적 -50%
    #       * 한국 1970대 : CPI 연 +15% 고인플레
    #       * 통계청 장래가구추계 : 2039 정점 후 가구수 감소 전환
    #       * 한국 1986-1991 : 5년 명목가격 +90% (88올림픽 전후)
    # =====================================================================

    # F1) inflation_grind_higher : CPI 연 +4% 누적 (10년 +48%)
    #     1970대 고인플레 대비 안한 수준, 2010-25 실측(2.3%)의 약 1.7배.
    #     명목 주택가격 자연적 상방 압력.
    def f_f1(x):
        x[:, col_idx["ecos__cpi_housing"]] = _trend(cur_cpi, 4.0, L)
        return x
    S["F1_inflation_grind_higher"] = _make(f_f1)

    # F2) inflation_grind_low : CPI 연 +0.5% (일본형 디스인플레)
    #     명목가격 장기 정체 압력.
    def f_f2(x):
        x[:, col_idx["ecos__cpi_housing"]] = _trend(cur_cpi, 0.5, L)
        return x
    S["F2_inflation_grind_low"] = _make(f_f2)

    # F3) demographic_decline : sale_sd 100→70 을 120개월 전체 ramp
    #     → 매 월 수요 약화 지속. 통계청 2039 정점 이후 가구수 감소 반영.
    #     2020년대 후반 생산가능인구 이미 감소 전환 → 10년 후 가구 형성 감소 분명.
    def f_f3(x):
        x[:, col_idx["reb__apt_sale_supply_demand"]]         = _ramp(cur_sale_sd, 70.0, L, L)
        x[:, col_idx["reb__apt_jeonse_supply_demand"]]       = _ramp(cur_jeon_sd, 72.0, L, L)
        x[:, col_idx["reb__apt_monthly_rent_supply_demand"]] = _ramp(cur_rent_sd, 75.0, L, L)
        return x
    S["F3_demographic_decline"] = _make(f_f3)

    # F4) secular_low_rate : base 2.5→1.0 (60개월 ramp 후 hold)
    #     일본형 장기 저금리 뉴노멈. 자금조달비용 지속 하락.
    def f_f4(x):
        x[:, col_idx["ecos__base_rate"]]         = _ramp(cur_base_rate,     1.0, L, 60)
        x[:, col_idx["ecos__mortgage_rate_new"]] = _ramp(cur_mortgage_rate, 2.8, L, 60)
        x[:, col_idx["ecos__cd_91d_rate"]]       = _ramp(cur_cd91,          1.3, L, 60)
        return x
    S["F4_secular_low_rate"] = _make(f_f4)

    # F5) secular_high_rate : base 2.5→4.5 (60개월 ramp 후 hold)
    #     글로벌 인플레 뉴노멈 (미연준 고대 고정화). 자금조달비용 지속 상승.
    def f_f5(x):
        x[:, col_idx["ecos__base_rate"]]         = _ramp(cur_base_rate,     4.5, L, 60)
        x[:, col_idx["ecos__mortgage_rate_new"]] = _ramp(cur_mortgage_rate, 6.2, L, 60)
        x[:, col_idx["ecos__cd_91d_rate"]]       = _ramp(cur_cd91,          4.7, L, 60)
        return x
    S["F5_secular_high_rate"] = _make(f_f5)

    # F6) japan_lost_decade : 디플레 + 공급과잉 + 저금리 콤보
    #     일본 1990-2010 재현. CPI 연 -1% trend, sale_sd →65, base →0.5
    def f_f6(x):
        x[:, col_idx["ecos__cpi_housing"]]                   = _trend(cur_cpi, -1.0, L)
        x[:, col_idx["reb__apt_sale_supply_demand"]]         = _ramp(cur_sale_sd, 65.0, L, 36)
        x[:, col_idx["reb__apt_jeonse_supply_demand"]]       = _ramp(cur_jeon_sd, 68.0, L, 36)
        x[:, col_idx["reb__apt_monthly_rent_supply_demand"]] = _ramp(cur_rent_sd, 70.0, L, 36)
        x[:, col_idx["ecos__base_rate"]]             = _ramp(cur_base_rate,     0.5, L, 36)
        x[:, col_idx["ecos__mortgage_rate_new"]]     = _ramp(cur_mortgage_rate, 2.3, L, 36)
        x[:, col_idx["ecos__cd_91d_rate"]]           = _ramp(cur_cd91,          0.8, L, 36)
        x[:, col_idx["ecos__unemployment_rate"]]     = _ramp(cur_unemp, cur_unemp + 1.2, L, 36)
        _set_cols(x, cov_cols, col_idx, ALL_POLICY_COLS, -1)
        return x
    S["F6_japan_lost_decade"] = _make(f_f6)

    # F7) korea_1986_1991_boom : 88올림픽급 장기 고호황 재현
    #     5년간 명목가격 +90%. CPI 연 +5% trend, sale_sd →125, sale_idx → 1.5배,
    #     base 연 0.5%p 점진 상승 (경기 과열 대응). 정책 완화 유지.
    def f_f7(x):
        x[:, col_idx["ecos__cpi_housing"]]             = _trend(cur_cpi, 5.0, L)
        x[:, col_idx["reb__apt_sale_supply_demand"]]   = _ramp(cur_sale_sd,  125.0, L, 24)
        x[:, col_idx["reb__apt_jeonse_supply_demand"]] = _ramp(cur_jeon_sd,  120.0, L, 24)
        x[:, col_idx["reb__apt_sale_index"]]           = _ramp(cur_sale_idx, cur_sale_idx * 1.5, L, 60)
        x[:, col_idx["ecos__base_rate"]]               = _ramp(cur_base_rate,     cur_base_rate     + 1.5, L, 60)
        x[:, col_idx["ecos__mortgage_rate_new"]]       = _ramp(cur_mortgage_rate, cur_mortgage_rate + 1.5, L, 60)
        _set_cols(x, cov_cols, col_idx, ALL_POLICY_COLS, -1)
        return x
    S["F7_korea_1986_1991_boom"] = _make(f_f7)

    return S


# 시나리오 카테고리 메타 (차트 / 보고용)
SCENARIO_CATEGORY = {
    "baseline": "Baseline",
    "A1_rate_shock_hike_2022":      "A_Rate",   "A2_rate_gradual_hike":     "A_Rate",
    "A3_rate_gradual_cut":          "A_Rate",   "A4_rate_zero_lower_covid": "A_Rate",
    "B1_policy_full_tight_moonjaein": "B_Policy", "B2_policy_full_loose_current": "B_Policy",
    "B3_demand_loose_tax_tight":      "B_Policy", "B4_demand_tight_tax_loose":    "B_Policy",
    "B5_neutral_supply_focus":        "B_Policy",
    "C1_stagflation_70s":           "C_Macro",  "C2_recession_imf_type":    "C_Macro",
    "C3_recovery_v_shape":          "C_Macro",  "C4_goldilocks":            "C_Macro",
    "D1_supply_shortage":           "D_Supply", "D2_supply_glut":           "D_Supply",
    "D3_jeonse_crisis":             "D_Supply", "D4_transaction_cliff":     "D_Supply",
    "E1_replay_2010_2014_stagnation": "E_Replay", "E2_replay_2017_2019_regulate": "E_Replay",
    "E3_replay_2020_2021_boom":       "E_Replay", "E4_replay_2022_shock":         "E_Replay",
    "E5_replay_2023_2025_recovery":   "E_Replay",
    "F1_inflation_grind_higher": "F_Structural",
    "F2_inflation_grind_low":    "F_Structural",
    "F3_demographic_decline":    "F_Structural",
    "F4_secular_low_rate":       "F_Structural",
    "F5_secular_high_rate":      "F_Structural",
    "F6_japan_lost_decade":      "F_Structural",
    "F7_korea_1986_1991_boom":   "F_Structural",
}


# ---------------------------------------------------------------------------
# [시나리오 구조적 trend prior - 단일 drift fallback]
# Diffusion 의 conditional sampling 만으로는 covariate perturbation 이
# mean-reversion prior 를 이기지 못해 모든 시나리오 median 이 횡보 수렴함.
# 시나리오 의미를 반영한 연환산 drift(%) 를 후처리로 주입한다.
# diffusion 출력은 short-term dynamics / stochastic noise 를 담당,
# 본 drift 가 long-horizon direction 을 담당.
#
# [SCENARIO_ANNUAL_DRIFT_PCT vs SCENARIO_PHASES 관계]
#   - 두 dict 는 상충하지 않고 **fallback 관계**:
#     build_drift_schedule() 에서 SCENARIO_PHASES 에 있으면 phase 사용,
#     없으면 본 dict 의 단일 drift 를 전 horizon 에 적용.
#   - 단일 drift 가 적절한 경우 = "국면 전환이 없는 장기 구조적 추세"
#     → 본 dict 에는 F1~F5 와 baseline 만 등재.
#   - 국면 전환이 있는 모든 시나리오(A/B/C/D/E/F6/F7) 는
#     SCENARIO_PHASES 에 정의 → 본 dict 의 fallback 미사용.
#
# 근거 (장기 단일 추세):
#   - F1/F2 : 인플레율(±) 의 부동산 명목가격 hedge 효과
#   - F3    : 통계청 인구추계 (장기 가구수 감소)
#   - F4/F5 : 뉴노멈 가정 - 영구 저/고 금리는 본질적으로 phase 전환 없음
# ---------------------------------------------------------------------------
SCENARIO_ANNUAL_DRIFT_PCT = {
    "baseline":                       0.0,   # exog carry-forward = drift 없음
    "Baseline_Actual":                0.0,   # 실측 covariate = drift 주입 금지
    # F. 구조적/장기 (phase 전환 없는 영구 추세만 등재)
    "F1_inflation_grind_higher":     +6.0,   # CPI hedge: +4%/yr 인플레 → 명목가 +6%
    "F2_inflation_grind_low":        +3.0,   # 저인플레 안정 상승
    "F3_demographic_decline":        -5.0,   # 인구 감소 수요 둔화 (통계청 추계)
    "F4_secular_low_rate":          +10.0,   # 영구 저금리 뉴노멈
    "F5_secular_high_rate":          -5.0,   # 영구 고금리 뉴노멈
    # A/B/C/D/E/F6/F7 는 SCENARIO_PHASES 에서 정의 (국면 전환 있음)
}


# ---------------------------------------------------------------------------
# [국면 전환 (regime-switching) 시나리오 phase schedule]
# 실물 경제는 하나의 국면이 무한 지속되지 않는다:
#   침체 지속 → 부양책(금리인하, 부동산 규제완화) → 회복
#   과열 지속 → 긴축(LTV 강화, 금리인상) → 조정 → 정상화
# 각 시나리오를 [(지속개월, 연환산 drift %), ...] 시퀀스로 표현하여
# 정책 반응 lag 와 historical episode 길이를 반영한다.
# 합계가 horizon 보다 짧으면 마지막 phase rate 로 채움.
# SCENARIO_PHASES 에 없는 시나리오는 SCENARIO_ANNUAL_DRIFT_PCT 단일값 사용
# (대개 F_ 장기 구조적 시나리오 - 자연스럽게 장기 유지).
# ---------------------------------------------------------------------------
SCENARIO_PHASES = {
    # =====================================================================
    # A. 금리 시나리오 (통화정책 사이클)
    # ---------------------------------------------------------------------
    # 부동산은 금리에 18~24개월 lag 로 반응(주택대출금리→매수심리→실거래).
    # 모든 금리 충격은 결국 중앙은행 정책 전환(완화↔긴축)으로 반전된다.
    # =====================================================================

    # 2022 한국 기준금리 0.5% → 3.5% 급등 (16개월 250bp).
    # 서울 아파트 실거래가 2022.06 정점 → 2023.01 약 -25% (KB).
    # 그 후 2023.07~ 부분 금리 동결/인하 기대로 점진 반등.
    # → 18m 급락(-22%) → 24m 횡보(-3%, 고금리 후유증) → 78m 정상화(+5%, 금리정상화 + 인플레 hedge)
    "A1_rate_shock_hike_2022":    [(18, -22), (24,  -3), (78, +5)],

    # 점진 인상(25bp×수회): 충격 흡수 가능. 2022 같은 급락 없이 24m 완만 하락,
    # 이후 안정. 한국 2010~2012 (2.0→3.25%) 시기 가격 +2~3%/yr 박스권 참고.
    "A2_rate_gradual_hike":       [(24,  -8), (96,  +2)],

    # 점진 인하 사이클: 주담대 금리 하락 → 매수 회복. 2014~2016 한국 사례
    # (3.0→1.25%) 서울 아파트 +8~12%/yr → 이후 정책 규제로 안정화.
    "A3_rate_gradual_cut":        [(24, +10), (96,  +3)],

    # 제로금리(2020.03 0.5%) + 코로나 유동성: 24m 폭등(+22%/yr 실제 관측치).
    # 2022 긴축 전환 시 12m 조정(-10%, 2022 shock 축소판) → 정상화.
    "A4_rate_zero_lower_covid":   [(24, +22), (12, -10), (84, +3)],

    # =====================================================================
    # B. 정책 시나리오 (부동산 규제 vs 완화)
    # ---------------------------------------------------------------------
    # 한국 부동산 정책은 5년 정권 주기로 강한 진폭. 규제 강화기에도
    # 매물 잠김(공급 위축) 으로 가격이 오히려 상승하는 역설이 반복됨.
    # =====================================================================

    # 문재인 정부(2017~2022) 26차례 규제: 의도와 달리 서울 아파트
    # +14%/yr CAGR (실측). 매물 잠김 + 다주택자 회피 + 똘똘한 한 채.
    # 60m 상승 후 차기 정권 규제완화/시장 피로감으로 24m 조정 → 정상.
    "B1_policy_full_tight_moonjaein": [(60, +14), (24,  -5), (36, +3)],

    # 윤정부 완화(LTV/DSR/세제): 단기 매수 회복 12%/yr × 2년,
    # 그 후 공급물량 회복으로 정상 +3%.
    "B2_policy_full_loose_current":   [(24, +12), (96,  +3)],

    # 수요완화(대출↑) + 세제강화(보유세↑): 두 효과 상쇄, 약한 양 +5%/yr,
    # 장기 +2% 박스권. 2018~2019 일부 시기 참고.
    "B3_demand_loose_tax_tight":      [(36,  +5), (84,  +2)],

    # 반대 조합: 수요억제(DSR↑) + 보유세 인하 → 단기 약세, 장기 미온적.
    "B4_demand_tight_tax_loose":      [(24,  -3), (96,  +1)],

    # 공급 중심(3기 신도시 등): 5년간 안정 상승, 입주 본격화 후 둔화.
    "B5_neutral_supply_focus":        [(60,  +4), (60,  +2)],

    # =====================================================================
    # C. 거시충격 시나리오 (글로벌/내부 위기)
    # ---------------------------------------------------------------------
    # 급격한 충격 후 정부 부양책(금리인하 + 재정 + 부동산 규제완화)이
    # 통상 12~24m 내 발동되어 V자 또는 U자 반등을 만듦.
    # =====================================================================

    # 1970년대 미국 스태그플레이션 재현: 고인플레+저성장 36m 침체(-10%/yr),
    # 인플레 둔화 후 부동산이 인플레 hedge 자산으로 24m 회복(+5%), 정상 +2%.
    "C1_stagflation_70s":         [(36, -10), (24,  +5), (60, +2)],

    # 1998 IMF 외환위기 재현: 18m 폭락(-22%/yr, 실제 -25~-30% 수준),
    # IMF 1999~2000 V자 회복(+18%/yr 24m), 이후 2002 정부 부양 +6% 장기.
    "C2_recession_imf_type":      [(18, -22), (24, +18), (78, +6)],

    # 2009 글로벌 금융위기 후 한국 V자(2010~) 사례. 단기 강반등 12m(+18%),
    # 회복 24m(+8%), 장기 정상 +3%.
    "C3_recovery_v_shape":        [(12, +18), (24,  +8), (84, +3)],

    # 미국 1990년대 골디락스(저물가+안정성장): 5년 견조 +12%/yr,
    # 사이클 후반 5년 완만 둔화 +3%.
    "C4_goldilocks":              [(60, +12), (60,  +3)],

    # =====================================================================
    # D. 수급 시나리오 (입주물량 / 거래량)
    # ---------------------------------------------------------------------
    # 한국 아파트 공급은 분양→입주 3년 lag. 부족/과잉 사이클 약 36m.
    # =====================================================================

    # 입주물량 절벽(2024~2025 서울 사례): 36m 강세 +20%/yr,
    # 공급 회복 후 24m 둔화 +5%, 장기 +2% 안정.
    "D1_supply_shortage":         [(36, +20), (24,  +5), (60, +2)],

    # 입주 폭탄(2018~2019 동탄/김포 사례): 36m -12%/yr, 24m 추가 약세 -3%,
    # 공급 흡수 후 0%.
    "D2_supply_glut":             [(36, -12), (24,  -3), (60,  0)],

    # 전세 급등(2020~2021 임대차3법) → 전세 보증금 매매 전환 수요:
    # 18m 매매가 견인 +8%, 안정 +3%, 장기 +2%.
    "D3_jeonse_crisis":           [(18,  +8), (24,  +3), (78, +2)],

    # 거래절벽(2022~2023 양도세 중과): 가격 하방 압력 18m -8%, 정책완화
    # 후 횡보 24m, 정상 회복 +2%.
    "D4_transaction_cliff":       [(18,  -8), (24,   0), (78, +2)],

    # =====================================================================
    # E. 역사 재현 시나리오 (실제 episode 시간축/CAGR 그대로 복원)
    # ---------------------------------------------------------------------
    # 한국 서울 아파트 실거래/KB 통계의 episode 길이와 CAGR 사용.
    # =====================================================================

    # 2010~2014 서울 장기침체: 글로벌 금융위기 여진 + 미분양 누적.
    # 48m 약세 -2%/yr (KB 실측 -1~-3%), 이후 72m 정체 0%.
    "E1_replay_2010_2014_stagnation": [(48,  -2), (72,   0)],

    # 2017~2019 규제기 상승: 36m +16%/yr (실측 +14~18% CAGR),
    # 2019말~2020초 잠시 조정 -10% × 12m, 이후 정상 +4%.
    "E2_replay_2017_2019_regulate":   [(36, +16), (12, -10), (72, +4)],

    # 2020~2021 코로나 폭등 (역사상 최대): 24m +30%/yr (실측 +27~35%),
    # 2022 shock 12m -25%, 2023 바닥 12m 0%, 2024~ 회복 +5% 장기.
    # 4-phase 가 가장 풍부 - 실제 한국이 거쳐온 trajectory.
    "E3_replay_2020_2021_boom":       [(24, +30), (12, -25), (12,  0), (72, +5)],

    # 2022 금리 shock 단독 재현: 12m -25%, 2023~24 점진 회복 +12%/18m,
    # 정상 +4%.
    "E4_replay_2022_shock":           [(12, -25), (18, +12), (90, +4)],

    # 2023~2025 회복기: 24m +12%/yr (실측 1267→1506 ≈ +9% + 가속),
    # 이후 정상 +4%.
    "E5_replay_2023_2025_recovery":   [(24, +12), (96,  +4)],

    # =====================================================================
    # F. 구조적/장기 시나리오 (10년 이상 지속되는 macro 추세)
    # ---------------------------------------------------------------------
    # F1~F5 는 본질적으로 장기 단일 추세 → SCENARIO_ANNUAL_DRIFT_PCT 사용.
    # F6/F7 만 명확한 phase 전환이 역사적으로 검증됨.
    # =====================================================================

    # 일본 잃어버린 10년 (1991~2001): 초반 5년 급락 -8%/yr,
    # 이후 5년 완만 하락 -3% (디플레 정착). 일본 부동산지수 실측 패턴.
    "F6_japan_lost_decade":       [(60,  -8), (60,  -3)],

    # 한국 1986~1991 3저 호황 (저금리/저유가/저환율) + 88올림픽 + 200만호 공약:
    # 5년간 주택가격지수 +250% → +28%/yr (실측). 1991 토지공개념 +
    # 200만호 입주 본격화로 24m 조정 -8%, 이후 정상 +3%.
    "F7_korea_1986_1991_boom":    [(60, +28), (24,  -8), (36, +3)],

    # F1_inflation_grind_higher    (+6%/yr 장기): 인플레 hedge 자산화
    # F2_inflation_grind_low       (+3%/yr 장기): 저인플레 안정 상승
    # F3_demographic_decline       (-5%/yr 장기): 인구감소 수요 둔화
    # F4_secular_low_rate         (+10%/yr 장기): 영구 저금리 가정
    # F5_secular_high_rate         (-5%/yr 장기): 영구 고금리 가정
    # → 단일 drift 가 장기 추세를 더 정확히 표현 (phase 전환 불필요)
}


def build_drift_schedule(scenario_name: str, total_months: int) -> np.ndarray:
    """시나리오의 월별 연환산 drift(%) 배열 반환 (shape (total_months,))."""
    phases = SCENARIO_PHASES.get(scenario_name)
    if phases is None:
        r = SCENARIO_ANNUAL_DRIFT_PCT.get(scenario_name, 0.0)
        return np.full(total_months, r, dtype=float)
    arr = np.zeros(total_months, dtype=float)
    pos = 0
    last_r = 0.0
    for dur, r in phases:
        end = min(pos + int(dur), total_months)
        arr[pos:end] = r
        last_r = r
        pos = end
        if pos >= total_months:
            break
    if pos < total_months:
        arr[pos:] = last_r
    return arr


def apply_scenario_trend(paths_orig: np.ndarray,
                          past_y_last_orig: float,
                          scenario_name: str) -> np.ndarray:
    """원 단위 paths 에 시나리오별 phase schedule trend 시프트 적용.

    shift[k] = past_y_last * (prod_{j=0..k-1} (1 + r_j/12) - 1)
    k=0 → shift=0 (anchor 보존). r_j 는 j 번째 달의 연환산 drift(%).
    국면 전환(예: boom→crash→정상화)이 cumulative compound 로 자연스럽게 반영됨.
    """
    L = paths_orig.shape[1]
    drift_yr = build_drift_schedule(scenario_name, L)          # (L,)
    if not np.any(drift_yr != 0.0):
        return paths_orig
    monthly_mult = 1.0 + drift_yr / 100.0 / 12.0               # (L,)
    cumprod = np.cumprod(monthly_mult)                         # (L,)
    shift_mult = np.empty(L, dtype=float)
    shift_mult[0] = 1.0
    shift_mult[1:] = cumprod[:-1]
    shift = past_y_last_orig * (shift_mult - 1.0)              # (L,)
    return paths_orig + shift[None, :]


# ===========================================================================
# 9. 후처리 유틸리티 (정규화 역변환, 지표, 차트)
# ===========================================================================
def inverse_target(arr_scaled: np.ndarray,
                   target_scaler: RobustScaler) -> np.ndarray:
    """RobustScaler 역변환 → 원 단위 가격(원/㎡)."""
    flat = arr_scaled.reshape(-1, 1)
    return target_scaler.inverse_transform(flat).reshape(arr_scaled.shape)


def summarize_paths(paths: np.ndarray, idx: pd.DatetimeIndex,
                    target_scaler: RobustScaler) -> pd.DataFrame:
    """100 path → (median, q10, q90, mean, std) timeline DataFrame."""
    orig = inverse_target(paths, target_scaler)   # (N, L)
    df = pd.DataFrame({
        "timestamp" : idx,
        "median"    : np.median(orig, axis=0),
        "q10"       : np.quantile(orig, 0.10, axis=0),
        "q90"       : np.quantile(orig, 0.90, axis=0),
        "mean"      : orig.mean(axis=0),
        "std"       : orig.std(axis=0),
    })
    return df


def compute_metrics(true_vals: np.ndarray,
                    median_pred: np.ndarray) -> dict:
    mae  = mean_absolute_error(true_vals, median_pred)
    rmse = math.sqrt(mean_squared_error(true_vals, median_pred))
    mape = float(np.mean(np.abs((true_vals - median_pred) /
                                np.where(true_vals == 0, 1e-9, true_vals)))) * 100
    r2   = r2_score(true_vals, median_pred) if len(true_vals) >= 2 else np.nan
    return {"MAE": mae, "RMSE": rmse, "MAPE_%": mape, "R2": r2}


# ===========================================================================
# 10. 시나리오 (1)+(2) : 백테스트 (2025-01 ~ 2025-12)
# ===========================================================================
def scenario_backtest(full_df: pd.DataFrame) -> dict:
    """2010 ~ 2024.12 학습 → 2025 12개월 × 100 path 생성, 실측 비교."""
    print("\n=== Scenario 1+2: Backtest 2025-01 ~ 2025-12 ===")
    train = full_df.loc[:TRAIN_END]
    valid = full_df.loc[VALID_START:VALID_END]

    target_scaler, cov_scaler, cov_cols = fit_scalers(train)
    y_tr, X_tr = transform_df(train, target_scaler, cov_scaler, cov_cols)
    y_va, X_va = transform_df(valid, target_scaler, cov_scaler, cov_cols)

    # 학습 윈도우
    p_y, p_X, f_X, f_y = build_sliding_windows(
        y_tr, X_tr, PAST_LEN, BACKTEST_HORIZON)
    print(f"[Backtest] training windows = {len(p_y)} "
          f"(past={PAST_LEN}, horizon={BACKTEST_HORIZON})")

    # recency + volatility 가중치 (2020+ 급등 구간 강조)
    w_np = compute_window_weights(f_y[:, :, 0] if f_y.ndim == 3 else f_y)
    print(f"[Backtest] weight stats : min={w_np.min():.3f} "
          f"max={w_np.max():.3f} mean={w_np.mean():.3f} "
          f"(last 12 windows mean={w_np[-12:].mean():.3f})")

    ds = TensorDataset(torch.from_numpy(p_y), torch.from_numpy(p_X),
                       torch.from_numpy(f_X), torch.from_numpy(f_y),
                       torch.from_numpy(w_np))
    # WeightedRandomSampler : 가중치 비례로 윈도우 재샘플 → 최근/변동 구간 oversample
    sampler = WeightedRandomSampler(weights=torch.from_numpy(w_np).double(),
                                    num_samples=len(w_np), replacement=True)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, sampler=sampler,
                        num_workers=0, drop_last=False)

    schedule = DiffusionSchedule()
    model = SSSDDiffusionNet(cov_dim=X_tr.shape[1]).to(DEVICE)
    print(f"[Backtest] model params = {sum(p.numel() for p in model.parameters()):,}")

    hist = train_diffusion(model, schedule, loader,
                           n_epochs=N_EPOCHS, lr=LEARNING_RATE,
                           log_csv=OUT_DIR / "backtest_train_loss.csv")

    # 추론 : past = train 마지막 PAST_LEN 개월
    past_y_np = y_tr[-PAST_LEN:]
    past_X_np = X_tr[-PAST_LEN:]
    pred_idx  = valid.index
    truth     = valid[TARGET_COL].values

    # ------------------------------------------------------------------
    # [국면별 시나리오 백테스트]
    # 기존 구현은 fut_X = X_va (실측 covariate) 단일 경로로만 sampling →
    # diffusion noise variation 만 보이고 구조적(금리/정책/매크로/구조) 분기는
    # 검증 단계에 노출되지 않아 100 path 모두 baseline 근처로 수렴(횡보).
    #
    # 개선 : future_X 에 대해 What-if 시나리오 22종 + Baseline_Actual(=X_va)
    # 을 모두 평가. 메트릭/compare 는 Baseline_Actual 기준(실측 covariate)으로
    # 산출하여 정확도 비교의 공정성을 유지하면서, 차트에는 모든 시나리오 median
    # 을 fan 으로 표시 → "실측이 어느 국면 가설에 가장 가까웠는가" 사후 진단 가능.
    # ------------------------------------------------------------------
    scenarios = build_whatif_covariates(X_va, cov_cols, cov_scaler)
    scenarios = {"Baseline_Actual": X_va, **scenarios}   # 실측 covariate 우선

    all_long_rows  = []
    summary_rows   = {}
    metrics_rows   = []
    # 시나리오 trend 시프트 기준점 (anchor 와 동일한 origin)
    past_y_last_orig = float(inverse_target(
        past_y_np[-1:].reshape(1, 1), target_scaler).ravel()[0])
    for name, fut_X_np in scenarios.items():
        print(f"  [backtest sampling] scenario = {name}")
        paths = sample_scenarios(model, schedule, past_y_np, past_X_np,
                                 fut_X_np, n_scenarios=N_SCENARIOS)
        paths_orig = inverse_target(paths, target_scaler)
        # 시나리오 의미를 반영한 구조적 trend 백본 주입 (k=0 shift=0 → anchor 보존)
        paths_orig = apply_scenario_trend(paths_orig, past_y_last_orig, name)
        # trend 적용된 원-단위 paths 로부터 summary 재계산
        summary = pd.DataFrame({
            "timestamp": pred_idx,
            "q10":  np.quantile(paths_orig, 0.10, axis=0),
            "median": np.quantile(paths_orig, 0.50, axis=0),
            "q90":  np.quantile(paths_orig, 0.90, axis=0),
            "mean": paths_orig.mean(axis=0),
            "std":  paths_orig.std(axis=0),
        })
        summary["scenario"] = name
        summary_rows[name]  = summary

        for i, p in enumerate(paths_orig):
            for ts, v in zip(pred_idx, p):
                all_long_rows.append({
                    "scenario": name, "scenario_id": i,
                    "timestamp": ts, "value": float(v),
                })

        m = compute_metrics(truth, summary["median"].values)
        m = {"scenario": name,
             "category": SCENARIO_CATEGORY.get(name, "Baseline"), **m}
        metrics_rows.append(m)

    # 저장 : long path, per-scenario summary, per-scenario metrics
    pd.DataFrame(all_long_rows).to_csv(
        OUT_DIR / "backtest_paths_long.csv", index=False)
    pd.concat(summary_rows.values(), ignore_index=True).to_csv(
        OUT_DIR / "backtest_summary_by_scenario.csv", index=False)
    metrics_df = pd.DataFrame(metrics_rows).sort_values("MAE")
    metrics_df.to_csv(OUT_DIR / "backtest_metrics_by_scenario.csv", index=False)

    # 기본(Baseline_Actual) 기준의 단일 compare/metrics (기존 호환)
    base_summary = summary_rows["Baseline_Actual"].drop(columns=["scenario"])
    compare = base_summary.copy()
    compare["actual"] = truth
    compare.to_csv(OUT_DIR / "backtest_compare.csv", index=False)
    base_metrics = compute_metrics(truth, base_summary["median"].values)
    pd.DataFrame([base_metrics]).to_csv(OUT_DIR / "backtest_metrics.csv",
                                        index=False)
    print(f"[Backtest] baseline metrics = {base_metrics}")
    print("[Backtest] top-5 best-matching scenarios:")
    print(metrics_df.head(5).to_string(index=False))

    # 차트 : 2010년 이후 전체 실측 + 시나리오 median fan + baseline 밴드
    hist_idx  = train.index
    hist_orig = inverse_target(y_tr, target_scaler)

    cat_colors = {
        "Baseline":    "#1f77b4",
        "A_Rate":      "#ff7f0e",
        "B_Policy":    "#2ca02c",
        "C_Macro":     "#d62728",
        "D_Supply":    "#9467bd",
        "E_Replay":    "#8c564b",
        "F_Structural":"#e377c2",
    }

    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(hist_idx, hist_orig, color="black", linewidth=1.2, label="historical")
    # 시나리오별 median 을 카테고리 색으로 표시
    drawn_cats = set()
    for name, summary in summary_rows.items():
        if name == "Baseline_Actual":
            continue
        cat = SCENARIO_CATEGORY.get(name, "Other")
        col = cat_colors.get(cat, "grey")
        lbl = cat if cat not in drawn_cats else None
        drawn_cats.add(cat)
        ax.plot(pred_idx, summary["median"], color=col, alpha=0.55,
                linewidth=1.0, label=lbl)
    # Baseline_Actual : 굵게 + 밴드
    base = summary_rows["Baseline_Actual"]
    ax.plot(pred_idx, base["median"], color="blue", linewidth=2.2,
            label="Baseline_Actual median")
    ax.fill_between(pred_idx, base["q10"], base["q90"],
                    color="blue", alpha=0.15, label="Baseline 10–90%")
    ax.plot(pred_idx, truth, color="red", linewidth=2.2, label="actual")
    ax.axvline(pd.Timestamp(VALID_START), color="grey", linestyle="--",
               linewidth=0.8, label="forecast start")
    ax.set_title(f"Backtest 2025 — {len(scenarios)} scenarios × {N_SCENARIOS} paths "
                 f"(category medians; baseline=실측 covariate)")
    ax.set_ylabel("apt sale avg price")
    ax.legend(loc="best", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "backtest_chart.png", dpi=130)
    plt.close(fig)

    return {"metrics": base_metrics, "summary": base_summary,
            "metrics_by_scenario": metrics_df, "history": hist}


# ===========================================================================
# 11. 시나리오 (3) : 미래 10년 × What-if 8종 × 100 path
# ===========================================================================
def scenario_future(full_df: pd.DataFrame) -> dict:
    """전체 데이터(2010~2025)로 재학습 → 2026~2035 What-if 8종 × 100 path."""
    print("\n=== Scenario 3: Future 2026-01 ~ 2035-12 with What-if ===")

    target_scaler, cov_scaler, cov_cols = fit_scalers(full_df)
    y_all, X_all = transform_df(full_df, target_scaler, cov_scaler, cov_cols)

    # 학습 윈도우 (horizon 은 backtest 와 동일 12 사용 - long horizon 직접 학습은
    # CPU 환경에서 비용 과대 → 12개월 단위 학습 모델로 sliding 추론도 가능하나
    # 본 파이프라인은 baseline 단순화를 위해 single-shot 120개월 학습 채택.
    # ※ 미래 horizon=120 로도 학습 가능하지만 윈도우 수가 너무 적어
    # generalization 이 떨어짐. 절충 : horizon=BACKTEST_HORIZON 로 학습한 모델을
    # 호출 시 horizon=FUTURE_HORIZON 로 확장.
    # → 본 구현은 horizon 가변 모델(1D conv) 의 특성을 활용해 FUTURE_HORIZON 로
    #   바로 학습하지 않고 BACKTEST_HORIZON 로 학습한 뒤 inference 시 길이 확장.
    p_y, p_X, f_X, f_y = build_sliding_windows(
        y_all, X_all, PAST_LEN, BACKTEST_HORIZON)
    print(f"[Future] training windows = {len(p_y)} "
          f"(past={PAST_LEN}, horizon={BACKTEST_HORIZON})")

    # recency + volatility 가중치 (2020+ 급등 구간 강조)
    w_np = compute_window_weights(f_y[:, :, 0] if f_y.ndim == 3 else f_y)
    print(f"[Future] weight stats : min={w_np.min():.3f} "
          f"max={w_np.max():.3f} mean={w_np.mean():.3f} "
          f"(last 12 windows mean={w_np[-12:].mean():.3f})")

    ds = TensorDataset(torch.from_numpy(p_y), torch.from_numpy(p_X),
                       torch.from_numpy(f_X), torch.from_numpy(f_y),
                       torch.from_numpy(w_np))
    sampler = WeightedRandomSampler(weights=torch.from_numpy(w_np).double(),
                                    num_samples=len(w_np), replacement=True)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, sampler=sampler,
                        num_workers=0, drop_last=False)

    schedule = DiffusionSchedule()
    model = SSSDDiffusionNet(cov_dim=X_all.shape[1]).to(DEVICE)

    hist = train_diffusion(model, schedule, loader,
                           n_epochs=N_EPOCHS, lr=LEARNING_RATE,
                           log_csv=OUT_DIR / "future_train_loss.csv")

    # baseline future covariate : 마지막 관측치 carry-forward (FUTURE_HORIZON 길이)
    fut_idx = pd.date_range(FUTURE_START, FUTURE_END, freq="MS")
    L_fut = len(fut_idx)
    last_X = X_all[-1:]                                 # (1, C)
    baseline_fut_X = np.repeat(last_X, L_fut, axis=0)   # (L_fut, C)

    scenarios = build_whatif_covariates(baseline_fut_X, cov_cols, cov_scaler)

    # ── 각 시나리오의 미래 covariate 경로를 원 단위로 복원하여 저장
    #    (어떤 시나리오가 어떤 변수 값을 갖는지 후속 분석/감사 가능)
    cov_long_rows = []
    for name, sc_arr in scenarios.items():
        orig = cov_scaler.inverse_transform(sc_arr)
        for ti, ts in enumerate(fut_idx):
            for ci, c in enumerate(cov_cols):
                cov_long_rows.append({
                    "scenario":  name,
                    "timestamp": ts,
                    "feature":   c,
                    "value":     float(orig[ti, ci]),
                })
    pd.DataFrame(cov_long_rows).to_csv(
        OUT_DIR / "future_scenario_covariates.csv", index=False)

    # 시나리오 카테고리 메타 저장
    meta = pd.DataFrame([
        {"scenario": n, "category": SCENARIO_CATEGORY.get(n, "Other")}
        for n in scenarios.keys()
    ])
    meta.to_csv(OUT_DIR / "future_scenario_meta.csv", index=False)

    past_y_np = y_all[-PAST_LEN:]
    past_X_np = X_all[-PAST_LEN:]
    past_y_last_orig = float(inverse_target(
        past_y_np[-1:].reshape(1, 1), target_scaler).ravel()[0])

    all_long_rows = []
    summary_rows  = {}

    for name, fut_X_np in scenarios.items():
        print(f"  [sampling] scenario = {name}")
        paths = sample_scenarios(model, schedule, past_y_np, past_X_np,
                                 fut_X_np, n_scenarios=N_SCENARIOS)
        paths_orig = inverse_target(paths, target_scaler)
        # 시나리오 의미 반영 trend 백본 (long-horizon 분기)
        paths_orig = apply_scenario_trend(paths_orig, past_y_last_orig, name)
        summary = pd.DataFrame({
            "timestamp": fut_idx,
            "q10":  np.quantile(paths_orig, 0.10, axis=0),
            "median": np.quantile(paths_orig, 0.50, axis=0),
            "q90":  np.quantile(paths_orig, 0.90, axis=0),
            "mean": paths_orig.mean(axis=0),
            "std":  paths_orig.std(axis=0),
        })
        summary["scenario"] = name
        summary_rows[name] = summary

        # 100 path long form
        for i, p in enumerate(paths_orig):
            for ts, v in zip(fut_idx, p):
                all_long_rows.append({
                    "scenario": name, "scenario_id": i,
                    "timestamp": ts, "value": v,
                })

    # 저장
    long_df = pd.DataFrame(all_long_rows)
    long_df.to_csv(OUT_DIR / "future_paths_long.csv", index=False)

    summary_all = pd.concat(summary_rows.values(), ignore_index=True)
    summary_all.to_csv(OUT_DIR / "future_summary.csv", index=False)

    # 시나리오별 차트 (medians 한 장 비교)
    fig, ax = plt.subplots(figsize=(12, 6))
    history_y = inverse_target(y_all, target_scaler)
    ax.plot(full_df.index, history_y, color="black",
            linewidth=1.5, label="history (2010-2025)")

    cmap = plt.get_cmap("tab20")
    for k, (name, summ) in enumerate(summary_rows.items()):
        ax.plot(summ["timestamp"], summ["median"],
                color=cmap(k % 20), linewidth=1.5, label=name, alpha=0.85)
    ax.set_title(f"Future 2026-2035 — {N_SCENARIOS} paths/scenario "
                 f"({len(scenarios)} scenarios)")
    ax.set_ylabel("apt sale avg price")
    ax.legend(loc="upper left", fontsize=7, ncol=2, bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    fig.savefig(OUT_DIR / "future_chart_all_medians.png", dpi=130,
                bbox_inches="tight")
    plt.close(fig)

    # 카테고리별 subplot (A/B/C/D/E + baseline 비교)
    cats: dict[str, list[str]] = {}
    for name in summary_rows.keys():
        c = SCENARIO_CATEGORY.get(name, "Other")
        cats.setdefault(c, []).append(name)
    cat_order = ["A_Rate", "B_Policy", "C_Macro", "D_Supply", "E_Replay"]
    n_cats = len(cat_order)
    fig, axes = plt.subplots(n_cats, 1, figsize=(13, 3.2 * n_cats), sharex=True)
    base_summ = summary_rows["baseline"]
    for ax_i, cat_name in zip(axes, cat_order):
        names = cats.get(cat_name, [])
        ax_i.plot(full_df.index[-60:], history_y[-60:],
                  color="black", linewidth=1.2, label="history(last 5y)")
        ax_i.plot(base_summ["timestamp"], base_summ["median"],
                  color="grey", linewidth=1.2, linestyle="--", label="baseline")
        for k, name in enumerate(names):
            s = summary_rows[name]
            color = cmap(k % 20)
            ax_i.plot(s["timestamp"], s["median"], color=color,
                      linewidth=1.6, label=name)
            ax_i.fill_between(s["timestamp"], s["q10"], s["q90"],
                              color=color, alpha=0.10)
        ax_i.set_title(cat_name)
        ax_i.set_ylabel("price")
        ax_i.legend(loc="upper left", fontsize=7, ncol=2)
        ax_i.grid(alpha=0.3)
    fig.suptitle(f"Future scenarios by category ({N_SCENARIOS} paths each)",
                 fontsize=13, y=1.001)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "future_chart_by_category.png", dpi=130,
                bbox_inches="tight")
    plt.close(fig)

    # baseline 시나리오만 별도 100-path 차트
    fig, ax = plt.subplots(figsize=(12, 6))
    base_paths = long_df[long_df["scenario"] == "baseline"]
    for sid, grp in base_paths.groupby("scenario_id"):
        ax.plot(grp["timestamp"], grp["value"],
                color="grey", alpha=0.08, linewidth=0.7)
    base_summ = summary_rows["baseline"]
    ax.plot(base_summ["timestamp"], base_summ["median"],
            color="blue", linewidth=2, label="baseline median")
    ax.fill_between(base_summ["timestamp"], base_summ["q10"], base_summ["q90"],
                    color="blue", alpha=0.15, label="10–90% band")
    ax.plot(full_df.index, history_y, color="black",
            linewidth=1.2, label="history")
    ax.set_title(f"Future baseline — all {N_SCENARIOS} sampled paths")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "future_baseline_paths.png", dpi=130)
    plt.close(fig)

    return {"summary": summary_all, "history": hist}


# ===========================================================================
# 12. 하이퍼파라미터 / 시나리오 정의 저장
# ===========================================================================
def save_hyperparams() -> None:
    hp = {
        "PAST_LEN": PAST_LEN,
        "BACKTEST_HORIZON": BACKTEST_HORIZON,
        "FUTURE_HORIZON": FUTURE_HORIZON,
        "DIFF_TIMESTEPS": DIFF_TIMESTEPS,
        "BETA_START": BETA_START, "BETA_END": BETA_END,
        "RES_CHANNELS": RES_CHANNELS, "SKIP_CHANNELS": SKIP_CHANNELS,
        "N_RES_LAYERS": N_RES_LAYERS, "COND_EMB_DIM": COND_EMB_DIM,
        "DIFF_STEP_EMB_DIM": DIFF_STEP_EMB_DIM,
        "BATCH_SIZE": BATCH_SIZE, "N_EPOCHS": N_EPOCHS,
        "LEARNING_RATE": LEARNING_RATE, "GRAD_CLIP": GRAD_CLIP,
        "N_SCENARIOS": N_SCENARIOS, "SEED": SEED,
        "TRAIN_END": TRAIN_END, "VALID_START": VALID_START,
        "VALID_END": VALID_END,
        "FUTURE_START": FUTURE_START, "FUTURE_END": FUTURE_END,
    }
    pd.DataFrame([hp]).to_csv(OUT_DIR / "hyperparameters.csv", index=False)
    with open(OUT_DIR / "hyperparameters.json", "w", encoding="utf-8") as f:
        json.dump(hp, f, indent=2, default=str)


# ===========================================================================
# 13. main
# ===========================================================================
def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("[Step A] Loading & aggregating data ...")
    raw = load_and_aggregate(DATA_CSV)
    raw.to_csv(OUT_DIR / "00_seoul_monthly_raw.csv")

    print("[Step A] Cleaning missing + policy normalize ...")
    clean = clean_missing_and_policy(raw)
    clean.to_csv(OUT_DIR / "01_seoul_monthly_clean.csv")

    print("[Step A] ADF diagnostic ...")
    run_adf_report(clean, OUT_DIR / "02_adf_report.csv")

    save_hyperparams()

    print("[Step B/C/D] Scenario 1+2 backtest ...")
    scenario_backtest(clean)

    print("[Step E] Scenario 3 future + what-if ...")
    scenario_future(clean)

    print("\n=== DONE ===  outputs in:", OUT_DIR)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        err = traceback.format_exc()
        print(err)
        try:
            (OUT_DIR / "_error_log.txt").write_text(err, encoding="utf-8")
        except Exception:
            pass
        sys.exit(1)
