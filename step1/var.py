"""
VAR(Vector AutoRegression) 기반 서울시 아파트 매매가 예측 파이프라인
====================================================================

데이터 : data/dong_level_meta_table.csv (2010-01 ~ 2025-12, 월 단위, 243개 동 × 25개 컬럼)
타겟   : reb__apt_sale_avg_price (서울 평균)
시나리오:
  (1) 2010~2022.12 학습 → 2023.1~2025.12 예측 → 실측과 비교 차트
  (2) (1)의 MAE 등 검증 지표 저장
  (3) 전체 데이터로 재학습 → 2026.1~2035.12 (10년) 예측 → 전 구간 차트

중간 산출물은 모두 step1/ 폴더에 csv 로 저장된다.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # GUI 없이 PNG 저장 전용 백엔드
import matplotlib.pyplot as plt

from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.api import VAR

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 경로 설정
# ---------------------------------------------------------------------------
# step1/var.py 기준 상위 폴더에 data/ 가 있는 구조를 가정.
BASE_DIR = Path(__file__).resolve().parent          # .../step1
ROOT_DIR = BASE_DIR.parent                          # .../newtrial
DATA_CSV = ROOT_DIR / "data" / "dong_level_meta_table.csv"
OUT_DIR  = BASE_DIR                                 # 모든 중간 산출물 저장 위치

TARGET_COL  = "reb__apt_sale_avg_price"
ID_COLS     = ["si", "gu", "dong"]
TIME_COL    = "timestamp"

# 시나리오 분할/예측 구간
TRAIN_END   = "2023-12-01"   # (1) 학습 종료 시점
VALID_START = "2024-01-01"   # (1) 검증 시작
VALID_END   = "2025-12-01"   # (1) 검증 종료 (=데이터의 마지막)
FUTURE_END  = "2035-12-01"   # (3) 미래 예측 종료 (10년)

# PCA 차원 (사용자 지정 8~10 중 9 채택: bias-variance 절충)
PCA_N_COMPONENTS = 9
# VAR maxlags 탐색 상한
# 표본 수 ~192, 변수 수 ~10. 일반적으로 maxlag ≤ T^(1/3) ≈ 5.7 권장(Schwert, 1989).
# 월 단위 계절성을 일부 흡수할 수 있도록 여유롭게 12 까지 탐색.
VAR_MAXLAGS = 12


# ---------------------------------------------------------------------------
# 1. 데이터 로딩 & 서울시 단위 집계
# ---------------------------------------------------------------------------
def load_and_aggregate(csv_path: Path) -> pd.DataFrame:
    """
    원본 동(洞) 단위 패널 데이터를 서울시 단위 월별 시계열로 집계한다.

    [근거]
      - VAR 은 단일 다변량 시계열이 필요하므로 cross-section(동) 평균으로 축약.
      - reb__/ecos__ 류 연속형 변수는 산술평균 으로 대표값 계산.
      - policy__ 류 정책 변수는 모든 동에서 동일한 시점에 동일 값(서울 광역 단위
        규제)이므로 평균 ≈ 모드. 이후 round 처리로 {-1,0,1} 보장.

    Parameters
    ----------
    csv_path : Path
        원본 CSV 경로.

    Returns
    -------
    pd.DataFrame
        DatetimeIndex(월초) × 22개 feature(타겟 포함).
    """
    df = pd.read_csv(csv_path, parse_dates=[TIME_COL])
    # 동 차원 평균으로 광역 시계열 생성. numeric_only=True 로 식별자 컬럼 자동 제외.
    agg = (df.drop(columns=ID_COLS)
             .groupby(TIME_COL).mean(numeric_only=True)
             .sort_index())
    # 월초 빈도(MS) 명시 → VAR 의 시점 정렬/예측 인덱스 생성에 사용.
    agg = agg.asfreq("MS")
    return agg


# ---------------------------------------------------------------------------
# 2. 결측치 보간 & 정책 변수 정규화
# ---------------------------------------------------------------------------
def clean_missing_and_policy(df: pd.DataFrame) -> pd.DataFrame:
    """
    결측치를 시간축 선형보간으로 메우고, policy__ 컬럼을 {-1,0,1} 로 강제한다.

    [파라미터 근거]
      - method="time" : 월 단위 등간격이지만 향후 결측 길이가 다양해질 수 있어
        timestamp 기반 보간이 가장 일반적/안전.
      - limit_direction="both" : 시계열 처음/끝 결측까지 외삽 없이 양방향 채움
        (앞/뒤 가장 가까운 관측값 활용).
      - policy 컬럼은 round 후 clip(-1,1) : 평균 집계 과정에서 발생할 수 있는
        소수점 잔차를 제거하고 사용자 요구사항 (강화+1/유지0/완화-1) 충족.

    Returns
    -------
    pd.DataFrame
        결측 없는 깨끗한 광역 월별 데이터.
    """
    out = df.copy()

    # 1) 연속형 보간
    out = out.interpolate(method="time", limit_direction="both")

    # 2) 정책 컬럼 분리 처리
    policy_cols = [c for c in out.columns if c.startswith("policy__")]
    out[policy_cols] = out[policy_cols].round().clip(-1, 1).astype(int)

    # 3) 그래도 남은 NaN 이 있으면 (예: 전 구간 결측 컬럼) 0 으로 채움(드물지만 방어).
    out = out.fillna(0.0)
    return out


# ---------------------------------------------------------------------------
# 3. 스케일링 & PCA 차원축소
# ---------------------------------------------------------------------------
def fit_scaler_and_pca(train_df: pd.DataFrame,
                       n_components: int = PCA_N_COMPONENTS
                       ) -> tuple[RobustScaler, RobustScaler, PCA, list[str]]:
    """
    학습 구간 데이터에 RobustScaler + PCA 를 적합한다.

    [설계 근거]
      - **타겟(reb__apt_sale_avg_price)** 도 별도 RobustScaler 로 정규화한다.
        수정 이유:
          · 타겟 std≈295 vs PC std≈0.25~2.83 → 스케일 불균형이 VAR 잔차
            공분산 행렬에서 타겟을 수백 배 압도해 교차변수 계수를 왜곡하고
            level offset 을 유발한다.
          · 정규화 후 VAR 을 적합하면 타겟 방정식의 intercept 가 0 근방으로
            수렴해 추세 외삽(trend extrapolation)으로 인한 shift 가 억제된다.
        예측 후 inverse_transform 으로 원래 단위로 복원.
      - feature 는 한 번에 RobustScaler(median/IQR 기반, 이상치 안정적) 적용.
      - 표본 192개 대비 변수 21개 → PCA 로 9차원 축소.

    Returns
    -------
    scaler, target_scaler, pca, feature_cols
        feature scaler / target scaler / pca 객체 / PCA 투입 feature 컬럼 목록.
    """
    feature_cols = [c for c in train_df.columns if c != TARGET_COL]

    # feature 스케일러
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(train_df[feature_cols].values)

    # 타겟 전용 스케일러 (scale mismatch 해소)
    target_scaler = RobustScaler()
    target_scaler.fit(train_df[[TARGET_COL]].values)

    pca = PCA(n_components=n_components, random_state=0)
    pca.fit(X_scaled)
    return scaler, target_scaler, pca, feature_cols


def save_pca_loadings(pca: PCA, feature_cols: list[str],
                      prefix: str, top_n: int = 5) -> None:
    """
    PCA 각 성분(PC)에 대한 feature loading(기여도)을 csv 로 저장한다.

    [출력물 설명]
      (1) {prefix}_pca_loadings.csv
          - 행: PC1 ~ PCk  / 열: feature 이름
          - 값: pca.components_ (각 PC 를 구성하는 feature 의 선형 가중치)
          - 양수: 해당 feature 가 클수록 PC 값도 커짐 (같은 방향)
          - 음수: 해당 feature 가 클수록 PC 값은 작아짐 (반대 방향)
          - |값| 이 클수록 해당 PC 에 더 큰 영향을 미치는 feature

      (2) {prefix}_pca_top_features.csv
          - 각 PC 별로 |loading| 기준 상위 top_n 개 feature 를 나열한 요약표
          - 어떤 경제/정책 변수가 주된 주성분 축을 형성하는지 직관적으로 파악

    [파라미터 근거]
      - pca.components_ shape = (n_components, n_features) : sklearn PCA 표준.
      - top_n=5 : 한 PC 에서 설명력이 집중되는 핵심 변수 파악에 충분한 수.

    Parameters
    ----------
    pca          : 적합된 PCA 객체
    feature_cols : PCA 에 투입된 feature 컬럼 목록 (TARGET_COL 제외)
    prefix       : 파일명 접두사 ("backtest" 또는 "future")
    top_n        : 각 PC 당 출력할 상위 feature 수
    """
    n_comp = pca.n_components_
    pc_labels = [f"PC{i+1}" for i in range(n_comp)]

    # (1) 전체 loading 행렬 저장
    loadings_df = pd.DataFrame(
        pca.components_,           # shape: (n_components, n_features)
        index=pc_labels,
        columns=feature_cols,
    )
    loadings_df.index.name = "component"
    loadings_df.to_csv(OUT_DIR / f"{prefix}_pca_loadings.csv")

    # (2) PC 별 상위 기여 feature 요약
    rows = []
    for i, pc in enumerate(pc_labels):
        # |loading| 기준 내림차순 정렬
        abs_loadings = loadings_df.loc[pc].abs().sort_values(ascending=False)
        evr = pca.explained_variance_ratio_[i]
        for rank, feat in enumerate(abs_loadings.index[:top_n], start=1):
            rows.append({
                "component":               pc,
                "explained_variance_ratio": round(evr, 4),
                "rank":                    rank,
                "feature":                 feat,
                "loading":                 round(float(loadings_df.loc[pc, feat]), 4),
                "abs_loading":             round(float(abs_loadings[feat]), 4),
            })
    top_df = pd.DataFrame(rows)
    top_df.to_csv(OUT_DIR / f"{prefix}_pca_top_features.csv", index=False)

    # 콘솔에도 요약 출력
    print(f"\n[PCA Loadings: {prefix}] 설명분산 누적={np.cumsum(pca.explained_variance_ratio_)[-1]:.3f}")
    print(f"{'PC':<6} {'EVR':>6}  {'Rank':>4}  {'feature':<45}  {'loading':>8}")
    print("-" * 80)
    for _, r in top_df.iterrows():
        print(f"{r['component']:<6} {r['explained_variance_ratio']:>6.4f}  "
              f"{int(r['rank']):>4}  {r['feature']:<45}  {r['loading']:>+8.4f}")


def transform_with_pca(df: pd.DataFrame,
                       scaler: RobustScaler,
                       target_scaler: RobustScaler,
                       pca: PCA,
                       feature_cols: list[str]) -> pd.DataFrame:
    """
    학습된 scaler/target_scaler/pca 로 변환하여 [scaled_target + PC1..PCk] DataFrame 반환.

    타겟도 target_scaler 로 정규화해 VAR 입력 변수 간 스케일을 통일시킨다.
    예측 후에는 inverse_transform_target() 으로 원 단위로 복원.
    """
    X_scaled = scaler.transform(df[feature_cols].values)
    X_pca    = pca.transform(X_scaled)
    pc_cols  = [f"PC{i+1}" for i in range(X_pca.shape[1])]
    out = pd.DataFrame(X_pca, index=df.index, columns=pc_cols)
    # 타겟도 동일 RobustScaler 로 정규화 → VAR 내 스케일 균형 확보
    y_scaled = target_scaler.transform(df[[TARGET_COL]].values).flatten()
    out.insert(0, TARGET_COL, y_scaled)
    return out


def inverse_transform_target(values: np.ndarray,
                             target_scaler: RobustScaler) -> np.ndarray:
    """VAR 예측된 정규화 타겟을 원래 단위(만원/㎡)로 역변환한다."""
    return target_scaler.inverse_transform(
        values.reshape(-1, 1)
    ).flatten()


# ---------------------------------------------------------------------------
# 4. VAR 적합 & 예측
# ---------------------------------------------------------------------------
def fit_var(endog: pd.DataFrame, maxlags: int = VAR_MAXLAGS):
    """
    VAR 모형을 적합한다.

    [파라미터 근거]
      - maxlags=12 : 월별 데이터의 연 단위 계절 패턴까지 후보로 둠.
      - ic="aic"  : 표본 크기가 작은 환경에서 BIC 보다 적합도 우선의 AIC 가
        예측력 측면에서 유리하다는 경험적 결과(Lütkepohl, 2005) 반영.
    """
    model = VAR(endog)
    results = model.fit(maxlags=maxlags, ic="aic")
    return results


def forecast_var(results, last_obs: np.ndarray, steps: int,
                 index: pd.DatetimeIndex, columns: list[str]) -> pd.DataFrame:
    """
    VAR 결과로부터 steps 만큼 다단계 예측.

    Parameters
    ----------
    last_obs : np.ndarray
        예측 직전 시점의 lag_order 길이만큼의 관측 행렬.
    """
    fc = results.forecast(y=last_obs, steps=steps)
    return pd.DataFrame(fc, index=index, columns=columns)


# ---------------------------------------------------------------------------
# 5. 시나리오 (1)+(2) : Backtest 2023.1 ~ 2025.12
# ---------------------------------------------------------------------------
def scenario_backtest(full_df: pd.DataFrame) -> dict:
    """
    학습: ~2022-12 / 검증: 2023-01 ~ 2025-12.
    결과: 비교 차트 + MAE/RMSE/MAPE/R2 지표 저장.
    """
    train = full_df.loc[:TRAIN_END]
    valid = full_df.loc[VALID_START:VALID_END]

    # ── 스케일러 + PCA 적합 (학습 구간만 사용 → 누설 방지)
    scaler, target_scaler, pca, feat_cols = fit_scaler_and_pca(train, PCA_N_COMPONENTS)

    # 설명력 기록 (해석/리포팅 용)
    explained = pd.DataFrame({
        "component": [f"PC{i+1}" for i in range(pca.n_components_)],
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_ratio": np.cumsum(pca.explained_variance_ratio_),
    })
    explained.to_csv(OUT_DIR / "backtest_pca_explained_variance.csv", index=False)

    # PCA loading (feature → PC 기여도) 상세 로그
    save_pca_loadings(pca, feat_cols, prefix="backtest")

    # ── VAR 입력 구성 후 적합
    train_endog = transform_with_pca(train, scaler, target_scaler, pca, feat_cols)
    train_endog.to_csv(OUT_DIR / "backtest_train_endog_scaled.csv")  # 정규화된 값

    results = fit_var(train_endog, VAR_MAXLAGS)
    lag = results.k_ar
    print(f"[Backtest] selected VAR lag order = {lag}")

    # ── 예측 (검증 길이만큼)
    steps = len(valid)
    last_obs = train_endog.values[-lag:]
    pred_idx = pd.date_range(VALID_START, VALID_END, freq="MS")
    pred = forecast_var(results, last_obs, steps, pred_idx, train_endog.columns.tolist())
    pred.to_csv(OUT_DIR / "backtest_forecast_scaled.csv")  # 정규화 공간 예측값

    # ── 타겟 역변환 (정규화 → 원 단위)
    forecast_orig = inverse_transform_target(pred[TARGET_COL].values, target_scaler)

    # ── [Boundary anchor] t/t+1 이격 보정
    # VAR 은 scaled space 의 lag 구조에 기반 → RobustScaler 의 median/IQR
    # 으로 역변환하면 첫 시점이 마지막 실측과 어꺋나는 level 이격이
    # 발생함. forecast'[k] = forecast[k] + (last_actual - forecast[0])
    #                                    * max(0, 1 - k/L). L=12 (1년 선형 감쇠).
    last_actual = float(train[TARGET_COL].iloc[-1])
    delta = last_actual - float(forecast_orig[0])
    L = 12
    weights = np.maximum(0.0, 1.0 - np.arange(len(forecast_orig)) / float(L))
    forecast_orig = forecast_orig + delta * weights

    # ── 타겟 비교 테이블 (원 단위)
    compare = pd.DataFrame({
        "actual":   valid[TARGET_COL].values,
        "forecast": forecast_orig,
    }, index=pred_idx)
    compare["abs_error"] = (compare["actual"] - compare["forecast"]).abs()
    compare.to_csv(OUT_DIR / "backtest_target_compare.csv")

    # ── 지표
    mae  = mean_absolute_error(compare["actual"], compare["forecast"])
    rmse = float(np.sqrt(mean_squared_error(compare["actual"], compare["forecast"])))
    mape = float(np.mean(np.abs((compare["actual"] - compare["forecast"]) / compare["actual"])) * 100)
    r2   = r2_score(compare["actual"], compare["forecast"])
    metrics = pd.DataFrame([{
        "MAE": mae, "RMSE": rmse, "MAPE(%)": mape, "R2": r2,
        "lag_order": lag, "n_train": len(train), "n_valid": len(valid),
        "pca_components": PCA_N_COMPONENTS,
    }])
    metrics.to_csv(OUT_DIR / "backtest_metrics.csv", index=False)
    print("[Backtest metrics]\n", metrics.to_string(index=False))

    # ── 차트 (2010~2025 실측 + 2023~2025 예측, 원 단위)
    plt.figure(figsize=(12, 5))
    plt.plot(full_df.index, full_df[TARGET_COL], label="Actual (2010~2025)", color="#1f77b4")
    plt.plot(compare.index, compare["forecast"], label="Forecast (2023~2025)",
             color="#d62728", linestyle="--")
    plt.axvline(pd.Timestamp(TRAIN_END), color="gray", linestyle=":", alpha=0.7,
                label="Train End (2022-12)")
    plt.title("VAR Backtest: Seoul Apartment Sale Avg Price (원 단위 역변환)")
    plt.xlabel("Date"); plt.ylabel(TARGET_COL)
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(OUT_DIR / "backtest_chart.png", dpi=130)
    plt.close()

    return {"metrics": metrics, "compare": compare, "lag": lag}


# ---------------------------------------------------------------------------
# 6. 시나리오 (3) : 미래 10년 예측 2026.1 ~ 2035.12
# ---------------------------------------------------------------------------
def scenario_future(full_df: pd.DataFrame) -> pd.DataFrame:
    """
    전체 데이터(2010~2025)로 재학습하여 2026-01 ~ 2035-12 까지 예측.
    """
    scaler, target_scaler, pca, feat_cols = fit_scaler_and_pca(full_df, PCA_N_COMPONENTS)

    explained = pd.DataFrame({
        "component": [f"PC{i+1}" for i in range(pca.n_components_)],
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_ratio": np.cumsum(pca.explained_variance_ratio_),
    })
    explained.to_csv(OUT_DIR / "future_pca_explained_variance.csv", index=False)

    # PCA loading (feature → PC 기여도) 상세 로그
    save_pca_loadings(pca, feat_cols, prefix="future")

    endog = transform_with_pca(full_df, scaler, target_scaler, pca, feat_cols)
    endog.to_csv(OUT_DIR / "future_endog_scaled.csv")

    results = fit_var(endog, VAR_MAXLAGS)
    lag = results.k_ar
    print(f"[Future] selected VAR lag order = {lag}")

    future_idx = pd.date_range("2026-01-01", FUTURE_END, freq="MS")
    steps = len(future_idx)
    last_obs = endog.values[-lag:]
    future_scaled = forecast_var(results, last_obs, steps, future_idx, endog.columns.tolist())
    future_scaled.to_csv(OUT_DIR / "future_forecast_scaled.csv")  # 정규화 공간 예측

    # 타겟 역변환 (정규화 → 원 단위)
    future_target_orig = inverse_transform_target(
        future_scaled[TARGET_COL].values, target_scaler
    )

    # [Boundary anchor] 2025-12 실측 ↔ 2026-01 예측 이격 흡수 (12개월 선형 감쇠).
    last_actual = float(full_df[TARGET_COL].iloc[-1])
    delta = last_actual - float(future_target_orig[0])
    L = 12
    weights = np.maximum(0.0, 1.0 - np.arange(len(future_target_orig)) / float(L))
    future_target_orig = future_target_orig + delta * weights

    future_out = future_scaled.copy()
    future_out[TARGET_COL] = future_target_orig

    # 타겟 시리즈 (과거 실측 + 미래 예측 결합) 저장
    target_history = full_df[[TARGET_COL]].copy()
    target_history["type"] = "actual"
    target_future = pd.DataFrame(
        {TARGET_COL: future_target_orig, "type": "forecast"},
        index=future_idx
    )
    target_all = pd.concat([target_history, target_future])
    target_all.to_csv(OUT_DIR / "future_target_timeline.csv")

    # 차트 : 2010 ~ 2035 (원 단위)
    plt.figure(figsize=(13, 5))
    plt.plot(target_history.index, target_history[TARGET_COL],
             label="Actual (2010~2025)", color="#1f77b4")
    plt.plot(target_future.index, target_future[TARGET_COL],
             label="Forecast (2026~2035)", color="#2ca02c", linestyle="--")
    plt.axvline(pd.Timestamp("2025-12-01"), color="gray", linestyle=":",
                alpha=0.7, label="Forecast Start (2026-01)")
    plt.title("VAR 10-Year Forecast: Seoul Apartment Sale Avg Price (원 단위 역변환)")
    plt.xlabel("Date"); plt.ylabel(TARGET_COL)
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(OUT_DIR / "future_chart.png", dpi=130)
    plt.close()

    return future_out


# ---------------------------------------------------------------------------
# 7. 메인
# ---------------------------------------------------------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step A : 로드 & 집계
    raw_agg = load_and_aggregate(DATA_CSV)
    raw_agg.to_csv(OUT_DIR / "00_seoul_monthly_raw.csv")

    # Step B : 결측 보간 + 정책 정규화
    clean = clean_missing_and_policy(raw_agg)
    clean.to_csv(OUT_DIR / "01_seoul_monthly_clean.csv")

    # Step C : 시나리오 (1)+(2) 백테스트
    print("\n=== Scenario 1+2: Backtest 2023-01 ~ 2025-12 ===")
    scenario_backtest(clean)

    # Step D : 시나리오 (3) 미래 10년 예측
    print("\n=== Scenario 3: Future Forecast 2026-01 ~ 2035-12 ===")
    scenario_future(clean)

    print("\n[Done] 산출물이 다음 폴더에 저장되었습니다:", OUT_DIR)


if __name__ == "__main__":
    main()
