"""
백테스트 검증 스크립트
======================
backtest_hybrid_scenarios.csv (VAR 시나리오 + TFT 보정 quantile)와
backtest_dong_timeline.csv (실제 집값)을 사용하여
VAR 단독 vs VAR+TFT 하이브리드 모델의 점 추정·구간 추정 성능을 검증.

출력:
  - 터미널: 종합 성적표 (RMSE, WAPE, MDA, PICP, MPIW, CRPS)
  - backtest_validation_report.pdf:
      Page 1  : VAR 시나리오 fan chart
      Page 2  : TFT 하이브리드 fan chart
      Page 3  : Actual vs VAR band vs Hybrid band (검증 chart)
      Page 4  : 50개 샘플 시나리오 - VAR/Hybrid 궤적 + 잔차 보정량 시각화
"""

import math
from pathlib import Path

import matplotlib
import matplotlib.dates as mdates
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False

# ── 경로 ─────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent
HYBRID_CSV   = BASE_DIR / "backtest_hybrid_scenarios.csv"
ACTUAL_CSV   = BASE_DIR / "backtest_dong_timeline.csv"
OUT_PDF      = BASE_DIR / "backtest_validation_report.pdf"
METRICS_CSV  = BASE_DIR / "backtest_validation_metrics.csv"

SAMPLE_N         = 50          # 마지막 페이지에 그릴 시나리오 수
COVERAGE_TARGET  = 0.80
DATE_FMT_FULL    = mdates.DateFormatter("%Y-%m")
DATE_FMT_SMALL   = mdates.DateFormatter("%y.%m")
Y_FMT            = mticker.FuncFormatter(lambda x, _: f"{x:,.0f}")

# ── 데이터 로드 ───────────────────────────────────────────────────────────
df_sc  = pd.read_csv(HYBRID_CSV, parse_dates=["timestamp"])
df_act = pd.read_csv(ACTUAL_CSV, parse_dates=["timestamp"])

dong_name = df_sc["dong"].iloc[0]
all_ids   = sorted(df_sc["scenario_id"].unique())
N_scen    = len(all_ids)
print(f"[load] 시나리오 {N_scen}개 / 실측 {len(df_act)}행 / 동={dong_name}")

# ── 시나리오 ↔ 시간 피벗 ─────────────────────────────────────────────────
pv_var  = df_sc.pivot_table(index="timestamp", columns="scenario_id",
                            values="reb__apt_sale_avg_price").sort_index()
pv_p10  = df_sc.pivot_table(index="timestamp", columns="scenario_id",
                            values="level_p10").sort_index()
pv_p50  = df_sc.pivot_table(index="timestamp", columns="scenario_id",
                            values="level_p50").sort_index()
pv_p90  = df_sc.pivot_table(index="timestamp", columns="scenario_id",
                            values="level_p90").sort_index()

timestamps = pv_var.index
actual_ser = (
    df_act.set_index("timestamp")["actual"]
          .reindex(timestamps)
          .astype(float)
)
y_true = actual_ser.values

# ── 분포 추출 ─────────────────────────────────────────────────────────────
# VAR 표본: 시나리오별 reb__apt_sale_avg_price (T × N)
var_samples = pv_var.values
# Hybrid 표본: 시나리오별 P10/P50/P90 을 풀링 (T × 3N) → VAR 불확실성 + TFT 분위
hyb_samples = np.concatenate(
    [pv_p10.values, pv_p50.values, pv_p90.values], axis=1
)

def quantiles(arr):
    p10 = np.nanpercentile(arr, 10, axis=1)
    p50 = np.nanpercentile(arr, 50, axis=1)
    p90 = np.nanpercentile(arr, 90, axis=1)
    return p10, p50, p90

var_p10, var_p50, var_p90 = quantiles(var_samples)
hyb_p10, hyb_p50, hyb_p90 = quantiles(hyb_samples)

# ── Metric 함수 ───────────────────────────────────────────────────────────
def rmse(y, p):
    return float(np.sqrt(np.nanmean((y - p) ** 2)))

def wape(y, p):
    return float(np.nansum(np.abs(y - p)) / np.nansum(np.abs(y))) * 100.0

def mda(y, p):
    # 전월 대비 방향 적중률
    dy = np.sign(np.diff(y))
    dp = np.sign(np.diff(p))
    mask = (~np.isnan(dy)) & (~np.isnan(dp))
    return float((dy[mask] == dp[mask]).mean()) * 100.0

def picp(y, lo, hi):
    inside = (y >= lo) & (y <= hi)
    return float(np.nanmean(inside)) * 100.0

def mpiw(lo, hi):
    return float(np.nanmean(hi - lo))

def crps_empirical(samples_TxM, y_T):
    """샘플 기반 평균 CRPS (T 시점 평균)."""
    out = []
    for t in range(len(y_T)):
        s = samples_TxM[t]
        s = s[~np.isnan(s)]
        if len(s) == 0 or np.isnan(y_T[t]):
            continue
        # CRPS = E|S - y| - 0.5 E|S - S'|
        term1 = np.mean(np.abs(s - y_T[t]))
        # 효율적인 두번째 항: 정렬 후 누적합 트릭
        s_sorted = np.sort(s)
        m = len(s_sorted)
        idx = np.arange(1, m + 1)
        term2 = (2.0 / (m * m)) * np.sum((2 * idx - m - 1) * s_sorted) / 2.0
        out.append(term1 - term2)
    return float(np.mean(out))

# ── Metric 계산 ───────────────────────────────────────────────────────────
metrics = {
    "RMSE (만원)":     (rmse(y_true, var_p50),  rmse(y_true, hyb_p50)),
    "WAPE (%)":        (wape(y_true, var_p50),  wape(y_true, hyb_p50)),
    "MDA (%)":         (mda(y_true,  var_p50),  mda(y_true,  hyb_p50)),
    "PICP (%, 목표80)":(picp(y_true, var_p10, var_p90),
                        picp(y_true, hyb_p10, hyb_p90)),
    "MPIW (만원)":     (mpiw(var_p10, var_p90), mpiw(hyb_p10, hyb_p90)),
    "CRPS":            (crps_empirical(var_samples, y_true),
                        crps_empirical(hyb_samples, y_true)),
}

# ── 성적표 출력 ───────────────────────────────────────────────────────────
def impact(metric, v_var, v_hyb):
    if "MDA" in metric:
        diff = v_hyb - v_var
        return f"📈 {diff:+.1f}%p"
    if "PICP" in metric:
        # 80에 가까울수록 좋음
        gap_var = abs(v_var - 80.0)
        gap_hyb = abs(v_hyb - 80.0)
        return f"🛡️ 목표대비 {gap_var:.1f}p → {gap_hyb:.1f}p"
    # 작을수록 좋은 지표
    if v_var == 0:
        return "—"
    pct = (v_var - v_hyb) / v_var * 100.0
    return f"📉 {pct:+.1f}% 변화"

header = f"\n{'='*78}\n  백테스트 검증 성적표  ({dong_name}, {timestamps.min():%Y-%m} ~ {timestamps.max():%Y-%m})\n{'='*78}"
print(header)
print(f"{'평가지표':<20}{'VAR 단독':>16}{'VAR+TFT':>16}   개선 성과")
print("-" * 78)
for k, (v_var, v_hyb) in metrics.items():
    print(f"{k:<20}{v_var:>16.3f}{v_hyb:>16.3f}   {impact(k, v_var, v_hyb)}")
print("=" * 78 + "\n")

# 저장
pd.DataFrame(
    [{"metric": k, "var_only": v[0], "var_tft_hybrid": v[1]} for k, v in metrics.items()]
).to_csv(METRICS_CSV, index=False, encoding="utf-8-sig")
print(f"[save] metrics → {METRICS_CSV.name}")

# ────────────────────────────────────────────────────────────────────────────
# 차트 헬퍼
# ────────────────────────────────────────────────────────────────────────────
def fan_chart(ax, pivot, color, title, ylabel):
    vals = pivot.values
    p10  = np.nanpercentile(vals, 10, axis=1)
    p50  = np.nanpercentile(vals, 50, axis=1)
    p90  = np.nanpercentile(vals, 90, axis=1)
    ts   = pivot.index
    for col in pivot.columns:
        ax.plot(ts, pivot[col].values, color=color, linewidth=0.4, alpha=0.10)
    ax.fill_between(ts, p10, p90, color=color, alpha=0.22, label="P10–P90 밴드")
    ax.plot(ts, p50, color=color, lw=2.0, ls="--", label="중앙값 (P50)")
    ax.plot(timestamps, y_true, color="black", lw=1.8, label="실제 (Actual)")
    ax.set_title(title, fontsize=11, pad=6)
    ax.set_xlabel("날짜", fontsize=9); ax.set_ylabel(ylabel, fontsize=9)
    ax.xaxis.set_major_formatter(DATE_FMT_FULL)
    ax.yaxis.set_major_formatter(Y_FMT)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.legend(fontsize=8, loc="upper left")


def comparison_chart(ax):
    ts = timestamps
    # VAR 밴드 (파랑)
    ax.fill_between(ts, var_p10, var_p90, color="#1f77b4", alpha=0.18,
                    label="VAR P10–P90")
    ax.plot(ts, var_p50, color="#1f77b4", lw=1.8, ls="--", label="VAR P50")
    # Hybrid 밴드 (빨강)
    ax.fill_between(ts, hyb_p10, hyb_p90, color="#d62728", alpha=0.18,
                    label="VAR+TFT P10–P90")
    ax.plot(ts, hyb_p50, color="#d62728", lw=2.0, label="VAR+TFT P50")
    # Actual
    ax.plot(ts, y_true, color="black", lw=2.2, label="실제 집값")

    # 본문에 metric 박스
    txt = (
        f"RMSE  : {metrics['RMSE (만원)'][0]:7.2f} → {metrics['RMSE (만원)'][1]:7.2f}\n"
        f"WAPE  : {metrics['WAPE (%)'][0]:6.2f}% → {metrics['WAPE (%)'][1]:6.2f}%\n"
        f"MDA   : {metrics['MDA (%)'][0]:6.1f}% → {metrics['MDA (%)'][1]:6.1f}%\n"
        f"PICP  : {metrics['PICP (%, 목표80)'][0]:6.1f}% → {metrics['PICP (%, 목표80)'][1]:6.1f}%\n"
        f"MPIW  : {metrics['MPIW (만원)'][0]:7.2f} → {metrics['MPIW (만원)'][1]:7.2f}\n"
        f"CRPS  : {metrics['CRPS'][0]:7.2f} → {metrics['CRPS'][1]:7.2f}"
    )
    ax.text(0.01, 0.99, txt, transform=ax.transAxes,
            fontsize=9, family="monospace",
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5",
                      facecolor="white", edgecolor="#888", alpha=0.85))

    ax.set_title(
        f"{dong_name}  VAR 단독 vs VAR+TFT 하이브리드  (Backtest 검증)",
        fontsize=12, pad=8,
    )
    ax.set_xlabel("날짜"); ax.set_ylabel("아파트 매매가 (만원/㎡)")
    ax.xaxis.set_major_formatter(DATE_FMT_FULL)
    ax.yaxis.set_major_formatter(Y_FMT)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.legend(fontsize=9, loc="lower left", ncol=2)


def sampled_scenarios_page(fig, sample_ids):
    """3-패널: (위) VAR 궤적, (중) Hybrid 궤적, (아래) 잔차 보정량."""
    gs = fig.add_gridspec(3, 1, hspace=0.35)
    ax_var = fig.add_subplot(gs[0, 0])
    ax_hyb = fig.add_subplot(gs[1, 0], sharex=ax_var)
    ax_cor = fig.add_subplot(gs[2, 0], sharex=ax_var)

    ts = timestamps
    for sid in sample_ids:
        ax_var.plot(ts, pv_var[sid].values,
                    color="#1f77b4", lw=0.5, alpha=0.35)
        ax_hyb.plot(ts, pv_p50[sid].values,
                    color="#d62728", lw=0.5, alpha=0.35)
        corr = pv_p50[sid].values - pv_var[sid].values
        ax_cor.plot(ts, corr, lw=0.5, alpha=0.35, color="#555555")

    # 중앙값/실제 오버레이
    ax_var.plot(ts, y_true, color="black", lw=2.0, label="실제")
    ax_var.plot(ts, var_p50, color="#1f77b4", lw=2.0, ls="--", label="VAR P50")
    ax_hyb.plot(ts, y_true, color="black", lw=2.0, label="실제")
    ax_hyb.plot(ts, hyb_p50, color="#d62728", lw=2.0, ls="--", label="Hybrid P50")
    ax_cor.axhline(0, color="black", lw=1.0)
    mean_corr = np.nanmean(pv_p50.values - pv_var.values, axis=1)
    ax_cor.plot(ts, mean_corr, color="#d62728", lw=2.0,
                label="평균 잔차 보정량 (TFT − VAR)")

    for ax, title, ylab in [
        (ax_var, f"VAR 생성 궤적  (sample {len(sample_ids)} of {N_scen})",
                 "VAR price"),
        (ax_hyb, "VAR + TFT 하이브리드 궤적 (level_p50)",
                 "Hybrid price"),
        (ax_cor, "잔차 보정량 (Hybrid − VAR)",
                 "보정 (만원)"),
    ]:
        ax.set_title(title, fontsize=10, pad=4)
        ax.set_ylabel(ylab, fontsize=9)
        ax.xaxis.set_major_formatter(DATE_FMT_FULL)
        ax.yaxis.set_major_formatter(Y_FMT)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right",
                 fontsize=8)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        ax.legend(fontsize=8, loc="upper left")

    fig.suptitle(
        f"{dong_name}  샘플 {len(sample_ids)} 시나리오: VAR / Hybrid / 잔차 보정",
        fontsize=12, y=0.995,
    )


# ────────────────────────────────────────────────────────────────────────────
# PDF 작성
# ────────────────────────────────────────────────────────────────────────────
rng = np.random.default_rng(42)
sampled = sorted(rng.choice(all_ids, size=min(SAMPLE_N, N_scen),
                            replace=False).tolist())

with PdfPages(OUT_PDF) as pdf:
    # Page 1: VAR fan
    fig, ax = plt.subplots(figsize=(14, 7))
    fan_chart(ax, pv_var, "#1f77b4",
              f"{dong_name}  VAR 시나리오별 예측 경로 (N={N_scen})",
              "아파트 매매가 (만원/㎡)")
    plt.tight_layout(); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
    print("[pdf] page 1 (VAR fan)")

    # Page 2: Hybrid fan (p50)
    fig, ax = plt.subplots(figsize=(14, 7))
    fan_chart(ax, pv_p50, "#d62728",
              f"{dong_name}  VAR+TFT 하이브리드 P50 경로 (N={N_scen})",
              "아파트 매매가 (만원/㎡)")
    plt.tight_layout(); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
    print("[pdf] page 2 (Hybrid fan)")

    # Page 3: Comparison
    fig, ax = plt.subplots(figsize=(14, 7))
    comparison_chart(ax)
    plt.tight_layout(); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
    print("[pdf] page 3 (Comparison + Metrics)")

    # Page 4: 50개 샘플 시나리오
    fig = plt.figure(figsize=(14, 12))
    sampled_scenarios_page(fig, sampled)
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
    print("[pdf] page 4 (Sampled trajectories + residual correction)")

    info = pdf.infodict()
    info["Title"]   = f"{dong_name} Backtest Validation Report"
    info["Subject"] = "VAR vs VAR+TFT Hybrid - Probabilistic Backtest"

print(f"\n[done] 리포트 저장 → {OUT_PDF.name}")
