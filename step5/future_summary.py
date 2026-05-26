"""
future_hybrid_scenarios.csv → 멀티페이지 PDF 리포트

  Page 1  : VAR 시나리오별 예측값  (reb__apt_sale_avg_price, 전체 시나리오 fan chart)
  Page 2  : TFT 보정 후 예측값     (level_p50, 전체 시나리오 fan chart)
  Page 3~ : 시나리오별 보정량 비교  (level_p50 - reb__apt_sale_avg_price, 음영, GRID_N개/page)
"""

import math
import matplotlib
import matplotlib.dates as mdates
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from pathlib import Path

# ── 한글 폰트 (Windows) ───────────────────────────────────────────────────
matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False

# ── 설정 ─────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent
CSV_PATH  = BASE_DIR / "future_hybrid_scenarios.csv"
OUT_PATH  = BASE_DIR / "future_scenario_report.pdf"

GRID_COLS = 4          # page 3~ 열 수
GRID_ROWS = 5          # page 3~ 행 수
GRID_N    = GRID_COLS * GRID_ROWS   # 한 페이지당 시나리오 수

DATE_FMT_FULL  = mdates.DateFormatter("%Y-%m")
DATE_FMT_SMALL = mdates.DateFormatter("%y.%m")
Y_FMT = mticker.FuncFormatter(lambda x, _: f"{x:,.0f}")

# ── 데이터 로드 ───────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH, parse_dates=["timestamp"])
dong_name  = df["dong"].iloc[0] if "dong" in df.columns else "동"
all_ids    = sorted(df["scenario_id"].unique())
N_scen     = len(all_ids)
print(f"로드 완료: {len(df):,} 행 / {N_scen} 시나리오")

# 피벗 (timestamp × scenario_id)
pv_var = df.pivot_table(index="timestamp", columns="scenario_id",
                         values="reb__apt_sale_avg_price").sort_index()
pv_tft = df.pivot_table(index="timestamp", columns="scenario_id",
                         values="level_p50").sort_index()

timestamps = pv_var.index  # DatetimeIndex


# ── 공통 헬퍼 ─────────────────────────────────────────────────────────────
def _fan(ax, pivot, color_line, color_band, title, ylabel):
    """전체 시나리오 fan chart (P10/P50/P90 밴드 + 개별 얇은 선)."""
    vals = pivot.values                             # (T, N)
    p10  = np.nanpercentile(vals, 10, axis=1)
    p50  = np.nanpercentile(vals, 50, axis=1)
    p90  = np.nanpercentile(vals, 90, axis=1)
    ts   = pivot.index

    # 개별 시나리오 (얇고 반투명)
    for col in pivot.columns:
        ax.plot(ts, pivot[col].values,
                color=color_line, linewidth=0.4, alpha=0.12)

    # 밴드
    ax.fill_between(ts, p10, p90, color=color_band, alpha=0.25,
                    label="P10–P90 밴드")
    ax.plot(ts, p50, color=color_band, linewidth=2.0,
            linestyle="--", label="중앙값 (P50)")

    ax.set_title(title, fontsize=11, pad=6)
    ax.set_xlabel("날짜", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.xaxis.set_major_formatter(DATE_FMT_FULL)
    ax.yaxis.set_major_formatter(Y_FMT)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.legend(fontsize=8, loc="upper left")


def _correction_grid(fig, axes_flat, subset_ids, df_src, page_no, total_pages):
    """보정량 서브플롯 배치 (1페이지 분)."""
    for ax_idx, sid in enumerate(subset_ids):
        ax = axes_flat[ax_idx]
        sub     = df_src[df_src["scenario_id"] == sid].sort_values("timestamp")
        ts      = sub["timestamp"].values
        var_raw = sub["reb__apt_sale_avg_price"].values
        tft_cor = sub["level_p50"].values
        corr    = tft_cor - var_raw

        ax.fill_between(ts, var_raw, tft_cor,
                        where=(tft_cor >= var_raw), interpolate=True,
                        alpha=0.28, color="#d62728")
        ax.fill_between(ts, var_raw, tft_cor,
                        where=(tft_cor <  var_raw), interpolate=True,
                        alpha=0.28, color="#1f77b4")
        ax.plot(ts, var_raw, color="#888888", linewidth=1.0,
                linestyle="--")
        ax.plot(ts, tft_cor, color="#d62728", linewidth=1.3)

        max_idx  = np.argmax(np.abs(corr))
        max_corr = corr[max_idx]
        ax.set_title(f"시나리오 {sid}  (최대보정 {max_corr:+.0f})",
                     fontsize=7, pad=2)
        ax.xaxis.set_major_formatter(DATE_FMT_SMALL)
        ax.tick_params(axis="x", labelsize=5, rotation=45)
        ax.tick_params(axis="y", labelsize=5)
        ax.yaxis.set_major_formatter(Y_FMT)
        ax.grid(axis="y", linestyle=":", alpha=0.35)

    # 빈 서브플롯 숨기기
    for ax_idx in range(len(subset_ids), len(axes_flat)):
        axes_flat[ax_idx].set_visible(False)

    # 페이지 공통 범례
    handles = [
        mlines.Line2D([0], [0], color="#888888", linestyle="--", lw=1.0),
        mlines.Line2D([0], [0], color="#d62728", lw=1.3),
        mpatches.Patch(color="#d62728", alpha=0.3),
        mpatches.Patch(color="#1f77b4", alpha=0.3),
    ]
    labels = ["VAR 원래 경로", "TFT 보정 후 (level_p50)",
              "보정 (+, 상향)", "보정 (-, 하향)"]
    fig.legend(handles, labels, loc="lower center", ncol=4,
               fontsize=7, bbox_to_anchor=(0.5, 0.00))
    fig.suptitle(
        f"{dong_name}  시나리오별 TFT 보정량  "
        f"(page {page_no} / {total_pages})",
        fontsize=10, y=1.01,
    )


# ── PDF 출력 ───────────────────────────────────────────────────────────────
n_corr_pages = math.ceil(N_scen / GRID_N)
total_pages  = 2 + n_corr_pages

with PdfPages(OUT_PATH) as pdf:

    # ── PAGE 1: VAR fan chart ──────────────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(14, 7))
    _fan(ax1, pv_var,
         color_line="#1f77b4", color_band="#1f77b4",
         title=f"{dong_name}  VAR 시나리오별 예측 경로\n"
               f"(reb__apt_sale_avg_price  |  N={N_scen}개 시나리오)",
         ylabel="아파트 매매가 (만원/㎡)")
    plt.tight_layout()
    pdf.savefig(fig1, bbox_inches="tight")
    plt.close(fig1)
    print("Page 1 완료 (VAR fan chart)")

    # ── PAGE 2: TFT fan chart ─────────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(14, 7))
    _fan(ax2, pv_tft,
         color_line="#d62728", color_band="#d62728",
         title=f"{dong_name}  TFT 보정 후 예측 경로\n"
               f"(level_p50  |  N={N_scen}개 시나리오)",
         ylabel="아파트 매매가 (만원/㎡)")
    plt.tight_layout()
    pdf.savefig(fig2, bbox_inches="tight")
    plt.close(fig2)
    print("Page 2 완료 (TFT fan chart)")

    # ── PAGE 3~: 보정량 그리드 ───────────────────────────────────────────
    for page_idx in range(n_corr_pages):
        start  = page_idx * GRID_N
        end    = min(start + GRID_N, N_scen)
        subset = all_ids[start:end]

        fig3, axes = plt.subplots(
            GRID_ROWS, GRID_COLS,
            figsize=(18, 14), sharey=False,
        )
        _correction_grid(
            fig3, axes.flatten(), subset, df,
            page_no    = page_idx + 3,
            total_pages= total_pages,
        )
        plt.tight_layout(rect=[0, 0.03, 1, 0.98])
        pdf.savefig(fig3, bbox_inches="tight")
        plt.close(fig3)
        print(f"Page {page_idx + 3} 완료 "
              f"(시나리오 {subset[0]}~{subset[-1]})")

    # ── PDF 메타데이터 ────────────────────────────────────────────────────
    info = pdf.infodict()
    info["Title"]   = f"{dong_name} 미래 예측 시나리오 리포트"
    info["Subject"] = "VAR + TFT Hybrid Forecast Report"

print(f"\n리포트 저장 완료: {OUT_PATH}  ({total_pages}페이지)")

