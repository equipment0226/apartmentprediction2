# -*- coding: utf-8 -*-
"""
Patch 3 for Diffusion.py:
  1. Log-return target (TARGET_COL = "target_logret")
  2. scenario_backtest / scenario_future: panel-aware + level reconstruction
  3. main(): load_panel, argparse TARGET_DONG
  4. Fix all stale index-based DataFrame access
"""

import re
from pathlib import Path

src = Path(__file__).parent / "Diffusion.py"
txt = src.read_text(encoding="utf-8")
ORIG = txt  # backup for diff check

# ============================================================
# HELPER
# ============================================================
def replace_one(old, new, label):
    global txt
    if old in txt:
        txt = txt.replace(old, new, 1)
        print(f"  OK  : {label}")
    else:
        print(f"  FAIL: {label}")

def replace_re(pattern, new, label, flags=re.DOTALL):
    global txt
    m = re.search(pattern, txt, flags)
    if m:
        txt = txt[:m.start()] + new + txt[m.end():]
        print(f"  OK  : {label}")
    else:
        print(f"  FAIL: {label}")

# ============================================================
# 1. Constants: split TARGET_COL into LEVEL + LOGRET
# ============================================================
print("\n[1] Constants")
replace_one(
    'TARGET_COL = "reb__apt_sale_avg_price"\n'
    'TIME_COL   = "timestamp"\n'
    'ID_COLS    = ["si", "gu", "dong"]',
    'TARGET_LEVEL_COL = "reb__apt_sale_avg_price"  # 원본 가격 (level)\n'
    'TARGET_COL       = "target_logret"             # 모델 학습 타겟 = 월간 로그수익률\n'
    'TIME_COL   = "timestamp"\n'
    'ID_COLS    = ["si", "gu", "dong"]',
    "TARGET_LEVEL_COL / TARGET_COL split"
)

# ============================================================
# 2. clean_missing_and_policy: add log-return column per dong
# ============================================================
print("\n[2] clean_missing_and_policy: add target_logret")
replace_one(
    '    out = out.reset_index().fillna(0.0)\n'
    '    policy_cols = [c for c in out.columns if c.startswith("policy__")]\n'
    '    out[policy_cols] = out[policy_cols].round().clip(-1, 1).astype(int)\n'
    '    return out',
    '    out = out.reset_index().fillna(0.0)\n'
    '    policy_cols = [c for c in out.columns if c.startswith("policy__")]\n'
    '    out[policy_cols] = out[policy_cols].round().clip(-1, 1).astype(int)\n'
    '    # log-return 타겟: 동별 월간 로그수익률 (TFT 와 동일 정통 방식)\n'
    '    # P_t / P_{t-1} log-diff → 정상(stationary) 시계열 확보\n'
    '    # fillna(0) : 첫 시점은 수익률 = 0 처리\n'
    '    out[TARGET_COL] = (\n'
    '        out.groupby("dong")[TARGET_LEVEL_COL]\n'
    '           .transform(lambda s: np.log(s).diff())\n'
    '    ).fillna(0.0).astype(np.float32)\n'
    '    return out',
    "target_logret per dong"
)

# ============================================================
# 3. run_adf_report: use TARGET_LEVEL_COL for ADF (level series)
# ============================================================
print("\n[3] run_adf_report")
replace_one(
    '        series = grp[TARGET_COL].dropna()',
    '        series = grp[TARGET_LEVEL_COL].dropna()  # ADF on level price',
    "ADF on level price"
)

# ============================================================
# 4. fit_scalers: exclude TARGET_LEVEL_COL from cov_cols
# ============================================================
print("\n[4] fit_scalers: exclude TARGET_LEVEL_COL")
replace_one(
    '        if c != TARGET_COL\n'
    '        and c not in ID_COLS + [TIME_COL]\n'
    '        and pd.api.types.is_numeric_dtype(train_df[c])',
    '        if c not in (TARGET_COL, TARGET_LEVEL_COL)\n'
    '        and c not in ID_COLS + [TIME_COL]\n'
    '        and pd.api.types.is_numeric_dtype(train_df[c])',
    "exclude TARGET_LEVEL_COL from cov_cols"
)

# ============================================================
# 5. inverse_target: rename to inverse_logret for clarity
#    Keep function but add level-restoration helper
# ============================================================
print("\n[5] Add logret_to_level helper after inverse_target")
old_inv = (
    'def inverse_target(arr_scaled: np.ndarray,\n'
    '                   target_scaler: RobustScaler) -> np.ndarray:\n'
    '    """RobustScaler 역변환 → 원 단위 가격(원/㎡)."""\n'
    '    flat = arr_scaled.reshape(-1, 1)\n'
    '    return target_scaler.inverse_transform(flat).reshape(arr_scaled.shape)'
)
new_inv = (
    'def inverse_target(arr_scaled: np.ndarray,\n'
    '                   target_scaler: RobustScaler) -> np.ndarray:\n'
    '    """RobustScaler 역변환 → 로그수익률 원단위(unscaled logret)."""\n'
    '    flat = arr_scaled.reshape(-1, 1)\n'
    '    return target_scaler.inverse_transform(flat).reshape(arr_scaled.shape)\n'
    '\n'
    '\n'
    'def logret_to_level(paths_logret: np.ndarray, last_price: float) -> np.ndarray:\n'
    '    """로그수익률 paths → 원 단위 가격 level 복원.\n'
    '\n'
    '    price_t = last_price * exp(sum_{k=0}^{t} logret_k)\n'
    '    last_price : 예측 시작 직전 시점의 실제 가격 (anchor).\n'
    '    paths_logret : (N_scenarios, horizon) unscaled log-returns.\n'
    '    Returns (N_scenarios, horizon) level prices.\n'
    '    """\n'
    '    return last_price * np.exp(np.cumsum(paths_logret, axis=1))'
)
replace_one(old_inv, new_inv, "logret_to_level helper")

# ============================================================
# 6. scenario_backtest: full rewrite
# ============================================================
print("\n[6] scenario_backtest")

OLD_BT = r'def scenario_backtest\(full_df.*?return \{.*?\}' + r'\s*\n(?=\n#)'
# Use regex with DOTALL
m = re.search(r'def scenario_backtest\(full_df.*?\n    return \{.*?\}\n', txt, re.DOTALL)
if m:
    OLD_BT_TEXT = m.group(0)
    NEW_BT_TEXT = '''\
def scenario_backtest(full_df: pd.DataFrame) -> dict:
    """패널 데이터로 학습 -> 2025 12개월 x 100 path, 실측 비교.

    TARGET_DONG 동(또는 서울 평균)의 과거 시퀀스를 기준으로 추론.
    학습: 모든 동 슬라이딩 윈도우 (Global Model).
    타겟: target_logret (로그수익률) -> inverse 후 level 복원.
    """
    print("\\n=== Scenario 1+2: Backtest 2025-01 ~ 2025-12 ===")
    train = full_df[full_df[TIME_COL] <= TRAIN_END]
    valid = full_df[(full_df[TIME_COL] >= VALID_START) &
                    (full_df[TIME_COL] <= VALID_END)]

    target_scaler, cov_scaler, cov_cols = fit_scalers(train)

    # 패널 전체 학습 윈도우 (동별 루프 + vstack)
    p_y, p_X, f_X, f_y = build_sliding_windows(
        train, target_scaler, cov_scaler, cov_cols,
        PAST_LEN, BACKTEST_HORIZON)
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
    sampler = WeightedRandomSampler(weights=torch.from_numpy(w_np).double(),
                                    num_samples=len(w_np), replacement=True)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, sampler=sampler,
                        num_workers=0, drop_last=False)

    schedule = DiffusionSchedule()
    model = SSSDDiffusionNet(cov_dim=len(cov_cols)).to(DEVICE)
    print(f"[Backtest] model params = {sum(p.numel() for p in model.parameters()):,}")

    hist = train_diffusion(model, schedule, loader,
                           n_epochs=N_EPOCHS, lr=LEARNING_RATE,
                           log_csv=OUT_DIR / "backtest_train_loss.csv")

    # ------------------------------------------------------------------
    # 추론: TARGET_DONG 동 또는 서울 전체 평균으로 past context 구성
    # ------------------------------------------------------------------
    pred_idx = pd.DatetimeIndex(sorted(valid[TIME_COL].unique()))
    tgt_dong = TARGET_DONG
    if tgt_dong is not None and tgt_dong in train["dong"].values:
        td_tr = train[train["dong"] == tgt_dong].sort_values(TIME_COL)
        td_va = valid[valid["dong"] == tgt_dong].sort_values(TIME_COL)
        y_td = target_scaler.transform(
            td_tr[[TARGET_COL]].values).astype(np.float32).flatten()
        X_td = cov_scaler.transform(td_tr[cov_cols].values).astype(np.float32)
        X_va = cov_scaler.transform(td_va[cov_cols].values).astype(np.float32)
        past_y_np = y_td[-PAST_LEN:]
        past_X_np = X_td[-PAST_LEN:]
        last_level = float(td_tr[TARGET_LEVEL_COL].iloc[-1])
        truth_level = td_va[TARGET_LEVEL_COL].values
        print(f"[Backtest] inference target = {tgt_dong!r}")
    else:
        # Seoul mean fallback
        mean_tr = (train.groupby(TIME_COL)[[TARGET_COL, TARGET_LEVEL_COL] + cov_cols]
                        .mean().sort_index())
        mean_va = (valid.groupby(TIME_COL)[[TARGET_COL, TARGET_LEVEL_COL] + cov_cols]
                        .mean().sort_index())
        y_td = target_scaler.transform(
            mean_tr[[TARGET_COL]].values).astype(np.float32).flatten()
        X_td = cov_scaler.transform(mean_tr[cov_cols].values).astype(np.float32)
        X_va = cov_scaler.transform(mean_va[cov_cols].values).astype(np.float32)
        past_y_np = y_td[-PAST_LEN:]
        past_X_np = X_td[-PAST_LEN:]
        last_level = float(mean_tr[TARGET_LEVEL_COL].iloc[-1])
        truth_level = mean_va[TARGET_LEVEL_COL].values
        print("[Backtest] inference target = Seoul mean")

    # ------------------------------------------------------------------
    # What-if 시나리오 코바리에이트 구성 + sampling
    # ------------------------------------------------------------------
    scenarios = build_whatif_covariates(X_va, cov_cols, cov_scaler)
    scenarios = {"Baseline_Actual": X_va, **scenarios}

    all_long_rows = []
    summary_rows  = {}
    metrics_rows  = []

    for name, fut_X_np in scenarios.items():
        print(f"  [backtest sampling] scenario = {name}")
        paths_scaled = sample_scenarios(model, schedule, past_y_np, past_X_np,
                                        fut_X_np, n_scenarios=N_SCENARIOS)
        # logret 역변환 -> level 복원
        paths_logret = inverse_target(paths_scaled, target_scaler)  # unscaled logret
        paths_level  = logret_to_level(paths_logret, last_level)    # (N, L) 가격

        summary = pd.DataFrame({
            "timestamp": pred_idx,
            "q10":    np.quantile(paths_level, 0.10, axis=0),
            "median": np.quantile(paths_level, 0.50, axis=0),
            "q90":    np.quantile(paths_level, 0.90, axis=0),
            "mean":   paths_level.mean(axis=0),
            "std":    paths_level.std(axis=0),
        })
        summary["scenario"] = name
        summary_rows[name]  = summary

        for i, p in enumerate(paths_level):
            for ts, v in zip(pred_idx, p):
                all_long_rows.append({
                    "scenario": name, "scenario_id": i,
                    "timestamp": ts, "value": float(v),
                })

        m = compute_metrics(truth_level, summary["median"].values)
        m = {"scenario": name,
             "category": SCENARIO_CATEGORY.get(name, "Baseline"), **m}
        metrics_rows.append(m)

    # 저장
    pd.DataFrame(all_long_rows).to_csv(OUT_DIR / "backtest_paths_long.csv", index=False)
    pd.concat(summary_rows.values(), ignore_index=True).to_csv(
        OUT_DIR / "backtest_summary_by_scenario.csv", index=False)
    metrics_df = pd.DataFrame(metrics_rows).sort_values("MAE")
    metrics_df.to_csv(OUT_DIR / "backtest_metrics_by_scenario.csv", index=False)

    base_summary = summary_rows["Baseline_Actual"].drop(columns=["scenario"])
    compare = base_summary.copy()
    compare["actual"] = truth_level
    compare.to_csv(OUT_DIR / "backtest_compare.csv", index=False)
    base_metrics = compute_metrics(truth_level, base_summary["median"].values)
    pd.DataFrame([base_metrics]).to_csv(OUT_DIR / "backtest_metrics.csv", index=False)
    print(f"[Backtest] baseline metrics = {base_metrics}")
    print("[Backtest] top-5 best-matching scenarios:")
    print(metrics_df.head(5).to_string(index=False))

    # 차트 : 학습 구간 historical (Seoul mean level) + 시나리오 fan
    hist_level_series = (train.groupby(TIME_COL)[TARGET_LEVEL_COL].mean().sort_index())
    hist_idx   = hist_level_series.index
    hist_orig  = hist_level_series.values

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
    base = summary_rows["Baseline_Actual"]
    ax.plot(pred_idx, base["median"], color="blue", linewidth=2.2,
            label="Baseline_Actual median")
    ax.fill_between(pred_idx, base["q10"], base["q90"],
                    color="blue", alpha=0.15, label="Baseline 10-90%")
    ax.plot(pred_idx, truth_level, color="red", linewidth=2.2, label="actual")
    ax.axvline(pd.Timestamp(VALID_START), color="grey", linestyle="--",
               linewidth=0.8, label="forecast start")
    ax.set_title(f"Backtest 2025 - {len(scenarios)} scenarios x {N_SCENARIOS} paths "
                 f"(level price; target={tgt_dong or 'Seoul mean'})")
    ax.set_ylabel("apt sale avg price (level)")
    ax.legend(loc="best", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "backtest_chart.png", dpi=130)
    plt.close(fig)

    return {"metrics": base_metrics, "summary": base_summary,
            "metrics_by_scenario": metrics_df, "history": hist}

'''
    txt = txt[:m.start()] + NEW_BT_TEXT + txt[m.end():]
    print("  OK  : scenario_backtest rewrite")
else:
    print("  FAIL: scenario_backtest NOT FOUND")

# ============================================================
# 7. scenario_future: full rewrite
# ============================================================
print("\n[7] scenario_future")
m = re.search(r'def scenario_future\(full_df.*?\n    return \{.*?\}\n', txt, re.DOTALL)
if m:
    NEW_FUT_TEXT = '''\
def scenario_future(full_df: pd.DataFrame) -> dict:
    """전체 패널 데이터로 재학습 -> 2026~2035 What-if 시나리오 x 100 path.

    TARGET_DONG 동(또는 서울 평균)의 마지막 past context 를 anchor 로 사용.
    log-return 예측 -> level 복원 (logret_to_level).
    """
    print("\\n=== Scenario 3: Future 2026-01 ~ 2035-12 with What-if ===")

    target_scaler, cov_scaler, cov_cols = fit_scalers(full_df)

    # 패널 전체 학습 윈도우
    p_y, p_X, f_X, f_y = build_sliding_windows(
        full_df, target_scaler, cov_scaler, cov_cols,
        PAST_LEN, BACKTEST_HORIZON)
    print(f"[Future] training windows = {len(p_y)} "
          f"(past={PAST_LEN}, horizon={BACKTEST_HORIZON})")

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
    model = SSSDDiffusionNet(cov_dim=len(cov_cols)).to(DEVICE)

    hist = train_diffusion(model, schedule, loader,
                           n_epochs=N_EPOCHS, lr=LEARNING_RATE,
                           log_csv=OUT_DIR / "future_train_loss.csv")

    # ------------------------------------------------------------------
    # 추론: TARGET_DONG 동 또는 서울 전체 평균으로 past context + anchor 구성
    # ------------------------------------------------------------------
    tgt_dong = TARGET_DONG
    if tgt_dong is not None and tgt_dong in full_df["dong"].values:
        td_all = full_df[full_df["dong"] == tgt_dong].sort_values(TIME_COL)
        y_td_all = target_scaler.transform(
            td_all[[TARGET_COL]].values).astype(np.float32).flatten()
        X_td_all = cov_scaler.transform(td_all[cov_cols].values).astype(np.float32)
        past_y_np = y_td_all[-PAST_LEN:]
        past_X_np = X_td_all[-PAST_LEN:]
        last_level = float(td_all[TARGET_LEVEL_COL].iloc[-1])
        last_X_orig = cov_scaler.inverse_transform(X_td_all[-1:])
        print(f"[Future] inference target = {tgt_dong!r}, last_level = {last_level:,.0f}")
    else:
        mean_all = (full_df.groupby(TIME_COL)[[TARGET_COL, TARGET_LEVEL_COL] + cov_cols]
                           .mean().sort_index())
        y_all_arr = target_scaler.transform(
            mean_all[[TARGET_COL]].values).astype(np.float32).flatten()
        X_all_arr = cov_scaler.transform(mean_all[cov_cols].values).astype(np.float32)
        past_y_np = y_all_arr[-PAST_LEN:]
        past_X_np = X_all_arr[-PAST_LEN:]
        last_level = float(mean_all[TARGET_LEVEL_COL].iloc[-1])
        last_X_orig = cov_scaler.inverse_transform(X_all_arr[-1:])
        print(f"[Future] inference target = Seoul mean, last_level = {last_level:,.0f}")

    # baseline future covariate: 마지막 관측치 carry-forward
    fut_idx = pd.date_range(FUTURE_START, FUTURE_END, freq="MS")
    L_fut = len(fut_idx)
    baseline_fut_X_orig = np.repeat(last_X_orig, L_fut, axis=0)
    baseline_fut_X = cov_scaler.transform(baseline_fut_X_orig).astype(np.float32)

    scenarios = build_whatif_covariates(baseline_fut_X, cov_cols, cov_scaler)

    # 각 시나리오 covariate 경로 원단위 저장 (감사용)
    cov_long_rows = []
    for name, sc_arr in scenarios.items():
        orig = cov_scaler.inverse_transform(sc_arr)
        for ti, ts in enumerate(fut_idx):
            for ci, c in enumerate(cov_cols):
                cov_long_rows.append({
                    "scenario": name, "timestamp": ts,
                    "feature": c, "value": float(orig[ti, ci]),
                })
    pd.DataFrame(cov_long_rows).to_csv(OUT_DIR / "future_scenario_covariates.csv",
                                       index=False)

    meta = pd.DataFrame([
        {"scenario": n, "category": SCENARIO_CATEGORY.get(n, "Other")}
        for n in scenarios.keys()
    ])
    meta.to_csv(OUT_DIR / "future_scenario_meta.csv", index=False)

    all_long_rows = []
    summary_rows  = {}

    for name, fut_X_np in scenarios.items():
        print(f"  [sampling] scenario = {name}")
        paths_scaled = sample_scenarios(model, schedule, past_y_np, past_X_np,
                                        fut_X_np, n_scenarios=N_SCENARIOS)
        paths_logret = inverse_target(paths_scaled, target_scaler)
        paths_level  = logret_to_level(paths_logret, last_level)

        summary = pd.DataFrame({
            "timestamp": fut_idx,
            "q10":    np.quantile(paths_level, 0.10, axis=0),
            "median": np.quantile(paths_level, 0.50, axis=0),
            "q90":    np.quantile(paths_level, 0.90, axis=0),
            "mean":   paths_level.mean(axis=0),
            "std":    paths_level.std(axis=0),
        })
        summary["scenario"] = name
        summary_rows[name] = summary

        for i, p in enumerate(paths_level):
            for ts, v in zip(fut_idx, p):
                all_long_rows.append({
                    "scenario": name, "scenario_id": i,
                    "timestamp": ts, "value": v,
                })

    long_df = pd.DataFrame(all_long_rows)
    long_df.to_csv(OUT_DIR / "future_paths_long.csv", index=False)
    summary_all = pd.concat(summary_rows.values(), ignore_index=True)
    summary_all.to_csv(OUT_DIR / "future_summary.csv", index=False)

    # ---- 차트 ----
    # historical: Seoul mean level (항상 서울 평균으로 표시)
    hist_level_series = (full_df.groupby(TIME_COL)[TARGET_LEVEL_COL].mean().sort_index())
    hist_idx_plt = hist_level_series.index
    history_y    = hist_level_series.values

    cmap = plt.get_cmap("tab20")

    # 전체 시나리오 median 비교 차트
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(hist_idx_plt, history_y, color="black",
            linewidth=1.5, label="history (2010-2025)")
    for k, (name, summ) in enumerate(summary_rows.items()):
        ax.plot(summ["timestamp"], summ["median"],
                color=cmap(k % 20), linewidth=1.5, label=name, alpha=0.85)
    ax.set_title(f"Future 2026-2035 - {N_SCENARIOS} paths/scenario "
                 f"({len(scenarios)} scenarios; target={tgt_dong or 'Seoul mean'})")
    ax.set_ylabel("apt sale avg price (level)")
    ax.legend(loc="upper left", fontsize=7, ncol=2, bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    fig.savefig(OUT_DIR / "future_chart_all_medians.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # 카테고리별 subplot
    cats: dict[str, list[str]] = {}
    for name in summary_rows.keys():
        c = SCENARIO_CATEGORY.get(name, "Other")
        cats.setdefault(c, []).append(name)
    cat_order = ["A_Rate", "B_Policy", "C_Macro", "D_Supply", "E_Replay"]
    n_cats = len(cat_order)
    fig, axes = plt.subplots(n_cats, 1, figsize=(13, 3.2 * n_cats), sharex=True)
    base_summ = summary_rows.get("baseline", next(iter(summary_rows.values())))
    for ax_i, cat_name in zip(axes, cat_order):
        names = cats.get(cat_name, [])
        ax_i.plot(hist_idx_plt[-60:], history_y[-60:],
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
    fig.savefig(OUT_DIR / "future_chart_by_category.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # baseline 100-path 차트
    fig, ax = plt.subplots(figsize=(12, 6))
    base_paths_df = long_df[long_df["scenario"] == "baseline"]
    for sid, grp in base_paths_df.groupby("scenario_id"):
        ax.plot(grp["timestamp"], grp["value"],
                color="grey", alpha=0.08, linewidth=0.7)
    ax.plot(base_summ["timestamp"], base_summ["median"],
            color="blue", linewidth=2, label="baseline median")
    ax.fill_between(base_summ["timestamp"], base_summ["q10"], base_summ["q90"],
                    color="blue", alpha=0.15, label="10-90% band")
    ax.plot(hist_idx_plt, history_y, color="black", linewidth=1.2, label="history")
    ax.set_title(f"Future baseline - all {N_SCENARIOS} sampled paths")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "future_baseline_paths.png", dpi=130)
    plt.close(fig)

    return {"summary": summary_all, "history": hist}

'''
    txt = txt[:m.start()] + NEW_FUT_TEXT + txt[m.end():]
    print("  OK  : scenario_future rewrite")
else:
    print("  FAIL: scenario_future NOT FOUND")

# ============================================================
# 8. main(): fix load call + add argparse + TARGET_DONG
# ============================================================
print("\n[8] main()")

OLD_MAIN = (
    'def main() -> None:\n'
    '    torch.manual_seed(SEED)\n'
    '    np.random.seed(SEED)\n'
    '\n'
    '    print("[Step A] Loading & aggregating data ...")\n'
    '    raw = load_and_aggregate(DATA_CSV)\n'
    '    raw.to_csv(OUT_DIR / "00_seoul_monthly_raw.csv")\n'
    '\n'
    '    print("[Step A] Cleaning missing + policy normalize ...")\n'
    '    clean = clean_missing_and_policy(raw)\n'
    '    clean.to_csv(OUT_DIR / "01_seoul_monthly_clean.csv")\n'
    '\n'
    '    print("[Step A] ADF diagnostic ...")\n'
    '    run_adf_report(clean, OUT_DIR / "02_adf_report.csv")\n'
    '\n'
    '    save_hyperparams()\n'
    '\n'
    '    print("[Step B/C/D] Scenario 1+2 backtest ...")\n'
    '    scenario_backtest(clean)\n'
    '\n'
    '    print("[Step E] Scenario 3 future + what-if ...")\n'
    '    scenario_future(clean)\n'
    '\n'
    '    print("\\n=== DONE ===  outputs in:", OUT_DIR)\n'
)
NEW_MAIN = (
    'def main() -> None:\n'
    '    import argparse\n'
    '    global TARGET_DONG\n'
    '    parser = argparse.ArgumentParser(description="Diffusion 시나리오 모델")\n'
    '    parser.add_argument("--target_dong", type=str, default=None,\n'
    '                        help="추론 대상 동 이름 (기본값: 서울 전체 평균)")\n'
    '    args, _ = parser.parse_known_args()\n'
    '    if args.target_dong is not None:\n'
    '        TARGET_DONG = args.target_dong\n'
    '    print(f"[Config] TARGET_DONG = {TARGET_DONG!r}")\n'
    '\n'
    '    torch.manual_seed(SEED)\n'
    '    np.random.seed(SEED)\n'
    '\n'
    '    print("[Step A] Loading panel data ...")\n'
    '    raw = load_panel(DATA_CSV)\n'
    '    raw.to_csv(OUT_DIR / "00_panel_raw.csv")\n'
    '\n'
    '    print("[Step A] Cleaning missing + policy normalize + log-return ...")\n'
    '    clean = clean_missing_and_policy(raw)\n'
    '    clean.to_csv(OUT_DIR / "01_panel_clean.csv")\n'
    '\n'
    '    print("[Step A] ADF diagnostic ...")\n'
    '    run_adf_report(clean, OUT_DIR / "02_adf_report.csv")\n'
    '\n'
    '    save_hyperparams()\n'
    '\n'
    '    print("[Step B/C/D] Scenario 1+2 backtest ...")\n'
    '    scenario_backtest(clean)\n'
    '\n'
    '    print("[Step E] Scenario 3 future + what-if ...")\n'
    '    scenario_future(clean)\n'
    '\n'
    '    print("\\n=== DONE ===  outputs in:", OUT_DIR)\n'
)
replace_one(OLD_MAIN, NEW_MAIN, "main() argparse + load_panel")

# ============================================================
# 9. Write
# ============================================================
src.write_text(txt, encoding="utf-8")
print(f"\n{'='*60}")
print(f"Patch 3 done. Lines: {len(txt.splitlines())}")

# Quick sanity
for kw in ["load_and_aggregate", "apply_scenario_trend", "y_tr, X_tr",
           "y_all, X_all", "full_df.loc[:", "train.index", "valid.index",
           "full_df.index"]:
    cnt = txt.count(kw)
    if cnt > 0:
        print(f"  WARN: '{kw}' still appears {cnt}x")

print("Stale-ref check done.")
