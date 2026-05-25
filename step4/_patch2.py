# -*- coding: utf-8 -*-
"""Patch 2: Remove orphaned apply_scenario_trend body, fix main() calls"""

import re
from pathlib import Path

src = Path(__file__).parent / "Diffusion.py"
txt = src.read_text(encoding="utf-8")

# ============================================================
# 1. Remove orphaned apply_scenario_trend body fragment
#    (from '# [시나리오 구조적...' to 'return paths_orig + shift[None, :]')
# ============================================================

# Find the orphaned comment block that precedes the leftover body
marker_start = "\n# ---------------------------------------------------------------------------\n# [시나리오 구조적 trend prior"
marker_end_str = "    return paths_orig + shift[None, :]\n"

idx_s = txt.find(marker_start)
if idx_s != -1:
    idx_e = txt.find(marker_end_str, idx_s)
    if idx_e != -1:
        idx_e += len(marker_end_str)
        txt = txt[:idx_s] + "\n" + txt[idx_e:]
        print("Orphaned apply_scenario_trend body: removed OK")
    else:
        print("End marker not found for orphaned body")
else:
    print("Orphaned body start not found")

# ============================================================
# 2. Fix main(): load_and_aggregate -> load_panel
# ============================================================
old_main_load = '    raw = load_and_aggregate(DATA_CSV)\n'
new_main_load = '    raw = load_panel(DATA_CSV)\n'
if old_main_load in txt:
    txt = txt.replace(old_main_load, new_main_load, 1)
    print("main() load_panel: OK")
else:
    print("main() load call: NOT FOUND")

# ============================================================
# 3. Fix scenario_backtest and scenario_future:
#    Replace old y_tr/X_tr split + build_sliding_windows calls
# ============================================================

# backtest: old call
old_bt_data = (
    '    target_scaler, cov_scaler, cov_cols = fit_scalers(train)\n'
    '    y_tr, X_tr = transform_df(train, target_scaler, cov_scaler, cov_cols)\n'
    '    y_va, X_va = transform_df(valid, target_scaler, cov_scaler, cov_cols)\n'
    '\n'
    '    # 학습 윈도우\n'
    '    p_y, p_X, f_X, f_y = build_sliding_windows(\n'
    '        y_tr, X_tr, PAST_LEN, BACKTEST_HORIZON)\n'
)
new_bt_data = (
    '    target_scaler, cov_scaler, cov_cols = fit_scalers(train)\n'
    '\n'
    '    # 패널 전체 학습 윈도우 (동별 루프 + vstack)\n'
    '    p_y, p_X, f_X, f_y = build_sliding_windows(\n'
    '        train, target_scaler, cov_scaler, cov_cols,\n'
    '        PAST_LEN, BACKTEST_HORIZON)\n'
)
if old_bt_data in txt:
    txt = txt.replace(old_bt_data, new_bt_data, 1)
    print("backtest build_sliding_windows: OK")
else:
    print("backtest build_sliding_windows: NOT FOUND – trying partial")
    # Try to just fix the build_sliding_windows call
    old_bsw = (
        '    p_y, p_X, f_X, f_y = build_sliding_windows(\n'
        '        y_tr, X_tr, PAST_LEN, BACKTEST_HORIZON)\n'
    )
    new_bsw = (
        '    p_y, p_X, f_X, f_y = build_sliding_windows(\n'
        '        train, target_scaler, cov_scaler, cov_cols,\n'
        '        PAST_LEN, BACKTEST_HORIZON)\n'
    )
    if old_bsw in txt:
        txt = txt.replace(old_bsw, new_bsw, 1)
        print("backtest BSW call (partial): OK")
    else:
        print("backtest BSW: FAILED")

# future: old call
old_fut_bsw = (
    '    p_y, p_X, f_X, f_y = build_sliding_windows(\n'
    '        y_all, X_all, PAST_LEN, BACKTEST_HORIZON)\n'
)
new_fut_bsw = (
    '    p_y, p_X, f_X, f_y = build_sliding_windows(\n'
    '        full_df, target_scaler, cov_scaler, cov_cols,\n'
    '        PAST_LEN, BACKTEST_HORIZON)\n'
)
if old_fut_bsw in txt:
    txt = txt.replace(old_fut_bsw, new_fut_bsw, 1)
    print("future BSW call: OK")
else:
    print("future BSW call: NOT FOUND")

# ============================================================
# 4. Fix backtest inference past context:
#    Replace y_tr[-PAST_LEN:] / X_tr[-PAST_LEN:] with TARGET_DONG-aware
# ============================================================
old_bt_past = (
    '    # 추론 : past = train 마지막 PAST_LEN 개월\n'
    '    past_y_np = y_tr[-PAST_LEN:]\n'
    '    past_X_np = X_tr[-PAST_LEN:]\n'
    '    pred_idx  = valid.index\n'
    '    truth     = valid[TARGET_COL].values\n'
)
new_bt_past = (
    '    # 추론: TARGET_DONG 동 또는 서울 전체 평균으로 past context 구성\n'
    '    pred_idx = pd.DatetimeIndex(sorted(valid[TIME_COL].unique()))\n'
    '    tgt_dong = TARGET_DONG\n'
    '    if tgt_dong is not None and tgt_dong in train["dong"].values:\n'
    '        td_tr = train[train["dong"] == tgt_dong].sort_values(TIME_COL)\n'
    '        td_va = valid[valid["dong"] == tgt_dong].sort_values(TIME_COL)\n'
    '        y_td = target_scaler.transform(\n'
    '            td_tr[[TARGET_COL]].values).astype(np.float32).flatten()\n'
    '        X_td = cov_scaler.transform(\n'
    '            td_tr[cov_cols].values).astype(np.float32)\n'
    '        y_va_arr = target_scaler.transform(\n'
    '            td_va[[TARGET_COL]].values).astype(np.float32).flatten()\n'
    '        X_va = cov_scaler.transform(\n'
    '            td_va[cov_cols].values).astype(np.float32)\n'
    '        past_y_np = y_td[-PAST_LEN:]\n'
    '        past_X_np = X_td[-PAST_LEN:]\n'
    '        truth = td_va[TARGET_COL].values\n'
    '    else:\n'
    '        # Seoul mean fallback\n'
    '        mean_tr = train.groupby(TIME_COL)[cov_cols + [TARGET_COL]].mean().sort_index()\n'
    '        mean_va = valid.groupby(TIME_COL)[cov_cols + [TARGET_COL]].mean().sort_index()\n'
    '        y_td = target_scaler.transform(\n'
    '            mean_tr[[TARGET_COL]].values).astype(np.float32).flatten()\n'
    '        X_td = cov_scaler.transform(\n'
    '            mean_tr[cov_cols].values).astype(np.float32)\n'
    '        y_va_arr = target_scaler.transform(\n'
    '            mean_va[[TARGET_COL]].values).astype(np.float32).flatten()\n'
    '        X_va = cov_scaler.transform(\n'
    '            mean_va[cov_cols].values).astype(np.float32)\n'
    '        past_y_np = y_td[-PAST_LEN:]\n'
    '        past_X_np = X_td[-PAST_LEN:]\n'
    '        truth = mean_va[TARGET_COL].values\n'
)
if old_bt_past in txt:
    txt = txt.replace(old_bt_past, new_bt_past, 1)
    print("backtest past context: OK")
else:
    print("backtest past context: NOT FOUND")

# ============================================================
# 5. Fix future inference past context:
#    Replace y_all[-PAST_LEN:] / X_all[-PAST_LEN:] with TARGET_DONG-aware
# ============================================================
old_fut_past = (
    '    past_y_np = y_all[-PAST_LEN:]\n'
    '    past_X_np = X_all[-PAST_LEN:]\n'
)
new_fut_past = (
    '    # TARGET_DONG 동 또는 서울 전체 평균으로 past context 구성\n'
    '    tgt_dong = TARGET_DONG\n'
    '    if tgt_dong is not None and tgt_dong in full_df["dong"].values:\n'
    '        td_all = full_df[full_df["dong"] == tgt_dong].sort_values(TIME_COL)\n'
    '        y_td_all = target_scaler.transform(\n'
    '            td_all[[TARGET_COL]].values).astype(np.float32).flatten()\n'
    '        X_td_all = cov_scaler.transform(\n'
    '            td_all[cov_cols].values).astype(np.float32)\n'
    '        past_y_np = y_td_all[-PAST_LEN:]\n'
    '        past_X_np = X_td_all[-PAST_LEN:]\n'
    '    else:\n'
    '        # Seoul mean fallback\n'
    '        mean_all = full_df.groupby(TIME_COL)[cov_cols + [TARGET_COL]].mean().sort_index()\n'
    '        past_y_np = target_scaler.transform(\n'
    '            mean_all[[TARGET_COL]].values).astype(np.float32).flatten()[-PAST_LEN:]\n'
    '        past_X_np = cov_scaler.transform(\n'
    '            mean_all[cov_cols].values).astype(np.float32)[-PAST_LEN:]\n'
)
if old_fut_past in txt:
    txt = txt.replace(old_fut_past, new_fut_past, 1)
    print("future past context: OK")
else:
    print("future past context: NOT FOUND")

# ============================================================
# 6. Fix main(): add argparse for --target_dong, update calls
# ============================================================
old_main = (
    'def main() -> None:\n'
    '    torch.manual_seed(SEED)\n'
    '    np.random.seed(SEED)\n'
    '\n'
    '    print("[Step A] Loading & aggregating data ...")\n'
    '    raw = load_panel(DATA_CSV)\n'
    '    raw.to_csv(OUT_DIR / "00_seoul_monthly_raw.csv")\n'
)
new_main = (
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
)
if old_main in txt:
    txt = txt.replace(old_main, new_main, 1)
    print("main() argparse: OK")
else:
    print("main() argparse: NOT FOUND")

# Also fix the clean.to_csv line
txt = txt.replace(
    '    clean.to_csv(OUT_DIR / "01_seoul_monthly_clean.csv")',
    '    clean.to_csv(OUT_DIR / "01_panel_clean.csv")',
)

# ============================================================
# 7. Fix scenario_backtest: remove now-unused y_tr/X_tr/y_va/X_va transform_df calls
# ============================================================
# The old code had transform_df calls after fit_scalers in backtest
old_transform = (
    '    target_scaler, cov_scaler, cov_cols = fit_scalers(train)\n'
    '\n'
    '    # 패널 전체 학습 윈도우 (동별 루프 + vstack)\n'
)
# Already handled above; just double-check there are no stale transform_df calls
# that we need y_tr/X_tr for in backtest's training section
remaining_ytr = txt.count('y_tr, X_tr = transform_df') + txt.count('y_va, X_va = transform_df')
print(f"Remaining stale transform_df calls: {remaining_ytr}")

# ============================================================
# 8. Fix future: remove stale transform_df calls
# ============================================================
old_fut_transform = (
    '    target_scaler, cov_scaler, cov_cols = fit_scalers(full_df)\n'
    '    y_all, X_all = transform_df(full_df, target_scaler, cov_scaler, cov_cols)\n'
    '\n'
    '    # 학습 윈도우 (horizon 은 backtest 와 동일 12 사용'
)
new_fut_transform = (
    '    target_scaler, cov_scaler, cov_cols = fit_scalers(full_df)\n'
    '\n'
    '    # 학습 윈도우 (horizon 은 backtest 와 동일 12 사용'
)
if old_fut_transform in txt:
    txt = txt.replace(old_fut_transform, new_fut_transform, 1)
    print("future transform_df: removed OK")
else:
    print("future transform_df: NOT FOUND (may already be removed)")

# Also fix future history chart that uses y_all
old_hist_yall = '    history_y = inverse_target(y_all, target_scaler)\n'
new_hist_yall = (
    '    # history: Seoul mean for chart\n'
    '    mean_all_hist = full_df.groupby(TIME_COL)[TARGET_COL].mean().sort_index()\n'
    '    history_y = inverse_target(\n'
    '        target_scaler.transform(mean_all_hist.values.reshape(-1,1)).astype(np.float32).flatten(),\n'
    '        target_scaler)\n'
)
if old_hist_yall in txt:
    txt = txt.replace(old_hist_yall, new_hist_yall, 1)
    print("future history_y: OK")
else:
    print("future history_y: NOT FOUND")

# Fix backtest hist_orig
old_hist_tr = (
    '    hist_idx  = train.index\n'
    '    hist_orig = inverse_target(y_tr, target_scaler)\n'
)
new_hist_tr = (
    '    hist_idx  = pd.DatetimeIndex(sorted(train[TIME_COL].unique()))\n'
    '    mean_tr_hist = train.groupby(TIME_COL)[TARGET_COL].mean().sort_index()\n'
    '    hist_orig = inverse_target(\n'
    '        target_scaler.transform(\n'
    '            mean_tr_hist.values.reshape(-1,1)).astype(np.float32).flatten(),\n'
    '        target_scaler)\n'
)
if old_hist_tr in txt:
    txt = txt.replace(old_hist_tr, new_hist_tr, 1)
    print("backtest hist_orig: OK")
else:
    print("backtest hist_orig: NOT FOUND")

# Fix backtest train/valid split (needs to be by TIME_COL not index)
old_split = (
    '    train = full_df.loc[:TRAIN_END]\n'
    '    valid = full_df.loc[VALID_START:VALID_END]\n'
)
new_split = (
    '    train = full_df[full_df[TIME_COL] <= TRAIN_END]\n'
    '    valid = full_df[(full_df[TIME_COL] >= VALID_START) &\n'
    '                    (full_df[TIME_COL] <= VALID_END)]\n'
)
if old_split in txt:
    txt = txt.replace(old_split, new_split, 1)
    print("backtest split: OK")
else:
    print("backtest split: NOT FOUND")

# ============================================================
# 9. Write final file
# ============================================================
src.write_text(txt, encoding="utf-8")
print(f"\nPatch 2 complete. File written: {src}")
