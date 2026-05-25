# -*- coding: utf-8 -*-
"""Patch Diffusion.py: panel-aware fit_scalers + build_sliding_windows + remove drift"""

import re
from pathlib import Path

src = Path(__file__).parent / "Diffusion.py"
txt = src.read_text(encoding="utf-8")

# ============================================================
# 1. Replace fit_scalers
# ============================================================
OLD_FIT = (
    'def fit_scalers(train_df: pd.DataFrame):\n'
    '    """타겟과 covariate 를 각각 RobustScaler 로 적합.\n'
    '\n'
    '    [근거] target 과 exog 의 스케일 차이가 매우 큼 (가격 ~1e8, 금리 ~1e0).\n'
    '           각각 별도 scaler 로 fit → inverse_transform 시 누설 없음.\n'
    '    """\n'
    '    target_scaler = RobustScaler(quantile_range=(1, 99)).fit(train_df[[TARGET_COL]].values)\n'
    '    cov_cols = [c for c in train_df.columns if c != TARGET_COL]\n'
    '    cov_scaler = RobustScaler(quantile_range=(1, 99)).fit(train_df[cov_cols].values)\n'
    '    # quantile_range=(1,99) : (5,95) 대비 극단값까지 완전 보존.\n'
    '    # 2020-21 고부양 +93% / 2022 급락 -30% 를 outlier 취급 안 하고\n'
    '    # 스케일 내에 보존 → 모델이 장기 큰 진폭 변동도 자연스레운 그랄로 학습.\n'
    '    return target_scaler, cov_scaler, cov_cols'
)

NEW_FIT = (
    'def fit_scalers(train_df: pd.DataFrame):\n'
    '    """패널 학습 데이터 전체에 대해 Global RobustScaler 적합.\n'
    '\n'
    '    [근거] 모든 동을 하나의 스케일러로 fit -> Global Model 학습에서\n'
    '           동간 가격 스케일이 일치하여 inverse_transform 일관성 보장.\n'
    '           quantile_range=(1,99): 2020-21 +93% / 2022 -30% 등 구조적 급등락도\n'
    '           outlier 취급 안 하고 스케일 내 보존.\n'
    '    """\n'
    '    target_scaler = RobustScaler(quantile_range=(1, 99)).fit(\n'
    '        train_df[[TARGET_COL]].values)\n'
    '    cov_cols = [\n'
    '        c for c in train_df.columns\n'
    '        if c != TARGET_COL\n'
    '        and c not in ID_COLS + [TIME_COL]\n'
    '        and pd.api.types.is_numeric_dtype(train_df[c])\n'
    '    ]\n'
    '    cov_scaler = RobustScaler(quantile_range=(1, 99)).fit(\n'
    '        train_df[cov_cols].values)\n'
    '    return target_scaler, cov_scaler, cov_cols'
)

if OLD_FIT in txt:
    txt = txt.replace(OLD_FIT, NEW_FIT, 1)
    print("fit_scalers: OK")
else:
    print("fit_scalers: NOT FOUND – trying fallback")
    # fallback: regex
    pat = r'def fit_scalers\(train_df.*?return target_scaler, cov_scaler, cov_cols'
    m = re.search(pat, txt, re.DOTALL)
    if m:
        print(f"  found at {m.start()}:{m.end()}")
        txt = txt[:m.start()] + NEW_FIT + txt[m.end():]
        print("fit_scalers: regex OK")
    else:
        print("fit_scalers: FAILED")

# ============================================================
# 2. Remove transform_df (no longer needed as standalone)
#    Keep it but it's still useful for single-series inference
# ============================================================

# ============================================================
# 3. Replace build_sliding_windows
# ============================================================
OLD_SLIDE = (
    'def build_sliding_windows(y: np.ndarray, X: np.ndarray,\n'
    '                          past_len: int, horizon: int):\n'
    '    """슬라이딩 윈도우 (past_y, past_X, future_X) → future_y 학습쌍 생성.\n'
    '\n'
    '    [근거]\n'
    '      - 학습 1 sample = (인코더 36 + 디코더 12) 의 multivariate window.\n'
    '      - exogenous (covariate) 는 학습 시점에서 미래 구간이 "known input" 으로\n'
    '        주어진다고 가정 (sklearn-style covariate / TFT 의 time_varying_known\n'
    '        대응). 이를 통해 What-if 시나리오를 covariate 만 바꿔 생성 가능.\n'
    '    """\n'
    '    T = len(y)\n'
    '    end = T - past_len - horizon + 1\n'
    '    past_y_list, past_X_list, fut_X_list, fut_y_list = [], [], [], []\n'
    '    for s in range(end):\n'
    '        past_y_list.append(y[s : s + past_len])\n'
    '        past_X_list.append(X[s : s + past_len])\n'
    '        fut_X_list .append(X[s + past_len : s + past_len + horizon])\n'
    '        fut_y_list .append(y[s + past_len : s + past_len + horizon])\n'
    '    return (np.stack(past_y_list),\n'
    '            np.stack(past_X_list),\n'
    '            np.stack(fut_X_list),\n'
    '            np.stack(fut_y_list))'
)

NEW_SLIDE = (
    'def build_sliding_windows(panel_df: pd.DataFrame,\n'
    '                          target_scaler: RobustScaler,\n'
    '                          cov_scaler: RobustScaler,\n'
    '                          cov_cols: list,\n'
    '                          past_len: int, horizon: int):\n'
    '    """패널 데이터를 동별로 순회하여 슬라이딩 윈도우 생성 후 vstack.\n'
    '\n'
    '    [근거]\n'
    '      - 동별 독립 시퀀스: 동 A의 윈도우와 동 B의 윈도우는 서로 다른\n'
    '        시계열이므로 옥스테킹 없음. Global Model 학습.\n'
    '      - 학습 샘플 수: ~(T-past_len-horizon+1) x N_dong 개.\n'
    '        단일 서울 집계 ~140 윈도우 대비 데이터 기아 해소.\n'
    '    """\n'
    '    all_py, all_pX, all_fX, all_fy = [], [], [], []\n'
    '    for dong, grp in panel_df.groupby("dong"):\n'
    '        grp = grp.sort_values(TIME_COL)\n'
    '        y_d = target_scaler.transform(\n'
    '            grp[[TARGET_COL]].values).astype(np.float32).flatten()\n'
    '        X_d = cov_scaler.transform(\n'
    '            grp[cov_cols].values).astype(np.float32)\n'
    '        T = len(y_d)\n'
    '        end = T - past_len - horizon + 1\n'
    '        if end < 1:\n'
    '            continue\n'
    '        for s in range(end):\n'
    '            all_py.append(y_d[s : s + past_len])\n'
    '            all_pX.append(X_d[s : s + past_len])\n'
    '            all_fX.append(X_d[s + past_len : s + past_len + horizon])\n'
    '            all_fy.append(y_d[s + past_len : s + past_len + horizon])\n'
    '    return (np.stack(all_py),\n'
    '            np.stack(all_pX),\n'
    '            np.stack(all_fX),\n'
    '            np.stack(all_fy))'
)

if OLD_SLIDE in txt:
    txt = txt.replace(OLD_SLIDE, NEW_SLIDE, 1)
    print("build_sliding_windows: OK")
else:
    print("build_sliding_windows: NOT FOUND – trying regex")
    pat = r'def build_sliding_windows\(y:.*?return \(np\.stack\(past_y_list\),.*?np\.stack\(fut_y_list\)\)'
    m = re.search(pat, txt, re.DOTALL)
    if m:
        txt = txt[:m.start()] + NEW_SLIDE + txt[m.end():]
        print("build_sliding_windows: regex OK")
    else:
        print("build_sliding_windows: FAILED")

# ============================================================
# 4. Remove SCENARIO_ANNUAL_DRIFT_PCT block (comment + dict)
# ============================================================
# Find the big comment block that starts before SCENARIO_ANNUAL_DRIFT_PCT
drift_marker = "SCENARIO_ANNUAL_DRIFT_PCT = {"
idx_start = txt.find("# ---------------------------------------------------------------------------\n# Diffusion 의 conditional sampling 만으로는")
if idx_start == -1:
    # Try alternate marker
    idx_start = txt.find("# Diffusion 의 conditional sampling 만으로는")
    if idx_start == -1:
        idx_start = txt.find(drift_marker)
        # Back up to find comment block
        tmp = txt.rfind("\n#", 0, idx_start)
        while tmp != -1 and txt[tmp:idx_start].count('\n') < 30:
            candidate = txt.rfind("\n#", 0, tmp)
            if candidate == -1:
                break
            tmp = candidate
        idx_start = tmp + 1 if tmp != -1 else idx_start

# Find end of SCENARIO_ANNUAL_DRIFT_PCT dict
idx_dict_start = txt.find(drift_marker)
idx_dict_end = txt.find("\n}", idx_dict_start)
idx_dict_end += 2  # include closing } and newline

print(f"SCENARIO_ANNUAL_DRIFT_PCT found at {idx_dict_start}, ends at {idx_dict_end}")

# ============================================================
# 5. Remove SCENARIO_PHASES block
# ============================================================
phases_marker = "SCENARIO_PHASES = {"
idx_phases_start = txt.find(phases_marker)
# Find comment block before it
idx_phases_comment = txt.rfind("# -----------", 0, idx_phases_start)
# Find end of SCENARIO_PHASES - it's a nested dict, find closing }
# Count braces
pos = txt.find("{", idx_phases_start)
depth = 0
while pos < len(txt):
    if txt[pos] == '{':
        depth += 1
    elif txt[pos] == '}':
        depth -= 1
        if depth == 0:
            idx_phases_end = pos + 1
            # consume trailing newlines
            while idx_phases_end < len(txt) and txt[idx_phases_end] in '\n':
                idx_phases_end += 1
            break
    pos += 1

print(f"SCENARIO_PHASES found at {idx_phases_start} (comment at {idx_phases_comment}), ends at {idx_phases_end}")

# Remove from comment before ANNUAL_DRIFT to end of SCENARIO_PHASES
txt = txt[:idx_start] + txt[idx_phases_end:]
print("SCENARIO_ANNUAL_DRIFT_PCT + SCENARIO_PHASES: removed OK")

# ============================================================
# 6. Remove build_drift_schedule function
# ============================================================
pat = r'\ndef build_drift_schedule\(.*?\n\n'
m = re.search(pat, txt, re.DOTALL)
if m:
    txt = txt[:m.start()] + "\n\n" + txt[m.end():]
    print("build_drift_schedule: removed OK")
else:
    print("build_drift_schedule: NOT FOUND")

# ============================================================
# 7. Remove apply_scenario_trend function
# ============================================================
pat = r'\ndef apply_scenario_trend\(.*?\n\n'
m = re.search(pat, txt, re.DOTALL)
if m:
    txt = txt[:m.start()] + "\n\n" + txt[m.end():]
    print("apply_scenario_trend: removed OK")
else:
    print("apply_scenario_trend: NOT FOUND")

# ============================================================
# 8. Remove apply_scenario_trend calls in scenario_backtest
# ============================================================
# There are two blocks to remove:
# (a) past_y_last_orig calculation + apply call in backtest
# (b) same in future

for fn_name in ["backtest", "future"]:
    # Remove apply_scenario_trend call lines
    pattern = r'        # 시나리오 의미를 반영한 구조적 trend 백본 주입.*?\n        paths_orig = apply_scenario_trend\(paths_orig, past_y_last_orig, name\)\n'
    m = re.search(pattern, txt, re.DOTALL)
    if m:
        txt = txt[:m.start()] + txt[m.end():]
        print(f"apply_scenario_trend call ({fn_name}): removed OK")
    else:
        # Try shorter pattern
        pattern2 = r'        paths_orig = apply_scenario_trend\(paths_orig, past_y_last_orig, name\)\n'
        m2 = re.search(pattern2, txt)
        if m2:
            txt = txt[:m2.start()] + txt[m2.end():]
            print(f"apply_scenario_trend call ({fn_name}): removed via short pattern")
        else:
            print(f"apply_scenario_trend call ({fn_name}): NOT FOUND")

# Remove the past_y_last_orig assignments
pattern3 = r'    # 시나리오 trend 시프트 기준점.*?\n    past_y_last_orig = float\(inverse_target\(\n        past_y_np\[-1:\]\.reshape\(1, 1\), target_scaler\)\.ravel\(\)\[0\]\)\n'
for _ in range(5):
    m = re.search(pattern3, txt, re.DOTALL)
    if m:
        txt = txt[:m.start()] + txt[m.end():]
        print("past_y_last_orig: removed OK")
    else:
        break

# Also remove simpler version without comment
pattern4 = r'    past_y_last_orig = float\(inverse_target\(\n        past_y_np\[-1:\]\.reshape\(1, 1\), target_scaler\)\.ravel\(\)\[0\]\)\n'
for _ in range(5):
    m = re.search(pattern4, txt)
    if m:
        txt = txt[:m.start()] + txt[m.end():]
        print("past_y_last_orig (simple): removed OK")
    else:
        break

# Remaining inline comment about trend
pattern5 = r'        # 시나리오 의미 반영 trend 백본 \(long-horizon 분기\)\n        paths_orig = apply_scenario_trend\(paths_orig, past_y_last_orig, name\)\n'
for _ in range(5):
    m = re.search(pattern5, txt)
    if m:
        txt = txt[:m.start()] + txt[m.end():]
        print("apply_scenario_trend (future call): removed OK")
    else:
        break

# ============================================================
# 9. Add TARGET_DONG constant near the top config section
# ============================================================
if "TARGET_DONG" not in txt:
    # Insert after SEED definition
    marker = "SEED              = 42\n"
    if marker in txt:
        insert_after = txt.find(marker) + len(marker)
        add_block = (
            "\n# ---------------------------------------------------------------------------\n"
            "# 추론 대상 동: None 이면 서울 전체 평균(mean aggregation) 사용,\n"
            "# 문자열이면 해당 동의 과거 시퀀스를 사용하여 100-path 생성.\n"
            "# CLI --target_dong 인자로 덮어쓸 수 있음.\n"
            "# ---------------------------------------------------------------------------\n"
            "TARGET_DONG: str | None = None   # 예: \"개포동\", \"압구정동\"\n"
        )
        txt = txt[:insert_after] + add_block + txt[insert_after:]
        print("TARGET_DONG: added OK")
    else:
        print("TARGET_DONG: SEED marker not found")
else:
    print("TARGET_DONG: already present")

# ============================================================
# 10. Write patched file
# ============================================================
src.write_text(txt, encoding="utf-8")
print(f"\nPatch complete. File written: {src}")
