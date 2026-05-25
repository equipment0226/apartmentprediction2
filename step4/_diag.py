import pandas as pd
df = pd.read_csv("step4/future_summary.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

targets = ["2027-01-01", "2030-01-01", "2035-12-01"]
scens = [
    "baseline",
    "C1_stagflation_70s",
    "C2_recession_imf_type",
    "C4_goldilocks",
    "A1_rate_shock_hike_2022",
    "B1_policy_full_tight_moonjaein",
    "E3_replay_2020_2021_boom",
    "E4_replay_2022_shock",
]

base_vals = {}
base_sub = df[df["scenario"] == "baseline"].set_index("timestamp")
for t in targets:
    try:    base_vals[t] = base_sub.loc[t, "median"]
    except: base_vals[t] = float("nan")

print(f"{'scenario':35s}  {'2027':>8s}  {'2030':>8s}  {'2035':>8s}  {'2027_diff%':>10s}  {'2035_diff%':>10s}")
print("-" * 90)
for sc in scens:
    sub = df[df["scenario"] == sc].set_index("timestamp")
    vals = []
    for t in targets:
        try:    vals.append(sub.loc[t, "median"])
        except: vals.append(float("nan"))
    d27 = (vals[0] - base_vals[targets[0]]) / base_vals[targets[0]] * 100 if sc != "baseline" else 0.0
    d35 = (vals[2] - base_vals[targets[2]]) / base_vals[targets[2]] * 100 if sc != "baseline" else 0.0
    print(f"{sc:35s}  {vals[0]:>8.0f}  {vals[1]:>8.0f}  {vals[2]:>8.0f}  {d27:>+9.1f}%  {d35:>+9.1f}%")

print()
print("=== 시나리오별 q10/median/q90 스프레드 (2035-12) ===")
t35 = pd.Timestamp("2035-12-01")
for sc in scens:
    sub = df[df["scenario"] == sc].set_index("timestamp")
    try:
        q10 = sub.loc[t35, "q10"];  med = sub.loc[t35, "median"];  q90 = sub.loc[t35, "q90"]
        print(f"  {sc:35s}  q10={q10:7.0f}  med={med:7.0f}  q90={q90:7.0f}  spread={q90-q10:6.0f}")
    except:
        print(f"  {sc:35s}  N/A")
