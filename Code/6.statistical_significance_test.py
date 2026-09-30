import os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# ===================== Paths Configuration =====================
BASE_DIR = "."

gated_path = os.path.join(BASE_DIR, "Results", "Result_A_GatedNet_Alone", "Final_Paper_Results", "PerFold_Metrics.csv")
ml_path = os.path.join(BASE_DIR, "Results", "3_Baseline_ML_Results", "PerFold_Baseline_Metrics.csv")

if not os.path.exists(gated_path):
    raise FileNotFoundError(f"GatedNet result file not found: {gated_path}")
if not os.path.exists(ml_path):
    raise FileNotFoundError(f"ML baselines result file not found: {ml_path}")

df_gated = pd.read_csv(gated_path)
df_ml = pd.read_csv(ml_path)

dl_model_names = ["Baseline-CNN", "TextCNN", "LSTM", "BiLSTM"]
dl_fold_dfs = {}

for mname in dl_model_names:
    p = os.path.join(BASE_DIR, "Results", "Result_DL_Baselines_Hybrid", mname, "Final_Paper_Results",
                     "PerFold_Metrics.csv")
    if not os.path.exists(p):
        print(f"Warning: DL baseline file not found for {mname}: {p}")
        continue
    dl_fold_dfs[mname] = pd.read_csv(p)

out_dir = os.path.join(BASE_DIR, "Results", "4_Statistical_Test")
os.makedirs(out_dir, exist_ok=True)

metrics_list = ["AUC", "MCC", "F1", "ACC"]
all_comparison_records = []

compare_model_list = list(df_ml["Model"].unique()) + [m for m in dl_model_names if m in dl_fold_dfs]

# ===================== Wilcoxon Signed-Rank Test =====================
for cmp_model in compare_model_list:
    if cmp_model in df_ml["Model"].unique():
        df_cmp = df_ml[df_ml["Model"] == cmp_model].sort_values("Fold").reset_index(drop=True)
    else:
        df_cmp = dl_fold_dfs[cmp_model].reset_index(drop=True)

    for met in metrics_list:
        gated_arr = df_gated[met].values
        cmp_arr = df_cmp[met].values

        diff = gated_arr - cmp_arr

        if np.all(diff == 0):
            stat, pval = 0.0, 1.0
        else:
            try:
                stat, pval = wilcoxon(gated_arr, cmp_arr, alternative="greater")
            except Exception as e:
                print(f"Error computing Wilcoxon test for {cmp_model} on {met}: {e}")
                stat, pval = np.nan, np.nan

        all_comparison_records.append({
            "Compare_Model": cmp_model,
            "Metric": met,
            "GatedNet_Mean": round(np.mean(gated_arr), 4),
            "Baseline_Mean": round(np.mean(cmp_arr), 4),
            "statistic": stat,
            "p_value": round(pval, 5) if not np.isnan(pval) else np.nan,
            "sig": "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else "ns"))
        })

# ===================== Save and Display Results =====================
df_stat = pd.DataFrame(all_comparison_records)
out_file = os.path.join(out_dir, "Table6_wilcoxon_test.csv")
df_stat.to_csv(out_file, index=False)

print("=" * 70)
print("==== Wilcoxon Test Results (DeepAntiox-GatedNet VS Baselines) ====")
print("=" * 70)
print(df_stat.to_string(index=False))

print(f"\nStatistical test results successfully saved to: {out_file}")