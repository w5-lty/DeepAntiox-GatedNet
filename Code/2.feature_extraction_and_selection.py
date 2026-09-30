# coding: utf-8
import pandas as pd
import numpy as np
import os
import warnings
import json
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# ===================== Global configuration for SCI figures =====================
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['xtick.major.width'] = 0.8
plt.rcParams['ytick.major.width'] = 0.8
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['figure.dpi'] = 1000
plt.rcParams['savefig.dpi'] = 1000
plt.rcParams['savefig.format'] = 'jpg'
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['legend.frameon'] = False

# ===================== Path Configuration =====================
BASE = "./Results"  # Modified to relative path for better portability
INPUT_CSV = os.path.join(BASE, "Result_AnOxPePred/v2_output", "03_p90_full_db_AnOxPePred_v2.csv")
OUT_ROOT = os.path.join(BASE, "2_Feature_Extraction_Selection_v2")
FIG_PATH = os.path.join(OUT_ROOT, "figures")
DATA_OUT = os.path.join(OUT_ROOT, "csv_output")

os.makedirs(OUT_ROOT, exist_ok=True)
os.makedirs(FIG_PATH, exist_ok=True)
os.makedirs(DATA_OUT, exist_ok=True)

STANDARD_AA = {'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
               'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'}
STANDARD_AA_LIST = list(STANDARD_AA)

CTD_PROPERTY_GROUPS = {
    'Polarity': {'Polar': ['R', 'K', 'E', 'D', 'Q', 'N'], 'Neutral': ['G', 'A', 'S', 'T', 'P', 'H', 'Y'],
                 'Nonpolar': ['F', 'L', 'I', 'M', 'W', 'V', 'C']},
    'Hydrophobicity': {'Strong': ['A', 'F', 'G', 'I', 'L', 'M', 'P', 'V', 'W'], 'Medium': ['C', 'H', 'Y'],
                       'Weak': ['D', 'E', 'N', 'Q', 'K', 'R', 'S', 'T']},
    'Charge': {'Positive': ['K', 'R'], 'Negative': ['D', 'E'],
               'Neutral': ['A', 'N', 'C', 'Q', 'G', 'H', 'I', 'L', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']},
    'SideChainVolume': {'Large': ['F', 'I', 'L', 'M', 'W', 'V'],
                        'Medium': ['A', 'C', 'H', 'K', 'N', 'Q', 'R', 'S', 'T', 'Y'], 'Small': ['D', 'E', 'G', 'P']},
    'Polarizability': {'High': ['F', 'H', 'I', 'L', 'M', 'W', 'V'],
                       'Medium': ['A', 'C', 'K', 'N', 'Q', 'R', 'S', 'T', 'Y'], 'Low': ['D', 'E', 'G', 'P']},
    'IsoelectricPoint': {'High': ['K', 'R'],
                         'Medium': ['A', 'C', 'F', 'G', 'H', 'I', 'L', 'M', 'N', 'P', 'Q', 'S', 'T', 'V', 'W', 'Y'],
                         'Low': ['D', 'E']},
    'HDonor': {'Many': ['R', 'K', 'H', 'N', 'Q'],
               'Few': ['A', 'C', 'F', 'G', 'I', 'L', 'M', 'P', 'S', 'T', 'V', 'W', 'Y'], 'None': ['D', 'E']},
    'HAcceptor': {'Many': ['D', 'E', 'N', 'Q'],
                  'Few': ['A', 'C', 'F', 'G', 'H', 'I', 'L', 'M', 'P', 'R', 'S', 'T', 'V', 'W', 'Y'], 'None': ['K']},
    'HydrophobicityIndex': {'High': ['I', 'V', 'L', 'F', 'C', 'M', 'A'], 'Medium': ['G', 'T', 'W', 'Y', 'P', 'H'],
                            'Low': ['E', 'D', 'R', 'K', 'Q', 'N', 'S']},
    'AAType': {'Aliphatic': ['A', 'G', 'I', 'L', 'P', 'V'], 'Aromatic': ['F', 'H', 'W', 'Y'],
               'PolarUncharged': ['C', 'N', 'Q', 'S', 'T'], 'Positive': ['K', 'R'], 'Negative': ['D', 'E']},
    'SideChainCharge': {'Positive': ['K', 'R', 'H'], 'Negative': ['D', 'E'],
                        'Neutral': ['A', 'C', 'F', 'G', 'I', 'L', 'M', 'N', 'P', 'Q', 'S', 'T', 'V', 'W', 'Y']},
    'SideChainAromatic': {'Aromatic': ['F', 'H', 'W', 'Y'],
                          'NonAromatic': ['A', 'C', 'D', 'E', 'G', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T',
                                          'V']},
    'SideChainHBond': {'Donor': ['R', 'K', 'H', 'N', 'Q'], 'Acceptor': ['D', 'E', 'N', 'Q', 'S', 'T', 'Y'],
                       'None': ['A', 'C', 'F', 'G', 'I', 'L', 'M', 'P', 'V', 'W']}
}

AA_PHYSICOCHEMICAL = {
    'A': [1.8, 0, 6.01, 8.1, 89.09, 67], 'C': [2.5, 0, 5.07, 5.5, 121.16, 86],
    'D': [-3.5, -1, 2.77, 13.0, 133.1, 91], 'E': [-3.5, -1, 3.22, 12.3, 147.13, 109],
    'F': [2.8, 0, 5.48, 5.2, 165.19, 135], 'G': [-0.4, 0, 5.97, 9.0, 75.07, 48],
    'H': [-3.2, 1, 7.59, 10.4, 155.16, 118], 'I': [4.5, 0, 6.02, 5.2, 131.18, 124],
    'K': [-3.9, 1, 9.74, 11.3, 146.19, 135], 'L': [3.8, 0, 6.0, 4.9, 131.18, 124],
    'M': [1.9, 0, 5.71, 5.7, 149.21, 124], 'N': [-3.5, 0, 5.41, 11.6, 132.12, 96],
    'P': [-1.6, 0, 6.3, 8.0, 115.13, 90], 'Q': [-3.5, 0, 5.65, 10.5, 146.15, 114],
    'R': [-4.5, 1, 10.76, 9.1, 174.2, 148], 'S': [-0.8, 0, 5.68, 9.2, 105.09, 73],
    'T': [-0.7, 0, 5.66, 8.6, 119.12, 93], 'V': [4.2, 0, 6.02, 5.9, 117.15, 105],
    'W': [-0.9, 0, 5.89, 5.4, 204.23, 163], 'Y': [-1.3, 0, 5.63, 6.2, 181.19, 141]
}
AA_PROPERTIES = ['Hydrophobicity', 'NetCharge', 'IsoelectricPoint', 'Polarity', 'MolecularWeight', 'SideChainVolume']


class PeptideFeatureExtractor:
    def __init__(self):
        self.aa_list = STANDARD_AA_LIST
        self.ctd_groups = CTD_PROPERTY_GROUPS
        self.aa_phys = AA_PHYSICOCHEMICAL
        self.aa_props = AA_PROPERTIES

    def extract_aac(self, seq):
        seq_len = len(seq)
        aac_values = [round(seq.count(aa) / seq_len, 6) for aa in self.aa_list]
        aac_names = [f"AAC_{aa}" for aa in self.aa_list]
        return aac_values, aac_names

    def extract_dpc(self, seq):
        seq_len = len(seq)
        dpc_values, dpc_names = [], []
        for aa1 in self.aa_list:
            for aa2 in self.aa_list:
                dipeptide = aa1 + aa2
                count = seq.count(dipeptide)
                value = round(count / (seq_len - 1), 6) if seq_len > 1 else 0
                dpc_values.append(value)
                dpc_names.append(f"DPC_{dipeptide}")
        return dpc_values, dpc_names

    def extract_asdc(self, seq, max_skip=3):
        asdc_values, asdc_names = [], []
        seq_len = len(seq)
        for skip in range(max_skip + 1):
            for aa1 in self.aa_list:
                for aa2 in self.aa_list:
                    count = 0
                    for i in range(seq_len - skip - 1):
                        if seq[i] == aa1 and seq[i + skip + 1] == aa2:
                            count += 1
                    denom = seq_len - skip - 1
                    val = round(count / denom, 6) if denom > 0 else 0
                    asdc_values.append(val)
                    asdc_names.append(f"ASDC_skip{skip}_{aa1}{aa2}")
        return asdc_values, asdc_names

    def extract_cksaap(self, seq, max_gap=2):
        cksaap_values, cksaap_names = [], []
        seq_len = len(seq)
        for gap in range(max_gap + 1):
            for aa1 in self.aa_list:
                for aa2 in self.aa_list:
                    cnt = 0
                    for i in range(seq_len - gap - 1):
                        if seq[i] == aa1 and seq[i + gap + 1] == aa2:
                            cnt += 1
                    denom = seq_len - gap - 1
                    val = round(cnt / denom, 6) if denom > 0 else 0
                    cksaap_values.append(val)
                    cksaap_names.append(f"CKSAAP_gap{gap}_{aa1}{aa2}")
        return cksaap_values, cksaap_names

    def extract_ctd(self, seq):
        ctd_values, ctd_names = [], []
        seq_len = len(seq)
        for prop_name, groups in self.ctd_groups.items():
            for g_name, g_aa in groups.items():
                aa_cnt = sum(1 for aa in seq if aa in g_aa)
                c_val = round(aa_cnt / seq_len, 6)
                ctd_values.append(c_val)
                ctd_names.append(f"CTD_{prop_name}_C_{g_name}")

            trans = 0
            if seq_len > 1:
                for i in range(seq_len - 1):
                    aa_i = seq[i]
                    aa_j = seq[i + 1]
                    g1_candidates = [k for k, v in groups.items() if aa_i in v]
                    g2_candidates = [k for k, v in groups.items() if aa_j in v]
                    g1 = g1_candidates[0] if len(g1_candidates) > 0 else None
                    g2 = g2_candidates[0] if len(g2_candidates) > 0 else None
                    if g1 is not None and g2 is not None and g1 != g2:
                        trans += 1
            t_val = round(trans / (seq_len - 1), 6) if seq_len > 1 else 0.0
            ctd_values.append(t_val)
            ctd_names.append(f"CTD_{prop_name}_T")

            for g_name, g_aa in groups.items():
                poslist = [i + 1 for i, aa in enumerate(seq) if aa in g_aa]
                if not poslist:
                    dvals = [0.0] * 5
                else:
                    dvals = [round(np.percentile(poslist, p) / seq_len, 6) for p in [0, 25, 50, 70, 100]]
                ctd_values.extend(dvals)
                ctd_names.extend([f"CTD_{prop_name}_D_{g_name}_P0",
                                  f"CTD_{prop_name}_D_{g_name}_P25",
                                  f"CTD_{prop_name}_D_{g_name}_P50",
                                  f"CTD_{prop_name}_D_{g_name}_P70",
                                  f"CTD_{prop_name}_D_{g_name}_P100"])
        return ctd_values, ctd_names

    def extract_paac(self, seq, lamda=10):
        aac_v, aac_n = self.extract_aac(seq)
        paac_v = aac_v.copy()
        paac_n = [f"PAAC_{x}" for x in aac_n]
        L = len(seq)
        for l in range(1, lamda + 1):
            if L > l:
                corr = sum(1 for i in range(L - l) if seq[i] == seq[i + l]) / (L - l)
            else:
                corr = 0.0
            paac_v.append(round(corr, 6))
            paac_n.append(f"PAAC_corr_{l}")
        return paac_v, paac_n

    def extract_pseaac(self, seq, lamda=10):
        v, n = self.extract_paac(seq, lamda)
        n = [x.replace("PAAC_", "PseAAC_") for x in n]
        return v, n

    def extract_aaindex(self, seq):
        vals, names = [], []
        for prop in self.aa_props:
            idx = self.aa_props.index(prop)
            arr = [self.aa_phys[aa][idx] for aa in seq if aa in self.aa_phys]
            if not arr:
                stats = [0.0] * 5
            else:
                stats = [round(np.mean(arr), 6), round(np.std(arr), 6),
                         round(np.max(arr), 6), round(np.min(arr), 6), round(np.sum(arr), 6)]
            vals.extend(stats)
            names.extend([f"AAindex_{prop}_mean", f"AAindex_{prop}_std",
                          f"AAindex_{prop}_max", f"AAindex_{prop}_min", f"AAindex_{prop}_sum"])
        return vals, names

    def extract_all(self, seq_list):
        f1, n1 = self.extract_aac(seq_list[0])
        f2, n2 = self.extract_dpc(seq_list[0])
        f3, n3 = self.extract_asdc(seq_list[0])
        f4, n4 = self.extract_cksaap(seq_list[0])
        f5, n5 = self.extract_ctd(seq_list[0])
        f6, n6 = self.extract_paac(seq_list[0])
        f7, n7 = self.extract_pseaac(seq_list[0])
        f8, n8 = self.extract_aaindex(seq_list[0])

        all_names = n1 + n2 + n3 + n4 + n5 + n6 + n7 + n8
        dim_info = {"AAC": len(n1), "DPC": len(n2), "ASDC": len(n3), "CKSAAP": len(n4),
                    "CTD": len(n5), "PAAC": len(n6), "PseAAC": len(n7), "AAindex": len(n8)}

        all_feats = []
        for idx, seq in enumerate(seq_list):
            if (idx + 1) % 200 == 0:
                print(f"Processing {idx + 1}/{len(seq_list)}")
            v1, _ = self.extract_aac(seq)
            v2, _ = self.extract_dpc(seq)
            v3, _ = self.extract_asdc(seq)
            v4, _ = self.extract_cksaap(seq)
            v5, _ = self.extract_ctd(seq)
            v6, _ = self.extract_paac(seq)
            v7, _ = self.extract_pseaac(seq)
            v8, _ = self.extract_aaindex(seq)
            all_feats.append(v1 + v2 + v3 + v4 + v5 + v6 + v7 + v8)
        return np.array(all_feats), all_names, dim_info


# ===================== Load Dataset and Filter Invalid Sequences =====================
df_raw = pd.read_csv(INPUT_CSV)
print(f"Total rows in raw dataset: {len(df_raw)}")


def is_valid_peptide(s):
    return all(ch in STANDARD_AA for ch in str(s))


df_filtered = df_raw[df_raw['SEQUENCE'].apply(is_valid_peptide)].copy()
invalid_cnt = len(df_raw) - len(df_filtered)
print(f"Filtered out invalid sequences: {invalid_cnt}")
if invalid_cnt > 0:
    bad_df = df_raw[~df_raw['SEQUENCE'].apply(is_valid_peptide)]
    print("Invalid sequence samples:", bad_df['SEQUENCE'].head().tolist())

seqs_all = df_filtered['SEQUENCE'].tolist()
y_all = df_filtered['label'].values
partition_all = df_filtered['partition'].values

extractor = PeptideFeatureExtractor()
X_all, feat_names, dim_info = extractor.extract_all(seqs_all)

df_feat_raw = pd.DataFrame(X_all, columns=feat_names)
df_feat_raw.insert(0, "SEQUENCE", df_filtered['SEQUENCE'])
df_feat_raw.insert(1, "label", df_filtered['label'])
df_feat_raw.insert(2, "partition", df_filtered['partition'])
df_feat_raw.to_csv(os.path.join(DATA_OUT, "01_AllRawFeatures.csv"), index=False)
print(f"Original feature dimensions: {X_all.shape[1]}")

# ===================== Nested CV for 3-Stage Feature Selection =====================
fold_ids = sorted(np.unique(partition_all))
fold_selection_records = []
fold_selected_feature_names = {}
all_sfs_curves = []

for outer_fold in fold_ids:
    print(f"\n==== Outer fold {outer_fold} ====")
    mask_test = partition_all == outer_fold
    mask_tv = partition_all != outer_fold
    inner_folds = [x for x in fold_ids if x != outer_fold]

    for inner_val_fold in inner_folds:
        mask_inner_val = partition_all == inner_val_fold
        mask_inner_train = mask_tv & (~mask_inner_val)

        X_in_train = X_all[mask_inner_train]
        y_in_train = y_all[mask_inner_train]
        X_in_val = X_all[mask_inner_val]
        y_in_val = y_all[mask_inner_val]

        scaler = StandardScaler()
        Xtr_scaled = scaler.fit_transform(X_in_train)
        Xval_scaled = scaler.transform(X_in_val)

        # Step 1: ANOVA F-test, retain top 80%
        f_scores, _ = f_classif(Xtr_scaled, y_in_train)
        f_th = np.percentile(f_scores, 80)
        mask_f = f_scores >= f_th
        X_f = Xtr_scaled[:, mask_f]
        names_f = [feat_names[i] for i in range(len(feat_names)) if mask_f[i]]

        # Step 2: Pearson Correlation |r| >= 0.7 to remove redundancy
        corr_mat = np.corrcoef(X_f.T)
        corr_mat = np.nan_to_num(corr_mat)
        keep = []
        nf = X_f.shape[1]
        for i in range(nf):
            if i in keep:
                continue
            keep.append(i)
            for j in range(i + 1, nf):
                if any(abs(corr_mat[k, j]) >= 0.7 for k in keep):
                    continue
                keep.append(j)
        X_corr = X_f[:, keep]
        names_corr = [names_f[i] for i in keep]

        # Step 3: Sequential Forward Selection (SFS) with Logistic Regression
        best_auc = 0
        best_set = []
        current = []
        sfs_history = []
        for it in range(30):
            best_it_auc = 0
            best_idx = -1
            for j in range(X_corr.shape[1]):
                if j in current: continue
                temp = current + [j]
                lr = LogisticRegression(max_iter=200)
                lr.fit(X_corr[:, temp], y_in_train)
                Xval_f = Xval_scaled[:, mask_f]
                Xval_corr = Xval_f[:, keep]
                auc_val = roc_auc_score(y_in_val, lr.predict_proba(Xval_corr[:, temp])[:, 1])
                if auc_val > best_it_auc:
                    best_it_auc = auc_val
                    best_idx = j
            if best_idx == -1: break
            current.append(best_idx)
            sfs_history.append({"iter": len(current), "auc": best_it_auc})
            if best_it_auc > best_auc:
                best_auc = best_it_auc
                best_set = current.copy()
            if len(current) > 5 and best_it_auc <= best_auc - 0.001:
                break

        final_names = [names_corr[i] for i in best_set]
        fold_selected_feature_names[(outer_fold, inner_val_fold)] = final_names
        all_sfs_curves.append(sfs_history)

        fold_selection_records.append({
            "outer_fold": outer_fold,
            "inner_val_fold": inner_val_fold,
            "n_original": X_all.shape[1],
            "n_after_Fscore": X_f.shape[1],
            "n_after_corr": X_corr.shape[1],
            "n_after_SFS": len(final_names),
            "best_val_auc": best_auc
        })

df_fold_sel = pd.DataFrame(fold_selection_records)
df_fold_sel.to_csv(os.path.join(DATA_OUT, "02_FoldWise_FeatureSelection_Log.csv"), index=False)
print("\nFold-wise feature selection completed and logged.")

# ===================== Figure 0: Feature Dimension Distribution =====================
plt.figure(figsize=(10, 5))
ftypes = list(dim_info.keys())
cnts = list(dim_info.values())
bars = plt.bar(ftypes, cnts, color="#1f77b4")
for b in bars:
    h = b.get_height()
    plt.text(b.get_x() + b.get_width() / 2., h + 10, str(int(h)), ha="center", va="bottom", fontsize=9)
plt.xticks(rotation=45, ha="right")
plt.title("Feature Dimension Distribution of Eight Feature Types", fontweight="bold")
plt.ylabel("Dimension Count")
plt.tight_layout()
plt.savefig(os.path.join(FIG_PATH, "Fig_FeatureDimension.jpg"))
plt.savefig(os.path.join(FIG_PATH, "Fig_FeatureDimension.pdf"), dpi=300)
plt.close()

# ===================== Save selected features & extract top 30 =====================
with open(os.path.join(DATA_OUT, "fold_selected_features.json"), "w", encoding="utf-8") as f:
    json.dump({str(k): v for k, v in fold_selected_feature_names.items()}, f, ensure_ascii=False, indent=2)

all_sel_feats = []
for k, v in fold_selected_feature_names.items():
    all_sel_feats.extend(v)

feat_counter = Counter(all_sel_feats)
top30_feat = [x[0] for x in feat_counter.most_common(30)]

print("\nTop 30 most frequent features selected:")
for idx, fname in enumerate(top30_feat, 1):
    print(f"{idx:2d}. {fname}")

df_raw_feat = pd.read_csv(os.path.join(DATA_OUT, "01_AllRawFeatures.csv"))
meta_cols = ["SEQUENCE", "label", "partition"]
df_final = df_raw_feat[meta_cols + top30_feat].copy()
df_final.to_csv(os.path.join(DATA_OUT, "03_Top30_Selected_Features.csv"), index=False)

freq_df = pd.DataFrame(feat_counter.items(), columns=["Feature", "Count"]).sort_values("Count", ascending=False)
freq_df.to_csv(os.path.join(DATA_OUT, "04_Feature_Select_Frequency.csv"), index=False)
print(f"\nOutput dataset saved: 03_Top30_Selected_Features.csv. Feature count: {len(top30_feat)}")

# ===================== Prepare data for plotting =====================
df_top30 = df_raw_feat[top30_feat].copy()
X_for_f = df_raw_feat[top30_feat].values
y_for_f = df_raw_feat["label"].values
f_scores, _ = f_classif(X_for_f, y_for_f)

df_frank = pd.DataFrame({"feat": top30_feat, "fscore": f_scores})
df_frank = df_frank.sort_values("fscore", ascending=True)
corr_matrix = df_top30.corr()

max_steps = 30
auc_matrix = np.full((len(all_sfs_curves), max_steps), np.nan)
for fold_idx, curve in enumerate(all_sfs_curves):
    for item in curve:
        step = item["iter"] - 1
        if step < max_steps:
            auc_matrix[fold_idx, step] = item["auc"]

mean_auc = np.nanmean(auc_matrix, axis=0)
x_axis = np.arange(1, max_steps + 1)
best_n = int(np.nanargmax(mean_auc) + 1)

# ===================== Figure A: Top 30 F-score Horizontal Bar Chart =====================
plt.figure(figsize=(12, 14))
colors_bar = ["#1f77b4"] * len(df_frank)
for idx, row in df_frank.iterrows():
    if row["feat"] in top30_feat[:best_n]:
        colors_bar[idx] = "#ff7f0e"

plt.barh(df_frank["feat"], df_frank["fscore"], color=colors_bar)
plt.xlabel("F-score")
plt.title("A. Top 30 Features Ranked by F-score (Blue=All, Orange=Selected Optimal Features)", loc="left")
plt.tick_params(axis='y', labelsize=7)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(FIG_PATH, "Fig_A_Top30_Fscore.jpg"))
plt.savefig(os.path.join(FIG_PATH, "Fig_A_Top30_Fscore.pdf"), dpi=300)
plt.close()

# ===================== Figure B: Feature Correlation Heatmap =====================
plt.figure(figsize=(14, 14))
sns.heatmap(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1,
            xticklabels=True, yticklabels=True, square=True, linewidths=0.2, cbar_kws={"shrink": 0.8})
plt.title("B. Pairwise Correlation of Selected Optimal Features", loc="left")
plt.tick_params(axis='both', labelsize=6)
plt.tight_layout()
plt.savefig(os.path.join(FIG_PATH, "Fig_B_Feature_Correlation.jpg"))
plt.savefig(os.path.join(FIG_PATH, "Fig_B_Feature_Correlation.pdf"), dpi=300)
plt.close()

# ===================== Figure C: SFS-AUC Curve =====================
plt.figure(figsize=(10, 6))
plt.plot(x_axis, mean_auc, marker='o', color="#1f77b4", linewidth=1.2, markersize=3)
plt.axvline(x=best_n, linestyle="--", color="#ff7f0e", label=f"Optimal={best_n}")
plt.xlabel("Feature number")
plt.ylabel("AUC value")
plt.title("C. SFS AUC Performance with Increasing Feature Count", loc="left")
plt.ylim([0.5, 1.0])
plt.grid(axis="y", alpha=0.3)
plt.legend(frameon=False)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(FIG_PATH, "Fig_C_SFS_AUC_Curve.jpg"))
plt.savefig(os.path.join(FIG_PATH, "Fig_C_SFS_AUC_Curve.pdf"), dpi=300)
plt.close()

print("\nAll individual figures saved successfully!")
print(f"\nExecution completed! Output directory: {OUT_ROOT}")