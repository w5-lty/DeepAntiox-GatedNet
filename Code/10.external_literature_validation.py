import os
import re
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

# ===================== Global Plotting Configuration =====================
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

# ===================== Paths Configuration =====================
BASE_DIR = "."
EXT_CSV = os.path.join(BASE_DIR, "Data", "external_aop.csv")
MODEL_PATH = os.path.join(BASE_DIR, "Results", "Result_A_GatedNet_Alone", "Trained_Models", "Final_Model_p90.keras")
TOP30_CSV = os.path.join(BASE_DIR, "Results", "2_Feature_Extraction_Selection_v2", "csv_output", "03_Top30_Selected_Features.csv")
EMBEDDING_FILE = os.path.join(BASE_DIR, "Data", "One-hot_encoding.txt")
FIG_OUT = os.path.join(BASE_DIR, "Results", "7_External_Literature_Val")
os.makedirs(FIG_OUT, exist_ok=True)

# ===================== Custom Keras Layers =====================
class PartialConv1D_Module(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, n_div=4, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.n_div = n_div
        self.dim_conv = filters // n_div
        self.dim_untouched = filters - self.dim_conv

    def build(self, input_shape):
        self.partial_conv = tf.keras.layers.Conv1D(
            filters=self.dim_conv, kernel_size=self.kernel_size,
            padding='same', use_bias=False, activation='swish'
        )
        super().build(input_shape)

    def call(self, x):
        x_conv, x_untouched = tf.split(x, [self.dim_conv, self.dim_untouched], axis=-1)
        x_conv = self.partial_conv(x_conv)
        return tf.concat([x_conv, x_untouched], axis=-1)

    def get_config(self):
        return {"filters": self.filters, "kernel_size": self.kernel_size, "n_div": self.n_div}


class DualPath_Attention(tf.keras.layers.Layer):
    def __init__(self, reduction=16, **kwargs):
        super().__init__(**kwargs)
        self.reduction = reduction

    def build(self, input_shape):
        ch = input_shape[-1]
        self.fc = tf.keras.models.Sequential([
            tf.keras.layers.Dense(ch // self.reduction, activation="swish", kernel_initializer="he_normal"),
            tf.keras.layers.Dense(ch, activation="sigmoid")
        ])
        super().build(input_shape)

    def call(self, inputs):
        avg = tf.keras.layers.GlobalAveragePooling1D()(inputs)
        mx = tf.keras.layers.GlobalMaxPooling1D()(inputs)
        e = self.fc(avg) + self.fc(mx)
        return inputs * tf.keras.layers.Reshape((1, inputs.shape[-1]))(e)

    def get_config(self):
        return {"reduction": self.reduction}


class BiGated_Fusion_Module(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.gate_seq = tf.keras.layers.Dense(self.units, activation="sigmoid")
        self.gate_feat = tf.keras.layers.Dense(self.units, activation="sigmoid")
        self.dense_out = tf.keras.layers.Dense(self.units, activation="swish")
        super().build(input_shape)

    def call(self, inputs):
        x_seq, x_feat = inputs
        gs = self.gate_seq(x_seq)
        gf = self.gate_feat(x_feat)
        fused = tf.keras.layers.Add()([x_seq * gs, x_feat * gf])
        return self.dense_out(fused)

    def get_config(self):
        return {"units": self.units}


# ===================== Feature Extraction Tools =====================
STANDARD_AA = {'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
               'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'}
STANDARD_AA_LIST = list(STANDARD_AA)

CTD_PROPERTY_GROUPS = {
    'Polarity': {'Polar': ['R', 'K', 'E', 'D', 'Q', 'N'], 'Neutral': ['G', 'A', 'S', 'T', 'P', 'H', 'Y'], 'Nonpolar': ['F', 'L', 'I', 'M', 'W', 'V', 'C']},
    'Hydrophobicity': {'Strong': ['A', 'F', 'G', 'I', 'L', 'M', 'P', 'V', 'W'], 'Medium': ['C', 'H', 'Y'], 'Weak': ['D', 'E', 'N', 'Q', 'K', 'R', 'S', 'T']},
    'Charge': {'Positive': ['K', 'R'], 'Negative': ['D', 'E'], 'Neutral': ['A', 'N', 'C', 'Q', 'G', 'H', 'I', 'L', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']},
    'SideChainVolume': {'Large': ['F', 'I', 'L', 'M', 'W', 'V'], 'Medium': ['A', 'C', 'H', 'K', 'N', 'Q', 'R', 'S', 'T', 'Y'], 'Small': ['D', 'E', 'G', 'P']},
    'Polarizability': {'High': ['F', 'H', 'I', 'L', 'M', 'W', 'V'], 'Medium': ['A', 'C', 'K', 'N', 'Q', 'R', 'S', 'T', 'Y'], 'Low': ['D', 'E', 'G', 'P']},
    'IsoelectricPoint': {'High': ['K', 'R'], 'Medium': ['A', 'C', 'F', 'G', 'H', 'I', 'L', 'M', 'N', 'P', 'Q', 'S', 'T', 'V', 'W', 'Y'], 'Low': ['D', 'E']},
    'HDonor': {'Many': ['R', 'K', 'H', 'N', 'Q'], 'Few': ['A', 'C', 'F', 'G', 'I', 'L', 'M', 'P', 'S', 'T', 'V', 'W', 'Y'], 'None': ['D', 'E']},
    'HAcceptor': {'Many': ['D', 'E', 'N', 'Q'], 'Few': ['A', 'C', 'F', 'G', 'H', 'I', 'L', 'M', 'P', 'R', 'S', 'T', 'V', 'W', 'Y'], 'None': ['K']},
    'HydrophobicityIndex': {'High': ['I', 'V', 'L', 'F', 'C', 'M', 'A'], 'Medium': ['G', 'T', 'W', 'Y', 'P', 'H'], 'Low': ['E', 'D', 'R', 'K', 'Q', 'N', 'S']},
    'AAType': {'Aliphatic': ['A', 'G', 'I', 'L', 'P', 'V'], 'Aromatic': ['F', 'H', 'W', 'Y'], 'PolarUncharged': ['C', 'N', 'Q', 'S', 'T'], 'Positive': ['K', 'R'], 'Negative': ['D', 'E']},
    'SideChainCharge': {'Positive': ['K', 'R', 'H'], 'Negative': ['D', 'E'], 'Neutral': ['A', 'C', 'F', 'G', 'I', 'L', 'M', 'N', 'P', 'Q', 'S', 'T', 'V', 'W', 'Y']},
    'SideChainAromatic': {'Aromatic': ['F', 'H', 'W', 'Y'], 'NonAromatic': ['A', 'C', 'D', 'E', 'G', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V']},
    'SideChainHBond': {'Donor': ['R', 'K', 'H', 'N', 'Q'], 'Acceptor': ['D', 'E', 'N', 'Q', 'S', 'T', 'Y'], 'None': ['A', 'C', 'F', 'G', 'I', 'L', 'M', 'P', 'V', 'W']}
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
                ctd_names.extend([
                    f"CTD_{prop_name}_D_{g_name}_P0",
                    f"CTD_{prop_name}_D_{g_name}_P25",
                    f"CTD_{prop_name}_D_{g_name}_P50",
                    f"CTD_{prop_name}_D_{g_name}_P70",
                    f"CTD_{prop_name}_D_{g_name}_P100"
                ])
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
                stats = [
                    round(np.mean(arr), 6), round(np.std(arr), 6),
                    round(np.max(arr), 6), round(np.min(arr), 6), round(np.sum(arr), 6)
                ]
            vals.extend(stats)
            names.extend([
                f"AAindex_{prop}_mean", f"AAindex_{prop}_std",
                f"AAindex_{prop}_max", f"AAindex_{prop}_min", f"AAindex_{prop}_sum"
            ])
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
        dim_info = {
            "AAC": len(n1), "DPC": len(n2), "ASDC": len(n3), "CKSAAP": len(n4),
            "CTD": len(n5), "PAAC": len(n6), "PseAAC": len(n7), "AAindex": len(n8)
        }
        all_feats = []
        for idx, seq in enumerate(seq_list):
            if (idx + 1) % 100 == 0:
                print(f"  Feature extraction progress: {idx+1}/{len(seq_list)}")
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

# ===================== Sequence Embedding Configuration =====================
MAX_SEQ_LEN = 30
aa_list, embedding_mat, aa_to_idx = [], [], {}

def load_aa_embedding(emb_file):
    global aa_list, embedding_mat, aa_to_idx
    vs = []
    if not os.path.exists(emb_file):
        print("Warning: Embedding file not found, initializing randomly.")
        aa_list = list("ACDEFGHIKLMNPQRSTVWY")
        embedding_mat = np.random.randn(20, 21).astype(np.float32)
    else:
        print(f"Successfully loaded embedding file: {emb_file}")
        with open(emb_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip() and not line.startswith("#"):
                    parts = line.split()
                    aa_list.append(parts[0])
                    vs.append(np.array([float(x) for x in parts[1:]], dtype=np.float32))
        embedding_mat = np.array(vs)
    aa_to_idx = {a: i for i, a in enumerate(aa_list)}

load_aa_embedding(EMBEDDING_FILE)

def encode_seqs(seqs):
    enc = np.zeros((len(seqs), MAX_SEQ_LEN, embedding_mat.shape[1]), dtype=np.float32)
    for i, s in enumerate(seqs):
        for j, c in enumerate(str(s)[:MAX_SEQ_LEN]):
            if c in aa_to_idx:
                enc[i, j] = embedding_mat[aa_to_idx[c]]
    return enc


# ===================== Main Execution Pipeline =====================
print("=" * 70)
print("Executing External Literature Independent Validation (DeepAntiox-GatedNet)")
print("=" * 70)

if not os.path.exists(EXT_CSV):
    raise FileNotFoundError(f"External dataset file not found: {EXT_CSV}")

print(f"Reading external dataset: {EXT_CSV}")
df_ext_raw = pd.read_csv(EXT_CSV)

seq_col = None
for candidate in ["SEQUENCE", "sequence", "Sequence", "pep", "peptide", "Peptide", "seq", "Seq"]:
    if candidate in df_ext_raw.columns:
        seq_col = candidate
        break

if seq_col is None:
    seq_col = df_ext_raw.columns[0]
    print(f"Warning: Standard sequence column not found, using the first column: {seq_col}")
else:
    print(f"Sequence column identified: '{seq_col}'")

def clean_sequence(s):
    if pd.isna(s):
        return None
    s_clean = re.sub(r'[^A-Za-z]', '', str(s)).upper()
    if 2 <= len(s_clean) <= 30 and all(ch in STANDARD_AA for ch in s_clean):
        return s_clean
    return None

df_ext = df_ext_raw.copy()
df_ext["_clean_seq"] = df_ext[seq_col].apply(clean_sequence)
invalid_cnt = df_ext["_clean_seq"].isna().sum()
if invalid_cnt > 0:
    print(f"Filtered out {invalid_cnt} invalid or over-length sequences.")
df_ext = df_ext.dropna(subset=["_clean_seq"]).reset_index(drop=True)
print(f"Total valid external peptides: {len(df_ext)}")

print(f"Loading pre-trained model: {MODEL_PATH}")
custom_objs = {
    "PartialConv1D_Module": PartialConv1D_Module,
    "DualPath_Attention": DualPath_Attention,
    "BiGated_Fusion_Module": BiGated_Fusion_Module
}
model = keras.models.load_model(MODEL_PATH, custom_objects=custom_objs)

print(f"Loading Top 30 features definition: {TOP30_CSV}")
df_top = pd.read_csv(TOP30_CSV, nrows=1)
top30_names = [c for c in df_top.columns if c not in ["SEQUENCE", "label", "partition"]]
print(f"Successfully identified {len(top30_names)} feature dimensions.")

print("Calculating features for external peptide sequences...")
extractor = PeptideFeatureExtractor()
seq_list = df_ext["_clean_seq"].tolist()
X_all_ext, feat_all_names, _ = extractor.extract_all(seq_list)
df_ext_feat = pd.DataFrame(X_all_ext, columns=feat_all_names)

missing_feats = [fn for fn in top30_names if fn not in df_ext_feat.columns]
if missing_feats:
    print(f"Warning: Imputing missing features with 0: {missing_feats}")
    for mf in missing_feats:
        df_ext_feat[mf] = 0.0

X_30_ext = df_ext_feat[top30_names].values.astype(np.float32)

print("Executing DeepAntiox-GatedNet model inference...")
seq_enc_ext = encode_seqs(df_ext["_clean_seq"].values)
pred_scores = model.predict([seq_enc_ext, X_30_ext], verbose=0).flatten()

df_ext["pred_score"] = np.round(pred_scores, 4)
df_ext["pred_label"] = (pred_scores >= 0.5).astype(int)
df_ext["classification"] = np.where(pred_scores >= 0.5, "Antioxidant", "Non-antioxidant")
df_ext["seq_len"] = df_ext["_clean_seq"].apply(len)

df_ext.drop(columns=["_clean_seq"], inplace=True)

pred_out_csv = os.path.join(FIG_OUT, "external_pred_result.csv")
df_ext.to_csv(pred_out_csv, index=False)
print(f"Prediction results saved to: {pred_out_csv}")

# ===================== Evaluation Metrics Calculation =====================
total_n = len(df_ext)
pos_n = int((df_ext["pred_label"] == 1).sum())
neg_n = total_n - pos_n
sensitivity = (pos_n / total_n) * 100.0 if total_n > 0 else 0.0
median_score = float(df_ext["pred_score"].median())
q25 = float(df_ext["pred_score"].quantile(0.25))
q75 = float(df_ext["pred_score"].quantile(0.75))
mean_score = float(df_ext["pred_score"].mean())

metrics_report = f"""======================================================================
  DeepAntiox-GatedNet External Literature Validation Summary
======================================================================
Total External Peptides (Literature Validated AOPs) : {total_n}
Correctly Identified as Antioxidant (Score >= 0.5)  : {pos_n} ({sensitivity:.2f}%)
Misclassified as Non-Antioxidant (Score < 0.5)      : {neg_n} ({100.0 - sensitivity:.2f}%)
Sensitivity / Recall (True Positive Rate)           : {sensitivity:.2f}%
Predicted Score Median [IQR]                        : {median_score:.4f} [{q25:.4f} - {q75:.4f}]
Predicted Score Mean ± Std                          : {mean_score:.4f} ± {df_ext['pred_score'].std():.4f}
======================================================================
"""
print("\n" + metrics_report)
report_file = os.path.join(FIG_OUT, "external_validation_metrics.txt")
with open(report_file, "w", encoding="utf-8") as f:
    f.write(metrics_report)

# ===================== Dual-panel Figure Generation =====================
print("Generating dual-panel validation figure...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

sns.histplot(
    df_ext["pred_score"], bins=20, kde=True,
    color="#1F77B4", stat="probability", ax=ax1,
    line_kws={"linewidth": 2, "color": "#0B406B"}
)
ax1.axvline(0.5, color="#D62728", linestyle="--", linewidth=1.5, label="Threshold (0.5)")
ax1.set_xlabel("Predicted Probability Score", fontsize=10, fontweight="bold")
ax1.set_ylabel("Probability Density", fontsize=10, fontweight="bold")
ax1.set_title("A. Score Distribution on External Peptides", fontsize=11, fontweight="bold", loc="left")
ax1.set_xlim(-0.05, 1.05)
ax1.legend(loc="upper left")
ax1.grid(axis="y", linestyle="--", alpha=0.3)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

info_text = f"Total = {total_n}\nSensitivity = {sensitivity:.1f}%\nMedian = {median_score:.3f}"
ax1.text(
    0.05, 0.65, info_text, transform=ax1.transAxes,
    fontsize=9, verticalalignment='top',
    bbox=dict(boxstyle='round,pad=0.5', facecolor='#F5F5F5', edgecolor='#BDBDBD', alpha=0.9)
)

df_ext["_group"] = "Literature-Reported AOPs"

sns.violinplot(
    x="_group", y="pred_score", data=df_ext, ax=ax2,
    inner="quartile", color="#A6CEE3", cut=0, linewidth=1.2
)
sns.stripplot(
    x="_group", y="pred_score", data=df_ext, ax=ax2,
    color="#1F77B4", alpha=0.6, jitter=0.25, size=5
)

ax2.axhline(0.5, color="#D62728", linestyle="--", linewidth=1.5, label="Threshold (0.5)")
ax2.set_xlabel("External Literature Dataset", fontsize=10, fontweight="bold")
ax2.set_ylabel("Predicted Probability Score", fontsize=10, fontweight="bold")
ax2.set_title("B. Density & Prediction Scores", fontsize=11, fontweight="bold", loc="left")
ax2.set_ylim(-0.05, 1.05)
ax2.legend(loc="lower left")
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()

fig_jpg = os.path.join(FIG_OUT, "Supple_FigX_external_validation.jpg")
fig_pdf = os.path.join(FIG_OUT, "Supple_FigX_external_validation.pdf")
fig.savefig(fig_jpg, dpi=1000)
fig.savefig(fig_pdf, dpi=300)
plt.close()

print(f"Dual-panel figure generated and saved:")
print(f"   JPG (1000 DPI) : {fig_jpg}")
print(f"   PDF (Vector)   : {fig_pdf}")

print("\n" + "=" * 70)
print(f"External literature validation completed! Output directory: {FIG_OUT}")
print("1. external_pred_result.csv         : Detailed predictions and scores")
print("2. external_validation_metrics.txt  : Quantitative metrics text report")
print("3. Supple_FigX_external_validation  : Dual-panel figure (Histogram KDE + Violin Scatter)")
print("=" * 70)