import os
import gc
import random
import warnings
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
from sklearn.metrics import roc_curve, auc, average_precision_score, confusion_matrix

# ===================== Global Setup =====================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['OMP_NUM_THREADS'] = '1'
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory growth enabled.")
    except RuntimeError as e:
        print(f"GPU configuration failed: {e}")

MAX_SEQ_LEN = 30
metrics_list = ['AUC', 'AUPR', 'SEN', 'SPE', 'PRE', 'F1', 'ACC', 'MCC']
BASE_DIR = "."
PART2_CSV = os.path.join(BASE_DIR, "Results", "2_Feature_Extraction_Selection_v2", "csv_output",
                         "03_Top30_Selected_Features.csv")
embedding_file = os.path.join(BASE_DIR, "Data", "One-hot_encoding.txt")
ABLATION_OUT = os.path.join(BASE_DIR, "Results", "00_Ablation_Result")
os.makedirs(ABLATION_OUT, exist_ok=True)


# ===================== Custom Layers =====================
class PartialConv1D_Module(layers.Layer):
    def __init__(self, filters, kernel_size, n_div=4, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.n_div = n_div
        self.dim_conv = filters // n_div
        self.dim_untouched = filters - self.dim_conv

    def build(self, input_shape):
        self.partial_conv = layers.Conv1D(
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


class DualPath_Attention(layers.Layer):
    def __init__(self, reduction=16, **kwargs):
        super().__init__(**kwargs)
        self.reduction = reduction

    def build(self, input_shape):
        ch = input_shape[-1]
        self.fc = models.Sequential([
            layers.Dense(ch // self.reduction, activation='swish', kernel_initializer='he_normal'),
            layers.Dense(ch, activation='sigmoid')
        ])
        super().build(input_shape)

    def call(self, inputs):
        avg = layers.GlobalAveragePooling1D()(inputs)
        mx = layers.GlobalMaxPooling1D()(inputs)
        e = self.fc(avg) + self.fc(mx)
        return inputs * layers.Reshape((1, inputs.shape[-1]))(e)

    def get_config(self):
        return {"reduction": self.reduction}


class BiGated_Fusion_Module(layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.gate_seq = layers.Dense(self.units, activation='sigmoid')
        self.gate_feat = layers.Dense(self.units, activation='sigmoid')
        self.dense_out = layers.Dense(self.units, activation='swish')
        super().build(input_shape)

    def call(self, inputs):
        x_seq, x_feat = inputs
        gs = self.gate_seq(x_seq)
        gf = self.gate_feat(x_feat)
        fused = layers.Add()([x_seq * gs, x_feat * gf])
        return self.dense_out(fused)

    def get_config(self):
        return {"units": self.units}


# ===================== Feature Loading & Encoding =====================
aa_list, embedding_mat, aa_to_idx = [], [], {}


def load_aa_embedding(emb_file):
    global aa_list, embedding_mat, aa_to_idx
    vs = []
    if not os.path.exists(emb_file):
        aa_list = list("ACDEFGHIKLMNPQRSTVWY")
        embedding_mat = np.random.randn(20, 21).astype(np.float32)
    else:
        with open(emb_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip() and not line.startswith("#"):
                    parts = line.split()
                    aa_list.append(parts[0])
                    vs.append(np.array([float(x) for x in parts[1:]], dtype=np.float32))
        embedding_mat = np.array(vs)
    aa_to_idx = {a: i for i, a in enumerate(aa_list)}


load_aa_embedding(embedding_file)


def encode_seqs(seqs):
    enc = np.zeros((len(seqs), MAX_SEQ_LEN, embedding_mat.shape[1]), dtype=np.float32)
    for i, s in enumerate(seqs):
        for j, c in enumerate(str(s)[:MAX_SEQ_LEN]):
            if c in aa_to_idx:
                enc[i, j] = embedding_mat[aa_to_idx[c]]
    return enc


def calc_metrics(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    mcc = (tp * tn - fp * fn) / (np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + 1e-8)
    return {
        "AUC": auc(fpr, tpr),
        "AUPR": average_precision_score(y_true, y_prob),
        "SEN": tp / (tp + fn + 1e-8),
        "SPE": tn / (tn + fp + 1e-8),
        "PRE": tp / (tp + fp + 1e-8),
        "F1": 2 * tp / (2 * tp + fp + fn + 1e-8),
        "ACC": (tp + tn) / (tp + tn + fp + fn + 1e-8),
        "MCC": mcc
    }


# ===================== Ablation Model Builder =====================
def build_ablation_model(feat_dim, use_partialconv: bool, use_dpatt: bool, use_gatefusion: bool):
    seq_in = layers.Input((MAX_SEQ_LEN, embedding_mat.shape[1]), name="Seq_In")
    x = layers.Conv1D(128, 1, padding="same")(seq_in)
    if use_partialconv:
        x = PartialConv1D_Module(128, 3)(x)
    x = layers.BatchNormalization()(x)
    shortcut = x

    if use_partialconv:
        x = PartialConv1D_Module(128, 5)(x)
    if use_dpatt:
        x = DualPath_Attention(8)(x)

    x = layers.Add()([x, shortcut])
    x_seq = layers.GlobalAveragePooling1D()(x)
    x_seq = layers.Dense(128, activation="swish")(x_seq)

    feat_in = layers.Input((feat_dim,), name="Feat_In")
    f = layers.BatchNormalization()(feat_in)
    f = layers.Dense(128, activation="swish")(f)
    f = layers.Dropout(0.3)(f)

    if use_gatefusion:
        fused = BiGated_Fusion_Module(128)([x_seq, f])
    else:
        fused = layers.concatenate([x_seq, f])
        fused = layers.Dense(128, activation="swish")(fused)

    z = layers.Dense(64, activation="swish")(fused)
    z = layers.Dropout(0.2)(z)
    out = layers.Dense(1, activation="sigmoid")(z)

    m = models.Model([seq_in, feat_in], out)
    m.compile(tf.keras.optimizers.Adam(2e-4), loss="binary_crossentropy", metrics=["accuracy"])
    return m


# ===================== Ablation Configurations =====================
ablation_configs = [
    {"name": "Base(xxx)", "use_partialconv": False, "use_dpatt": False, "use_gatefusion": False},
    {"name": "PConv(vxx)", "use_partialconv": True, "use_dpatt": False, "use_gatefusion": False},
    {"name": "DPAtt(xvx)", "use_partialconv": False, "use_dpatt": True, "use_gatefusion": False},
    {"name": "GateFus(xxv)", "use_partialconv": False, "use_dpatt": False, "use_gatefusion": True},
    {"name": "PConv+DPAtt(vvx)", "use_partialconv": True, "use_dpatt": True, "use_gatefusion": False},
    {"name": "PConv+GateFus(vxv)", "use_partialconv": True, "use_dpatt": False, "use_gatefusion": True},
    {"name": "DPAtt+GateFus(xvv)", "use_partialconv": False, "use_dpatt": True, "use_gatefusion": True},
    {"name": "All(vvv)", "use_partialconv": True, "use_dpatt": True, "use_gatefusion": True},
]

df_all = pd.read_csv(PART2_CSV)
meta_cols = ["SEQUENCE", "label", "partition"]
feat_cols = [c for c in df_all.columns if c not in meta_cols]
info = {
    "seq": df_all["SEQUENCE"].values,
    "label": df_all["label"].values,
    "partition": df_all["partition"].values,
    "feat": df_all[feat_cols].values.astype(np.float32),
    "feat_dim": len(feat_cols)
}

all_ablation_fold_dfs = {}

# ===================== Evaluation Loop =====================
for cfg in ablation_configs:
    model_tag = cfg["name"]
    print(f"\n==== Running Ablation Group: {model_tag} ====")
    group_out = os.path.join(ABLATION_OUT, model_tag)
    pred_path = os.path.join(group_out, "Predictions")
    res_path = os.path.join(group_out, "Results")
    os.makedirs(pred_path, exist_ok=True)
    os.makedirs(res_path, exist_ok=True)

    fold_pred_records = []
    fold_metrics = pd.DataFrame()

    for fold in range(5):
        val_f = (fold + 1) % 5
        p_arr = info["partition"]
        mask_test = p_arr == fold
        mask_val = p_arr == val_f
        mask_train = (~mask_test) & (~mask_val)

        xtr_seq = encode_seqs(info["seq"][mask_train])
        xtr_feat = info["feat"][mask_train]
        ytr = info["label"][mask_train]

        xval_seq = encode_seqs(info["seq"][mask_val])
        xval_feat = info["feat"][mask_val]
        yval = info["label"][mask_val]

        xte_seq = encode_seqs(info["seq"][mask_test])
        xte_feat = info["feat"][mask_test]
        yte = info["label"][mask_test]
        seq_te = info["seq"][mask_test]

        K.clear_session()
        gc.collect()

        model = build_ablation_model(
            feat_dim=info["feat_dim"],
            use_partialconv=cfg["use_partialconv"],
            use_dpatt=cfg["use_dpatt"],
            use_gatefusion=cfg["use_gatefusion"]
        )

        es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=25, restore_best_weights=True)
        model.fit(
            [xtr_seq, xtr_feat], ytr,
            validation_data=([xval_seq, xval_feat], yval),
            epochs=150, batch_size=32, verbose=0, callbacks=[es]
        )

        yp = model.predict([xte_seq, xte_feat], verbose=0).flatten()
        met = calc_metrics(yte, yp)
        fold_metrics = pd.concat([fold_metrics, pd.DataFrame([met])], ignore_index=True)

        for s, yt, ypb in zip(seq_te, yte, yp):
            fold_pred_records.append({
                "fold": fold,
                "sequence": s,
                "y_true": int(yt),
                "y_pred_prob": float(ypb),
                "y_pred_label": int(1 if ypb >= 0.5 else 0)
            })
        del model
        gc.collect()

    df_pred = pd.DataFrame(fold_pred_records)
    df_pred.to_csv(os.path.join(pred_path, "ablation_fivefold_pred.csv"), index=False)
    fold_metrics.to_csv(os.path.join(res_path, "PerFold_Metrics.csv"), index=False)

    summary_df = pd.DataFrame({"Mean": fold_metrics.mean(), "Std": fold_metrics.std()})
    summary_df.to_csv(os.path.join(res_path, "Summary_MeanStd.csv"))

    all_ablation_fold_dfs[model_tag] = fold_metrics.copy()

print("\nRunning statistical significance testing (Wilcoxon Test)...")

# ===================== Statistical Significance Testing =====================
target_model = "All(vvv)"
df_target = all_ablation_fold_dfs[target_model]
test_metrics = ['AUC', 'AUPR', 'SEN', 'SPE', 'PRE', 'F1', 'ACC', 'MCC']

detailed_p_records = []

for cfg in ablation_configs:
    m_name = cfg["name"]
    if m_name == target_model:
        continue

    df_cmp = all_ablation_fold_dfs[m_name]

    for met in test_metrics:
        all_vals = df_target[met].values
        cmp_vals = df_cmp[met].values
        diff = all_vals - cmp_vals

        if np.all(diff == 0):
            stat, pval = 0.0, 1.0
        else:
            try:
                stat, pval = wilcoxon(all_vals, cmp_vals, alternative="greater")
            except Exception as e:
                stat, pval = np.nan, np.nan

        sig_mark = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else "ns"))
        detailed_p_records.append({
            "Reference": target_model,
            "Ablation_Group": m_name,
            "Metric": met,
            "Target_Mean": round(np.mean(all_vals), 4),
            "Ablation_Mean": round(np.mean(cmp_vals), 4),
            "Delta": round(np.mean(all_vals) - np.mean(cmp_vals), 4),
            "statistic_W": stat,
            "p_value": round(pval, 5) if not np.isnan(pval) else np.nan,
            "Significance": sig_mark
        })

df_wilcoxon_ablation = pd.DataFrame(detailed_p_records)
df_wilcoxon_ablation.to_csv(os.path.join(ABLATION_OUT, "Table5_Ablation_Wilcoxon_Detailed.csv"), index=False)

# ===================== Generate Final Summary Table =====================
table5_rows = []
for cfg in ablation_configs:
    m_name = cfg["name"]
    df_m = all_ablation_fold_dfs[m_name]
    row_data = {"Model": m_name}

    for met in ['AUC', 'MCC', 'F1', 'ACC', 'AUPR']:
        m_mean = df_m[met].mean()
        m_std = df_m[met].std()
        cell_str = f"{m_mean:.4f} ± {m_std:.4f}"

        if m_name != target_model:
            sub = df_wilcoxon_ablation[
                (df_wilcoxon_ablation["Ablation_Group"] == m_name) &
                (df_wilcoxon_ablation["Metric"] == met)
                ]
            if not sub.empty:
                sig = sub["Significance"].values[0]
                pval = sub["p_value"].values[0]
                cell_str += f" ({sig}, p={pval:.4f})"
        else:
            cell_str += " (Ref)"

        row_data[met] = cell_str

    table5_rows.append(row_data)

df_table5 = pd.DataFrame(table5_rows)
df_table5.to_csv(os.path.join(ABLATION_OUT, "Table5_Ablation_Summary_with_Significance.csv"), index=False)

print("\n==== Ablation Study Summary (Mean ± Std & Wilcoxon p-value) ====")
print(df_table5.to_string(index=False))

print(f"\nAblation study completed successfully! Output directory: {ABLATION_OUT}")