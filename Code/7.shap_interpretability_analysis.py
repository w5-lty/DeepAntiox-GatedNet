import os
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import shap

# ===================== Paths Configuration =====================
BASE_DIR = "."
MODEL_PATH = os.path.join(BASE_DIR, "Results", "Result_A_GatedNet_Alone", "Trained_Models", "Final_Model_p90.keras")
DATA_CSV = os.path.join(BASE_DIR, "Results", "2_Feature_Extraction_Selection_v2", "csv_output", "03_Top30_Selected_Features.csv")
EMBEDDING_FILE = os.path.join(BASE_DIR, "Data", "One-hot_encoding.txt")
FIG_OUT = os.path.join(BASE_DIR, "Results", "5_Shap_Analysis", "figures")
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


# ===================== Load Model and Embeddings =====================
print("Loading trained GatedNet model...")
custom_objs = {
    "PartialConv1D_Module": PartialConv1D_Module,
    "DualPath_Attention": DualPath_Attention,
    "BiGated_Fusion_Module": BiGated_Fusion_Module
}
model = keras.models.load_model(MODEL_PATH, custom_objects=custom_objs)

MAX_SEQ_LEN = 30
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

load_aa_embedding(EMBEDDING_FILE)

def encode_seqs(seqs):
    enc = np.zeros((len(seqs), MAX_SEQ_LEN, embedding_mat.shape[1]), dtype=np.float32)
    for i, s in enumerate(seqs):
        for j, c in enumerate(str(s)[:MAX_SEQ_LEN]):
            if c in aa_to_idx:
                enc[i, j] = embedding_mat[aa_to_idx[c]]
    return enc

# ===================== Prepare Data for SHAP =====================
print(f"Reading dataset: {DATA_CSV}")
df = pd.read_csv(DATA_CSV)
meta_cols = ["SEQUENCE", "label", "partition"]
feat_names = [c for c in df.columns if c not in meta_cols]

sample_size = min(150, len(df))
df_sub = df.iloc[:sample_size].copy()
X_feat = df_sub[feat_names].values.astype(np.float32)

sub_seq_encoded = encode_seqs(df_sub["SEQUENCE"].values)
mean_seq_vector = np.mean(sub_seq_encoded, axis=0, keepdims=True)

def model_predict_wrapper(x_feat_perturb):
    batch_size = x_feat_perturb.shape[0]
    seq_matched_input = np.repeat(mean_seq_vector, batch_size, axis=0)
    preds = model.predict([seq_matched_input, x_feat_perturb], verbose=0).flatten()
    return preds

# ===================== Compute SHAP Values =====================
print("Initializing KernelExplainer and computing SHAP values (this may take a few minutes)...")
background_data = shap.sample(X_feat, 30, random_state=42)
explainer = shap.KernelExplainer(model_predict_wrapper, background_data)
shap_values = explainer.shap_values(X_feat, nsamples=100)

if isinstance(shap_values, list):
    shap_vals_arr = shap_values[0]
else:
    shap_vals_arr = shap_values

base_expected_value = explainer.expected_value
if isinstance(base_expected_value, (list, np.ndarray)):
    base_expected_value = float(base_expected_value[0])

# ===================== Visualization & Export =====================
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

print("Generating SHAP visualization plots...")

# Plot A: SHAP Summary Beeswarm Plot
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_vals_arr, X_feat, feature_names=feat_names, show=False)
plt.title("A. SHAP Value Distribution Across Top Features", loc="left", fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIG_OUT, "Fig15a_SHAP_summary.jpg"), dpi=1000)
plt.savefig(os.path.join(FIG_OUT, "Fig15a_SHAP_summary.pdf"), dpi=300)
plt.close()

# Plot B: Global Feature Importance Bar Chart
mean_abs_shap = np.mean(np.abs(shap_vals_arr), axis=0)
df_imp = pd.DataFrame({"Feature": feat_names, "Importance": mean_abs_shap})
df_imp = df_imp.sort_values("Importance", ascending=True)

plt.figure(figsize=(9, 10))
plt.barh(df_imp["Feature"], df_imp["Importance"], color="#1f77b4", edgecolor="none", height=0.7)
plt.xlabel("Mean |SHAP Value| (Average Impact on Model Output)")
plt.title("B. Global Feature Importance Ranking", loc="left", fontsize=11, fontweight="bold")
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(FIG_OUT, "Fig15b_Global_Importance.jpg"), dpi=1000)
plt.savefig(os.path.join(FIG_OUT, "Fig15b_Global_Importance.pdf"), dpi=300)
plt.close()

# Plot C: Single Sample Waterfall Plot
plt.figure(figsize=(9, 7))
explanation_sample = shap.Explanation(
    values=shap_vals_arr[0],
    base_values=base_expected_value,
    data=X_feat[0],
    feature_names=feat_names
)
shap.waterfall_plot(explanation_sample, max_display=12, show=False)
plt.title("C. Feature Attribution Waterfall for Single Peptide Sample", loc="left", fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIG_OUT, "Fig15c_Waterfall_single_sample.jpg"), dpi=1000)
plt.savefig(os.path.join(FIG_OUT, "Fig15c_Waterfall_single_sample.pdf"), dpi=300)
plt.close()

df_imp_sorted = df_imp.sort_values("Importance", ascending=False).reset_index(drop=True)
df_imp_sorted.to_csv(os.path.join(FIG_OUT, "Feature_SHAP_Importance.csv"), index=False)

print("=" * 70)
print(f"SHAP Analysis completed successfully! Output directory: {FIG_OUT}")
print("Generated Files:")
print("1. Fig15a_SHAP_summary.jpg/pdf")
print("2. Fig15b_Global_Importance.jpg/pdf")
print("3. Fig15c_Waterfall_single_sample.jpg/pdf")
print("4. Feature_SHAP_Importance.csv")
print("=" * 70)