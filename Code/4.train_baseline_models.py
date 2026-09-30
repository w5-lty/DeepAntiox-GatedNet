# coding: utf-8
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, average_precision_score, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef,
    roc_curve, precision_recall_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

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
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['legend.frameon'] = False

COLORS = {
    'LR': '#1f77b4', 'RF': '#ff7f0e', 'XGB': '#2ca02c',
    'SVM': '#d62728', 'KNN': '#9467bd', 'LGB': '#8c564b', 'MLP': '#e377c2'
}
MODEL_LIST = ['LR', 'RF', 'XGB', 'SVM', 'KNN', 'LGB', 'MLP']
METRICS = ['AUC', 'AUPR', 'SEN', 'SPE', 'PRE', 'F1', 'ACC', 'MCC']

# ===================== Paths Configuration =====================
BASE = "./"
INPUT_CSV = os.path.join(BASE, "Results/2_Feature_Extraction_Selection_v2/csv_output/03_Top30_Selected_Features.csv")
EMBEDDING_FILE = os.path.join(BASE, "Data/One-hot_encoding.txt")
OUT_DIR = os.path.join(BASE, "Results/3_Baseline_ML_Results")
FIG_DIR = os.path.join(OUT_DIR, "Figures")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


# ===================== Feature Extraction Methods =====================
def load_aa_embedding(emb_file):
    aa_dict = {}
    if not os.path.exists(emb_file):
        print(f"Warning: Embedding file {emb_file} not found. Falling back to 20-dim One-hot initialization.")
        std_aas = list("ACDEFGHIKLMNPQRSTVWY")
        dim = len(std_aas)
        for idx, aa in enumerate(std_aas):
            vec = np.zeros(dim, dtype=np.float32)
            vec[idx] = 1.0
            aa_dict[aa] = vec
        return aa_dict, dim

    with open(emb_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                parts = line.split()
                aa = parts[0]
                vec = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                aa_dict[aa] = vec
    dim = len(next(iter(aa_dict.values())))
    print(f"Successfully loaded embeddings: {emb_file}, Dimension: {dim}")
    return aa_dict, dim


def extract_sequence_pooled_features(seqs, aa_dict, emb_dim):
    pooled_feats = []
    for s in seqs:
        s_clean = str(s).strip().upper()
        vecs = [aa_dict[aa] for aa in s_clean if aa in aa_dict]
        if len(vecs) == 0:
            vecs = [np.zeros(emb_dim, dtype=np.float32)]
        vecs = np.array(vecs, dtype=np.float32)

        gap_vec = np.mean(vecs, axis=0)
        pooled_feats.append(gap_vec)

    return np.array(pooled_feats, dtype=np.float32)


# ===================== Modeling & Evaluation =====================
def get_models():
    return {
        'LR': LogisticRegression(max_iter=2000, random_state=42),
        'RF': RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1),
        'XGB': xgb.XGBClassifier(n_estimators=150, random_state=42, n_jobs=-1, eval_metric='logloss'),
        'SVM': SVC(probability=True, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'LGB': lgb.LGBMClassifier(n_estimators=150, random_state=42, n_jobs=-1, verbosity=-1),
        'MLP': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=42)
    }


def evaluate(y_true, y_pred, y_prob):
    sen = recall_score(y_true, y_pred, zero_division=0)
    spe = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    pre = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    auc_val = roc_auc_score(y_true, y_prob)
    aupr = average_precision_score(y_true, y_prob)
    return auc_val, aupr, sen, spe, pre, f1, acc, mcc


# ===================== Visualization =====================
def plot_roc_curves_from_history(roc_history, save_base_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    for name in MODEL_LIST:
        mean_tpr = np.mean(roc_history[name]['tprs'], axis=0)
        mean_auc = np.mean(roc_history[name]['aucs'])
        ax.plot(np.linspace(0, 1, 100), mean_tpr,
                label=f'{name} (AUC={mean_auc:.3f})',
                color=COLORS[name], linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves (5-Fold Mean)', fontweight='bold')
    ax.legend(loc='lower right')
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_base_path + ".pdf", format='pdf')
    plt.savefig(save_base_path + ".jpg", format='jpg')
    plt.close()


def plot_pr_curves_from_history(pr_history, save_base_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    for name in MODEL_LIST:
        mean_pr = np.mean(pr_history[name]['prs'], axis=0)
        mean_aupr = np.mean(pr_history[name]['auprs'])
        ax.plot(np.linspace(0, 1, 100), mean_pr,
                label=f'{name} (AUPR={mean_aupr:.3f})',
                color=COLORS[name], linewidth=2)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curves (5-Fold Mean)', fontweight='bold')
    ax.legend(loc='lower left')
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_base_path + ".pdf", format='pdf')
    plt.savefig(save_base_path + ".jpg", format='jpg')
    plt.close()


def plot_metrics_bar(df_mean, save_base_path):
    df_plot = df_mean.melt(id_vars=['Model'], value_vars=METRICS, var_name='Metric', value_name='Score')
    fig, ax = plt.subplots(figsize=(14, 5))
    sns.barplot(data=df_plot, x='Metric', y='Score', hue='Model', palette=COLORS, ax=ax)
    ax.set_title('Baseline Models Performance Comparison', fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_base_path + ".pdf", format='pdf')
    plt.savefig(save_base_path + ".jpg", format='jpg')
    plt.close()


# ===================== Main Execution Pipeline =====================
if not os.path.exists(INPUT_CSV):
    raise FileNotFoundError(f"Input file not found: {INPUT_CSV}")

print(f"Loading dataset: {INPUT_CSV}")
df = pd.read_csv(INPUT_CSV)
meta_cols = ["SEQUENCE", "label", "partition"]
feat_cols = [c for c in df.columns if c not in meta_cols]

X_handcrafted = df[feat_cols].values.astype(np.float32)

aa_dict, emb_dim = load_aa_embedding(EMBEDDING_FILE)
X_seq_pooled = extract_sequence_pooled_features(df["SEQUENCE"].values, aa_dict, emb_dim)

X = np.hstack([X_handcrafted, X_seq_pooled])
y = df["label"].values
partition = df["partition"].values

print(
    f"Feature construction completed: Handcrafted dims = {X_handcrafted.shape[1]}, Sequence GAP dims = {X_seq_pooled.shape[1]}, Total dims = {X.shape[1]}")

all_metric_records = []
all_sample_prediction = []

roc_plot_data = {m: {'tprs': [], 'aucs': []} for m in MODEL_LIST}
pr_plot_data = {m: {'prs': [], 'auprs': []} for m in MODEL_LIST}

for fold in range(5):
    print(f"\n==== Fold {fold + 1}/5 ====")
    train_mask = partition != fold
    test_mask = partition == fold
    test_index = df.index[test_mask]

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    models = get_models()

    for m_name in MODEL_LIST:
        clf = models[m_name]
        clf.fit(X_train, y_train)
        y_prob = clf.predict_proba(X_test)[:, 1]
        y_pred_label = (y_prob >= 0.5).astype(int)

        auc_val, aupr, sen, spe, pre, f1, acc, mcc = evaluate(y_test, y_pred_label, y_prob)
        all_metric_records.append({
            "Fold": fold,
            "Model": m_name,
            "AUC": auc_val, "AUPR": aupr, "SEN": sen, "SPE": spe,
            "PRE": pre, "F1": f1, "ACC": acc, "MCC": mcc
        })

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_plot_data[m_name]['tprs'].append(np.interp(np.linspace(0, 1, 100), fpr, tpr))
        roc_plot_data[m_name]['aucs'].append(auc_val)

        p, r, _ = precision_recall_curve(y_test, y_prob)
        pr_plot_data[m_name]['prs'].append(np.interp(np.linspace(0, 1, 100), r[::-1], p[::-1]))
        pr_plot_data[m_name]['auprs'].append(aupr)

        for i, idx in enumerate(test_index):
            all_sample_prediction.append({
                "Model": m_name,
                "Fold": fold,
                "Sequence": df.loc[idx, "SEQUENCE"],
                "y_true": int(df.loc[idx, "label"]),
                "y_pred_prob": float(y_prob[i]),
                "y_pred_label": int(y_pred_label[i])
            })

df_perfold_metric = pd.DataFrame(all_metric_records)
df_perfold_metric.to_csv(os.path.join(OUT_DIR, "PerFold_Baseline_Metrics.csv"), index=False)

df_all_pred = pd.DataFrame(all_sample_prediction)
df_all_pred.to_csv(os.path.join(OUT_DIR, "Predictions_Baseline_All.csv"), index=False)

df_mean_metric = df_perfold_metric.groupby("Model")[METRICS].mean().reset_index()
df_mean_metric.to_csv(os.path.join(OUT_DIR, "Baseline_Mean_Metrics.csv"), index=False)

df_std_metric = df_perfold_metric.groupby("Model")[METRICS].std().reset_index()
df_summary = pd.DataFrame({'Model': df_mean_metric['Model']})
for col in METRICS:
    df_summary[col] = df_mean_metric[col].map('{:.4f}'.format) + " ± " + df_std_metric[col].map('{:.4f}'.format)
df_summary.to_csv(os.path.join(OUT_DIR, "Baseline_Mean_Std_Metrics.csv"), index=False)

plot_roc_curves_from_history(roc_plot_data, os.path.join(FIG_DIR, "Baseline_ROC"))
plot_pr_curves_from_history(pr_plot_data, os.path.join(FIG_DIR, "Baseline_PR"))
plot_metrics_bar(df_mean_metric, os.path.join(FIG_DIR, "Baseline_Metrics_Bar"))

print("\n" + "=" * 70)
print("Baseline ML models evaluation completed successfully!")
print(f"Output directory: {OUT_DIR}")
print("Generated files:")
print("1. PerFold_Baseline_Metrics.csv")
print("2. Predictions_Baseline_All.csv")
print("3. Baseline_Mean_Metrics.csv")
print("4. Baseline_Mean_Std_Metrics.csv")
print("5. Figures/Baseline_ROC.jpg/pdf")
print("6. Figures/Baseline_PR.jpg/pdf")
print("7. Figures/Baseline_Metrics_Bar.jpg/pdf")
print("=" * 70)