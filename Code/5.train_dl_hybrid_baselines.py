# coding: utf-8
import matplotlib

matplotlib.use('Agg')
import numpy as np
import tensorflow as tf
import pandas as pd
import os
import warnings
import gc
import random
from scipy import stats
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix
from tensorflow.keras import layers, models, backend as K

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

# ===================== Paths Configuration =====================
BASE_DIR = "."
PART2_CSV = os.path.join(BASE_DIR, "Results", "2_Feature_Extraction_Selection_v2", "csv_output",
                         "03_Top30_Selected_Features.csv")
embedding_file = os.path.join(BASE_DIR, "Data", "One-hot_encoding.txt")

DL_BASE_OUT = os.path.join(BASE_DIR, "Results", "Result_DL_Baselines_Hybrid")
os.makedirs(DL_BASE_OUT, exist_ok=True)

# ===================== Utilities & Metrics =====================
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


def bootstrap_ci(y_true, y_prob, n_bootstraps=1000, alpha=0.95):
    rng = np.random.RandomState(SEED)
    bootstrapped_scores = {m: [] for m in ['AUPR', 'SEN', 'SPE', 'PRE', 'F1', 'ACC', 'MCC']}
    for i in range(n_bootstraps):
        indices = rng.randint(0, len(y_prob), len(y_prob))
        if len(np.unique(y_true[indices])) < 2:
            continue
        m_dict = calc_metrics(y_true[indices], y_prob[indices])
        for m in bootstrapped_scores.keys():
            bootstrapped_scores[m].append(m_dict[m])

    lower_p = ((1.0 - alpha) / 2.0) * 100
    upper_p = (alpha + ((1.0 - alpha) / 2.0)) * 100
    cis = {}
    for m in bootstrapped_scores.keys():
        cis[m] = (
            np.percentile(bootstrapped_scores[m], lower_p),
            np.percentile(bootstrapped_scores[m], upper_p)
        )
    return cis


def compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def fastDeLong(preds_trans, pos_cnt):
    m = pos_cnt
    n = preds_trans.shape[1] - m
    pos = preds_trans[:, :m]
    neg = preds_trans[:, m:]
    k = preds_trans.shape[0]
    tx = np.empty([k, m])
    ty = np.empty([k, n])
    tz = np.empty([k, m + n])
    for r in range(k):
        tx[r, :] = compute_midrank(pos[r, :])
        ty[r, :] = compute_midrank(neg[r, :])
        tz[r, :] = compute_midrank(preds_trans[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    cov = sx / m + sy / n
    return aucs, cov


def auc_95ci(y_true, y_score, alpha=0.95):
    order = (-y_true).argsort()
    y_sorted = y_true[order]
    s_sorted = y_score[order]
    pred = np.vstack((s_sorted,))
    poscnt = int(np.sum(y_sorted == 1))
    negcnt = int(np.sum(y_sorted == 0))
    if poscnt < 1 or negcnt < 1:
        a = auc(y_true, y_score)
        return a, 0.0, 1.0
    aucs, cov = fastDeLong(pred, poscnt)
    aucv = aucs[0]
    cov = np.atleast_2d(cov)
    se = np.sqrt(cov[0, 0])
    z = stats.norm.ppf((1 + alpha) / 2)
    low = max(0.0, aucv - z * se)
    high = min(1.0, aucv + z * se)
    return aucv, low, high


# ===================== Dual-Input DL Baseline Models =====================
def build_cnn_baseline(seq_shape, feat_dim):
    inp_seq = layers.Input(seq_shape, name="Seq_In")
    x = layers.Conv1D(filters=128, kernel_size=3, padding="same", activation="swish")(inp_seq)
    x = layers.GlobalAveragePooling1D()(x)
    x_seq = layers.Dense(64, activation="swish")(x)

    inp_feat = layers.Input((feat_dim,), name="Feat_In")
    f = layers.BatchNormalization()(inp_feat)
    x_feat = layers.Dense(64, activation="swish")(f)

    concat = layers.concatenate([x_seq, x_feat])
    z = layers.Dense(64, activation="swish")(concat)
    z = layers.Dropout(0.3)(z)
    out = layers.Dense(1, activation="sigmoid")(z)

    model = models.Model([inp_seq, inp_feat], out)
    model.compile(tf.keras.optimizers.Adam(2e-4), loss="binary_crossentropy", metrics=["accuracy"])
    return model


def build_textcnn(seq_shape, feat_dim):
    inp_seq = layers.Input(seq_shape, name="Seq_In")
    c3 = layers.Conv1D(64, kernel_size=3, padding="same", activation="swish")(inp_seq)
    c4 = layers.Conv1D(64, kernel_size=4, padding="same", activation="swish")(inp_seq)
    c5 = layers.Conv1D(64, kernel_size=5, padding="same", activation="swish")(inp_seq)

    p3 = layers.GlobalMaxPooling1D()(c3)
    p4 = layers.GlobalMaxPooling1D()(c4)
    p5 = layers.GlobalMaxPooling1D()(c5)

    cat_seq = layers.concatenate([p3, p4, p5])
    x_seq = layers.Dense(64, activation="swish")(cat_seq)

    inp_feat = layers.Input((feat_dim,), name="Feat_In")
    f = layers.BatchNormalization()(inp_feat)
    x_feat = layers.Dense(64, activation="swish")(f)

    concat = layers.concatenate([x_seq, x_feat])
    z = layers.Dense(64, activation="swish")(concat)
    z = layers.Dropout(0.3)(z)
    out = layers.Dense(1, activation="sigmoid")(z)

    model = models.Model([inp_seq, inp_feat], out)
    model.compile(tf.keras.optimizers.Adam(2e-4), loss="binary_crossentropy", metrics=["accuracy"])
    return model


def build_lstm(seq_shape, feat_dim):
    inp_seq = layers.Input(seq_shape, name="Seq_In")
    x = layers.LSTM(128, dropout=0.2, return_sequences=False)(inp_seq)
    x_seq = layers.Dense(64, activation="swish")(x)

    inp_feat = layers.Input((feat_dim,), name="Feat_In")
    f = layers.BatchNormalization()(inp_feat)
    x_feat = layers.Dense(64, activation="swish")(f)

    concat = layers.concatenate([x_seq, x_feat])
    z = layers.Dense(64, activation="swish")(concat)
    z = layers.Dropout(0.3)(z)
    out = layers.Dense(1, activation="sigmoid")(z)

    model = models.Model([inp_seq, inp_feat], out)
    model.compile(tf.keras.optimizers.Adam(2e-4), loss="binary_crossentropy", metrics=["accuracy"])
    return model


def build_bilstm(seq_shape, feat_dim):
    inp_seq = layers.Input(seq_shape, name="Seq_In")
    x = layers.Bidirectional(layers.LSTM(128, dropout=0.2, return_sequences=False))(inp_seq)
    x_seq = layers.Dense(64, activation="swish")(x)

    inp_feat = layers.Input((feat_dim,), name="Feat_In")
    f = layers.BatchNormalization()(inp_feat)
    x_feat = layers.Dense(64, activation="swish")(f)

    concat = layers.concatenate([x_seq, x_feat])
    z = layers.Dense(64, activation="swish")(concat)
    z = layers.Dropout(0.3)(z)
    out = layers.Dense(1, activation="sigmoid")(z)

    model = models.Model([inp_seq, inp_feat], out)
    model.compile(tf.keras.optimizers.Adam(2e-4), loss="binary_crossentropy", metrics=["accuracy"])
    return model


DL_BASELINE_MODELS = {
    "Baseline-CNN": build_cnn_baseline,
    "TextCNN": build_textcnn,
    "LSTM": build_lstm,
    "BiLSTM": build_bilstm
}

# ===================== Dataset Loading & Preparation =====================
print(f"Loading dataset: {PART2_CSV}")
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

print(f"Total samples: {len(df_all)}, Hand-crafted feature dimension: {info['feat_dim']}")
seq_shape = (MAX_SEQ_LEN, embedding_mat.shape[1])

# ===================== Evaluation Loop (Nested 5-Fold CV) =====================
for model_name, model_builder in DL_BASELINE_MODELS.items():
    print(f"\n==================== Running Model: {model_name} ====================")
    model_out_root = os.path.join(DL_BASE_OUT, model_name)
    pred_save_path = os.path.join(model_out_root, "Predictions")
    final_result_path = os.path.join(model_out_root, "Final_Paper_Results")
    for p in [model_out_root, pred_save_path, final_result_path]:
        os.makedirs(p, exist_ok=True)

    fold_pred_records = []
    fold_metrics = pd.DataFrame()

    for fold in range(5):
        print(f"\n--- {model_name} Fold {fold + 1}/5 ---")
        val_f = (fold + 1) % 5
        p_arr = info["partition"]
        mask_test = p_arr == fold
        mask_val = p_arr == val_f
        mask_train = (~mask_test) & (~mask_val)

        xtr_seq = encode_seqs(info["seq"][mask_train])
        xtr_feat = info["feat"][mask_train]

        xval_seq = encode_seqs(info["seq"][mask_val])
        xval_feat = info["feat"][mask_val]

        xte_seq = encode_seqs(info["seq"][mask_test])
        xte_feat = info["feat"][mask_test]

        ytr = info["label"][mask_train]
        yval = info["label"][mask_val]
        yte = info["label"][mask_test]
        seq_te = info["seq"][mask_test]

        K.clear_session()
        gc.collect()

        model = model_builder(seq_shape, info["feat_dim"])
        es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=25, restore_best_weights=True)

        model.fit([xtr_seq, xtr_feat], ytr,
                  validation_data=([xval_seq, xval_feat], yval),
                  epochs=150, batch_size=32, verbose=0, callbacks=[es])

        yp = model.predict([xte_seq, xte_feat], verbose=0).flatten()
        met = calc_metrics(yte, yp)
        fold_metrics = pd.concat([fold_metrics, pd.DataFrame([met])])

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
    df_pred.to_csv(os.path.join(pred_save_path, f"{model_name}_FiveFold_Prediction.csv"), index=False)
    fold_metrics.to_csv(os.path.join(final_result_path, "PerFold_Metrics.csv"), index=False)

    y_all_true = df_pred["y_true"].values
    y_all_prob = df_pred["y_pred_prob"].values
    auc_val, ci_low_auc, ci_high_auc = auc_95ci(y_all_true, y_all_prob)

    print(f"Calculating 95% CI for {model_name} via Bootstrap (1000 resamples)...")
    other_cis = bootstrap_ci(y_all_true, y_all_prob)

    summary = pd.DataFrame({
        "Metric": metrics_list,
        "Mean": [fold_metrics[m].mean() for m in metrics_list],
        "Std": [fold_metrics[m].std() for m in metrics_list]
    })

    summary["95CI_low"] = 0.0
    summary["95CI_high"] = 0.0
    summary.loc[summary["Metric"] == "AUC", "95CI_low"] = ci_low_auc
    summary.loc[summary["Metric"] == "AUC", "95CI_high"] = ci_high_auc

    for m in ['AUPR', 'SEN', 'SPE', 'PRE', 'F1', 'ACC', 'MCC']:
        summary.loc[summary["Metric"] == m, "95CI_low"] = other_cis[m][0]
        summary.loc[summary["Metric"] == m, "95CI_high"] = other_cis[m][1]

    summary["95CI_low"] = summary["95CI_low"].round(4)
    summary["95CI_high"] = summary["95CI_high"].round(4)
    summary.to_csv(os.path.join(final_result_path, "Summary_MeanStd_CI.csv"), index=False)

    print(f"\n==== {model_name} (Hybrid: Seq + Hand-crafted) Nested 5-fold CV Summary ====")
    print(summary)

print("\n" + "=" * 70)
print("All hybrid DL baseline models (CNN/TextCNN/LSTM/BiLSTM) evaluated successfully!")
print(f"Output directory: {DL_BASE_OUT}")
print("1. Predictions/*_FiveFold_Prediction.csv       : Sample-wise predictions")
print("2. Final_Paper_Results/PerFold_Metrics.csv    : Metrics for each fold")
print("3. Final_Paper_Results/Summary_MeanStd_CI.csv : Mean±Std and 95% CI summary")
print("=" * 70)