# -*- coding: utf-8 -*-
import sys
import os
import csv
import re
import numpy as np
import pandas as pd
from Bio import SeqIO

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTextEdit,
    QPushButton, QFileDialog, QTableWidget, QTableWidgetItem, QProgressBar,
    QLabel, QMessageBox, QHeaderView, QGroupBox, QFrame, QSplitter, QGridLayout
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QColor, QFont
import tensorflow as tf
from tensorflow import keras

# ===================== Global Paths Configuration =====================
BASE_DIR = "."
MODEL_PATH = os.path.join(BASE_DIR, "Results", "Result_A_GatedNet_Alone", "Trained_Models", "Final_Model_p90.keras")
EMBEDDING_FILE = os.path.join(BASE_DIR, "Data", "One‑hot_encoding.txt")
TOP30_CSV = os.path.join(BASE_DIR, "Results", "2_Feature_Extraction_Selection_v2", "csv_output",
                         "03_Top30_Selected_Features.csv")

STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
MAX_SEQ_LEN = 30

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


CUSTOM_OBJ = {
    "PartialConv1D_Module": PartialConv1D_Module,
    "DualPath_Attention": DualPath_Attention,
    "BiGated_Fusion_Module": BiGated_Fusion_Module
}

# ===================== Feature Extraction Tools (完整原版A特征计算) =====================
CTD_GROUPS = {
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

AA_PHYS = {
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
AA_PROPS = ['Hydrophobicity', 'NetCharge', 'IsoelectricPoint', 'Polarity', 'MolecularWeight', 'SideChainVolume']


class FastFeatureExtractor:
    def __init__(self):
        self.aa_list = list(STANDARD_AA)

    def extract_single(self, seq):
        seq = str(seq).strip().upper()
        L = len(seq)
        f_dict = {}
        for aa in self.aa_list:
            f_dict[f"AAC_{aa}"] = round(seq.count(aa) / L, 6)
        for a1 in self.aa_list:
            for a2 in self.aa_list:
                dp = a1 + a2
                f_dict[f"DPC_{dp}"] = round(seq.count(dp) / (L - 1), 6) if L > 1 else 0.0
        for skip in range(4):
            denom = L - skip - 1
            for a1 in self.aa_list:
                for a2 in self.aa_list:
                    cnt = sum(1 for i in range(denom) if seq[i] == a1 and seq[i + skip + 1] == a2) if denom > 0 else 0
                    f_dict[f"ASDC_skip{skip}_{a1}{a2}"] = round(cnt / denom, 6) if denom > 0 else 0.0
        for gap in range(3):
            denom = L - gap - 1
            for a1 in self.aa_list:
                for a2 in self.aa_list:
                    cnt = sum(1 for i in range(denom) if seq[i] == a1 and seq[i + gap + 1] == a2) if denom > 0 else 0
                    f_dict[f"CKSAAP_gap{gap}_{a1}{a2}"] = round(cnt / denom, 6) if denom > 0 else 0.0
        for prop_name, groups in CTD_GROUPS.items():
            for g_name, g_aa in groups.items():
                cnt = sum(1 for aa in seq if aa in g_aa)
                f_dict[f"CTD_{prop_name}_C_{g_name}"] = round(cnt / L, 6)
            trans = 0
            if L > 1:
                for i in range(L - 1):
                    g1 = next((k for k, v in groups.items() if seq[i] in v), None)
                    g2 = next((k for k, v in groups.items() if seq[i + 1] in v), None)
                    if g1 and g2 and g1 != g2:
                        trans += 1
            f_dict[f"CTD_{prop_name}_T"] = round(trans / (L - 1), 6) if L > 1 else 0.0
            for g_name, g_aa in groups.items():
                poslist = [i + 1 for i, aa in enumerate(seq) if aa in g_aa]
                if not poslist:
                    dvals = [0.0] * 5
                else:
                    dvals = [round(np.percentile(poslist, p) / L, 6) for p in [0, 25, 50, 70, 100]]
                for idx, p in enumerate([0, 25, 50, 70, 100]):
                    f_dict[f"CTD_{prop_name}_D_{g_name}_P{p}"] = dvals[idx]
        for l in range(1, 11):
            corr = (sum(1 for i in range(L - l) if seq[i] == seq[i + l]) / (L - l)) if L > l else 0.0
            f_dict[f"PAAC_corr_{l}"] = round(corr, 6)
            f_dict[f"PseAAC_corr_{l}"] = round(corr, 6)
        for aa in self.aa_list:
            f_dict[f"PAAC_AAC_{aa}"] = f_dict[f"AAC_{aa}"]
            f_dict[f"PseAAC_AAC_{aa}"] = f_dict[f"AAC_{aa}"]
        for prop in AA_PROPS:
            idx = AA_PROPS.index(prop)
            arr = [AA_PHYS[aa][idx] for aa in seq if aa in AA_PHYS]
            if not arr:
                stats = [0.0] * 5
            else:
                stats = [round(np.mean(arr), 6), round(np.std(arr), 6), round(np.max(arr), 6), round(np.min(arr), 6),
                         round(np.sum(arr), 6)]
            for s_name, val in zip(["mean", "std", "max", "min", "sum"], stats):
                f_dict[f"AAindex_{prop}_{s_name}"] = val
        return f_dict


def load_embedding_dict(emb_file):
    aa_list, vs = [], []
    if not os.path.exists(emb_file):
        aa_list = list("ACDEFGHIKLMNPQRSTVWY")
        mat = np.random.randn(20, 21).astype(np.float32)
    else:
        with open(emb_file, "r", encoding="utf‑8") as f:
            for line in f:
                if line.strip() and not line.startswith("#"):
                    parts = line.split()
                    aa_list.append(parts[0])
                    vs.append([float(x) for x in parts[1:]])
        mat = np.array(vs, dtype=np.float32)
    aa_to_idx = {a: i for i, a in enumerate(aa_list)}
    return mat, aa_to_idx


def encode_single_seq(seq, embedding_mat, aa_to_idx):
    enc = np.zeros((1, MAX_SEQ_LEN, embedding_mat.shape[1]), dtype=np.float32)
    for j, c in enumerate(str(seq)[:MAX_SEQ_LEN]):
        if c in aa_to_idx:
            enc[0, j] = embedding_mat[aa_to_idx[c]]
    return enc


# ===================== Background Inference Thread 支持中止+逐条返回 =====================
class PredictThread(QThread):
    result_item = pyqtSignal(dict)
    progress_signal = pyqtSignal(int)
    error_signal = pyqtSignal(str)
    finish_signal = pyqtSignal(bool)

    def __init__(self, seq_list, model, extractor, top30_feats, emb_mat, aa_to_idx):
        super().__init__()
        self.seq_list = seq_list
        self.model = model
        self.extractor = extractor
        self.top30 = top30_feats
        self.emb_mat = emb_mat
        self.aa_to_idx = aa_to_idx
        self._is_running = True

    def stop(self):
        self._is_running = False

    def run(self):
        total = len(self.seq_list)
        try:
            for idx, seq in enumerate(self.seq_list):
                if not self._is_running:
                    self.finish_signal.emit(True)
                    return
                seq_enc = encode_single_seq(seq, self.emb_mat, self.aa_to_idx)
                f_dict = self.extractor.extract_single(seq)
                feat_vec = np.array([[f_dict.get(fname, 0.0) for fname in self.top30]], dtype=np.float32)
                pred_score = float(self.model.predict([seq_enc, feat_vec], verbose=0)[0, 0])
                label = "Antioxidant" if pred_score >= 0.5 else "Non‑antioxidant"
                self.result_item.emit({
                    "seq": seq,
                    "length": len(seq),
                    "score": round(pred_score, 4),
                    "label": label
                })
                self.progress_signal.emit(int(100 * (idx + 1) / total))
            self.finish_signal.emit(False)
        except Exception as e:
            self.error_signal.emit(str(e))


# ===================== GUI Main Window 美化UI =====================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DeepAntiox‑GatedNet Antioxidant Peptide Prediction Platform")
        self.resize(1200, 800)
        self.results_data = []
        self.predict_thread = None

        if not os.path.exists(MODEL_PATH):
            QMessageBox.critical(self, "Error", f"Model file not found:\n{MODEL_PATH}")
            sys.exit(1)

        self.model = keras.models.load_model(MODEL_PATH, custom_objects=CUSTOM_OBJ)
        self.emb_mat, self.aa_to_idx = load_embedding_dict(EMBEDDING_FILE)

        self.top30_feats = []
        if os.path.exists(TOP30_CSV):
            df_top = pd.read_csv(TOP30_CSV, nrows=1)
            meta = ["SEQUENCE", "label", "partition"]
            self.top30_feats = [c for c in df_top.columns if c not in meta]

        self.extractor = FastFeatureExtractor()
        self.apply_styles()
        self.init_ui()

    def apply_styles(self):
        self.setStyleSheet("""
        QMainWindow { background‑color: #f0f2f5; }
        QGroupBox { font‑weight: bold; border: 2px solid #dcdfe6; border‑radius:10px; margin‑top:20px; background‑color:white; }
        QGroupBox::title { subcontrol‑origin:margin; left:15px; padding:0 5px; color:#1890ff; }
        QPushButton { border‑radius:6px; padding:10px; color:white; font‑weight:bold; font‑size:13px; }
        QPushButton#btn_start { background‑color:#2CA02C; }
        QPushButton#btn_stop { background‑color:#ff4d4f; }
        QPushButton#btn_upload { background‑color:#1890ff; }
        QPushButton#btn_export { background‑color:#faad14; }
        QProgressBar { border:1px solid #dcdfe6; border‑radius:8px; height:20px; text‑align:center; }
        QProgressBar::chunk { background‑color:#1890ff; border‑radius:8px; }
        QTableWidget { border:none; gridline‑color:#f0f0f0; alternate‑background‑color:#fafafa; }
        QHeaderView::section { background‑color:#fafafa; padding:8px; border:1px solid #f0f0f0; font‑weight:bold; }
        """)

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(20, 20, 20, 20)

        header = QLabel("DeepAntiox‑GatedNet Antioxidant Peptide Prediction Platform")
        header.setFont(QFont("Arial", 18, QFont.Bold))
        sub_header = QLabel("High‑throughput screening for antioxidant peptides (length: 2‑30 standard amino acids)")
        main_layout.addWidget(header)
        main_layout.addWidget(sub_header)

        splitter = QSplitter(Qt.Horizontal)
        left_panel = QFrame()
        lay_left = QVBoxLayout(left_panel)

        input_group = QGroupBox("Input Configuration")
        v_in = QVBoxLayout()
        self.text_input = QTextEdit()
        self.text_input.setPlaceholderText("Paste peptide sequences here (one per line)\nOr upload FASTA / CSV / TXT file.")
        v_in.addWidget(self.text_input)
        self.btn_upload = QPushButton("📁 Upload FASTA/CSV/TXT")
        self.btn_upload.setObjectName("btn_upload")
        self.btn_upload.clicked.connect(self.load_file)
        v_in.addWidget(self.btn_upload)
        input_group.setLayout(v_in)
        lay_left.addWidget(input_group)

        ctrl_group = QGroupBox("Control Panel")
        g_layout = QGridLayout()
        self.btn_start = QPushButton("🚀 Run Prediction")
        self.btn_start.setObjectName("btn_start")
        self.btn_start.clicked.connect(self.run_predict)
        self.btn_stop = QPushButton("⏹ Abort Task")
        self.btn_stop.setObjectName("btn_stop")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.abort_task)
        self.btn_export = QPushButton("💾 Export Result")
        self.btn_export.setObjectName("btn_export")
        self.btn_export.clicked.connect(self.export_result)

        g_layout.addWidget(self.btn_start, 0, 0)
        g_layout.addWidget(self.btn_stop, 0, 1)
        g_layout.addWidget(self.btn_export, 1, 0, 1, 2)
        ctrl_group.setLayout(g_layout)
        lay_left.addWidget(ctrl_group)

        self.progress_label = QLabel("Progress: 0%")
        lay_left.addWidget(self.progress_label)
        self.progress = QProgressBar()
        lay_left.addWidget(self.progress)
        lay_left.addStretch()

        right_panel = QFrame()
        lay_right = QVBoxLayout(right_panel)
        res_group = QGroupBox("Prediction Results")
        v_res = QVBoxLayout()
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Sequence", "Length", "Probability Score", "Classification"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setAlternatingRowColors(True)
        v_res.addWidget(self.table)
        res_group.setLayout(v_res)
        lay_right.addWidget(res_group)

        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(1, 2)
        main_layout.addWidget(splitter)

    def load_file(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Open File", "",
                                               "Supported (*.fasta *.fa *.csv *.txt);;FASTA(*.fasta *.fa);;CSV(*.csv);;TXT(*.txt)")
        if not fname:
            return
        seqs = []
        try:
            if fname.lower().endswith((".fasta", ".fa")):
                for rec in SeqIO.parse(fname, "fasta"):
                    seqs.append(str(rec.seq).upper())
            elif fname.lower().endswith(".csv"):
                df = pd.read_csv(fname)
                col_candidate = [c for c in df.columns if "SEQUENCE" in c.upper()]
                use_col = col_candidate[0] if col_candidate else df.columns[0]
                seqs = df[use_col].astype(str).str.upper().tolist()
            else:
                with open(fname, "r", encoding="utf‑8") as f:
                    for line in f:
                        s = line.strip().upper()
                        if s and not s.startswith(">"):
                            seqs.append(s)
            self.text_input.setPlainText("\n".join(seqs))
            QMessageBox.information(self, "Success", f"Loaded {len(seqs)} sequences.")
        except Exception as e:
            QMessageBox.critical(self, "File Error", str(e))

    def run_predict(self):
        raw_lines = self.text_input.toPlainText().splitlines()
        seq_clean = []
        for line in raw_lines:
            s = line.strip().upper()
            if not s or s.startswith(">"):
                continue
            s_alpha = re.sub(r'[^A‑Z]', '', s)
            if 2 <= len(s_alpha) <= 30 and all(ch in STANDARD_AA for ch in s_alpha):
                seq_clean.append(s_alpha)
        if not seq_clean:
            QMessageBox.warning(self, "Warning", "No valid peptide!\nLength 2‑30, only standard 20 amino acids.")
            return
        if len(self.top30_feats) == 0:
            QMessageBox.critical(self, "Error", "Top30 feature csv file missing!")
            return

        self.results_data.clear()
        self.table.setRowCount(0)
        self.progress.setValue(0)
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_upload.setEnabled(False)

        self.predict_thread = PredictThread(seq_clean, self.model, self.extractor,
                                            self.top30_feats, self.emb_mat, self.aa_to_idx)
        self.predict_thread.result_item.connect(self.on_one_result)
        self.predict_thread.progress_signal.connect(lambda v: (self.progress.setValue(v), self.progress_label.setText(f"Progress: {v}%")))
        self.predict_thread.error_signal.connect(self.on_err)
        self.predict_thread.finish_signal.connect(self.on_finish)
        self.predict_thread.start()

    def abort_task(self):
        if self.predict_thread:
            self.predict_thread.stop()

    def on_one_result(self, d):
        self.results_data.append(d)
        row = self.table.rowCount()
        self.table.insertRow(row)
        item_seq = QTableWidgetItem(d["seq"])
        item_len = QTableWidgetItem(str(d["length"]))
        item_score = QTableWidgetItem(f"{d['score']:.4f}")
        item_label = QTableWidgetItem(d["label"])
        for it in [item_seq, item_len, item_score, item_label]:
            it.setTextAlignment(Qt.AlignCenter)
        if d["label"] == "Antioxidant":
            item_label.setBackground(QColor("#FFCDD2"))
            item_label.setForeground(QColor("#B71C1C"))
        else:
            item_label.setBackground(QColor("#E8F5E9"))
            item_label.setForeground(QColor("#1B5E20"))
        self.table.setItem(row, 0, item_seq)
        self.table.setItem(row, 1, item_len)
        self.table.setItem(row, 2, item_score)
        self.table.setItem(row, 3, item_label)
        self.table.scrollToBottom()

    def on_err(self, msg):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_upload.setEnabled(True)
        QMessageBox.critical(self, "Prediction Error", msg)

    def on_finish(self, is_aborted):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_upload.setEnabled(True)
        if is_aborted:
            QMessageBox.warning(self, "Notice", "Prediction task was aborted by user.")
        else:
            QMessageBox.information(self, "Complete", f"Finished, total {len(self.results_data)} peptides processed.")

    def export_result(self):
        if not self.results_data:
            QMessageBox.warning(self, "Warning", "No prediction data to export!")
            return
        save_path, _ = QFileDialog.getSaveFileName(self, "Save results", "Prediction_Results.csv",
                                                  "CSV (*.csv);;Excel(*.xlsx)")
        if not save_path:
            return
        df_out = pd.DataFrame(self.results_data)
        df_out.columns = ["Sequence", "Length", "Probability_Score", "Classification"]
        try:
            if save_path.endswith(".xlsx"):
                df_out.to_excel(save_path, index=False)
            else:
                df_out.to_csv(save_path, index=False, encoding="utf‑8‑sig")
            QMessageBox.information(self, "Saved", f"File saved:\n{save_path}")
        except Exception as e:
            QMessageBox.critical(self, "Export failed", str(e))


if __name__ == "__main__":
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)
    app.setFont(QFont("Arial", 10))
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
