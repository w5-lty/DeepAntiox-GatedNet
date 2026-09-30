import numpy as np
import pandas as pd
import tensorflow as tf
from Bio import pairwise2
import os
from sklearn.metrics import *
from tensorflow.keras.layers import *
from tensorflow.keras import Model
from tensorflow.keras import backend as K


# --- 1. Data Preprocessing & Alignment ---
def homology_reduction(a_df, a_cutoff=.9):
    _pos_raw = a_df[a_df['label'] == 1]
    _neg_raw = a_df[a_df['label'] == 0]
    _pos = hmlg_reduc(_pos_raw, cutoff=a_cutoff)
    _neg = hmlg_reduc(_neg_raw, cutoff=a_cutoff)
    return pd.concat([_pos, _neg]).reset_index(drop=True)


def hmlg_reduc(a_df, cutoff):
    if a_df.empty: return a_df
    a_columns = a_df.columns
    a_list = np.array(a_df).tolist()
    b_list = []
    seq_idx = 0
    for a_val in a_list:
        if not b_list:
            b_list.append(a_val)
        else:
            for b_val in b_list:
                ident = calc_ident(a_val[seq_idx], b_val[seq_idx])
                if ident >= cutoff: break
            else:
                b_list.append(a_val)
    return pd.DataFrame(data=b_list, columns=a_columns)


def calc_ident(a_seq, b_seq):
    match = pairwise2.align.globalms(str(a_seq), str(b_seq), 1, 0, -10, -10, penalize_end_gaps=False, score_only=True)
    return match / max(len(str(a_seq)), len(str(b_seq)))


def visualize_data(a_df, idx_name):
    pos = a_df[a_df['label'] == 1].shape[0]
    neg = a_df[a_df['label'] == 0].shape[0]
    return pd.DataFrame([[pos, neg, a_df.shape[0]]], index=[idx_name], columns=['Positive(1)', 'Negative(0)', 'TOTAL'])


# --- 2. Negative Sample Generation ---
def pep_generator(a_df, fsa_file, a_nr):
    from itertools import cycle
    pos_df = a_df[a_df['label'] == 1]
    unique, counts = np.unique(pos_df.Sequence.apply(len).values, return_counts=True)
    pep_dict = dict(zip(unique, counts))

    def convert_fsa_list(fasta_file):
        with open(fasta_file, 'r') as h:
            seq = [line.strip() for line in h if not line.startswith('>')]
        return "".join(seq)

    full_seq = convert_fsa_list(fsa_file)
    ran_pep_cycle = cycle(full_seq)
    pep_list = []
    for length in pep_dict:
        for _ in range(a_nr):
            seq = ''.join([next(ran_pep_cycle) for _ in range(length)])
            pep_list.append([seq, 'Random', 0])
    return pd.DataFrame(pep_list, columns=['Sequence', 'Source', 'label'])


def reduce_df(a_df, b_df, max_len=1):
    the_df = pd.concat([a_df, b_df, b_df]).drop_duplicates(subset=['Sequence'], keep=False)
    n_df = pd.DataFrame()
    for i in range(2, 31):
        tmp_df = the_df[the_df.Sequence.apply(len) == i]
        n_df = pd.concat([n_df, tmp_df.iloc[:max_len]])
    return n_df.reset_index(drop=True)


# --- 3. Partitioning & Auxiliary Logic ---
def homology_partition(a_s, ident, parts):
    sort_s = a_s.copy()
    a_list = []
    for seq in sort_s:
        tmp_s = sort_s.apply(lambda x: calc_ident(x, seq))
        a_list.append(tmp_s[tmp_s >= ident].index.values)
    c_list = [list(i) for i in a_list]
    while True:
        merged = False
        for i in range(len(c_list)):
            for j in range(i + 1, len(c_list)):
                if not set(c_list[i]).isdisjoint(c_list[j]):
                    c_list[i] = list(set(c_list[i] + c_list[j]))
                    c_list.pop(j)
                    merged = True;
                    break
            if merged: break
        if not merged: break
    hom_list = sorted(c_list, key=len, reverse=True)
    par_list = [[] for _ in range(parts)]
    import itertools
    idx_cycle = itertools.cycle(range(parts))
    for val in hom_list:
        par_list[next(idx_cycle)] += val
    return [[num for num, v in enumerate(a_s.index.values) if v in p] for p in par_list], len(c_list), c_list


def hc_part_visualizer(a_df, hom_list, hom_name):
    r_df = pd.DataFrame()
    for num, hom in enumerate(hom_list):
        tmp2_df = pd.DataFrame()
        for i in range(len(hom)):
            tmp_df = a_df.iloc[hom[i]]
            tmp3_df = pd.DataFrame([[len(tmp_df), tmp_df.label.sum()]], columns=[f'Sum_P{i + 1}', f'pos_P{i + 1}'])
            tmp2_df = pd.concat([tmp2_df, tmp3_df], axis=1)
        tmp2_df.index = [f'Ident_{hom_name[num]}%']
        r_df = pd.concat([r_df, tmp2_df])
    return r_df


def gini(x):
    if np.mean(x) <= 0: return 0
    mad = np.abs(np.subtract.outer(x, x)).mean()
    return 0.5 * mad / np.mean(x)


# --- 4. Network Construction & Data Augmentation ---
def split_into_parts(a_data, test_part, val_part, embed_file):
    def get_xy(df):
        if df.empty: return np.array([]), np.array([])
        x = data_augmentation(df.SEQUENCE.to_numpy(), embed_file)
        y = df[['label']].values
        return x, y

    x_test, y_test = get_xy(a_data[a_data.partition == test_part])
    x_val, y_val = get_xy(a_data[a_data.partition == val_part])
    x_train, y_train = get_xy(a_data[~a_data.partition.isin([test_part, val_part])])
    return x_test, y_test, x_val, y_val, x_train, y_train


def create_AnOxPePred_v1(hps):
    y_out = hps.get('y_out', 1)

    class AnOxPePred_v1_Model(Model):
        def __init__(self, output_dim):
            super(AnOxPePred_v1_Model, self).__init__()
            self.conv = Conv1D(128, 3, activation='elu', padding='same')
            self.pool = AveragePooling1D(3, 3)
            self.dropout = Dropout(0.2)
            self.flat = Flatten()
            self.d1 = Dense(256, activation='elu')
            self.d2 = Dense(output_dim, activation='sigmoid')

        def call(self, x):
            x = self.dropout(self.pool(self.conv(x)))
            x = self.flat(x)
            x = self.dropout(self.d1(x))
            return self.d2(x)

    model = AnOxPePred_v1_Model(y_out)
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(3e-5), metrics=['accuracy'])
    return model


def data_augmentation(a_array, embed_file):
    e_dic = {}
    with open(embed_file, 'r') as f:
        for line in f:
            if not line.startswith('#'):
                parts = line.split()
                if len(parts) > 1: e_dic[parts[0]] = [float(i) for i in parts[1:]]

    def pad(seq, length=30):
        num_x = length - len(seq)
        if num_x < 0: return seq[:length]
        return ('X' * int(np.ceil(num_x / 2))) + seq + ('X' * int(np.floor(num_x / 2)))

    x_gen = [pad(str(s)) for s in a_array]
    embed_dim = len(next(iter(e_dic.values())))
    return np.array([np.array([e_dic.get(j, [0.0] * embed_dim) for j in i]) for i in x_gen])


def get_fsa_file(fasta_file):
    peptides = []
    with open(fasta_file, 'r') as f:
        name, seq = None, []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if name: peptides.append([name, "".join(seq)])
                name, seq = line[1:], []
            else:
                seq.append(line)
        if name: peptides.append([name, "".join(seq)])
    return peptides


# --- 5. Metrics Calculation & Evaluation ---
def calc_metrics(y_label, y_pred, idx='metrics'):
    y_true, y_prob = y_label.flatten(), y_pred.flatten()
    thresholds = np.arange(0.1, 0.9, 0.05)

    try:
        mccs = [matthews_corrcoef(y_true, (y_prob >= t).astype(int)) for t in thresholds]
        opt_t = thresholds[np.argmax(mccs)]
    except:
        opt_t = 0.5

    def get_row(t):
        y_bin = (y_prob >= t).astype(int)
        try:
            auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
            f1 = f1_score(y_true, y_bin, zero_division=0)
            mcc = matthews_corrcoef(y_true, y_bin)
        except:
            auc, f1, mcc = 0.5, 0.0, 0.0
        return [auc, f1, mcc]

    res = pd.DataFrame([get_row(0.5) + get_row(opt_t)],
                       columns=['0.5_AUC', '0.5_F1', '0.5_MCC', 'custom_AUC', 'custom_F1', 'custom_MCC'],
                       index=[idx])
    res['Threshold'] = opt_t
    return res


def output_df(x_data, y_label, y_pred, tag, sequences=None):
    """
    Create a DataFrame containing prediction results.

    Parameters:
    -----------
    x_data : numpy array
        Input data (embedded representation)
    y_label : numpy array
        True labels
    y_pred : numpy array
        Predicted probabilities
    tag : str
        Tag identifier for the prediction batch
    sequences : array-like, optional
        List of original sequence strings to retain sequence information

    Returns:
    --------
    pd.DataFrame
        DataFrame containing prediction probabilities, labels, lengths, and sequences.
    """
    df_pred = pd.DataFrame(y_pred.flatten(), columns=['Pred_Prob'])
    df_label = pd.DataFrame(y_label.flatten(), columns=['label'])

    lengths = [int(np.sum(np.any(sample != 0, axis=1))) for sample in x_data]
    df_length = pd.DataFrame(lengths, columns=['Length'])

    result_df = pd.concat([df_pred, df_label, df_length], axis=1).reset_index(drop=True)
    result_df['tag'] = tag

    if sequences is not None:
        df_seq = pd.DataFrame(list(sequences), columns=['SEQUENCE'])
        result_df = pd.concat([result_df, df_seq], axis=1).reset_index(drop=True)

    return result_df


def length_metrics(pred_df, bins, x_labels=None):
    if 'Length' not in pred_df.columns:
        return pd.DataFrame()

    results = []
    for i, b in enumerate(bins):
        mask = pred_df['Length'].apply(lambda l: l in b if isinstance(b, (range, list)) else l == b)
        group = pred_df[mask]

        if not group.empty and len(np.unique(group['label'])) > 1:
            m = calc_metrics(group['label'].values, group['Pred_Prob'].values, idx=x_labels[i] if x_labels else i)
            results.append(m)

    return pd.concat(results) if results else pd.DataFrame()


# --- 6. k-NN Prediction ---
def kNN_pred(row, train_df, k=5):
    test_seq = row['Sequence']
    similarities = train_df['Sequence'].apply(lambda x: calc_ident(test_seq, x))
    neighbor_indices = similarities.sort_values(ascending=False).index[1:k + 1]
    neighbors = train_df.loc[neighbor_indices]
    res_label = neighbors['label'].mean()
    return pd.Series([res_label, 0.0])