# coding: utf-8
import numpy as np
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from importlib import reload
import AnOxPePred_funcs as AOf

# --------------------------
# SCI Figure Style Configuration
# --------------------------
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['xtick.major.width'] = 0.8
plt.rcParams['ytick.major.width'] = 0.8
# Vector graphics: embed fonts
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
# High resolution for line/bar charts, default save format is JPG
plt.rcParams['figure.dpi'] = 1000
plt.rcParams['savefig.dpi'] = 1000
plt.rcParams['savefig.format'] = 'jpg'
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['legend.frameon'] = False

# Color scheme (Differentiating positive/negative samples)
colors = {'positive': '#D62728', 'negative': '#1F77B4', 'reduction': '#2CA02C', 'partition': '#FF7F0E'}

# --- 1. Environment & Paths ---
# Use relative paths for GitHub repository standard
base_data = './Data'
base_result = './Results/Result_AnOxPePred'
base_figure = './Results/Homology_Reduction'

# Create output subdirectories (v2) to avoid overwriting previous results
result_path = os.path.join(base_result, "v2_output")
figure_path = os.path.join(base_figure, "v2_figures")
onehot_path = os.path.join(base_data, 'One-hot_encoding.txt')

os.makedirs(result_path, exist_ok=True)
os.makedirs(figure_path, exist_ok=True)

print("=" * 60)
print("Starting binary classification data processing pipeline (V2)")
print("=" * 60)

# 2. Load complete dataset
full_df = pd.read_csv(os.path.join(base_data, 'combined_data.csv'), index_col=0)
full_df = full_df[['SEQUENCE', 'label']].reset_index(drop=True)
full_df = full_df[full_df['SEQUENCE'].apply(len) <= 30]
full_df['seq_length'] = full_df['SEQUENCE'].apply(len)

print("\n[Step 1] Initial dataset statistics:")
data_stats = AOf.visualize_data(full_df, 'Initial_Full_Data')
print(data_stats)

pos_count = full_df[full_df['label'] == 1].shape[0]
neg_count = full_df[full_df['label'] == 0].shape[0]
total_count = full_df.shape[0]

print(f"  Positive samples: {pos_count}")
print(f"  Negative samples: {neg_count}")
print(f"  Total samples: {total_count}")

# --------------------------
# Figure 1: Initial Data Distribution
# --------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.bar(['Positive (1)', 'Negative (0)'], [pos_count, neg_count],
        color=[colors['positive'], colors['negative']], width=0.6)
ax1.set_ylabel('Number of Samples', fontsize=10)
ax1.set_title('Sample Distribution', fontsize=11, fontweight='bold')
for i, v in enumerate([pos_count, neg_count]):
    ax1.text(i, v + 50, f'{v}', ha='center', va='bottom', fontsize=9)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

pos_lengths = full_df[full_df['label'] == 1]['seq_length']
neg_lengths = full_df[full_df['label'] == 0]['seq_length']
bp = ax2.boxplot([pos_lengths, neg_lengths], labels=['Positive', 'Negative'],
                 patch_artist=True, widths=0.6)
for patch, color in zip(bp['boxes'], [colors['positive'], colors['negative']]):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax2.set_ylabel('Sequence Length', fontsize=10)
ax2.set_title('Sequence Length Distribution', fontsize=11, fontweight='bold')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

fig.suptitle('Initial Dataset Statistics', fontsize=12, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(figure_path, 'V2_Figure1_Initial_Data_Distribution.jpg'))
plt.close()

# 3. Sequence-identity reduction
print("\nExecuting sequence-identity reduction (90% threshold) ...")
reduced_df = AOf.homology_reduction(full_df, 0.9)
reduced_df['seq_length'] = reduced_df['SEQUENCE'].apply(len)

# Save reduced dataset to v2 folder
reduced_df.to_csv(os.path.join(result_path, '02_processed_full_db_AnOxPePred_v2.csv'))

print("\n[Step 2] Dataset statistics after sequence-identity reduction:")
reduced_stats = AOf.visualize_data(reduced_df, 'Post_Identity_Reduction')
print(reduced_stats)

red_pos_count = reduced_df[reduced_df['label'] == 1].shape[0]
red_neg_count = reduced_df[reduced_df['label'] == 0].shape[0]
red_total_count = reduced_df.shape[0]

# --------------------------
# Figure 2: Sequence-Identity Reduction Comparison
# --------------------------
fig, ax = plt.subplots(1, 1, figsize=(8, 5))
categories = ['Positive', 'Negative', 'Total']
before = [pos_count, neg_count, total_count]
after = [red_pos_count, red_neg_count, red_total_count]
x = np.arange(len(categories))
width = 0.35

ax.bar(x - width / 2, before, width, label='Before Reduction', color=colors['negative'], alpha=0.8)
ax.bar(x + width / 2, after, width, label='After Reduction (90% sequence identity)', color=colors['reduction'],
       alpha=0.8)
ax.set_ylabel('Number of Samples', fontsize=10)
ax.set_title('Sample Count Before/After Sequence-Identity Reduction', fontsize=11, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend(fontsize=9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

for i, (b, a) in enumerate(zip(before, after)):
    ax.text(i - width / 2, b + 50, f'{b}', ha='center', va='bottom', fontsize=8)
    ax.text(i + width / 2, a + 50, f'{a}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
fig.savefig(os.path.join(figure_path, 'V2_Figure2_Identity_Reduction_Comparison.jpg'))
plt.close()

# ======================================================
# Core: Sequence-Identity Partitioning & Cluster-size Gini Calculation
# ======================================================
print("\nPerforming 5-Fold sequence-identity partitioning ...")
hom_parts = []
gr_nrs = []
all_cluster_sizes_list = []
thresholds = [0.6, 0.7, 0.8, 0.9]
threshold_labels = ['60%', '70%', '80%', '90%']

for idt in thresholds:
    print(f"  Processing {int(idt * 100)}% sequence-identity threshold ...")
    pos_seqs = reduced_df[reduced_df['label'] == 1]['SEQUENCE']
    neg_seqs = reduced_df[reduced_df['label'] == 0]['SEQUENCE']

    # Returns partition, cluster count, and indices list per cluster
    pos_part, pos_cluster_cnt, pos_c_list = AOf.homology_partition(pos_seqs, ident=idt, parts=5)
    neg_part, neg_cluster_cnt, neg_c_list = AOf.homology_partition(neg_seqs, ident=idt, parts=5)

    merged_part = []
    for fold in range(5):
        pos_idx = reduced_df[reduced_df['label'] == 1].iloc[pos_part[fold]].index
        neg_idx = reduced_df[reduced_df['label'] == 0].iloc[neg_part[fold]].index
        merged_part.append(list(pos_idx) + list(neg_idx))

    hom_parts.append(merged_part)
    gr_nrs.append(pos_cluster_cnt + neg_cluster_cnt)

    # Collect cluster sample sizes for Gini calculation
    pos_sizes = [len(cl) for cl in pos_c_list]
    neg_sizes = [len(cl) for cl in neg_c_list]
    all_sizes = np.array(pos_sizes + neg_sizes)
    all_cluster_sizes_list.append(all_sizes)

# Calculate Gini coefficient based on cluster sizes
gini_values = [AOf.gini(sizes) for sizes in all_cluster_sizes_list]

# Save cluster sizes in long format
long_records = []
for lab, sizes_arr in zip(threshold_labels, all_cluster_sizes_list):
    for sz in sizes_arr:
        long_records.append({"sequence_identity_threshold": lab, "cluster_sample_size": int(sz)})
cluster_size_df = pd.DataFrame(long_records)
cluster_size_df.to_csv(os.path.join(result_path, "03_cluster_sizes_v2.csv"), index=False)

Data_info = pd.DataFrame(
    [gr_nrs, gini_values],
    columns=['AO_p60', 'AO_p70', 'AO_p80', 'AO_p90'],
    index=['Clusters', 'Gini (cluster-size)']
)

print("\n[Step 4] Partition quality report (Data_info_v2):")
print(Data_info)
Data_info.to_csv(os.path.join(result_path, '03_Data_Info_AnOxPePred_v2.csv'))

# --------------------------
# Figure 3: Partition Quality Analysis
# --------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

ax1.plot(threshold_labels, gr_nrs, marker='o', linewidth=2,
         color=colors['partition'], markersize=6)
ax1.set_xlabel('Sequence-Identity Threshold', fontsize=10)
ax1.set_ylabel('Number of Clusters', fontsize=10)
ax1.set_title('Cluster Count by Sequence-Identity Threshold', fontsize=11, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

ax2.bar(threshold_labels, gini_values, color=colors['partition'], alpha=0.8, width=0.6)
ax2.set_xlabel('Sequence-Identity Threshold', fontsize=10)
ax2.set_ylabel('Gini Coefficient', fontsize=10)
ax2.set_title('Cluster-Size Distribution (Gini Coefficient)', fontsize=11, fontweight='bold')
for i, v in enumerate(gini_values):
    offset = 0.01 if v > 0.01 else 0.002
    ax2.text(i, v + offset, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.set_ylim(0, max(gini_values) * 1.2)

fig.suptitle('Partition Quality Analysis', fontsize=12, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(figure_path, 'V2_Figure3_Partition_Quality.jpg'))
plt.close()

# --------------------------
# Figure 4: Partition Sample Distribution Heatmap (90% Threshold)
# --------------------------
partition_90 = hom_parts[-1]
partition_df = pd.DataFrame({'partition': -1}, index=reduced_df.index)

for p_idx, indices in enumerate(partition_90):
    partition_df.loc[indices, 'partition'] = p_idx
reduced_df['partition'] = partition_df['partition']

partition_stats = pd.crosstab(reduced_df['partition'], reduced_df['label'])
partition_stats.columns = ['Negative (0)', 'Positive (1)']

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
sns.heatmap(partition_stats, annot=True, fmt='d', cmap='RdBu_r', ax=ax,
            cbar_kws={'label': 'Number of Samples'}, linewidths=0.5)
ax.set_xlabel('Sample Class', fontsize=10)
ax.set_ylabel('5-Fold Partition Index', fontsize=10)
ax.set_title('Sample Distribution in 5-Fold Partitions (90% Sequence-Identity Threshold)',
             fontsize=11, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(figure_path, 'V2_Figure4_Partition_Distribution_Heatmap.jpg'))
plt.close()

# 6. Save partition CSVs for all thresholds
print("\nSaving partitioned training data ...")
for idt, part in zip([60, 70, 80, 90], hom_parts):
    train_df = reduced_df.copy()
    train_df['partition'] = -1
    for p_idx, indices in enumerate(part):
        train_df.loc[indices, 'partition'] = p_idx
    train_df.to_csv(os.path.join(result_path, f'03_p{idt}_full_db_AnOxPePred_v2.csv'))
    print(f"  Saved {idt}% threshold partition file: 03_p{idt}_full_db_AnOxPePred_v2.csv")

# Save fold split indices to JSON for reproducibility
with open(os.path.join(result_path, "fold_split_indices_v2.json"), "w", encoding="utf-8") as f:
    json.dump({
        "thresholds": [0.6, 0.7, 0.8, 0.9],
        "partitions": hom_parts
    }, f, indent=2)

print("\n" + "=" * 60)
print("V2 processing completed successfully!")
print(f"Figure output directory: {figure_path}")
print(f"Table/JSON output directory: {result_path}")
print("Output list:")
print("  1. V2_*.jpg (SCI-formatted figures)")
print("  2. 03_Data_Info_AnOxPePred_v2.csv (Cluster counts & Gini coefficients)")
print("  3. 03_cluster_sizes_v2.csv (Long format cluster sample sizes)")
print("  4. fold_split_indices_v2.json (Fold split indices for GitHub reproducibility)")
print("  5. 03_pXX_full_db_AnOxPePred_v2.csv (Partitioned datasets per threshold)")
print("=" * 60)