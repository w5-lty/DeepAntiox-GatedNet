# DeepAntiox‑GatedNet: An Interpretable Multimodal Deep Learning Framework with an Adaptive Gating Mechanism for Antioxidant Peptide Prediction
# Description
We presented DeepAntiox‑GatedNet, a novel interpretable multimodal dual‑branch deep learning framework for antioxidant peptide (AOP) prediction. We designed three customized modules: PartialConv1D multiscale convolution, dual‑path attention (DPA), and adaptive gated multimodal fusion (AGMF). The sequence branch extracts local multiscale peptide patterns from one‑hot encoded sequences, while the physicochemical branch leverages an optimized feature subset obtained via a three‑stage feature‑selection pipeline. The AGMF module dynamically balances contributions from sequence embeddings and hand‑crafted physicochemical features to mitigate multimodal feature‑fusion conflicts. Comprehensive evaluations under multiple sequence‑identity thresholds (P60‑P90) demonstrated that our model achieves competitive performance compared with conventional machine learning algorithms and state‑of‑the‑art deep‑learning predictors. Furthermore, we adopted strict cluster‑based nested 5‑fold cross‑validation to avoid data leakage. SHAP interpretability analysis was applied to reveal core predictive features. We further implemented a PyQt5 graphical platform for high‑throughput peptide screening and performed in‑vitro DPPH radical‑scavenging assays for experimental validation.
# Dataset
We collected three public benchmark datasets for antioxidant peptide prediction: the AnOxPePred dataset from Olson et al., the AnOxPP dataset from Qin et al., and the AOPP dataset from Li et al.
The AnOxPePred dataset was curated in 2020 from the BIOPEP‑UWM database, containing 676 experimentally verified antioxidant positive samples and 728 negative samples. The AnOxPP dataset was compiled in 2023 based on DFBP and BIOPEP‑UWM databases with 1060 positive and 1060 negative peptide entries. The AOPP dataset is a comprehensive resource integrating DFBP, BIOPEP‑UWM, APD, PlantPepDB and FermFooDb, consisting of 1511 positive AOP samples and 1511 negative controls.
After merging three sources, we filtered sequences with non‑standard amino acids and restricted peptide length within 2‑30 residues. CD‑HIT was utilized for redundancy removal under 90 % sequence identity threshold. The consolidated non‑redundant dataset contains 3379 peptide entries for nested cross‑validation. An independent literature‑sourced external test set containing 37 experimentally validated positive AOPs was used for additional generalization assessment.

  - Raw merged dataset: /data/raw/
  - Processed non‑redundant dataset (CD‑HIT 90%): /data/processed/
  - Cluster‑based partition files for P60/P70/P80/P90 cross‑validation are provided in /data/cluster_split/
  - Script for three‑stage feature selection (F‑test, Pearson filtering, SFS): /feature_engineer/feature_selection.py
  - Script to reproduce machine‑learning baseline comparison: /reproduce/ml_baseline/aop_ml_compare.py
  - Script to reproduce deep‑learning baseline comparison & ablation study: /reproduce/dl_baseline/ablation_run.py
  - Script for full nested‑cross‑validation training of DeepAntiox‑GatedNet: /final_model/deepantiox_gatednet_train.py

  # Getting Started

  # Python packages
  - python==3.10
  - tensorflow==2.15.0
  - pyqt5==5.15.10
  - shap==0.44.0
  - scikit‑learn==1.3.2
  - pandas==2.1.4
  - numpy==1.26.2
  - cd‑hit (external tool, add to system PATH)

  # Executing program
  - run /reproduce/dl_baseline/ablation_run.py to reproduce ablation experiment results
  - run /reproduce/ml_baseline/aop_ml_compare.py to reproduce traditional machine‑learning baseline results
  - run /final_model/deepantiox_gatednet_train.py to train DeepAntiox‑GatedNet under cluster‑based nested 5‑fold cross‑validation
  - run /interpretability/shap_analysis.py to perform SHAP interpretability analysis
  - run /gui/run_gui.py to launch PyQt5 graphical prediction platform for single/batch peptide prediction

  # Acknowledgments
  We thank the authors of AnOxPePred, AnOxPP and AOPP for sharing their public datasets. We also acknowledge open‑source resources including CD‑HIT, SHAP and TensorFlow.
*Hu J, Shen L, Sun G. Squeeze‑and‑Excitation Networks. Proceedings of the IEEE conference on computer vision and pattern recognition. 2018: 7132‑7141.*
*Chen J, Yang Z, Zhang L, et al. Run, Don’t Walk: Chasing Higher FLOPS for Faster Neural Networks. CVPR, 2023.*
