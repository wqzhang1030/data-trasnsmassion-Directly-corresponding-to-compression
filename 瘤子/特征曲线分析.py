# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import (
    roc_curve, roc_auc_score, precision_recall_curve, average_precision_score,
    f1_score, accuracy_score, classification_report, confusion_matrix
)
from sklearn.inspection import permutation_importance

# ===== 路径 =====
OUT_DIR = r"C:\Users\zwq\Desktop\DS\ml_out"
CSV_OOF = os.path.join(OUT_DIR, "features_with_oof.csv")   # 上一步生成的
CSV_FEAT= os.path.join(OUT_DIR, "features.csv")            # 原始特征
MODEL_PKL = os.path.join(OUT_DIR, "svm_calibrated_pipeline.pkl")
SUMMARY_XLSX = os.path.join(OUT_DIR, "results_summary.xlsx")

os.makedirs(OUT_DIR, exist_ok=True)

# ===== 1) 读取 OOF 结果，画 ROC / PR =====
df_oof = pd.read_csv(CSV_OOF)
y_true = df_oof["y"].values
y_score = df_oof["proba"].values
y_pred  = df_oof["pred"].values

# ROC
fpr, tpr, _ = roc_curve(y_true, y_score)
auc = roc_auc_score(y_true, y_score)

plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.3f}")
plt.plot([0,1],[0,1],"k--", lw=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Radiomics + SVM)")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "ROC_curve.png"), dpi=300)
plt.close()

# PR
prec, rec, _ = precision_recall_curve(y_true, y_score)
ap = average_precision_score(y_true, y_score)
plt.figure(figsize=(6,6))
plt.plot(rec, prec, lw=2, label=f"AP = {ap:.3f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend(loc="lower left")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "PR_curve.png"), dpi=300)
plt.close()

# ===== 2) 置换重要性（对原始特征做） =====
# 读取原始特征表
df_feat = pd.read_csv(CSV_FEAT)
feat_cols = [c for c in df_feat.columns if not c.startswith("meta_")]
X = df_feat[feat_cols].values

# 用“等效直径>=6mm”的规则生成同样的标签（保证与训练一致）
y = (df_feat["meta_eq_diameter_mm"].values >= 6.0).astype(int)

# 加载已校准模型（里头自带：标准化/PCA/SVM）
model = joblib.load(MODEL_PKL)

# 某些 sklearn 版本 CalibratedClassifierCV 的底座访问方式不同，这里不用去拿底座，
# 直接对整个 model 做 permutation importance（它会把 X 经过 pipeline 再进分类器）
print("[INFO] computing permutation importance (this may take a bit)...")
pi = permutation_importance(
    model, X, y,
    scoring="roc_auc",
    n_repeats=5,
    random_state=42,
    n_jobs=1   # 保守起见，避免多线程在某些环境里抽风
)

importances = pi.importances_mean
stds = pi.importances_std
idx = np.argsort(importances)[-20:][::-1]  # 取 Top-20

plt.figure(figsize=(8,6))
plt.barh(np.array(feat_cols)[idx][::-1], importances[idx][::-1], xerr=stds[idx][::-1], alpha=0.9)
plt.xlabel("Permutation Importance (mean ΔAUC)")
plt.title("Top-20 Radiomics Features")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "Top20_features_permutation.png"), dpi=300)
plt.close()

# ===== 3) 汇总表 =====
acc = accuracy_score(y_true, y_pred)
f1  = f1_score(y_true, y_pred)
cm  = confusion_matrix(y_true, y_pred)

summary = pd.DataFrame({
    "Metric": ["AUC", "Average Precision", "Accuracy", "F1-score"],
    "Value": [auc, ap, acc, f1]
})
# 也附上混淆矩阵单元
cm_df = pd.DataFrame(cm, index=["True 0","True 1"], columns=["Pred 0","Pred 1"])

with pd.ExcelWriter(SUMMARY_XLSX) as w:
    summary.to_excel(w, sheet_name="summary", index=False)
    pd.DataFrame({"feature": np.array(feat_cols)[idx],
                  "importance": importances[idx],
                  "std": stds[idx]}).to_excel(w, sheet_name="top_features", index=False)
    cm_df.to_excel(w, sheet_name="confusion_matrix")

print(f"[DONE] 已输出到：{OUT_DIR}")
print(" - ROC_curve.png")
print(" - PR_curve.png")
print(" - Top20_features_permutation.png")
print(" - results_summary.xlsx")
