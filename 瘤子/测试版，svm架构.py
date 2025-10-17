# ml_pipeline_robust.py
# -*- coding: utf-8 -*-
import os, sys, glob, json, subprocess, tempfile, uuid, gc
import numpy as np
import pandas as pd

# ========== 你的路径 ==========
IMAGE_DIR = r"C:\Users\zwq\Desktop\DS\image"
MASK_DIR  = r"C:\Users\zwq\Desktop\DS\nodule_mask"
OUT_DIR   = r"C:\Users\zwq\Desktop\DS\ml_out"
os.makedirs(OUT_DIR, exist_ok=True)

FEATURE_CSV       = os.path.join(OUT_DIR, "features.csv")
FAILED_LIST_TXT   = os.path.join(OUT_DIR, "failed_cases.txt")
FEATURE_WITH_OOF  = os.path.join(OUT_DIR, "features_with_oof.csv")
MODEL_PKL         = os.path.join(OUT_DIR, "svm_calibrated_pipeline.pkl")

# ========== SVM 配置 ==========
SIZE_THRESH_MM = 6.0    # 等效直径 >= 6mm 作为正类（若你有真实标签，换掉函数 make_label）
N_SPLITS = 5
USE_PCA  = True
PCA_VAR  = 0.95

# ========== 环境稳健性设置 ==========
# 限制 ITK 多线程，避免底层并发崩（有时有用）
os.environ.setdefault("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "1")

# ================= 工具 =================
def yprint(*a):
    print(*a, flush=True)

def list_cases():
    nii_list = sorted([p for p in glob.glob(os.path.join(IMAGE_DIR, "*.nii*"))])
    items = []
    for ip in nii_list:
        fn = os.path.basename(ip)
        mp = os.path.join(MASK_DIR, fn)
        if os.path.exists(mp):
            items.append((fn, ip, mp))
    return items

def load_existing_cases():
    if not os.path.exists(FEATURE_CSV): return set()
    try:
        df = pd.read_csv(FEATURE_CSV)
        return set(df["meta_case"].astype(str).tolist())
    except Exception:
        return set()

# ========== 子进程工作模式（单病例提特征） ==========
def worker_extract_one(image_path, mask_path, out_json_path):
    # 单病例提取：只在子进程里运行，崩了也不影响父进程
    import SimpleITK as sitk
    from radiomics import featureextractor, setVerbosity
    setVerbosity(40)

    def read_itk(path):
        img = sitk.ReadImage(path)
        arr = sitk.GetArrayFromImage(img)  # (z,y,x)
        arr = np.transpose(arr, (2,1,0))   # (x,y,z)
        return arr, img

    def eq_diameter_from_mask(mask_bool, spacing_xyz):
        vx = float(spacing_xyz[0]*spacing_xyz[1]*spacing_xyz[2])
        vol_mm3 = mask_bool.sum() * vx
        if vol_mm3 <= 0: return 0.0
        return float((6.0*vol_mm3/np.pi) ** (1.0/3.0))

    # 基础 sanity check，尽量避免触发底层崩溃
    arr_img, itk_img = read_itk(image_path)
    arr_msk, itk_msk = read_itk(mask_path)
    if arr_img.shape != arr_msk.shape:
        raise RuntimeError(f"shape mismatch: {arr_img.shape} vs {arr_msk.shape}")
    # 掩膜至少有一定体素数
    mask_bool = arr_msk > 0
    if mask_bool.sum() < 5:
        raise RuntimeError("mask too small (<5 voxels)")
    # spacing 全为正
    sp = itk_img.GetSpacing()
    if any([s <= 0 for s in sp]):
        raise RuntimeError(f"invalid spacing: {sp}")

    # radiomics 配置：只开 Original；不开奇怪的设置；避免 YAML
    params = {
        "imageType": {"Original": {}},
        "setting": {
            "resampledPixelSpacing": [1.0, 1.0, 1.0],
            "interpolator": "sitkBSpline",
            "binWidth": 25,
            "normalize": False,
            "force2D": False,
            "correctMask": True
        }
    }
    extractor = featureextractor.RadiomicsFeatureExtractor(params)
    extractor.enableAllFeatures()

    fv = extractor.execute(itk_img, itk_msk)
    feat = {k: float(v) for k, v in fv.items() if k.startswith("original_") and np.isfinite(v)}

    eqd = eq_diameter_from_mask(mask_bool, sp)
    feat["meta_case"] = os.path.basename(image_path)
    feat["meta_voxels"] = int(mask_bool.sum())
    feat["meta_eq_diameter_mm"] = float(eqd)
    feat["meta_sx"] = float(sp[0]); feat["meta_sy"] = float(sp[1]); feat["meta_sz"] = float(sp[2])

    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(feat, f, ensure_ascii=False)

# ========== 父进程主流程 ==========
def run_parent():
    yprint("[INFO] scanning cases ...")
    items = list_cases()
    already = load_existing_cases()
    yprint(f"[INFO] total pairs: {len(items)}, already done: {len(already)}")

    # 若已存在 features.csv，先读入，后续增量追加
    if os.path.exists(FEATURE_CSV):
        df_all = pd.read_csv(FEATURE_CSV)
    else:
        df_all = pd.DataFrame()

    failed = []

    for idx, (fname, ip, mp) in enumerate(items, 1):
        if fname in already:
            yprint(f"[SKIP] {idx}/{len(items)} {fname} (exists)")
            continue

        yprint(f"[DO]   {idx}/{len(items)} {fname}")
        tmp_json = os.path.join(OUT_DIR, f"_tmpfeat_{uuid.uuid4().hex}.json")

        # 调用“自己”作为子进程（--worker 模式）
        cmd = [sys.executable, __file__, "--worker", ip, mp, tmp_json]
        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=600)
            if res.returncode != 0:
                yprint(f"[FAIL] {fname} rc={res.returncode}")
                yprint(res.stderr[-4000:])
                failed.append(fname)
            else:
                # 读取 JSON 结果并追加到 CSV
                with open(tmp_json, "r", encoding="utf-8") as f:
                    feat = json.load(f)
                df_row = pd.DataFrame([feat])
                if df_all.empty:
                    df_all = df_row
                else:
                    df_all = pd.concat([df_all, df_row], ignore_index=True)
                df_all.to_csv(FEATURE_CSV, index=False, encoding="utf-8-sig")
                yprint(f"[OK]   {fname} -> features appended (total={len(df_all)})")
        except subprocess.TimeoutExpired:
            yprint(f"[TIMEOUT] {fname}")
            failed.append(fname)
        except Exception as e:
            yprint(f"[EXC] {fname}: {e}")
            failed.append(fname)
        finally:
            try:
                if os.path.exists(tmp_json): os.remove(tmp_json)
            except Exception:
                pass
        # 主动清理
        gc.collect()

    # 记录失败列表
    if failed:
        with open(FAILED_LIST_TXT, "w", encoding="utf-8") as f:
            f.write("\n".join(failed))
        yprint(f"[WARN] failed cases: {len(failed)} -> {FAILED_LIST_TXT}")
    else:
        yprint("[INFO] no failed cases 🎉")

    # ========== SVM 训练 ==========
    if not os.path.exists(FEATURE_CSV):
        yprint("[ERR] features.csv not found, abort SVM.")
        return

    df = pd.read_csv(FEATURE_CSV)
    if len(df) < 10:
        yprint("[ERR] too few successful cases for training.")
        return

    # ---- 训练并评估 ----
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVC
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, classification_report, confusion_matrix
    import joblib

    def make_label(_df: pd.DataFrame):
        return (_df["meta_eq_diameter_mm"].values >= SIZE_THRESH_MM).astype(int)

    feat_cols = [c for c in df.columns if not c.startswith("meta_")]
    X = df[feat_cols].values
    y = make_label(df)
    yprint(f"[INFO] SVM data: X={X.shape}, pos={(y==1).sum()}, neg={(y==0).sum()}")

    steps = [("scaler", StandardScaler())]
    if USE_PCA:
        steps.append(("pca", PCA(n_components=PCA_VAR, svd_solver="full")))
    steps.append(("svc", SVC(kernel="rbf", C=1.0, gamma="scale",
                             class_weight="balanced", probability=True)))
    base = Pipeline(steps)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    oof_pred = np.zeros(len(y), dtype=float)
    oof_lbl  = y.copy()

    best_auc, best_model = -1, None
    for k, (tr, va) in enumerate(skf.split(X, y), 1):
        Xtr, Xva = X[tr], X[va]
        ytr, yva = y[tr], y[va]
        base.fit(Xtr, ytr)
        clf = CalibratedClassifierCV(base, method="sigmoid", cv=3)
        clf.fit(Xtr, ytr)

        proba = clf.predict_proba(Xva)[:, 1]
        pred  = (proba >= 0.5).astype(int)

        auc  = roc_auc_score(yva, proba)
        ap   = average_precision_score(yva, proba)
        f1   = f1_score(yva, pred)
        yprint(f"[FOLD {k}] AUC={auc:.3f}  AP={ap:.3f}  F1={f1:.3f}")

        oof_pred[va] = proba
        if auc > best_auc:
            best_auc, best_model = auc, clf

    auc = roc_auc_score(oof_lbl, oof_pred)
    ap  = average_precision_score(oof_lbl, oof_pred)
    pred = (oof_pred >= 0.5).astype(int)
    f1   = f1_score(oof_lbl, pred)
    yprint("\n[OOF] AUC={:.3f}  AP={:.3f}  F1={:.3f}".format(auc, ap, f1))
    yprint("[OOF] Confusion:\n" + str(confusion_matrix(oof_lbl, pred)))
    yprint("[OOF] Report:\n" + classification_report(oof_lbl, pred, digits=3))

    out = df.copy()
    out["y"] = oof_lbl
    out["proba"] = oof_pred
    out["pred"]  = pred
    out.to_csv(FEATURE_WITH_OOF, index=False, encoding="utf-8-sig")
    yprint("[DONE] saved:", FEATURE_WITH_OOF)

    if best_model is not None:
        import joblib
        joblib.dump(best_model, MODEL_PKL)
        yprint("[DONE] model saved:", MODEL_PKL)

# ========== 入口 ==========
if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--worker":
        # 子进程模式：--worker <image_path> <mask_path> <out_json>
        _, _, ip, mp, oj = sys.argv
        try:
            worker_extract_one(ip, mp, oj)
            sys.exit(0)
        except Exception as e:
            # 子进程失败时打印错误并返回非零，父进程会记录
            print(str(e), file=sys.stderr, flush=True)
            sys.exit(1)
    else:
        run_parent()
