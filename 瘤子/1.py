# lung_nodule_pipeline.py
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import SimpleITK as sitk
from scipy.ndimage import (
    gaussian_filter,
    binary_opening, binary_closing, binary_fill_holes, binary_erosion,
    distance_transform_edt as bwdist,
)
from skimage.filters import threshold_otsu


# ========= 1) 输入路径 =========
IMAGE_PATH = r"C:\Users\zwq\Desktop\DS\image\LIDC-IDRI-0003_R_3.nii.gz"
MASK_PATH  = r"C:\Users\zwq\Desktop\DS\nodule_mask\LIDC-IDRI-0003_R_3.nii.gz"

# ========= 2) 输出目录（纯英文+有写权限） =========
OUT_DIR = r"C:\Users\zwq\LIDC-IDRI-0003_R_3"
os.makedirs(OUT_DIR, exist_ok=True)


# ========= 工具函数 =========
def sitk_read_xyz(path):
    img = sitk.ReadImage(path)
    arr_zyx = sitk.GetArrayFromImage(img)       # (z,y,x)
    arr_xyz = np.transpose(arr_zyx, (2, 1, 0))  # (x,y,z)
    meta = {"spacing": img.GetSpacing(), "origin": img.GetOrigin(), "direction": img.GetDirection()}
    return arr_xyz, meta


def window_ct_hu(vol, lo=-1000, hi=400):
    v = vol.astype(np.float32)
    v = np.clip(v, lo, hi)
    v = (v - lo) / float(hi - lo + 1e-6)
    return v


def get_middle_slice_index(vol_xyz):
    return vol_xyz.shape[2] // 2


def show_and_save_overlay(img2d, mask2d, seg2d, outpath, title="overlay"):
    plt.figure(figsize=(6,6))
    plt.imshow(img2d, cmap='gray')
    if mask2d is not None:
        plt.contour(mask2d.astype(float), levels=[0.5], colors='r', linewidths=1.0)
    if seg2d is not None:
        plt.contour(seg2d.astype(float), levels=[0.5], colors='g', linewidths=1.0)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def safe_write_excel(path, df):
    try:
        with pd.ExcelWriter(path, engine='xlsxwriter') as w:
            df.to_excel(w, index=False)
    except Exception:
        df.to_csv(path.replace(".xlsx", ".csv"), index=False, encoding="utf-8-sig")


# ========= 评估指标 =========
def dice(seg, gt):
    seg = seg.astype(bool); gt = gt.astype(bool)
    inter = np.sum(seg & gt)
    denom = np.sum(seg) + np.sum(gt)
    return (2.0 * inter / denom) if denom > 0 else 0.0


def voe(seg, gt):
    seg = seg.astype(bool); gt = gt.astype(bool)
    inter = np.sum(seg & gt)
    denom = np.sum(seg) + np.sum(gt)
    return (1 - 2*inter/denom) if denom > 0 else 1.0


def rel_vol_diff(seg, gt):
    seg = seg.astype(bool); gt = gt.astype(bool)
    s = float(np.sum(seg)); g = float(np.sum(gt))
    return (s - g) / s if s > 0 else np.inf


def surface_distances(seg, gt):
    seg = seg.astype(bool); gt = gt.astype(bool)
    if seg.shape != gt.shape:
        raise ValueError("seg 和 gt 形状不一致")

    seg_er = binary_erosion(seg, structure=np.ones((3,3,3)), border_value=0)
    gt_er  = binary_erosion(gt,  structure=np.ones((3,3,3)), border_value=0)
    surf_seg = seg ^ seg_er
    surf_gt  = gt  ^ gt_er

    dt_to_gt  = bwdist(gt)
    dt_to_seg = bwdist(seg)

    d1 = dt_to_gt[surf_seg]
    d2 = dt_to_seg[surf_gt]

    if d1.size == 0 and d2.size == 0:
        return 0.0, 0.0
    all_d = np.concatenate([d1, d2]) if (d1.size > 0 and d2.size > 0) else (d1 if d2.size == 0 else d2)
    return float(np.mean(all_d)), float(np.max(all_d))


# ========= 主流程 =========
if __name__ == "__main__":
    print("[INFO] 读取图像与掩膜 ...")
    img_xyz, _ = sitk_read_xyz(IMAGE_PATH)
    mask_xyz, _ = sitk_read_xyz(MASK_PATH)
    mask_xyz = mask_xyz > 0

    print("[INFO] 预处理 & Otsu 分割 ...")
    img_win = window_ct_hu(img_xyz)
    img_smooth = gaussian_filter(img_win, sigma=1.0)
    th = threshold_otsu(img_smooth.ravel())
    seg = img_smooth > th
    seg = binary_opening(seg, structure=np.ones((3,3,3)))
    seg = binary_closing(seg, structure=np.ones((3,3,3)))
    seg = binary_fill_holes(seg)

    print("[INFO] 评估分割质量 ...")
    DICE = dice(seg, mask_xyz)
    VOE  = voe(seg, mask_xyz)
    RVD  = rel_vol_diff(seg, mask_xyz)
    ASD, MSD = surface_distances(seg, mask_xyz)

    metrics_dict = {
        "Dice": DICE,
        "VOE": VOE,
        "RelativeVolumeDiff": RVD,
        "AvgSurfaceDist(voxels)": ASD,
        "MaxSurfaceDist(voxels)": MSD
    }
    print("[METRICS]", metrics_dict)
    pd.DataFrame([metrics_dict]).to_csv(os.path.join(OUT_DIR, "segmentation_metrics.csv"),
                                        index=False, encoding="utf-8-sig")

    # 可视化中间切片
    k = get_middle_slice_index(img_xyz)
    sa_img, sa_gt, sa_seg = img_win[:,:,k], mask_xyz[:,:,k], seg[:,:,k]
    show_and_save_overlay(sa_img, sa_gt, sa_seg,
                          os.path.join(OUT_DIR, f"overlay_slice_z{k}.png"),
                          title=f"Overlay z={k} (red=GT, green=Otsu)")

    # ========= Radiomics =========
    print("[INFO] 提取 PyRadiomics 3D 体积特征 ...")
    from radiomics import featureextractor, setVerbosity
    setVerbosity(40)

    params = {
        "imageType": {"Original": {}},
        "setting": {
            "resampledPixelSpacing": [1.0, 1.0, 1.0],
            "interpolator": "sitkBSpline",
            "binWidth": 25,
            "normalize": False,
            "force2D": False
        }
    }

    imageITK = sitk.ReadImage(IMAGE_PATH)
    maskITK  = sitk.ReadImage(MASK_PATH)
    extractor = featureextractor.RadiomicsFeatureExtractor(params)
    extractor.enableAllFeatures()  # 用代码启用所有特征
    fv = extractor.execute(imageITK, maskITK)

    row = {k: v for k, v in fv.items() if k.startswith("original_")}
    feat_df = pd.DataFrame(list(row.items()), columns=["feature_name", "value"])
    feat_df.to_csv(os.path.join(OUT_DIR, "radiomics_3d_features.csv"),
                   index=False, encoding="utf-8-sig")
    safe_write_excel(os.path.join(OUT_DIR, "radiomics_3d_features.xlsx"), feat_df)

    print(f"[DONE] 结果已输出到：{OUT_DIR}")
    print(" - segmentation_metrics.csv")
    print(" - overlay_slice_*.png")
    print(" - radiomics_3d_features.(csv/xlsx)")
