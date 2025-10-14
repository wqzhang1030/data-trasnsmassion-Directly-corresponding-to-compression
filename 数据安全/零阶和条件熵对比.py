import numpy as np
from pathlib import Path

# ========= 工具函数 =========
def load_raw(filename, H, W, C, layout="BSQ", dtype=np.uint8):
    """
    读取 RAW 图像为 numpy 数组，形状 (H,W,C)
    layout: BSQ/BIL/BIP
    """
    raw = np.fromfile(filename, dtype=dtype)
    assert raw.size == H * W * C, f"尺寸不匹配：期望{H*W*C}，实际{raw.size}"
    layout = layout.upper()
    if layout == "BSQ":
        arr = raw.reshape(C, H, W).transpose(1, 2, 0)
    elif layout == "BIL":
        arr = raw.reshape(H, C, W).transpose(0, 2, 1)
    elif layout == "BIP":
        arr = raw.reshape(H, W, C)
    else:
        raise ValueError("layout 必须是 BSQ/BIL/BIP")
    return arr

def entropy0_from_array(x: np.ndarray) -> float:
    """零阶熵 H0"""
    vals, cnts = np.unique(x, return_counts=True)
    p = cnts / cnts.sum()
    return float(-(p * np.log2(p)).sum())

def cond_entropy_markov_order1(gray_2d: np.ndarray) -> float:
    """
    一阶马尔可夫条件熵 H(X_t | X_{t-1})
    使用向量化 bincount 统计 256x256 转移计数，避免 Python 循环
    """
    s = gray_2d.astype(np.uint16).ravel()
    if s.size <= 1:
        return 0.0
    prev = s[:-1]
    curr = s[1:]
    # 索引映射到 [0, 65535]
    idx = (prev.astype(np.int32) << 8) + curr.astype(np.int32)
    T = np.bincount(idx, minlength=256*256).reshape(256, 256)

    row_sums = T.sum(axis=1, keepdims=True)  # 每个前像素值 a 的计数
    with np.errstate(divide="ignore", invalid="ignore"):
        P_cond = np.divide(T, row_sums, where=row_sums > 0)  # P(b|a)

    P_prev = row_sums.flatten() / row_sums.sum()             # P(a)
    # 行条件熵 H(P(.|a))
    H_row = np.zeros(256, dtype=np.float64)
    for a in range(256):
        p = P_cond[a]
        p = p[p > 0]
        if p.size:
            H_row[a] = -(p * np.log2(p)).sum()
    H_cond = float((P_prev * H_row).sum())
    return H_cond

# ========= 主程序 =========
if __name__ == "__main__":
    # === 你的 RAW 文件（park 帧） ===
    filename = r"C:\Users\zwq\Desktop\数据传输与安全\data\park_sequence\park-frame207-u8be-3x720x1280.raw"
    H, W, C = 720, 1280, 3

    img = load_raw(filename, H, W, C, layout="BSQ", dtype=np.uint8)
    print("图像形状:", img.shape)

    # 分别计算 R/G/B 的零阶熵与条件熵
    H0_list = []
    H1_list = []
    for ch, name in zip([0, 1, 2], ["R", "G", "B"]):
        chan = img[:, :, ch]
        H0 = entropy0_from_array(chan)
        H1 = cond_entropy_markov_order1(chan)
        H0_list.append(H0)
        H1_list.append(H1)
        print(f"{name} 通道：零阶熵 H0 = {H0:.6f} bit/样本；条件熵 H1 = {H1:.6f} bit/样本")

    # 整体熵（按你的要求：三通道“相加”）
    H0_total = sum(H0_list)
    H1_total = sum(H1_list)

    print("-" * 60)
    print(f"整体零阶熵（R+G+B 相加） = {H0_total:.6f} bit/像素")
    print(f"整体条件熵（R+G+B 相加） = {H1_total:.6f} bit/像素")
