# -*- coding: utf-8 -*-
"""
计算 BSQ(u8) 3×1512×2268 图像的平均游程长度（ARL）
- 文件按 BSQ 存：R 平面 | G 平面 | B 平面
- ARL = 像素总数 / 游程数（游程：连续相同值的一段）
"""

import numpy as np
from pathlib import Path

# ===== 参数区 =====
path = r"C:\Users\zwq\Desktop\数据传输与安全\data\flower-bsq-u8be-3x1512x2268.raw"
H, W, C = 1512, 2268, 3   # Y, X, 通道数（BSQ）
dtype = np.uint8          # u8

# ===== 读取与检查 =====
p = Path(path)
data = np.fromfile(p, dtype=dtype)
expected = H * W * C
if data.size != expected:
    raise ValueError(f"文件大小不匹配：期望 {expected}，实际 {data.size}")

# ===== 按 BSQ 拆分通道 =====
band_size = H * W
channels = []
for c in range(C):
    start = c * band_size
    end = (c + 1) * band_size
    ch = data[start:end]        # 按行优先的栅格序（raster order）
    channels.append(ch)

# ===== 计算 ARL（1D 栅格序）=====
def average_run_length(vec: np.ndarray) -> float:
    """平均游程长度：像素总数 / 游程数"""
    if vec.size == 0:
        return float("nan")
    # 找到相邻像素发生变化的位置，变化次数+1 = 游程数
    runs = np.count_nonzero(np.diff(vec) != 0) + 1
    return vec.size / runs

arl_per_channel = [average_run_length(ch) for ch in channels]

# “整体”=把 R、G、B 三个平面顺序拼接（仍是 BSQ 的顺序）
arl_all = average_run_length(data)

# ===== 打印结果 =====
names = ["R", "G", "B"]
for n, a in zip(names, arl_per_channel):
    print(f"{n} 通道 ARL = {a:.3f}")
print(f"整体（R|G|B 拼接）ARL = {arl_all:.3f}")

# 简单结论：ARL 阈值参考
def rle_comment(arl: float) -> str:
    if arl < 1.3:
        return "游程很短，RLE 基本无效"
    if arl < 2.0:
        return "游程一般，RLE 收益有限"
    if arl < 5.0:
        return "存在明显游程，RLE 有一定效果"
    return "游程很长，RLE 效果显著"

print("\n判读（按通道）：")
for n, a in zip(names, arl_per_channel):
    print(f"- {n}: {rle_comment(a)}")
print(f"- 整体: {rle_comment(arl_all)}")
