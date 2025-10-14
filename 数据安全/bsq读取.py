#!/usr/bin/env python3
"""
读取并解析 Band-Sequential (BSQ) 格式的原始 (RAW) 图像文件。
用户可以修改文件路径、宽度、高度、通道数和数据类型。
"""

import numpy as np
import os

# --- 【用户配置信息】 ---
# 图像文件路径
IMAGE_PATH = "C:\\Users\\zwq\\Desktop\\数据传输与安全\\data\\park_sequence\\park-frame207-u8be-3x720x1280.raw"
# 图像宽度 (W)
WIDTH = 1280
# 图像高度 (H)
HEIGHT = 720
# 图像通道数 (C) (例如: RGB=3, 灰度=1)
CHANNELS = 3
# 数据类型 (例如: u8be -> 8-bit unsigned integer, big endian)
# 请根据实际文件格式调整。常用的有：
# 'u1' (8-bit, unsigned), 'u2' (16-bit, unsigned), 'i4' (32-bit, signed)
DTYPE = np.dtype('>u1')  # '>u1' 表示 Big Endian (>) 8-bit unsigned integer (u1)


# --- ---------------------- ---

def read_bsq_image(file_path, width, height, channels, dtype):
    """
    从 RAW 文件中读取数据，并按照 BSQ 顺序进行重塑 (Reshape)。

    BSQ (Band Sequential) 顺序意味着数据存储为:
    [R1, R2, ..., Rn] [G1, G2, ..., Gn] [B1, B2, ..., Bn]
    """

    if not os.path.exists(file_path):
        print(f"错误：文件未找到。请检查路径是否正确：{file_path}")
        return None

    # 1. 计算总像素数和文件大小
    total_pixels = width * height
    total_samples = total_pixels * channels
    bytes_per_sample = dtype.itemsize
    expected_file_size = total_samples * bytes_per_sample

    print("=" * 60)
    print("RAW 图像 BSQ 读取器")
    print("-" * 60)
    print(f"预期文件大小: {expected_file_size / (1024 * 1024):.2f} MB ({expected_file_size} bytes)")
    print(f"图像尺寸: {width}x{height}，通道数: {channels}")
    print(f"数据类型: {dtype.name} ({bytes_per_sample} bytes/sample)")

    try:
        # 2. 读取整个文件到 NumPy 数组
        # 'C' order (行主序) 是默认的
        data = np.fromfile(file_path, dtype=dtype)

        # 3. 验证读取的数据量
        if data.size != total_samples:
            print(f"警告：读取的样本数 ({data.size}) 与预期样本数 ({total_samples}) 不匹配！")
            # 尝试继续，但结果可能不正确

        # 4. 按照 BSQ 顺序重塑 (Reshape)
        # 形状为 (通道数, 高度, 宽度)
        bsq_data = data.reshape((channels, height, width))

        print(f"数据成功读取并重塑为形状: {bsq_data.shape}")

        # 5. 提取并打印 R, G, B 分量的前几个像素值进行验证
        if channels >= 3:
            # 按照 BSQ 顺序，R 是第一个通道 (index 0)
            red_band = bsq_data[0]
            green_band = bsq_data[1]
            blue_band = bsq_data[2]

            print("-" * 60)
            print("各通道前 10 个像素值示例 (BSQ 格式):")
            print(f"R 分量 (Band 0): {red_band.flatten()[:10]}")
            print(f"G 分量 (Band 1): {green_band.flatten()[:10]}")
            print(f"B 分量 (Band 2): {blue_band.flatten()[:10]}")
            print("-" * 60)

            # 6. 额外：获取一个特定像素的 RGB 值 (例如：中心像素)
            center_x = width // 2
            center_y = height // 2
            print(f"中心像素 ({center_x}, {center_y}) 的 RGB 值:")
            print(
                f"R: {red_band[center_y, center_x]}, G: {green_band[center_y, center_x]}, B: {blue_band[center_y, center_x]}")

        return bsq_data

    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return None


if __name__ == "__main__":
    bsq_image = read_bsq_image(IMAGE_PATH, WIDTH, HEIGHT, CHANNELS, DTYPE)

    if bsq_image is not None:
        print("程序执行完毕。")
    else:
        print("程序因错误终止。")
