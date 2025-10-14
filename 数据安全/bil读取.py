#!/usr/bin/env python3
"""
读取并解析 Band-Interleaved by Line (BIL) 格式的原始 (RAW) 图像文件。
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
# '>u1' 表示 Big Endian (>) 8-bit unsigned integer (u1)
DTYPE = np.dtype('>u1')


# --- ---------------------- ---

def read_bil_image(file_path, width, height, channels, dtype):
    """
    从 RAW 文件中读取数据，并按照 BIL 顺序进行重塑 (Reshape)。

    BIL (Band Interleaved by Line) 顺序意味着数据存储为:
    [R11, G11, B11, R12, G12, B12, ...] (第一行),
    [R21, G21, B21, R22, G22, B22, ...] (第二行), ...

    重塑形状为 (高度, 宽度, 通道数)。
    """

    if not os.path.exists(file_path):
        print(f"错误：文件未找到。请检查路径是否正确：{file_path}")
        return None

    # 1. 计算总像素数和文件大小
    total_samples = width * height * channels
    bytes_per_sample = dtype.itemsize
    expected_file_size = total_samples * bytes_per_sample

    print("=" * 60)
    print("RAW 图像 BIL 读取器")
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

        # 4. 按照 BIL 顺序重塑 (Reshape)
        # 形状为 (高度, 宽度, 通道数)
        bil_data = data.reshape((height, width, channels))

        print(f"数据成功读取并重塑为形状: {bil_data.shape}")

        # 5. 提取并打印 R, G, B 分量的前几个像素值进行验证
        # 在 BIL/BIP 格式中，数据已经按像素组织。
        if channels >= 3:
            # 获取第一行的前 5 个像素
            first_row_pixels = bil_data[0, :5, :]

            print("-" * 60)
            print("第一行前 5 个像素值示例 (BIL 格式):")

            # 打印每个像素的 (R, G, B) 值
            for i, pixel in enumerate(first_row_pixels):
                # 假设通道顺序是 RGB (索引 0=R, 1=G, 2=B)
                print(f"像素 (0, {i}): R={pixel[0]}, G={pixel[1]}, B={pixel[2]}")

            print("-" * 60)

            # 6. 额外：获取一个特定像素的 RGB 值 (例如：中心像素)
            center_x = width // 2
            center_y = height // 2
            center_pixel = bil_data[center_y, center_x, :]

            print(f"中心像素 ({center_x}, {center_y}) 的 RGB 值:")
            print(f"R: {center_pixel[0]}, G: {center_pixel[1]}, B: {center_pixel[2]}")

        return bil_data

    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return None


if __name__ == "__main__":
    bil_image = read_bil_image(IMAGE_PATH, WIDTH, HEIGHT, CHANNELS, DTYPE)

    if bil_image is not None:
        print("程序执行完毕。")
    else:
        print("程序因错误终止。")
