# -*- coding: utf-8 -*-
"""
霍夫曼压缩 + 熵与效率对比（八位小数、人类可读科学计数法）
Author: you
"""

import os
import math
import heapq
from collections import defaultdict
from typing import Dict, Tuple, List

# ===================== 用户配置（按需修改） =====================
RAW_PATH = r"C:\Users\zwq\Desktop\数据传输与安全\data\flower-bsq-u8be-3x1512x2268.raw"
PPM_PATH = r"C:\Users\zwq\Desktop\数据传输与安全\data\flower_with_header.ppm"

# 输出压缩文件与同目录同名（扩展名 .huff）
RAW_OUT = os.path.splitext(RAW_PATH)[0] + ".huff"
PPM_OUT = os.path.splitext(PPM_PATH)[0] + ".huff"

# 分块读取大小
CHUNK_SIZE = 1024 * 1024  # 1 MB
# ===================== 结束配置 =====================


# ---------- 工具函数 ----------
def human(n: int) -> str:
    """将字节数格式化为可读字符串"""
    size = float(n)
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0 or unit == 'TB':
            return f"{size:.2f} {unit}"
        size /= 1024.0

def sci_human(x: float, digits: int = 8) -> str:
    """人类可读科学计数法（mant × 10^exp），保留 digits 位小数"""
    if x == 0 or x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "0"
    exp = int(math.floor(math.log10(abs(x))))
    mant = x / (10 ** exp)
    return f"{mant:.{digits}f} × 10^{exp}"

def file_size(path: str) -> int:
    return os.path.getsize(path)


# ---------- 统计与熵 ----------
def build_freq_table(path: str) -> Tuple[Dict[int, int], int]:
    """按字节（0~255）统计频率，返回频率表与总符号数"""
    freq: Dict[int, int] = defaultdict(int)
    total = 0
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break
            total += len(chunk)
            for b in chunk:
                freq[b] += 1
    return freq, total

def entropy_bits(freq: Dict[int, int], total: int) -> float:
    """香农熵 H（bit/符号），以字节为符号"""
    if total == 0:
        return 0.0
    H = 0.0
    for c in freq.values():
        if c == 0:
            continue
        p = c / total
        H -= p * math.log2(p)
    return H


# ---------- 霍夫曼构建 ----------
class Node:
    __slots__ = ("w", "sym", "left", "right")
    def __init__(self, w: int, sym: int = None, left=None, right=None):
        self.w = w
        self.sym = sym
        self.left = left
        self.right = right
    def __lt__(self, other):
        # Python 的 heapq 需要可比较；用权重打破平局
        if self.w != other.w:
            return self.w < other.w
        # 为稳定性，优先叶子（符号小的在前）
        sa = self.sym if self.sym is not None else 256
        sb = other.sym if other.sym is not None else 256
        return sa < sb

def build_huffman_code_lengths(freq: Dict[int, int]) -> Dict[int, int]:
    """构造霍夫曼树，返回各符号码长（不生成实际比特串）"""
    items = [(w, s) for s, w in freq.items() if w > 0]
    if not items:
        return {}
    # 特例：只有一个符号 -> 码长给 1
    if len(items) == 1:
        only_sym = items[0][1]
        return {only_sym: 1}

    heap: List[Node] = [Node(w, sym=s) for w, s in items]
    heapq.heapify(heap)
    while len(heap) > 1:
        a = heapq.heappop(heap)
        b = heapq.heappop(heap)
        heapq.heappush(heap, Node(a.w + b.w, left=a, right=b))
    root = heap[0]

    lengths: Dict[int, int] = {}
    def dfs(node: Node, depth: int):
        if node.sym is not None:
            lengths[node.sym] = max(1, depth)
            return
        dfs(node.left, depth + 1)
        dfs(node.right, depth + 1)
    dfs(root, 0)
    return lengths

def build_canonical_codes(code_lengths: Dict[int, int]) -> Dict[int, Tuple[int, int]]:
    """
    根据码长生成规范霍夫曼码（Canonical Huffman Codes）
    返回: {symbol: (code, length)}
    """
    if not code_lengths:
        return {}

    # 统计每个长度的数量
    max_len = max(code_lengths.values())
    bl_count = [0] * (max_len + 1)
    for ln in code_lengths.values():
        bl_count[ln] += 1

    # 计算每个长度的起始码值（标准做法）
    next_code = [0] * (max_len + 1)
    code = 0
    for bits in range(1, max_len + 1):
        code = (code + bl_count[bits - 1]) << 1
        next_code[bits] = code

    # 按 (长度, 符号) 排序并分配码字
    codes: Dict[int, Tuple[int, int]] = {}
    for sym, ln in sorted(code_lengths.items(), key=lambda kv: (kv[1], kv[0])):
        code_val = next_code[ln]
        codes[sym] = (code_val, ln)
        next_code[ln] += 1
    return codes


# ---------- 简单容器头（便于解压） ----------
def write_header_with_freqs(fout, total_symbols: int, freqs: Dict[int, int]):
    """
    容器格式：
    magic(4B) = b'HUF0'
    original_size(8B, little-endian)
    table: 256 * uint32 小端频率表
    """
    fout.write(b'HUF0')
    fout.write(total_symbols.to_bytes(8, 'little', signed=False))
    for i in range(256):
        fout.write(freqs.get(i, 0).to_bytes(4, 'little', signed=False))


# ---------- 压缩主流程 ----------
def huffman_compress(in_path: str, out_path: str) -> Dict[str, float]:
    """
    压缩并返回综合指标：
    {
      'H', 'Lavg', 'orig_bytes', 'comp_bytes',
      'efficiency'(η=H/Lavg), 'redundancy'(R=Lavg-H),
      'CR'(原/压缩), 'reduction'(节省百分比)
    }
    """
    # 1) 频率与熵
    freq, total = build_freq_table(in_path)
    H = entropy_bits(freq, total) if total > 0 else 0.0

    # 2) 码长与平均码长
    lengths = build_huffman_code_lengths(freq)
    if total == 0 or not lengths:
        # 空文件或异常：只写容器头
        with open(out_path, 'wb') as fout:
            write_header_with_freqs(fout, 0, {})
        comp_size = file_size(out_path)
        return {
            'H': 0.0, 'Lavg': 0.0,
            'orig_bytes': 0, 'comp_bytes': comp_size,
            'efficiency': 0.0, 'redundancy': 0.0,
            'CR': float('inf'), 'reduction': 0.0
        }

    Lavg = sum((freq[sym] / total) * lengths[sym] for sym in lengths)

    # 3) 生成规范码并编码
    codes = build_canonical_codes(lengths)

    with open(in_path, 'rb') as fin, open(out_path, 'wb') as fout:
        # 文件头（含原始大小与频率表）
        write_header_with_freqs(fout, total, freq)

        bitbuf = 0  # 累积到最低位
        bitcount = 0

        def flush_byte():
            nonlocal bitbuf, bitcount
            fout.write(bytes([bitbuf & 0xFF]))
            bitbuf >>= 8
            bitcount -= 8

        while True:
            chunk = fin.read(CHUNK_SIZE)
            if not chunk:
                break
            for b in chunk:
                code, ln = codes[b]
                # 将 code 的低位对齐当前 bitbuf（bitbuf 低位先出）
                bitbuf |= (code << bitcount)
                bitcount += ln
                while bitcount >= 8:
                    flush_byte()
        # 尾部不足 8 位也写出（零填充）
        if bitcount > 0:
            fout.write(bytes([bitbuf & 0xFF]))

    comp_size = file_size(out_path)

    # 4) 指标（注意：压缩大小包含容器头开销，更贴近“实际压缩文件”）
    CR = (total / comp_size) if comp_size > 0 else float('inf')  # 压缩比（越大越好）
    reduction = (1 - comp_size / total) * 100 if total > 0 else 0.0  # 节省百分比
    efficiency = (H / Lavg) if Lavg > 0 else 0.0
    redundancy = Lavg - H

    return {
        'H': H, 'Lavg': Lavg,
        'orig_bytes': total, 'comp_bytes': comp_size,
        'efficiency': efficiency, 'redundancy': redundancy,
        'CR': CR, 'reduction': reduction
    }


# ---------- 报告输出 ----------
def report_one(name: str, stats: Dict[str, float]):
    print(f"\n=== {name} ===")
    print(f"原始大小: {stats['orig_bytes']} 字节 ({human(stats['orig_bytes'])})")
    print(f"压缩后:   {stats['comp_bytes']} 字节 ({human(stats['comp_bytes'])})")
    print(f"信息熵 H: {sci_human(stats['H'])} bit/符号（按字节）")
    print(f"平均码长 Lavg: {sci_human(stats['Lavg'])} bit/符号")
    print(f"编码效率 η = H/Lavg: {sci_human(stats['efficiency'])}")
    print(f"冗余 R = Lavg - H: {sci_human(stats['redundancy'])} bit/符号")
    print(f"压缩比 CR = 原/压缩: {sci_human(stats['CR'])}")
    print(f"压缩率（节省%）: {sci_human(stats['reduction'])} %")

def main():
    # RAW
    raw_stats = huffman_compress(RAW_PATH, RAW_OUT)
    # PPM（包含文本头 + 像素数据，作为普通字节流整体压缩）
    ppm_stats = huffman_compress(PPM_PATH, PPM_OUT)

    report_one("RAW 原始数据", raw_stats)

    report_one("PPM（含 header）", ppm_stats)

    # 对比汇总
    print("\n=== 对比总结 ===")
    print(f"H 对比（越小越易压）：RAW={sci_human(raw_stats['H'])} , PPM={sci_human(ppm_stats['H'])}")
    print(f"Lavg 对比：RAW={sci_human(raw_stats['Lavg'])} , PPM={sci_human(ppm_stats['Lavg'])}")
    print(f"压缩比 CR：RAW={sci_human(raw_stats['CR'])} , PPM={sci_human(ppm_stats['CR'])}")
    print(f"压缩率（节省%）：RAW={sci_human(raw_stats['reduction'])} % , PPM={sci_human(ppm_stats['reduction'])} %")
    print(f"编码效率 η：RAW={sci_human(raw_stats['efficiency'])} , PPM={sci_human(ppm_stats['efficiency'])}")

if __name__ == "__main__":
    main()
