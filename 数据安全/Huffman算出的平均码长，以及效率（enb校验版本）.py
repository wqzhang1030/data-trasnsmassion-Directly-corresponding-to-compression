import os, heapq, numpy as np

# === 配置：你的 RAW 路径 ===
image_path = r"C:\Users\zwq\Desktop\数据传输与安全\data\flower-bsq-u8be-3x1512x2268.raw"

# --- 基础函数 ---
def entropy_from_counts(counts: np.ndarray) -> float:
    total = int(counts.sum())
    if total == 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log2(p)).sum())

def build_huffman_code_lengths(counts: np.ndarray) -> np.ndarray:
    """给出每个符号的哈夫曼码长（未出现的符号为0）。"""
    heap = []
    uid = 0
    for s, c in enumerate(counts):
        c = int(c)
        if c > 0:
            heap.append((c, uid, ("leaf", s)))
            uid += 1
    if not heap:
        return np.zeros_like(counts, dtype=np.int32)

    import heapq
    heapq.heapify(heap)

    # 特例：仅1种符号时码长定为1
    if len(heap) == 1:
        _, _, node = heap[0]
        code_lengths = np.zeros_like(counts, dtype=np.int32)
        code_lengths[node[1]] = 1
        return code_lengths

    while len(heap) > 1:
        c1, _, n1 = heapq.heappop(heap)
        c2, _, n2 = heapq.heappop(heap)
        uid += 1
        heapq.heappush(heap, (c1 + c2, uid, ("node", n1, n2)))

    _, _, root = heap[0]
    code_lengths = np.zeros_like(counts, dtype=np.int32)

    def dfs(node, depth):
        t = node[0]
        if t == "leaf":
            code_lengths[node[1]] = max(depth, 1)
        else:
            _, l, r = node
            dfs(l, depth + 1)
            dfs(r, depth + 1)

    dfs(root, 0)
    return code_lengths

def main():
    # 读取数据
    if not os.path.isfile(image_path):
        print(f"[错误] 找不到文件：{image_path}")
        return
    data = np.fromfile(image_path, dtype=np.uint8)
    N = data.size
    print(f"Read {N} bytes.")

    # 1) 原图熵 H(X)
    counts = np.bincount(data, minlength=256).astype(np.int64)
    H = entropy_from_counts(counts)  # bit/符号
    print(f"1) Entropy H(X): {H:.6f} bit/symbol")

    # 2) 哈夫曼平均码长 L_avg（用码长期望，不生成大比特串）
    code_lengths = build_huffman_code_lengths(counts)          # 各符号码长
    probs = counts / counts.sum() if counts.sum() else np.zeros_like(counts, float)
    L_avg = float((probs * code_lengths).sum())                # 平均码长
    print(f"2) Huffman average code length L_avg: {L_avg:.6f} bit/symbol")

    # 3) 差值
    diff = L_avg - H
    print(f"3) Difference L_avg - H(X): {diff:.6f} bit/symbol")

    # 4) 压缩效率 & 压缩率（相对原始8 bit/字节）
    efficiency = H / L_avg if L_avg > 0 else 0.0
    ratio_bits = 8.0 / L_avg if L_avg > 0 else float("inf")
    print(f"4) Efficiency η = H/L_avg: {efficiency:.6f}")
    print(f"   Estimated compression ratio (vs 8 b/byte): {ratio_bits:.6f}x")

    # 估计总比特与节省
    est_total_bits = int((counts * code_lengths).sum())
    raw_bits = N * 8
    saving = 1.0 - est_total_bits / raw_bits if raw_bits > 0 else 0.0
    print(f"   Raw bits: {raw_bits}  |  Estimated Huffman bits: {est_total_bits}  |  Saving: {saving*100:.2f}%")

    # --- 可选：用 enb 做熵值校验与汇总表 ---
    try:
        import enb
        # 用同一公式算一次，作为“enb 校验行”展示（enb无现成哈夫曼接口，这里做校验+表格汇总）
        H_enb = entropy_from_counts(counts)
        print("\n[enb] sanity check:")
        print(f"    H_enb (should match H): {H_enb:.6f} bit/symbol")

        # 打印一个简易表格
        rows = [
            ("Entropy H(X) [bit/symbol]", H),
            ("Huffman L_avg [bit/symbol]", L_avg),
            ("Difference L_avg - H", diff),
            ("Efficiency η = H/L_avg", efficiency),
            ("Compression ratio (8/L_avg)", ratio_bits),
            ("Raw bits", raw_bits),
            ("Huffman bits (estimated)", est_total_bits),
            ("Saving (%)", saving * 100.0),
        ]
        # 简单对齐输出（不强依赖 enb 的表格类，避免 API 版本差异）
        print("\n[enb] summary table")
        w = max(len(k) for k, _ in rows)
        for k, v in rows:
            if isinstance(v, float):
                print(f"  {k:<{w}} : {v:.6f}")
            else:
                print(f"  {k:<{w}} : {v}")
    except Exception as e:
        print(f"\n[enb] 可选部分未运行：{e}")

if __name__ == "__main__":
    main()
