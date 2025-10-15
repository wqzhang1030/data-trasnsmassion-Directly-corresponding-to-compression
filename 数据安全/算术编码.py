import os, struct, heapq, math
import numpy as np

# ====== 配置 ======
SRC_PATH = r"C:\Users\zwq\Desktop\数据传输与安全\data\mandrill-u8be-3x512x512.raw"
OUT_AC  = r"C:\Users\zwq\Desktop\数据传输与安全\data\mandrill-u8be-3x512x512.ac"
OUT_DEC = r"C:\Users\zwq\Desktop\数据传输与安全\data\mandrill-u8be-3x512x512.dec"

# ====== 工具：熵 & 频数 ======
def entropy_from_counts(counts: np.ndarray) -> float:
    total = int(counts.sum())
    if total == 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log2(p)).sum())

def build_cdf_from_counts(counts: np.ndarray) -> np.ndarray:
    """返回长度=257 的累计频率表 cdf，cdf[0]=0，cdf[-1]=total。确保每个符号至少 1（避免零概率）。"""
    counts = counts.astype(np.int64).copy()
    # 防 0：把为 0 的项补成 1（会带来极小冗余，但保证可编码）
    zero_mask = counts == 0
    if zero_mask.any():
        counts[zero_mask] = 1
    cdf = np.zeros(len(counts) + 1, dtype=np.int64)
    np.cumsum(counts, out=cdf[1:])
    return cdf

# ====== 位流 ======
class BitWriter:
    def __init__(self):
        self.buf = bytearray()
        self.cur = 0
        self.n = 0
        self.total_bits = 0

    def write_bit(self, b: int):
        self.cur = (self.cur << 1) | (1 if b else 0)
        self.n += 1
        self.total_bits += 1
        if self.n == 8:
            self.buf.append(self.cur)
            self.cur = 0
            self.n = 0

    def flush(self) -> int:
        pad = 0
        if self.n > 0:
            pad = 8 - self.n
            self.cur <<= pad
            self.buf.append(self.cur)
            self.cur = 0
            self.n = 0
        return pad

    def get_bytes(self) -> bytes:
        return bytes(self.buf)

class BitReader:
    def __init__(self, data: bytes, valid_bits: int):
        self.data = data
        self.valid_bits = valid_bits
        self.pos_bits = 0
        self.i = 0
        self.cur = 0
        self.n = 0

    def read_bit(self) -> int:
        if self.pos_bits >= self.valid_bits:
            return 0  # 安全返回 0（不应被使用）
        if self.n == 0:
            if self.i >= len(self.data):
                self.cur = 0
            else:
                self.cur = self.data[self.i]
                self.i += 1
            self.n = 8
        b = (self.cur >> 7) & 1
        self.cur = (self.cur << 1) & 0xFF
        self.n -= 1
        self.pos_bits += 1
        return b

# ====== 算术编码（32-bit 区间，E1/E2/E3 规则） ======
# 参考思想：Witten–Neal–Cleary 算法与常见 range coder 结构
MASK = (1 << 32) - 1
HALF = 1 << 31
QUARTER = 1 << 30
THREE_QUARTER = 3 << 30

def ac_encode(data: np.ndarray, cdf: np.ndarray) -> (bytes, int):
    """返回 (payload_bytes, valid_bits)。cdf 长度=257，总量 total=cdf[-1]"""
    total = int(cdf[-1])
    bw = BitWriter()
    low = 0
    high = MASK
    pending = 0

    def output_bit(b: int):
        nonlocal pending
        bw.write_bit(b)
        # 把积累的 underflow 按相反位输出
        while pending > 0:
            bw.write_bit(1 - b)
            pending -= 1

    for sym in data:
        sym = int(sym)
        # 区间更新
        rng = (high - low + 1)
        lo = cdf[sym]
        hi = cdf[sym + 1]
        # 高端是闭区间，所以 -1
        high = low + (rng * hi) // total - 1
        low  = low + (rng * lo) // total

        # 归一化输出
        while True:
            if high < HALF:
                # MSB 都是 0
                output_bit(0)
                low  = (low  << 1) & MASK
                high = ((high << 1) & MASK) | 1
            elif low >= HALF:
                # MSB 都是 1
                output_bit(1)
                low  = ((low  - HALF) << 1) & MASK
                high = (((high - HALF) << 1) & MASK) | 1
            elif low >= QUARTER and high < THREE_QUARTER:
                # E3 underflow
                pending += 1
                low  = ((low  - QUARTER) << 1) & MASK
                high = (((high - QUARTER) << 1) & MASK) | 1
            else:
                break

    # 终止：强行区分区间
    pending += 1
    if low < QUARTER:
        output_bit(0)
    else:
        output_bit(1)

    pad = bw.flush()
    payload = bw.get_bytes()
    valid_bits = bw.total_bits  # 有效位（不含 pad）
    return payload, valid_bits

def ac_decode(payload: bytes, valid_bits: int, cdf: np.ndarray, out_len: int) -> np.ndarray:
    total = int(cdf[-1])
    br = BitReader(payload, valid_bits)
    low = 0
    high = MASK
    code = 0
    # 读入初始 32 位
    for _ in range(32):
        code = ((code << 1) & MASK) | br.read_bit()

    out = np.empty(out_len, dtype=np.uint8)

    # 二分查找函数：给定 cum，找到符号 s 使得 cdf[s] <= cum < cdf[s+1]
    def find_symbol(cum: int) -> int:
        lo, hi = 0, 256
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if cdf[mid] <= cum:
                lo = mid
            else:
                hi = mid
        return lo

    for i in range(out_len):
        rng = (high - low + 1)
        # 计算当前 cum 落点
        # 与编码端完全对应的公式：
        cum = (( (code - low + 1) * total - 1) // rng)
        sym = find_symbol(cum)

        out[i] = sym

        # 收缩到该符号区间
        lo_c = cdf[sym]
        hi_c = cdf[sym + 1]
        high = low + (rng * hi_c) // total - 1
        low  = low + (rng * lo_c) // total

        # 归一化读位
        while True:
            if high < HALF:
                # do nothing, just scale
                pass
            elif low >= HALF:
                low  -= HALF
                high -= HALF
                code -= HALF
            elif low >= QUARTER and high < THREE_QUARTER:
                low  -= QUARTER
                high -= QUARTER
                code -= QUARTER
            else:
                break
            low  = (low  << 1) & MASK
            high = ((high << 1) & MASK) | 1
            code = ((code << 1) & MASK) | br.read_bit()

    return out

# ====== 主流程 ======
def main():
    if not os.path.isfile(SRC_PATH):
        print(f"[错误] 找不到文件：{SRC_PATH}")
        return

    data = np.fromfile(SRC_PATH, dtype=np.uint8)
    N = data.size
    raw_bits = N * 8
    print(f"Read {N} bytes ({raw_bits} bits).")

    # 1) 熵
    counts = np.bincount(data, minlength=256).astype(np.int64)
    H = entropy_from_counts(counts)
    print(f"Entropy H(X): {H:.6f} bit/symbol")

    # 2) 构建 CDF（每符号至少 1）
    cdf = build_cdf_from_counts(counts)
    total = int(cdf[-1])

    # 3) 算术编码
    payload, valid_bits = ac_encode(data, cdf)
    # 写文件：自定义简单容器
    # magic(4) 'AC01' | N(uint64) | valid_bits(uint64) | counts(256*uint32 LE) | payload
    with open(OUT_AC, "wb") as f:
        f.write(b"AC01")
        f.write(struct.pack("<Q", int(N)))
        f.write(struct.pack("<Q", int(valid_bits)))
        f.write(struct.pack("<256I", *[int(x) for x in counts]))
        f.write(payload)
    print(f"[encode] wrote: {OUT_AC}  size={os.path.getsize(OUT_AC)} bytes")

    # 4) 算术解码（验证）
    with open(OUT_AC, "rb") as f:
        magic = f.read(4); assert magic == b"AC01"
        N2 = struct.unpack("<Q", f.read(8))[0]
        valid_bits2 = struct.unpack("<Q", f.read(8))[0]
        counts2 = np.array(struct.unpack("<256I", f.read(4*256)), dtype=np.int64)
        payload2 = f.read()
    assert N2 == N and valid_bits2 == valid_bits
    cdf2 = build_cdf_from_counts(counts2)
    dec = ac_decode(payload2, valid_bits, cdf2, N)
    assert np.array_equal(dec, data), "解码不一致！"
    with open(OUT_DEC, "wb") as f:
        dec.tofile(f)
    print(f"[decode] ok, wrote: {OUT_DEC}")

    # 5) 指标
    L_avg = valid_bits / N
    diff = L_avg - H
    eta = H / L_avg if L_avg > 0 else 0.0
    cr = 8.0 / L_avg if L_avg > 0 else float("inf")
    saving = 1.0 - valid_bits / raw_bits

    print("\n=== Metrics ===")
    print(f"L_avg (arith, actual)  : {L_avg:.6f} bit/symbol")
    print(f"L_avg - H(X)           : {diff:.6f} bit/symbol")
    print(f"Efficiency η = H/L_avg : {eta:.6f}")
    print(f"Compression ratio 8/L  : {cr:.6f}x")
    print(f"Saving vs raw          : {saving*100:.2f}%")

if __name__ == "__main__":
    main()
