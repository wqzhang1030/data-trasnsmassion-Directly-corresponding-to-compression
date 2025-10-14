import numpy as np
import matplotlib.pyplot as plt
import heapq
from collections import Counter

# === Read image ===
path = r"C:\Users\zwq\Desktop\数据传输与安全\data\mandrill-u8be-3x512x512.raw"
data = np.fromfile(path, dtype=np.uint8)
print(f"Read {data.size} bytes from file.")

# === Build Huffman codes ===
def build_huffman_codes(symbols):
    counts = Counter(symbols)
    heap = [[weight, [sym, ""]] for sym, weight in counts.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    return dict(sorted(heapq.heappop(heap)[1:], key=lambda p: (len(p[-1]), p)))

# === Encode using Huffman ===
codes = build_huffman_codes(data)
encoded_bits = "".join(codes[b] for b in data)

# === Pack bits into bytes ===
padding = (8 - len(encoded_bits) % 8) % 8
encoded_bits += "0" * padding
encoded_bytes = np.frombuffer(int(encoded_bits, 2).to_bytes(len(encoded_bits)//8, 'big'), dtype=np.uint8)

# === Plot histograms ===
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(data, bins=256, color='skyblue', alpha=0.8)
axes[0].set_title("Original Image Byte Distribution")
axes[0].set_xlabel("Pixel Value")
axes[0].set_ylabel("Frequency")

axes[1].hist(encoded_bytes, bins=256, color='orange', alpha=0.8)
axes[1].set_title("Huffman Encoded Byte Distribution")
axes[1].set_xlabel("Byte Value (0–255)")
axes[1].set_ylabel("Frequency")

plt.tight_layout()
plt.show()
