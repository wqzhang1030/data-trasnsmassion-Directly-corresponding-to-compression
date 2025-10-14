import math

def entropy(probabilities):
    """计算单个事件的信息熵（单位：bit）"""
    return -sum(p * math.log2(p) for p in probabilities if p > 0)

# 假设三个独立事件的概率分布
# 事件 A 有 2 种结果
p_A = [0.7, 0.3]

# 事件 B 有 3 种结果
p_B = [0.5, 0.25, 0.25]

# 事件 C 有 2 种结果
p_C = [0.9, 0.1]

# 分别计算各自熵
H_A = entropy(p_A)
H_B = entropy(p_B)
H_C = entropy(p_C)

# 独立时的联合熵
H_total = H_A + H_B + H_C

print(f"H(A) = {H_A:.4f} bits")
print(f"H(B) = {H_B:.4f} bits")
print(f"H(C) = {H_C:.4f} bits")
print(f"H(A,B,C) = {H_total:.4f} bits")
