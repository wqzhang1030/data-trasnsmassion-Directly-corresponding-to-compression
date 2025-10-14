import math

def entropy(probabilities):
    return -sum(p * math.log2(p) for p in probabilities if p > 0)

# 假设一个事件 X，有三种可能结果，概率如下：
p_X = [0.5, 0.25, 0.25]

H_X = entropy(p_X)
print(f"H(X) = {H_X:.4f} bits")




from fractions import Fraction
import math

def entropy(probabilities):
    return -sum(float(p) * math.log2(float(p)) for p in probabilities if p > 0)

# 用分数表示概率
p_X = [Fraction(1, 2), Fraction(1, 4), Fraction(1, 4)]

H_X = entropy(p_X)
print(f"H(X) = {H_X:.4f} bits")

