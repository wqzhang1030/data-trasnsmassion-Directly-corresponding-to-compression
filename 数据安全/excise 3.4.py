import math

def info_from_prob(p):
    assert 0 < p <= 1
    return -math.log2(p)

def info_from_outcomes(n):
    # n 个等概率结果
    assert n >= 1
    return math.log2(n)

# 示例：把每个小问都写成函数，运行时替换为相应参数
def ex_coin_fair_once():
    return info_from_outcomes(2)

def ex_two_independent_coins():
    return info_from_outcomes(2) + info_from_outcomes(2)

def ex_coin_and_die():
    return info_from_outcomes(2*6)

def ex_broken_alarm_always_off():
    # 唯一结果
    return info_from_outcomes(1)

# 你可以按需 print 这些函数的返回值进行验证

