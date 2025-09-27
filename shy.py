# ababab 6

def sov(s):
    len_num = len(s)
    half_len_num = int(len_num / 2)
    if len_num == 0 or 1:
        return False
    for i in range(half_len_num):
        str = s[:i]
        if str * (half_len_num / i) == s:
            return True
    return False


Sov = sov('abab')

