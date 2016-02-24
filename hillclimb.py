import numpy
import math
import copy

def bits_for_precision(start, end, q):
    return math.ceil(math.log((end - start) * math.pow(10, q), 2))

def gen_bin_list(bits):
    return list(numpy.random.randint(2, size=bits))

def bin_list_to_int(bits):
    acc = 0
    for bit in bits:
        acc = acc * 2
        acc = acc + bit
    return acc

def bin_list_to_floats(bits, bits_per_value, start, end):
    res = []
    pos = bits_per_value
    for i in range(len(bits) / bits_per_value):
        intval = bin_list_to_int(bits[pos - bits_per_value:pos])
        pos = pos + bits_per_value
        floatval = float(intval) / (2 ** bits_per_value) * (end - start) + start
        res.append(floatval)
    return res

class all_neighbors:
    def __init__(self, bin_list):
        self.base = bin_list
        self.index = 0
    def __iter__(self):
        return self
    def next(self):
        if self.index >= len(self.base):
            raise StopIteration()
        res = copy.deepcopy(self.base)
        res[self.index] = int(not self.base[self.index])
        self.index += 1
        return res

# f6(x)=10·n+sum(x(i)^2-10·cos(2·pi·x(i))), i=1:n; -5.12<=x(i)<=5.12.
def rastrigin2(list):
    res = 10 * n + sum( (x ** 2 - 10 * math.cos(2 * math.pi * x)) for x in list )