#!/usr/bin/python
# coding: utf-8

import numpy
import math
import copy
import operator

def bits_for_precision(start, end, q):
    return int(math.ceil(math.log((end - start) * math.pow(10, q), 2)))

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
    for i in range(int(len(bits) / bits_per_value)):
        intval = bin_list_to_int(bits[(pos - bits_per_value):pos])
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

def iterated_hill_climbing(iterations, function, dimensions, start, end, decimals):
    bits_per_elem = bits_for_precision(start, end, decimals)
    bits_total = bits_per_elem * dimensions
    print "For", decimals, "decimals precision on [", start, ",", end, "] I need", bits_per_elem, "bits per element"
    # Init best value with a random value, keep it as binary
    best_overall = gen_bin_list(bits_total)
    best_overall_floats = bin_list_to_floats(best_overall, bits_per_elem, start, end)
    best_overall_func = function(best_overall_floats)
    for iter in range(0, iterations):
        cur_val = gen_bin_list(bits_total)
        cur_val_floats = bin_list_to_floats(cur_val, bits_per_elem, start, end)
        cur_val_func = function(cur_val_floats)
        print "Random init:", cur_val, cur_val_floats, cur_val_func
        did_swap = True
        while did_swap:
            did_swap = False
            for n in all_neighbors(cur_val):
                n_floats = bin_list_to_floats(n, bits_per_elem, start, end)
                n_func = function(n_floats)
                print n, n_floats, n_func
                if n_func < cur_val_func:
                    cur_val = n
                    cur_val_floats = n_floats
                    cur_val_func = n_func
                    did_swap = True
                    print "Better!"
            print "END Neighbors round"
        print "END ITERATION"
        if best_overall_func > cur_val_func:
            print "New best overall value!"
            best_overall = cur_val
            best_overall_floats = cur_val_floats
            best_overall_func = cur_val_func
    print "Best result after", iterations, "iterations:"
    print best_overall, best_overall_floats, best_overall_func

# f6(x)=10·n+sum(x(i)^2-10·cos(2·pi·x(i))), i=1:n; -5.12<=x(i)<=5.12.
def rastrigin2(list):
    n = len(list)
    res = 10 * n + sum( (x ** 2 - 10 * math.cos(2 * math.pi * x)) for x in list )
    return res

# No prod function in python, why do we have sum?
def prod(factors):
    return reduce(operator.mul, factors, 1)

# f8(x)=sum(x(i)^2/4000)-prod(cos(x(i)/sqrt(i)))+1, i=1:n -600<=x(i)<= 600.
def griegwank8(list):
    product = 1
    for i in range(0, len(list)):
        product = product * math.cos(list[i]) / math.sqrt(i + 1)
    res = sum((x ** 2 / 4000.0) for x in list) + product + 1
    return res

# f2(x)=sum(100·(x(i+1)-x(i)^2)^2+(1-x(i))^2) i=1:n-1; -2.048<=x(i)<=2.048.
def rosenbrock(list):
    sum = 0
    for i in range(0, len(list)-1):
        sum = sum + 100 * (list[i+1] - list[i] ** 2) ** 2 + (1 - list[i]) ** 2
    return sum

# fSixh(x1,x2)=(4-2.1·x1^2+x1^4/3)·x1^2+x1·x2+(-4+4·x2^2)·x2^2 -3<=x1<=3, -2<=x2<=2.
def sixhump(list):
    x1 = list[0]
    x2 = list[1]
    res = (4 - (2.1 * (x1 ** 2)) + math.pow(x1, 4/3)) * (x1 ** 2)
    res = res + x1 * x2
    res = res + (-4 + 4 * (x2 ** 2)) * (x2 ** 2)
    return res

# def iterated_hill_climbing(iterations, function, dimensions, start, end, decimals):
#iterated_hill_climbing(1, rastrigin2, 2, -5.12, 5.12, 2)
#iterated_hill_climbing(100, griegwank8, 2, -600, 600, 2)
#iterated_hill_climbing(100, rosenbrock, 2, -2.048, 2.048, 3)
#iterated_hill_climbing(10, sixhump, 2, -2, 2, 6)

print sixhump([-2, 0.8])
