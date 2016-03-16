#!/usr/bin/python
# coding: utf-8
import math
import numpy
import pprint
import copy

import ga_standard

# x3-60x2+900x+100
def func(x):
    return x[0] ** 3 - 60 * x[0] ** 2 + 900 * x[0] + 100

def fitness_func(x):
    return x

def find_attraction(bestfit):
    results = {}
    for i in range(0, 32):
        start = ga_standard.Item()
        start.components = 1
        start.intervals = [[0, 31]]
        start.decimals = 0
        start.value_func = func
        start.fitness_func = fitness_func
        start.setup_bits()
        start.bits = ga_standard.int_to_bin_list(i, start.bit_len)
        start.update_values()
        start.hillclimb_run(bestfit)
        start.update_values()
        floatint = int(start.floats[0])
        if not floatint in results:
            results[floatint] = [i]
        else:
            results[floatint].append(i)
        #print i, "converges to", floatint
    return results


def pretty_display(d):
    for k, v in d.iteritems():
        print k, "(", func([k]), ")", ":", v

if __name__ == "__main__":
    print "Attraction basins, best fit"
    resbest = find_attraction(True)
    pretty_display(resbest)
    #pprint.pprint(resbest)
    print "Attraction basins, first fit"
    resfirst = find_attraction(False)
    pretty_display(resfirst)
    #pprint.pprint(resfirst)