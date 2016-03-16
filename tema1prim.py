#!/usr/bin/python
# coding: utf-8
import math
import numpy
import pprint
import copy

import gastandard

# x3-60x2+900x+100
def func(x):
    return x ** 3 - 60 * x ** 2 + 900 * x + 100

def fitness_func(x):
    return x

def find_attraction(bestfit):
    results = {}
    for i in range(0, 31):
        start = Item()



if __name__ == "__main__":
    print "Attraction basins, best fit"
    find_attraction(True)
    print "Attraction basins, first fit"
    find_attraction(False)