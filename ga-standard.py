#!/usr/bin/python
# coding: utf-8
import math
import numpy
import pprint
import copy

# Helper functions pulled from hill climbing
def bits_for_precision(start, end, q):
    print "bits_for_precision", start, end, q
    return int(math.ceil(math.log((end - start) * math.pow(10, q), 2)))

def gen_bin_list(bits):
    return list(numpy.random.randint(2, size=bits))

def bin_list_to_int(bits):
    acc = 0
    for bit in bits:
        acc = acc * 2
        acc = acc + bit
    return acc

# fSixh(x1,x2)=(4-2.1·x1^2+x1^4/3)·x1^2+x1·x2+(-4+4·x2^2)·x2^2 -3<=x1<=3, -2<=x2<=2.
def sixhump(list):
    #print "sixhump called with", list
    x1 = list[0]
    x2 = list[1]
    res1 = (4 - (2.1 * (x1 ** 2)) + (x1 ** 4)/3) * (x1 ** 2)
    res2 = x1 * x2
    res3 = (-4 + 4 * (x2 ** 2)) * (x2 ** 2)
    #print res1, res2, res3
    res = res1 + res2 + res3
    return res

def sixhump_fitness(val):
    return 1/(val + 1.05) # Since we know the minimum to be 1.03

class Item(object):

    def __init__(self):
        self.components = 0     # Number of (float) components
        self.intervals = []     # [(-2, 5), (-1, 2)] - one pair per component
        self.decimals = 3       # Desired precision - for all components
        self.bits_per_comp = [] # To be calculated
        self.bit_len = 0        # Total length in bits
        self.bits = []          # The bit string for a value, when it holds a value
        self.floats = []        # The float values, calculated from the bits
        self.value = None       # The function value
        self.value_func = None
        self.fitness = None     # The fitness function value, can be the same or different (ex Six Hump)
        self.fitness_func = None

    def setup_bits(self):
        self.bits_per_comp = []
        self.bit_len = 0
        for i in self.intervals:
            comp_len = bits_for_precision(i[0], i[1], self.decimals)
            self.bits_per_comp.append(comp_len)
            self.bit_len += comp_len

    def bin_list_to_floats(bits, bits_per_value, start, end):
        res = []
        pos = bits_per_value
        for i in range(int(len(bits) / bits_per_value)):
            intval = bin_list_to_int(bits[(pos - bits_per_value):pos])
            pos = pos + bits_per_value
            floatval = float(intval) / (2 ** bits_per_value) * (end - start) + start
            res.append(floatval)
        return res

    def floats_from_bits(self):
        self.floats = []
        pos = 0
        for i in zip(self.intervals, self.bits_per_comp):
            start = i[0][0]
            end = i[0][1]
            bpc = i[1]
            intval = bin_list_to_int(self.bits[pos:(pos + bpc)])
            floatval = float(intval) / ((2 ** bpc) - 1) * (end - start) + start
            self.floats.append(floatval)

    def random_init(self):
        self.bits = gen_bin_list(self.bit_len)

    def update_values(self):
        self.floats_from_bits()
        self.value = self.value_func(self.floats)
        if self.fitness_func != None:
            self.fitness = self.fitness_func(self.value)

class Population(object):

    def __init__(self):
        # Population and global params
        self.pop = []
        self.popsize = 0
        self.bestidx = -1;
        # Parameters for the Items
        self.components = 0     # Number of (float) components
        self.intervals = []     # [(-2, 5), (-1, 2)] - one pair per component
        self.decimals = 3       # Desired precision - for all components
        self.value_func = None
        self.fitness_func = None
        # And parameters for the genetic part
        self.elitism = 1 # How many of the best fitness chromozomes to retain before normal selection
        self.crossoverprob = 0.7
        self.mutationprob = 0.01 # bit by bit

    def gen_random_pop(self):
        print "gen_random_pop, popsize=", self.popsize
        self.pop = []
        for i in range(0, self.popsize):
            item = Item()
            item.components = self.components
            item.intervals = self.intervals
            item.decimals = self.decimals
            item.value_func = self.value_func
            item.fitness_func = self.fitness_func
            item.setup_bits()
            item.bits = list(numpy.random.randint(2, size=item.bit_len))
            item.update_values()
            self.pop.append(item)

    def dump_pop(self):
        counter = 1
        for p in self.pop:
            print "Item", counter
            print "bits:", p.bits
            print "floats:", p.floats
            print "value:", p.value
            print "fitness:", p.fitness
            counter += 1

    def eval(self):
        for i in range(0, self.popsize):
            self.pop[i].update_values()

    def selection(self):
        #TODO elitism
        new_pop = []
        total_fitness = 0.0
        for i in range(0, self.popsize):
            total_fitness = total_fitness + self.pop[i].fitness
        for i in range(0, self.popsize):
            point = numpy.random.random()
            #print "Generated point:", point
            acc = 0.0
            for i in range(0, self.popsize):
                acc += self.pop[i].fitness
                print i, acc, acc/total_fitness
                if (acc/total_fitness) >= point:
                    # print "Selected index", i
                    # No choice, because we need multiple copies of some chromozomes
                    new_pop.append(copy.deepcopy(self.pop[i]))
                    break
        pop = new_pop

    def cross_two(self, i1, i2):
        #print "cross_two", i1, i2
        p1 = numpy.random.randint(0, i1.bit_len)
        p2 = p1
        # cross at least one bit
        while p2 == p1:
            p2 = numpy.random.randint(0, i1.bit_len)
        # Need them in left, right order
        if p2 < p1:
            tmp = p1
            p1 = p2
            p2 = tmp
        print "from", p1, "to", p2
        # Swap mid sequence
        i2_middle = i2.bits[p1:p2]
        i2.bits[p1:p2] = i1.bits[p1:p2]
        i1.bits[p1:p2] = i2_middle



    def crossover(self):
        sel_list = []
        for i in xrange(0, self.popsize):
            r = numpy.random.random()
            if r < self.crossoverprob:
                sel_list.append(i)
        print "Selection list for crossing:", sel_list
        for i in xrange(0, len(sel_list), 2):
            if i < len(sel_list) - 1:
                #print "Crossing", sel_list[i], sel_list[i+1]
                #print self.pop[sel_list[i]].bits
                #print self.pop[sel_list[i+1]].bits
                self.cross_two(self.pop[sel_list[i]], self.pop[sel_list[i+1]])
                #print "After"
                #print self.pop[sel_list[i]].bits
                #print self.pop[sel_list[i+1]].bits

    def mutation(self):
        for i in xrange(0, self.popsize):
            for j in xrange(0, self.pop[i].bit_len):
                r = numpy.random.random()
                if r < self.mutationprob:
                    print "Mutating chromosome", i, "bit", j
                    self.pop[i].bits[j] = not self.pop[i].bits[j]

#aa = Item()
#aa.comps = 2
#aa.intervals = [(-2, 2), (-2, 2)]
#aa.decimals = 0

#aa.setup_bits()
#pprint.pprint(aa.__dict__)

#aa.bits = [1, 1, 1, 1]
#aa.floats_from_bits()

#pprint.pprint(aa.__dict__)

#aa.value_func = sixhump
#aa.fitness_func = sixhump_fitness
#aa.update_values()

#pprint.pprint(aa.__dict__)

pp = Population()
pp.popsize = 10
pp.components = 2
pp.intervals = [(-3, 3), (-2, 2)]
pp.decimals = 1
pp.value_func = sixhump
pp.fitness_func = sixhump_fitness

pp.gen_random_pop()
pp.dump_pop()

pp.selection()

pp.crossover()

pp.dump_pop()
pp.mutationprob = 0.01
pp.mutation()
pp.dump_pop()
