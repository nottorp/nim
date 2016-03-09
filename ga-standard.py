#!/usr/bin/python
# coding: utf-8
import math
import numpy
import pprint
import copy

# Helper functions pulled from hill climbing
def bits_for_precision(start, end, q):
    #print "bits_for_precision", start, end, q
    return int(math.ceil(math.log((end - start) * math.pow(10, q), 2)))

def gen_bin_list(bits):
    return list(numpy.random.randint(2, size=bits))

def bin_list_to_int(bits):
    acc = 0
    for bit in bits:
        acc = acc * 2
        acc = acc + bit
    return acc

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

def rastrigin2(list):
    n = len(list)
    res = 10 * n + sum( (x ** 2 - 10 * math.cos(2 * math.pi * x)) for x in list )
    return res

def rastrigin2_fitness(val):
    return 1/(val + 0.0000001)

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
        # print self.intervals, self.bits_per_comp
        for i in zip(self.intervals, self.bits_per_comp):
            start = i[0][0]
            end = i[0][1]
            bpc = i[1]
            # print start, end, bpc
            intval = bin_list_to_int(self.bits[pos:(pos + bpc)])
            floatval = float(intval) / ((2 ** bpc) - 1) * (end - start) + start
            self.floats.append(floatval)
            pos = pos + bpc

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
        self.bestfitness = 0
        self.bestitem = None
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
        self.hybridprob = 0.1
        self.maxnochangeiter = 10 # When the best fitness doesn't change for this many iterations, stop
        self.itercount = 0
        self.nofitchangecount = 0
        self.bestiter = 0

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

    def dump_one(self, item):
        print "bits:", item.bits
        print "floats:", item.floats
        print "value:", item.value
        print "fitness:", item.fitness


    def dump_pop(self):
        counter = 1
        for p in self.pop:
            print "Item", counter
            self.dump_one(p)
            counter += 1

    def eval(self):
        for i in range(0, self.popsize):
            self.pop[i].update_values()

    # Replaces current population with new set from selection
    def selection(self):
        #TODO elitism
        new_pop = []
        total_fitness = 0.0
        for i in range(0, self.popsize):
            total_fitness = total_fitness + self.pop[i].fitness
        self.pop.sort(key = lambda item: item.fitness, reverse = True)
        # self.dump_pop()
        for i in range(0, self.elitism):
            new_pop.append(copy.deepcopy(self.pop[i]))
        for i in range(0, self.popsize - self.elitism):
            point = numpy.random.random()
            #print "Generated point:", point
            acc = 0.0
            for i in range(0, self.popsize):
                acc += self.pop[i].fitness
                #print i, acc, acc/total_fitness
                if (acc/total_fitness) >= point:
                    # print "Selected index", i
                    # No choice, because we need multiple copies of some chromozomes
                    new_pop.append(copy.deepcopy(self.pop[i]))
                    break
        pop = new_pop

    # Crosses two chromosomes on random positions
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
        #print "from", p1, "to", p2
        # Swap mid sequence
        i2_middle = i2.bits[p1:p2]
        i2.bits[p1:p2] = i1.bits[p1:p2]
        i1.bits[p1:p2] = i2_middle

    # Modifies current population
    def crossover(self):
        sel_list = []
        for i in xrange(self.elitism, self.popsize):
            r = numpy.random.random()
            if r < self.crossoverprob:
                sel_list.append(i)
        #print "Selection list for crossing:", sel_list
        for i in xrange(0, len(sel_list), 2):
            if i < len(sel_list) - 1:
                #print "Crossing", sel_list[i], sel_list[i+1]
                #print self.pop[sel_list[i]].bits
                #print self.pop[sel_list[i+1]].bits
                self.cross_two(self.pop[sel_list[i]], self.pop[sel_list[i+1]])
                #print "After"
                #print self.pop[sel_list[i]].bits
                #print self.pop[sel_list[i+1]].bits

    # Mutates current population - walks bits, probability calculated for each bit
    def mutation(self):
        for i in xrange(self.elitism, self.popsize):
            for j in xrange(0, self.pop[i].bit_len):
                r = numpy.random.random()
                if r < self.mutationprob:
                    #print "Mutating chromosome", i, "bit", j
                    self.pop[i].bits[j] = 0 if self.pop[i].bits[j] == 1 else 1

    # Hillclimb, one iteration, best improvement
    def hillclimb_one(self, i):
        tmp = None
        for n in all_neighbors(self.pop[i].bits):
            aa = copy.deepcopy(pp.pop[0])
            aa.bits = n
            aa.update_values()
            if aa.fitness > self.pop[i].fitness:
                if tmp <> None:
                    if aa.fitness > tmp.fitness:
                        tmp = aa
        if tmp <> None:
            self.pop[i] = tmp

    def hillclimb(self):
        for i in xrange(self.elitism, self.popsize):
            if numpy.random.random() < self.hybridprob:
                self.hillclimb_one(i)

    # Assumes fitness has been calculated
    def best_fitness(self):
        curbestfitness = 0
        curbestidx = -1
        for i in xrange(0, self.popsize):
            if curbestfitness < self.pop[i].fitness:
                curbestfitness = self.pop[i].fitness
                curbestidx = i
        return (curbestfitness, curbestidx)

    # Assumes pop is filled in, including floats and fitness
    def ga_step(self):
        self.selection()
        self.crossover()
        self.mutation()
        if self.hybridprob > 0:
            self.hillclimb()
        self.eval()
        (cbf, cbi) = self.best_fitness()
        self.itercount += 1
        print "Iteration", self.itercount
        print "Current best fitness:", cbf, "for chromosome"
        self.dump_one(self.pop[cbi])
        oldbestfitness = self.bestfitness
        if cbf > self.bestfitness:
            print "FOUND NEW BEST FITNESS!"
            self.bestfitness = cbf
            self.bestitem = copy.deepcopy(self.pop[cbi])
            self.bestiter = self.itercount
            self.nofitchangecount = 0
        else:
            self.nofitchangecount += 1
        if self.maxnochangeiter > 0:
            if self.nofitchangecount > self.maxnochangeiter:
                return True
        return False



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
pp.popsize = 100
pp.components = 2
pp.decimals = 3

pp.elitism = 1 # How many of the best fitness chromozomes to retain before normal selection
pp.crossoverprob = 0.7
pp.mutationprob = 0.01 # bit by bit
pp.hybridprob = 0.1
pp.maxnochangeiter = 25

# Six hump
pp.intervals = [(-3, 3), (-2, 2)]
pp.value_func = sixhump
pp.fitness_func = sixhump_fitness
# Rastrigin
#pp.intervals = [(-3, 3), (-3, 3)]
#pp.value_func = rastrigin2
#pp.fitness_func = rastrigin2_fitness


#pp.gen_random_pop()

#print "Starting from:", pp.pop[0].bits, pp.pop[0].floats, pp.pop[0].value
#for n in all_neighbors(pp.pop[0].bits):
#    aa = copy.deepcopy(pp.pop[0])
#    aa.bits = n
#    aa.update_values()
#    print "N:", aa.bits, aa.floats, aa.value

pp.gen_random_pop()
pp.eval()
#pp.dump_pop()
#pp.selection()
#pp.dump_pop()
#
for i in xrange(0, 100):
    print "*****************************************************"
    if pp.ga_step():
        print "Decided to stop after ", pp.itercount, "iterations"
        break

print "Best value found", pp.bestitem.floats, pp.bestitem.value, pp.bestitem.fitness
print "On iteration", pp.bestiter

#pp.dump_pop()

#pp.selection()

#pp.crossover()

#pp.dump_pop()
#pp.mutationprob = 0.01
#pp.mutation()
#pp.dump_pop()
