import math
import numpy as np
import utility
import copy
from random import Random

# reuse the individual code from ev3
'''class Individual:
    """
    Individual
    """
    minSigma = 1e-100
    maxSigma = 1
    learningRate = 1
    minLimit = None
    maxLimit = None
    uniprng = None
    normprng = None
    fitFunc = None

    def __init__(self):
        self.x = self.uniprng.uniform(self.minLimit, self.maxLimit)
        self.fit = self.__class__.fitFunc(self.x)
        self.mutRate = self.uniprng.uniform(0.9, 0.1)  # use "normalized" mutRate

    def crossover(self, other):
        # perform crossover "in-place"
        alpha = self.uniprng.random()

        tmp = self.x * alpha + other.x * (1 - alpha)
        other.x = self.x * (1 - alpha) + other.x * alpha
        self.x = tmp

        self.fit = None
        other.fit = None

    def mutate(self):
        self.mutRate = self.mutRate * math.exp(self.learningRate * self.normprng.normalvariate(0, 1))
        if self.mutRate < self.minSigma:
            self.mutRate = self.minSigma
        if self.mutRate > self.maxSigma:
            self.mutRate = self.maxSigma

        self.x = self.x + (self.maxLimit - self.minLimit) * self.mutRate * self.normprng.normalvariate(0, 1)
        self.fit = None

    def evaluateFitness(self):
        if self.fit == None:
            self.fit = self.fitFunc(self.x)

    def __str__(self):
        return '%0.8e' % self.x + '\t' + '%0.8e' % self.fit + '\t' + '%0.8e' % self.mutRate
'''


class Individual:
    """
    Individual
    """
    minMutRate = 1e-100
    maxMutRate = 1
    learningRate = None
    uniprng = None
    normprng = None
    fitFunc = None

    def __init__(self):
        self.fit = self.__class__.fitFunc(self.x)
        self.mutRate = self.uniprng.uniform(0.9, 0.1)  # use "normalized" mutRate

    def mutateMutRate(self):
        self.mutRate = self.mutRate * math.exp(self.learningRate * self.normprng.normalvariate(0, 1))
        if self.mutRate < self.minMutRate: self.mutRate = self.minMutRate
        if self.mutRate > self.maxMutRate: self.mutRate = self.maxMutRate

    def evaluateFitness(self):
        if self.fit == None: self.fit = self.__class__.fitFunc(self.x)


MB_INFO = utility.MotherBoardInput('mother_board.png', 'rectangles.json').info_extraction()
RECT_LIST = MB_INFO[0]
GLUE_WIDTH = MB_INFO[1]
PATH_TOOL = utility.PathToolBox(RECT_LIST, GLUE_WIDTH, MB_INFO[2])


class FullPath(Individual):
    minMutRate = 5e-4
    maxMutRate = 1
    learningRate = 1
    uniprng = Random()
    normprng = Random()
    costType = None
    fitFunc = None
    crossFunc = None

    def __init__(self):
        # super().__init__()
        # self.length = 5
        self.mutRate = self.uniprng.uniform(0.9, 0.1)  # use "normalized" mutRate
        self.length = len(PATH_TOOL.target_regions)
        self.fitFunc = cost_func
        self.x = np.array([Rectangle(r) for r in range(self.length)])
        self.fit = None
        self.corner_initialize()
        self.uniprng.shuffle(self.x)
        self.evaluateFitness()

    def corner_initialize(self):
        np.vectorize(self.set_corner_pair)(self.x)

    def set_corner_pair(self, r):
        r.i = self.uniprng.randint(0, 3)
        r.o = PATH_TOOL.get_outcorner(r.rect, r.i)

    def mutate(self):
        super().mutateMutRate()
        mutated = False
        if self.mutRate / 5 > self.uniprng.random():
            self.uniprng.shuffle(self.x)

        elif self.mutRate > self.uniprng.random():
            t = self.uniprng.choice([1, 1, 1, 1, 2, 2, 2, 3, 3, 4])
            index_list = [i for i in range(self.length)]
            while t > 0:
                interval = get_interval(self.length, self.uniprng)
                tmp_list = [index_list[:interval[0]], index_list[interval[0]:interval[1]], index_list[interval[1]:]]
                self.uniprng.shuffle(tmp_list)
                index_list = []
                for intv in tmp_list:
                    index_list.extend(intv)
                t -= 1
            tmpind = []
            for index in index_list:
                tmpind.append(copy.deepcopy(self.x[index]))
            self.x = np.array(tmpind)
            mutated = True

        if self.mutRate > self.uniprng.random() or self.mutRate >= 0.5:
            self.corner_initialize()
            mutated = True
        else:
            rate = self.mutRate
            for r in self.x:
                if rate * 2 > self.uniprng.random():
                    self.set_corner_pair(r)
                    mutated = True

        if mutated:
            self.fit = None
        # self.evaluateFitness()

    def crossover(self, other):
        parents = (copy.deepcopy(self.x), copy.deepcopy(other.x))
        self.x, other.x = crossing(parents, self.uniprng)
        self.fit = None
        other.fit = None

        # self.mutate()
        # other.mutate()

        # return copy.deepcopy(self), copy.deepcopy(other)

    def evaluateFitness(self):
        if self.fit == None:
            self.fit = self.fitFunc(self.x, self.costType)

    def __str__(self):
        output = ''
        for rect in self.x:
            output += rect.__str__()
        return output


# serial number of rectangles
class Rectangle:
    def __init__(self, rect_num):
        self.rect = rect_num
        self.i = None
        self.o = None

    def __str__(self):
        return str(self.rect) + ': ' + str((self.i, self.o)) + ', '

    def __eq__(self, other):
        if self.rect == other.rect:
            return True
        else:
            return False


# 起點、終點的cost怎麼算？
def cost_func(path, cost_type):
    total = 0
    if (len(path)):
        last_r = path[0]
        if cost_type:
            for this_r in path:
                total += PATH_TOOL.dist_max(RECT_LIST[last_r.rect][last_r.o], RECT_LIST[this_r.rect][this_r.i])
                total += PATH_TOOL.dist_max(RECT_LIST[this_r.rect][this_r.i], RECT_LIST[this_r.rect][this_r.o])
                last_r = this_r
        else:
            for this_r in path:
                total += PATH_TOOL.dist_euler(RECT_LIST[last_r.rect][last_r.o], RECT_LIST[this_r.rect][this_r.i])
                total += PATH_TOOL.dist_euler(RECT_LIST[this_r.rect][this_r.i], RECT_LIST[this_r.rect][this_r.o])
                last_r = this_r
    return total

def crossing(parents, prng=None):
    p1x = parents[0]
    p2x = parents[1]
    length = len(p1x)
    interval = get_interval(length, prng)
    child1 = np.full((length,), Rectangle(-1))
    child2 = np.full((length,), Rectangle(-1))
    remain1 = np.array([i for i in range(length)])
    remain2 = np.array([i for i in range(length)])

    p1_p = p1x[interval[0]:interval[1]]
    p2_p = p2x[interval[0]:interval[1]]

    #
    for i in range(interval[0], interval[1]):
        remain1 = np.delete(remain1, index_of(remain1, i)[0])
        remain2 = np.delete(remain2, index_of(remain2, i)[0])
        child1[i] = copy.copy(p1x[i])
        child2[i] = copy.copy(p2x[i])

    # find i&j
    ij_pairs1 = []
    result1 = np.in1d(p2_p, p1_p)
    for index in range(len(result1)):
        if not result1[index]:
            ij_pairs1.append((p2_p[index], p1_p[index]))
    ij_pairs2 = []
    result2 = np.in1d(p1_p, p2_p)
    for index in range(len(result2)):
        if not result2[index]:
            ij_pairs2.append((p1_p[index], p2_p[index]))

    for ij in ij_pairs1:
        k = index_of(p2x, ij[1])
        while (interval[0] <= k < interval[1]):
            k = index_of(p2x, p1x[k])

        try:
            remain1 = np.delete(remain1, index_of(remain1, k)[0])
        except Exception as e:
            print(e.with_traceback())
        child1[k] = ij[0]
    for i in remain1:
        child1[i] = p2x[i]

    for ij in ij_pairs2:
        k = index_of(p1x, ij[1])
        while (interval[0] <= k < interval[1]):
            k = index_of(p1x, p2x[k])
        try:
            remain2 = np.delete(remain2, index_of(remain2, k)[0])
        except Exception as e:
            print(e.with_traceback())
        child2[k] = ij[0]
    for i in remain2:
        child2[i] = p1x[i]

    return child1, child2

def index_of(arr, e):
    return np.where(arr == e)[0]


def get_interval(len, prng=None):
    if prng:
        i1 = prng.randint(0, len - 2)
        i2 = prng.randint(i1 + 1, len)
    else:
        i1 = np.random.randint(0, len - 2)
        i2 = np.random.randint(i1 + 1, len)
    while (i2 - i1) < 7 or (i2 - i1) == (len - 6):
        # print('Too narrow!', (i1, i2))
        return get_interval(len, prng)
    else:
        return [i1, i2]


if __name__ == '__main__':
    for i in range(20):
        print(get_interval(34))

    '''print('p1.x:', p1, p1.fit)
    print('p2.x:', p2, p2.fit)
    print()

    p2.crossover(p1)
    p2.evaluateFitness()
    p1.evaluateFitness()
    print('After crossing:')
    print('p1.x:', p1, p1.fit)
    print('p2: ', p2, p2.fit)
    print()

    p2.mutate()
    p2.evaluateFitness()
    print('After mutating:')
    p2.evaluateFitness()
    print(p2, p2.fit)'''
