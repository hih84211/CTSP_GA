import math
import numpy as np
import utility
import copy

# reuse the individual code from ev3
class Individual:
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
        self.sigma = self.uniprng.uniform(0.9, 0.1)  # use "normalized" sigma

    def crossover(self, other):
        # perform crossover "in-place"
        alpha = self.uniprng.random()

        tmp = self.x * alpha + other.x * (1 - alpha)
        other.x = self.x * (1 - alpha) + other.x * alpha
        self.x = tmp

        self.fit = None
        other.fit = None

    def mutate(self):
        self.sigma = self.sigma * math.exp(self.learningRate * self.normprng.normalvariate(0, 1))
        if self.sigma < self.minSigma:
            self.sigma = self.minSigma
        if self.sigma > self.maxSigma:
            self.sigma = self.maxSigma

        self.x = self.x + (self.maxLimit - self.minLimit) * self.sigma * self.normprng.normalvariate(0, 1)
        self.fit = None

    def evaluateFitness(self):
        if self.fit == None:
            self.fit = self.__class__.fitFunc(self.x)

    def __str__(self):
        return '%0.8e' % self.x + '\t' + '%0.8e' % self.fit + '\t' + '%0.8e' % self.sigma


MB_INFO = utility.MotherBoardInput('mother_board.png', 'rectangles.json').info_extraction()
RECT_LIST = MB_INFO[0]
GLUE_WIDTH = MB_INFO[1]
PATH_TOOL = utility.PathToolBox(RECT_LIST, GLUE_WIDTH)

class FullPath(Individual):
    minSigma = 1e-2
    maxSigma = 1
    learningRate = 1
    uniprng = None
    normprng = None
    fitFunc = None

    def __init__(self):
        # super().__init__()
        # self.length = 5
        self.sigma = self.uniprng.uniform(0.9, 0.1)  # use "normalized" sigma
        self.length = len(RECT_LIST)
        self.fitFunc = cost_func
        self.x = np.array([Rectangle(r) for r in range(self.length)])
        self.fit = None
        self.corner_initialize()
        np.random.shuffle(self.x)
        self.evaluateFitness()

    def corner_initialize(self):
        np.vectorize(self.__set_corner_pair)(self.x)

    def __set_corner_pair(self, r):
        r.i = self.uniprng.randint(0, 3)
        r.o = PATH_TOOL.get_outcorner(r.rect, r.i)

    def mutate(self):
        self.sigma = self.sigma * math.exp(self.learningRate * self.normprng.normalvariate(0, 1))
        if self.sigma < self.minSigma:
            self.sigma = self.minSigma
        if self.sigma > self.maxSigma:
            self.sigma = self.maxSigma

        for i in range(self.length):
            if self.sigma > self.uniprng.random():
                # self.uniprng.shuffle(self.x)
                self.corner_initialize()
                if self.sigma * 10 > self.uniprng.random():
                    self.uniprng.shuffle(self.x)
                self.fit = None

    def crossover(self, other):
        remain1 = np.array([i for i in range(self.length)])
        remain2 = np.array([i for i in range(self.length)])
        interval = get_interval(self.length)

        p1 = copy.copy(self)
        p2 = copy.copy(other)
        child1 = np.full((self.length,), Rectangle(-1))
        child2 = np.full((self.length,), Rectangle(-1))

        p1_p = p1.x[interval[0]:interval[1]]
        p2_p = p2.x[interval[0]:interval[1]]

        #
        for i in range(interval[0], interval[1]):
            remain1 = np.delete(remain1, index_of(remain1, i)[0])
            remain2 = np.delete(remain2, index_of(remain2, i)[0])
            child1[i] = copy.copy(p1.x[i])
            child2[i] = copy.copy(p2.x[i])

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
            k = index_of(p2.x, ij[1])
            while (interval[0] <= k < interval[1]):
                k = index_of(p2.x, p1.x[k])

            try:
                remain1 = np.delete(remain1, index_of(remain1, k)[0])
            except Exception as e:
                print(e.with_traceback())
            child1[k] = ij[0]
        for i in remain1:
            child1[i] = p2.x[i]

        for ij in ij_pairs2:
            k = index_of(p1.x, ij[1])
            while (interval[0] <= k < interval[1]):
                k = index_of(p1.x, p2.x[k])
            try:
                remain2 = np.delete(remain2, index_of(remain2, k)[0])
            except Exception as e:
                print(e.with_traceback())
            child2[k] = ij[0]
        for i in remain2:
            child2[i] = p1.x[i]
        self.x = child1
        self.fit = None
        other.x = child2
        other.fit = None

    def evaluateFitness(self):
        if self.fit == None:
            self.fit = self.fitFunc(self.x)

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
def cost_func(path):
    total = 0
    if(len(path)):
        last_r = path[0]
        for this_r in path:
            total += PATH_TOOL.dist_euler(RECT_LIST[last_r.rect][last_r.o], RECT_LIST[this_r.rect][this_r.i])
            total += PATH_TOOL.dist_euler(RECT_LIST[this_r.rect][this_r.i], RECT_LIST[this_r.rect][this_r.o])
            last_r = this_r
    return total


def index_of(arr, e):
    return np.where(arr == e)[0]


def get_interval(len):
    i1 = np.random.randint(0, len - 2)
    i2 = np.random.randint(i1 + 1, len)
    while((i2 - i1) < 2 or (i2 - i1) == (len-1)):
        # print('Too narrow!', (i1, i2))
        return get_interval(len)
    else:
        return [i1, i2]


if __name__ == '__main__':
    from random import Random
    FullPath.fitFunc = cost_func
    FullPath.uniprng = Random()
    FullPath.normprng = Random()
    p1 = FullPath()
    p2 = FullPath()

    print('p1.x:', p1, p1.fit)
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
    print(p2, p2.fit)


