import copy
import math
from operator import attrgetter
from Individual import *
import multiprocessing as mp


class Population:
    """
    Population
    """
    uniprng = None
    crossoverFraction = None

    def __init__(self, populationSize, problem_num=0, minmax=0):
        """
        Population constructor
        """
        self.population = []
        self.minmax = minmax
        for i in range(populationSize):
            self.population.append(FullPath())

    def __len__(self):
        return len(self.population)

    def __getitem__(self, key):
        return self.population[key]

    def __setitem__(self, key, newValue):
        self.population[key] = newValue

    def copy(self):
        return copy.deepcopy(self)

    def evaluateFitness(self):
        '''for individual in self.population:
            individual.evaluateFitness()
            # print('ind: ', individual.x)
            # print('fit: ', individual.fit)
        '''
        p = mp.Pool(mp.cpu_count())
        p.map_async(self.ind_fitness, self.population, chunksize=2000)

    def ind_fitness(self, individual):
        individual.evaluateFitness()

    def mutate(self):
        '''for individual in self.population:
            individual.mutate()
        '''
        p = mp.Pool(mp.cpu_count())
        p.map_async(self.ind_mutate, self.population, chunksize=2000)

    def ind_mutate(self, individual):
        individual.mutate()

    def crossover(self):
        indexList1 = list(range(len(self)))
        indexList2 = list(range(len(self)))
        self.uniprng.shuffle(indexList1)
        self.uniprng.shuffle(indexList2)
        tmp = copy.deepcopy(self)

        if self.crossoverFraction == 1.0:
            for index1, index2 in zip(indexList1, indexList2):
                # self[index1].crossover(self[index2])
                self[index1].crossover(tmp[index2])
        else:
            for index1, index2 in zip(indexList1, indexList2):
                rn = self.uniprng.random()
                if rn < self.crossoverFraction:
                    # print('[index1]: ', index1, self[index1])
                    # print('[index2]: ', index2, tmp[index2])
                    # self[index1].crossover(self[index2])
                    self[index1].crossover(tmp[index2])

    def conductTournament(self):
        # binary tournament
        indexList1 = list(range(len(self)))
        indexList2 = list(range(len(self)))

        self.uniprng.shuffle(indexList1)
        self.uniprng.shuffle(indexList2)

        # do not allow self competition
        for i in range(len(self)):
            if indexList1[i] == indexList2[i]:
                temp = indexList2[i]
                if i == 0:
                    indexList2[i] = indexList2[-1]
                    indexList2[-1] = temp
                else:
                    indexList2[i] = indexList2[i - 1]
                    indexList2[i - 1] = temp

        # compete
        newPop = []
        if self.minmax == 0:
            '''for index1, index2 in zip(indexList1, indexList2):
                if self[index1].fit < self[index2].fit:
                    newPop.append(copy.deepcopy(self[index1]))
                elif self[index1].fit > self[index2].fit:
                    newPop.append(copy.deepcopy(self[index2]))
                else:
                    rn = self.uniprng.random()
                    if rn > 0.5:
                        newPop.append(copy.deepcopy(self[index1]))
                    else:
                        newPop.append(copy.deepcopy(self[index2]))
            '''
            p = mp.Pool(mp.cpu_count())
            result = p.map_async(self.ind_tournament_lt, zip(indexList1, indexList2), chunksize=2000)
            results = result.get()
            for item in results:
                if not item:
                    print('NoneType detected!')
            newPop.extend(results)
        elif self.minmax == 1:
            for index1, index2 in zip(indexList1, indexList2):
                if self[index1].fit > self[index2].fit:
                    newPop.append(copy.deepcopy(self[index1]))
                elif self[index1].fit < self[index2].fit:
                    newPop.append(copy.deepcopy(self[index2]))
                else:
                    rn = self.uniprng.random()
                    if rn > 0.5:
                        newPop.append(copy.deepcopy(self[index1]))
                    else:
                        newPop.append(copy.deepcopy(self[index2]))

        # overwrite old pop with newPop
        self.population = newPop

    def ind_tournament_lt(self, indexes):
        if self[indexes[0]].fit > self[indexes[1]].fit:
            return copy.deepcopy(self[indexes[0]])
        elif self[indexes[0]].fit < self[indexes[1]].fit:
            return copy.deepcopy(self[indexes[1]])
        else:
            rn = self.uniprng.random()
            if rn > 0.5:
                return copy.deepcopy(self[indexes[0]])
            else:
                return copy.deepcopy(self[indexes[1]])

    def combinePops(self, otherPop):
        self.population.extend(otherPop.population)

    def truncateSelect(self, newPopSize):
        # sort by fitness
        self.population.sort(key=attrgetter('fit'))

        # then truncate the bottom
        self.population = self.population[:newPopSize]

    def __str__(self):
        s = ''
        for ind in self:
            s += str(ind) + '\n'
        return s

class Population_MP(Population):
    def __init__(self):
        super().__init__()


