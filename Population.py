import copy
import math
from random import Random
from operator import attrgetter
import numpy as np
import main
from Individual import *
import multiprocessing as mp
from functools import partial


class Population:
    """
    Population
    """
    uniprng = None
    crossoverFraction = None
    individualType=None

    def __init__(self, populationSize, minmax=0):
        """
        Population constructor
        """
        self.population = []
        self.minmax = minmax
        for i in range(populationSize):
            self.population.append(self.__class__.individualType())

    def __len__(self):
        return len(self.population)

    def __getitem__(self, key):
        return self.population[key]

    def __setitem__(self, key, newValue):
        self.population[key] = newValue

    def copy(self):
        return copy.deepcopy(self)

    def evaluateFitness(self):
        print('  Evaluating...')
        for individual in self.population:
            individual.evaluateFitness()
            # print('ind: ', individual.x)
            # print('fit: ', individual.fit)

        print('  Evaluation done!')

    def ind_fitness(self, individual):
        individual.evaluateFitness()
        return individual

    def mutate(self):
        '''inds = []
        states = [ind.x for ind in inds]
        fitnesses = procPool.map(self.__class__.individualType.mutate, states)
        for i in range(len(inds)): inds[i].fit = fitnesses[i]'''
        print('  Mutating...')
        for individual in self.population:
            individual.mutate()
        print('  Mutation done!')

    def crossover(self):
        print('  Crossing...')
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
        print('  Crossover done!')

    def conductTournament(self):
        # binary tournament
        print('  Conducting binary tournament...')
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
            for index1, index2 in zip(indexList1, indexList2):
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
        print('  Binary tournament done!')

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


class PopulationMP(Population):
    uniprng = None
    crossoverFraction = None
    CORE_RESERVE = None
    CHUNKSIZE = None

    def __init__(self, populationSize, minmax=0):
        super().__init__(populationSize, minmax)

    def __len__(self):
        #super().__len__()
        return len(self.population)

    def __getitem__(self, key):
        #super().__getitem__(key)
        return self.population[key]

    def __setitem__(self, key, newValue):
        #super().__setitem__(key, newValue)
        self.population[key] = newValue

    def copy(self):
        super().copy()

    def evaluateFitness(self, procPool):
        if procPool:
            print('  Evaluating...')
            inds = []
            # only need to eval individuals whose states actually changed
            for individual in self.population:
                if individual.fit == None:
                    inds.append(individual)

            # compute all individual fitnesses using parallel process Pool
            #
            states = [ind.x for ind in inds]
            fitnesses = procPool.map(self.__class__.individualType.fitFunc, states)
            for i in range(len(inds)): inds[i].fit = fitnesses[i]
            print('  Evaluation done!')
        else:
            super().evaluateFitness()


        '''
        temp = []
        print('  Evaluating...')
        p = mp.Pool(mp.cpu_count() - self.CORE_RESERVE)
        result = p.map(self.ind_fitness, self.population, chunksize=self.CHUNKSIZE)
        temp.extend(result)
        print('  Evaluation done!')
        p.close()
        p.join()
        self.population = temp'''

    def ind_fitness(self, individual):
        individual.evaluateFitness()
        return individual

    def mutate(self, procPool = None):

        if procPool:
            print('  Mutating...')
            temp = []
            result = procPool.map(self.ind_mutate, self.population, chunksize=self.CHUNKSIZE)
            temp.extend(result)
            self.population = temp
            print('  Mutation done!')
        else:
            super().mutate()



    def ind_mutate(self, individual):
        individual.mutate()
        return individual

    def crossover(self, procPool=None):
        if procPool:
            print('  Crossing...')

            indexList1 = list(range(len(self)))
            indexList2 = list(range(len(self)))

            self.uniprng.shuffle(indexList1)
            self.uniprng.shuffle(indexList2)

            # tmp1 = [self.population[i] for i in indexList1]
            # tmp2 = [self.population[i] for i in indexList2]
            p1 = [self.population[i].x for i in indexList1]
            p2 = [self.population[i].x for i in indexList2]
            newStates = []
            if self.crossoverFraction == 1.0:
                result = procPool.map(self.__class__.individualType.crossFunc, zip(p1, p2), chunksize=self.CHUNKSIZE)
                for i in result:
                    newStates.extend(i)
            else:
                p_cross = partial(self.ind_cross, prng=self.uniprng, cf=self.crossoverFraction)
                result = procPool.map(p_cross, zip(p1, p2), chunksize=self.CHUNKSIZE)
                for i in result:
                    newStates.extend(i)

            for i in range(self.__len__()):
                self.population[i].x = newStates[i]
                self.population[i].fit = None
            print('  Cross over done!')
        else:
            super().crossover()

    def ind_cross(self, parents, prng, cf):
        rn = prng
        if rn.random() < cf:
            return self.__class__.individualType.crossFunc(parents)
        else:
            return parents

    def conductTournament(self):
        # binary tournament

        super().conductTournament()
        '''
        print('  Conducting binary tournament...')
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
        p = mp.Pool(mp.cpu_count() - CORE_RESERVE)
        result = None
        tmp1 = [self[i] for i in indexList1]
        tmp2 = [self[i] for i in indexList2]
        if self.minmax == 0:
            result = p.map(self.ind_tournament_lt, zip(tmp1, tmp2), chunksize=CHUNKSIZE)

        elif self.minmax == 1:
            result = p.map(self.ind_tournament_gt, zip(tmp1, tmp2), chunksize=CHUNKSIZE)

        newPop.extend(result)

        # overwrite old pop with newPop
        self.population = newPop
        p.close()
        p.join()
        print('  Binary tournament done!')
        '''

    def ind_tournament_lt(self, pairs):
        if pairs[0].fit > pairs[1].fit:
            return copy.deepcopy(pairs[0])
        else:  #self[indexes[0]].fit < self[indexes[1]].fit:
            return copy.deepcopy(pairs[1])


    def ind_tournament_gt(self, indexes):
        if self[indexes[0]].fit < self[indexes[1]].fit:
            return copy.deepcopy(self[indexes[0]])
        elif self[indexes[0]].fit > self[indexes[1]].fit:
            return copy.deepcopy(self[indexes[1]])
        else:
            rn = self.uniprng.random()
            if rn > 0.5:
                return copy.deepcopy(self[indexes[0]])
            else:
                return copy.deepcopy(self[indexes[1]])

    def combinePops(self, otherPop):
        super().combinePops(otherPop)

    def truncateSelect(self, newPopSize):
        super().truncateSelect(newPopSize)

    def has_none(self):
        i = 0
        j = 0
        for ind in self.population:
            if not ind.fit:
                i += 1
            else:
                j += 1
        if i:
            msg = "None detected! " + str(i)
            print('Popsize: ', self.__len__())
            print('Not none: ', j)
            print(msg)
            raise "None detected! "

    def output(self):
        print(self.population)

    def __str__(self):
        s = ''
        for ind in self:
            s += str(ind) + '\n'
        return s


if __name__ == '__main__':
    seed = 1212
    crossoverFraction = 0.8
    pop_size = 2000
    uniprng = Random()
    uniprng.seed(seed)
    normprng = Random()
    normprng.seed(seed + 101)

    FullPath.uniprng = uniprng
    FullPath.normprng = normprng
    Population.uniprng = uniprng
    Population.crossoverFraction = crossoverFraction
    Population.CORE_RESERVE = 4
    Population.CHUNKSIZE = int(pop_size/(mp.cpu_count() - Population.CORE_RESERVE))
    PopulationMP.uniprng = uniprng
    PopulationMP.crossoverFraction = crossoverFraction
    PopulationMP.CORE_RESERVE = 4
    PopulationMP.CHUNKSIZE = int(pop_size / (mp.cpu_count() - PopulationMP.CORE_RESERVE))

    minmax = 0
    p1 = PopulationMP(populationSize=pop_size, minmax=minmax)
    p1.evaluateFitness()
    #print('p1.fit ', p1.x)
    main.printStats(minmax=0, pop=p1, gen=0, lb=np.Inf)
    p1.mutate()
    p1.evaluateFitness()
    main.printStats(minmax=0, pop=p1, gen=0, lb=np.Inf)
    p1.conductTournament()
    p1.crossover()
    a = [4, 3]
    t1 = (1, 2)
    a.extend(t1)
    print(a)



