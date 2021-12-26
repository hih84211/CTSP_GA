import optparse

import numpy as np
import multiprocessing as mp
from Population import *
from Individual import  *
from random import Random
import yaml
import sys
import time
import utility

MB_INFO = utility.MotherBoardInput('mother_board.png', 'rectangles.json').info_extraction()
RECT_LIST = MB_INFO[0]
GLUE_WIDTH = MB_INFO[1]
PATH_TOOL = utility.PathToolBox(RECT_LIST, GLUE_WIDTH, MB_INFO[2])

class CTSP_Config:
    """
    Configuration class
    """
    # class variables
    sectionName = 'CTSP'
    options = {'CPUCoresPreserved': (int, True),
               'populationSize': (int, True),
               'generationCount': (int, True),
               'randomSeed': (int, True),
               'crossoverFraction': (float, True)}

    # constructor
    def __init__(self, inFileName):
        # read YAML config and get EV3 section
        infile = open(inFileName, 'r')
        ymlcfg = yaml.safe_load(infile)
        infile.close()
        eccfg = ymlcfg.get(self.sectionName, None)
        if eccfg is None:
            raise Exception('Missing {} section in cfg file'.format(self.sectionName))

        # iterate over options
        for opt in self.options:
            if opt in eccfg:
                optval = eccfg[opt]

                # verify parameter type
                if type(optval) != self.options[opt][0]:
                    raise Exception('Parameter "{}" has wrong type'.format(opt))

                # create attributes on the fly
                setattr(self, opt, optval)
            else:
                if self.options[opt][1]:
                    raise Exception('Missing mandatory parameter "{}"'.format(opt))
                else:
                    setattr(self, opt, None)

    # string representation for class data
    def __str__(self):
        return str(yaml.dump(self.__dict__, default_flow_style=False))



def CTSP_problem(cfg):
    minmax = 0
    current_best_fit = np.Inf

    start = time.process_time()
    pool = mp.Pool(initializer=initClassVars, initargs=(cfg,), processes=(mp.cpu_count() - cfg.CPUCoresPreserved))

    initClassVars(cfg)

    # create initial Population (random initialization)
    population = PopulationMP(populationSize=cfg.populationSize, minmax=minmax)
    population.evaluateFitness(pool)
    print(type(population))

    # print initial pop stats
    printStats(minmax=minmax, pop=population, gen=0, lb=current_best_fit, total_t=(time.process_time() - start))

    # evolution main loop
    for i in range(cfg.generationCount):
        tic = time.process_time()
        # create initial offspring population by copying parent pop
        offspring = copy.deepcopy(population)

        # select mating pool
        offspring.conductTournament()

        # perform crossover
        offspring.crossover(pool)

        # random mutation
        offspring.mutate()

        # update fitness values
        #print(offspring.population)
        offspring.evaluateFitness(pool)

        # survivor selection: elitist truncation using parents+offspring
        population.combinePops(offspring)
        population.truncateSelect(cfg.populationSize)

        toc = time.process_time()
        # print population stats
        current_best_fit = printStats(minmax=minmax, pop=population, gen=i + 1, tictoc=(toc - tic),
                                      total_t=(toc - start), lb=current_best_fit)


# Print some useful stats to screen
# 可由minmax參數選擇呈現最大值或最小值
def printStats(minmax, pop, gen, lb, tictoc=.0, total_t=.0):
    print('Generation:', gen)
    print('Population size: ', len(pop.population))
    avgval = 0
    mval = pop[0].fit
    sigma = pop[0].mutRate
    min_ind = pop[0]
    if minmax == 0:
        for ind in pop:
            avgval += ind.fit
            if ind.fit < mval:
                mval = ind.fit
                sigma = ind.mutRate
                min_ind = ind
        if(mval < lb):
            PATH_TOOL.path_plot(min_ind.x)
        print('Min fitness', mval)
    elif minmax == 1:
        for ind in pop:
            avgval += ind.fit
            if ind.fit > mval:
                mval = ind.fit
                sigma = ind.mutRate
            print(ind)

        print('Max fitness', mval)

    print('Sigma', sigma)
    print('Avg fitness', avgval / len(pop))
    print('Gen runtime ', tictoc)
    print('Total runtime ', total_t)
    print('')
    return mval


def main(argv=None):
    if argv is None:
        argv = sys.argv
    try:
        #
        # get command-line options
        #
        parser = optparse.OptionParser()
        parser.add_option("-i", "--input", action="store", dest="inputFileName", help="input filename", default=None)
        parser.add_option("-q", "--quiet", action="store_true", dest="quietMode", help="quiet mode", default=False)
        parser.add_option("-d", "--debug", action="store_true", dest="debugMode", help="debug mode", default=False)
        (options, args) = parser.parse_args(argv)

        # validate options
        if options.inputFileName is None:
            raise Exception("Must specify input file name using -i or --input option.")

        # Get CTSP config params
        cfg = CTSP_Config(options.inputFileName)

        # print config params
        print(cfg)

        print('Clustered-TSP path length minimization start!')
        CTSP_problem(cfg)
        if not options.quietMode:
            print('Clustered-TSP path length minimization Completed!\n')


    except Exception as info:
        if 'options' in vars() and options.debugMode:
            from traceback import print_exc
            print_exc()
        else:
            print(info)

def initClassVars(cfg):
    uniprng = Random()
    uniprng.seed(cfg.randomSeed)
    normprng = Random()
    normprng.seed(cfg.randomSeed + 101)
    FullPath.uniprng = uniprng
    FullPath.normprng = normprng
    FullPath.fitFunc = cost_func
    FullPath.crossFunc = crossing

    Population.individualType = FullPath
    Population.uniprng = uniprng
    Population.crossoverFraction = cfg.crossoverFraction
    Population.CORE_RESERVE = 4
    Population.CHUNKSIZE = int(cfg.populationSize / (mp.cpu_count() - Population.CORE_RESERVE))

    PopulationMP.CORE_RESERVE = 4
    PopulationMP.CHUNKSIZE = int(cfg.populationSize / (mp.cpu_count() - PopulationMP.CORE_RESERVE))
    PopulationMP.uniprng = uniprng
    PopulationMP.crossoverFraction = cfg.crossoverFraction


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main(['-i', 'CTSP.cfg', '-d'])

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
