import array
import random
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from functools import partial
from scoop import futures

import DEAPTSP as tsp
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

TSP_NAME = "bayg29"
tsp = tsp.TravelingSalesmanProblem(TSP_NAME)


POPULATION_SIZE = 300
MAX_GENERATIONS = 200
HALL_OF_FAME_SIZE = 1
P_CROSSOVER = 0.9
P_MUTATION = 0.1


toolbox = base.Toolbox()


creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", array.array, typecode='i', fitness=creator.FitnessMin)


toolbox.register("randomOrder", random.sample, range(len(tsp)), len(tsp))
toolbox.register("individualCreator", tools.initIterate, creator.Individual, toolbox.randomOrder)
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)

def tpsDistance(individual):
    return tsp.getTotalDistance(individual),

toolbox.register("evaluate", tpsDistance)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", tools.cxOrdered)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=1.0/len(tsp))
toolbox.register("migrate", tools.migRing, k=15, selection=tools.selBest)
toolbox.register("map", futures.map)

def main():

    population = toolbox.populationCreator(n=POPULATION_SIZE)

    NISLES = 5
    FREQ = 10
    islands = [toolbox.populationCreator for i in range(NISLES)]

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)


    hof = tools.HallOfFame(HALL_OF_FAME_SIZE)


    toolbox.register("algorithm", algorithms.eaSimple(population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION,
                                              ngen=MAX_GENERATIONS, stats=stats, halloffame=hof, verbose=True))

    population, logbook = algorithms.eaSimple(population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION,
                                              ngen=MAX_GENERATIONS, stats=stats, halloffame=hof, verbose=True)

    for i in range(0,MAX_GENERATIONS, FREQ):
        results = toolbox.map(toolbox.algorithm, islands)
        islands = [island for island, logbook in results]
        toolbox.migrate(islands)

        return islands

    best = hof.items[0]
    print("-- Best Ever Individual = ", best)
    print("-- Best Ever Fitness = ", best.fitness.values[0])

    plt.figure(1)
    tsp.plotData(best)

    minFitnessValues, meanFitnessValues = logbook.select("min", "avg")
    plt.figure(2)
    sns.set_style("whitegrid")
    plt.plot(minFitnessValues, color='red')
    plt.plot(meanFitnessValues, color='green')
    plt.xlabel('Generation')
    plt.ylabel('Min / Average Fitness')
    plt.title('Min and Average fitness over Generations')

    plt.show()

if __name__ == "__main__":
    main()
