from deap import base
from deap import creator
from deap import tools
from deap import algorithms

import random
import array

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import DEAPTSP as tsp


RANDOM_SEED = 42
random.seed(RANDOM_SEED)

TSP_NAME = "eil51"
tsp = tsp.TravelingSalesmanProblem(TSP_NAME)


POPULATION_SIZE = 1000
MAX_GENERATIONS = 300
HALL_OF_FAME_SIZE = 1
P_CROSSOVER = 0.7
P_MUTATION = 0.3

toolbox = base.Toolbox()


creator.create("FitnessMin", base.Fitness, weights=(-1.0,))


creator.create("Individual", array.array, typecode='i', fitness=creator.FitnessMin)


toolbox.register("randomOrder", random.sample, range(len(tsp)), len(tsp))


toolbox.register("individualCreator", tools.initIterate, creator.Individual, toolbox.randomOrder)


toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)



def tpsDistance(individual):
    return tsp.getTotalDistance(individual),


toolbox.register("evaluate", tpsDistance)
toolbox.register("select", tools.selTournament, tournsize=6)
toolbox.register("mate", tools.cxUniform, indpb=1.0)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=1.0/len(tsp))





def main():


    population = toolbox.populationCreator(n=POPULATION_SIZE)


    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)


    hof = tools.HallOfFame(HALL_OF_FAME_SIZE)


    population, logbook = algorithms.eaSimple(population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION,
                                              ngen=MAX_GENERATIONS,  stats=stats, halloffame=hof, verbose=True)


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