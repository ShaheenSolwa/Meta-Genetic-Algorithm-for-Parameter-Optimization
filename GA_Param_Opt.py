import numpy as np, random
from GA_FINAL import GeneticAlgorithm


def initialize_Population(numberOfParents, M):
    popSize = np.empty([numberOfParents, 1], dtype=np.uint8)
    K = np.empty([numberOfParents,1], dtype=np.uint8)
    Pm = np.empty([numberOfParents,1])
    R = np.empty([numberOfParents,1])
    generations = np.empty([numberOfParents,1])

    for i in range(numberOfParents):
        popSize[i] = round(random.randint(2,20))
        K[i] = random.randrange(2, 10)
        Pm[i] = random.random()
        R[i] = random.randint(3,7)
        generations[i] = random.randint(1,5)

    population = np.concatenate((popSize, K, Pm, R, generations), axis=1)

    return population


#bring fitness here         ---->      DO NOT FORGET
#bring the entire GA here   ---->      DO NOT FORGET
def train_population(map,mapper,maps,population):
    fit = []
    for i in range(population.shape[0]):
        param = {
            'popSize': population[i][0],
            'K': population[i][1],
            'Pm': population[i][2],
            'R': population[i][3],
            'gen': population[i][4]
        }
        print(param)

        GATrain = GeneticAlgorithm(map=map,maps=maps,mapper=mapper,popSize=int(param['popSize']),k=int(param['K']), R=param['R'], Pm=param['Pm'],generations=int(param['gen']))
        for i in range(len(GATrain)):
            fit.append(GATrain[i][1])
    return fit

def new_parent_selection(population, fitness, numParents):
    selectedParents = np.empty((numParents, population.shape[1]))

    for parentId in range(numParents):
        bestFitnessId = np.where(fitness == np.max(fitness))
        bestFitnessId = bestFitnessId[0][0]
        selectedParents[parentId, :] = population[bestFitnessId, :]
        fitness[bestFitnessId] = -1

    return selectedParents


def crossover_uniform(parents, childrenSize):
    crossoverPointIndex = np.arange(0, np.uint8(childrenSize[1]), 1, dtype=np.uint8)
    crossoverPointIndex1 = np.random.randint(0, np.uint8(childrenSize[1]), np.uint8(childrenSize[1]/2))
    crossoverPointIndex2 = np.array(list(set(crossoverPointIndex)- set(crossoverPointIndex1)))

    children = np.empty(childrenSize)

    for i in range(childrenSize[0]):
        parent1_index = i % parents.shape[0]
        parent2_index = (i+1) % parents.shape[0]

        children[i, crossoverPointIndex1] = parents[parent1_index, crossoverPointIndex1]
        children[i, crossoverPointIndex2] = parents[parent2_index, crossoverPointIndex2]

    return children


def mutation(crossover, numberofParameters, M):
        minMaxValue = np.zeros((numberofParameters, 2))

        minMaxValue[0:] = [2, 8]  # popSize
        minMaxValue[1, :] = [2, 10]  # K
        minMaxValue[2, :] = [0.1, 1]  # Pm
        minMaxValue[3, :] = [0.1,1]  # R
        minMaxValue[4, :] = [1,5]  # generations

        mutationValue = 0
        parameterSelect = np.random.randint(0, 4, 1)
        if parameterSelect == 0:  # popSize
            mutationValue = round(np.random.uniform(2, 8))
        if parameterSelect == 1:  # K
            mutationValue = np.random.randint(2, 10)
        if parameterSelect == 2:  # Pm
            mutationValue = np.random.uniform(0.1, 1)
        if parameterSelect == 3:  # R
            mutationValue = np.random.uniform(3,7)
        if parameterSelect == 4:  # Generations
            mutationValue = np.random.uniform(1,5)

        for idx in range(crossover.shape[0]):
            crossover[idx, parameterSelect] = crossover[idx, parameterSelect] + mutationValue
            if (crossover[idx, parameterSelect] > minMaxValue[parameterSelect, 1]):
                crossover[idx, parameterSelect] = minMaxValue[parameterSelect, 1]
            if (crossover[idx, parameterSelect] < minMaxValue[parameterSelect, 0]):
                crossover[idx, parameterSelect] = minMaxValue[parameterSelect, 0]
        return crossover


M = 64
map = {
    0:complex(-7,7), 1:complex(-7,5), 2:complex(-7,1), 3:complex(-7,3), 4:complex(-7,-7), 5:complex(-7,-5), 6:complex(-7,-1), 7:complex(-7,-3),
    8:complex(-5,7), 9:complex(-5,5), 10:complex(-5,1), 11:complex(-5,3), 12:complex(-5,-7), 13:complex(-5,-5), 14:complex(-5,-1), 15:complex(-5,-3),
    16:complex(-1,7), 17:complex(-1,5), 18:complex(-1,1), 19:complex(-1,3), 20:complex(-1,-7), 21:complex(-1,-5), 22:complex(-1,-1), 23:complex(-1,-3),
    24:complex(-3,7), 25:complex(-3,5), 26:complex(-3,1), 27:complex(-3,3), 28:complex(-3,-7), 29:complex(-3,-5), 30:complex(-3,-1), 31:complex(-3,-3),
    32:complex(7,7), 33:complex(7,5), 34:complex(7,1), 35:complex(7,3), 36:complex(7,-7), 37:complex(7,-5), 38:complex(7,-1), 39:complex(7,-3),
    40:complex(5,7), 41:complex(5,5), 42:complex(5,1), 43:complex(5,3), 44:complex(5,-7), 45:complex(5,-5), 46:complex(5,-1), 47:complex(5,-3),
    48:complex(1,7), 49:complex(1,5), 50:complex(1,1), 51:complex(1,3), 52:complex(1,-7), 53:complex(1,-5), 54:complex(1,-1), 55:complex(1,-3),
    56:complex(3,7), 57:complex(3,5), 58:complex(3,1), 59:complex(3,3), 60:complex(3,-7), 61:complex(3,-5), 62:complex(3,-1), 63:complex(3,-3)
}


mapper1 = 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20, \
          21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40, \
          41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60, \
          61,62,63

mapper2 = 0,55,52,3,50,5,6,49,62,9,10,61,12,59,56,15,38,17,18,37,20,35,32,23,24,47,44,27,42,29,30,41, \
    22,33,34,21,36,19,16,39,40,31,28,43,26,45,46,25,48,7,51,4,2,53,54,1,14,57,58,13,60,11,8,63

mapper3 = 18,21,16,19,22,17,20,23,42,45,40,43,46,41,44,47,2,5,0,3,6,8,4,7,26,29,24,27,30,25,28,31, \
          50,53,48,51,54,49,52,55,10,13,1,11,14,9,12,15,34,37,32,35,38,33,36,39,58,61,56,59,62,57,60,63


maps = [mapper1, mapper2,mapper3]


numberOfParents = 10
numberOfParentsMating = 10
numberOfParameters = 5
numberOfGenerations = 10

populationSize = (numberOfParents, numberOfParameters)
population = initialize_Population(numberOfParents, M)
fitnessHistory = np.empty([numberOfGenerations+1, numberOfParents])
populationHistory = np.empty([(numberOfGenerations+1)*numberOfParents, numberOfParameters])
populationHistory[0:numberOfParents, :] = population


for generation in range(numberOfGenerations):
    print("This is number %s generation" % (generation))
    fitnessValue = train_population(map,mapper1,maps,population)[:numberOfParents]
    print(fitnessValue)
    fitnessHistory[generation, :] = fitnessValue
    print("Best Fitness: ", fitnessHistory[generation, :])
    print("Best Fitness Index: ", np.argmax(fitnessValue))
    parents = new_parent_selection(population,fitnessValue,numberOfParentsMating)
    children = crossover_uniform(parents, (populationSize[0] - parents.shape[0], numberOfParameters))
    children_mutated = mutation(children,numberOfParameters,M)
    population[0:parents.shape[0], :] = parents  # fittest parents
    population[parents.shape[0]:, :] = children_mutated  # children

    populationHistory[(generation+1) * numberOfParents: (generation+1) * numberOfParents + numberOfParents, :] = population



