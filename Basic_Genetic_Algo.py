# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 15:10:22 2022

@author: Nazifa
"""

import pandas as pd
import numpy as np


Y = pd.read_csv('D:/Nazifa/Thesis/WindramTrainingData.csv', index_col=[0]).T.values
mData = pd.read_csv('D:/Nazifa/Thesis/WindramMetaData.csv', index_col=[0])

matrix = pd.read_csv('D:/Nazifa/Thesis/WindramTrainingData.csv', index_col=[0])
N, D = Y.shape

#print('Time Points: %s, Genes: %s'%(N, D))
mData.head()

np.random.seed(10)
sigma_t = 3.
prior_mean = mData['capture'].values[:, None]

X_mean = [prior_mean[i, 0] + sigma_t * np.random.randn(1) for i in range(0, N)]
v= type(X_mean)
#print(v)

# Python3 program to create target string, starting from
# random string using Genetic Algorithm
import random

# Number of individuals in each generation
POPULATION_SIZE = 100

# Valid genes
GENES = X_mean

# Target string to be generated
TARGET = sorted(X_mean)
v= type(TARGET)
print("X_mean", TARGET)
#print(v)
minIndex = GENES.index(min(GENES))
maxIndex = GENES.index(max(GENES))
avg = (GENES[minIndex]+ GENES[maxIndex])/2

belowAVG = [gene for gene in GENES if gene <= avg]
MoreThanAVG = [gene for gene in GENES if gene > avg]

print("avg", avg)
class Individual(object):
    '''
	Class representing individual in population
	'''
    def __init__ (self, chromosome):
        self.chromosome = chromosome
        self.fitness = self.cal_fitness()
        
    @classmethod
    def mutated_genes(self, flag):
        
            global GENES
          
            minIndex = GENES.index(min(GENES))
            maxIndex = GENES.index(max(GENES))
            avg = (GENES[minIndex]+ GENES[maxIndex])/2
           # print("avg", avg)
            belowAVG = [gene for gene in GENES if gene <= avg]
            MoreThanAVG = [gene for gene in GENES if gene > avg]

            if flag == 0:
                gene = random.choice(belowAVG)
                while not gene <= avg:
                    gene = random.choice(belowAVG)
                return gene
            elif flag == 1:
                gene = random.choice(MoreThanAVG)
                while not gene > avg:
                    gene = random.choice(MoreThanAVG)
                return gene
    
    @classmethod
    def create_gnome(self):
        '''
		create chromosome or string of genes
		'''
        global TARGET
        gnome_len = len(TARGET)
        half = 12;
        flag = 0
        firstHalf = []
        firstHalf = [self.mutated_genes(flag) for _ in range(half)]
        flag = 1
        #print("count: ",len(firstHalf))
        secondHalf = []
        secondHalf = [self.mutated_genes(flag) for _ in range(half)]
        firstHalf = firstHalf + secondHalf
        #print("count: ",len(firstHalf))

        return firstHalf
    
    def mate(self, par2):
        '''
		Perform mating and produce new offspring
		'''
        # chromosome for offspring
        child_chromosome = []
        index = 1;
        for gp1, gp2 in zip(self.chromosome, par2.chromosome):
            
            # random probability
            prob = random.random()
            
            # if prob is less than 0.45, insert gene
			# from parent 1
            if prob < 0.45:
                child_chromosome.append(gp1)
            
            # otherwise insert random gene(mutate),
			# for maintaining diversity
            elif prob < 0.09:
                child_chromosome.append(gp2)
                
            else:
                if index >= 12:
                    child_chromosome.append(self.mutated_genes(1))
                else:
                    child_chromosome.append(self.mutated_genes(0))
                    
            index += 1
            # create new Individual(offspring) using
		    # generated chromosome for offspring
#        print("\nMMchild")
#        print(child_chromosome)
#        print("\n")
#      
        child_chromosome[0:12] = list(map(lambda x: x if x <= avg else random.choice(belowAVG) , child_chromosome[0:12]))
       # print("after changing:", child_chromosome)
        child_chromosome[12:24] = list(map(lambda x: x if x > avg else random.choice(MoreThanAVG) , child_chromosome[12:24]))
       # print("after changing:", child_chromosome)
#        if len(belowAVGIn) > 0:
#            for i in belowAVGIn:    
#                gene = random.choice(belowAVG)
#                print("index:", i[0])
#                child_chromosome[i[0]] = gene
        return Individual(child_chromosome)
        
    def cal_fitness(self):
        '''
		Calculate fitness score, it is the number of
		characters in string which differ from target
		string.
		'''
        global TARGET
        fitness = 0
        #res = [self.chromosome.index(n) for m, n in
        #zip(self.chromosome, TARGET) if n != m]
#        res = [idx for idx, elem in enumerate(self.chromosome)
#                           if elem != TARGET[idx]]
#        #array3 = np.where((self.chromosome - TARGET) != 0)
       # fitness = len(res)
        #print("gnome", self.chromosome)
        for gs, gt in zip(self.chromosome, TARGET): 
            v = gs
            v2 = gt
            if v != v2: fitness+= 1
      #  print("fitness %d",fitness)
        return fitness
    
def main():
        print("started")
        global POPULATION_SIZE
        
        generation = 1
         
        found = False
        population = []
        index = 1
        for _ in range (POPULATION_SIZE):
         #   print("start")
            gnome = Individual.create_gnome()
            index += 1
          #  print(index)
#            print(_)
#            print(gnome)
#            #print(gnome[_])
#            print(type(gnome))
#            
##            print(_)
#            print("\n")
            population.append(Individual(gnome))
            
        while not found:
            population = sorted(population, key = lambda x:x.fitness)
            
            ###########
#            print("\n")
#            print(population[0].chromosome)
#            print(type(population[0]))
#            print(generation)
            ##############
            print("\nfitt")
            print(population[0].chromosome)
            print("fitness",population[0].fitness)
            if population[0].fitness <=0:
                found = True
                print(found)
                break
            
            new_generation = []
            
            s = int ((10*POPULATION_SIZE)/100) #2.4
            new_generation.extend(population[:s])
            
            s= int((90*POPULATION_SIZE)/100) ##21.6
            for _ in range(s):
                parent1 = random.choice(population[:50])
                parent2 = random.choice(population[50:])
                child = parent1.mate(parent2)
#                print(type(child))
#                print("\nparent1")
#                print(child)
#                print("\n")
                new_generation.append(child)
                
            population = new_generation
            generation+= 1
            print ("Generation:", generation)
            
           # print("Generation: {}\tString: {}\tFitness:{}".\)
           ## print("Generation: {}\tString: {}\tFitness: {}".\
           
#            print(population[0].chromosome, population[0].fitness )
            
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    