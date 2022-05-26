import abc
from abc import ABC
from random import randint
import numpy as np
from config import *
#from gui import PyGui
import random
import matplotlib.pyplot as plt

def random_line(size_of_board):
    line = []
    for j in range(size_of_board):
        line.append(j+1)
        line = random.shuffle(line)
    return line

def decision(probability):
    return random.random() < probability


def random_Solution(size_of_board):
    return np.random.randint(1, size_of_board+1, size=(size_of_board, size_of_board))

class GeneticAlgorithm(ABC):
        def __init__(self, path, mutation_probs, elite_lim, counter):
            self.mutation_probs=mutation_probs
            self.elite_lim =elite_lim
            self.counter=counter
            self.size_of_board, self.coordinates_and_digits,self.coordinates_of_signs = self.get_problem_data(path)
            self.gen = self.first_generation()
            self.best_fitness = []
            self.worst_fitness = -1
            self.avg_fitness = []
            self.sum_fitness = 0
            self.index_max_sol = -1
            self.bests_solution = []
            
        
            
            
        def first_generation(self):
            gen = []
            for i in range(POPULATION):
                gen.append(random_Solution(self.size_of_board))
            return gen
    
        
        def get_problem_data(self,file_path):
            coordinates_and_digits = []
            coordinates_of_signs = []
            try:
                file = open(file_path, "+r")
                size_of_board = int(file.readline())
                number_of_digits_given = int(file.readline())
                for i in range(number_of_digits_given):            
                    coor_and_val =  file.readline()            
                    coor_and_val_split = coor_and_val.split(' ')
                    coordinates_and_digits.append(((coor_and_val_split[0], coor_and_val_split[1]), coor_and_val_split[2]))
                number_of_signs = int(file.readline())
                for i in range(number_of_signs):
                    coor_and_coor =  file.readline()
                    coor_and_coor_split = coor_and_coor.split(' ')
                    coor1 = (coor_and_coor_split[0], coor_and_coor_split[1])
                    coor2 = (coor_and_coor_split[2], coor_and_coor_split[3])
                    coordinates_of_signs.append((coor1, coor2))
                return size_of_board, coordinates_and_digits,coordinates_of_signs 
            except:
                print("File not found!!!")
        
        def mutate_line(self, line):
            for i in range(len(line)):
                '''if decision(0.7):
                    j = randint(0, len(line)-1)
                    temp = line[i]
                    line[i] = j
                    line[j] = temp'''
                if decision(0.1):
                    line[i] = randint(0, len(line)-1) 
            return np.array(line)
                        
        def mutation(self, sol):
            mutant = []
            for row in sol:
                mutant.append(self.mutate_line(row))
            mutant = self.leave_given_digits_in_place(mutant)
            return np.array(mutant)
        
        def crossover(self, parent1, parent2):
            new_solution = []
            limit = randint(0, len(parent1)-1) 
            for i in range(len(parent1)):
                if i < limit:
                    new_solution.append(np.array(parent1[i]))
                else:
                    new_solution.append(np.array(parent2[i]))
            new_solution = self.leave_given_digits_in_place(new_solution)        
            return np.array(new_solution)
        
        def choose_random_sol(self):
            return 6
        
        def line_fitness(self, line):
            fitness = 0
            for i in range(len(line)):                   
                if i+1  not in line:
                    fitness = fitness + 1
            return fitness/self.size_of_board
            
            
        def fitness_function(self, arr_solution, coordinates_of_signs):
            fitness = 0
            for line in arr_solution:                
                for i in range(len(line)):                   
                    if i+1  not in line:
                        fitness = fitness + 1
            for line in arr_solution.T:
                    for i in range(len(line)):
                        if i+1  not in line:
                            fitness = fitness + 1
            for sign in coordinates_of_signs:
                if arr_solution[int(sign[0][0])-1][int(sign[0][1])-1] < arr_solution[int(sign[1][0])-1][int(sign[1][1])-1]:
                    fitness = fitness + 1
            return fitness
        
        def lowest_50_fitnesses(self,generation):
            lowest_fitnesses = []
            for gen in generation:
                lowest_fitnesses.append([gen,self.fitness_function(gen, self.coordinates_of_signs)])
            lowest_fitnesses = sorted(lowest_fitnesses, key=lambda x:x[1] )
            #lowest_fitnesses = lowest_fitnesses[0:50]
            #higher_fitness = lowest_fitnesses[50:99]
            return lowest_fitnesses
            
        def leave_given_digits_in_place(self, sol):
            for num in self.coordinates_and_digits:
                sol[int(num[0][0])-1][int(num[0][1])-1] = num[1]
            return sol


        def create_2D_arr_from_input(size_of_board, coordinates_and_digits):
            arr = np.zeros( (size_of_board, size_of_board) , dtype=np.int64)
            for num in coordinates_and_digits:
                arr[int(num[0][0])-1][int(num[0][1])-1] = num[1]
            return arr
    
        def new_generation(self, generation):
            new_generation = []
            lowest_fitnesses = self.lowest_50_fitnesses(generation)           
            for gen in generation:
                if decision(0.7):
                    i = randint(0, self.elite_lim)
                    j = randint(0, self.elite_lim)
                else:
                    i = randint(0, self.elite_lim)
                    j = randint(self.elite_lim, 98)
                while i == j:
                    j = randint(0, self.elite_lim)
                new_sol = self.crossover(lowest_fitnesses[i][0], lowest_fitnesses[j][0])
                new_generation.append(new_sol)
            return new_generation
        
        def find_if_solved(self,new_gen):
            #print(new_gen)
            if self.fitness_function(new_gen, self.coordinates_of_signs) == 0:
                return self.fitness(new_gen, self.coordinates_of_signs)
            return self.fitness(new_gen, self.coordinates_of_signs)
        
        def make_graph(self, best,worst, average, fitness):
            #script for generating the graphs
            plt.ion()
            plt.rcParams["figure.figsize"] = (10,10)
            fig, ax = plt.subplots() 
            plt.title('mutation probability:' + str(self.mutation_probs) + ' elitism limit:' + str(self.elite_lim) + ' fitness:'+str(fitness))
            plt.xlabel("Generation")
            plt.ylabel("Fitness")
            plt.plot(best , 'b')
            plt.plot(worst, 'r')
            plt.plot(average, 'k')
            plt.show()
            plt.savefig('lemark 2 graph 7 tricky' + str(self.counter)+'.png')
            plt.close('all')
        
        def solve(self):
            minimum = 99
            maximum = 0
            average = 0
            all_time_minimum = 99
            gen = self.first_generation()
            temp = gen
            best = []
            worst = []
            average_list = []
            for i in range(ITERATIONS):
                new_gen = self.new_generation(temp)
                average=0
                maximum = 0
                minimum=99
                for gen in new_gen:
                    if decision(self.mutation_probs):
                        gen = self.mutation(gen)
                    fitness=self.find_if_solved(gen)
                    if fitness == 0:
                        print(gen)
                        self.make_graph(best,worst, average_list, fitness)
                        return gen
                    average += fitness
                    minimum = min(minimum,fitness)
                    maximum = max(maximum, fitness)
                    #avoid local minimum
                all_time_minimum = min(minimum,all_time_minimum)
                average_list.append(average/100)        
                best.append(minimum)
                worst.append(maximum)
                #print(minimum)
                temp = new_gen
            print('all_time_minimum '+ str(all_time_minimum))
            self.make_graph(best,worst, average_list, fitness)
            return -1
        
        def optimize(self, sol):
            fitness = self.fitness_function(sol, self.coordinates_of_signs)
            new_sol = sol
            counter = 0
            for coordinate in self.coordinates_of_signs:
                greater_cell = coordinate[0]
                cell = coordinate[1]
                if new_sol[int(greater_cell[0])-1][int(greater_cell[1])-1] <  new_sol[int(cell[0])-1][int(cell[1])-1]:
                    temp = new_sol[int(greater_cell[0])-1][int(greater_cell[1])-1] 
                    new_sol[int(greater_cell[0])-1][int(greater_cell[1])-1]  = new_sol[int(cell[0])-1][int(cell[1])-1]
                    new_sol[int(cell[0])-1][int(cell[1])-1] = temp
                    counter +=1
                if counter == 7:
                    break
                new_fitness = self.fitness_function(new_sol, self.coordinates_of_signs)
                if new_fitness < fitness:
                    sol = new_sol
            return sol
        @abc.abstractmethod
        def fitness(self):
            raise NotImplementedError


class reg_genetic_algo(GeneticAlgorithm):
    def fitness(self, arr_solution, coordinates_of_signs):
        return self.fitness_function(arr_solution, coordinates_of_signs)
    
    
    
class lemark_genetic_algo(GeneticAlgorithm):
    
    def fitness(self, arr_solution, coordinates_of_signs):
        return self.fitness_function(arr_solution, coordinates_of_signs)
    
    
    #

    
    
    def solve(self):
        minimum = 99
        maximum = 0
        average = 0
        all_time_minimum = 99
        best = []
        worst = []
        average_list = []
        minimum = 99
        gen = self.first_generation()
        temp = gen
        for i in range(ITERATIONS):
            new_gen = self.new_generation(temp)
            average=0
            maximum = 0
            minimum=99
            for gen in new_gen:
                #avoid local minimum
                if decision(self.mutation_probs):
                    gen = self.mutation(gen)
                gen = self.optimize(gen)
                fitness=self.find_if_solved(gen)
                if fitness == 0:
                    print(gen)
                    self.make_graph(best,worst, average_list, fitness)
                    return gen
                average += fitness
                minimum = min(minimum,fitness)
                maximum = max(maximum, fitness)
            all_time_minimum = min(minimum,all_time_minimum)
            average_list.append(average/100)        
            best.append(minimum)
            worst.append(maximum)

            #print(minimum)
            temp = new_gen
        print('all_time_minimum '+ str(all_time_minimum))
        self.make_graph(best,worst, average_list, fitness)        
        return -1
    
    
    
    
    
    
    
    
class darwin_genetic_algo(GeneticAlgorithm):
    
    def fitness(self, arr_solution, coordinates_of_signs):
        return self.fitness_function(arr_solution, coordinates_of_signs)
    
    
    #    
    def solve(self):
        minimum = 99
        maximum = 0
        average = 0
        all_time_minimum = 99
        best = []
        worst = []
        average_list = []
        minimum = 99
        gen = self.first_generation()
        temp = gen
        for i in range(ITERATIONS):
            new_gen = self.new_generation(temp)
            average=0
            maximum = 0
            minimum=99
            for gen in new_gen:
                #avoid local minimum
                if decision(self.mutation_probs):
                    gen = self.mutation(gen)
                temp = gen
                gen = self.optimize(gen)
                fitness=self.find_if_solved(gen)
                if fitness == 0:
                    print(gen)
                    self.make_graph(best,worst, average_list, fitness)
                    return gen
                gen = temp
                average += fitness
                minimum = min(minimum,fitness)
                maximum = max(maximum, fitness)
            all_time_minimum = min(minimum,all_time_minimum)
            average_list.append(average/100)        
            best.append(minimum)
            worst.append(maximum)

            #print(minimum)
            temp = new_gen
        print('all_time_minimum '+ str(all_time_minimum))
        self.make_graph(best,worst, average_list, fitness)        
        return -1