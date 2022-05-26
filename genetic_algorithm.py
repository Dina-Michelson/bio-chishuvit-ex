import abc
from abc import ABC
from random import randint
import numpy as np
from config import *
import random
import matplotlib.pyplot as plt

# returns a random line so that we can use it to generate a random generation
def random_line(size_of_board):
    line = []
    for j in range(size_of_board):
        line.append(j+1)
        line = random.shuffle(line)
    return line

def decision(probability):
    return random.random() < probability

# random solution to be used initially
def random_Solution(size_of_board):
    return np.random.randint(1, size_of_board+1, size=(size_of_board, size_of_board))

class GeneticAlgorithm(ABC):
        def __init__(self, path, mutation_probs, elite_lim, counter):
            self.mutation_probs=mutation_probs  # hyperparameter
            self.elite_lim =elite_lim  #hyperparameter
            self.counter=counter
            self.size_of_board, self.coordinates_and_digits,self.coordinates_of_signs = self.get_problem_data(path)
            self.gen = self.first_generation()
            self.best_fitness = []
            self.worst_fitness = -1
            self.avg_fitness = []
            self.sum_fitness = 0
            self.index_max_sol = -1
            self.bests_solution = []
            
        
            
        # create the first generation-- random
        def first_generation(self):
            gen = []
            for i in range(POPULATION):
                gen.append(random_Solution(self.size_of_board))
            return gen
    
        # In this function we get the data based on the given input file
        def get_problem_data(self,file_path):
            coordinates_and_digits = []
            coordinates_of_signs = []
            # read the file to get needed information
            try:
                file = open(file_path, "+r")
                size_of_board = int(file.readline())
                number_of_digits_given = int(file.readline())
                # get the location of each digit on the board
                for i in range(number_of_digits_given):            
                    coor_and_val =  file.readline()            
                    coor_and_val_split = coor_and_val.split(' ')
                    coordinates_and_digits.append(((coor_and_val_split[0], coor_and_val_split[1]), coor_and_val_split[2]))
                number_of_signs = int(file.readline())
                # get the location of each <,> sign on the board
                for i in range(number_of_signs):
                    coor_and_coor =  file.readline()
                    coor_and_coor_split = coor_and_coor.split(' ')
                    coor1 = (coor_and_coor_split[0], coor_and_coor_split[1])
                    coor2 = (coor_and_coor_split[2], coor_and_coor_split[3])
                    coordinates_of_signs.append((coor1, coor2))
                return size_of_board, coordinates_and_digits,coordinates_of_signs 
            except:
                print("File not found!!!")

        # function to create a mutation in the genetic algorithms, we mutate the lines only with a certain probability
        def mutate_line(self, line):
            for i in range(len(line)):
                if decision(0.1):
                    line[i] = randint(1, len(line)-1) 
            return np.array(line)
                        
        def mutation(self, sol):
            mutant = []
            for row in sol:
                mutant.append(self.mutate_line(row))
            mutant = self.leave_given_digits_in_place(mutant)
            return np.array(mutant)

        # crossover between the entire population to make the next solution.
        # Better solutions have a higher probability.
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

        # checks how many mistakes per line- we require each line to contain every digit
        def line_fitness(self, line):
            fitness = 0
            for i in range(len(line)):                   
                if i+1  not in line:
                    fitness = fitness + 1
            return fitness/self.size_of_board
            
        # checks how many mistakes on the board- each digit must be in each row/column and must follow the <,> rules
        def fitness_function(self, arr_solution, coordinates_of_signs):
            fitness = 0
            # checking that each digit is present in each line
            for line in arr_solution:                
                for i in range(len(line)):                   
                    if i+1  not in line:
                        fitness = fitness + 1
            for line in arr_solution.T:
                    for i in range(len(line)):
                        if i+1  not in line:
                            fitness = fitness + 1
            # checks that the signs are being followed properly
            for sign in coordinates_of_signs:
                if arr_solution[int(sign[0][0])-1][int(sign[0][1])-1] < arr_solution[int(sign[1][0])-1][int(sign[1][1])-1]:
                    fitness = fitness + 1
            return fitness

        # return the 50 solutions with the least mistakes
        def lowest_50_fitnesses(self,generation):
            lowest_fitnesses = []
            # sort all solutions according to fitness and return the 50 lowest
            for gen in generation:
                lowest_fitnesses.append([gen,self.fitness_function(gen, self.coordinates_of_signs)])
            lowest_fitnesses = sorted(lowest_fitnesses, key=lambda x:x[1] )
            #lowest_fitnesses = lowest_fitnesses[0:50]
            #higher_fitness = lowest_fitnesses[50:99]
            return lowest_fitnesses

        # We are given certain digits initially that we can't move.
        # During randomization we might mess them up, so we want to make sure to keep their original places.
        def leave_given_digits_in_place(self, sol):
            for num in self.coordinates_and_digits:
                sol[int(num[0][0])-1][int(num[0][1])-1] = num[1]
            return sol

        # generates a board based on user input
        def create_2D_arr_from_input(size_of_board, coordinates_and_digits):
            arr = np.zeros( (size_of_board, size_of_board) , dtype=np.int64)
            for num in coordinates_and_digits:
                arr[int(num[0][0])-1][int(num[0][1])-1] = num[1]
            return arr

        # generates a new generation to see a pool of new solutions
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

        # find the solution if the board it solved-- if the fitness is zero (no errors)
        def find_if_solved(self,new_gen):
            if self.fitness_function(new_gen, self.coordinates_of_signs) == 0:
                return self.fitness(new_gen, self.coordinates_of_signs)
            return self.fitness(new_gen, self.coordinates_of_signs)

        # generates a graph with best, worst and average solution of each generation
        # has the hyperparameters stated on top
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
            plt.savefig('reg graph 5 tricky' + str(self.counter)+'.png')
            plt.close('all')

        # Generates generations and iterates through to find the maximal solution.
        # We found that the ideal number of iterations was 140 to find maximal solution in limited time.
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
            temp_min_sol = 0
            temp_min_sol2 =0
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
                        print('found correct solution with regular genetic algorithm')
                        print(gen)
                        self.make_graph(best,worst, average_list, fitness)
                        return gen
                    average += fitness
                    if fitness < minimum:
                        minimum = min(minimum,fitness)
                        temp_min_sol = gen
                    maximum = max(maximum, fitness)
                    #avoid local minimum
                if minimum < all_time_minimum:
                    temp_min_sol2=temp_min_sol
                all_time_minimum = min(minimum,all_time_minimum)
                average_list.append(average/100)        
                best.append(minimum)
                worst.append(maximum)
                temp = new_gen
            self.make_graph(best,worst, average_list, fitness)
            print('regular minimum solution')
            print(temp_min_sol2)
            return -1
        
            
            
            
        # In this function we manually switch between cells if we see that one of the <,> rules are not being followed.
        # Once we swap between the cells, we see if the solution was improved. If so, return the new one.
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
                if counter == 5:
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

    # Like the regular genetic algorithm with a minor change- the fitness is calculated only after optimization.
    # The next generation is generated according to the solution AFTER optimization.
    def solve(self):
        minimum = 99
        maximum = 0
        average = 0
        all_time_minimum = 99
        best = []
        worst = []
        average_list = []
        minimum = 99
        temp_min_sol = 0
        temp_min_sol2 =0
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
                    print('found correct solution with lemark')
                    print(gen)
                    self.make_graph(best,worst, average_list, fitness)
                    return gen
                average += fitness
                if fitness < minimum:
                    minimum = min(minimum,fitness)
                    temp_min_sol = gen
                maximum = max(maximum, fitness)
            if minimum < all_time_minimum:
                temp_min_sol2=temp_min_sol
            all_time_minimum = min(minimum,all_time_minimum)
            average_list.append(average/100)        
            best.append(minimum)
            worst.append(maximum)
            temp = new_gen
        print('lemark minimum solution')
        print(temp_min_sol2)
        self.make_graph(best,worst, average_list, fitness)        
        return -1
    
    

    
class darwin_genetic_algo(GeneticAlgorithm):
    
    def fitness(self, arr_solution, coordinates_of_signs):
        return self.fitness_function(arr_solution, coordinates_of_signs)
    
    
    # Each solution is optimized and only then is the fitness calculated.
    # Next generation is generated according to original solution BEFORE optimization.
    def solve(self):
        minimum = 99
        maximum = 0
        average = 0
        all_time_minimum = 99
        best = []
        worst = []
        average_list = []
        minimum = 99
        temp_min_sol = 0
        temp_min_sol2 =0
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
                    print('found correct solution with darwin')
                    print(gen)
                    self.make_graph(best,worst, average_list, fitness)
                    return gen
                gen = temp
                average += fitness
                if fitness < minimum:
                    minimum = min(minimum,fitness)
                    temp_min_sol = gen
                maximum = max(maximum, fitness)
            if minimum < all_time_minimum:
                temp_min_sol2=temp_min_sol
            all_time_minimum = min(minimum,all_time_minimum)
            average_list.append(average/100)        
            best.append(minimum)
            worst.append(maximum)
            temp = new_gen
        print('darwin minimum solution')
        print(temp_min_sol2)
        self.make_graph(best,worst, average_list, fitness)        
        return -1