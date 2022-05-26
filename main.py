# Simple pygame program

# Import and initialize the pygame library
#import pygame
#pygame.init()
import sys
import numpy as np
from genetic_algorithm import *


population_size = 100


# From this file we run all the different types of genetic algorithms-- regular, Darwin and Lemark

# makes the board
def create_2D_arr_from_input(size_of_board, coordinates_and_digits):
    arr = np.zeros( (size_of_board, size_of_board) , dtype=np.int64)
    for num in coordinates_and_digits:
        arr[int(num[0][0])-1][int(num[0][1])-1] = num[1]
    return arr

    
file_path = sys.argv[1]


#hyperparameters
mutation_probs = 0.4
elite_lim = 30
counter = 0
reg_gen_algo = reg_genetic_algo(file_path, mutation_probs, elite_lim, counter) 
reg_gen_algo.solve()
darwin_gen_algo = darwin_genetic_algo(file_path, mutation_probs, elite_lim, counter) 
darwin_gen_algo.solve()
lemark_gen_algo = lemark_genetic_algo(file_path, mutation_probs, elite_lim, counter) 
lemark_gen_algo.solve()






