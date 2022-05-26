# Simple pygame program

# Import and initialize the pygame library
#import pygame
#pygame.init()
import sys
import numpy as np
from genetic_algorithm import *


population_size = 100





def create_2D_arr_from_input(size_of_board, coordinates_and_digits):
    arr = np.zeros( (size_of_board, size_of_board) , dtype=np.int64)
    for num in coordinates_and_digits:
        arr[int(num[0][0])-1][int(num[0][1])-1] = num[1]
    return arr

    
file_path = sys.argv[1]
#gen_algo = reg_genetic_algo(file_path)
#lemark_gen_algo = lemark_genetic_algo(file_path)
#print(lemark_gen_algo.solve())
mutation_probs=0.6
elite_lim = 35
counter =0
#gen_algo = reg_genetic_algo(file_path, mutation_probs, elite_lim, counter) 
#print(gen_algo.solve())

mutation_probs = 0.4
elite_lim = 30
counter = 0
for i in range(4):
    mutation_probs+=0.05
    elite_lim = 29
    for j in range(7):
        counter+=1
        elite_lim += 1 
        gen_algo = lemark_genetic_algo(file_path, mutation_probs, elite_lim, counter) 
        #gen_algo.solve()
        print(gen_algo.solve())


#size_of_board, coordinates_and_digits,coordinates_of_signs = get_problem_data(file_path)
#board = create_2D_arr_from_input(size_of_board, coordinates_and_digits)
#create_population(size_of_board,population_size)
#print(fitness_function(board, coordinates_of_signs))





'''
SCREEN_WIDTH = 500
SCREEN_HEIGHT= 500
# Set up the drawing window
pygame.display.set_caption('fetuccini alfredo')
screen = pygame.display.set_mode([SCREEN_WIDTH, SCREEN_HEIGHT])
number_font = pygame.font.SysFont( None, 24 )

# Run until the user asks to quit
running = True
while running:

    # Did the user click the window close button?
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Fill the background with white
    screen.fill((255, 255, 255))

    # Create a surface and pass in a tuple containing its length and width
    surf = pygame.Surface((50, 50))

    # Give the surface a color to separate it from the background
    surf.fill((0, 100, 0))
    text = '5' 
    antialias = True
    color = (70, 0, 0)
    surface = number_font.render(f"{text}", antialias, color)
    rect = surf.get_rect()
    screen.blit(surface, (SCREEN_WIDTH/2, SCREEN_HEIGHT/2))
    # Flip the display
    pygame.display.flip()

# Done! Time to quit.
pygame.quit()'''