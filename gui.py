import pygame
import sys

'''class PyGui:
    pygame.display.set_caption('fetuccini alfredo')
    pause = False
    block_size = 20
    background_color = (255, 255, 255)
    def __init__(self, rows, cols):
        self.num_row =  rows
        self.num_col = cols
        pygame.init()
        self.TitleFont = pygame.font.Font(None, 30)
        self.TitleFont.underline = True
        self.Font = pygame.font.Font(None, 20)
        # size = (self.width, self.height) = 700, 700
        size = (self.width, self.height) = (self.num_col * self.block_size) + 200, (self.num_row * self.block_size) + 200
        self.board = pygame.display.set_mode(size)
        
    def text_objects(self, text, font):
        textSurface = font.render(text, True, (0, 0, 0), self.background_color)
        return textSurface, textSurface.get_rect()

    def update_data(self, text, data, width, height, font):
        TextSurf, TextRect = self.text_objects(text + str(data), font)
        TextRect.center = (width, height)
        self.board.blit(TextSurf, TextRect)'''
        
        


#pg = PyGui(5,5)