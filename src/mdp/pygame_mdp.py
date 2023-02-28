import sys
import time

import pygame
from pygame.locals import QUIT

from .mdp import MDP

class PyGameMDP(MDP):
    width:int
    height:int
    display:bool
    operator:str
    surface:pygame.Surface
    game_clock:pygame.time.Clock
    font_default:pygame.font.Font

    def __init__(self, name:str, display:bool) -> None:
        super().__init__(name)

        self.display = display
        self.debounce_val = 100
        self.debounce = time.time_ns()

    def start(self) -> float:
        assert self.operator != None, "Set agent operator parameter before starting"

    def init_display(self) -> None:
        pygame.init()
        self.game_clock = pygame.time.Clock()
        self.surface = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("PyGame MDP")

    def is_quit(self) -> None:
        for event in pygame.event.get():
            if event.type == QUIT:
                self.quit()

    def quit(self) -> None:
        pygame.quit()
        sys.exit()

    def check_input(self) -> bool:
        if ((time.time_ns() - self.debounce) / 1e6) < self.debounce_val:
            return False
        else:
            self.debounce = time.time_ns()
            return True
