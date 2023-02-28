import math
from typing import Dict
from typing import Tuple

import numpy as np
import pygame
from pygame.locals import *

from .pygame_mdp import PyGameMDP
from model import Actor
from model import StepResult
from util.constants import *

class GridTargetMDP(PyGameMDP):
    def __init__(self,
                 fps:int,
                 width:int,
                 height:int,
                 grid_dim:int,
                 agent_position:Tuple[int, int],
                 target_position:Tuple[int, int],
                 display:bool,
                 values:bool,
                 trail:bool) -> None:
        super().__init__(GRID_TARGET_MDP, display)

        self.n_state = grid_dim ** 2
        self.n_action = 4 # NORTH, EAST, SOUTH, WEST
        self.d_state = (grid_dim, grid_dim)

        self.fps = fps
        self.width = width
        self.height = height
        self.show_trail = trail
        self.show_values = values
        self.grid_dim = grid_dim
        self.agent_start_position = agent_position
        self.target_start_position = target_position
        self.cell_size = (round(self.width/self.grid_dim), round(self.height/self.grid_dim))

        self.agent = None
        self.target = None
        self.values = None
        self.operator = None

        self.agent_image = pygame.image.load("./assets/agent.png")
        self.target_image = pygame.image.load("./assets/star.png")

        if self.display:
            self.init_display()
            self.font_default = pygame.font.Font(pygame.font.get_default_font(), 16)

    def start(self) -> np.ndarray:
        super().start()
        self.values = None
        self.target = self.build_actor(self.target_start_position, BLUE)
        self.agent = self.build_actor(self.agent_start_position, RED)
        state = self.get_state()
        self.logger.debug(f"state:\n{state}")
        self.logger.debug(f"agent position: {self.agent.get_position()} ({self.get_display_position(self.agent.get_position())})")
        self.logger.debug(f"target position: {self.target.get_position()} ({self.get_display_position(self.target.get_position())})")
        self.update_display()
        return state

    def step(self, action:int, *args) -> Tuple[float, np.ndarray, bool, Dict[str, object]]:
        self.values:np.ndarray = args[0] if len(args) > 0 else None
        self.update_agent(action)
        result = self.get_step_result()
        self.update_display()

        return result.reward, result.state, result.is_terminal, {}

    def get_state(self) -> np.ndarray:
        state = np.zeros((self.grid_dim, self.grid_dim), dtype=np.int32)
        state[self.target.get_position_idx()] = 1
        state[self.agent.get_position_idx()] = 1
        return state

    def update_display(self) -> None:
        if self.display:
            self.is_quit()
            self.surface.fill(THEME_BLUE_DARK)
            self.draw_grid_lines_x(THEME_DARK)
            self.draw_grid_lines_y(THEME_DARK)
            self.draw_trail()
            self.draw_values()
            self.draw_target()
            self.draw_agent()
            self.draw_status()
            pygame.display.update()
            self.game_clock.tick(self.fps)

    def draw_grid_lines_x(self, colour:Tuple[int, int, int]) -> None:
        buffer = 0
        x = round(self.width/self.grid_dim)
        for _ in range(0, self.grid_dim):
            start_pos = (buffer + x, 0)
            end_pos = (buffer + x, self.height)
            pygame.draw.line(self.surface, colour, start_pos, end_pos, width=2)
            buffer += x

    def draw_grid_lines_y(self, colour:Tuple[int, int, int]) -> None:
        buffer = 0
        y = round(self.height/self.grid_dim)
        for _ in range(0, self.grid_dim):
            start_pos = (0, buffer + y)
            end_pos = (self.width, buffer + y)
            pygame.draw.line(self.surface, colour, start_pos, end_pos, width=2)
            buffer += y

    def draw_trail(self) -> None:
        if self.show_trail and len(self.agent.position_history) > 1:
            for i in range(1, len(self.agent.position_history)):
                line_start_pos = self.get_display_position(self.agent.position_history[i - 1])
                line_end_pos = self.get_display_position(self.agent.position_history[i])
                pygame.draw.line(self.surface, THEME_BLUE_MEDIUM, line_start_pos, line_end_pos, width=2)

    def draw_values(self) -> None:
        if isinstance(self.values, np.ndarray):
            values = self.values
            for x in range(0, self.grid_dim):
                for y in range(0, self.grid_dim):
                    max_idx = values.argmax()
                    text_values = []

                    text_north = self.font_default.render(f"{values[NORTH]:.2f}", True, THEME_BLUE_LIGHT)
                    text_x = (self.cell_size[X] * x) + (self.cell_size[X] // 2)
                    text_y = (self.cell_size[Y] * y) + math.floor(self.cell_size[Y] * .1)
                    text_north_rect = text_north.get_rect()
                    text_north_rect.center = (text_x, text_y)
                    text_values.append((text_north, text_north_rect))

                    text_east = self.font_default.render(f"{values[EAST]:.2f}", True, THEME_BLUE_LIGHT)
                    text_x = (self.cell_size[X] * x) + math.floor(self.cell_size[X] * .9)
                    text_y = (self.cell_size[Y] * y) + (self.cell_size[Y] // 2)
                    text_east_rect = text_east.get_rect()
                    text_east_rect.center = (text_x, text_y)
                    text_values.append((text_east, text_east_rect))

                    text_south = self.font_default.render(f"{values[SOUTH]:.2f}", True, THEME_BLUE_LIGHT)
                    text_x = (self.cell_size[X] * x) + (self.cell_size[X] // 2)
                    text_y = (self.cell_size[Y] * y) + math.floor(self.cell_size[Y] * .9)
                    text_south_rect = text_south.get_rect()
                    text_south_rect.center = (text_x, text_y)
                    text_values.append((text_south, text_south_rect))

                    text_west = self.font_default.render(f"{values[WEST]:.2f}", True, THEME_BLUE_LIGHT)
                    text_x = (self.cell_size[X] * x) + math.floor(self.cell_size[X] * .1)
                    text_y = (self.cell_size[Y] * y) + (self.cell_size[Y] // 2)
                    text_west_rect = text_west.get_rect()
                    text_west_rect.center = (text_x, text_y)
                    text_values.append((text_west, text_west_rect))

                    if not all([v == 0 for v in values]):
                        topleft = (text_values[max_idx][1].topleft[0] - 2, text_values[max_idx][1].topleft[1] - 2)
                        size = (text_values[max_idx][1].size[X] + 4, text_values[max_idx][1].size[Y] + 4)
                        pygame.draw.rect(self.surface, THEME_BLUE_MEDIUM, pygame.Rect(topleft, size))

                    self.surface.blit(text_values[NORTH][0], text_values[NORTH][1])
                    self.surface.blit(text_values[EAST][0], text_values[EAST][1])
                    self.surface.blit(text_values[SOUTH][0], text_values[SOUTH][1])
                    self.surface.blit(text_values[WEST][0], text_values[WEST][1])

    def draw_target(self) -> None:
        pos = self.get_display_position(self.target.get_position())
        rect = self.target_image.get_rect()
        rect.center = pos
        self.surface.blit(self.target_image, rect)

    def draw_agent(self) -> None:
        pos = self.get_display_position(self.agent.get_position())
        if self.agent.get_position() != self.target.get_position():
            rect = self.agent_image.get_rect()
            rect.center = pos
            self.surface.blit(self.agent_image, rect)

    def draw_status(self) -> None:
        if self.agent.get_position() == self.target.get_position():
            text = "TERMINAL"
            text_x = self.width//2
            text_y = self.height//2
            text_rendered = self.font_default.render(text, True, THEME_DARK)
            text_rect = text_rendered.get_rect()
            text_rect.center = (text_x, text_y)
            self.surface.blit(text_rendered, text_rect)

    def build_actor(self, position:Tuple[int, int], colour:Tuple[int, int, int]) -> Actor:
        radius = round((self.cell_size[0] if self.cell_size[0] <= self.cell_size[1] else self.cell_size[1]) / 2) - 4
        return Actor(radius, colour, position)

    def get_display_position(self, position:Tuple[int, int]) -> Tuple[int, int]:
        x = self.cell_size[X] * position[X] - (self.cell_size[X] // 2)
        y = self.cell_size[Y] * position[Y] - (self.cell_size[Y] // 2)
        return x, y

    def update_agent(self, action:int) -> None:
        agent_moved = False
        if self.operator == HUMAN:
            action = -1
            if self.check_input():
                pressed = pygame.key.get_pressed()
                if pressed[K_UP]:
                    action = NORTH
                    agent_moved = True
                elif pressed[K_RIGHT]:
                    action = EAST
                    agent_moved = True
                elif pressed[K_DOWN]:
                    action = SOUTH
                    agent_moved = True
                elif pressed[K_LEFT]:
                    action = WEST
                    agent_moved = True
        else:
            agent_moved = True

        position = self.agent.get_position()
        if action == NORTH and position[Y] > 1:
            position = (position[X], position[Y] - 1)
        elif action == EAST and position[X] < self.grid_dim:
            position = (position[X] + 1, position[Y])
        elif action == SOUTH and position[Y] < self.grid_dim:
            position = (position[X], position[Y] + 1)
        elif action == WEST and position[X] > 1:
            position = (position[X] - 1, position[Y])

        if position != self.agent.get_position():
            self.agent.update_position(position)

        if agent_moved:
            self.log_state_debug()

    def get_step_result(self) -> StepResult:
        state = self.get_state()

        # reward = 1. if self.agent.get_position() == self.target.get_position() else 0.
        # reward = 0. if self.agent.get_position() == self.target.get_position() else -1.
        reward = 1. if self.agent.get_position() == self.target.get_position() else -1.

        is_terminal = self.agent.get_position() == self.target.get_position()

        result = StepResult()
        result.is_terminal = is_terminal
        result.reward = reward
        result.state = state
        return result

    def log_state_debug(self) -> None:
        state = self.get_state()
        self.logger.debug(f"agent: ({self.agent.get_position()})")
        self.logger.debug(f"target: ({self.target.get_position()})")
        self.logger.debug(f"state:\n{state}")
