import numpy as np
import matplotlib.pyplot as plt
from utils import choice, possible_directions, get_distance_between

__all__ = ["World"]


class World:
    """
    Environment. An actor takes 1 pixel il the board
    """

    actor_radius = 1
    noise = 0.2
    max_iterations = None
    current_iter = 0

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.actors = []
        self.initial_positions = []
        self.first_draw = True

    def random_position(self):
        return np.random.randint(0, self.width), np.random.randint(0, self.height)

    def add_actor(self, actor, position=None):
        self.initial_positions.append(position)
        actor.set_size_board(self.width, self.height)
        self.actors.append(actor)

    def reset(self):
        self.current_iter = 0
        positions = []
        for k, actor in enumerate(self.actors):
            position = self.initial_positions[k]
            position = self.random_position() if position is None else position
            positions.append(position)
            actor.set_position(position)
        return positions

    def step(self):
        self.current_iter += 1
        terminal_state = False if self.max_iterations is None else self.current_iter >= self.max_iterations
        actions = []
        positions = []
        for actor in self.actors:
            obs = []
            nb_patrol_around_target = 0
            for other_actor in self.actors:
                if actor.type == 'target' and other_actor.type == 'patrol':
                    if other_actor.position in actor.view_area:
                        obs.append(other_actor)
                    if get_distance_between(actor.limit_board, actor.position, other_actor.position) == 1:
                        nb_patrol_around_target += 1
            if nb_patrol_around_target >= 2:
                terminal_state = True
            action = actor.draw_action(obs)
            actions.append(action)
            if np.random.rand() < self.noise:
                # We select a position at random and not the one selected
                action = choice(possible_directions(actor.limit_board, actor.position))
            actor.set_position(action)
            positions.append(actor.position)
        return positions, actions, terminal_state

    def draw_board(self):
        plt.ylim(bottom=0, top=self.height)
        plt.xlim(left=0, right=self.width)
        plt.grid(True)
        for actor in self.actors:
            x, y = actor.position
            circle = plt.Circle((x, y), radius=self.actor_radius, color=actor.color)
            plt.gcf().gca().add_artist(circle)
        plt.show()
