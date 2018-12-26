import numpy as np
import scipy.spatial
from utils import choice

__all__ = ["Patrol", "Target", "Civilian"]


class Actor:
    type_id = None
    type = None
    position = None
    color = 'red'
    limit_board = None
    view_radius = 10  # manhattan distance

    history = []

    def __init__(self, name=None):
        assert (self.type_id is not None and
                self.type is not None), "This actor does not have any type information."
        if name is None:
            name = self.type + " " + str(np.random.randint(0, 1000))
        self.name = name

    def set_position(self, position):
        self.history.append(position)
        self.position = position

    @property
    def view_area(self):
        """
        Defines the points in the board the actor can see
        """
        positions = [self.position]
        self.view_area_from(self.position, self.view_radius, positions)
        return positions

    def view_area_from(self, position, hops, positions):
        """
        Defines the points in the board the actor can see from the given position
        Args:
            positions: list of seen positions
            position: original position
            hops: Number of allowed hops
        """
        if hops == 0:
            return
        possible_positions = self.possible_directions(position)
        for new_position in possible_positions:
            if new_position not in positions:
                positions.append(new_position)
                self.view_area_from(new_position, hops - 1, positions)
        return

    def get_distance_between(self, position_1, position_2):
        if position_1 in self.possible_directions(position_2):
            return 1
        x, y = position_1
        x_2, y_2 = position_2
        x_new, y_new = position_1
        if x < x_2:
            if y < y_2:
                x_new, y_new = x+1, y+1
            elif y == y_2:
                x_new = x+1
            else:
                x_new, y_new = x+1, y-1
        elif x == x_2:
            if y < y_2:
                y_new = y+1
            elif y > y_2:
                y_new = y-1
        else:
            if y < y_2:
                x_new, y_new = x-1, y+1
            elif y == y_2:
                x_new = x-1
            else:
                x_new, y_new = x-1, y-1
        return 1 + self.get_distance_between((x_new, y_new), position_2)

    def set_size_board(self, width, height):
        self.limit_board = (width, height)

    def possible_directions(self, position=None):
        lim_x, lim_y = self.limit_board
        x, y = self.position if position is None else position
        possible_direction = []
        if x > 0:
            possible_direction.append((x - 1, y))
            if y > 0:
                possible_direction.append((x - 1, y - 1))
            if y < lim_y - 1:
                possible_direction.append((x - 1, y + 1))
        if y > 0:
            possible_direction.append((x, y - 1))
        if x < lim_x - 1:
            possible_direction.append((x + 1, y))
            if y > 0:
                possible_direction.append((x + 1, y - 1))
            if y < lim_y - 1:
                possible_direction.append((x + 1, y + 1))
        if y < lim_y - 1:
            possible_direction.append((x, y + 1))
        return possible_direction

    def draw_action(self, obs):
        """
        Select the action to perform
        Args:
            obs: information on where are the other actors. List of actors.
        """
        return choice(self.possible_directions())


class Patrol(Actor):
    type_id = 0
    type = "patrol"
    color = 'blue'


class Target(Actor):
    type_id = 1
    type = "target"

    def distance_to_patrolers(self, obs, position=None):
        position = self.position if position is None else position
        sum_dist = np.inf
        for actor in obs:
            if actor.type == 'patrol':
                if sum_dist == np.inf:
                    sum_dist = 0
                sum_dist += self.get_distance_between(position, actor.position)
        return sum_dist

    def draw_action(self, obs):
        """
        Choose the position as far away from patrol
        Args:
            obs: information on where are the other actors. List of actors.
        """
        if self.distance_to_patrolers(obs) == np.inf:
            return self.position
        possible_directions = self.possible_directions()
        position = max(possible_directions, key=lambda pos: self.distance_to_patrolers(obs, pos))
        return position


class Civilian(Actor):
    type_id = 2
    type = "civilian"
    color = "black"
