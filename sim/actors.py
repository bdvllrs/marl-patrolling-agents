import numpy as np
from utils import choice, possible_directions, get_distance_between

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
        possible_positions = possible_directions(self.limit_board, position)
        for new_position in possible_positions:
            if new_position not in positions:
                positions.append(new_position)
                self.view_area_from(new_position, hops - 1, positions)
        return

    def set_size_board(self, width, height):
        self.limit_board = (width, height)

    def draw_action(self, obs):
        """
        Select the action to perform
        Args:
            obs: information on where are the other actors. List of actors.
        """
        return choice(possible_directions(self.limit_board, self.position))


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
                sum_dist += get_distance_between(self.limit_board, position, actor.position)
        return sum_dist

    def draw_action(self, obs):
        """
        Choose the position as far away from patrol
        Args:
            obs: information on where are the other actors. List of actors.
        """
        if self.distance_to_patrolers(obs) == np.inf:
            return self.position
        directions = possible_directions(self.limit_board, self.position)
        position = max(directions, key=lambda pos: self.distance_to_patrolers(obs, pos))
        return position


class Civilian(Actor):
    type_id = 2
    type = "civilian"
    color = "black"
