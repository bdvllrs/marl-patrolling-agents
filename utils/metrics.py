import numpy as np


class Metrics:
    def __init__(self):
        self.data = {
            "losses": [],
            "loss_actor": [],
            "returns": [],
            "collisions": []
        }

        self.loss_buffer = []
        self.returns_buffer = []
        self.loss_actor = []
        self.collision_count_buffer = []

    def add_return(self, value):
        self.returns_buffer.append(value)

    def add_collision_count(self, value):
        self.collision_count_buffer.append(value)

    def add_loss(self, value):
        self.loss_buffer.append(value)

    def add_loss_actor(self, value):
        self.loss_actor.append(value)

    def compute_averages(self):
        if self.loss_buffer:
            self.data["losses"].append(np.mean(self.loss_buffer))
            self.loss_buffer = []
        if self.returns_buffer:
            self.data["returns"].append(np.mean(self.returns_buffer))
            self.returns_buffer = []
        if self.loss_actor:
            self.data["loss_actor"].append(np.mean(self.loss_actor))
            self.loss_actor = []
        if self.collision_count_buffer:
            self.data["collisions"].append(np.mean(self.collision_count_buffer))
            self.collision_count_buffer = []

    def plot(self, key, episode, ax, legend=None):
        x = np.linspace(0, episode, len(self.data[key]))
        if legend is not None:
            ax.plot(x, self.data[key], label=legend)
        else:
            ax.plot(x, self.data[key])

    def plot_returns(self, episode, ax, legend):
        self.plot("returns", episode, ax, legend)

    def plot_losses(self, episode, ax, legend):
        self.plot("losses", episode, ax, legend)

    def plot_losses_actor(self, episode, ax, legend):
        self.plot("loss_actor", episode, ax, legend)

    def plot_collision_counts(self, episode, ax):
        self.plot("collisions", episode, ax)
