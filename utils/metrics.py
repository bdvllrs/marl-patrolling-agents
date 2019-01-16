import numpy as np


class Metrics:
    def __init__(self):
        self.data = {
            "losses": [],
            "returns": []
        }

        self.loss_buffer = []
        self.returns_buffer = []

    def add_return(self, value):
        self.returns_buffer.append(value)

    def add_loss(self, value):
        self.loss_buffer.append(value)

    def compute_averages(self):
        if self.loss_buffer:
            self.data["losses"].append(np.mean(self.loss_buffer))
            self.loss_buffer = []
        if self.returns_buffer:
            self.data["returns"].append(np.mean(self.returns_buffer))
            self.returns_buffer = []

    def plot(self, key, episode, ax, legend):
        x = np.linspace(0, episode, len(self.data[key]))
        ax.plot(x, self.data[key], label=legend)

    def plot_returns(self, episode, ax, legend):
        self.plot("returns", episode, ax, legend)

    def plot_losses(self, episode, ax, legend):
        self.plot("losses", episode, ax, legend)
