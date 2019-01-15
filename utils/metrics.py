import numpy as np


class Metrics:
    def __init__(self):
        self.data = {
            "losses": [],
            "returns": []
        }

        self.loss_buffer = []

    def add_return(self, value):
        self.data["returns"].append(value)

    def add_loss(self, value):
        self.loss_buffer.append(value)

    def compute_loss_average(self):
        self.data["losses"].append(np.mean(self.loss_buffer))
        self.loss_buffer = []

    def plot(self, key, ax, legend):
        ax.plot(list(range(0, len(self.data[key]))), self.data[key], legend=legend)

    def plot_returns(self, ax, legend):
        self.plot("returns", ax, legend)

    def plot_losses(self, ax, legend):
        self.plot("losses", ax, legend)
