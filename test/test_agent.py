import unittest
import torch
from model.dqn import DQNUnit
from utils.config import Config
config = Config('./builds')

class Testagent(unittest.TestCase):

    def test_DQN_unit(self):
        n = config.agents.number_preys + config.agents.number_predators
        input = torch.randn(128, n)
        model = DQNUnit()
        with self.assertRaises(TypeError):
            output = model(input)


if __name__ == '__main__':
    unittest.main()
