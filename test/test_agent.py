import unittest
import torch
from model.DQN import DQN_unit
from utils.config import Config
config = Config('./builds')

class Testagent(unittest.TestCase):

    def test_DQN_unit(self):
        n = config.agents.number_preys + config.agents.number_predators
        input = torch.randn(128, n)
        model = DQN_unit()
        with self.assertRaises(TypeError):
            output = model(input)


    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)

if __name__ == '__main__':
    unittest.main()
