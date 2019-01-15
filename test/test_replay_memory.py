import unittest
from sim import ReplayMemory
from utils import Config

config = Config("./config")


class TestEnv(unittest.TestCase):

    def test_size(self):
        rm = ReplayMemory(5)
        self.assertTrue(len(rm) <= config.replay_memory)

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
