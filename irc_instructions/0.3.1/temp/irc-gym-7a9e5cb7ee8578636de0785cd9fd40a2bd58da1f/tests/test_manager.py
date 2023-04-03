import unittest

import random, shutil

from jarvis.config import Config
from irc.manager import AgentManager


class TestAgentManager(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = 'test_'+''.join([
            '{:X}'.format(random.choice(range(16))) for _ in range(4)
        ])

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_config(self):
        manager = AgentManager(self.tmp_dir)
        with self.assertRaises(Exception):
            manager.get_config()

        env_class = 'irc.examples.FoodBoxesEnv'
        manager = AgentManager(
            self.tmp_dir, {'env._target_': env_class},
            verbose=False,
        )
        config = manager.get_config()
        self.assertIsInstance(config, Config)
        self.assertEqual(config.env._target_, env_class)

        env_param = (0.2, 0.15, 0.1, 0.05, 0.8, 0.8, 0.9, 5., -1.)
        with self.assertRaises(Exception):
            manager = AgentManager(
                self.tmp_dir, {'env._target_': env_class, 'env_param': tuple(*env_param, 100)},
                verbose=False,
            )
            config = manager.get_config()
            manager.setup(config)
        manager = AgentManager(
            self.tmp_dir, {'env._target_': env_class, 'env_param': env_param},
            verbose=False,
        )
        config = manager.get_config()
        manager.setup(config)
        self.assertEqual(
            manager.configs._to_hashable(config),
            manager.configs._to_hashable(manager.config),
        )
        self.assertEqual(env_param, tuple(manager.agent.model.env.get_param()))

    def test_one_epoch(self):
        config = {
            'env._target_': 'irc.examples.FoodBoxesEnv',
            'env_param': (0.2, 0.15, 0.1, 0.05, 0.8, 0.8, 0.9, 5., -1.),
            'estimate.p_o_s.init': {'num_samples': 1000},
            'collect.init': {'num_steps': 50},
            'train.init.batch_size': 32,
        }
        manager = AgentManager(self.tmp_dir, verbose=False)
        config = manager.get_config(config)
        manager.setup(config)
        manager.init_ckpt()
        manager.eval()
        manager.save_ckpt()
        manager.load_ckpt()
        manager.train()


if __name__=='__main__':
    unittest.main()
