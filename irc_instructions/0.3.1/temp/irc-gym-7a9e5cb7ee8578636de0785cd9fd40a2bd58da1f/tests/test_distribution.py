import unittest

import numpy as np
import torch
from gym.spaces import MultiDiscrete, Box

from irc.distribution import BasicDiscretePotential, BasicDiscreteDistribution
from irc.alias import Tensor

from utils import _create_tensor_samples


class TestPotential(unittest.TestCase):

    def _create_discrete(self, space=None):
        if space is None:
            space = MultiDiscrete([2, 3])
        return BasicDiscretePotential(space)

    def test_basic(self):
        for phi in [self._create_discrete()]:
            # test get param_vec
            param_vec = phi.get_param_vec()
            self.assertIsInstance(param_vec, Tensor)
            self.assertEqual(len(param_vec.shape), 1)
            self.assertEqual(len(param_vec), phi.num_params)
            # test forward pass
            num_samples = 8
            xs = _create_tensor_samples(phi.space, num_samples)
            self.assertEqual(xs.shape[1], phi.num_vars)
            es = phi(xs)
            self.assertIsInstance(es, Tensor)
            self.assertEqual(es.shape, (num_samples,))
            self.assertTrue(es.requires_grad)
            param_vec = torch.randn(phi.num_params)
            with torch.no_grad():
                es = phi(xs, param_vec)
            # test set param_vec
            phi.set_param_vec(param_vec)
            self.assertTrue(
                np.allclose(es.data.numpy(), phi(xs).data.numpy()),
            )

    def test_discrete(self):
        # one binary variable
        space = MultiDiscrete([2])
        prob_dict = {
            (0,): 0.1, (1,): 0.9,
        }
        phi = self._create_discrete(space)
        phi.set_from_prob(prob_dict)
        xs = torch.tensor(np.array([(0,), (1,)]), dtype=torch.long)
        with torch.no_grad():
            es = phi(xs)
        self.assertAlmostEqual(np.exp((es[1]-es[0]).item()), 9, places=4)
        # two discrete variables
        space = MultiDiscrete([3, 4])
        prob_dict = {
            (0, 0): 3, (0, 2): 1, (2, 2): 9,
        }
        phi = self._create_discrete(space)
        phi.set_from_prob(prob_dict)
        xs = torch.tensor(np.array([(0, 2), (2, 2)]), dtype=torch.long)
        with torch.no_grad():
            es = phi(xs)
        self.assertAlmostEqual(np.exp((es[1]-es[0]).item()), 9, places=4)


class TestDistribution(unittest.TestCase):

    def test_init(self):
        space = MultiDiscrete([2, 3, 4])
        BasicDiscreteDistribution(space)

        # test setting idxs
        idxs = [[0, 1]]
        with self.assertRaises(Exception):
            BasicDiscreteDistribution(space, idxs=idxs)
        idxs = [[0, 1], [2, 0]]
        BasicDiscreteDistribution(space, idxs=idxs)

        # test setting phis
        phis = [None]
        with self.assertRaises(Exception):
            BasicDiscreteDistribution(space, idxs=idxs, phis=phis)
        phis = [None, None]
        BasicDiscreteDistribution(space, idxs=idxs, phis=phis)

    def test_param_vec(self):
        space = MultiDiscrete([2, 3, 4])

        # test default idxs
        p_x = BasicDiscreteDistribution(space)
        param_vec = p_x.get_param_vec()
        self.assertEqual(param_vec.shape, (24,))
        p_x.set_param_vec(param_vec)

        # test structured idxs
        idxs = [[0], [1, 2]]
        p_x = BasicDiscreteDistribution(space, idxs=idxs)
        param_vec = p_x.get_param_vec()
        self.assertEqual(param_vec.shape, (14,))
        p_x.set_param_vec(param_vec)

    def test_energy(self):
        x_space = MultiDiscrete([2, 3])

        # test energy forward pass
        p_x = BasicDiscreteDistribution(x_space)
        num_samples = 8
        xs = _create_tensor_samples(x_space, num_samples)
        es = p_x.energy(xs)
        self.assertEqual(es.shape, (num_samples,))
        self.assertTrue(es.requires_grad)

        # test energy values
        idxs = [[0], [1]]
        p_x = BasicDiscreteDistribution(x_space, idxs=idxs)
        param_vec = torch.tensor([2, 0, -1, 0, 1], dtype=torch.float)
        p_x.set_param_vec(param_vec)
        xs = torch.tensor(np.array([[0, 0], [1, 1]]), dtype=torch.long)
        with torch.no_grad():
            es = p_x.energy(xs)
        self.assertAlmostEqual(es[0]-es[1], 1, places=4) # (2-1)-(0+0)

    def test_sample(self):
        x_space = MultiDiscrete([2, 3])
        idxs = [[0], [1]]
        p_x = BasicDiscreteDistribution(x_space, idxs=idxs)
        param_vec = torch.tensor([2, 0, -1, 0, 1], dtype=torch.float)
        p_x.set_param_vec(param_vec)
        num_samples = 10000
        xs = p_x.sample(num_samples)
        self.assertAlmostEqual((xs[:, 0]==1).mean(), 1/(1+np.exp(2)), places=1)
        self.assertAlmostEqual(
            (xs[:, 1]==1).mean(), 1/(1+np.exp(-1)+np.exp(1)), places=1,
        )
        self.assertAlmostEqual(
            ((xs[:, 0]==1)&(xs[:, 1]==1)).mean(),
            1/(1+np.exp(2))/(1+np.exp(-1)+np.exp(1)), places=1,
        )

    def test_estimate(self):
        space = MultiDiscrete([2, 2])
        p_x = BasicDiscreteDistribution(space)
        num_samples = 1000
        xs = np.concatenate([
            np.array([[0, 0]]*num_samples),
            np.array([[0, 1]]*num_samples*2),
            np.array([[1, 0]]*num_samples*3),
            np.array([[1, 1]]*num_samples*4),
        ])
        p_x.estimate(xs)
        with torch.no_grad():
            param_vec = p_x.get_param_vec()
        p = torch.softmax(param_vec, dim=0).numpy()
        delta = np.abs(p-np.array([0.1, 0.2, 0.3, 0.4])).max()
        self.assertLess(delta, 0.02)


if __name__=='__main__':
    unittest.main()
