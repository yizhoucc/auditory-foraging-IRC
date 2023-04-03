import unittest

import numpy as np
import torch
from gym.spaces import MultiDiscrete

from irc.net import MultiLayerPerceptronNet, CompleteEmbedWrapper

from utils import _create_tensor_samples

class TestMlpNet(unittest.TestCase):

    def test_init(self):
        x_space = MultiDiscrete([2, 2, 3])
        y_spaces = [MultiDiscrete([2]), MultiDiscrete([4])]
        # test forward pass
        net = MultiLayerPerceptronNet(x_space, y_spaces)
        num_samples = 8
        ys = [_create_tensor_samples(y_space, num_samples) for y_space in y_spaces]
        param_vecs = net(*ys)
        self.assertEqual(param_vecs.shape, (num_samples, 12))
        # test p_x structure
        p_x = {'idxs': [[0, 1], [2]]}
        net = MultiLayerPerceptronNet(x_space, y_spaces, p_x)
        param_vecs = net(*ys)
        self.assertEqual(param_vecs.shape, (num_samples, 7))
        # test mlp_features
        for mlp_features in [None, [], [32], [64, 16]]:
            net = MultiLayerPerceptronNet(x_space, y_spaces, mlp_features=mlp_features)


class TestCompleteNet(unittest.TestCase):

    def test_init(self):
        x_space = MultiDiscrete([2, 2, 3])
        y_spaces = [MultiDiscrete([5, 4])]
        # test wrapping
        net = CompleteEmbedWrapper(x_space, y_spaces)
        net = CompleteEmbedWrapper(x_space, y_spaces, p_x={'idxs': [[0], [1], [2]]})
        net = CompleteEmbedWrapper(x_space, y_spaces, net={'mlp_features': []})
        # test forward pass
        num_samples = 8
        ys = [_create_tensor_samples(y_space, num_samples) for y_space in y_spaces]
        param_vecs = net(*ys)
        self.assertEqual(param_vecs.shape, (num_samples, 12))

    def test_estimate(self):
        x_space = MultiDiscrete([2])
        y_space = MultiDiscrete([2])
        num_samples = 100
        xys = np.concatenate([
            np.array([[0, 0]]*num_samples),
            np.array([[0, 1]]*num_samples*2),
            np.array([[1, 0]]*num_samples*3),
            np.array([[1, 1]]*num_samples*4),
        ])
        xs = xys[:, 0][:, None]
        ys = xys[:, 1][:, None]
        p_x_y = CompleteEmbedWrapper(x_space, [y_space])
        p_x_y.estimate(xs, [ys], num_epochs=40)
        with torch.no_grad():
            param_vecs = p_x_y(
                torch.tensor(np.array([(0,), (1,)]), dtype=torch.long),
            )
        p = torch.softmax(param_vecs, dim=1).numpy()
        delta = np.abs(p-np.array([[1/4, 3/4], [1/3, 2/3]])).max()
        self.assertLess(delta, 0.02)


if __name__=='__main__':
    unittest.main()
