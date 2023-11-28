import unittest

from pathlib import Path
import numpy as np
import torch

from irc.examples import IdenticalBoxesEnv
from irc.model import ReplayBuffer, SamplingBeliefModel

rng = np.random.default_rng()


class TestBuffer(unittest.TestCase):

    def _create_episode(self, num_steps):
        episode = {
            'actions': rng.choice(3, size=(num_steps,)),
            'observations': np.stack([
                rng.choice(2, size=(num_steps+1,)),
                rng.choice(4, size=(num_steps+1,)),
            ]),
        }
        return episode

    def test_capacity(self):
        buffer = ReplayBuffer(capacity=10)
        buffer.add_episode(self._create_episode(num_steps=6))
        self.assertEqual(len(buffer), 6)
        buffer.add_episode(self._create_episode(num_steps=4))
        self.assertEqual(len(buffer), 10)
        buffer.add_episode(self._create_episode(num_steps=3))
        self.assertEqual(len(buffer), 7)
        buffer.add_episode(self._create_episode(num_steps=8))
        self.assertEqual(len(buffer), 8)
        with self.assertRaises(Exception):
            buffer.add_episode(self._create_episode(num_steps=11))


class TestSamplingBeliefModel(unittest.TestCase):

    def _example_model(self, pretrained=0):
        env = IdenticalBoxesEnv(num_boxes=2, num_shades=2, lambda_center=0.25)
        model = SamplingBeliefModel(
            env, p_s={'idxs': [[0, 1], [2]]}, p_o={'idxs': [[0], [1], [2]]}, device='cpu',
        )
        if pretrained>0:
            saved = torch.load(Path(__file__).parent/'fixtures'/'model_example.pt')
            model.init_net.load_state_dict(saved['init_net'])
            model.observe_net.load_state_dict(saved['observe_net'])
            if pretrained>1:
                model.buffer.load_state_dict(saved['buffer'])
            if pretrained>2:
                model.update_net.load_state_dict(saved['update_net'])
        return model

    def test_basic(self):
        model = self._example_model()
        self.assertEqual(model.api, 'v26')
        belief = model.reset()
        self.assertEqual(belief.shape, (7,))

    def test_init(self, eps=0.1):
        model = self._example_model()

        model.train_init_net(num_samples=1000, num_epochs=30)
        states = torch.tensor(np.array([
            [0, 0, 2], [0, 1, 2], [1, 0, 2], [1, 1, 2],
        ]), dtype=torch.long)
        observations = torch.tensor(np.array([
            [0, 0, 2], [0, 1, 2], [1, 0, 2], [1, 1, 2],
        ]), dtype=torch.long)
        for observation in observations:
            with torch.no_grad():
                model.p_s.set_param_vec(
                    model.init_net(observation[None])[0]
                )
                p_est = model.p_s.loglikelihoods(states).exp().numpy()
            p_true = np.array([1, 0, 0, 0])
            delta = np.abs(p_true-p_est).max()
            self.assertLess(delta, eps)

        model.train_observe_net(num_samples=4000, num_epochs=30, use_replay=False)
        for pos in [0, 2]:
            states = torch.tensor(np.array([
                [0, 0, pos], [0, 1, pos], [1, 0, pos], [1, 1, pos],
            ]), dtype=torch.long)
            observations = torch.tensor(np.array([
                [0, 0, pos], [0, 1, pos], [1, 0, pos], [1, 1, pos],
            ]), dtype=torch.long)
            for state in states:
                with torch.no_grad():
                    model.p_o.set_param_vec(
                        model.observe_net(state[None])[0]
                    )
                    p_est = model.p_o.loglikelihoods(observations).exp().numpy()
                p_true = []
                for observation in observations:
                    _p = 1.
                    for i in range(2):
                        if observation[i]==state[i]:
                            _p *= 0.7 if pos==2 else 0.6
                        else:
                            _p *= 0.3 if pos==2 else 0.4
                    p_true.append(_p)
                p_true = np.array(p_true)
                delta = np.abs(p_true-p_est).max()
                self.assertLess(delta, eps)

    def test_rollouts(self):
        model = self._example_model(pretrained=1)

        num_steps, max_steps = 40, 15
        collect_stats = model.collect_rollouts(total_steps=num_steps, max_steps=max_steps)
        self.assertEqual(len(model.buffer.episodes), collect_stats['num_episodes'])
        self.assertEqual(len(model.buffer), num_steps)
        self.assertEqual(model.buffer.episodes[0]['beliefs'].shape, (16, 7))

    def test_belief_net(self, eps=0.05):
        model = self._example_model(pretrained=2)
        model.train_update_net()

#         model.update_mode = 'N'
#         episode = model.run_one_episode()

#         episode = rng.choice(model.buffer.episodes)
#         actions = episode['actions']
#         observations = episode['observations']
#         beliefs, probs = {}, {}
#         for mode in ['S', 'N']:
#             model.update_mode = mode
#             beliefs[mode] = []
#             model.init_belief(observations[0])
#             beliefs[mode].append(model.get_belief())
#             for i in range(len(actions)):
#                 model.update_belief(actions[i], observations[i+1])
#                 beliefs[mode].append(model.get_belief())
#             beliefs[mode] = np.array(beliefs[mode])

#             probs[mode] = []
#             for belief in beliefs[mode]:
#                 model.p_s.set_param_vec(torch.tensor(belief, dtype=torch.float))
#                 probs[mode].append(model.p_s.loglikelihoods(model.p_s._all_xs).exp().data.numpy())
#             probs[mode] = np.array(probs[mode])
#         delta = np.abs(probs['S']-probs['N']).mean()
#         self.assertLess(delta, eps)


if __name__=='__main__':
    unittest.main()
