import numpy as np
import torch

from .model import BaseBeliefModel
from .alias import Array, SB3Algo


class BeliefAgent:
    r"""An agent with an internal model and uses belief for decision."""

    def __init__(self,
        model: BaseBeliefModel,
        algo: SB3Algo,
    ):
        r"""
        Args
        ----
        model:
            The internal model of assumed environment, which describes an MDP on
            the belief.
        algo:
            The reinforcement learning algorithm for the policy taking belief as
            input.

        """
        self.model = model
        self.algo = algo

    def state_dict(self):
        r"""Returns state dictionary."""
        return {
            'model': self.model.state_dict(),
            'policy': self.algo.policy.state_dict(),
        }

    def load_state_dict(self, state):
        r"""Loads state dictionary."""
        self.model.load_state_dict(state['model'])
        self.algo.policy.load_state_dict(state['policy'])

    @staticmethod
    def _return(gamma, rewards):
        r"""Returns cumulative discounted reward."""
        w = gamma**np.flip(np.arange(len(rewards)))
        g = (w*rewards).sum()
        return g

    def run_one_episode(self, **kwargs):
        return self.model.run_one_episode(policy=self.algo.policy, **kwargs)

    def evaluate(self, num_episodes: int, max_steps: int) -> dict:
        r"""Evaluates current policy with respect to internal model.

        Args
        ----
        num_episodes:
            Number of evaluation episodes.
        max_steps:
            See `BaseBeliefModel.run_one_episode` for more details.

        Returns
        -------
        eval_record:
            Evaluation record, containing episode returns.

        """
        num_steps = [] # number of steps of episodes
        rates = [] # reward rates of episodes
        returns = [] # cumulative discounted rewards of episodes
        for _ in range(num_episodes):
            episode = self.run_one_episode(max_steps=max_steps, q_states=[])
            num_steps.append(episode['num_steps'])
            rewards = episode['rewards']
            rates.append(np.mean(rewards))
            returns.append(self._return(self.algo.gamma, rewards))
        eval_record = {
            'num_episodes': num_episodes,
            'max_steps': max_steps,
            'num_steps': np.array(num_steps),
            'rates': np.array(rates),
            'returns': np.array(returns),
        }
        return eval_record

    def episode_logps(self,
        observations: Array,
        actions: Array,
    ) -> Array:
        r"""Returns the likelihood of given episode.

        Args
        ----
        observations, actions:
            Time series of observations and actions, see
            `BaseBeliefModel.get_beliefs` for more details.

        Returns
        -------
        beliefs:
            Time series of beleif vectors, see `BaseBeliefModel.get_beliefs` for
            more details.
        logps: (num_steps,)
            Log likelihoods log(p(actions, observations)) of time steps [0, t).

        """
        device = self.model.device
        self.algo.policy.eval().to(device)

        beliefs = self.model.get_beliefs(observations, actions)
        with torch.no_grad():
            dists = self.algo.policy.get_distribution(
                torch.tensor(beliefs, dtype=torch.float, device=device),
            )
        logps = dists.log_prob(torch.tensor(actions, dtype=torch.long, device=device)).cpu().numpy()
        return beliefs, logps
