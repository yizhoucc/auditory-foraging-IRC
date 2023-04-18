import numpy as np
import torch
from torch.distributions.categorical import Categorical
from gym.spaces import MultiDiscrete

from typing import Optional, Union
from collections.abc import Collection

from jarvis.config import Config

from . import rcParams
from .net import BaseDistributionNet, _create_net
from .model import BaseBeliefModel, DistilledBeliefModel
from .alias import Array, EnvParam, Tensor, SB3Policy


class BaseBeliefAgent:
    r"""An agent with an internal model and uses belief for decision."""

    def __init__(self,
        model: BaseBeliefModel,
        policy: SB3Policy,
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
        self.policy = policy

    def state_dict(self):
        r"""Returns state dictionary."""
        return {
            'model': self.model.state_dict(),
            'policy': self.policy.state_dict(),
        }

    def load_state_dict(self, state):
        r"""Loads state dictionary."""
        self.model.load_state_dict(state['model'])
        self.policy.load_state_dict(state['policy'])

    @staticmethod
    def _return(gamma, rewards):
        r"""Returns cumulative discounted reward."""
        w = gamma**np.flip(np.arange(len(rewards)))
        g = (w*rewards).sum()
        return g

    def run_one_episode(self, **kwargs) -> dict:
        r"""Runs one episode using current policy.

        Returns
        -------
        episode: dict
            See `BaseBeliefModel.run_one_episode` for more details.

        """
        return self.model.run_one_episode(policy=self._get_distribution, **kwargs)

    def run_episodes(self,
        num_episodes: Optional[int] = None,
        total_steps: Optional[int] = None,
        max_steps: Optional[int] = None,
        **kwargs,
    ) -> list[dict]:
        r"""Runs multiple episodes using current policy.

        Args
        ----
        num_episodes:
            Number of episodes to run. Should be ``None`` if `total_steps` is
            specified.
        total_steps:
            Total time steps to run. Should be ``None`` if `num_episodes` is
            specified.
        max_steps:
            See `BaseBeliefModel.run_one_episode` for more details.

        Returns
        -------
        episodes:
            A list of episode data, see `BaseBeliefModel.run_one_episode` for
            more details.

        """
        assert (num_episodes is None)!=(total_steps is None), (
            "One and only one of 'num_episodes' and 'total_steps' must be specified."
        )
        episodes = []
        if num_episodes is not None:
            for _ in range(num_episodes):
                episode = self.run_one_episode(max_steps=max_steps, **kwargs)
                episodes.append(episode)
        if total_steps is not None:
            count = 0
            while count<total_steps:
                episode = self.run_one_episode(
                    max_steps=min(max_steps, total_steps-count), **kwargs,
                )
                episodes.append(episode)
                count += episode['num_steps']
        return episodes

    def _get_distribution(self, beliefs: Tensor) -> Categorical:
        dists = self.policy.get_distribution(beliefs.to(self.model.device)).distribution
        return dists

    def compute_likelihoods(self,
        observations: Array,
        actions: Array,
    ) -> tuple[Tensor, Tensor]:
        r"""Returns the likelihood of given episode.

        Args
        ----
        observations, actions:
            Time series of observations and actions, see
            `BaseBeliefModel.compute_beliefs` for more details.

        Returns
        -------
        beliefs: (num_steps+1, belief_dim)
            Time series of beleif vectors, see `BaseBeliefModel.compute_beliefs`
            for more details.
        logps: (num_steps,)
            Log likelihoods log(p(actions|beliefs)) of time steps [0, t).

        """
        beliefs = self.model.compute_beliefs(observations, actions)
        dists = self._get_distribution(beliefs[:-1])
        logps = dists.log_prob(
            torch.tensor(actions, dtype=torch.long, device=self.model.device),
        )
        return beliefs, logps


class DistilledBeliefAgent(BaseBeliefAgent):

    model: DistilledBeliefModel
    policy: BaseDistributionNet

    def __init__(self,
        model: DistilledBeliefModel,
        policy: Union[BaseDistributionNet, dict, None] = None,
    ):
        _rcParams = Config(rcParams.get('agent.DistilledBeliefAgent._init_'))
        if policy is None or isinstance(policy, dict):
            policy = Config(policy)
            policy.fill(_rcParams.get('policy_net'))
        _action_space = MultiDiscrete([model.action_space.n])
        policy = _create_net(
            policy, _action_space, [model.belief_space, model.theta_space],
            {'_target_': 'irc.distribution.BasicDiscreteDistribution', 'rng': model.rng},
        )
        super().__init__(model, policy)

    def _get_distribution(self, beliefs: Array) -> Categorical:
        return Categorical(logits=self.policy.forward(
            beliefs, torch.tile(self.model.theta, (len(beliefs), 1)),
        ))

    def set_env_param(self,
        env_param: EnvParam,
        gamma: Optional[float] = None,
        ent_coef: Optional[float] = None,
    ) -> None:
        _rcParams = Config(rcParams.get('agent.DistilledBeliefAgent.set_env_param'))
        if gamma is None:
            gamma = _rcParams.gamma
            print("Use default discount parameter gamma={:g}".format(gamma))
        if ent_coef is None:
            ent_coef = _rcParams.ent_coef
            print("Use default curiosity parameter ent_coef={:g}".format(ent_coef))
        assert len(env_param)==len(self.model.theta)-2, "Incorrect environment parameter length."
        self.model.env.set_param(env_param)
        param = torch.tensor([*env_param, gamma, ent_coef]).to(self.model.theta)
        self.model.theta.data = self.model._param2theta(param[None])[0]

    def train_policy_net(self,
        beliefs: Array,
        params: Array,
        logits: Array,
        **kwargs,
    ):
        r"""Trains update network.

        Args
        ----
        beliefs: (*, belief_dim)
            Beliefs at time t of different agents.
        env_params: (*, param_dim)
            Paramters of different environments.
        logits: (*, action_dim)
            Logits at time t given by policy network of different agents.

        Returns
        -------
        train_stats:
            Statistics of training procedure.

        """
        _rcParams = Config(rcParams.get('agent.DistilledBeliefAgent.train_policy_net'))
        kwargs = Config(kwargs)
        kwargs.fill(_rcParams)
        thetas = self.model._param2theta(torch.tensor(params)).numpy()
        assert not np.any(np.isnan(thetas)), "Environment parameter boundary is incorrectly set."
        return self.policy.regress(logits, [beliefs, thetas], **kwargs)

    def sga_inference(self,
        observations: Array, actions: Array,
        init_env_param: Optional[EnvParam] = None,
        init_gamma: Optional[float] = None,
        init_ent_coef: Optional[float] = None,
        mask: Optional[Collection[bool]] = None,
        segment_len: Optional[int] = None,
        lr: Optional[float] = None,
        momentum: Optional[float] = None,
        num_steps: Optional[int] = None,
    ):
        _rcParams = Config(rcParams.get('agent.DistilledBeliefAgent.sga_inference'))
        if init_env_param is None:
            init_env_param = self.model.get_param()
            print("Use current environment parmater as initial point.")
        if mask is None:
            mask = torch.tensor([1]*len(init_env_param)+[0, 0]).to(self.model.theta)
        else:
            assert len(mask)==len(self.model.theta), "Incorrect mask length."
            mask = torch.tensor([bool(m) for m in mask]).to(self.model.theta)
        if segment_len is None:
            segment_len = int(0.6*len(actions))
        lr = lr or _rcParams.lr
        momentum = momentum or _rcParams.momentum
        num_steps = num_steps or _rcParams.num_steps

        self.set_env_param(init_env_param, init_gamma, init_ent_coef)
        theta_0 = self.model.theta.data
        theta = torch.nn.Parameter(theta_0.clone())
        optimizer = torch.optim.SGD([theta], lr=lr, momentum=momentum)
        thetas, lls = [], [] # parameter and log likelihood trajectory
        for _ in range(num_steps):
            _, logps = self.compute_likelihoods(observations, actions)
            idx = self.model.rng.choice(len(actions)-segment_len)
            loss = -logps[idx:(idx+segment_len)].sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self.model.theta = theta*mask+theta_0*(1-mask)
            thetas.append(theta.data.clone())
            lls.append(logps.sum().item())
        thetas = torch.stack(thetas)
        params = self.model._theta2param(thetas).cpu().numpy()
        env_params = params[:, :-2]
        gammas = env_params[:, -2]
        ent_coefs = env_params[:, -1]
        lls = np.array(lls)
        return env_params, gammas, ent_coefs, lls


# TODO add memoryless agent and omniscient agent
