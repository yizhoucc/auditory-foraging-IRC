import numpy as np
import torch
from typing import Optional, Union
from collections.abc import Callable, Iterable
from scipy.special import logsumexp
from jarvis.utils import numpy_dict, tensor_dict

from .model import BaseBeliefModel, SamplingBeliefModel
from .alias import Array, GymEnv, SB3Algo


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
            The reinfocement learning algorithm for learning the policy taking
            belief as input.

        """
        self.model = model
        self.algo = algo

    def state_dict(self):
        r"""Returns state dictionary."""
        return {
            'model_state': self.model.state_dict(),
            'policy_state': numpy_dict(self.algo.policy.state_dict()),
        }

    def load_state_dict(self, state):
        r"""Loads state dictionary."""
        self.model.load_state_dict(state['model_state'])
        self.algo.policy.load_state_dict(tensor_dict(state['policy_state'], self.algo.device))

    @staticmethod
    def _return(gamma, rewards):
        r"""Returns cumulative discounted reward."""
        w = gamma**np.flip(np.arange(len(rewards)))
        g = (w*rewards).sum()
        return g

    def evaluate(self, num_episodes: int = 10, num_steps: int = 40) -> dict:
        r"""Evaluates current policy with respect to internal model.

        Args
        ----
        num_episodes:
            Number of evaluation episodes.
        num_steps:
            Number of time steps per episode.

        Returns
        -------
        eval_record:
            Evaluation record, used for keeping track of training progress.

        """
        returns = [] # cumulative discounted rewards of all episodes
        if isinstance(self.model, SamplingBeliefModel):
            optimalities = [] # belief update optimalities
        for _ in range(num_episodes):
            episode = self.run_one_episode(num_steps=num_steps)
            rewards = episode['rewards']
            returns.append(self._return(self.algo.gamma, rewards))
            if isinstance(self.model, SamplingBeliefModel):
                optimalities.append(np.nanmean(episode['optimalities']))
        eval_record = {
            'num_episodes': num_episodes,
            'num_steps': num_steps,
            'returns': returns,
        }
        if isinstance(self.model, SamplingBeliefModel):
            eval_record['optimalities'] = optimalities
        return eval_record

    def run_one_episode(self,
        env: Optional[GymEnv] = None,
        num_steps: int = 40,
        q_states: Union[Callable[[GymEnv], Iterable[Array]], Iterable[Array], None] = None,
    ) -> dict:
        r"""Runs one episode.

        Args
        ----
        env:
            The actual environment to interact with, if not provided, the agent
            interacts with the internal model.
        num_steps:
            Maximum number of time steps of each episode.
        q_states: (num_queries, *state_dim)
            Query states for presenting the belief. Probabilities of each
            queried state represented by the belief vector will be returned. It
            can also be a callable function that takes the current environment
            as input, because sometimes the states of interest is dynamically
            decided based on current observation.

        Returns
        -------
        episode:
            Results of one episode.

        """
        _to_restore_train = self.algo.policy.training # policy will be set to evaluation mode temporarily
        self.algo.policy.set_training_mode(False)
        
        actions, rewards, states, observations, beliefs = [], [], [], [], []

        if q_states is None:
            get_queries = lambda env: env.query_states()
        elif isinstance(q_states, Iterable):
            get_queries = lambda env: q_states
        else:
            assert callable(q_states)
            get_queries = q_states
        try:
            _q_states = get_queries(self.model.env)
            _q_states, _q_probs = [], []
        except:
            get_queries = None
        if isinstance(self.model, SamplingBeliefModel):
            optimalities, fvus = [], []

        belief, info = self.model.reset(env, return_info=True)
        states.append(info['state'])
        observations.append(info['observation'])
        beliefs.append(belief)
        if get_queries is not None:
            _q_states.append(np.array(get_queries(self.model.env)))
            _q_probs.append(self.model.query_probs(_q_states[-1]))
        t = 0
        while True:
            action, _ = self.algo.predict(belief)
            action = int(action) # SB3 policy returns an array
            actions.append(action)
            belief, reward, done, info = self.model.step(action, env)
            rewards.append(reward)
            states.append(info['state'])
            observations.append(info['observation'])
            beliefs.append(belief)
            if get_queries is not None:
                _q_states.append(np.array(get_queries(self.model.env)))
                _q_probs.append(self.model.query_probs(_q_states[-1]))
            if isinstance(self.model, SamplingBeliefModel):
                optimalities.append(self.model.p_s.est_stats['optimality'])
                fvus.append(self.model.p_s.est_stats['fvu'])
            t += 1
            if done or t==num_steps:
                break
        
        episode = {
            'num_steps': t,
            'actions': np.array(actions), # [0, t)
            'rewards': np.array(rewards), # [0, t)
            'states': np.array(states), # [0, t]
            'observations': np.array(observations), # [0, t]
            'beliefs': np.array(beliefs), # [0, t]
        }

        if get_queries is not None:
            diffs = ((_q_states-_q_states[0])**2).reshape(len(_q_states), -1).sum(axis=1)
            if np.all(diffs<1e-8): # merge fixed query set
                _q_states = _q_states[0]
            episode['q_states'] = np.array(_q_states) # (num_queries, state_dim, t+1) or (num_queries, state_dim)
            episode['q_probs'] = np.array(_q_probs) # (num_queries, t+1)
        if isinstance(self.model, SamplingBeliefModel):
            episode['optimalities'] = np.array(optimalities), # [1, t]
            episode['fvus'] = np.array(fvus), # [1, t]
        self.algo.policy.set_training_mode(_to_restore_train)
        return episode

    def episode_likelihood(self,
        actions: Array,
        observations: Array,
        seed: int = 0,
    ) -> float:
        r"""Returns the likelihood of given episode.

        Args
        ----
        actions: (num_steps,)
            Actions taken by the agent, in [0, t).
        observations: (num_steps+1, observation_dim)
            Observations to the agent, in [0, t].
        seed:
            Random seed to diversify belief trajectories.

        Returns
        -------
        logp:
            Log likelihood log(p(actions, observations|agent)) for a sampled
            belief trajectory.

        """
        device = self.model.p_s.get_param_vec().device
        self.algo.policy.eval().to(device)
        self.model.seed(seed)

        logps = []
        for t in range(len(actions)):
            observation = observations[t] # last observation will not be used
            if t==0:
                with torch.no_grad():
                    self.model.p_s.set_param_vec(
                        self.model.p_s_o.param_net(np.array(observation)[None])[0]
                    )
            else:
                self.model.update_belief(actions[t-1], observation)
            belief = self.model.p_s.get_param_vec()
            pi = self.algo.policy.get_distribution(belief[None].to(device))
            logps.append(pi.log_prob(torch.tensor(actions[t], dtype=torch.long, device=device)).item())
        return sum(logps)


    # Lokesh modified this method for incorporating explicit belief update
    def episode_likelihood(self,
        actions: Array,
        observations: Array,
        seed: int = 0,
    ) -> float:
        r"""Returns the likelihood of given episode.

        Args
        ----
        actions: (num_steps,)
            Actions taken by the agent, in [0, t).
        observations: (num_steps+1, observation_dim)
            Observations to the agent, in [0, t].
        seed:
            Random seed to diversify belief trajectories.

        Returns]'
        -------
        logp:
            Log likelihood log(p(actions, observations|agent)) for a sampled
            belief trajectory.

        """
        # device = self.model.p_s.get_param_vec().device
        device = self.model.device

        self.algo.policy.eval().to(device)
        self.model.seed(seed)

        logps = []
        for t in range(len(actions)):
            observation = observations[t] # last observation will not be used
            # print((observation[0],))
            if t==0:

                # with torch.no_grad():
                #     self.model.p_s.set_param_vec(
                #         self.model.p_s_o.param_net(np.array(observation)[None])[0]
                #     )
                self.model.init_belief((observation[0],))
                # print(f' belief here is {self.model.belief}')
            else:
                # self.model.update_belief(actions[t-1], observation)
                self.model.update_belief(actions[t-1], observation)
                # print(f' belief here is {self.model.belief}')
            
            # belief = self.model.p_s.get_param_vec()

            

            # pi = self.algo.policy.get_distribution(belief[None].to(device))
            pi = self.algo.policy.get_distribution(torch.from_numpy(self.model.belief)[None].to(device))

            logps.append(pi.log_prob(torch.tensor(actions[t], dtype=torch.long, device=device)).item())
        return sum(logps)


# Lokesh added this method to check if polcicies are consistent
    def agent_action_distribution(self,
            beliefs = None,
        ) -> list:
            r"""Returns action distributions for given belief vectors.

            Args
            ----
            beliefs:
                A numpy array containting belief vectors.

            Returns
            -------
            action_distributions:
                A list contatining action distributions for corresponding belief vectors.

            """
            device = self.model.device
            self.algo.policy.eval().to(device)
            # self.model.seed(seed)

            # _to_restore_train = self.algo.policy.training # policy will be set to evaluation mode temporarily
            # self.algo.policy.set_training_mode(False)
            if beliefs is None:
                raise NotImplementedError("Current version requires you to input the beliefs.")
            action_distributions = []
            for belief in beliefs:
                pi = self.algo.policy.get_distribution(torch.from_numpy(belief)[None].to(device))
                action_distributions.append([np.exp(pi.log_prob(torch.tensor(action, dtype=torch.long, device=device)).item()) for action in range(self.model.action_space.n)])
            # self.algo.policy.set_training_mode(_to_restore_train)
            return action_distributions