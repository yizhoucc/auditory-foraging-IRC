# irc-gym
Inverse rational control (IRC) package based on Gym

## Requirements
* `python==3.9`
* `gym==0.21.0`
* `stable-baselines3==1.7.0`
* `jarvis>=0.6.2`

## Installation
Run the `pip` install commands at the repository folder
```bash
pip install .
```

## Usage

### Verify a custom environment is compatible
The IRC package assumes that each environment is parameterized, and the continuous parameters are
summarized by a parameter vector. It needs the custom environment satisfies as a `gym` environment
with additional attributes and methods implemented, listed below:
- `state_space`
- `get_param() -> env_param: Collection[float]`
- `set_param(env_param: Collection[float])`
- `get_state() -> state: StateType`
- `set_state(state: StateType)`

We provide a foraging task as an example environment `irc.examples.FoodBoxesEnv`, in which an agent
moves in a room containing several food boxes. Color cue outside each food box is associated with
the food state, and the agent needs to decide which box to move to and open to gather food as much
as faster. A simplified version of this environment is provided as `irc.examples.IdenticalBoxesEnv`,
in which box properties are the same. More details can be found in the doc strings.
```python
from irc.examples import IdenticalBoxesEnv

env = IdenticalBoxesEnv()
observation, info = env.reset(seed=0)

# state space of the environment
print(f'state_space: {env.state_space}')
# get and set environment parameter
env_param = env.get_param()
print(f'environment parameter: {env_param}')
env.set_param(env_param)
# get and set environment state
state = env.get_state()
print(f'environment state: {state}')
env.set_state(state)
```

#### Supported state space
At this moment, only discrete spaces are supported for state, observation and action. `irc` requires
the environment `state_space` and `observation_space` to be `gym.spaces.MultiDiscrete`, while
`action_space` to be `gym.spaces.Discrete`.

### Initialize a manager for rational agents
A manager saves the configurations and training checkpoints of rational agents at the local machine.
The default path is `'irc_store'` at the working directory.

The environment class of agent internal models along with the non-continuous parameters needs to be
specified at manager initialization via the argument `defaults`. The argument `defaults` can be a
dictionary whose key `'env._target_'` is a string of the environment class, or it can be a yaml file
name that contains a dictionary of such structure.
```python
from irc.manager import AgentManager

defaults = {'env._target_': 'irc.examples.IdenticalBoxesEnv'}
manager = AgentManager(defaults=defaults)
```

### Train and inspect a rational agent
A rational agent is optimal with respect to its assumed model about the environment. `irc` uses
standard reinforcement learning algorithms from `stable-baselines3` to train one.
```python
agent, key = manager.train_agent() # using default environment parameter

env_param = (0.2, 0.05, 0.8, 0.1, 10., -1.)
agent, key = manager.train_agent(env_param=env_param, num_epochs=10)
```

It takes about 5 mins to initialize the belief update network and 40 secs per training epoch to
fine-tune the belief network and learn the policy on a laptop with i7-1280P CPU.

Each agent is assigned with a unique string as key, and we can use it to inspect the training
progress.
```python
agent, fig = manager.inspect_agent(key)
```

After the agent is thoroughly trained, we can run it in a given environment that does not have to be
the same as its assumed one.
```python
# if `env` is not specified, agent interacts with its assumed environment
episode = agent.run_one_episode()

# run the agent in an environment different from assumption
env = IdenticalBoxesEnv()
episode = agent.run_one_episode(env=env, max_steps=60)
```

The dictionary `episode` contains actions, observations, rewards, states and beliefs at each time
step of an episode, along with probabilities of queried states of the assumed environment. More
details can be found in the documentation of `run_one_episode` method.

### Sweep over a grid of environment parameters
`irc` computes episode likelihood conditioned on different assumed environments, and infer the most
likely environment from the results. The feature is implemented by a method `train_agents` that
sweeps over a family of rational agents characterized by a grid of environment parameters. The
sweeping can be run in parallel on a cluster of machines using orchestration tools such as
`kubernetes`.
```python
param_grid = [
    [0.1, 0.2, 0.3],    # p_appear
    [],                 # p_vanish
    [0.6, 0.8],         # p_cue
    [],                 # lambda_center
    [10., 5.],          # r_food
    [],                 # r_move
]
seeds = range(2)
manager.train_agents_on_grid(param_grid=param_grid, seeds=seeds, num_epochs=10)
```

### Monitor the training progress of agents
The overall status of agents defined by an environment parameter grid trained in parallel can be
viewed using `monitor_agents_on_grid` method, which takes similar input arguments as
`train_agents_on_grid`.
```python
report = manager.overview_agents_on_grid(env_param_grid=env_param_grid, seeds=seeds)
```

### Compute likelihood of episode data
Given a trained agent, we can compute the likelihood of a sequence of actions $a_{1:t}$ and
observations $o_{1:t}$ as $P(a_{1:t}, o_{1:t}) = \prod_t \pi(a_t|b_t)$, in which $b_t$ is the belief
at time $t$ and $\pi(\cdot|\cdot)$ is the policy of the agent. Note that $b_t$ is updated from
$b_{t-1}$ with $a_{t-1}$ and $o_t$.

Time series of beliefs $b_t$ and log likelihoods $\ln \pi(a_t|b_t)$ is computed by the agent method
`compute_likelihoods`.
```python
beliefs, logps = agent.compute_likelihoods(observations, actions)
```

### Distill a single agent from a family of agents
When enough number of agents are trained for different environment parameters, we can distill a
single master agent that takes environment parameter as additional input to mimic behavior of each
individual agents.
```python
d_agent, train_stats = manager.distill_agents_on_grid(param_grid, save_path='wukong.pt')
```
When a `save_path` is provided, the trained agent will be saved in the file. If the file already
exists, the training will resume from the saved state.

### Inferring environment parameter with distilled agent
When the distilled agent is prepared, it can be used to estimate parameters of the agent's internal
model from the behavior data. The method `sga_inference` uses stochastic gradient ascent method to
find the maximum likelihood estimation.
```python
d_agent.sga_inference(observations, actions, init_env_param)
print(d_agent.model.get_param())
```

### Use custom belief functions
*Not tested for 0.3.1*

`irc` does not require the user to provide belief update functions. Instead, the default belief
model `irc.model.SamplingBeliefModel` uses a sampling-based approach to update belief vectors, and
later trains a belief update network as a proxy.

While this method is both flexible in representing distributions and fast in execution, it is albeit
an approximation method. If the user can provide a belief update function, `irc` can thus make use
of it by replacing `SamplingBeliefModel` with `FuncBeliefModel`.

To be compatible of `FuncBeliefModel`, a few additional methods of the environment are needed:
- `init_belief(observation: ObsType) -> belief: Array`
- `update_belief(belief: Array, action: int, observation: ObsType) -> belief: Array`
- `sample_state(belief: Array) -> state: StateType`
- `query_probs(belief: Array, states: Iterable[StateType]) -> probs: Array` (optional)

To use the `FuncBeliefModel` class, the user can simply change the `defaults` argument as in:
```python
defaults = {
    'env._target_': 'irc.examples.FoodBoxesEnv',
    'model._target_': 'irc.model.FuncBeliefModel',
}
manager = AgentManager(defaults=defaults)
```
