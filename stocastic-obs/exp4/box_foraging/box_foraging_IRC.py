import os
import sys
sys.path.append(f'{os.getcwd()}/irc_package')

from irc.examples import IdenticalBoxesEnv
env = IdenticalBoxesEnv()
observation, info = env.reset(seed=0)

print(f'state_space: {env.state_space}')
env_param = env.get_param()
print(f'environment parameter: {env_param}')
env.set_param(env_param)
state = env.get_state()
print(f'environment state: {state}')
env.set_state(state)


from irc.manager import AgentManager
defaults = {'env._target_': 'irc.examples.IdenticalBoxesEnv'}
manager = AgentManager(defaults=defaults)

agent = manager.train_agent() # using default environment parameter

#Alternate
env_param = (0.2, 0.05, 0.8, 0.1, 10., -1.)
agent = manager.train_agent(env_param=env_param, num_epochs=10)

# agent, fig = manager.inspect_agent(env_param)

#Alternate
env = IdenticalBoxesEnv()
episode = agent.run_one_episode(env = env, max_steps=60)

env_param_grid = [
    [0.1, 0.2, 0.3],    # p_appear
    [],                 # p_vanish
    [0.6, 0.8],         # p_cue
    [],                 # lambda_center
    [10., 5.],          # r_food
    [],                 # r_move
]
seeds = range(2)
# manager.train_agents(env_param_grid=env_param_grid, seeds=seeds, num_epochs=10)

report = manager.overview_agents(env_param_grid=env_param_grid, seeds=seeds)