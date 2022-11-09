import numpy as np
from scipy.special import logsumexp

episode_path = 'saved_episode.pickle'

import matplotlib.pyplot as plt
from irc import BeliefAgentFamily
from irc.examples import FoodBoxEnv # to change to your custom environment
bafam = BeliefAgentFamily(FoodBoxEnv)


import json
from itertools import product
import time
from jarvis.utils import progress_str, time_str

env_param = (0.05, 0.8, 10.)
max_seed, num_repeats = 2, 3

# with open('jsons/single_box_envs.json', 'r') as f:
#     envs_spec = json.load(f)

#Lokesh commented to check quick example
# p_appear_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
# p_cue_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# r_food_list = [2, 5, 10]
p_appear_list = [0.05, 0.1]
p_cue_list = [0.1, 0.2]
r_food_list = [2, 5]

logps = np.zeros((len(p_appear_list), len(p_cue_list), len(r_food_list)))
print("Fetching episode log likelihood for all environment parameters...")
tic = time.time()
for idx, (i, j, k) in enumerate(product(range(len(p_appear_list)), range(len(p_cue_list)), range(len(r_food_list))), 1):
    env_param = (p_appear_list[i], p_cue_list[j], r_food_list[k])
    _logps = bafam.episode_likelihood(env_param, episode_path, seeds=range(max_seed), num_repeats=num_repeats)
    logps[i, j, k] = logsumexp(_logps)-np.log(_logps.size)
    
    if idx%(-(-logps.size//10))==0 or idx==logps.size:
        print('{} ({})'.format(progress_str(idx, logps.size), time_str(time.time()-tic, idx/logps.size)))

print("true environment parameter:")
print(saved['true_env_spec'])

vmin, vmax = logps.min(), logps.max()
_, axes = plt.subplots(1, len(r_food_list), figsize=(5*len(r_food_list), 3))
for i, ax in enumerate(axes):
    h = ax.imshow(logps[..., i], vmin=vmin, vmax=vmax, cmap='summer')
    ax.set_xticks([0, len(p_cue_list)-1])
    ax.set_xticklabels([p_cue_list[0], p_cue_list[-1]])
    ax.set_xlabel(r'$p_\mathrm{cue}$')
    ax.set_yticks([0, len(p_appear_list)-1])
    if i==0:
        ax.set_yticklabels([p_appear_list[0], p_appear_list[-1]])
        ax.set_ylabel(r'$p_\mathrm{appear}$')
    else:
        ax.set_yticklabels([])
    ax.set_title(r'$r_\mathrm{food}$='+str(r_food_list[i]))
plt.colorbar(h, ax=axes, shrink=0.8, label=r'$\lnp(\bar{o},\bar{a}|\theta)$')
plt.show()