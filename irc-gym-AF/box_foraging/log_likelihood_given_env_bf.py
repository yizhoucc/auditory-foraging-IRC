episode_path = 'saved_episode.pickle'

import matplotlib.pyplot as plt
from irc import BeliefAgentFamily
from irc.examples import FoodBoxEnv # to change to your custom environment
bafam = BeliefAgentFamily(FoodBoxEnv)


from scipy.special import logsumexp

env_param = (0.05, 0.8, 10.)
max_seed, num_repeats = 2, 3
logps = bafam.episode_likelihood(env_param, episode_path, seeds=range(max_seed), num_repeats=num_repeats)

_, ax = plt.subplots(figsize=(5, 3))
h = ax.imshow(logps, cmap='summer')
# plt.colorbar(h, label=r'$\lnp(\bar{o},\bar{a}|\pi,\bar{b})$')
ax.set_xlabel('Belief traces index')
ax.set_ylabel('Agent index')
ax.set_title('Mean log likelihood {:.2f}'.format(logsumexp(logps)-np.log(logps.size)))
plt.show()