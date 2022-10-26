import os, argparse, json
from irc import BeliefAgentFamily
from irc.examples import FoodBoxEnv # to change to your custom environment
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 15, 'lines.linewidth': 2, 'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 1200})
import pickle

if __name__=='__main__':
    bafam = BeliefAgentFamily(FoodBoxEnv)
    env_param = (0.1, 0.8, 10.)
    seed = 0
    config = bafam.to_config(env_param=env_param, seed=seed)
    _, ckpt, _ = bafam.load_ckpt(config)
    num_episodes, num_steps = 10, 40
    epochs, r_means, r_stds = [], [], []
    for epoch, val in ckpt['eval_records'].items():
        epochs.append(epoch)
        r_means.append(np.mean(val['returns']))
        r_stds.append(np.std(val['returns']))
    epochs = np.array(epochs)
    r_means = np.array(r_means)
    r_stds = np.array(r_stds)

    print("Training progress of reinforcement learning")
    print("Agent is evaluated on {} episodes at each checkpoint, and return calculated over {} time steps".format(num_episodes, num_steps))
    _, ax = plt.subplots(figsize=(5, 2.5))
    ax.plot(epochs, r_means, marker='s')
    ax.fill_between(epochs, r_means-r_stds/num_episodes**0.5, r_means+r_stds/num_episodes**0.5, alpha=0.2)
    ax.set_xlabel('RL epochs')
    ax.set_ylabel('Episode return')
    ax.set_title('PPO with internal model')
    plt.show()
