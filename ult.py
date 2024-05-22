from scipy.signal import savgol_filter
import numpy as np
import pandas as pd
import torch
import scipy.stats as stats
from matplotlib import pyplot as plt
from collections import OrderedDict, defaultdict, Counter
import seaborn as sns
import os
import pickle
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
import multiprocess
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from contextlib import contextmanager, ExitStack, redirect_stderr, redirect_stdout
# ---notification------
import requests
import configparser
config = configparser.ConfigParser()
config.read_file(open('privateconfig'))
token = config['Notification']['token']


def notify(msg='plots ready', group='lab', title='plot'):
    notification = "https://api.day.app/{}/{}/{}?group={}".format(
        token, title, msg, group)
    requests.get(notification)


# ---plot configs------

font_dirs = ['fonts/computer-modern', ]
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
for font_file in font_files:
    font_manager.fontManager.addfont(font_file)
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['mathtext.default'] = 'regular'
plt.rcParams['pdf.fonttype'] = 42

cmaps = OrderedDict()
cmaps['Qualitative'] = ['Pastel1', 'Pastel2', 'Paired', 'Accent',
                        'Dark2', 'Set1', 'Set2', 'Set3',
                        'tab10', 'tab20', 'tab20b', 'tab20c']

plt.rcParams.update({
    'font.size': 15, 'lines.linewidth': 2,
    'xtick.labelsize': 13, 'ytick.labelsize': 13,
    'axes.spines.top': False, 'axes.spines.right': False,
    'savefig.dpi': 1200,
})


# color choice --------------
lick_color, attn_color='tab:orange', 'tab:blue'
sig_color='black'
noise_color='white'
hit_color='green'
fa_color='red'
miss_color='grey'


start_rgb = (0.0, 1, 0.7) # low reward
end_rgb = (0.0, 0.2, 0.2) # high reward
rewardcmap = LinearSegmentedColormap.from_list('rewardmap', [start_rgb, end_rgb])

colors = [attn_color, 'white', lick_color]
stops = [0.0, 0.5, 1.0]
attnlickcmap = LinearSegmentedColormap.from_list(
    'attmap', list(zip(stops, colors)))

start_rgb = (0.2, 0.7, 0.9) # low reward
end_rgb = (0.2, 0.2, 0.2) # high reward
pcmap = LinearSegmentedColormap.from_list('pcmap', [start_rgb, end_rgb])
# pcmap(np.linspace(0, 1, len(plist)))

start_rgb = (0.6, 0.1, 0.5) # low 
end_rgb = (0.0, 0.2, 0.2) # high 
nodecmap = LinearSegmentedColormap.from_list('nodecmap', [start_rgb, end_rgb])


def find_block_lengths(numbers):
    block_lengths = []
    current_length = 1

    for i in range(1, len(numbers)):
        if numbers[i] == numbers[i - 1] + 1:
            current_length += 1
        else:
            block_lengths.append(current_length)
            current_length = 1

    # Append the length of the last block
    block_lengths.append(current_length)

    return block_lengths

@contextmanager
def suppress(out=True, err=False):
    with ExitStack() as stack:
        with open(os.devnull, "w") as null:
            if out:
                stack.enter_context(redirect_stdout(null))
            if err:
                stack.enter_context(redirect_stderr(null))
            yield


def quicksave(name, modelname='default', fig=None):
    '''save to pdf. 
    name, name of figure.
    modelname, name of the folder'''
    directory = f'fig/{modelname}'
    if not os.path.exists(directory):
        os.makedirs(directory)
    if not fig:
        plt.savefig(f'{directory}/{name}.pdf',
                    dpi='figure', format='pdf', bbox_inches="tight")
    else:
        fig.savefig(f'{directory}/{name}.pdf',
                    dpi='figure', format='pdf', bbox_inches="tight")


def process_one_episode(episode, task):
    '''process the episode data into result lists'''
    hit_count = 0
    miss_count = 0
    false_alarm_count = 0
    noise_time_before_lick = 0
    signal_time_before_lick = 0
    total_noise_time = 0
    total_signal_time = 0
    total_reward = 0

    attention_time_points_across_episodes = []
    episode_length_across_episodes = []
    hit_reaction_time_across_episodes = []
    fa_reaction_time_across_episodes = []

    # process single ep data
    if episode['actions'][-1][0] == 1:
        if episode['states'][-1][0] > task.no_signal_nodes + task.no_penalty_nodes:
            hit_count += 1
            signal_time_before_lick_curr_episode = count_no_elements(
                episode['states'], 1, task.no_signal_nodes)
            hit_reaction_time_across_episodes.append(
                signal_time_before_lick_curr_episode)
            signal_time_before_lick += signal_time_before_lick_curr_episode
        else:
            false_alarm_count += 1
            noise_time_before_lick_curr_episode = count_no_elements(
                episode['states'], 0, 0)
            fa_reaction_time_across_episodes.append(
                noise_time_before_lick_curr_episode)
            noise_time_before_lick += noise_time_before_lick_curr_episode
    else:
        miss_count += 1
    total_signal_time += count_no_elements(
        episode['states'], 1, task.no_signal_nodes)
    total_noise_time += count_no_elements(episode['states'], 0, 0)
    total_reward += sum(episode['rewards'])
    # attention_time_points_across_episodes.append(np.where(episode['actions'] % env.no_attention_modes > 0)[0])
    attention_time_points_across_episodes.append(
        np.where(np.array([elt[1] for elt in episode['actions']]) == 1)[0])
    episode_length_across_episodes.append(len(episode['states']))
    # end process single ep data
    food_reward_idx = episode['trial_food_reward_idx']
    return (food_reward_idx,
            hit_count,
            miss_count,
            false_alarm_count,
            noise_time_before_lick,
            signal_time_before_lick,
            total_noise_time,
            total_signal_time,
            total_reward,
            attention_time_points_across_episodes,
            episode_length_across_episodes,
            hit_reaction_time_across_episodes,
            fa_reaction_time_across_episodes
            )


def count_no_elements(np_array, min_allowed, max_allowed):
    return len([elt for elt in list(np_array.flatten()) if elt >= min_allowed and elt <= max_allowed])


def run_one_episode(task, taskbelief, agent,
                    num_steps=1000, deterministic=True):
    '''modified run one ep function.'''
    q_states = [[i] for i in range(task.no_nodes)]

    actions, rewards, states, observations, beliefs = [], [], [], [], []
    trial_food_reward = []

    def get_queries(env): return q_states
    try:
        _q_states = get_queries(taskbelief.env)
        _q_states, _q_probs = [], []
    except:
        get_queries = None
    task.reset()
    p1p2 = task.obs_certainity_possible
    belief, info = taskbelief.reset(task, return_info=True)
    states.append(info['state'])
    observations.append(info['observation'])
    beliefs.append(belief)
    if get_queries is not None:
        _q_states.append(np.array(get_queries(taskbelief.env)))
        _q_probs.append(taskbelief.query_probs(_q_states[-1]))
    t = 0
    while True:

        action, _ = agent.predict(belief, deterministic=deterministic)
        # action, _ = agent.predict(belief)
        action = action

        actions.append(action)
        belief, reward, done, info = taskbelief.step(action, task)
        rewards.append(reward)
        states.append(info['state'])
        observations.append(info['observation'])
        beliefs.append(belief)
        if get_queries is not None:
            _q_states.append(np.array(get_queries(taskbelief.env)))
            _q_probs.append(taskbelief.query_probs(_q_states[-1]))
        t += 1
        if done or t == num_steps:
            break

    episode = {
        'trial_food_reward_idx': task.food_reward_idx,
        'num_steps': t,
        'actions': np.array(actions),  # [0, t)
        'rewards': np.array(rewards),  # [0, t)
        'states': np.array(states),  # [0, t]
        'observations': np.array(observations),  # [0, t]
        'beliefs': np.array(beliefs),  # [0, t]
        'p1p2': p1p2
    }

    if get_queries is not None:
        diffs = ((_q_states-_q_states[0]) **
                 2).reshape(len(_q_states), -1).sum(axis=1)
        if np.all(diffs < 1e-8):  # merge fixed query set
            _q_states = _q_states[0]
        # (num_queries, state_dim, t+1) or (num_queries, state_dim)
        episode['q_states'] = np.array(_q_states)
        episode['q_probs'] = np.array(_q_probs)  # (num_queries, t+1)

    return episode


def find_activation(agent, belief):
    features = torch.tensor(belief, dtype=torch.float32)
    with torch.no_grad():
        policy_net_0_results = agent.policy.mlp_extractor.policy_net[0](
            features)
        policy_net_1_results = agent.policy.mlp_extractor.policy_net[1](
            policy_net_0_results)
        policy_net_2_results = agent.policy.mlp_extractor.policy_net[2](
            policy_net_1_results)
        policy_net_3_results = agent.policy.mlp_extractor.policy_net[3](
            policy_net_2_results)

    return policy_net_3_results


def pad_lists(list_of_lists):
    max_length = max(len(lst) for lst in list_of_lists)
    padded_lists = [lst + [0] * (max_length - len(lst))
                    for lst in list_of_lists]
    return padded_lists


def pad_zero_lick(data):
    num_rows = len(data)
    num_cols = max(data) + 1
    grid = np.zeros((num_rows, num_cols))
    for i, arr in enumerate(data):
        grid[i, arr] = 1
    return grid


def pad_zero_attention(data):
    num_rows = len(data)
    num_cols = 50
    try:
        num_cols = max(max(arr)
                       for arr in data if arr is not None and len(arr) != 0) + 1
    except:
        pass
    grid = np.zeros((num_rows, num_cols))
    for i, arr in enumerate(data):
        grid[i, arr] = 1
    return grid


def offset_to_first(data):
    offset_data = [[x - min(inner_list) for x in inner_list]
                   for inner_list in data]
    return offset_data


def find_gap(arr):
    '''find gap in binary arr (for each row)'''
    gap_lengths = []
    for row in arr:
        count = 0
        for value in row:
            if value == 0:
                count += 1
            elif value == 1:
                if count > 0:
                    gap_lengths.append(count)
                    count = 0
    return np.array(gap_lengths)


def gap_histogram_all_rows(arr):
    gap_lengths = []
    for row in arr:
        count = 0
        for value in row:
            if value == 0:
                count += 1
            elif value == 1:
                if count > 0:
                    gap_lengths.append(count)
                    count = 0

    # Compute histogram for all gap lengths together
    if gap_lengths==[]:
        return # None for attn all time

    histogram = np.histogram(gap_lengths, bins=np.arange(1, max(gap_lengths) + 2))
    return histogram


def plot_over_episodes(data, title, plot_no_episodes=100):
    '''modified, add color bar'''
    if title == 'attention time':

        plt.figure()
        plt.plot(np.sum(pad_zero_attention(data), axis=0)/len(data))
        plt.xlabel('time in trial')
        plt.ylabel('normalized count')
        plt.title(title)
        plt.show()

        plot_data = data[:plot_no_episodes]
        trial_lens = [len(a) for a in data]
        sortidx = np.argsort(trial_lens)
        plt.figure()
        # c=plt.imshow(pad_zero_attention(plot_data)[sortidx], cmap='viridis',
        #            aspect='auto', interpolation='none')
        c = plt.imshow(pad_zero_attention(plot_data), cmap='viridis',
                       aspect='auto', interpolation='none')
        plt.colorbar(c)
        plt.xlabel('time in trial')
        plt.ylabel('episode no.')
        plt.title(title)
        plt.show()

        plt.figure()
        plt.plot(np.sum(pad_zero_attention(
            offset_to_first(data)), axis=0)/len(data))
        plt.xlabel('time in trial')
        plt.ylabel('normalized count')
        plt.title('offset ' + title)
        plt.show()

        plot_data = offset_to_first(data[:plot_no_episodes])
        plt.figure()
        c = plt.imshow(pad_zero_attention(plot_data), cmap='viridis',
                       aspect='auto', interpolation='none')
        plt.colorbar(c)
        plt.xlabel('time in trial')
        plt.ylabel('episode no.')
        plt.title('offset' + title)
        plt.show()

        # plt.figure()
        # tot_sum = np.sum(pad_zero_attention(offset_to_first(data)), axis = 0)/len(data)
        # plt.plot(tot_sum[2:])
        # plt.xlabel('time in trial')
        # plt.ylabel('normalized count')
        # plt.title('offset and removed first ' + title)
        # plt.show()

        plt.figure()
        histogram = gap_histogram_all_rows(pad_zero_attention(data))
        plt.bar(histogram[1][:-1], histogram[0], width=0.8, align='center')
        plt.xlabel('Gap Length')
        plt.ylabel('Frequency')
        plt.title('Histogram of gaps between subsequent attentions')
        plt.show()

    elif title == 'lick time':

        plt.figure()
        plt.plot(np.sum(pad_zero_lick(data), axis=0)/len(data))
        plt.xlabel('time in trial')
        plt.ylabel('normalized count')
        plt.title(title)
        plt.show()

        plot_data = data[:plot_no_episodes]
        plt.figure()
        c = plt.imshow(pad_zero_lick(plot_data), cmap='viridis',
                       aspect='auto', interpolation='none')
        plt.colorbar(c)
        plt.xlabel('time in trial')
        plt.ylabel('episode no.')
        plt.title(title)
        plt.show()

    else:

        plt.figure()
        plt.plot(np.sum(pad_lists(data), axis=0)/len(data))
        plt.xlabel('time in trial')
        plt.ylabel('normalized count')
        plt.title(title)
        plt.show()

        plot_data = data[:plot_no_episodes]
        plt.figure()
        c = plt.imshow(pad_lists(plot_data), cmap='viridis',
                       aspect='auto', interpolation='none')
        plt.colorbar(c)
        plt.xlabel('time in trial')
        plt.ylabel('episode no.')
        plt.title(title)
        plt.show()


def find_activation(agent, belief):
    '''modified, with only 3 activations instead of 4'''
    features = torch.tensor(belief, dtype=torch.float32)
    with torch.no_grad():
        policy_net_0_results = agent.policy.mlp_extractor.policy_net[0](
            features)
        policy_net_1_results = agent.policy.mlp_extractor.policy_net[1](
            policy_net_0_results)
        policy_net_2_results = agent.policy.mlp_extractor.policy_net[2](
            policy_net_1_results)
        policy_net_3_results = agent.policy.mlp_extractor.policy_net[3](
            policy_net_2_results)

    return policy_net_3_results


def plot_hit_miss_FA(food_reward_list, hit_count_list, miss_count_list, false_alarm_list, title=''):
    '''modified to probability'''
    f = plt.figure()
    hit_count_list, miss_count_list, false_alarm_list = np.array(
        hit_count_list), np.array(miss_count_list), np.array(false_alarm_list)
    hit_prob = [hit_count_list[i]/(hit_count_list[i]+miss_count_list[i] +
                                   false_alarm_list[i]) for i in range(len(hit_count_list))]
    miss_prob = [miss_count_list[i]/(hit_count_list[i]+miss_count_list[i] +
                                     false_alarm_list[i]) for i in range(len(hit_count_list))]
    fa_prob = [false_alarm_list[i]/(hit_count_list[i]+miss_count_list[i] +
                                    false_alarm_list[i]) for i in range(len(hit_count_list))]

    plt.plot(food_reward_list, hit_prob, '-*g')
    plt.plot(food_reward_list, miss_prob, '-*b')
    plt.plot(food_reward_list, fa_prob, '-*r')
    plt.legend(['hit', 'miss', 'false alarm'])
    plt.xlabel('food reward')
    plt.ylabel('probability')
    plt.xticks(food_reward_list, [f'{a:.0f}' for a in food_reward_list])
    plt.title(title)
    plt.show()
    return f


def plot_hit_miss(food_reward_list, hit_count_list, miss_count_list, title=''):
    '''hit and miss, no fa'''
    plt.figure()
    hit_count_list, miss_count_list = np.array(
        hit_count_list), np.array(miss_count_list)
    hit_prob = [hit_count_list[i]/(hit_count_list[i]+miss_count_list[i])
                for i in range(len(hit_count_list))]
    miss_prob = [miss_count_list[i]/(hit_count_list[i]+miss_count_list[i])
                 for i in range(len(hit_count_list))]
    plt.plot(food_reward_list, hit_prob, '-*g')
    plt.plot(food_reward_list, miss_prob, '-*b')
    plt.legend(['hit', 'miss', 'false alarm'])
    plt.xlabel('food reward')
    plt.ylabel('probability')
    plt.xticks(food_reward_list, [f'{a:.0f}' for a in food_reward_list])
    plt.title(title)
    plt.show()


def plot_hit_fa(food_reward_list, hit_count_list, false_alarm_list, title=''):
    '''hit and fa'''
    plt.figure()
    hit_count_list, false_alarm_list = np.array(
        hit_count_list), np.array(false_alarm_list)
    hit_prob = [hit_count_list[i]/(hit_count_list[i]+false_alarm_list[i])
                for i in range(len(hit_count_list))]
    fa_prob = [false_alarm_list[i]/(hit_count_list[i]+false_alarm_list[i])
               for i in range(len(hit_count_list))]

    plt.plot(food_reward_list, hit_prob, '-*g')
    plt.plot(food_reward_list, fa_prob, '-*r')
    plt.legend(['hit', 'false alarm'])
    plt.xlabel('food reward')
    plt.ylabel('probability')
    plt.xticks(food_reward_list, [f'{a:.0f}' for a in food_reward_list])
    plt.title(title)
    plt.show()


def find_blocks(alist):
    '''return att blocks in (s,e)'''
    blocks=[]
    cur=-2
    for a in alist:
        # print(a,cur, blocks)
        if a==cur+1: # cont
            blocks[-1][-1]=a
        else:
            blocks.append([a,a])
        cur=a
    return blocks

def find_gaps(alist):
    '''find gaps in att list'''
    blocks=find_blocks(alist)
    gaps=[]
    for (_,s),(e,_) in zip(blocks, blocks[1:]):
        gaps.append(e-s)
    return gaps
    # print(gaps)

def find_blocksize(alst):
    '''find att block size. 
    alst: attention list such as [10,22,23,45]'''
    lst=np.diff(alst)
    one_indices = np.where(lst == 1)[0]
    blocks = np.split(one_indices, np.where(np.diff(one_indices) != 1)[0] + 1)
    block_sizes=[len(block)+1 for block in blocks if block.size > 0]
    return block_sizes



def count_less_equal_index(lst):
    '''count number of trial end at i'''
    result = []
    for i in range(max(lst)):
        count = sum(1 for x in lst if x == i)
        result.append(count)
    return np.array(result)


def smooth_list(lst, window_size=3, polynomial_order=1):
    '''smoothing'''
    return savgol_filter(lst, window_size, polynomial_order)


def previous_block_gap(lst):
    '''we define block as continus int. this function finds the gap of each item's block to previous block end'''
    res = [0]
    r = 0
    for i in lst:
        if i == r+1:  # continues in block
            # the answer should be the same as begining of the block
            res.append(res[-1])
        else:  # not in block. r is previous block end
            res.append(i-r)
        r = i
    return res[1:]

def trialsb(b):
    sb=b[:-1,1:26]
    sb=np.sum(sb,axis=1)
    return sb

def trialnextatt(alist, b):
    r=0
    nextatt=[]
    for t in range(len(b)):
        while r<len(alist)-1 and t>alist[r]:
            r+=1
        # print(r, t)
        nextatt.append(alist[r]-t)
    nextatt=np.array(nextatt)
    return nextatt

def previous_item_gap(lst, remove_first=True):
    '''we define block as continus int. this function finds the gap of each item to previous item. '''
    if remove_first:  # remove begining of trial as an attention
        return np.diff(lst)
    res=np.diff(np.append([0], lst))
    res=res[res!=1]
    return res

def find_blockgap(alst):
    '''find att block gap. 
    alst: attention list such as [10,22,23,45]'''
    lst=np.diff(alst)
    one_indices = np.where(lst == 1)[0]
    blocks = np.split(one_indices, np.where(np.diff(one_indices) != 1)[0] + 1)
    block_sizes=[len(block)+1 for block in blocks if block.size > 0]
    res=[]
    for e,s in zip(blocks, blocks[1:]):
        res.append(s[0]-s[-1])
    return np.array(res)

def find_blocksize(alst):
    '''find att block size. 
    alst: attention list such as [10,22,23,45]'''
    lst=np.diff(alst)
    one_indices = np.where(lst == 1)[0]
    blocks = np.split(one_indices, np.where(np.diff(one_indices) != 1)[0] + 1)
    block_sizes=[len(block)+1 for block in blocks if block.size > 0]
    # alst, lst, one_indices,block_sizes
    return block_sizes

def previous_block_size(lst):
    '''we define block as continus int. this function finds the previous block size'''
    block_starts = [i for i in range(
        len(lst)) if i == 0 or lst[i] != lst[i-1] + 1]
    block_sizes = []
    for i in range(len(lst)):
        current_block_index = next(
            (index for index in block_starts[::-1] if index <= i), None)
        if current_block_index is not None:
            prev_block_index = next(
                (index for index in block_starts[::-1] if index < current_block_index), None)
            if prev_block_index is not None:
                block_sizes.append(current_block_index - prev_block_index)
            else:
                # If no previous block found, length is up to current block start.
                block_sizes.append(current_block_index + 1)
        else:
            block_sizes.append(0)  # If not in any block, length is 0.
    return block_sizes


def compute_action_probs(belief, agent):
    def find_lick_prob(pi):
        prob_val = 0
        prob_val += find_action_comb_probs(lick_choice=1, att_choice=0, pi=pi)
        prob_val += find_action_comb_probs(lick_choice=1, att_choice=1, pi=pi)
        return prob_val

    def find_att_prob(pi):
        prob_val = 0
        prob_val += find_action_comb_probs(lick_choice=0, att_choice=1, pi=pi)
        prob_val += find_action_comb_probs(lick_choice=1, att_choice=1, pi=pi)
        return prob_val

    def find_action_comb_probs(lick_choice, att_choice, pi):
        return np.exp(pi.log_prob(torch.tensor([np.array([lick_choice, att_choice])], dtype=torch.long, device='cpu')).item())

    pi = agent.policy.get_distribution(torch.tensor(belief)[None].to('cpu'))
    lick_prob = find_lick_prob(pi)
    att_prob = find_att_prob(pi)
    return lick_prob, att_prob


def compute_action_probs(belief, agent):
    '''same function rewrite just for clarity'''
    pi = agent.policy.get_distribution(torch.tensor(belief)[None].to('cpu'))
    nolicknoatt = np.exp(pi.log_prob(torch.tensor(
        [np.array([0, 0])], dtype=torch.long, device='cpu')).item())
    nolickatt = np.exp(pi.log_prob(torch.tensor(
        [np.array([0, 1])], dtype=torch.long, device='cpu')).item())
    licknoatt = np.exp(pi.log_prob(torch.tensor(
        [np.array([1, 0])], dtype=torch.long, device='cpu')).item())
    lickatt = np.exp(pi.log_prob(torch.tensor(
        [np.array([1, 1])], dtype=torch.long, device='cpu')).item())
    lick_prob = licknoatt+lickatt
    att_prob = nolickatt+lickatt
    return lick_prob, att_prob


def get_att_lick_probs_new(episode, agent):
    '''new from lokesh may 6th'''
    lick_probs = []
    att_probs = []
    time = []
    noise_probs = []
    start_time = 1

    # Last belief is not used for action
    for ind in range(start_time, len(episode['beliefs'][:-1])):
        belief = episode['beliefs'][:-1][ind]
        lick_prob, att_prob = compute_action_probs(belief, agent)
        lick_probs.append(lick_prob)
        att_probs.append(att_prob)
        noise_probs.append(belief[0])
        time.append(ind)

    return att_probs, lick_probs, noise_probs, time


def compute_autocorrelation(sequence):
    autocorr = np.correlate(sequence, sequence, mode='full')
    autocorr /= np.max(autocorr)
    return autocorr


def open_pickle_file(file_name):
    open_file = open(file_name, "rb")
    data = pickle.load(open_file)
    open_file.close()
    return data


def save_pickle_file(file_name, data):
    directory = os.path.dirname(file_name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)


def assign_state_class(true_state, no_signal_nodes, no_penalty_nodes):
    if true_state == 0:
        state_class = true_state
    elif true_state <= no_signal_nodes:
        state_class = 1
    elif true_state <= no_signal_nodes + no_penalty_nodes:
        state_class = 2
    else:
        state_class = 3
    return state_class


def plot_AF_episode(episode, env, agent, nodes_from_zero = 20, time_steps_before_lick = 10):        
    obs_certainity_possible = env.obs_certainity_possible
    dict_action_possible = env.dict_action_possible #{key: (lick_choice,attention_choice)}
    no_attention_modes = env.no_attention_modes

    #for episodic
    no_signal_nodes = env.no_signal_nodes
    no_penalty_nodes = env.no_penalty_nodes
    no_ITI_nodes = env.no_ITI_nodes
    no_nodes = env.no_nodes

    licking_action_keys = [action_key for action_key in dict_action_possible if dict_action_possible[action_key][0] == 1]
    attention_action_keys = []
    for attention_choice in range(no_attention_modes):
        attention_action_keys.append([action_key for action_key in dict_action_possible if dict_action_possible[action_key][1] == attention_choice])
    
    num_steps = episode['num_steps']
    states = episode['states']
    observations = episode['observations']
    actions = episode['actions']
    rewards = episode['rewards']
    probs = episode['q_probs']
    num_states = states.max()+1

    attention_color_list = ['blue','blueviolet','indigo','magenta','darkcyan','cyan']

    offset_actions = np.vstack((actions, [0,0]))
    offset_rewards = np.append(rewards, 0) #offset actions to match time steps, as it looks like the rewards at last time step is not computed.

    correct_lick_choice_idxs = list(map(lambda action_choice: True if action_choice[0] == 1 else False, offset_actions))&(offset_rewards>=0)
    wrong_lick_choice_idxs = list(map(lambda action_choice: True if action_choice[0] == 1 else False, offset_actions))&(offset_rewards<0)
    attention_choice_idxs = []
    for attention_choice in range(no_attention_modes):

        attention_choice_idxs.append(list(map(lambda action_choice: True if action_choice[1] == 1 else False, offset_actions)))
    env_plotter = PlotHelper(num_steps, no_attention_modes, correct_lick_choice_idxs, wrong_lick_choice_idxs, attention_choice_idxs, obs_certainity_possible, num_states)    
    figs = []
    
    state_classes = np.array([[assign_state_class(true_state, no_signal_nodes, no_penalty_nodes)] for true_state in states.flatten()])
    fig = env_plotter.make_episode_plot(plot_variable = state_classes, label = 'State class', feature_color_list = ['khaki','green','red','maroon'], attention_color_list = attention_color_list)
    figs.append(fig)
    fig = env_plotter.make_episode_plot(plot_variable = observations, label = 'Observation', feature_color_list = ['black','white','red','maroon'], attention_color_list = attention_color_list)
    figs.append(fig)
    fig = env_plotter.make_episode_plot(plot_variable = probs, label = 'Belief', for_belief = True, attention_color_list = attention_color_list)    
    figs.append(fig)


    return figs


class PlotHelper():
    def __init__(self, num_steps, no_attention_modes, correct_lick_choice_idxs, wrong_lick_choice_idxs, attention_choice_idxs, obs_certainity_possible, num_states, figsize=(15, 3),bbox_to_anchor=(1.5, 1.05)):
        self.num_steps = num_steps
        self.no_attention_modes = no_attention_modes
        self.correct_lick_choice_idxs = correct_lick_choice_idxs
        self.wrong_lick_choice_idxs = wrong_lick_choice_idxs
        self.attention_choice_idxs = attention_choice_idxs
        self.obs_certainity_possible = obs_certainity_possible
        self.num_states = num_states
        self.figsize = figsize
        self.bbox_to_anchor = bbox_to_anchor

    def make_episode_plot(self, plot_variable, label, feature_color_list = [], for_belief = False, attention_color_list = ['blue','blueviolet','indigo','magenta','darkcyan','cyan']):
        fig_w, fig_h = self.figsize
        aspect = self.num_steps*fig_h/fig_w*1.5
        fig, ax = plt.subplots(figsize=self.figsize) 
        attention_choice_lists = []
        attention_choice_text = []
        for attention_choice in range(self.no_attention_modes):
            attention_choice_lists.append(ax.scatter(np.arange(self.num_steps+1)[self.attention_choice_idxs[attention_choice]], .3 * np.ones(sum(self.attention_choice_idxs[attention_choice])), color=attention_color_list[attention_choice], marker='o', s=50))
            attention_choice_text.append(f'Obs. conf. {self.obs_certainity_possible[attention_choice]}')
        
        correct_lick_choice = ax.scatter(np.arange(self.num_steps+1)[self.correct_lick_choice_idxs], -0.3 * np.ones(sum(self.correct_lick_choice_idxs)), color='darkorange', marker='^', s=50)
        wrong_lick_choice = ax.scatter(np.arange(self.num_steps+1)[self.wrong_lick_choice_idxs], -0.3 * np.ones(sum(self.wrong_lick_choice_idxs)), color='darkorange', marker='v', s=50)
        feature_colors = ListedColormap(feature_color_list)
        if for_belief:
            h = ax.imshow(plot_variable.T, aspect=aspect, extent=[-0.5, self.num_steps+0.5, -0.5, 0.5], vmin=0, vmax=1, origin='lower', cmap='gist_gray',)
            cbar = plt.colorbar(h, label=label)
            cbar.set_ticks(np.arange(2))
            ax.set_yticks(1/self.num_states * np.arange(self.num_states) - 0.5 + 0.5 * 1/self.num_states)
            ax.set_yticklabels([f'{i}' for i in range(self.num_states)])
        else:
            h = ax.imshow(plot_variable.T, aspect=aspect, extent=[-0.5, self.num_steps+0.5, -0.5, 0.5], vmin=np.min(plot_variable) - 0.5, vmax=np.max(plot_variable) + 0.5, origin='lower', cmap=feature_colors)

            # print('problem somwhere here')
            # print(plot_variable)
            # print(np.where(plot_variable == 2))


            cbar = plt.colorbar(h, label=label)
            cbar.set_ticks(np.arange(np.min(plot_variable), np.max(plot_variable)+1))
            ax.set_yticks([])
            ax.set_ylabel('')
        action_legend_list = [correct_lick_choice,wrong_lick_choice] + attention_choice_lists
        ax.legend(action_legend_list, ['correct_lick_choice','wrong_lick_choice']+attention_choice_text, bbox_to_anchor=self.bbox_to_anchor, fontsize=12)
        ax.set_xlim([-0.5, self.num_steps+0.5])
        ax.set_xticks([0, self.num_steps])
        ax.set_xlabel('Time')
        return fig
    
    #for episodic
    def make_hist_plots(self, offset_actions, observations, states, no_signal_nodes, no_attention_modes):
        flatten_observations = observations.flatten()
        flatten_states = states.flatten()
        fig_w, fig_h = self.figsize
        fig, axs = plt.subplots(3,2,figsize=(fig_w, 6*fig_h))
        axs[0,0].hist(offset_actions[np.flatnonzero(flatten_observations == 3)]%no_attention_modes, bins = np.arange(-.5,no_attention_modes,1))
        axs[0, 0].set(xticks = range(no_attention_modes), xlabel = 'attention levels', ylabel = 'count', title = 'attention during ITI')
        axs[0,1].hist((offset_actions[np.flatnonzero(flatten_observations == 3)]>= no_attention_modes).astype(int), bins = np.arange(-.5,2,1))
        axs[0, 1].set(xticks = range(2), xlabel = 'lick choice', ylabel = 'count', title = 'lick during ITI')
        axs[1, 0].hist(offset_actions[np.flatnonzero(flatten_states == 0)]%no_attention_modes, bins = np.arange(-.5,no_attention_modes,1))
        axs[1, 0].set(xticks = range(no_attention_modes), xlabel = 'attention levels', ylabel = 'count', title = 'attention during noise')
        axs[1, 1].hist((offset_actions[np.flatnonzero(flatten_states == 0)]>= no_attention_modes).astype(int), bins = np.arange(-.5,2,1))
        axs[1, 1].set(xticks = range(2), xlabel = 'lick choice', ylabel = 'count', title = 'lick during noise')
        axs[2, 0].hist(offset_actions[np.flatnonzero((flatten_states >= 1) & (flatten_states <= no_signal_nodes))]%no_attention_modes, bins = np.arange(-.5,no_attention_modes,1))
        axs[2, 0].set(xticks = range(no_attention_modes), xlabel = 'attention levels', ylabel = 'count', title = 'attention during signal')
        axs[2, 1].hist((offset_actions[np.flatnonzero((flatten_states >= 1) & (flatten_states <= no_signal_nodes))]>= no_attention_modes).astype(int), bins = np.arange(-.5,2,1))
        axs[2, 1].set(xticks = range(2), xlabel = 'lick choice', ylabel = 'count', title = 'lick during signal')
        return fig
    
    def policy_for_gaussian_beliefs(self, agent, no_nodes, dict_action_possible, licking_action_keys, attention_action_keys, no_signal_nodes, mu_max = None, sigma_max_factor = 6, mu_length = 20, sigma_length = 10):
        no_signal_and_noise_nodes = no_signal_nodes + 1
        mu_max = no_signal_and_noise_nodes-1 if mu_max == None else mu_max
        sigma_max = mu_max/sigma_max_factor
        mu_vector = np.linspace(0, mu_max, mu_length)
        sigma_vector = np.linspace(0.1, sigma_max, sigma_length)
        belief_matrix = np.zeros((mu_length, sigma_length, no_nodes))
        action_prob_matrix = np.zeros((mu_length, sigma_length, len(dict_action_possible)))
        lick_prob_matrix = np.zeros((mu_length, sigma_length))
        attention_prob_matrix = np.zeros((mu_length, sigma_length, len(attention_action_keys)))
        for mu_ind in range(mu_length):
            for sigma_ind in range(sigma_length):
                for belief_ind in range(0, no_signal_nodes+1):
                    belief_matrix[mu_ind, sigma_ind, belief_ind] = 1/np.sqrt(2 * np.pi * sigma_vector[sigma_ind]**2) * np.exp(-.5 * ((belief_ind-mu_vector[mu_ind])/sigma_vector[sigma_ind])**2)
                belief_matrix[mu_ind, sigma_ind, :] /= np.sum(belief_matrix[mu_ind, sigma_ind, :])
                action_prob_matrix[mu_ind, sigma_ind, :] = agent.agent_action_distribution(np.array([belief_matrix[mu_ind, sigma_ind, :]]))[0]
        for lick_key in licking_action_keys:
            lick_prob_matrix += action_prob_matrix[:, :, lick_key]
        for attention_choice in range(len(attention_action_keys)):
            for attention_keys in attention_action_keys[attention_choice]:
                attention_prob_matrix[:,:,attention_choice] += action_prob_matrix[:, :, attention_keys]
        
        fig_w, fig_h = self.figsize

        no_xticks = int(no_nodes/20)
        no_yticks = int(mu_length/5)
        no_rows_in_subplot = 10
        xticks = np.around(np.linspace(0, no_signal_and_noise_nodes-1, no_xticks)).astype(int)
        yticks = np.around(np.linspace(0, len(mu_vector)-1, no_yticks)).astype(int)
        fig1, axs = plt.subplots(no_rows_in_subplot,figsize=(1.5*fig_w, 15*fig_h))
        for sigma_ind in range(len(sigma_vector)):
            h = axs[sigma_ind].imshow(belief_matrix[:, sigma_ind, :no_signal_and_noise_nodes])
            cbar = plt.colorbar(h, label= 'prob.')
            cbar.set_ticks(np.round([np.min(belief_matrix[:, sigma_ind, :no_signal_and_noise_nodes]), np.max(belief_matrix[:, sigma_ind, :no_signal_and_noise_nodes])],4))
            axs[sigma_ind].set(xticks = xticks, xticklabels = np.around(np.arange(no_nodes)[xticks],1), xlabel = 'node index')
            axs[sigma_ind].set(yticks = yticks, yticklabels = np.around(mu_vector[yticks],1), ylabel = 'mean (node)')
            axs[sigma_ind].set(title = f's.t.d of {round(sigma_vector[sigma_ind],2)}')


        no_xticks = int(sigma_length/2)
        no_yticks = int(mu_length/2)
        no_rows_in_subplot = 2
        no_cols_in_subplot = 3
        xticks = np.around(np.linspace(0, len(sigma_vector)-1, no_xticks)).astype(int)
        yticks = np.around(np.linspace(0, len(mu_vector)-1, no_yticks)).astype(int)
        plt.figure()
        fig2 = plt.imshow(lick_prob_matrix)
        plt.colorbar(label= 'prob.', ticks = np.round([np.min(lick_prob_matrix), np.max(lick_prob_matrix)],2))
        plt.xlabel('s.t.d (node)')
        plt.ylabel('mean (node)')
        plt.xticks(ticks = xticks, labels = np.around(sigma_vector[xticks],1))
        plt.yticks(ticks = yticks, labels = np.around(mu_vector[yticks],1))
        plt.title(f'prob. of licking')

        
        fig3, axs = plt.subplots(no_rows_in_subplot,no_cols_in_subplot,figsize=(1.2*fig_w, 4.75*fig_h))
        for attention_choice in range(len(attention_action_keys)):
            row_ind = attention_choice//no_cols_in_subplot
            col_ind = attention_choice%no_cols_in_subplot
            h = axs[row_ind, col_ind].imshow(attention_prob_matrix[:,:,attention_choice])
            cbar = plt.colorbar(h, label= 'prob.', shrink = .7)
            cbar.set_ticks(np.round([np.min(attention_prob_matrix[:,:,attention_choice]), np.max(attention_prob_matrix[:,:,attention_choice])],2))
            axs[row_ind, col_ind].set(xticks = xticks, xticklabels = np.around(sigma_vector[xticks],1), xlabel = 's.t.d (node)')
            axs[row_ind, col_ind].set(yticks = yticks, yticklabels = np.around(mu_vector[yticks],1), ylabel = 'mean (node)')
            axs[row_ind, col_ind].set(title = f'prob. of attention {attention_choice}')
        
        return fig1, fig2, fig3

    def policy_for_all_signal_noise_durations(self, agent, env, states, probs, licking_action_keys, attention_action_keys, no_signal_nodes, no_additional_episodes = 0, attention_color_list = ['blue','blueviolet','indigo','magenta','darkcyan','cyan']):
        no_signal_and_noise_nodes = no_signal_nodes + 1
        episodes_states = states
        episodes_beliefs = probs
        for _ in range(no_additional_episodes):
            episode = agent.run_one_episode(env=env, num_steps=10000, q_states = [[i] for i in range(env.no_nodes)])
            episodes_states = np.concatenate((episodes_states,episode['states']))
            episodes_beliefs = np.concatenate((episodes_beliefs,episode['q_probs']))
        chosen_time = [i for i in range(len(episodes_states)) if episodes_states[i] < no_signal_and_noise_nodes]
        chosen_beliefs = episodes_beliefs[chosen_time]
        sorted_indices = np.argsort(np.sum(chosen_beliefs[:,1:no_signal_and_noise_nodes],1))
        sorted_signal_prob = np.sum(chosen_beliefs[:,1:no_signal_and_noise_nodes],1)[sorted_indices]
        sorted_beliefs = chosen_beliefs[sorted_indices]

        action_prob_matrix = np.array([agent.agent_action_distribution(np.array([sorted_beliefs[index, :]]))[0] for index in range(len(sorted_beliefs))])
        lick_prob_matrix = np.zeros(len(sorted_beliefs))
        attention_prob_matrix = np.zeros((len(sorted_beliefs), len(attention_action_keys)))
        for lick_key in licking_action_keys:
            lick_prob_matrix += action_prob_matrix[:,lick_key]
        for attention_choice in range(len(attention_action_keys)):
            for attention_keys in attention_action_keys[attention_choice]:
                attention_prob_matrix[:,attention_choice] += action_prob_matrix[:, attention_keys]
        # plt.figure(figsize = self.figsize)
        # fig1 = plt.imshow(sorted_beliefs[:,0:no_signal_and_noise_nodes].T)
        # plt.colorbar(label= 'prob.', ticks = np.round([np.min(sorted_beliefs[:,:no_signal_and_noise_nodes]), np.max(sorted_beliefs[:,:no_signal_and_noise_nodes])],2), shrink = .5)
        # plt.xlabel('belief index')
        # plt.ylabel('node index')
        # plt.title(f'Sampled beliefs')
        # plt.figure(figsize = self.figsize)
        # fig2 = plt.plot(sorted_signal_prob, 'o-')
        # plt.xlabel('belief index')
        # plt.ylabel('signal probability')
        plt.figure()
        fig3 = plt.plot(sorted_signal_prob, lick_prob_matrix, 'o-', color = 'r')
        plt.xlabel('signal belief')
        plt.title(f'prob. of licking')
        for attention_choice in range(len(attention_action_keys)):
            plt.plot(sorted_signal_prob, attention_prob_matrix[:,attention_choice], 'o-', color = attention_color_list[attention_choice])
        plt.legend(['lick']+[f'attention {attention_choice}' for attention_choice in range(len(attention_action_keys))], bbox_to_anchor=(1.5, 1.05), fontsize=12)
        plt.xlabel('signal prob.')
        plt.ylabel('prob.')
        plt.title('')
        plt.figure(figsize = self.figsize)
        # fig4 = plt.plot(lick_prob_matrix, 'o-', color = 'r')
        # plt.xlabel('belief index')
        # plt.title(f'prob. of licking')
        # for attention_choice in range(len(attention_action_keys)):
        #     plt.plot(attention_prob_matrix[:,attention_choice], 'o-', color = attention_color_list[attention_choice])
        # plt.legend(['lick']+[f'attention {attention_choice}' for attention_choice in range(len(attention_action_keys))], bbox_to_anchor=(1.5, 1.05), fontsize=12)
        # plt.xlabel('belief index')
        # plt.ylabel('prob.')
        # plt.title('')
        # return fig1, fig2, fig3, fig4
        return fig3

    def count_success_streak(self, agent, env, states, no_signal_nodes, no_additional_episodes = 10):
        no_signal_and_noise_nodes = no_signal_nodes + 1
        episodes_states = states
        success_streak_list = [len([i for i in range(len(episodes_states)) if episodes_states[i] < no_signal_and_noise_nodes and episodes_states[i-1] >= no_signal_and_noise_nodes])]
        success_rate_list = [success_streak_list[-1]/len(episodes_states)]
        episode_time_list = [len(episodes_states)]
        for _ in range(no_additional_episodes):
            episode = agent.run_one_episode(env=env, num_steps=10000, q_states = [[i] for i in range(env.no_nodes)])
            episodes_states = episode['states']
            episodes_beliefs = episode['q_probs']
            success_streak_list.append(len([i for i in range(len(episodes_states)) if episodes_states[i] < no_signal_and_noise_nodes and episodes_states[i-1] >= no_signal_and_noise_nodes]))
            success_rate_list.append(success_streak_list[-1]/len(episodes_states))
            episode_time_list.append(len(episodes_states))
        average_success_streak = sum(success_streak_list)/(no_additional_episodes+1)
        average_success_rate = sum(success_rate_list)/(no_additional_episodes+1)
        average_episode_time = sum(episode_time_list)/(no_additional_episodes+1)
        return average_success_streak, average_success_rate, average_episode_time, success_streak_list, success_rate_list, episode_time_list
    
    def policy_for_signal_noise_durations(self, agent, states, observations, probs, licking_action_keys, attention_action_keys, no_signal_nodes, nodes_from_zero, time_steps_before_lick, attention_color_list = ['blue','blueviolet','indigo','magenta','darkcyan','cyan']):
        no_signal_and_noise_nodes = no_signal_nodes + 1
        episodes_states = states
        episodes_beliefs = probs
        chosen_start_time = [i for i in range(len(episodes_states)) if episodes_states[i] < no_signal_and_noise_nodes and episodes_states[i-1] >= no_signal_and_noise_nodes]
        chosen_end_time = [i for i in range(len(episodes_states)) if episodes_states[i] < no_signal_and_noise_nodes and episodes_states[i+1] >= no_signal_and_noise_nodes]
        no_acquisitions = len(chosen_start_time)
        if no_acquisitions == 1:
            return None
        fig_w, fig_h = self.figsize
        fig, axs = plt.subplots(no_acquisitions, 2, figsize=(1.5*fig_w, 15*fig_h))
        for acquisition_no in range(no_acquisitions):
            start_time = chosen_start_time[acquisition_no]
            end_time = chosen_end_time[acquisition_no]
            chosen_beliefs = episodes_beliefs[start_time:end_time+1][-time_steps_before_lick:]
            
            # chosen_observations = observations[start_time:end_time+1][-time_steps_before_lick:]
            # obs_favoring_noise = [1 if val == 0 else 0 for val in chosen_observations]
            
            action_prob_matrix = np.array([agent.agent_action_distribution(np.array([chosen_beliefs[index, :]]))[0] for index in range(len(chosen_beliefs))])
            lick_prob_matrix = np.zeros(len(chosen_beliefs))
            attention_prob_matrix = np.zeros((len(chosen_beliefs), len(attention_action_keys)))
            for lick_key in licking_action_keys:
                lick_prob_matrix += action_prob_matrix[:,lick_key]
            for attention_choice in range(len(attention_action_keys)):
                for attention_keys in attention_action_keys[attention_choice]:
                    attention_prob_matrix[:,attention_choice] += action_prob_matrix[:, attention_keys]
            h = axs[acquisition_no, 0].imshow(chosen_beliefs[:,0:nodes_from_zero].T)
            plt.colorbar(h, label= 'prob.', shrink = .7)
            axs[acquisition_no, 0].set(xticks = np.arange(time_steps_before_lick), xticklabels = [str(-i) for i in range(time_steps_before_lick-1,-1,-1)], xlabel = 'time')
            axs[acquisition_no, 0].set(yticks = np.arange(nodes_from_zero), ylabel = 'node index')
            axs[acquisition_no, 0].set(title = f'Beliefs before licking')
            
            axs[acquisition_no, 1].plot(lick_prob_matrix, 'v-', color = 'r')
            for attention_choice in range(len(attention_action_keys)):
                axs[acquisition_no, 1].plot(attention_prob_matrix[:,attention_choice], 'o-', color = attention_color_list[attention_choice])
            axs[acquisition_no, 1].plot(np.sum(chosen_beliefs[:,1:no_signal_and_noise_nodes],1), '*-')
            # axs[acquisition_no, 1].stem(obs_favoring_noise, '--')
            axs[acquisition_no, 1].set(xticks = np.arange(time_steps_before_lick), xticklabels = [str(-i) for i in range(time_steps_before_lick-1,-1,-1)], xlabel = 'time')
            axs[acquisition_no, 1].set(yticks = np.arange(0,1,.1), ylabel = 'prob.')
            axs[acquisition_no, 1].set(title = f'Policy before licking')
            axs[acquisition_no, 1].legend(['lick']+[f'attention {attention_choice}' for attention_choice in range(len(attention_action_keys))]+['signal prob.'], bbox_to_anchor=(1.5, 1.05), fontsize=12)
        return fig
    

def centerax(ax):
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')



def empirical_autocorrelation(list_of_seqs, min_no_samples):
    # input is a list of lists, each inner list being a sequence
    '''
    list_of_seqs: list of lists
    min_no_samples: min window?
    '''

    def first_index_above_value(array, value):
        indices = np.where(array >= value)
        if len(indices[0]) == 0:
            return None
        return indices[0][0]

    def pad_lists_with_zeros(lists):
        max_length = max(len(inner_list) for inner_list in lists) 
        padded_lists = []
        for inner_list in lists:
            current_length = len(inner_list)
            total_padding = max_length - current_length
            left_padding = total_padding // 2
            right_padding = total_padding - left_padding
            padded_list = [0] * left_padding + inner_list + [0] * right_padding
            padded_lists.append(padded_list)
        return padded_lists

    def generate_autocorr_count(n):
        no_samples_one_direction = [n - i for i in range(n)]
        no_samples_both_directions = list(reversed(no_samples_one_direction[1:])) + no_samples_one_direction
        return no_samples_both_directions 

    def convert_to_float(list_of_lists):
        return [[float(elt) for elt in sublist] for sublist in list_of_lists]
    
    list_of_seqs = convert_to_float(list_of_seqs)
    list_of_counts = []
    list_of_unorm_corrs = []
    for seq in list_of_seqs:
        unorm_corr = np.correlate(seq, seq, mode='full')
        list_of_unorm_corrs.append(list(unorm_corr))
        list_of_counts.append(generate_autocorr_count(len(seq)))
    total_counts = np.sum(np.array(pad_lists_with_zeros(list_of_counts)), 0)
    autocorr = np.sum(np.array(pad_lists_with_zeros(list_of_unorm_corrs)), 0)/total_counts

    ind = first_index_above_value(total_counts, min_no_samples)
    if ind != 0:
        return autocorr[ind:-ind], total_counts[ind:-ind]
    
    return autocorr, total_counts

# exampel from lokesh
# x = [[1.0, 2.0, 3.0],[4.0, 5.0]]
# print(empirical_autocorrelation(x,0))


def compute_bernoulli_entropy(prob_list):
    entropy_list = [- p * np.log(p) - (1-p) * np.log(1-p) for p in prob_list]
    return entropy_list


def quickleg(ax, loc='lower right', bbox_to_anchor=(0,0)):

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    leg = ax.legend(by_label.values(), by_label.keys(),
                    loc='lower right', bbox_to_anchor=bbox_to_anchor)
    for lh in leg.legendHandles:
        lh.set_alpha(1)


def empirical_autocorrelation_new(list_of_seqs, min_no_samples):
    # input is a list of lists, each inner list being a sequence

    def get_mean_and_variance(list_of_seqs):
        flattened_list = [item for sublist in list_of_seqs for item in sublist]
        np_array = np.array(flattened_list)
        mean = np.mean(np_array)
        variance = np.var(np_array)
        return mean, variance
        
    def normalize(np_array, mean, variance):
        return (np_array - mean**2)/variance
    
    def first_index_above_value(array, value):
        indices = np.where(array >= value)
        if len(indices[0]) == 0:
            return None
        return indices[0][0]

    def pad_lists_with_zeros(lists):
        max_length = max(len(inner_list) for inner_list in lists) 
        padded_lists = []
        for inner_list in lists:
            current_length = len(inner_list)
            total_padding = max_length - current_length
            left_padding = total_padding // 2
            right_padding = total_padding - left_padding
            padded_list = [0] * left_padding + inner_list + [0] * right_padding
            padded_lists.append(padded_list)
        return padded_lists

    def generate_autocorr_count(n):
        no_samples_one_direction = [n - i for i in range(n)]
        no_samples_both_directions = list(reversed(no_samples_one_direction[1:])) + no_samples_one_direction
        return no_samples_both_directions 

    def convert_to_float(list_of_lists):
        return [[float(elt) for elt in sublist] for sublist in list_of_lists]
    
    list_of_seqs = convert_to_float(list_of_seqs)
    mean, variance = get_mean_and_variance(list_of_seqs)
    list_of_counts = []
    list_of_unorm_corrs = []
    for seq in list_of_seqs:
        unorm_corr = np.correlate(seq, seq, mode='full')
        list_of_unorm_corrs.append(list(unorm_corr))
        list_of_counts.append(generate_autocorr_count(len(seq)))
    total_counts = np.sum(np.array(pad_lists_with_zeros(list_of_counts)), 0)
    autocorr = np.sum(np.array(pad_lists_with_zeros(list_of_unorm_corrs)), 0)/total_counts

    ind = first_index_above_value(total_counts, min_no_samples)
    if ind != 0:
        return normalize(autocorr[ind:-ind], mean, variance), total_counts[ind:-ind]
    if np.sum(variance)==0: # all time att
        res= np.zeros_like(autocorr)
        # res[len(res)//2]=1
        return res, total_counts
    return normalize(autocorr, mean, variance), total_counts


