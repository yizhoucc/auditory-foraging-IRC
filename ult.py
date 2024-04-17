import numpy as np
import torch
from matplotlib import pyplot as plt

plt.rcParams.update({
    'font.size': 15, 'lines.linewidth': 2,
    'xtick.labelsize': 13, 'ytick.labelsize': 13,
    'axes.spines.top': False, 'axes.spines.right': False,
    'savefig.dpi': 1200,
})


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
                    num_steps=1000):
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

    belief, info = taskbelief.reset(task, return_info=True)
    states.append(info['state'])
    observations.append(info['observation'])
    beliefs.append(belief)
    if get_queries is not None:
        _q_states.append(np.array(get_queries(taskbelief.env)))
        _q_probs.append(taskbelief.query_probs(_q_states[-1]))
    t = 0
    while True:

        action, _ = agent.predict(belief, deterministic=True)
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


def get_att_lick_probs(episode,agent):
    '''add agent arg'''
    activations_list = []
    for belief in episode['beliefs'][:-1]:  # Last belief is not used for action
        activations_list.append(find_activation(agent, belief).numpy())
    lick_feature = [activations_list[ind][0]
                    for ind in range(len(activations_list))]
    attention_feature = [activations_list[ind][1]
                         for ind in range(len(activations_list))]
    lick_prob = [(1/(1+np.exp(-2 * elt))) for elt in lick_feature]
    attention_prob = [(1/(1+np.exp(-2 * elt))) for elt in attention_feature]
    return attention_prob, lick_prob


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
    num_cols = max(max(arr) for arr in data if len(arr) != 0) + 1
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
    histogram = np.histogram(
        gap_lengths, bins=np.arange(1, max(gap_lengths) + 2))
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
        plt.figure()
        c=plt.imshow(pad_zero_attention(plot_data), cmap='viridis',
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
        c=plt.imshow(pad_zero_attention(plot_data), cmap='viridis',
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
        c=plt.imshow(pad_zero_lick(plot_data), cmap='viridis',
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
        c=plt.imshow(pad_lists(plot_data), cmap='viridis',
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
        policy_net_0_results = agent.policy.mlp_extractor.policy_net[0](features)
        policy_net_1_results = agent.policy.mlp_extractor.policy_net[1](policy_net_0_results)
        policy_net_2_results = agent.policy.mlp_extractor.policy_net[2](policy_net_1_results)
        policy_net_3_results = agent.policy.mlp_extractor.policy_net[3](policy_net_2_results)

    return policy_net_3_results

def plot_hit_miss_FA(food_reward_list, hit_count_list, miss_count_list, false_alarm_list, title = ''):
    '''modified to probability'''
    plt.figure()
    hit_count_list,miss_count_list,false_alarm_list=np.array(hit_count_list),np.array(miss_count_list),np.array(false_alarm_list)
    hit_prob=[hit_count_list[i]/(hit_count_list[i]+miss_count_list[i]+false_alarm_list[i]) for i in range(len(hit_count_list))]
    miss_prob=[miss_count_list[i]/(hit_count_list[i]+miss_count_list[i]+false_alarm_list[i]) for i in range(len(hit_count_list))]
    fa_prob=[false_alarm_list[i]/(hit_count_list[i]+miss_count_list[i]+false_alarm_list[i]) for i in range(len(hit_count_list))]
    
    plt.plot(food_reward_list, hit_prob, '-*g')
    plt.plot(food_reward_list, miss_prob, '-*b')
    plt.plot(food_reward_list, fa_prob, '-*r')
    plt.legend(['hit', 'miss', 'false alarm'])
    plt.xlabel('food reward')
    plt.ylabel('probability')
    plt.xticks(food_reward_list,food_reward_list)
    plt.title(title)
    plt.show()