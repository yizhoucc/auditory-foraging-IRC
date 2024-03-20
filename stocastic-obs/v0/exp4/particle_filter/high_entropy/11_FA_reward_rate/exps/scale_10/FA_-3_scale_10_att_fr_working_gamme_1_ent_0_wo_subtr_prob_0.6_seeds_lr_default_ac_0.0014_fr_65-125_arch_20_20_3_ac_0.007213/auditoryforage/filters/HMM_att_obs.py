import os, copy
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from ..utils import save_pickle_file, open_pickle_file

class HMM():
    def __init__(self, agent, env, episode, end_index):
        self.agent = agent
        self.env = env
        self.end_index = end_index
        self.state_list, self.lick_actions = self.extract_data(episode)
        self.observation_matrix = env.find_observation_matrix()
        self.transition_matrix = env.find_transition_matrix()
        self.HMM_IO = {}
        self.HMM_IO['input'] = {}
        self.HMM_IO['input']['root_episode'] = episode
        
    def extract_data(self, episode):
        state_list = [list_of_state[0] for list_of_state in episode['states']]
        lick_actions = [self.env.dict_action_possible[action][0] for action in episode['actions']]
        state_list = state_list[:self.end_index]
        lick_actions = lick_actions[:self.end_index]
        return state_list, lick_actions
    
    def possible_attention_series(self):
        return [list(elt) for elt in product(self.env.attention_possible, repeat=self.end_index)]
    
    def possible_obs_series(self):
        return [list(elt) for elt in product(self.env.observation_possible, repeat=self.end_index)]

    def possible_obs_attention_combs(self):
        return list(product(self.env.observation_possible, self.env.attention_possible))
    
    def possible_obs_attention_series(self):
        return [list(elt) for elt in product(self.possible_obs_attention_combs(), repeat=self.end_index)]
    
    def possible_obs_attention_series_given_attention_series(self, attention_series):
        return [list(list(zip(obs_series, attention_series))) for obs_series in self.possible_obs_series()]
    
    def find_action_key(self, lick, attention):
        return [key for key, val in self.env.dict_action_possible.items() if val == (lick, attention)][0]
    
    def compute_joint(self, obs_attention_series):
        def output_relevant_variables(time):
            prev_lick = self.lick_actions[time-1] if time != 0 else None
            prev_attention = obs_attention_series[time-1][1] if time != 0 else None
            prev_action = [key for key, val in self.env.dict_action_possible.items() if val == (prev_lick, prev_attention)][0] if time != 0 else None
            prev_state = self.state_list[time-1] if time != 0 else None
            current_state = self.state_list[time]
            current_obs = obs_attention_series[time][0]
            current_lick = self.lick_actions[time]
            current_attention = obs_attention_series[time][1]
            current_action = self.find_action_key(current_lick, current_attention)
            return prev_lick, prev_attention, prev_action, prev_state, current_state, current_obs, current_action
        _, _, _, _, current_state, current_obs, current_action = output_relevant_variables(0)
        belief = self.env.init_belief([current_obs])
        action_probs = self.agent.agent_action_distribution(np.array([belief]))[0]
        joint_prob = self.observation_matrix[current_obs, current_state, 0] * action_probs[current_action]
        for time in range(1, len(obs_attention_series)):
            prev_lick, prev_attention, prev_action, prev_state, current_state, current_obs, current_action = output_relevant_variables(time)
            joint_prob *= self.transition_matrix[prev_state, current_state, prev_lick]
            joint_prob *= self.observation_matrix[current_obs, current_state, prev_attention]
            belief = self.env.update_belief(belief, prev_action, current_obs)
            if belief is not None:
                action_probs = self.agent.agent_action_distribution(np.array([belief]))[0]
                joint_prob *= action_probs[current_action]
            else:
                joint_prob = 0
                break
        return joint_prob
    
    def marginalize(self, series_samples):
        marginal = 0
        for series in series_samples:
            marginal += self.compute_joint(series)
        return marginal
    
    def compute_posterior_for_given_attention_series(self, attention_series):
        posterior = self.marginalize(self.possible_obs_attention_series_given_attention_series(attention_series))
        posterior /= self.marginalize(self.possible_obs_attention_series())
        return posterior
    
    def compute_posterior_for_given_obs_attention_series(self, obs_attention_series):
        posterior = self.marginalize([obs_attention_series])
        posterior /= self.marginalize(self.possible_obs_attention_series())
        return posterior
    
    def save_output(self, subfolder):
        store_folder = 'store/filters/' + subfolder + '/'
        if not os.path.exists(store_folder): os.makedirs(store_folder)
        file_name = store_folder + f'HMM_ei_{self.end_index}_rn_{np.random.randint(0,100)}.pkl'
        save_pickle_file(self.HMM_IO, file_name)
        print(f'Saved data to file {file_name}.')
    
    def compute_posterior_across_attention_series(self, do_save = True):
        _to_restore_train = self.agent.algo.policy.training 
        self.agent.algo.policy.set_training_mode(False)
        attention_series_list = self.possible_attention_series()
        posterior_list = []
        for attention_series in attention_series_list:
            posterior_list.append(copy.deepcopy(self.compute_posterior_for_given_attention_series(attention_series)))
        self.agent.algo.policy.set_training_mode(_to_restore_train)
        sorted_indices = list(np.argsort(-1 * np.array(posterior_list)))
        attention_series_list = [attention_series_list[ind] for ind in sorted_indices]
        posterior_list = [posterior_list[ind] for ind in sorted_indices]
        self.HMM_IO['output'] = {}
        self.HMM_IO['output']['attention_series_list'] = attention_series_list
        self.HMM_IO['output']['posterior_list'] = posterior_list
        if do_save: self.save_output('HMM_attention')
        return self.HMM_IO
    
    def compute_posterior_across_attention_obs_series(self, do_save = True):
        _to_restore_train = self.agent.algo.policy.training 
        self.agent.algo.policy.set_training_mode(False)
        obs_attention_series_list = self.possible_obs_attention_series()
        posterior_list = []
        for obs_attention_series in obs_attention_series_list:
            posterior_list.append(copy.deepcopy(self.compute_posterior_for_given_obs_attention_series(obs_attention_series)))
        self.agent.algo.policy.set_training_mode(_to_restore_train)
        sorted_indices = list(np.argsort(-1 * np.array(posterior_list)))
        obs_attention_series_list = [obs_attention_series_list[ind] for ind in sorted_indices]
        posterior_list = [posterior_list[ind] for ind in sorted_indices]
        self.HMM_IO['output'] = {}
        self.HMM_IO['output']['obs_attention_series_list'] = obs_attention_series_list
        self.HMM_IO['output']['posterior_list'] = posterior_list
        if do_save: self.save_output('HMM_att_obs')
        return self.HMM_IO
    
def plot_PF_HMM_comparison(PF_IO, HMM_IO):
    PF_posterior_ranked = PF_IO['output']['sorted_results']['PF_posterior_ranked']
    PF_action_obs_seq_ranked = PF_IO['output']['sorted_results']['action_obs_seq_ranked']
    IRC_based_posterior_ranked = PF_IO['output']['sorted_results']['IRC_based_posterior_ranked']
    HMM_obs_attention_series_list = HMM_IO['output']['obs_attention_series_list']
    
    HMM_obs_attention_seq = []
    for obs_attention_series in HMM_obs_attention_series_list:
        temp_dict = {}
        temp_dict['observations'] = []
        temp_dict['attentions'] = []
        for obs_attention_pair in obs_attention_series:
            temp_dict['observations'].append(copy.deepcopy([obs_attention_pair[0]]))
            temp_dict['attentions'].append(copy.deepcopy(obs_attention_pair[1]))
        HMM_obs_attention_seq.append(copy.deepcopy(temp_dict))
    PF_obs_attention_seq_ranked = copy.deepcopy(PF_action_obs_seq_ranked)
    for dict_elt in PF_obs_attention_seq_ranked:
        dict_elt['attentions'] = list(dict_elt['attentions'])
        dict_elt['observations'] = list(dict_elt['observations'])
        _ = dict_elt.pop('actions')
    HMM_posterior_list = HMM_IO['output']['posterior_list']
    PF_rank_in_HMM = []
    HMM_posterior_arranged_wrt_PF_rank = []
    for PF_obs_attention_seq in PF_obs_attention_seq_ranked:
        HMM_rank = HMM_obs_attention_seq.index(PF_obs_attention_seq)
        PF_rank_in_HMM.append(copy.deepcopy(HMM_rank))
        HMM_posterior_arranged_wrt_PF_rank.append(copy.deepcopy(HMM_posterior_list[HMM_rank]))
    plt.scatter(PF_posterior_ranked, HMM_posterior_arranged_wrt_PF_rank)
    PF_exact_match = np.linspace(min(PF_posterior_ranked), max(PF_posterior_ranked),len(PF_posterior_ranked))
    plt.plot(PF_exact_match, PF_exact_match, 'r')
    plt.xlabel('PF posterior')
    plt.ylabel('HMM posterior')
    plt.show()
    plt.scatter(HMM_posterior_arranged_wrt_PF_rank, IRC_based_posterior_ranked)
    HMM_exact_match = np.linspace(min(HMM_posterior_arranged_wrt_PF_rank), max(HMM_posterior_arranged_wrt_PF_rank),len(HMM_posterior_arranged_wrt_PF_rank))
    plt.plot(HMM_exact_match, HMM_exact_match, 'r')
    plt.xlabel('HMM posterior')
    plt.ylabel('IRC based posterior')
    plt.show()
    plt.plot(PF_posterior_ranked, '-o')
    plt.xlabel('attention + observation sequence')
    plt.ylabel('PF posterior')
    plt.show()
    plt.plot(HMM_posterior_list,'-o')
    plt.xlabel('attention + observation sequence')
    plt.ylabel('HMM posterior')
    plt.show()
    plt.scatter(np.arange(len(PF_rank_in_HMM)), PF_rank_in_HMM)
    plt.plot(np.arange(min(PF_rank_in_HMM),max(PF_rank_in_HMM)+1), 'r')
    plt.xlabel('rank in PF')
    plt.ylabel('rank in HMM')
    plt.show()

def compare_PF_HMM(PF_file_name, HMM_file_name):
    PF_IO = open_pickle_file(PF_file_name)
    HMM_IO = open_pickle_file(HMM_file_name)
    plot_PF_HMM_comparison(PF_IO, HMM_IO)