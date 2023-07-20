import matplotlib.pyplot as plt
import numpy as np
import copy, pickle, os
from itertools import product

class AttentionPosterior():

    ### AFTER COMPUTING ATTENTIONS, MIGHT HAVE TO ALIGN W/ OR W/O LICK DEPENDING ON EPISODE FORMATTING!!

    def __init__(self, agent, env, episode, end_index):
        self.agent = agent
        self.env = env
        self.state_list, self.lick_actions = self.extract_data(episode, end_index)
        self.observation_matrix = env.find_observation_matrix
        self.transition_matrix = env.find_transition_matrix
        
    def extract_data(self, episode, end_index):
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
    
    def possible_obs_attention_series_given_attentions(self, attention_series):
        return [list(list(zip(obs_series, attention_series))) for obs_series in self.possible_obs_series()]
    
    def find_action_key(self, lick, attention):
        return [key for key, val in self.env.dict_action_possible.items() if val == (lick, attention)][0]
    
    def compute_joint(self, obs_attention_series):
        def output_relevant_variables(self, time):
            prev_lick = self.lick_actions[time-1] if time != 0 else None
            prev_attention = obs_attention_series[time-1][1] if time != 0 else None
            prev_action = [key for key, val in self.env.dict_action_possible.items() if val == (prev_lick, prev_attention)][0] if time != 0 else None
            prev_state = self.state_list[time-1] if time != 0 else None
            current_state = self.state_list[time]
            current_obs = obs_attention_series[time][0]
            current_lick = self.lick_actions[time]
            current_attention = obs_attention_series[time][1]
            current_action = self.find_action_key(self, current_lick, current_attention)
            return prev_lick, prev_attention, prev_action, prev_state, current_state, current_obs, current_lick, current_attention, current_action
        _, _, _, _, current_state, current_obs, current_lick, current_attention, current_action = output_relevant_variables(0)
        belief = self.env.init_belief(current_obs)
        action_probs = self.agent.agent_action_distribution(belief)[0]
        joint_prob = self.observation_matrix(current_obs, current_state, 0) * action_probs[current_action]
        for time in range(1, len(obs_attention_series)):
            prev_lick, prev_attention, prev_action, prev_state, current_state, current_obs, current_lick, current_attention, current_action = output_relevant_variables(time)
            joint_prob *= self.transition_matrix(prev_state, current_state, prev_lick)
            joint_prob *= self.observation_matrix(current_obs, current_state, prev_attention)
            belief = self.env.update_belief(self, belief, prev_action, current_obs)
            action_probs = self.agent.agent_action_distribution(belief)[0]
            joint_prob *= action_probs[current_action]
        return joint_prob
    
    def marginalize(self, series_samples):
        marginal = 0
        for series in series_samples:
            marginal += self.compute_joint(series)
        return marginal
    
    def compute_posterior(self):
        _to_restore_train = self.agent.algo.policy.training 
        self.agent.algo.policy.set_training_mode(False)

        self.agent.algo.policy.set_training_mode(_to_restore_train)
    
    def save_output(self, particle_filter_IO):
        pass


    def do_main(self, episode, do_save = True):
            attention_posterior_IO = self.compute_posterior(self.lick_actions, self.state_list)
            attention_posterior_IO['input']['root_episode'] = episode
            if do_save: self.save_output(attention_posterior_IO)
            return attention_posterior_IO
