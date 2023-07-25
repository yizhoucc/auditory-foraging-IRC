#A: Assumption, #C: Concerns, #H: Hacks.
# A1: Assuming we start with ITI.
# H1: We sample at last time step regardless if it's a multiple of sampling frequency or not.
# C1: Might have to have this in tensor
# C2: check if [0] is required, depending on observer_step in new env code. Also note how observe_step comes after env.step.
# overdue, resampling same instant

import matplotlib.pyplot as plt
import numpy as np
import copy, os
from collections import Counter
from .utils import open_pickle_file, save_pickle_file, plot_AF_episode
from itertools import product
import multiprocessing

class ParticleFilter():

    def __init__(self, agent, env, default_no_particles = 100, default_sampling_freq = 1, verbose = False):
        r"""Performs particle filter to generate attention and observation sequences.

        Args
        ----
        agent:
            Agent with learned policy.
        env:
            Environment.
        verbose:
            Flag to indicate if time and other information needs to be printed as the particle filter runs.

        """
        
        self.agent = agent
        self.env = env
        self.no_particles = default_no_particles
        self.sampling_freq = default_sampling_freq
        self.verbose = verbose
                
    def generate_wrt_reference_episode(self, episode, no_particles = None, sampling_freq = None, end_index = None, do_save = True, req_output_posterior = False):
        r"""Performs particle filter to generate attention and observation sequences.

        Args
        ----
        episode:
            Trajectory stored in IRC compatible 'episode dictionary format'.
        no_particles:
            Number of particles to be used in the particle filter.
        sampling_freq:
            Particle filter samples particles once in every sampling_freq steps.
        end_index:
            Up until what point in the episode should be considered for doing particle filter.

        Returns
        -------
        particle_filter_IO:
            Output of filter method.
        """
        no_particles = self.no_particles if no_particles is None else no_particles
        sampling_freq = self.sampling_freq if sampling_freq is None else sampling_freq
        state_list = [list_of_state[0] for list_of_state in episode['states']]
        lick_actions = [self.env.dict_action_possible[action][0] for action in episode['actions']]
        end_index = len(state_list) if end_index is None else end_index
        state_list = state_list[:end_index]
        lick_actions = lick_actions[:end_index]
        particle_filter_IO = self.filter(lick_actions, state_list, no_particles, sampling_freq)
        particle_filter_IO['input']['root_episode'] = episode
        particle_filter_IO = self.compute_posterior(particle_filter_IO) if req_output_posterior else particle_filter_IO
        if do_save: self.save_filter_output(particle_filter_IO, end_index)
        return particle_filter_IO

    def filter(self, lick_actions, state_list, no_particles = None, sampling_freq = None, do_save = False):
        r"""Performs particle filter to generate attention and observation sequences.

        Args
        ----
        lick_actions:
            A list contatining true lick choice time series.
        state_list:
            A list contatining true state time series.
        no_particles:
            Number of particles to be used in the particle filter.
        sampling_freq:
            Particle filter samples particles once in every sampling_freq steps.

        Returns
        -------
        particle_filter_IO:
            A dictionary containing the input and ouput of particle filter stored in values corresponding to keys 'input', and 'output'.
            Most important part of that dictionary being the (sub) key 'particle_filter_output', described below.
            
            particle_filter_output:
                A dictionary contatining the following keys.
                'particles_likelihoods': 
                    A numpy array containing the absolute likelihood value of each particle's trajectory.
                'sampling count tracker': 
                    A dictionary where keys are the time instances where additional sampling was done, and values 
                    represent the number of times sampling had to be repeated. 
                'generated_episodes':
                    A list with each element as a particle's trajectory stored in IRC compatible 'episode dictionary format'. The order of
                    the list is in decreasing order of particles' likelihoods. 
        """
            
        agent = self.agent
        env = self.env
        verbose = self.verbose
        no_particles = self.no_particles if no_particles is None else no_particles
        sampling_freq = self.sampling_freq if sampling_freq is None else sampling_freq
        
        _to_restore_train = agent.algo.policy.training 
        agent.algo.policy.set_training_mode(False)
        observation_matrix = env.find_observation_matrix()
        env.state = state_list[0]
        observation = env.observe_step(0) #A1
        belief = env.init_belief(observation)
        belief_list = [[belief] for _ in range(no_particles)]
        observation = observation[0]
        particles_observation_probs = [1 for _ in range(no_particles)] #A1
        observation_list = [[[observation]] for _ in range(no_particles)]        
        particles_distribution = 1/no_particles * np.ones(no_particles)
        particles_likelihoods = np.ones(no_particles)
        action_list = [[] for _ in range(no_particles)]
        overdue_status = [False for _ in range(no_particles)]
        sampling_count_tracker = {} 
        
        for time in range(len(lick_actions)):
            if verbose: print(time)
            if time != len(lick_actions) - 1: env.state = state_list[time + 1]
            step_particle = True
            sampling_count = 0

            while step_particle:
                
                instant_likelihood = []
                sampling_count += 1
                
                for particle in range(no_particles):
                    if not overdue_status[particle]:
                        action, _ = agent.algo.predict(belief_list[particle][-1]) #C1
                        action_list[particle].append(action.item())
                        instant_action_probs = agent.agent_action_distribution(np.array([belief_list[particle][-1]]))[0]
                        if lick_actions[time] == 1:
                            if action >= env.no_attention_modes:
                                particle_action_prob = instant_action_probs[action]/sum(instant_action_probs[env.no_attention_modes:])
                            else:
                                particle_action_prob = 0
                        elif lick_actions[time] == 0:
                            if action >= env.no_attention_modes:
                                particle_action_prob = 0
                            else: 
                                particle_action_prob = instant_action_probs[action]/sum(instant_action_probs[:env.no_attention_modes])
                        else:
                            raise Exception("Lick actions can only be 0 or 1.")
                        instant_likelihood.append(particles_observation_probs[particle] * particle_action_prob)
                        _, attention_choice = env.dict_action_possible[int(action)]
                        
                        if time != len(lick_actions) - 1: 
                            observation = env.observe_step(attention_choice)[0] #C2
                            observation_list[particle].append([observation])
                            particles_observation_probs[particle] = observation_matrix[observation, env.state, attention_choice]
                            next_belief = env.update_belief(belief_list[particle][-1], action, observation)
                            belief_list[particle].append(next_belief)
                            if next_belief is None:
                                if particle_action_prob != 0:
                                    raise Exception('Error: Liklihood should have been zero when wrong belief update happens!')
                                else:
                                    overdue_status[particle] = True
                    else:
                        action_list[particle].append(None)
                        instant_likelihood.append(0)
                        if time != len(lick_actions) - 1:
                            observation_list[particle].append([None])
                            belief_list[particle].append(None) 
                
                if sum(instant_likelihood) == 0:
                    if verbose: print(f'Need to sample again for time {time}')
                    for particle_ind in range(len(belief_list)):
                        action_list[particle_ind] = action_list[particle_ind][:-1]
                        observation_list[particle_ind] = observation_list[particle_ind][:-1]
                        belief_list[particle_ind] = belief_list[particle_ind][:-1]
                else:
                    step_particle = False
            
            if sampling_count > 1:
                sampling_count_tracker[time] = sampling_count
            
            particles_likelihoods = np.multiply(particles_likelihoods, np.array(instant_likelihood))
            particles_distribution = np.multiply(particles_distribution, np.array(instant_likelihood))
            particles_distribution = particles_distribution/np.sum(particles_distribution)
            
            if time%sampling_freq == 0 or time == len(lick_actions) - 1: #H1
                overdue_status = [False for _ in range(no_particles)]
                temp_observation_list = [[] for _ in range(no_particles)]
                temp_belief_list = [[] for _ in range(no_particles)]
                temp_action_list = [[] for _ in range(no_particles)]
                temp_particles_likelihoods = [[] for _ in range(no_particles)]
                temp_particles_observation_probs = [[] for _ in range(no_particles)]
                for particle in range(no_particles):
                    chosen_particle = np.random.choice(no_particles, p = particles_distribution)
                    temp_observation_list[particle] = copy.deepcopy(observation_list[chosen_particle])
                    temp_belief_list[particle] = copy.deepcopy(belief_list[chosen_particle])
                    temp_action_list[particle] = copy.deepcopy(action_list[chosen_particle])
                    temp_particles_likelihoods[particle] = particles_likelihoods[chosen_particle]
                    temp_particles_observation_probs[particle] = particles_observation_probs[chosen_particle]
                observation_list = copy.deepcopy(temp_observation_list)
                belief_list = copy.deepcopy(temp_belief_list)
                action_list = copy.deepcopy(temp_action_list)
                particles_observation_probs = copy.deepcopy(temp_particles_observation_probs)
                particles_likelihoods = np.array(temp_particles_likelihoods)
                particles_distribution = 1/no_particles * np.ones(no_particles)
        
        agent.algo.policy.set_training_mode(_to_restore_train) 
        
        particle_filter_output = {}
        generated_episodes = []
        sorted_indices = np.argsort(-1 * particles_likelihoods)
        episode_states = np.array([[state] for state in state_list])
        for ind in sorted_indices:
            temp_dict = {}
            temp_dict['states'] = episode_states
            temp_dict['actions'] = np.array(action_list[ind])
            temp_dict['observations'] = np.array(observation_list[ind])
            temp_dict['q_probs'] = np.array(belief_list[ind])
            temp_dict['num_steps'] =  len(temp_dict['actions'])
            generated_episodes.append(temp_dict)
        particle_filter_output['generated_episodes'] = generated_episodes
        particle_filter_output['particles_likelihoods'] = particles_likelihoods[sorted_indices]
        particle_filter_output['sampling_count_tracker'] = sampling_count_tracker
        particle_filter_output['filter_specs'] = {'no_particles': no_particles, 'sampling_freq': sampling_freq}
        particle_filter_IO = self.package_PF_IO(particle_filter_output, state_list, lick_actions)
        if do_save: self.save_filter_output(particle_filter_IO)
        return particle_filter_IO
    
    def package_PF_IO(self, particle_filter_output, input_state_list, input_lick_actions, root_episode = None):
        input_time_series = {'state_list': input_state_list, 'lick_actions': input_lick_actions}
        input = {'root_episode': root_episode, 'input_time_series': input_time_series}
        particle_filter_IO = {'input': input, 'output': particle_filter_output}
        return particle_filter_IO

    def save_filter_output(self, particle_filter_IO, end_index):
        no_particles = particle_filter_IO['output']['filter_specs']['no_particles']
        sampling_freq = particle_filter_IO['output']['filter_specs']['sampling_freq']
        store_folder = 'store/particle_filter/'
        if not os.path.exists(store_folder): os.makedirs(store_folder)
        file_name = store_folder + f'PF_np_{no_particles}_sf_{sampling_freq}_ei_{end_index}_rn_{np.random.randint(0,100)}.pkl'
        save_pickle_file(particle_filter_IO, file_name)
        print(f'Saved data to file {file_name}.')

    def multiple_filtering(self, episode, no_particles_list, sampling_freq_list, end_index, req_output_posterior = False):
        for no_particles in no_particles_list:
            for sampling_freq in sampling_freq_list:
                print(f'Filter with no_particles = {no_particles} and sampling_freq = {sampling_freq} in action.')
                _ = self.generate_wrt_reference_episode(episode, no_particles, sampling_freq, end_index, req_output_posterior = req_output_posterior)

    def multiple_filtering_in_parallel(self, episode, no_particles_list, sampling_freq_list, end_index, req_output_posterior = False):
        def run_and_save_particle_filter(no_particles_sampling_freq_tuple):
            no_particles, sampling_freq = no_particles_sampling_freq_tuple
            _ = self.generate_wrt_reference_episode(episode, no_particles, sampling_freq, end_index, req_output_posterior = req_output_posterior)
            print(f'Completed filter with no_particles = {no_particles} and sampling_freq = {sampling_freq}.', flush = True)
        no_particles_sampling_freq_tuples_list = list(product(no_particles_list, sampling_freq_list))
        with multiprocessing.Pool() as pool:
            pool.map(run_and_save_particle_filter, no_particles_sampling_freq_tuples_list)

    def compute_posterior(self, particle_filter_IO):
        generated_actions = [list(generated_episode['actions']) for generated_episode in particle_filter_IO['output']['generated_episodes']]
        element_counts = Counter([tuple(action_seq) for action_seq in generated_actions])
        posterior_ranked, attention_seq_ranked = [], []
        for action_seq, count in element_counts.items():
            posterior_ranked.append(count)
            attention_seq_ranked.append(np.array(action_seq)%self.env.no_attention_modes)
        posterior_ranked = np.array(posterior_ranked)/sum(posterior_ranked)
        sorted_ind = np.argsort(-1 * posterior_ranked)
        posterior_ranked = posterior_ranked[sorted_ind]
        attention_seq_ranked = [list(attention_seq_ranked[ind]) for ind in sorted_ind]
        particle_filter_IO['output']['posterior'] = {}
        particle_filter_IO['output']['posterior']['posterior_ranked'] = posterior_ranked
        particle_filter_IO['output']['posterior']['attention_seq_ranked'] = attention_seq_ranked
        return particle_filter_IO

    def plot_generated_episode(self, file_name, likelihood_rank, nodes_from_zero = 40, time_steps_before_lick = 20):
        particle_filter_IO = open_pickle_file(file_name)
        generated_ep = particle_filter_IO['output']['generated_episodes'][likelihood_rank]
        generated_ep['rewards'] = np.zeros(len(generated_ep['actions']))
        fig = plot_AF_episode(generated_ep, self.env, self.agent, nodes_from_zero = nodes_from_zero, time_steps_before_lick = time_steps_before_lick)

class PF_results_analyzer():

    def __init__(self, file_name, likelihood_rank = 0):
        self.particle_filter_IO = open_pickle_file(file_name)
        self.likelihood_rank = likelihood_rank
   
    def return_pf_results(self):
        return self.particle_filter_IO
    
    def plot_actions(self, start= 0, stop = None):
        generated_ep = self.particle_filter_IO['output']['generated_episodes'][self.likelihood_rank]
        reference_ep = self.particle_filter_IO['input']['root_episode']
        stop = len(generated_ep['actions']) if stop is None else stop
        def tranform_actions(episode):
            return episode['actions'].T[start:stop], ''
        trans_reference, trans_string = tranform_actions(reference_ep)
        trans_generated, trans_string = tranform_actions(generated_ep)
        max_action = max(max(trans_reference), max(trans_generated))
        fig, ax = plt.subplots(2)
        ax[0].stem(trans_reference)
        ax[0].set_title('Reference actions ' + f'({trans_string})')
        ax[0].set_xlabel('time')
        ax[0].set_ylabel('action')
        ax[0].set_yticks(np.arange(max_action + 1))
        ax[1].stem(trans_generated)
        ax[1].set_title('Generated actions ' + f'({trans_string})')
        ax[1].set_xlabel('time')
        ax[1].set_ylabel('action')
        ax[1].set_yticks(np.arange(max_action + 1))
        fig.tight_layout()
        plt.show()
        
    def plot_beliefs(self, start= 0, stop = None, min_color_val = -50, max_color_val = 0):
        generated_ep = self.particle_filter_IO['output']['generated_episodes'][self.likelihood_rank]
        reference_ep = self.particle_filter_IO['input']['root_episode']
        stop = len(generated_ep['actions']) if stop is None else stop
        def tranform_beleifs(episode):
            return np.log(episode['q_probs'].T[:,start:stop]), 'log'
        trans_reference, trans_string = tranform_beleifs(reference_ep)
        trans_generated, trans_string = tranform_beleifs(generated_ep)
        fig, ax = plt.subplots(2)
        num_states, num_steps  = np.shape(trans_reference)
        im1 = ax[0].imshow(trans_reference, vmin=min_color_val, vmax=max_color_val, extent=[-0.5, num_steps+0.5, num_states+0.5, -0.5], aspect='auto', cmap='viridis') 
        ax[0].set_title('Reference beliefs ' + f'(after {trans_string})')
        ax[0].set_xlabel('time')
        ax[0].set_ylabel('state')
        im2 = ax[1].imshow(trans_generated, vmin=min_color_val, vmax=max_color_val, extent=[-0.5, num_steps+0.5, num_states+0.5, -0.5], aspect='auto', cmap='viridis')
        ax[1].set_title('Generated beliefs ' + f'(after {trans_string})')
        ax[1].set_xlabel('time')
        ax[1].set_ylabel('state')
        fig.tight_layout()
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
        fig.colorbar(im2, cax=cbar_ax)
        plt.show()
    
    def plot_absolute_particle_likelihood(self):
        particles_likelihoods = self.particle_filter_IO['output']['particles_likelihoods']
        if np.sum(particles_likelihoods) == 0:
            print('\n Absolute particle likelihoods for all particles are close to 0.')
        else:
            plt.stem(self.particle_filter_IO['output']['particles_likelihoods'])
            plt.show()
    
    def plot_posterior(self):
        plt.stem(self.particle_filter_IO['output']['posterior']['posterior_ranked'])
        plt.xlabel('Attention sequence')
        plt.ylabel('PF posterior')
        plt.show()
    
    def print_repetition_dicitonary(self):
        repetition_dicitonary = self.particle_filter_IO['output']['sampling_count_tracker']
        if len(repetition_dicitonary) == 0:
            print(f"No sampling repetition was needed. Good stuff!")
        else:
            print(f"Sampling repeatition dictionary is {repetition_dicitonary}.")
    
    def check_for_different_trajectories(self):
        no_particles = len(self.particle_filter_IO['output']['particles_likelihoods'])
        matching_indices = []
        did_it_converge = False
        ind = 0
        diff_in_actions = 0
        while diff_in_actions == 0:
            ind += 1
            if ind > no_particles - 1:
                did_it_converge = True
                print('All particles converged to the same trajectory.')
                break
            a = self.particle_filter_IO['output']['generated_episodes'][0]['actions']
            b = self.particle_filter_IO['output']['generated_episodes'][ind]['actions']
            diff_in_actions = np.sum(np.abs(a-b))/len(a)
        if not did_it_converge:
            matching_indices.append(0)
            matching_indices.append(ind)
            print(f'\n Found at least two particles with different trajectories, namely particle {ind} and particle 0.')
            print(f'Rate of absolute difference in actions between them is {diff_in_actions} per time step.')
        return matching_indices

    def compute_attention_difference(self, no_attentions):
        generated_actions = self.particle_filter_IO['output']['generated_episodes'][self.likelihood_rank]['actions']
        actual_actions = self.particle_filter_IO['input']['root_episode']['actions'][:len(generated_actions)]
        generated_attention = np.array([action % no_attentions for action in generated_actions])
        actual_attention = np.array([action % no_attentions for action in actual_actions])
        error_rate = np.sum(np.abs(generated_attention-actual_attention))/len(generated_attention)
        print(f'\n Avgerage attention error rate (abs) between generated episode likelihood_rank-{self.likelihood_rank} and true episode is {error_rate} per time step.')
    
    def analyze_results(self):
        self.plot_actions()
        self.plot_beliefs()
        self.plot_posterior()
        # self.check_for_different_trajectories()
        self.plot_absolute_particle_likelihood()
        self.print_repetition_dicitonary()
        # self.compute_attention_difference(3)


