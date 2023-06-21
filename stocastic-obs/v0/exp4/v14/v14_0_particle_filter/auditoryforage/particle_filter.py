# A1: Assuming we start with ITI.
# C1: Might have to have this in tensor
# C2: check if [0] is required, depending on observer_step in new env code. Also note how observe_step comes after env.step.
# overdue, resampling one instant

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
import copy

def particle_filter(agent, env, lick_actions, state_list, no_particles = 10, sampling_freq = 1, verbose = False):
        
        _to_restore_train = agent.algo.policy.training 
        agent.algo.policy.set_training_mode(False)
        observation_matrix = env.find_observation_matrix()
        env.state = state_list[0]
        observation = env.observe_step(0) #A1
        belief = env.init_belief(observation)
        belief_list = [[belief] for _ in range(no_particles)]
        observation = observation[0]
        particle_observation_prob = 1 #A1
        observation_list = [[[observation]] for _ in range(no_particles)]        
        particles_distribution = 1/no_particles * np.ones(no_particles)
        particles_likelihoods = np.ones(no_particles)
        action_list = [[] for _ in range(no_particles)]
        overdue_status = [False for _ in range(no_particles)]

        sampling_count_tracker = {} #key are the time indices where additional sampling was done, and values represent number of times. 
        
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
                        instant_likelihood.append(particle_observation_prob * particle_action_prob)
                        _, attention_choice = env.dict_action_possible[int(action)]
                        
                        if time != len(lick_actions) - 1: 
                            observation = env.observe_step(attention_choice)[0] #C2
                            observation_list[particle].append([observation])
                            particle_observation_prob = observation_matrix[observation, env.state, attention_choice]
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
            
            if time%sampling_freq == 0:
                overdue_status = [False for _ in range(no_particles)]
                temp_observation_list = [[] for _ in range(no_particles)]
                temp_belief_list = [[] for _ in range(no_particles)]
                temp_action_list = [[] for _ in range(no_particles)]
                temp_particles_likelihoods = [[] for _ in range(no_particles)]
                for particle in range(no_particles):
                    chosen_particle = np.random.choice(no_particles, p = particles_distribution)
                    temp_observation_list[particle] = copy.deepcopy(observation_list[chosen_particle])
                    temp_belief_list[particle] = copy.deepcopy(belief_list[chosen_particle])
                    temp_action_list[particle] = copy.deepcopy(action_list[chosen_particle])
                    temp_particles_likelihoods[particle] = particles_likelihoods[chosen_particle]
                observation_list = copy.deepcopy(temp_observation_list)
                belief_list = copy.deepcopy(temp_belief_list)
                action_list = copy.deepcopy(temp_action_list)
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
        particle_filter_output['sampling count tracker'] = sampling_count_tracker
        
        return particle_filter_output