import numpy as np
import matplotlib.pyplot as plt
from auditoryforage.utils import open_pickle_file, find_recent_epoch
# no_seeds = 5
# folder_name = './store/comparison/'
# pattern = f'#seeds_{no_seeds}_#epochs_'
start_model_no = 0
# epoch_increments_saved = 10
# max_epoch_val = find_recent_epoch(directory = folder_name, pattern = pattern)
# num_models = int(max_epoch_val/epoch_increments_saved)

def setup_measure_matrix(start_model_no, num_models, measure, no_seeds):
    # In general
        # dim 0 different epochs.
        # dim 1 is seed
    
        # dim 2 different exp no.
    # For best_seed_list
        # dim 0 different epochs.
        # dim 1 different exp no.

    epochs_list =  [(elt + 1) * epoch_increments_saved for elt in range(start_model_no, num_models)]
    measure_matrix = []
    for epoch in epochs_list:
        file_name = folder_name + f'#seeds_{no_seeds}_#epochs_{epoch}.pkl'
        detection_measures = open_pickle_file(file_name)
        if measure == 'best_seed_list':
            measure_matrix.append([curr_elt for curr_elt in detection_measures['output'][measure]])
        elif measure in ['attention_time_points_list', 'episode_length_list','hit_reaction_time_list','fa_reaction_time_list']:
            measure_matrix.append([])
            for seed_index in range(no_seeds):
                measure_matrix[-1].append([curr_elt[seed_index] for curr_elt in detection_measures['output'][measure]])
        elif measure in ['hit_count_list', 'miss_count_list', 'false_alarm_list', 'total_reward_list_across_exps']:
            measure_matrix.append([])
            for seed_index in range(no_seeds):
                measure_matrix[-1].append([curr_elt[seed_index]/detection_measures['input']['no_episodes'] for curr_elt in detection_measures['output'][measure]])
        
    
    if measure not in ['attention_time_points_list', 'episode_length_list','hit_reaction_time_list','fa_reaction_time_list']: 
        measure_matrix = np.array(measure_matrix)
    food_reward_list = detection_measures['input']['food_reward_list']
    return epochs_list, food_reward_list, measure_matrix

def top_k_combinations(total_reward_matrix, k):
    # takes total reward matrix as input and gives top k (epoch, seed) combinations for each experiemnt. 
    # input: across_seeds_dict['total_reward_matrix'], where dim 0 is epoch_ind, dim 1 is seed_ind and dim 2 is exp_ind.
    # output: list with exp_len no. of elements. Where each element is a list with k elements. Where each element is a tuple (epoch_ind, seed_ind).
    
    _, _, exp_len = total_reward_matrix.shape
    top_k_epoch_seed_indices = []

    # Iterate over each entry in the third dimension
    for i in range(exp_len):
        # Find indices of top k values for the current entry
        flat_indices = np.argpartition(total_reward_matrix[:, :, i], -k, axis=None)[-k:]
        flat_values = total_reward_matrix[:, :, i].flatten()[flat_indices]
        sorted_indices = np.argsort(flat_values)[::-1]  # Sort indices in descending order

        # Convert flattened indices to (epoch_ind, seed_ind) indices
        epoch_indices, seed_indices = np.unravel_index(flat_indices[sorted_indices], total_reward_matrix[:, :, i].shape)

        # Append the top k (e, s) combinations to the list
        top_k_epoch_seed_indices.append([(e_idx, s_idx) for e_idx, s_idx in zip(epoch_indices, seed_indices)])
    return top_k_epoch_seed_indices


def extract_info_across_seeds(no_seeds, choose_top_k_no = 1):
    # For best_seed_list
    # dim 0 different exp no.
    # dim 1 (epoch_ind, seed_ind).

    across_seeds_dict = {}
    across_seeds_dict['epochs_list'], across_seeds_dict['food_reward_list'], across_seeds_dict['total_reward_matrix'] = setup_measure_matrix(start_model_no, num_models, 'total_reward_list_across_exps', no_seeds)
    _, _, across_seeds_dict['hit_count_matrix'] = setup_measure_matrix(start_model_no, num_models, 'hit_count_list', no_seeds)
    _, _, across_seeds_dict['miss_count_matrix'] = setup_measure_matrix(start_model_no, num_models, 'miss_count_list', no_seeds)
    _, _, across_seeds_dict['false_alarm_matrix'] = setup_measure_matrix(start_model_no, num_models, 'false_alarm_list', no_seeds)
    _, _, across_seeds_dict['best_seed_matrix'] = setup_measure_matrix(start_model_no, num_models, 'best_seed_list', no_seeds)
    _, _, across_seeds_dict['attention_time_points_matrix'] = setup_measure_matrix(start_model_no, num_models, 'attention_time_points_list', no_seeds)
    _, _, across_seeds_dict['episode_length_matrix'] = setup_measure_matrix(start_model_no, num_models, 'episode_length_list', no_seeds)
    _, _, across_seeds_dict['hit_reaction_time_matrix'] = setup_measure_matrix(start_model_no, num_models, 'hit_reaction_time_list', no_seeds)
    _, _, across_seeds_dict['fa_reaction_time_matrix'] = setup_measure_matrix(start_model_no, num_models, 'fa_reaction_time_list', no_seeds)
    max_total_rewards = np.max(across_seeds_dict['total_reward_matrix'], axis=(0, 1))
    best_epoch_seed_inds = [np.where(across_seeds_dict['total_reward_matrix'][:,:,exp_ind] == max_total_rewards[exp_ind]) for exp_ind in range(len(max_total_rewards))]

    print(f"shape of total reward matrix is {np.shape(across_seeds_dict['total_reward_matrix'])}")
    print(f"shape of best_epoch_seed_inds is {np.shape(best_epoch_seed_inds)}")
    print(f"best_epoch_seed_inds is {best_epoch_seed_inds}")


    # # Manual check
    # epoch = int(max_epoch_val/epoch_increments_saved-1)
    # seed = 0
    # best_epoch_seed_inds = [(np.array([epoch]), np.array([seed])), (np.array([epoch]), np.array([seed])), (np.array([epoch]), np.array([seed])), (np.array([epoch]), np.array([seed])), (np.array([epoch]), np.array([seed])), (np.array([epoch]), np.array([seed])), (np.array([epoch]), np.array([seed])), (np.array([epoch]), np.array([seed])), (np.array([epoch]), np.array([seed])), (np.array([epoch]), np.array([seed])), (np.array([epoch]), np.array([seed]))]
    
    across_seeds_dict['best_epoch_seed_inds'] = [(elt[0][0], elt[1][0]) for elt in best_epoch_seed_inds]
    across_seeds_dict['top_k_epoch_seed_inds'] = top_k_combinations(across_seeds_dict['total_reward_matrix'], choose_top_k_no)

    print(f"across_seeds_dict is {across_seeds_dict['best_epoch_seed_inds']}")


    return across_seeds_dict    
    
def visualize_top_k_output(no_seeds):
    across_seeds_dict = extract_info_across_seeds(no_seeds)
    total_reward_matrix = across_seeds_dict['total_reward_matrix']
    hit_count_matrix = across_seeds_dict['hit_count_matrix']
    miss_count_matrix = across_seeds_dict['miss_count_matrix']
    false_alarm_matrix = across_seeds_dict['false_alarm_matrix']
    best_epoch_seed_inds = across_seeds_dict['best_epoch_seed_inds']
    top_k_epoch_seed_inds = across_seeds_dict['top_k_epoch_seed_inds']
    food_reward_list = across_seeds_dict['food_reward_list']
    epochs_list = across_seeds_dict['epochs_list']
    attention_time_points_matrix = across_seeds_dict['attention_time_points_matrix']
    episode_length_matrix = across_seeds_dict['episode_length_matrix']
    hit_reaction_time_matrix = across_seeds_dict['hit_reaction_time_matrix']
    fa_reaction_time_matrix = across_seeds_dict['fa_reaction_time_matrix']

    print(f'top_k_epoch_seed_inds is {top_k_epoch_seed_inds}')

    def plot_hit_miss_FA(x_axis_list, hit_count_list, miss_count_list, false_alarm_list, title = ''):
        plt.figure()
        plt.plot(x_axis_list, hit_count_list, '-*g')
        plt.plot(x_axis_list, miss_count_list, '-*b')
        plt.plot(x_axis_list, false_alarm_list, '-*r')
        plt.legend(['hit', 'miss', 'false alarm'])
        plt.xlabel('food reward') if x_axis_list == food_reward_list else plt.xlabel('epochs')
        plt.title(title)
        plt.show()
            
    def plot_hit_miss_FA_for_top_k():
        hit_count_list = np.zeros(len(food_reward_list))
        miss_count_list = np.zeros(len(food_reward_list))
        false_alarm_list = np.zeros(len(food_reward_list))
        choose_top_k_no = len(top_k_epoch_seed_inds[0])

        for kth_choice in range(choose_top_k_no):
            hit_count_list += [hit_count_matrix[top_k_epoch_seed_inds[exp_ind][kth_choice][0], top_k_epoch_seed_inds[exp_ind][kth_choice][1], exp_ind] for exp_ind in range(len(food_reward_list))]
            miss_count_list += [miss_count_matrix[top_k_epoch_seed_inds[exp_ind][kth_choice][0], top_k_epoch_seed_inds[exp_ind][kth_choice][1], exp_ind] for exp_ind in range(len(food_reward_list))]
            false_alarm_list += [false_alarm_matrix[top_k_epoch_seed_inds[exp_ind][kth_choice][0], top_k_epoch_seed_inds[exp_ind][kth_choice][1], exp_ind] for exp_ind in range(len(food_reward_list))]
        hit_count_list /= choose_top_k_no
        miss_count_list /= choose_top_k_no
        false_alarm_list /= choose_top_k_no
        
        title = 'Best across epochs and seeds'
        plot_hit_miss_FA(food_reward_list, hit_count_list, miss_count_list, false_alarm_list, title)
        print(f'food_reward_list is {food_reward_list}')         
        print(f'hit_count_list is {hit_count_list}')         
        print(f'miss_count_list is {miss_count_list}')         
        print(f'false_alarm_list is {false_alarm_list}')  

    def plot_no_high_att_for_best():
        attention_time_points_list = [attention_time_points_matrix[best_epoch_seed_inds[exp_ind][0]][best_epoch_seed_inds[exp_ind][1]][exp_ind] for exp_ind in range(len(food_reward_list))]
        no_attention_list = []
        for ind in range(len(attention_time_points_list)):
            no_attention_list.append([len(episode) for episode in attention_time_points_list[ind]])
                
        plt.figure()
        plt.boxplot(no_attention_list, showfliers=False)
        plt.xticks([ind+1 for ind in range(len(food_reward_list))], [str(food_reward) for food_reward in food_reward_list])
        plt.xlabel('food reward')
        plt.ylabel(f'Number of high attention time instances')
        plt.title(f'Boxlplot - Number of high attention time instances Vs food rewawrds')
        plt.show()

        plt.figure()
        mean_list = [np.mean(no_attention_list[exp_ind]) for exp_ind in range(len(food_reward_list))]
        sem_list = [np.std(no_attention_list[exp_ind])/np.sqrt(np.size(no_attention_list[exp_ind])) for exp_ind in range(len(food_reward_list))]
        plt.errorbar([str(food_reward) for food_reward in food_reward_list], mean_list, yerr= sem_list, fmt='o', markersize=6, capsize=4)
        plt.xlabel('food reward')
        plt.ylabel(f'Number of high attention time instances')
        plt.title(f'Error bar - Number of high attention time instances for best')
        plt.xticks([ind+1 for ind in range(len(food_reward_list))], [str(food_reward) for food_reward in food_reward_list])
        plt.grid()
        plt.show()

    def plot_trial_length_for_best():
        episode_length_list = [episode_length_matrix[best_epoch_seed_inds[exp_ind][0]][best_epoch_seed_inds[exp_ind][1]][exp_ind] for exp_ind in range(len(food_reward_list))]
        plt.figure()
        plt.boxplot(episode_length_list, showfliers=False)
        plt.xticks([ind+1 for ind in range(len(food_reward_list))], [str(food_reward) for food_reward in food_reward_list])
        plt.ylabel(f'trial length')
        plt.title(f'Box plot - trial length for best')
        plt.xlabel('food reward')
        plt.show()

        plt.figure()
        mean_list = [np.mean(episode_length_matrix[best_epoch_seed_inds[exp_ind][0]][best_epoch_seed_inds[exp_ind][1]][exp_ind]) for exp_ind in range(len(food_reward_list))]
        sem_list = [np.std(episode_length_matrix[best_epoch_seed_inds[exp_ind][0]][best_epoch_seed_inds[exp_ind][1]][exp_ind])/np.sqrt(np.size(episode_length_matrix[best_epoch_seed_inds[exp_ind][0]][best_epoch_seed_inds[exp_ind][1]][exp_ind])) for exp_ind in range(len(food_reward_list))]
        plt.errorbar([str(food_reward) for food_reward in food_reward_list], mean_list, yerr= sem_list, fmt='o', markersize=6, capsize=4)
        plt.xlabel('food reward')
        plt.ylabel(f'trial length')
        plt.title(f'Error bar - trial length for best')
        plt.xticks([ind+1 for ind in range(len(food_reward_list))], [str(food_reward) for food_reward in food_reward_list])
        plt.grid()
        plt.show()

    def plot_hit_reaction_time_for_best():
        hit_reaction_time_list = [hit_reaction_time_matrix[best_epoch_seed_inds[exp_ind][0]][best_epoch_seed_inds[exp_ind][1]][exp_ind] for exp_ind in range(len(food_reward_list))]
        plt.figure()
        plt.boxplot(hit_reaction_time_list, showfliers=False)
        plt.xticks([ind+1 for ind in range(len(food_reward_list))], [str(food_reward) for food_reward in food_reward_list])
        plt.ylabel(f'hit reaction time')
        plt.title(f'Boxplot - hit reaction time for best')
        plt.xlabel('food reward')
        plt.show()

        plt.figure()
        mean_list = [np.mean(hit_reaction_time_matrix[best_epoch_seed_inds[exp_ind][0]][best_epoch_seed_inds[exp_ind][1]][exp_ind]) for exp_ind in range(len(food_reward_list))]
        sem_list = [np.std(hit_reaction_time_matrix[best_epoch_seed_inds[exp_ind][0]][best_epoch_seed_inds[exp_ind][1]][exp_ind])/np.sqrt(np.size(hit_reaction_time_matrix[best_epoch_seed_inds[exp_ind][0]][best_epoch_seed_inds[exp_ind][1]][exp_ind])) for exp_ind in range(len(food_reward_list))]
        plt.errorbar([str(food_reward) for food_reward in food_reward_list], mean_list, yerr= sem_list, fmt='o', markersize=6, capsize=4)
        plt.xlabel('food reward')
        plt.ylabel(f'hit reaction time')
        plt.title(f'Error bar - hit reaction time for best')
        plt.xticks([ind+1 for ind in range(len(food_reward_list))], [str(food_reward) for food_reward in food_reward_list])
        plt.grid()
        plt.show()

        return hit_reaction_time_list

    def plot_fa_reaction_time_for_best():
        fa_reaction_time_list = [fa_reaction_time_matrix[best_epoch_seed_inds[exp_ind][0]][best_epoch_seed_inds[exp_ind][1]][exp_ind] for exp_ind in range(len(food_reward_list))]
        plt.figure()
        plt.boxplot(fa_reaction_time_list, showfliers=False)
        plt.xticks([ind+1 for ind in range(len(food_reward_list))], [str(food_reward) for food_reward in food_reward_list])
        plt.ylabel(f'fa reaction time')
        plt.title(f'Boxplot - fa reaction time for best')
        plt.xlabel('food reward')
        plt.show()

        plt.figure()
        mean_list = [np.mean(fa_reaction_time_matrix[best_epoch_seed_inds[exp_ind][0]][best_epoch_seed_inds[exp_ind][1]][exp_ind]) for exp_ind in range(len(food_reward_list))]
        sem_list = [np.std(fa_reaction_time_matrix[best_epoch_seed_inds[exp_ind][0]][best_epoch_seed_inds[exp_ind][1]][exp_ind])/np.sqrt(np.size(fa_reaction_time_matrix[best_epoch_seed_inds[exp_ind][0]][best_epoch_seed_inds[exp_ind][1]][exp_ind])) for exp_ind in range(len(food_reward_list))]
        plt.errorbar([str(food_reward) for food_reward in food_reward_list], mean_list, yerr= sem_list, fmt='o', markersize=6, capsize=4)
        plt.xlabel('food reward')
        plt.ylabel(f'fa reaction time')
        plt.title(f'Error bar - hit reaction time for best')
        plt.xticks([ind+1 for ind in range(len(food_reward_list))], [str(food_reward) for food_reward in food_reward_list])
        plt.grid()
        plt.show()
    
    def plot_total_reward_for_best():
        plt.figure()
        plt.plot(food_reward_list, [total_reward_matrix[best_epoch_seed_inds[exp_ind][0]][best_epoch_seed_inds[exp_ind][1]][exp_ind] for exp_ind in range(len(food_reward_list))],'*-')
        plt.xlabel('food reward')
        plt.ylabel('total rewards per trial')
        plt.title(f'Total rewards for best')
        plt.show()

    def plot_reward_progression_per_seed_per_exp(seed_index, exp_index):
        plt.figure()
        plt.plot(epochs_list, [total_reward_matrix[epoch_ind][seed_index][exp_index] for epoch_ind in range(len(epochs_list))],'*-')
        plt.xlabel('epoch number')
        plt.ylabel('rewards per trial')
        plt.title(f'Rewards per trial progression  across epochs for exp_ind {exp_index} and seed {seed_index}')
        plt.show()
        