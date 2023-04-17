import matplotlib.pyplot as plt
import numpy as np

# each element is of the form {exp_name: (attention_coefficient, attention_temperature)}
exps_list = {0: (25, 10), 2: (38, 15), 3: (50, 20), 4: (63, 25), 6: (13, 5), 7: (13, .75), 8: (13, 1), 9: (13, .65), 10: (13, .5)}

def plot_exp_settings(exp_params, no_attention_modes = 6):
    attention_cost_coeff, attention_cost_temp = exp_params
    obs_certainity_possible = 1/(2*(no_attention_modes-1)) * np.arange(no_attention_modes) + 0.5
    attention_cost = np.array([-attention_cost_coeff * np.exp(certainity/attention_cost_temp) for certainity in obs_certainity_possible])
    plt.plot(obs_certainity_possible, attention_cost - attention_cost[0], 'o-')
exp_name_list = []
for exp_name in exps_list.keys():
    exp_name_list.append(str(exp_name))
    plot_exp_settings(exps_list[exp_name])
plt.xlabel('certainity')
plt.ylabel('attention cost')
plt.legend(exp_name_list)
plt.show()