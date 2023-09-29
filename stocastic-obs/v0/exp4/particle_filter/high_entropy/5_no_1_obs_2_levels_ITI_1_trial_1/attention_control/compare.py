
no_episodes = 1
lick_cost_list = [0]
food_reward_list = [10.0, 50.0, 100.0, 500.0, 1000.0]
attention_cost_coeff_list = [.08]
attention_cost_temp_list = [.25]
penalty_cost_list = [0]
iti_cost_list = [0]

for lick_cost in lick_cost_list:
    for food_reward in food_reward_list:
        for attention_cost_coeff in attention_cost_coeff_list:
            for attention_cost_temp in attention_cost_temp_list:
                for penalty_cost in penalty_cost_list:
                    for iti_cost in iti_cost_list:
                        env_param = [lick_cost, food_reward, attention_cost_coeff, attention_cost_temp, penalty_cost, iti_cost]
                        hit_count = 0
                        miss_count = 0
                        false_alarm_count = 0
                        for episide_no in range(no_episodes):
                            episode = agent.run_one_episode(env=env, num_steps=10000, q_states = [[i] for i in range(env.no_nodes)])
                            if episode['rewards'][-1] == food_reward:
                                hit_count += 1
                            elif episode['states'][-1] == env.no_signal_nodes + 1:
                                false_alarm_count += 1
                            elif episode['states'][-1] > env.no_signal_nodes + 1:
                                miss_count += 1
                        