# Correct usage is python train.py <food_reward> <att_coeff> <att_temp> <penalty_cost> <num_epochs> <seed_value>
# Correct usage is python compare_across_seeds.py <food_reward_list> <att_coeff_list> <att_temp_list> <penalty_cost_list> <num_epochs> <seed_list> <no_episodes>

# food_reward_list=(0 0.2 0.4 0.6 0.8 1)
# food_reward_list=(0.03125 0.0625 0.09375 0.125 0.15625 0.1875 0.21875 0.25)
food_reward_list=(100 200 300 400 500 600)
# food_reward_list=(0.03125)
# att_coeff=0.00149215543
# att_coeff=0.007213
att_coeff=0.037213
att_temp=0.25
# penalty_cost=-0.15
penalty_cost=-3
time_in_game_reward=0

# no_episodes=1000
no_episodes=1000

start_num_epochs=10
step_num_epochs=10

# num_epochs=400
# num_epochs=10000
num_epochs=1000

# num_seeds=10
num_seeds=1

for ((inter_num_epochs = $start_num_epochs; inter_num_epochs <= ${num_epochs}; inter_num_epochs+= $step_num_epochs)); do 
    for ((seed_value = 0; seed_value < ${num_seeds}; seed_value++)); do 
        for ((exp_ind = 0; exp_ind < ${#food_reward_list[@]}; exp_ind++)); do 
            python train.py ${food_reward_list[$exp_ind]} $att_coeff $att_temp $penalty_cost $time_in_game_reward $inter_num_epochs $seed_value &
        done
    done
    wait
    food_reward_list_str="${food_reward_list[@]}"
    python compare_across_seeds.py "$food_reward_list_str" $att_coeff $att_temp $penalty_cost $time_in_game_reward $inter_num_epochs $seed_value $no_episodes &
    wait
done
wait
echo "Multi-train, and comparison code finished running!"