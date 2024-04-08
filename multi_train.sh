#!/bin/zsh

food_reward_list=(555 777 999 1111 1333 1500)
att_coeff=0.087213
att_temp=0.25
penalty_cost=-30
time_in_game_reward=0
no_episodes=1000
start_num_epochs=10
step_num_epochs=10
num_epochs=55
num_seeds=9

for ((inter_num_epochs = $start_num_epochs; inter_num_epochs <= ${num_epochs}; inter_num_epochs+= $step_num_epochs)); do 
    for ((seed_value = 0; seed_value < ${num_seeds}; seed_value++)); do 
        for ((exp_ind = 1; exp_ind < ${#food_reward_list[@]}; exp_ind++)); do 
            python train.py ${food_reward_list[$exp_ind]} $att_coeff $att_temp $penalty_cost $time_in_game_reward $inter_num_epochs $seed_value &
            # echo  ${food_reward_list[$exp_ind]} $att_coeff $att_temp $penalty_cost $time_in_game_reward $inter_num_epochs $seed_value
        done
    done
    wait
    food_reward_list_str="${food_reward_list[@]}"
    python compare_across_seeds.py "$food_reward_list_str" $att_coeff $att_temp $penalty_cost $time_in_game_reward $inter_num_epochs $seed_value $no_episodes &
    wait
done
wait
echo "Multi-train, and comparison code finished running!"
