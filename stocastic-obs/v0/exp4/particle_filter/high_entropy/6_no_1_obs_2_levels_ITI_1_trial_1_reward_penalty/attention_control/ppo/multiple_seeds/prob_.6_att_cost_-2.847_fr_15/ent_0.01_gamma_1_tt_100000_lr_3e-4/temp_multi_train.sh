# Correct usage is python train.py <food_reward> <att_coeff> <att_temp> <penalty_cost> <num_epochs> <seed_value>
# Correct usage is python compare_across_seeds.py <food_reward_list> <att_coeff_list> <att_temp_list> <penalty_cost_list> <num_epochs> <seed_list> <no_episodes>

food_reward=15
att_coeff=0.28412737113
att_temp=0.25
penalty_cost_list=(0 -1 -2 -3 -4 -5 -6 -7 -8 -9 -10)

# no_episodes=1000
no_episodes=500

start_num_epochs=400

step_num_epochs=50

# num_epochs=400
num_epochs=400

# num_seeds=5
num_seeds=1

for ((inter_num_epochs = $start_num_epochs; inter_num_epochs <= ${num_epochs}; inter_num_epochs+= $step_num_epochs)); do 
    for ((seed_value = 0; seed_value < ${num_seeds}; seed_value++)); do 
        for ((exp_ind = 0; exp_ind < ${#penalty_cost_list[@]}; exp_ind++)); do 
            python train.py $food_reward $att_coeff $att_temp ${penalty_cost_list[$exp_ind]} $inter_num_epochs $seed_value &
        done
    done
    wait
    penalty_cost_list_str="${penalty_cost_list[@]}"
    python temp_compare_across_seeds.py $food_reward $att_coeff $att_temp "$penalty_cost_list_str" $inter_num_epochs $seed_value $no_episodes &
    wait
done
wait
echo "Multi-train, and comparison code finished running!"