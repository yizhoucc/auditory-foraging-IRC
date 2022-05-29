import gym
import numpy as np
from gym import Env
from gym.spaces import Discrete, MultiDiscrete

# POMDP model:
# State 0 (tone cloud without target) and states representing tone cloud with target are treated as partially observable states.
# Remaining states are considered fully observable. i.e. inter trial time interval states (pink noise duration), and penaltly states (penalty time duration).
# If the agent pays attention and the agent's next state is a partially observable state, the observation is 1/0 depending on whether food is present/absent.
# If the agent does not pay attention and the agent's next state is a partially observable state, the observation is 0.5.
# If the agent's next state is a fully observable state, the observation is the next state's node index in the graphical model.

# Definitions:
# prob_01 - Probability of transitioning from state 0 (tone cloud without target) to state 1 (tone cloud with target)
# no_signal_nodes - Maximum duration of tone cloud with target
# no_penalty_nodes - Penalty duration, in the case the agent licks in state 0 (during tone cloud without target)
# no_ITI_nodes - Inter trial time interval when pink noise is played.
# lick_cost - Cost of performing a lick.
# food_reward - Reward of licking food.
# attention_possible - A numpy array of all possible attention levels. in the increasing order of attention.
# attention_cost - A numpy array of corresponding cost of attention, in the increasing order of attention.
# no_nodes - Total number of nodes in the graphical model.
# observation_possible - A numpy array of possible values of observation.
# dict_observation_possible - Dictionary version of 'observation_possible'. (key, value) pair in the dicitonary is the (index,observation) pair in the array.
# dict_action_possible - Dictionary with values as action tuples, each tuple representing (lick_choice, attention_choice).

# Problem-setup parameters
prob_01 = .7
no_signal_nodes = 4
no_penalty_nodes = 7
no_ITI_nodes =  4
lick_cost = 1
food_reward = 10
attention_possible = np.array([0, 1]) # increasing order
attention_cost = np.array([0, 2]) # increasing order of attention
no_nodes =  1 + no_signal_nodes + no_penalty_nodes + no_ITI_nodes
observation_possible = np.concatenate((np.array([0,0.5,1]),np.arange(1 + no_signal_nodes,no_nodes)))

# Look-up dictionaries to map numbers used in OpenAI version (dict keys) to physical quantities in the foraging task (dict values).
dict_observation_possible = dict(enumerate(observation_possible))
dict_action_possible = dict(enumerate([(lick_choice,attention_choice) for lick_choice in range(2) for attention_choice in attention_possible]))

# Environment class for OpenAI gym
class AuditoryForaging(Env):
    def __init__(self):
        # Agent's observation space, which is different from experimentalist's observation space!
        self.obs_space = MultiDiscrete(len(dict_observation_possible))

        # Agent's action space, note that the experimentalist might not have direct access to attention choice!
        self.action_space = Discrete(len(dict_action_possible))

        # Initial state is the beginning of pink noise
        self.state = 1 + no_signal_nodes + no_penalty_nodes

        # no. of rounds
        self.rounds = 20

        # Initial rewards collected is 0
        self.collected_reward = 0

    def step(self, action):
        done = False
        info = {}
        rw = 0
        self.rounds -= 1

        lick_choice, attention_choice = dict_action_possible[action]
        current_state = self.state

        # Reward
        attention_cost_value = attention_cost[list(attention_possible).index(attention_choice)]
        lick_cost_value = lick_choice * lick_cost
        if self.state>=1 and self.state<=no_signal_nodes and lick_choice == 1:
            food_reward_value = food_reward
        else:
            food_reward_value = 0
        rw = food_reward_value - (attention_cost_value + lick_cost_value)
        self.collected_reward += rw

        # State transition
        # If current state is node 0 (tone cloud without target)
        if self.state == 0:
            if lick_choice == 1: #penalty
                next_state = 1 + no_signal_nodes #no penalty
            else:
                next_state = self.state + np.random.choice(2, p=[1-prob_01, prob_01])
        # If current state is in the beginning of tone cloud with target
        if self.state>=1 and self.state<no_signal_nodes:
            if lick_choice == 0: #time passes by
                next_state = self.state + 1
            else: #goes to ITI
                next_state = 1 + no_signal_nodes + no_penalty_nodes
        # If current state is in the end of tone cloud without target
        if self.state == no_signal_nodes:
            next_state = 1 + no_signal_nodes + no_penalty_nodes
        # If current state is anywhere in between beginning of penalty period or just before the end of ITI
        if self.state>=1 + no_signal_nodes and self.state<no_nodes-1:
            next_state = self.state + 1
        # If current state is in the end of ITI
        if self.state == no_nodes-1:
            next_state = 0
        self.state = next_state

        # Observation
        if self.state == 0:
            if attention_choice == 0:
                obs = list(observation_possible).index(0.5)
            else:
                obs = list(observation_possible).index(0)
        if self.state>=1 and self.state<=no_signal_nodes:
            if attention_choice == 0:
                obs = list(observation_possible).index(0.5)
            else:
                obs = list(observation_possible).index(1)
        if self.state>=1 + no_signal_nodes and self.state<no_nodes:
            obs = list(observation_possible).index(self.state)

        if self.rounds == 0:
            done = True

        self.render(current_state, action, rw, obs)

        return obs, self.collected_reward, done, info

    def reset(self):
        self.state = 1 + no_signal_nodes + no_penalty_nodes
        return self.state

    def render(self, current_state, action, rw, obs):
        print(f"Round : {self.rounds}\nCurrent State : {current_state}\nLick Choice : {dict_action_possible[action][0]}\nAttention Choice : {dict_action_possible[action][1]}\nReward Received: {rw}\nNext State: {self.state}\nNext Observation: {observation_possible[obs]}")
        print(f"Total Reward : {self.collected_reward}")
        print("=============================================================================")

env = AuditoryForaging()
done = False
state = env.reset()
while not done:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
