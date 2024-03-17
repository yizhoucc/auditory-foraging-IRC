import numpy as np
class AuditoryForaging():

    def __init__(self, signal_start, signal_duration, total_duration, no_attention_modes = 2
    ):  
        self.signal_start = signal_start
        self.signal_duration = signal_duration
        self.total_duration = total_duration
        self.signal_states = [elt for elt in range(self.signal_start, self.signal_start+self.signal_duration)]
        self.no_attention_modes = no_attention_modes
        self.obs_certainity_possible = 0.1/(self.no_attention_modes-1) * np.arange(self.no_attention_modes) + 0.5 
        self.state = 0
        self.start_obs = 2


    def find_reward(self, lick_choice):
        """
        Computes the reward, given the choice of licking and the amount of attention.
        """
        if self.state in self.signal_states and lick_choice == 1:
            return 1
        else:
            return 0

    def transition_step(self, lick_choice):
        """
        Based on the current state and the lick choice, the state value is updated
        from the current state value to the future state value.
        """

        # If current state is node 0 (tone cloud without target)
        if self.state == 0:
            if lick_choice == 1: #penalty
                next_state = 1 + self.no_signal_nodes
            else: #no penalty
                next_state = self.state + np.random.choice(2, p=[1-self.prob_01, self.prob_01])

        # If current state is in the beginning of tone cloud with target
        if self.state>=1 and self.state<self.no_signal_nodes:
            if lick_choice == 0: #time passes by
                next_state = self.state + 1
            else: #goes to ITI
                next_state = self.no_signal_nodes + self.no_penalty_nodes + 1

        # If current state is in the end of tone cloud without target
        if self.state == self.no_signal_nodes:
            next_state = self.no_signal_nodes + self.no_penalty_nodes + 1

        # If current state is anywhere in between beginning of penalty period or just before the end of ITI
        if self.state>=1 + self.no_signal_nodes and self.state<self.no_nodes-1:
            next_state = self.state + 1

        # If current state is in the end of ITI
        if self.state == self.no_nodes-1:
            next_state = 0

        self.state = next_state

    def observe_step(self, attention_choice):
        """
        Provides the observation given the choice of attention provided at the previous time step, and the current state.
        """
        if self.state == 0:
            obs = self.start_obs
        elif self.state in self.signal_states:
            obs = np.random.binomial(size=1, n=1, p= self.obs_certainity_possible[attention_choice])[0]
        elif self.state not in self.signal_states:
            obs = 1 - np.random.binomial(size=1, n=1, p= self.obs_certainity_possible[attention_choice])[0]
        else:
            print("Problem: State has to be in range.")
        return obs

    def step(self, action):
        """
        One time step in the POMDP.
        """

        done = False
        info = {}

        lick_choice, attention_choice = action

        # Reward
        rw = self.find_reward(lick_choice, attention_choice)
        self.collected_reward += rw
        
        # State transition
        self.transition_step(lick_choice)

        # Observation
        obs = self.observe_step(attention_choice)

        if self.state > self.no_signal_nodes:
            done = True

        return obs, rw, done, info

    def reset(self):
        """
        Resetting to beginning of ITI period.
        """
        self.state = 0
        obs = self.observe_step(0)
        self.time = 0
        return obs


    def update_belief(self, previous_belief, action, observation):
        """
        Updating belief, given previous belief, new observation, and past action.
        """
        lick_choice, attention_choice = action

        transition_matrix = self.find_transition_matrix()
        observation_matrix = self.find_observation_matrix()
        new_belief = np.zeros(self.no_nodes)
        
        for state in range(self.no_nodes):
            # note the transpose below, because of the way we made transition_matrix: (current state, next state, action)
            new_belief[state] = observation_matrix[observation,state,int(attention_choice)] * np.reshape(np.transpose(transition_matrix[:,state,int(lick_choice)]),(1,self.no_nodes)) @ previous_belief
        
        if np.sum(new_belief) == 0:
            # print('Error: Mistake in belief update as all probabilities are coming out to be 0 somehow. Returned None!')
            new_belief = None
        else:
            new_belief = new_belief/np.sum(new_belief) #Normalization
        
        return new_belief


    def find_transition_matrix(self):
        """
        Function returns the transition matrix of the form transition_matrix(current_state,future_state,current_lick_choice).
        Note that although the usual convention is transition_matrix(next state, current state, action),
        we set it up as transition_matrix(current_state,future_state,current_lick_choice).
        Because of the above choice, some places we use np.transpose() while using this matrix.
        """

        transition_matrix = np.zeros((self.no_nodes,self.no_nodes,2))

        # no lick cases
        transition_matrix[(0,0,0)] = 1 - self.prob_01
        transition_matrix[(0,1,0)] = self.prob_01

        transition_matrix[(self.no_signal_nodes, self.no_signal_nodes + self.no_penalty_nodes + 1, 0)] = 1
        
        transition_matrix[(self.no_nodes-1,0,0)] = 1
        for i in range(1,self.no_nodes - 1):
            if i != self.no_signal_nodes:
                transition_matrix[(i,i+1,0)] = 1

        # lick cases
        transition_matrix[(0,self.no_signal_nodes+1,1)] = 1

        for i in range(1,self.no_signal_nodes+1):
            transition_matrix[(i, self.no_signal_nodes + self.no_penalty_nodes + 1, 1)] = 1
        
        for i in range(self.no_signal_nodes+1,self.no_nodes-1):
            transition_matrix[(i,i+1,1)] = 1
        transition_matrix[(self.no_nodes-1,0,1)] = 1

        return transition_matrix


    def find_observation_matrix(self):
        """
        Function returns the observation matrix of the form observation_matrix(obs at (t+1), state at (t+1), action at (t)).
        Representing the probability O(obs at (t+1)|state at (t+1),action at (t))
        """

        observation_matrix = np.zeros((len(self.observation_possible),self.no_nodes,len(self.attention_possible)))
        
        #Considering 0th state (partially observable)
        for attention in range(len(self.attention_possible)):
            observation_matrix[0,0,attention] = self.obs_certainity_possible[attention]
            observation_matrix[1,0,attention] = 1 - self.obs_certainity_possible[attention]
        # Considering 'food' states (partially observable)
        for i in range(1,self.no_signal_nodes+1):
            for attention in range(len(self.attention_possible)):
                observation_matrix[0,i,attention] = 1 - self.obs_certainity_possible[attention]
                observation_matrix[1,i,attention] = self.obs_certainity_possible[attention]
        
        # Considering 'non-trial' (fully observable) nodes
        for i in range(self.no_signal_nodes+1,self.no_signal_nodes+1+self.no_penalty_nodes):
            observation_matrix[2,i,:] = 1
        for i in range(self.no_signal_nodes+1+self.no_penalty_nodes,self.no_nodes):
            observation_matrix[3,i,:] = 1
        
        return observation_matrix

    def init_belief(self, obs):
        r"""Initializes belief with observation.
        """
        if obs == self.start_obs:
            belief = np.zeros(shape=self.total_duration)
            for ind in range(1,self.total_duration):
                belief[ind] = 1/(self.total_duration-1) 
        else:
            print(f'Problem: First observation should have been {self.start_obs}')
        
        return belief