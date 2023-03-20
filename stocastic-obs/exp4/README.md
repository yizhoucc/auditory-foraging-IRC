Stochastic observation model

6 attention levels

changing probability of transition to (1/mean) * tau = (1/5) * .02 = 0.004
increasing # of signal nodes to 150.
increasing # iti nodes to 150.
setting 0 reward for iti and penalty.
reducing # of epochs to 20.
changing to episodic tasks.
    - chaning observation model (should not have info on time)
    - set done when it reaches the penalty node. 
    - changing transition to iti nodes.
    - change transition matrix as well, and check if belief updates are right.
    - check if the experement starts in the beginning of ITI.
    - Assign penalty cost as the termination cost.
set defaults-learn to 2000 in manager.yaml in irc
set runtime-num_steps to 10000 in manager.yaml in irc
changed utils for visualization.

increasing # of total_timesteps.
