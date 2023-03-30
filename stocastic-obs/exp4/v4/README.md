Stochastic observation model

6 attention levels

changing probability of transition to (1/mean) * tau = (1/5) * .02 = 0.004
increasing # of signal nodes to 150.
increasing # iti nodes to 150.
setting 0 reward for iti and penalty.
reducing # of epochs to 20.
changing to episodic tasks.
    - chaning observe_step (should not have info on time).
    - change find_observation_matrix.
    - set done when it reaches the penalty node. 
    - changing transition to iti nodes.
    - change transition matrix as well, and check if belief updates are right.
    - check if the experement starts in the beginning of ITI.
    - Assign penalty cost as the termination cost.
    - Assigned penalty cost to reward when licking at state 0, instead of penalizing after moving to penalty state. 
    - Adjusted the attention costs in a way that least attention gives only attention cost of 0.
    - Set high cost for penalty state to strongly penalise it (behavior could be quite sensitive, if too much then might never lick).
set defaults-learn to 10000 in manager.yaml in irc
set runtime-num_steps to 100000 in manager.yaml in irc
changed utils for visualization.

remove prob_01 from run_episode line in notebook (not specific to this exp). 

To do:
increasing # of total_timesteps.
change num_epochs for param_grid part.

TAKEAWAY:
Previously: In v0 we observe most are purple while there is not much change in variety of attention. On that note, we make the following change.
Now: Increase temperature to flatter cost more, and increase scaling to discourage purple always.
