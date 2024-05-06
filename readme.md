# AF project

# about this branch
for testing purpose mainly. has some plotting functions.
the structure is train then analysis.

# recent update (code)
5.6 udpate: 
sync the notebooks. now just need to copy the training config cell from train to anaylysis, to generate plots.

# recent update (plot and idea)
please refer to latex

# intructions
- run ./multi_train.sh.  
    - this file has 2 parts. 
        - 1, it trains 5 agent under different food reward.
        - 2, it evaludate (generate data for later plots) the trained agents

- use 'compare across seeds and epoch.ipynb' to generate plots. we want the reward to match with the plots in 'expected result' folder.


# file directory

- audiotoryforage. the AF_env.py defines the task. 

- enviorment folder. pip and conda env file. no need to install all packages. pay attension to zsh (1 index) and bash (0 index) for shell script, and gym<=0.21 for full function (define and log random seed for reproducebility. i temperarlly comment out the random seed function in irc_gym/ircmodel.py line 65)

- irc_gym folder. a wraper of stablebaseline. no need to use it in my opionion. you can just run training individually with more flexibility. example in yc.ipynb.

- store. store the data occur during this.

# hints
 - curiculum training. learning a complex task is hard. learning without punishment coudl be a first step. we can add punishment back later.
 - we already have the simlar trend compared to expected result. we just lack the smoothness. modeling the reward as a input paramter (reward as one dim of the observation space) could link differetn reward tasks together and generate smooth behavior.