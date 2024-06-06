# AF project

# todo

0605. vary node np case. the problem is what we dicussed earlier. the current autocorr plot is not realy for this. should use the min wait time plot for eval.

# state about this pr
this is the version we have at the date of nips deadline.
the files are pretty messy, wie dont have time to do a full reorganization.
but they can be split into 2 groups.
plot notebooks. they are done on macbook laptop. 
i use them for plottings. 
the load agent load data should be on my laptop. 
(i dont have a function to check exist to avoid overwrite when save fig save data yet, i should have one soon.)

the vary notebooks.
they are used for training, and usually do not need extra file to run.
rare case, they are continue training. can just change the epoch index to 0 and start fresh. 
because of the set seed, we should have same result.
the trained agents are on the linux and mac server, i still have all the agents checkpoints.

reorganize plans.
curently, the notebooks for training and eval mainly have 2 versions. 
i did not finish the reorg. 
old version, i have the dict as data stucture because we only vary reward.
new version, i have the df as data structure for easy querying when vary for multi variables.
we will fully make this to df.

after that, we will thinking about improving smoothness of varying p1 and n nodes.
need to intergrate them into the aganet family.
but things are harder than reward.
eg, varying p1 changes belief directly. 
from my past exp, the agent do not think the extra knowledge of p1 help very much, and nearly ignore the extra dimention.
maybe we need some belief representation, since the current belief is not information dense.

we also want either a template or pipepline(prefered) to run experiemtns.
running exp, should have proper log, checkpoint saved, figure exported and visualized on some webpage. now i have to check it manually or show in vscode, make the notebook huge and slow.
better to keep to modular.



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

# example privateconfig (put it under the repo folder)

[Datafolder]
data = 'path to data folder'

[Codefolder]
workspace = 'path to repo folder'

[Notification]
token= 'bark token' 


# file directory

- audiotoryforage. the AF_env.py defines the task. 

- enviorment folder. pip and conda env file. no need to install all packages. pay attension to zsh (1 index) and bash (0 index) for shell script, and gym<=0.21 for full function (define and log random seed for reproducebility. i temperarlly comment out the random seed function in irc_gym/ircmodel.py line 65)

- irc_gym folder. a wraper of stablebaseline. no need to use it in my opionion. you can just run training individually with more flexibility. example in yc.ipynb.

- store. store the data occur during this.

- plot xxx notebooks. to generate the plots

- vary xxx notebooks, to train the agents.

# hints
 - curiculum training. learning a complex task is hard. learning without punishment coudl be a first step. we can add punishment back later.
 - we already have the simlar trend compared to expected result. we just lack the smoothness. modeling the reward as a input paramter (reward as one dim of the observation space) could link differetn reward tasks together and generate smooth behavior.

