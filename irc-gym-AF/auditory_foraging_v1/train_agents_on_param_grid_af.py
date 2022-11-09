import sys
sys.path.append('/home/lokesh/Documents/Projects/irc-gym-AF/irc-gym-main/')


import os, argparse, json
from irc import BeliefAgentFamily
#from irc.examples import FoodBoxEnv # to change to your custom environment
from AF_env import AuditoryForaging # to change to your custom environment

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--store-dir', default='cache',
        help="Directory for storing data.",
    )
    parser.add_argument('--param-grid-path', default='cache/param_grid.json',
        help="The path of environment parameter grid json file.",
    )
    parser.add_argument('--max-seed', default=1, type=int,
        help="Maximum seed number for each environment parameter."
    )
    parser.add_argument('--num-epochs', default=20, type=int,
        help="Number of reinforcement learning epochs for training agents."
    )
    parser.add_argument('--num-works', default=0, type=int,
        help="Number of agents to be trained before exit. '0' means to train all agents."
    )
    parser.add_argument('--patience', default=12, type=float,
        help="Number of hours from last modification of a training to be considered as running."
    )
    args = parser.parse_args()

    bafam = BeliefAgentFamily(AuditoryForaging, store_dir=args.store_dir) # replace 'FoodBoxEnv' with your environment class
    if os.path.exists(args.param_grid_path):
        with open(args.param_grid_path, 'r') as f:
            param_grid = json.load(f)
    else:
        env_param = bafam.default_env_param()
        param_grid = [[val] for val in env_param]
        with open(args.param_grid_path, 'w') as f:
            json.dump(param_grid, f)
        print(f"Default parameter grid is saved in {args.param_grid_path}")
    bafam.train_agents_on_param_grid(
        param_grid, args.max_seed, args.num_epochs,
        num_works=args.num_works, patience=args.patience,
    )
