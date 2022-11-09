EST_SPEC = { # default specification for distribution estimation used by belief model
    'state_prior': {
        'num_samples': 2000,
        'optim_kwargs': {
            'batch_size': 64, 'num_epochs': 40,
            'lr': 0.01, 'momentum': 0.9, 'weight_decay': 1e-4,
        },
    },
    'obs_conditional': {
        'num_samples': 2000,
        'optim_kwargs': {
            'batch_size': 64, 'num_epochs': 40,
            'lr': 0.01, 'momentum': 0.9, 'weight_decay': 1e-4,
        },
    },
    'belief': {
        'num_samples': 200,
        'optim_kwargs': {
            'batch_size': 16, 'num_epochs': 5,
            'lr': 0.01, 'momentum': 0.9, 'weight_decay': 1e-4,
        },
    },
}

#ALGO_KWARGS = { # RL algorithm kwargs
#    'n_steps': 16,'batch_size': 64,'n_envs': 5, 'gamma': 0.99, #original batch_size was 16
#}
ALGO_KWARGS = { # RL algorithm kwargs
    'n_steps': 64,'batch_size': 256, 'gamma': 0.99, #original batch_size was 16
}  

POLICY_KWARGS = {} # policy kwargs

LEARN_KWARGS = { # learning kwargs
    'total_timesteps': 1024,
}

WARNING_OPTIMALITY = 0.8

EVAL_KWARGS = { # kwargs for evaluating an agent during RL
    'num_episodes': 10,
    'num_steps': 40,
}

#n_steps: The number of steps to run for each environment per update
#(i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel)
#(i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
# `batch_size = (n_steps * n_envs) // nminibatches``
# Added warning for ``PPO`` when ``n_steps * n_envs`` is not a multiple of ``batch_size``
# ``total_timesteps``: Total number of timesteps (steps in the environments)
#64 * n = 32
#https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
#16, 64, 256
#M ≤ N T
#M is the mini bactch size
#K epochs
# T timesteps we run policy for
# N actors are used
