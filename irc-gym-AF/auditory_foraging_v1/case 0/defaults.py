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

ALGO_KWARGS = { # RL algorithm kwargs
    'n_steps': 64, 'batch_size': 16, 'gamma': 0.99,
}

POLICY_KWARGS = {} # policy kwargs

LEARN_KWARGS = { # learning kwargs
    'total_timesteps': 256,
}

WARNING_OPTIMALITY = 0.8

EVAL_KWARGS = { # kwargs for evaluating an agent during RL
    'num_episodes': 10,
    'num_steps': 40,
}
