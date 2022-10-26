# irc-gym
Inverse rational control (IRC) package based on Gym

## Requirements
* `gym`
* `stable-baselines3`
* [`jarvis`](<https://github.com/lizhe07/jarvis/tree/0.4>)

## Train agents for a grid of environment parameters
Copy the script file `scripts/train_agents_on_param_grid.py` to your own project folder, and replace
the environment class in line 3 and 27 to your environment class.

An example to train just one agent is
```
python train_agents_on_param_grid.py --num-works 1 --patience 0
```
You can specify the environment parameters in a json file (`cache/param_grid.json` by default),
which contains a list of lists of float numbers, containing the possible environment parameter
values in each dimension. Multiple agents will be trained for one environment parameter combination
depending on the `max-seed` specified.
```
python train_agents_on_param_grid.py --param-grid-path [json_path] --max-seed 2
```
