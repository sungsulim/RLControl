# RLControl
Implementation of Continuous Control RL Algorithms.

## Available Algorithms
* Q-learning methods
  * [Actor-Expert, Actor-Expert+](https://arxiv.org/abs/1810.09103)
  * Actor-Expert with PICNN
  * [Wire-Fitting](http://www.leemon.com/papers/1993bk3.pdf) 
  * [Normalized Advantage Functions(NAF)](https://arxiv.org/abs/1603.00748)
  * [(Partial) Input Convex Neural Networks(ICNN)](https://arxiv.org/abs/1609.07152) - adapted from github.com/locuslab/icnn
  * [QT-Opt (single, mixture gaussian)](https://arxiv.org/abs/1806.10293)
  
* Policy Gradient methods
  * [Deep Deterministic Policy Gradient(DDPG)](https://arxiv.org/abs/1509.02971)
  * Advantage Actor-Critic baseline with Replay Buffer : This is different from ACER

## Usage
Settings for available environments and agents are provided in `jsonfiles/` directory

**Example:**

ENV=Pendulum-v0 (must match jsonfiles/environment/*.json name)

AGENT=ddpg (must match jsonfiles/agent/*.json name)

INDEX=0 (useful for running sweeps over different settings and doing multiple runs)


**Run:** `python3 main.py --env_json jsonfiles/environment/$ENV.json --agent_json jsonfiles/agent/$AGENT.json --index $INDEX`


(`--render` and `--monitor` is optional, to visualize/monitor the agents' training, only available for openai gym or mujoco environments. `--write_plot` is also available to plot the learned action-values and policy on Bimodal1DEnv domain.)


* ENV.json is used to specify evaluation settings:
  * TotalMilSteps: Total training steps to be run (in million)
  * EpisodeSteps: Steps in an episode (Use -1 to use the default setting)
  * EvalIntervalMilSteps: Evaluation Interval steps during training (in million)
  * EvalEpisodes: Number of episodes to evaluate in a single evaluation
  
* AGENT.json is used to specify agent hyperparameter settings: 
  * norm: type of normalization to use
  * exploration_policy: "ou_noise", "none": Use "none" if the algorithm has its own exploration mechanism
  * actor/critic l1_dim, l2_dim: layer dimensions
  * learning rate
  * other algorithm specific settings
  
