# RLControl
Implementation of Continuous Control RL Algorithms

## Available Algorithms
* Value-based methods
  * [Normalized Advantage Functions(NAF)](https://arxiv.org/abs/1603.00748)
  * [Input Convex Neural Networks(ICNN)](https://arxiv.org/abs/1609.07152) (Work in progress)
  * [Wire-Fitting](http://www.leemon.com/papers/1993bk3.pdf) (Work in progress)
  
* Policy Gradient methods
  * [Deep Deterministic Policy Gradient(DDPG)](https://arxiv.org/abs/1509.02971)

## Usage
Settings for available environments and agents are provided in `jsonfiles/` directory

**Example:**

ENV=Pendulum-v0

AGENT=ddpg

INDEX=0 (useful for running sweeps over different settings and doing multiple runs)


Run: `python3 main.py --env_json jsonfiles/environment/$ENV.json --agent_json jsonfiles/agent/$AGENT.json --index $INDEX --render --monitor`


(`--render` and `--monitor` is optional, to visualize/monitor the agents' training)

* ENV.json is used to specify evaluation settings:
  * TotalMilSteps: Total training steps to be run (in million)
  * EpisodeSteps: Steps in an episode (Use -1 to use the default setting)
  * EvalIntervalMilSteps: Evaluation Interval steps during training (in million)
  * EvalEpisodes: Number of episodes to evaluate in a single evaluation
  
* AGENT.json is used to specify agent hyperparameter settings: 
  * norm: type of normalization to use: "none", "input_norm", "layer", "batch"
  * exploration_policy: "ou_noise", "none": Use "none" if the algorithm has its own exploration mechanism
  * actor/critic l1_dim, l2_dim: layer dimensions
  * learning rate
  
