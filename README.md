[//]: # (Image References)

[image1]: ./reports/gif_tennis_800px.gif "Trained Agent"

# Project 3: Collaboration and Competition

## Introduction

This project is the third major assignment of the [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) course offered by UDACITY.

For this project, you will work with the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.

![Trained Agent][image1]

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

## Setup Instructions

### Repository Setup

Clone the course repository and set up the Python environment:

```bash
# Clone the required repositories
git clone https://github.com/udacity/deep-reinforcement-learning.git
git clone https://github.com/Yoshiokha/Deep_Reinforcement_Learning_Nanodegree_UDACITY/Project2_Continuous_Control.git

# Navigate to the project directory
cd Project2_Continuous_Control

# Create and activate the conda environment
conda create --name DRL_udacity_cpu python=3.6
conda activate DRL_udacity_cpu

# Install PyTorch (Windows specific link provided, for others, visit PyTorch's previous versions page)
pip install https://download.pytorch.org/whl/cpu/torch-0.4.0-cp36-cp36m-win_amd64.whl

# Update unityagents in requirements.txt and install dependencies
pip install -r requirements.txt

# Add the environment to Jupyter
python -m ipykernel install --user --name DRL_udacity_cpu --display-name "Python 3.6 (DRL_udacity_cpu)"
```

### Environment Setup

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

2. Extract the downloaded environment into the `Project3_Collab-compet/` directory., and unzip (or decompress) the file. 


### Instructions

Follow the instructions in `Tennis.ipynb` to get started with training your own agent!  

## Training the Agent

To start training the agent:

1. Launch Jupyter Notebook by running `jupyter notebook` in your terminal.
2. Open the notebook `Tennis.ipynb`.
3. Update the path to the Tennis environment in the notebook to match your local setup.
4. Execute the notebook cells to begin training the agent.

The training process concludes once the agents achieve an average score of +0.5 over 100 consecutive episodes. Models, tracking indicator data, and plots are archived in dedicated folders. Each training session is sequentially saved in the 'tests' directory for subsequent performance analysis.

## Abstract

In this [Report.pdf](./reports/Report.pdf), we tackled the challenge of the "Tennis" environment from the Udacity Deep Reinforcement Learning Nanodegree, employing a Multi-Agent Deep Deterministic Policy Gradient (MADDPG) approach. This involved adapting DDPG algorithms to support cooperative and competitive interactions between two agents with the goal of maintaining a ball in play. Our approach, grounded in shared learning and exploration of advanced multi-agent coordination techniques, culminated in achieving the target average score, showcasing the effectiveness of collaborative reinforcement learning strategies in a controlled setting and setting a foundation for further exploration in complex multi-agent scenarios.
