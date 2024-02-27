random_seed = 1
target_score = 0.5
n_episodes=3500            # maximum number of training episodes
max_t = 1000               # maximum number of timesteps per episode # 10000
moving_average_period = 100
print_interval = 100
BUFFER_SIZE = int(1e6)     # replay buffer size # 1e4 , 1e6
BATCH_SIZE = 1024           # minibatch size # 128 , 512, 256 , 64, 1024, 1280      1024
GAMMA = 0.99               # discount factor # 0.99
TAU = 1e-3                 # for soft update of target parameters # 1e-3
LR_ACTOR = 1e-3            # learning rate of the actor # 1e-4
LR_CRITIC = 1e-3           # learning rate of the critic # 1e-3, 1e-4
WEIGHT_DECAY = 0           # L2 weight decay # 0 # 1e-4, 1e-5, 0, 1e-2
LEARNING_ITERATIONS = 1    # number of times the agent learn at each time step, based on samples taken from the replay memory # 1
NOISE_SIGMA_INITIAL = 0.2  # Initial sigma value (higher= greater exploration) 0.2
NOISE_DECAY_EPISODES_TO_MIN_SIGMA = 800 # Number of episodes to reach SIGMA min 
NOISE_SIGMA_MIN = 0.01     # Minimum sigma value to prevent noise from disappearing completely
PER_ALPHA = 0.8            #controls the degree of prioritization used in Prioritized Experience Replay (PER). A value of 0 corresponds to uniform random sampling (no prioritization), while a value closer to 1 increases the focus on high-error experiences. Alpha helps to balance the exploration of the experience space (alpha near 0) with focused learning on high-priority experiences (alpha near 1).
PER_BETA_INITIAL = 0.4     # Initial beta value  0.4
PER_BETA_MAX = 1.0         # Maximum beta value  1.0
PER_BETA_N_EPISODES_MAX = 800 # Number of episodes to reach BETA max

