import carla
EXPERIMENT_NAME = "prueba"

################################################
####              VAE TRAINING              ####
################################################

Z_SIZE = 512  # Only used for random features
# Input dimension for VAE
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 128
N_CHANNELS = 3 #3

VAE_SEED = 0
VAE_N_SAMPLES = -10024
VAE_BATCH_SIZE = 64
VAE_LEARNING_RATE = 3e-4
VAE_KERNEL_SIZE = 5
VAE_KL_TOLERANCE = 0
VAE_BETA = 2
VAE_N_EPOCHS = 100
VAE_SEMANTIC_SEGMENTATION = 1
VAE_DATA_AUGMENTATION = 0
VAE_NORM_MODE = "rl"
VAE_GREEN_MULTIPLIER = 1.0
VAE_NAME = ""
VAE_VERBOSE = 0

INPUT_DIM = (IMAGE_HEIGHT, IMAGE_WIDTH, N_CHANNELS)

################################################
####          DATASET COLLECTION            ####
################################################

COLLECTION_STEPS_PER_RESET = 2000
COLLECTION_STEPS_PER_MAP = 200000
COLLECTION_RESETS_PER_MAP = 20
COLLECTION_TARGET_SPEED = 10
COLLECTION_INPUT_NOISE = True
COLLECTION_MAPS = [ "Town02"]#"Town01",, "Town07"]
COLLECTION_HIDDEN_OBJECTS = [carla.CityObjectLabel.Buildings, carla.CityObjectLabel.Fences, carla.CityObjectLabel.Other] 
                            # Nunca usar, crashea carla.CityObjectLabel.Dynamic
                            #Ultra lento carla.CityObjectLabel.Water carla.CityObjectLabel.Terrain
                            #Lento carla.CityObjectLabel.GuardRail carla.CityObjectLabel.Vegetation carla.CityObjectLabel.Walls carla.CityObjectLabel.Static
COLLECTION_WEATHER = "random"

################################################
####              RL TRAINING               ####
################################################

CRITIC_LEARNING_RATE = 3e-4 
ACTOR_LEARNING_RATE = 3e-4 
ALPHA_LEARNING_RATE = 3e-4
RL_BATCH_SIZE = 256
TARGET_UPDATE_TAU = 0.005 
TARGET_UPDATE_PERIOD = 1
GAMMA = 0.99
REWARD_SCALE_FACTOR = 1.0
NUM_EPISODES = 100
NUM_STEPS = 1000000
INITIAL_COLLECT_STEPS = 10000
STEPS_PER_EPISODE = 1000
GRADIENT_STEPS = 1
REPLAY_BUFFER_CAPACITY = 300000
ACTOR_FC_LAYER_PARAMS = (256, 256)
CRITIC_JOINT_FC_LAYER_PARAMS = (256, 256)


# Reward parameters
THROTTLE_REWARD_WEIGHT = 0.1 #0.1
JERK_REWARD_WEIGHT = 0 #-0.1 #0.0
BASE_REWARD = 1 #1
# Negative reward for getting off the road
REWARD_CRASH = -10
# Penalize the agent even more when being fast
CRASH_SPEED_WEIGHT = 5 / 20

# Min speed the agent has to achieve
MIN_SPEED = 1.0
# Maximum number of ticks that the agent can stay without increasing speed above MIN_SPEED
NUM_TICK_WITHOUT_MIN_SPEED = 40

# Max cross track error (used in normal mode to reset the car)
MAX_CTE_ERROR = 1.3 #2.0
NUM_EPISODES_FOR_MAP_CHANGE = 10
# Level to use for training
LEVEL = 0

# Action repeat
FRAME_SKIP = 1
# Symmetric command
MAX_STEERING = 1
MIN_STEERING = - MAX_STEERING
# very smooth control: 10% -> 0.2 diff in steering allowed (requires more training)
# smooth control: 15% -> 0.3 diff in steering allowed
MAX_STEERING_DIFF = 1 #0.15
# Simulation config
MIN_THROTTLE = 0.4 #0.4
# max_throttle: 0.6 for level 0 and 0.5 for level 1
MAX_THROTTLE = 0.6 #0.6
# Number of past commands to concatenate with the input
N_COMMAND_HISTORY = 30 #30 #20

LEADING_INSTRUCTIONS = 10
INCLUDE_SPEED = 1
INCLUDE_ACCEL = 1
INCLUDE_CTE = 0
INCLUDE_ANGLE_DIFF = 0
INCLUDE_JUNCTION = 0
INCLUDE_VAE = 1

IMAGE_NOISE = 0
LATENT_NOISE = 0
ACTION_NOISE = 0