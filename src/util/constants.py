LOCAL_REGISTRY = "local-registry"

GRID_TARGET_MDP = "grid-target"

SKELETON_AGENT = "skl"
STABLE_BASELINES3_AGENT = "sb3"

DQN = "dqn"
HUMAN = "human"
MACHINE = "machine"
MACHINE_TRAINING = "machine-training"
MONTE_CARLO = "monte-carlo"
MONTE_CARLO_POLICY_GRADIENT = "monte-carlo-pg"
Q_LEARNING = "q-learning"
Q_NETWORK = "q-network"

NORTH = 0
EAST  = 1
SOUTH = 2
WEST  = 3

BLACK = (  0,   0,   0)
BLUE  = (  0,   0, 255)
GREEN = (  0, 255,   0)
RED   = (255,   0,   0)
GREY  = (245, 245, 245)
WHITE = (255, 255, 255)


THEME_BLUE_DARK   = (  0,  52,  89)
THEME_BLUE_LIGHT  = (  0, 168, 232)
THEME_BLUE_MEDIUM = (  0, 126, 167)
THEME_DARK        = (  0,  23,  31)


X = 0
Y = 1

SB3_DQN = f"{STABLE_BASELINES3_AGENT}-{DQN}"
SKL_QLEARNING = f"{SKELETON_AGENT}-{Q_LEARNING}"
