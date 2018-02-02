
class Config:
    VERSION = 1.05
    MEMORY_SIZE = 100000
    NUM_LAST_FRAMES = 4
    LEVEL = "snakeai/levels/10x10-blank.json"
    NUM_EPISODES = 10000
    BATCH_SIZE = 64
    DISCOUNT_FACTOR = 0.95
    USE_PRETRAINED_MODEL = False
    PRETRAINED_MODEL = "dqn-00000000.model"
    MAX_EXPLORATION = 1.0
    MIN_EXPLORATION = 0.1
    # Either sarsa, dqn, ddqn
    LEARNING_METHOD = "ddqn"
    MULTI_STEP_REWARD = False
    #foodspeed =0 no movement. foodspeed =2 food moves one step every 2 timesteps
    FOODSPEED = 0
    LOG_FREQUENCY = 10