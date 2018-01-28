
class Config:
    VERSION = 1.02
    MEMORY_SIZE = -1
    NUM_LAST_FRAMES = 4
    LEVEL = "snakeai/levels/10x10-blank.json"
    NUM_EPISODES = 100
    BATCH_SIZE = 64
    DISCOUNT_FACTOR = 0.95
    USE_PRETRAINED_MODEL = False
    PRETRAINED_MODEL = "dqn-00090000.model"
    MAX_EXPLORATION = 1.0
    MIN_EXPLORATION = 0.1
    SARSA = True
    #foodspeed =0 no movement. foodspeed =2 food moves one step every 2 timesteps
    FOODSPEED = 0