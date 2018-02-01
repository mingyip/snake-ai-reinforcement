
class Config:
    VERSION = 1.08
    MEMORY_SIZE = 100000
    NUM_LAST_FRAMES = 4
    LEVEL = "snakeai/levels/10x10-blank.json"
    NUM_EPISODES = 50000
    BATCH_SIZE = 64
    DISCOUNT_FACTOR = 0.95
    USE_PRETRAINED_MODEL = True
    PRETRAINED_MODEL = "outputs/20180131-233536_sarsa_50000epsiodes_10x10-blank.json/dqn-00020000.model"
    MAX_EXPLORATION = 0.1
    MIN_EXPLORATION = 0.1
    # Either sarsa, dqn, ddqn
    LEARNING_METHOD = "sarsa"
    #foodspeed =0 no movement. foodspeed =2 food moves one step every 2 timesteps
    FOODSPEED = 0
    LOG_FREQUENCY = 10