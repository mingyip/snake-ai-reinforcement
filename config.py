class Config:
	    VERSION = 1.04
	    MEMORY_SIZE = 100000
	    NUM_LAST_FRAMES = 4
	    LEVEL = "snakeai/levels/10x10-blank.json"
	    NUM_EPISODES = 50000 #### changed
	    BATCH_SIZE = 64
	    DISCOUNT_FACTOR = 0.95
	    USE_PRETRAINED_MODEL = False
	    PRETRAINED_MODEL = "dqn-00090000.model"
	    MAX_EXPLORATION = 1.0
	    MIN_EXPLORATION = 0.1
	    # Either sarsa, dqn, ddqn
	    LEARNING_METHOD = "sarsa" #### changed 
	    #foodspeed =0 no movement. foodspeed =2 food moves one step every 2 timesteps
	    FOODSPEED = 2 ### changed
	    LOG_FREQUENCY = 100
