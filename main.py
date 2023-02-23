from rl.rl_training import RLTraining
from rl.data_collecting import DataCollection
from carla_env.carla_env import CarlaEnv
from carla_agents.rl_agent import RLAgent
from carla_agents.auto_agent import AutoAgent
from carla_agents.mario_agent import MarioAgent
import numpy as np
import sys
from config import MAX_CTE_ERROR, MAX_STEERING_DIFF, COLLECTION_TARGET_SPEED, COLLECTION_INPUT_NOISE

def noisy_control(steer, throttle, info_dict):
    sd = (1 - (abs(info_dict["cte"]) / MAX_CTE_ERROR)) * (1 - abs(info_dict["angle_diff"]))
    sd = max(sd, 0)
    new_steer = max(min(np.random.normal(steer, sd * .5), 1), -1)

    return new_steer, throttle

agent = AutoAgent('localhost', 2000, target_speed = COLLECTION_TARGET_SPEED)

#agent = MarioAgent('localhost', 2000)

agent_list = [("agente", agent)]
#La anchura multiplo de 64

env = CarlaEnv( [s[0] for s in agent_list],
                [s[1].sensors() for s in agent_list],
                [['vehicle.lincoln.mkz_2017', {}]],
                ["rgb", "semantic_segmentation"], 
                [noisy_control])

#training = RLTraining(env, agent_list)
#training.run(save_data = False, render = True, render_mode = "rgb", input_noise = True)
#training.run(save_data = True, render_mode = "all", input_noise = True)
data_collection = DataCollection(env, agent_list)
data_collection.run(render=False, render_mode="rgb", input_noise=True)