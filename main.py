from rl.rl_training import RLTraining
from carla_env.carla_training_env import CarlaEnv
from carla_agents.rl_agent import RLAgent
from carla_agents.auto_agent import AutoAgent
from carla_agents.mario_agent import MarioAgent
import numpy as np
import sys
from config import MAX_CTE_ERROR, MAX_STEERING_DIFF

def noisy_control(steer, throttle, info_dict):
    sd = (1 - (abs(info_dict["cte"]) / MAX_CTE_ERROR)) * (1 - abs(info_dict["angle_diff"]))
    sd = max(sd, 0)
    new_steer = max(min(np.random.normal(steer, sd), 1), -1)

    return new_steer, throttle

agent = AutoAgent('localhost', 2000)

#agent = MarioAgent('localhost', 2000)

agent_list = [("agente2", agent)]
#La anchura multiplo de 64
env = CarlaEnv( [s[0] for s in agent_list],
                [s[1].sensors() for s in agent_list],
                [['vehicle.lincoln.mkz_2017', {}]],
                ["rgb", "semantic_segmentation"],
                [noisy_control])

training = RLTraining(env, agent_list)

training.run(save_data = False, render = False, render_mode = "rgb", input_noise = False)
#training.run(save_data = True, render_mode = "all", input_noise = True)