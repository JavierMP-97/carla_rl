
import os
import keyboard
from typing import List, Tuple, Dict, Any, Union
from leaderboard.autoagents.autonomous_agent import AutonomousAgent
import cv2
import gym
import carla
import time

#agent = RLAgent()
#env = CarlaEnv(agent.sensors())

class RLTraining():
    """A class that executes a custom loop for training or testing agents in a random way in the carla environment
    
        Special keybinds:
        :keybind Q: Close the environment and exit the loop
        :keybind W: Resets the environment
        :keybind E: Resets the environment and changes the simulator map
    """
    
    def __init__(self, env: gym.Env, agent_list: List[Tuple[str, AutonomousAgent]]) -> None:
        """
        Constructor
        
        :param env: A carla environment
        :param agen_list: A list which contains a tuple for each agent, with its string id and the agent object
        """
        self.env: gym.Env = env
        self.agent_list: List[Tuple[str, AutonomousAgent]] = agent_list
        '''
        self.state_list: List[Dict[str, Tuple[int, Any]]] = [None]
        self.reward_list: List[float] = [None]
        self.game_over_list: List[bool] = [None]
        self.info_list: List[Dict[str, Any]] = [None]
        '''
                    
    def run(self, save_data: bool = True, render: bool = False, render_mode: str = "none", input_noise: bool = False) -> None:
        """
        A function that starts the execution loop

        :param bool save_data: A flag that indicates if the step data must be saved
        :param bool render: A flag that indicates if the state must be rendered
        :param str render: A string that indicates the camera type that will be rendered. If "all" is specified, every type will be rendered and stacked
        :param bool input_noise: A flag that indicates if the loop will apply noise to the controls of the agents
        """

        runing = True
        key_pressed = {}
        key_pressed["e"] = False
        key_pressed["w"] = False
        key_pressed["q"] = False
        self.game_over_list = [False]
        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 0.0
        control.hand_brake = False
        
        while runing and not key_pressed["q"]:
            
            self.state_list, self.info_list = self.env.reset(change_map=key_pressed["e"])
            if render:
                self.env.render(idx = 0, mode = render_mode)
            if save_data:
                self.env.save_current_step(save_data, save_data)

            for idx, _ in enumerate(self.state_list):
                self.state_list[idx]["game_over"] = False
                self.state_list[idx]["reward"] = 0
                self.state_list[idx]["first"] = True
            
            key_pressed["e"] = False
            key_pressed["w"] = False
            key_pressed["q"] = False

            while not self.game_over_list[0] and not key_pressed["e"] and not key_pressed["w"]:
                

                key_pressed["e"] = keyboard.is_pressed("e")
                key_pressed["w"] = keyboard.is_pressed("w")
                key_pressed["q"] = keyboard.is_pressed("q")

                action = self.agent_list[0][1].run_step(self.state_list[0], self.info_list[0]["timestamp"], self.info_list[0])             

                self.state_list, self.reward_list, self.game_over_list, self.info_list = self.env.step([action], input_noise)

                if render:
                    self.env.render(idx = 0, mode = render_mode)
                if save_data:
                    self.env.save_current_step(save_data, save_data)
                
                for idx, _ in enumerate(self.state_list):
                    self.state_list[idx]["reward"] = self.reward_list[idx]
                    self.state_list[idx]["game_over"] = self.game_over_list[idx]
                    self.state_list[idx]["first"] = False
                
                #agent._train()

                #agent.collect_tuple(state, action, next_state, reward, game_over)

                if key_pressed["e"] or key_pressed["w"] or key_pressed["q"]:
                    self.game_over_list[0] = True
            
            #agent.add_to_buffer(state, action)

            self.game_over_list[0] = False
            
        self.env.close()

#save_data(agent_list, state_list, reward_list, game_over_list, info_list)