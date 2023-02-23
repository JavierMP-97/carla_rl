
import os
import keyboard
from typing import List, Tuple, Dict, Any, Union
from leaderboard.autoagents.autonomous_agent import AutonomousAgent
import cv2
import gym
from config import COLLECTION_STEPS_PER_RESET, COLLECTION_RESETS_PER_MAP, COLLECTION_MAPS, COLLECTION_HIDDEN_OBJECTS, COLLECTION_WEATHER


class DataCollection():
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
                    
    def run(self, render: bool = False, render_mode: str = "none", input_noise: bool = False) -> None:
        """
        A function that starts the execution loop

        :param bool save_data: A flag that indicates if the step data must be saved
        :param bool render: A flag that indicates if the state must be rendered
        :param str render: A string that indicates the camera type that will be rendered. If "all" is specified, every type will be rendered and stacked
        :param bool input_noise: A flag that indicates if the loop will apply noise to the controls of the agents
        """

        runing = True
        key_pressed = {}
        def reset_keys():
            key_pressed["e"] = False
            key_pressed["w"] = False
            key_pressed["q"] = False
        reset_keys()
        def press_e():
            key_pressed["e"] = True
        def press_w():
            key_pressed["w"] = True
        def press_q():
            key_pressed["q"] = True
        keyboard.on_press_key("ctrl+e", lambda e: press_e() )
        keyboard.on_press_key("ctrl+w", lambda e: press_w() )
        keyboard.on_press_key("ctrl+q", lambda e: press_q() )
        self.game_over_list = [False]     

        episodes = 0

        while runing and not key_pressed["q"] and episodes < len(COLLECTION_MAPS) * COLLECTION_RESETS_PER_MAP:

            map_name = COLLECTION_MAPS[int(episodes / COLLECTION_RESETS_PER_MAP)]
            
            self.state_list, self.info_list = self.env.reset(   change_map=(key_pressed["e"] or (episodes != 0 and episodes % COLLECTION_RESETS_PER_MAP == 0)), 
                                                                change_weather = True, weather = COLLECTION_WEATHER,
                                                                map_name = map_name, hide_objects = True, hidden_objects_list = COLLECTION_HIDDEN_OBJECTS)
            if render:
                self.env.render(idx = 0, mode = render_mode)
            self.env.save_current_step(True, True)
            
            reset_keys()

            steps = 0

            while not self.game_over_list[0] and not key_pressed["e"] and not key_pressed["w"] and steps < COLLECTION_STEPS_PER_RESET:

                action = self.agent_list[0][1].run_step(self.state_list[0], self.info_list[0]["timestamp"], self.info_list[0])             

                self.state_list, self.reward_list, self.game_over_list, self.info_list = self.env.step([action], input_noise)

                if render:
                    self.env.render(idx = 0, mode = render_mode)
                
                self.env.save_current_step(True, True)

                if key_pressed["e"] or key_pressed["w"] or key_pressed["q"]:
                    self.game_over_list[0] = True

                steps += 1

            self.game_over_list[0] = False
            episodes += 1
        self.env.close()