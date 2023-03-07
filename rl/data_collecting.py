
import os
import keyboard
from typing import List, Tuple, Dict, Any, Union
from leaderboard.autoagents.autonomous_agent import AutonomousAgent
import cv2
import gym
from config import COLLECTION_STEPS_PER_RESET, COLLECTION_RESETS_PER_MAP, COLLECTION_MAPS, COLLECTION_HIDDEN_OBJECTS, COLLECTION_WEATHER, COLLECTION_STEPS_PER_MAP


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
                    
    def run(self, render: bool = False, render_mode: str = "none", input_noise: bool = False, change_map_by_steps: bool = True) -> None:
        """
        A function that starts the execution loop

        :param bool save_data: A flag that indicates if the step data must be saved
        :param bool render: A flag that indicates if the state must be rendered
        :param str render: A string that indicates the camera type that will be rendered. If "all" is specified, every type will be rendered and stacked
        :param bool input_noise: A flag that indicates if the loop will apply noise to the controls of the agentsç
        :param bool change_map_by_steps: A flag that indicates if the map should be changed once enough steps have been taken or if enough resets have been taken
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
        keyboard.add_hotkey("ctrl+e", lambda: press_e() )
        keyboard.add_hotkey("ctrl+w", lambda: press_w() )
        keyboard.add_hotkey("ctrl+q", lambda: press_q() )
        self.game_over_list = [False]     

        episodes = 0

        total_steps = 0

        steps = 0

        while runing and not key_pressed["q"] and (
                (not change_map_by_steps and episodes < len(COLLECTION_MAPS) * COLLECTION_RESETS_PER_MAP) or 
                (change_map_by_steps and total_steps < len(COLLECTION_MAPS) * COLLECTION_STEPS_PER_MAP)
                ):
            if total_steps % 10000 == 0:
                print(total_steps)
            map_name = "Town01"
            if change_map_by_steps:
                map_name = COLLECTION_MAPS[int(total_steps / COLLECTION_STEPS_PER_MAP)]
            else:
                map_name = COLLECTION_MAPS[int(episodes / COLLECTION_RESETS_PER_MAP)]

            change_map = False
            if not change_map_by_steps and episodes != 0 and episodes % COLLECTION_RESETS_PER_MAP == 0:
                change_map = True
            elif change_map_by_steps and total_steps != 0 and total_steps % COLLECTION_STEPS_PER_MAP == 0:
                change_map = True
            
            self.state_list, self.info_list = self.env.reset(   change_map=(key_pressed["e"] or change_map), 
                                                                change_weather = True, weather = COLLECTION_WEATHER,
                                                                map_name = map_name, hide_objects = True, hidden_objects_list = COLLECTION_HIDDEN_OBJECTS)
            if render:
                self.env.render(idx = 0, mode = render_mode)
            self.env.save_current_step(True, True)
            
            reset_keys()

            steps += 1
            total_steps += 1

            while not self.game_over_list[0] and not key_pressed["e"] and not key_pressed["w"] and steps < COLLECTION_STEPS_PER_RESET:
                if total_steps % 10000 == 0:
                    print(total_steps)
                if change_map_by_steps and total_steps % COLLECTION_STEPS_PER_MAP == 0:
                    break

                action = self.agent_list[0][1].run_step(self.state_list[0], self.info_list[0]["timestamp"], self.info_list[0])             

                self.state_list, self.reward_list, self.game_over_list, self.info_list = self.env.step([action], input_noise)

                if render:
                    self.env.render(idx = 0, mode = render_mode)
                
                self.env.save_current_step(True, True)

                if key_pressed["e"] or key_pressed["w"] or key_pressed["q"]:
                    self.game_over_list[0] = True

                steps += 1
                total_steps += 1

            self.game_over_list[0] = False
            episodes += 1
            steps = 0
        input("Press Enter to continue...")
        self.env.close()