"""
Environment for reinforcement learning with carla
"""

import sys
import os
import glob
import time
from typing import List, Tuple, Dict, Any, Callable, Optional, Union

import weakref
import re
import random
import numpy as np
import math

import cv2

import gym

import carla
#from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from leaderboard.envs.sensor_interface import CallBack, OpenDriveMapReader, SpeedometerReader, SensorInterface
from carla_env.sensors.collision_sensor import CollisionSensor, ModCallBack
from srunner.scenariomanager.timer import GameTime
from agents.navigation.local_planner import LocalPlanner, RoadOption
from agents.navigation.global_route_planner import GlobalRoutePlanner
#from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
from config import REWARD_CRASH, CRASH_SPEED_WEIGHT, MAX_CTE_ERROR, NUM_EPISODES_FOR_MAP_CHANGE, MIN_SPEED, NUM_TICK_WITHOUT_MIN_SPEED

ROAD_OPTIONS = [RoadOption.LEFT, RoadOption.STRAIGHT, RoadOption.RIGHT] #, RoadOption.LANEFOLLOW, RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT]
ALL_ROAD_OPTIONS = [RoadOption.LEFT, RoadOption.STRAIGHT, RoadOption.RIGHT, RoadOption.CHANGELANELEFT, RoadOption.LANEFOLLOW, RoadOption.CHANGELANERIGHT]

DEFAULT = ["Default"]
CLEAR = ["ClearNoon", "ClearSunset"]
CLOUD = ["CloudyNoon", "CloudySunset"]
WET = ["WetNoon", "WetSunset"]
CLOUDWET = ["WetCloudyNoon", "WetCloudySunset"]
STORM = ["HardRainNoon", "HardRainSunset", "MidRainSunset", "MidRainyNoon", "SoftRainNoon", "SoftRainSunset"]
MAPS = ["Town01", "Town02", "Town07"]
THROTTLE = 1
STEERING = 0

control_zero = carla.VehicleControl()

control_zero.steer = 0.0
control_zero.throttle = 0.0
control_zero.brake = 0.0
control_zero.hand_brake = False

#class Vehicle():
#    pass

class CEnvAgent():
    """Class that contains info and carla objects related to an agent"""
    def __init__(self, name: str, vehicle_description: Tuple[str, Dict[str, Any]], sensor_description_list: List[Dict[str, Any]]):
        """
        Constructor

        :param str name: Name id of the agent
        :param Tuple[str, Dict[str, Any]] vehicle_description: A tuple with the blueprint name of the vehicle and a dict with its config
        :param List[Dict[str, Any]] sensor_description_list: A list with the description of each sensor of the agent. See sensors() from carla leaderboards
        """

        self.actor: carla.Vehicle = None
        self.name: str = name
        self.vehicle_description: Tuple[str, Dict[str, Any]] = vehicle_description
        self.sensor_list: List = []
        self.sensor_description_list: List[Dict[str, Any]] = sensor_description_list
        self.route: List[Tuple[carla.Waypoint, RoadOption]] = None
        self.sensor_interface: SensorInterface = None
        self.speed_array: List[float] = [MIN_SPEED] * NUM_TICK_WITHOUT_MIN_SPEED
        self.last_throttle: float = 0.0
        self.last_steering: float = 0.0
        self.last_noisy_throttle: float = 0.0
        self.last_noisy_steering: float = 0.0
        self.cte: float = 0.0
        self.angle_diff: float = 0.0
        self.junction_distance: float = 0.0
        self.success: int = 0
        self.next_instruction: int = -1
        self.next_instruction_distance: float = 0.0
        self.speed_sensor: SpeedometerReader = None
        self.collision_sensor: CollisionSensor = None
        self.data_dict: Dict[str, Tuple[int, Any]] = None
        self.info: Dict[str, Any] = None
        self.current_idx: int = 0
        agent_path = os.path.realpath(os.path.dirname(__file__)) + "\\..\\log\\" + self.name
        if not os.path.isdir(agent_path):
            os.mkdir(agent_path)
            self.current_idx = 0
        else:
            onlyfiles = [f for f in os.listdir(agent_path) if os.path.isfile(os.path.join(agent_path, f))]
            if len(onlyfiles) > 0:
                last_idx = int(onlyfiles[-1].split(".")[0].split("_")[-1])
                self.current_idx = last_idx + 1
            else:
                self.current_idx = 0
        self.input_noise_function: Callable[[float, float, Optional[Dict[str, Any]]], Tuple[float, float]] = None

    def clear(self) -> None:
        for sensor_idx, sensor in enumerate(self.sensor_list):
            if sensor:
                #if isinstance(sensor, carla.Actor):
                sensor.destroy()
                self.sensor_list[sensor_idx] = None

        self.speed_sensor.destroy()
        self.collision_sensor.destroy()
        self.speed_sensor = None
        self.collision_sensor = None
        self.sensor_list = []
        self.sensor_interface = None
        if self.actor is not None:
            self.actor.destroy()
        self.actor = None
        self.reset_variables()

    def reset_variables(self) -> None:
        self.data_dict = None
        self.info = None
        self.route = None
        self.speed_array = [MIN_SPEED] * NUM_TICK_WITHOUT_MIN_SPEED
        self.last_throttle = 0.0
        self.last_steering = 0.0
        self.last_noisy_throttle = 0.0
        self.last_noisy_steering = 0.0
        self.cte = 0.0
        self.angle_diff = 0.0
        self.junction_distance = 0.0
        self.success = 0
        self.next_instruction = -1
        self.next_instruction_distance = 0.0
        if self.collision_sensor is not None:
            self.collision_sensor.reset()

    def get_actor(self) -> carla.Vehicle:
        return self.actor
    def get_name(self) -> str:
        return self.name
    def get_vehicle_description(self) -> Tuple[str, Dict[str, Any]]:
        return self.vehicle_description
    def get_sensor_list(self) -> List:
        return self.sensor_list
    def get_sensor(self, idx: int):
        if idx >= len(self.sensor_list) or idx < -len(self.sensor_list):
            print("Error: index {} out of bounds".format(idx))
            return None
        return self.sensor_list[idx]
    def get_sensor_description_list(self) -> List[Dict[str, Any]]:
        return self.sensor_description_list
    def get_sensor_description(self, idx: int) -> Dict[str, Any]:
        if idx >= len(self.sensor_description_list) or idx < -len(self.sensor_description_list):
            print("Error: index {} out of bounds".format(idx))
            return None
        return self.sensor_description_list[idx]
    def get_route(self) -> List[Tuple[carla.Waypoint, RoadOption]]:
        return self.route
    def get_sensor_interface(self) -> SensorInterface:
        return self.sensor_interface
    def get_speed_array(self) -> List[float]:
        return self.speed_array
    def get_speed(self) -> float:
        return self.speed_array[0]
    def get_last_throttle(self) -> float:
        return self.last_throttle
    def get_last_steering(self) -> float:
        return self.last_steering
    def get_last_noisy_throttle(self) -> float:
        return self.last_noisy_throttle
    def get_last_noisy_steering(self) -> float:
        return self.last_noisy_steering
    def get_cte(self) -> float:
        return self.cte
    def get_angle_diff(self) -> float:
        return self.angle_diff
    def get_junction_distance(self) -> float:
        return self.junction_distance
    def get_success(self) -> int:
        return self.success
    def get_next_instruction(self) -> int:
        return self.next_instruction
    def get_next_instruction_distance(self) -> float:
        return self.next_instruction_distance
    def get_speed_sensor(self) -> SpeedometerReader:
        return self.speed_sensor
    def get_collision_sensor(self) -> CollisionSensor:
        return self.collision_sensor
    def get_data_dict(self) -> Dict[str, Tuple[int, Any]]:
        return self.data_dict
    def get_info(self) -> Dict[str, Any]:
        return self.info
    def get_current_idx(self) -> int:
        return self.current_idx
    def get_input_noise_function(self) -> Callable[[float, float, Optional[Dict[str, Any]]], Tuple[float, float]]:
        return self.input_noise_function

    def set_actor(self, actor: carla.Vehicle):
        self.actor = actor
    def set_name(self, name: str):
        self.name = name
    def set_vehicle_description(self, vehicle_description: Tuple[str, Dict[str, Any]]):
        self.vehicle_description = vehicle_description
    def set_sensor_list(self, sensor_list: List):
        self.sensor_list = sensor_list
    def set_sensor(self, sensor, idx: int):
        if idx >= len(self.sensor_list) or idx < -len(self.sensor_list):
            print("Error: index {} out of bounds".format(idx))
        else:
            self.sensor_list[idx] = sensor
    def append_sensor(self, sensor):
        self.sensor_list.append(sensor)
    def set_sensor_description_list(self, sensor_description_list: List[Dict[str, Any]]):
        self.sensor_description_list = sensor_description_list
    def set_route(self, route: List[Tuple[carla.Waypoint, RoadOption]]):
        self.route = route
    def set_sensor_interface(self, sensor_interface: SensorInterface):
        self.sensor_interface = sensor_interface
    def set_speed_array(self, speed_array: List[float]):
        self.speed_array = speed_array
    def add_speed_meassurement(self, speed: float):
        self.speed_array[1:] = self.speed_array[0:-1]
        self.speed_array[0] = speed
    def set_last_throttle(self, last_throttle: float):
        self.last_throttle = last_throttle
    def set_last_steering(self, last_steering: float):
        self.last_steering = last_steering
    def set_last_noisy_throttle(self, last_noisy_throttle: float):
        self.last_noisy_throttle = last_noisy_throttle
    def set_last_noisy_steering(self, last_noisy_steering: float):
        self.last_noisy_steering = last_noisy_steering
    def set_cte(self, cte: float):
        self.cte = cte
    def set_angle_diff(self, angle_diff: float):
        self.angle_diff = angle_diff
    def set_junction_distance(self, junction_distance: float):
        self.junction_distance = junction_distance
    def set_success(self, success: int):
        self.success = success
    def set_next_instruction(self, next_instruction: int):
        self.next_instruction = next_instruction
    def set_next_instruction_distance(self, next_instruction_distance: float):
        self.next_instruction_distance = next_instruction_distance
    def set_speed_sensor(self, speed_sensor: SpeedometerReader):
        self.speed_sensor = speed_sensor
    def set_collision_sensor(self, collision_sensor: CollisionSensor):
        self.collision_sensor = collision_sensor
    def set_data_dict(self, data_dict: Dict[str, Tuple[int, Any]]):
        self.data_dict = data_dict
    def set_info(self, info: Dict[str, Any]):
        self.info = info
    def set_current_idx(self, current_idx: int):
        self.current_idx = current_idx
    def set_input_noise_function(self, input_noise_function: Callable[[float, float, Optional[Dict[str, Any]]], Tuple[float, float]]):
        self.input_noise_function = input_noise_function
    



class CarlaEnv(gym.Env):
    # Init connection
    # Init managers
    # Reset?
    def __init__(   self, agent_names: List[str], sensor_descriptions: List[List[Dict[str, Any]]], 
                    vehicle_description: List[Tuple[str, Dict[str, Any]]], 
                    camera_types: List[str] = ["rgb"], 
                    input_noise_function_list: Optional[Callable[[float, float, Optional[Dict[str, Any]]], Tuple[float, float]]] = None
                ) -> None:
        super(CarlaEnv, self).__init__()
        """
        Constructor

        :param List[str] agent_names: A list with the name ids of the agents
        :param sensor_descriptions: A list with one list of sensor descriptions for each agent. See sensors() from carla leaderboards
        :param vehicle_description: A list with a tuple for each agent with the blueprint name of the vehicle and a dict with its config
        :param List[str] camera_types: A list with the camera types that you want the agent to have: "rgb", "semantic_segmentation", "depth"
        :param input_noise_function_list: (Optional) A list with a function for each agent
                that will introduce noise in its controls
                The params of that function will be:
                    :param float steer: steer to be modified
                    :param float throttle: throttle to be modified
                    :return: a tuple with the modified steer and throttle
        """
        self._grp: GlobalRoutePlanner = None
        self.change_map_counter: int = 0
        self.first_episode: bool = True
        self.map_idx: int = 0
        self.frame_rate: int = 20

        self.route_resolution: float = 1.0

        self.frame: int = 0

        self.spawn_index: int = 0

        self.agent_list: List[CEnvAgent] = []

        

        for agent_idx, (an, sd, vd) in enumerate(zip(agent_names, sensor_descriptions, vehicle_description)):
            self.agent_list.append(CEnvAgent(an, vd, sd))
            if input_noise_function_list != None:
                self.agent_list[agent_idx].set_input_noise_function(input_noise_function_list[agent_idx])
        
        self.camera_types: List[str] = camera_types

        try:
            self.client: carla.Client = carla.Client('localhost', 2000)
            self.client.set_timeout(100.0)
            self.world: carla.World = self.client.get_world()
        except Exception as e:
            print("Error connecting to Carla")
            print(e)

        '''
        settings = self.world.get_settings()
        if True: #self._args.sync:  
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 1.0 / self.frame_rate
        self.world.apply_settings(settings)

        '''

            
    # Execute one time step within the environment
    # Read env
    # Act
    def step(self, control: carla.VehicleControl, input_noise: bool = False) -> Tuple[List[Dict[str, Any]], List[float], List[bool], List[Dict[str, Any]]]:
        """
        This function takes a list of vehicle controls, apply each control for each agent and advances the environment one step

        :param control: A list of carla vehicle controls, one for each agent
        :param bool input_noise: A flag that indicates if the input noise function will be used to alter the controls
        :return: A tuple with
            - data_dict_list: One dictionary for each agent with the data from each sensor. The key to access the data of each sensor will be the id of that senor (except cameras)
            - reward: One reward value for each agent
            - game_over: A flag for each agent that indicates if has crashed or failed
            - info_list: One dictionary for each agent with any data that you could ever wish for 
        """
        self.take_action(control, input_noise)

        self._tick()

        spectator = self.world.get_spectator()
        transform = self.agent_list[0].get_actor().get_transform()
        spectator.set_transform(carla.Transform(transform.location + carla.Location(z=50), carla.Rotation(pitch=-90)))

        #self.collision_history = self.collision_sensor.history
        #self.lane_crossed = self.lane_sensor.lane_crossed
        self.get_ct_angle_dif_instruction()

        ### AÑADIR LA VELOCIDAD AL AGENTE ###       

        info_list = []
        data_dict_list = []
        for agent in self.agent_list:
            data_dict = agent.get_sensor_interface().get_data(self.frame)

            agent.add_speed_meassurement(agent.get_speed_sensor()._get_forward_speed())

            data_dict_list.append(data_dict)
            
            info = {"cte":agent.get_cte(),
                    "angle_diff":agent.get_angle_diff(),
                    "throttle": agent.get_last_throttle(),
                    "steering": agent.get_last_steering(),
                    "noisy_throttle": agent.get_last_noisy_throttle(),
                    "noisy_steering": agent.get_last_noisy_steering(),
                    "instruction": agent.get_next_instruction(),
                    "instruction_distance": agent.get_next_instruction_distance(),
                    "junction_distance": agent.get_junction_distance(),
                    "reward":self.calc_reward_agent(agent),
                    "is_game_over":int(self.is_game_over_agent(agent)),
                    "success":agent.get_success(),
                    "timestamp": self.frame, #GameTime.GameTime.get_time()
                    "CEnvAgent": agent
                    }
            
            info_list.append(info)

            agent.set_data_dict(data_dict)
            agent.set_info(info)            

        return data_dict_list, self.calc_reward(), self.is_game_over(), info_list
        
    # Reset the state of the environment to an initial state
    # Destroy actors
    # Change town
    # Change weather 
    # Respawn actors (different position)
    def reset(  self, change_map: bool = False, map_name: Optional[str] = None, change_weather: bool = True, weather: Optional[str] = "random", 
                hide_objects: bool = False, hidden_objects_list: List[str] = []) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        This function resets the environment and starts a new episode

        :param bool change_map: A flag that indicates if the simulator should change its map
        :param str map: (Optional) The name of the map that carla will load
        :param bool change_weather: A flag that indicates if the simulator should change its weather
        :param str weather: A string with the name of the weather that carla will load. If "random", a random weather will be generated. If None, a weather will be chosen from 
                            the list at the beginning of the script
        :param bool hide_objects: A flag that indicates if the simulator should hide some objects
        :param List[str] hidden_objects_list: A list with the type of objects that can be hidden
        :return: A tuple with:
            - data_dict_list: One dictionary for each agent with the data from each sensor. The key to access the data of each sensor will be the id of that senor (except cameras)
            - info_list: One dictionary for each agent with any data that you could ever wish for 
        """
        if self.change_map_counter >= NUM_EPISODES_FOR_MAP_CHANGE  or self.first_episode or change_map:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            self.world.apply_settings(settings)
            if self.first_episode:
                self.first_episode = False
                self.client.reload_world()
            else:
                self._cleanup()
                #self.lane_sensor.destroy()
                #self.collision_sensor.destroy()
                self.change_map_counter = 0

            self.spawn_index = 0

            if map_name == None:
                map_name = MAPS[self.map_idx]
                self.map_idx += 1
                self.map_idx %= len(MAPS)
            
            self._load_and_wait_for_world(map_name)

            self._prepare_ego_vehicles(respawn=True)

            self._setup_sensors()
          
            #self.lane_sensor = LaneInvasionSensor(self.ego_vehicles[0])
            #self.collision_sensor = CollisionSensor(self.ego_vehicles[0])
        else:
            #self._tick()

            self._prepare_ego_vehicles(respawn=False)

            #self.lane_sensor.reset()
            #self.collision_sensor.reset()

        if change_weather:
            self.next_weather(next_weather = weather)

        if hide_objects:
            self.hide_objects(hidden_objects_list = hidden_objects_list)

        self._tick()

        settings = self.world.get_settings()
        if True: #self._args.sync:  
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 1.0 / self.frame_rate
        self.world.apply_settings(settings)

        self._tick()

        spectator = self.world.get_spectator()
        transform = self.agent_list[0].get_actor().get_transform()
        spectator.set_transform(carla.Transform(transform.location + carla.Location(z=50), carla.Rotation(pitch=-90)))

        self.change_map_counter += 1

        self._tick()

        self.get_ct_angle_dif_instruction()

        info_list = []
        data_dict_list = []
        for agent in self.agent_list:
            
            data_dict = agent.get_sensor_interface().get_data(self.frame)

            agent.add_speed_meassurement(agent.get_speed_sensor()._get_forward_speed())

            data_dict_list.append(data_dict)

            info = {"cte":agent.get_cte(),
                    "angle_diff":agent.get_angle_diff(),
                    "throttle": agent.get_last_throttle(),
                    "steering": agent.get_last_steering(),
                    "noisy_throttle": agent.get_last_noisy_throttle(),
                    "noisy_steering": agent.get_last_noisy_steering(),
                    "instruction": agent.get_next_instruction(),
                    "instruction_distance": agent.get_next_instruction_distance(),
                    "junction_distance": agent.get_junction_distance(),
                    "reward":self.calc_reward_agent(agent),
                    "is_game_over":int(self.is_game_over_agent(agent)),
                    "success":agent.get_success(),
                    "timestamp": self.frame, #GameTime.GameTime.get_time()
                    "CEnvAgent": agent
                    }
            
            info_list.append(info)

            agent.set_data_dict(data_dict)
            agent.set_info(info)
        

        return data_dict_list, info_list

        
    # Render the environment to the screen
    def render(self, idx: int = 0, mode: str ='rgb', image_transforms: List[Callable[[np.ndarray], np.ndarray]] = []) -> None:
        """
        This function renders the data from the cameras with opencv

        :param int idx: idx of the agent to be rendered
        :param str mode: type of camera that is wanted to be rendered. If "all" is specified, every type will be rendered and stacked
        :param image_trnasforms: A list of functions to modify the image of each camera mode to be rendered
            The params and return of the function will be:
                :param input_image: a numpy array with the input image
                :return: The image modified by the function
        """
        img = None
        image_transform = None
        if len(image_transforms) == 1:
            image_transform = image_transforms[0]
        if mode == "all":
            for type_idx, camera_type in enumerate(self.camera_types):
                if len(image_transforms) == len(self.camera_types):
                    image_transform = image_transforms[type_idx]
                if type_idx == 0:
                    img = self._concatenate_sensors(idx = idx, mode = camera_type, image_transform = image_transform)
                else:
                    aux_img = self._concatenate_sensors(idx = idx, mode = camera_type, image_transform = image_transform)
                    img = np.concatenate((img, aux_img), axis = 0)
        else:
            img = self._concatenate_sensors(idx = idx, mode = mode, image_transform = image_transform)

        cv2.imshow("frame", img)
        cv2.waitKey(1)
    
    # Close the environment
    # 1- Destroy actors
    # 2- Close connection
    def close(self) -> None:
        """
        A function that closes the environment cleanly
        """
        self._cleanup()

    def save_current_step(self, save_state: bool = True, save_info: bool = True) -> None:
        """
        A function that saves the information a the current step

        :param bool save_state: A flag that indicates if the state (sensors) of each agent must be saved
        :param bool save_info: A flag that indicates if the info dict of each agent must be saved
        """
        for idx, agent in enumerate(self.agent_list):
            agent_path = os.path.realpath(os.path.dirname(__file__)) + "\\..\\log\\" + agent.get_name()
            state = agent.get_data_dict()
            info = agent.get_info()

            if save_state:
                state_string = ""
                sensor_names = []

                for sensor in agent.get_sensor_description_list():
                    if sensor["type"].startswith("sensor.camera."):
                        for camera_type in self.camera_types:
                            data = state[sensor["id"] + "_" + camera_type][1][:,:,0:3]
                            cv2.imwrite(agent_path + "\\" + sensor["id"] + "_" + camera_type + "_" + str(agent.get_current_idx()).zfill(7) + ".png", data)
                    elif sensor["type"] == 'sensor.speedometer':
                        if state_string != "":
                            state_string += ","
                        state_string += str(state[sensor["id"]][1]["speed"])
                        sensor_names.append(sensor["type"].split(".")[-1])
                    elif sensor["type"] == 'sensor.other.imu' or sensor["type"] == "sensor.other.gnss":
                        for data in state[sensor["id"]][1]:
                            if state_string != "":
                                state_string += ","
                            state_string += str(data)
                        sensor_names.append(sensor["type"].split(".")[-1])
                file_name = ""
                for sensor_name in sensor_names:
                    if file_name != "":
                        file_name += "_"
                    file_name += sensor_name
                file = open(agent_path + "\\" + file_name + "_" + str(agent.get_current_idx()).zfill(7) + ".txt", "w")
                file.write(state_string)
                file.close()
            if save_info:

                info_string = ""
                for key in info:
                    if key != "CEnvAgent":
                        if info_string != "":
                            info_string += ","
                        info_string += str(info[key])
                        file_name += sensor_name

                file = open(agent_path + "\\" + "info_" + str(agent.get_current_idx()).zfill(7) + ".txt", "w")
                file.write(info_string)
                file.close()

            agent.set_current_idx(agent.get_current_idx() + 1)

    def _concatenate_sensors(self, idx: int = 0, mode: str ='rgb', image_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None ) -> np.ndarray:
        img = None
        for sensor_idx, sensor_description in enumerate(self.agent_list[idx].get_sensor_description_list()):
            if sensor_description["type"].startswith("sensor.camera."):
                id = sensor_description["id"]
                if sensor_idx == 0:
                    img = self.agent_list[idx].get_data_dict()[id + "_" + mode][1][:,:,0:3]                   
                    if image_transform != None:
                        img = image_transform(img)
                else:
                    aux_img = self.agent_list[idx].get_data_dict()[id + "_" + mode][1][:,:,0:3]
                    
                    if image_transform != None:
                        aux_img = image_transform(aux_img)
                    img = np.concatenate((img, aux_img), axis = 1)
        
        return img

    def _tick(self) -> None:
        if self.world.get_settings().synchronous_mode:
            self.world.tick()
        else:
            self.world.wait_for_tick()
        snapshot = self.world.get_snapshot()
        timestamp = snapshot.timestamp
        self.frame = timestamp.frame
        GameTime.on_carla_tick(timestamp)

    def _load_and_wait_for_world(self, town: str) -> None:
            """
            Load a new CARLA world and provide data to CarlaDataProvider
            """

            self.world = self.client.load_world(town)
            self.world = self.client.get_world()

            # Wait for the world to be ready
            self._tick()     


    def _prepare_ego_vehicles(self, respawn: bool = True) -> None:
        """
        Spawn or update the ego vehicles
        """

        for vehicle_idx, agent in enumerate(self.agent_list):
            
            vehicle_name = agent.get_vehicle_description()[0]
            vehicle_attributes = agent.get_vehicle_description()[1]
            if respawn:
                actor_bp = self.world.get_blueprint_library().find(vehicle_name)
                for attribute, value in vehicle_attributes.items():
                    actor_bp.set_attribute(attribute, str(value))

                actor = None
                while actor == None:
                    spawn_point = self.world.get_map().get_spawn_points()[self.spawn_index]
                    _spawn_point = carla.Transform(carla.Location(), spawn_point.rotation)
                    _spawn_point.location.x = spawn_point.location.x
                    _spawn_point.location.y = spawn_point.location.y
                    _spawn_point.location.z = spawn_point.location.z + 0.2
                    self.spawn_index = (self.spawn_index + 1) % len(self.world.get_map().get_spawn_points())
                    actor = self.world.try_spawn_actor(actor_bp, _spawn_point)
                    if actor is not None:
                        self.agent_list[vehicle_idx].set_actor(actor)
                        self.agent_list[vehicle_idx].get_actor().set_light_state(carla.VehicleLightState.HighBeam)
                    else:
                        print("Error: Couldn't spawn actor")
            else:        
                self.agent_list[vehicle_idx].get_actor().set_target_velocity(carla.Vector3D())
                self.agent_list[vehicle_idx].get_actor().set_target_angular_velocity(carla.Vector3D())
                
                self.take_single_action(control_zero, vehicle_idx)
                
                spawn_point = self.world.get_map().get_spawn_points()[self.spawn_index]
                self.spawn_index = (self.spawn_index + 1) % len(self.world.get_map().get_spawn_points())
                self.agent_list[vehicle_idx].reset_variables()
            
            #UN TICK AQUI????
            #self._tick()

            destination_point = random.choice(self.world.get_map().get_spawn_points()) if self.world.get_map().get_spawn_points() else carla.Transform()
            i = 0
            while spawn_point == destination_point:
                i += 1
                if i == 50000:
                    print("Error: Stuck trying to decide spawn point")
                destination_point = random.choice(self.world.get_map().get_spawn_points()) if self.world.get_map().get_spawn_points() else carla.Transform()
            route = self.create_route(spawn_point.location, destination_point.location)

            idx = 0
            while (route[idx][1] == RoadOption.CHANGELANELEFT or route[idx][1] == RoadOption.CHANGELANERIGHT) and (idx < len(route)):
                idx += 1
            if idx == len(route):
                print("PERO POR QUE, POR QUEEEE!!!")

            if route[idx][0].transform.location.distance(self.agent_list[vehicle_idx].get_actor().get_location()) >= MAX_CTE_ERROR or idx > 0:
                rt = route[idx][0].transform
                new_route_transform = carla.Transform(carla.Location(), rt.rotation)
                new_route_transform.location.x = rt.location.x
                new_route_transform.location.y = rt.location.y
                new_route_transform.location.z = rt.location.z + 0.2
                self.agent_list[vehicle_idx].get_actor().set_transform(new_route_transform)
            self.agent_list[vehicle_idx].set_route(route)

        # sync state
        self._tick()

    def _cleanup(self) -> None:
        """
        Remove and destroy all actors
        """
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        self.world.apply_settings(settings)

        self._grp = None

        #self.speed_sensor.destroy()
        #self.speed_sensor = None

        for agent_idx, _ in enumerate(self.agent_list):
            self.agent_list[agent_idx].clear()

    def _setup_sensors(self) -> None:

        for agent in self.agent_list:
            agent.set_sensor_interface(SensorInterface())
            
            for sensor_description in agent.get_sensor_description_list():
                transform = carla.Transform()
                if "x" in sensor_description:
                    if "roll" in sensor_description:
                        transform = carla.Transform(carla.Location(x=sensor_description['x'], y=sensor_description['y'], z=sensor_description['z']), carla.Rotation(roll=sensor_description['roll'], pitch=sensor_description['pitch'], yaw=sensor_description['yaw']))
                    else:
                        transform = carla.Transform(carla.Location(x=sensor_description['x'], y=sensor_description['y'], z=sensor_description['z']))

                if sensor_description["type"] == "sensor.camera.rgb":
                    for image_type in self.camera_types:
                        cam_bp = self.world.get_blueprint_library().find('sensor.camera.' + image_type)
                        cam_bp.set_attribute('image_size_x', str(sensor_description['width']))
                        cam_bp.set_attribute('image_size_y', str(sensor_description['height']))
                        cam_bp.set_attribute('fov', str(sensor_description['fov']))                         
                        sensor = self.world.spawn_actor(cam_bp, transform, attach_to=agent.get_actor(), attachment_type=carla.AttachmentType.Rigid)
                        sensor.listen(CallBack(sensor_description['id'] + '_' + image_type, 'sensor.camera.' + image_type, sensor, agent.get_sensor_interface()))

                        ### REVISAR ###
                        agent.append_sensor(sensor)
                elif sensor_description["type"] == "sensor.opendrive_map":
                    pass
                elif sensor_description["type"] == "sensor.speedometer":
                    speed_sensor = SpeedometerReader(agent.get_actor(), reading_frequency = self.frame_rate)
                    speed_sensor.listen(CallBack(sensor_description['id'], 'sensor.speedometer', speed_sensor, agent.get_sensor_interface()))
                    agent.set_speed_sensor(speed_sensor)
                elif sensor_description["type"] == "sensor.other.collision":
                    collision_sensor = CollisionSensor(agent.get_actor(), reading_frequency = self.frame_rate)
                    collision_sensor.listen(ModCallBack(sensor_description['id'], 'sensor.other.collision', collision_sensor, agent.get_sensor_interface()))
                    agent.set_collision_sensor(collision_sensor)
                else:
                    sensor_bp = self.world.get_blueprint_library().find(sensor_description["type"])
                    if sensor_description["type"] == "sensor.other.radar":
                        sensor_bp.set_attribute('fov', str(sensor_description['fov']))                         
                    sensor = self.world.spawn_actor(sensor_bp, transform, attach_to=agent.get_actor(), attachment_type=carla.AttachmentType.Rigid)
                    sensor.listen(CallBack(sensor_description['id'], sensor_description["type"], sensor, agent.get_sensor_interface()))
                    
                    ### REVISAR ###
                    agent.append_sensor(sensor)
            if agent.get_speed_sensor() is None:
                speed_sensor = SpeedometerReader(agent.get_actor(), reading_frequency = self.frame_rate)
                speed_sensor.listen(CallBack("Speed", 'sensor.speedometer', speed_sensor, agent.get_sensor_interface()))
                agent.set_speed_sensor(speed_sensor)
            if agent.get_collision_sensor() is None:
                collision_sensor = CollisionSensor(agent.get_actor(), reading_frequency = self.frame_rate)
                collision_sensor.listen(ModCallBack("Collision", 'sensor.other.collision', collision_sensor, agent.get_sensor_interface()))
                agent.set_collision_sensor(collision_sensor)
                
        self._tick()


    def get_ct_angle_dif_instruction(self) -> None:

        for agent in self.agent_list:

            transform = agent.get_actor().get_transform()

            location = transform.location
            rotation = transform.rotation.yaw

            distance = 9999999
            close_w = None

            for idx, (w, ins) in enumerate(agent.get_route()):
                d = w.transform.location.distance(location)

                if d < distance:
                    close_w = w
                    close_idx = idx
                distance = min(distance, d)

            w_rotation = close_w.transform.rotation.yaw
            w_location = close_w.transform.location

            angle_diff = rotation - w_rotation

            relative_position = (location.x - w_location.x, location.y - w_location.y)
            distance = math.sqrt(relative_position[0]**2 + relative_position[1]**2)

            relative_angle_rad = math.atan2(relative_position[1], relative_position[0])

            relative_angle_diff_rad = relative_angle_rad - math.radians(w_rotation)

            cte = math.sin(relative_angle_diff_rad) * distance  

            while angle_diff > 180:
                angle_diff -= 360
            while angle_diff < -180:
                angle_diff += 360


            angle_diff /= 180

            next_instruction = RoadOption.LANEFOLLOW
            next_instruction_distance = 0.0
            while next_instruction == RoadOption.LANEFOLLOW:
                for i in range(close_idx, len(agent.get_route())):
                    if agent.get_route()[i][1] != RoadOption.LANEFOLLOW:
                        if agent.get_route()[i][1] == RoadOption.CHANGELANELEFT or agent.get_route()[i][1] == RoadOption.CHANGELANERIGHT:
                            print(agent.get_route()[i][1])
                        else:
                            next_instruction = ROAD_OPTIONS.index(agent.get_route()[i][1])
                            next_instruction_distance = agent.get_actor().get_location().distance(agent.get_route()[i][0].transform.location)
                            break
                if next_instruction == RoadOption.LANEFOLLOW:
                    destination_point = random.choice(self.world.get_map().get_spawn_points()) if self.world.get_map().get_spawn_points() else carla.Transform()
                    i = 0
                    while agent.get_route()[-1][0] == destination_point:
                        i += 1
                        if i == 50000:
                            print("Error: Stuck trying to decide spawn point")
                        destination_point = random.choice(self.world.get_map().get_spawn_points()) if self.world.get_map().get_spawn_points() else carla.Transform()
                    route = self.create_route(agent.get_route()[-1][0].transform.location, destination_point.location)
                    agent.set_route(agent.get_route() + route[1:])
                    print("bien")

            junction_distance = 0.0
            for i in range(close_idx, len(agent.get_route())):
                if agent.get_route()[i][0].is_junction:
                    junction_distance = agent.get_actor().get_location().distance(agent.get_route()[i][0].transform.location)
                    break
            
            prev_idx = 0
            if close_idx > 0:
                prev_idx = close_idx - 1
            agent.set_route(agent.get_route()[prev_idx:])

            success = 0
            #if close_idx == len(agent.get_route()) - 1:
            #    success = 1

            agent.set_cte(cte)
            
            agent.set_angle_diff(angle_diff)
            
            agent.set_junction_distance(junction_distance)
            
            agent.set_next_instruction(next_instruction)
            
            agent.set_next_instruction_distance(next_instruction_distance)
            
            agent.set_success(success)
            
            

            '''
            print("\n")
            print("Trans: ", transform)
            print("W_Trans: ", close_w.transform)
            print("relative_position: ", relative_position)
            print("distance: ", distance)
            print("relative_angle: ", math.degrees(relative_angle_rad))
            print("relative_angle_diff: ", math.degrees(relative_angle_diff_rad))
            print("cte: ", cte)
            '''

    def take_single_action(self, control: carla.VehicleControl, idx: int, input_noise: bool = False) -> None:
        
        self.agent_list[idx].set_last_throttle(control.throttle)
        self.agent_list[idx].set_last_steering(control.steer)

        if input_noise:
            control.steer, control.throttle = self.agent_list[idx].get_input_noise_function()(control.steer, control.throttle, self.agent_list[idx].get_info())

        self.agent_list[idx].set_last_noisy_throttle(control.throttle)
        self.agent_list[idx].set_last_noisy_steering(control.steer)
        
        self.agent_list[idx].get_actor().apply_control(control)

    def take_action(self, control_list: List[carla.VehicleControl], input_noise: bool = False) -> None:
        for idx, control in enumerate(control_list):
            self.take_single_action(control, idx, input_noise)

    def calc_reward_agent(self, agent: CEnvAgent) -> float:
        if agent.get_collision_sensor().__call__() > 0:
            #print("Crash: ", REWARD_CRASH - agent.get_collision_sensor().__call__() * CRASH_SPEED_WEIGHT)
            return REWARD_CRASH - agent.get_collision_sensor().__call__() * CRASH_SPEED_WEIGHT
            #if self.world_frame - self.collision_history[-1][0] < 10:
                #print("Crash: ", REWARD_CRASH - self.collision_history[-1][1] * CRASH_SPEED_WEIGHT)
                #return REWARD_CRASH - self.collision_history[-1][1] * CRASH_SPEED_WEIGHT
        #    reward_list.append(REWARD_CRASH - agent.get_speed() * CRASH_SPEED_WEIGHT)
        #    continue
        #if self.controller.crash:
            #print("reward crash")
        #    return REWARD_CRASH - self.speed * CRASH_SPEED_WEIGHT

        #if self.lane_crossed:
        #    return REWARD_CRASH - self.speed * CRASH_SPEED_WEIGHT
        if all([s < MIN_SPEED for s in agent.get_speed_array()]):
            print("Too Slow")
            return 0

        #print("Reward: ", BASE_REWARD + THROTTLE_REWARD_WEIGHT * self.last_throttle)
        if abs(agent.get_cte()) > MAX_CTE_ERROR:
            
            transform = agent.get_actor().get_transform()

            location = transform.location

            cte = 9999999
            close_w = None

            for w, _ in agent.get_route():
                d = w.transform.location.distance(location)

                if d < cte:
                    close_w = w
                cte = min(cte, d)

            #print("CAR: ", transform)
            #print("WP: ", close_w.transform)
            #print("Instruction: ", agent.get_close_ins())

            return REWARD_CRASH - agent.get_speed() * CRASH_SPEED_WEIGHT


        #return BASE_REWARD + THROTTLE_REWARD_WEIGHT * self.last_throttle

        return agent.get_last_throttle() + (1 - abs(agent.get_cte()) / MAX_CTE_ERROR) + (1 - abs(agent.get_angle_diff()))

    def calc_reward(self) -> List[float]:
        #print(self.controller.crash)
        reward_list = []
        for agent in self.agent_list:
            reward_list.append(self.calc_reward_agent(agent))

        return reward_list

    def is_game_over_agent(self, agent: CEnvAgent) -> bool:
        if agent.get_collision_sensor().__call__() > 0:
            print("Collision")
            return True

        if all([s < MIN_SPEED for s in agent.get_speed_array()]):
            print("Too Slow")
            return True

        margin = 0
        if agent.get_junction_distance() < self.route_resolution + 2:
            margin = 2
        if abs(agent.get_cte()) > MAX_CTE_ERROR + margin:
            print("CTE")
            return True

        return False

    def is_game_over(self) -> List[bool]:
        game_over_list = []
        for agent in self.agent_list:
            game_over_list.append(self.is_game_over_agent(agent))
        return game_over_list

    def next_weather(self, next_weather: Optional[str] = "random", reverse: bool = False) -> None:
        """Get next weather setting"""
        '''

        '''
        parameters = None
        if next_weather is None:
            weather = ""
            n = random.random()
            if n < 0.1:
                weather = random.choice(DEFAULT)
            elif n < 0.37:
                weather = random.choice(CLEAR)
            elif n < 0.55:
                weather = random.choice(CLOUD)
            elif n < 0.68:
                weather = random.choice(WET)
            elif n < 0.84:
                weather = random.choice(CLOUDWET)
            else:
                weather = random.choice(STORM)
            parameters = getattr(carla.WeatherParameters, weather)
        elif next_weather != "random":
            weather = next_weather
            parameters = getattr(carla.WeatherParameters, weather)
        else: 
            parameters = carla.WeatherParameters()
            parameters.sun_azimuth_angle = np.random.uniform(0,360)
            parameters.sun_altitude_angle = np.random.uniform(-90,90)
            parameters.wind_intensity = np.random.uniform(0,100)
            parameters.mie_scattering_scale = max(0, np.random.normal(0.03, 0.01))
            parameters.rayleigh_scattering_scale = max(0, np.random.normal(parameters.rayleigh_scattering_scale, parameters.rayleigh_scattering_scale * 0.3))
            parameters.cloudiness = np.random.uniform(0,100)
            parameters.precipitation_deposits = np.random.uniform(0,100)
            #parameters.wetness 
            n = random.random()
            if n < 0.25:
                """clear"""
                pass
            elif n < 0.5:
                """rain"""
                parameters.precipitation = np.random.uniform(0,100)
            elif n < 0.75:
                """fog"""
                parameters.fog_density = np.random.uniform(0,100)
                parameters.fog_distance = max(0, np.random.normal(300, 100))
                parameters.fog_falloff = np.random.uniform(0,10)
                parameters.scattering_intensity = max(np.random.normal(0.03, 0.01), 0)
            else:
                """sandstorm"""
                parameters.precipitation_deposits = 0
                parameters.dust_storm = np.random.uniform(0,100)

        self.world.set_weather(parameters)

    def hide_objects(self, hidden_objects_list: List[str] = [], base_hide_probability: float = 0.6):
        """
        Hides a random set of objects of the categories provided
        """
        
        hide_probability = max(0.25, min(1, np.random.normal(base_hide_probability, 0.2)))

        for object_type in hidden_objects_list:
            object_list = self.world.get_environment_objects(object_type)
            for obj in object_list:
                self.world.enable_environment_objects([obj.id], random.random() < hide_probability)

    def create_route(self, origin: carla.Location, destination: carla.Location) -> List[Tuple[carla.Waypoint, RoadOption]]:
        """
        This method creates a list of waypoints from agent's position to destination location
        based on the route returned by the global router
        """
        start_waypoint = self.world.get_map().get_waypoint(origin)
        end_waypoint = self.world.get_map().get_waypoint(destination)
        route_trace = self._trace_route(start_waypoint, end_waypoint)
        return route_trace

    def _trace_route(self, start_waypoint: carla.Waypoint, end_waypoint: carla.Waypoint) -> List[Tuple[carla.Waypoint, RoadOption]]:
        """
        This method sets up a global router and returns the optimal route
        from start_waypoint to end_waypoint
        """

        # Setting up global router
        #if self._grp is None:
        if not isinstance(self._grp, GlobalRoutePlanner):
            grp = GlobalRoutePlanner(self.world.get_map(), self.route_resolution)
            self._grp = grp

        # Obtain route plan
        route = self._grp.trace_route(
            start_waypoint.transform.location,
            end_waypoint.transform.location)

        return route
