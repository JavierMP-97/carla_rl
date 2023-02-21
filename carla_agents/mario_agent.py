from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track
from agents.navigation.controller import PIDLateralController, PIDLongitudinalController
from agents.navigation.local_planner import RoadOption
import carla
import time
from collections import deque
from config import MAX_STEERING_DIFF
import cv2
import tensorflow as tf
import os
import numpy as np

ROAD_OPTIONS = [RoadOption.STRAIGHT, RoadOption.LEFT, RoadOption.RIGHT]

class ModPIDLongitudinalController(PIDLongitudinalController):
    def __init__(self, K_P=1.0, K_I=0.05, K_D=0.0, dt=1.0 / 20.0):
        """
        Constructor method.

            :param vehicle: actor to apply to local planner logic onto
            :param K_P: Proportional term
            :param K_D: Differential term
            :param K_I: Integral term
            :param dt: time differential in seconds
        """
        self._k_p = K_P
        self._k_i = K_I
        self._k_d = K_D
        self._dt = dt
        self._error_buffer = deque(maxlen=10) 
    def run_step(self, target_speed, current_speed, debug=False):
        """
        Execute one step of longitudinal control to reach a given target speed.

            :param target_speed: target speed in Km/h
            :param debug: boolean for debugging
            :return: throttle control
        """
        if debug:
            print('Current speed = {}'.format(current_speed))

        return self._pid_control(target_speed, current_speed)

class ModPIDLateralController(PIDLateralController):
    def __init__(self, offset=0, K_P=1.95, K_I=0.05, K_D=0.2, dt=1.0 / 20.0):
        """
        Constructor method.

            :param vehicle: actor to apply to local planner logic onto
            :param offset: distance to the center line. If might cause issues if the value
                is large enough to make the vehicle invade other lanes.
            :param K_P: Proportional term
            :param K_D: Differential term
            :param K_I: Integral term
            :param dt: time differential in seconds
        """
        self._k_p = K_P
        self._k_i = K_I
        self._k_d = K_D
        self._dt = dt
        self._offset = offset
        self._e_buffer = deque(maxlen=10)

    def run_step(self, waypoint, current_transform):
        """
        Execute one step of lateral control to steer
        the vehicle towards a certain waypoin.

            :param waypoint: target waypoint
            :return: steering control in the range [-1, 1] where:
            -1 maximum steering to left
            +1 maximum steering to right
        """
        return self._pid_control(waypoint, current_transform)

class MarioAgent(AutonomousAgent):
    def __init__(self, carla_host, carla_port, debug=False):
        super().__init__(carla_host, carla_port, debug)
        self.long_pid = ModPIDLongitudinalController()
        self.lat_pid = ModPIDLateralController()
        self.past_steering = 0
        self.max_brake = 0.3
        self.max_throt = 0.75
        self.max_steer = 0.8
        self.model = tf.keras.models.load_model(os.path.realpath(os.path.dirname(__file__)) + "\\models\\bestmodel.h5")


    def setup(self, path_to_conf_file):
        """
        Initialize everything needed by your agent and set the track attribute to the right type:
            Track.SENSORS : CAMERAS, LIDAR, RADAR, GPS and IMU sensors are allowed
            Track.MAP : OpenDRIVE map is also allowed
        """
        self.track = Track.SENSORS
        
    def sensors(self):  # pylint: disable=no-self-use
        sensors = [ {'type': 'sensor.camera.rgb', 'x': 1.2, 'y': 0.0, 'z': 1.2, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0, 'width': 512, 'height': 384, 'fov': 90, 'id': 'middle'},
                    {'type': 'sensor.speedometer', 'id': 'Speed'}]

        return sensors

    def run_step(self, input_data, timestamp, info_dict = None):
        """
        Execute one step of navigation.
        :return: control
        """
        t = time.time()
        min_cte=-2.254151817
        max_cte=3.047694543
        dif_cte = max_cte-min_cte
        min_angle=-0.769588314
        max_angle=0.731862545
        dif_angle = max_angle-min_angle
        min_steer=-0.800000012
        max_steer=0.800000012
        dif_steer= max_steer- min_steer

        acceleration = self.long_pid.run_step(20, info_dict["CEnvAgent"].get_speed())

        control = carla.VehicleControl()
        if acceleration >= 0.0:
            control.throttle = min(acceleration, self.max_throt)
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = min(abs(acceleration), self.max_brake)

        imagen_list=[]

        #procesamos la imegen para poder pasarla a la red
        imagen = input_data["middle_rgb"][1]
        image = imagen[:, :, :3]
        #Esta línea solo si se entrena con BGR y no RGB
        #img = image[:,:,::-1]
        image = cv2.resize(image,(400,300))
        crop_image = image[0:250,0:400]
        imagen_list.append(crop_image)
        imagen_list= np.array(imagen_list)
        imagen_list= imagen_list.astype("float16") / 255.0

        #Si el coche esta en una zona con un semáforo y esta en rojo devuelve True si no False
        straight = 0
        left = 0
        right = 0

        data_list = []

        #Normalizacion de datos
        cte_normalized = (info_dict["cte"]-min_cte)/dif_cte
        data_list.append(cte_normalized)

        angle_normalized = (info_dict["angle_diff"]-min_angle)/dif_angle
        data_list.append(angle_normalized)

        close_idx = 0
        second_idx = 0
        best_distance = 999999
        second_distance = 999999
        location = info_dict["CEnvAgent"].get_actor().get_transform().location
        for idx, (w, ins) in enumerate(info_dict["CEnvAgent"].get_route()):
                d = w.transform.location.distance(location)
                if d < best_distance:
                    second_idx = close_idx
                    close_idx = idx
                    second_distance = best_distance
                    best_distance = d
                elif d < second_distance:
                    second_idx = idx
                    second_distance = d
        
        new_idx = max(close_idx, second_idx)

        instruction = info_dict["CEnvAgent"].get_route()[new_idx][1]

        if instruction == RoadOption.LEFT:
            left = 1
        elif instruction == RoadOption.STRAIGHT:
            straight = 1
        elif instruction == RoadOption.RIGHT:
            right = 1            

        #añadimos todo a la lista para pasarlo a la red
        data_list.append(straight)
        data_list.append(left)
        data_list.append(right)
        data_list.append(int(info_dict["junction_distance"] < 1))
        data_list=np.array(data_list)
        data_list = np.expand_dims(data_list,axis=0)

        #prediccion del steer
        with tf.device('/cpu:0'):
            steer_normalized = self.model.predict([imagen_list,data_list])

        #desnormalización
        steer= (steer_normalized*dif_steer)+ min_steer

        control.steer= float(steer[0][0])

        return control 

    def destroy(self):
        """
        Destroy (clean-up) the agent
        :return:
        """
        pass