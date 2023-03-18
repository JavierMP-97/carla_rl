import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass
from carla_agents.conditional_imitation_learning.model import create_model

from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track
from agents.navigation.controller import PIDLateralController, PIDLongitudinalController
import carla
from collections import deque
from config import MAX_STEERING_DIFF
import cv2

SIDE_IMAGES = True
MODEL_NAME = "mobilenetv3"

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

def get_entry_point():
    return 'CILAgent'

class CILAgent(AutonomousAgent):
    def __init__(self, carla_host, carla_port, debug=False, target_speed = 5):
        super(CILAgent, self).__init__(carla_host, carla_port, debug)
        self.long_pid = ModPIDLongitudinalController()
        self.lat_pid = ModPIDLateralController()
        self.past_steering = 0
        self.max_brake = 0.3
        self.max_throt = 0.75
        self.max_steer = 1 #0.8
        self.target_speed = target_speed
        self.model, _, _ = create_model(side_images = SIDE_IMAGES, model_name = MODEL_NAME, weights_path="D:/PC-Javier/Desktop/Carla14/carla_rl/carla_agents/conditional_imitation_learning/logs/MN3-freeze1-lr-4-BIG_exploding-gradients/best_weights.hdf5")


    def setup(self, path_to_conf_file):
        """
        Initialize everything needed by your agent and set the track attribute to the right type:
            Track.SENSORS : CAMERAS, LIDAR, RADAR, GPS and IMU sensors are allowed
            Track.MAP : OpenDRIVE map is also allowed
        """
        self.track = Track.SENSORS
        
    def sensors(self):  # pylint: disable=no-self-use
        sensors = [ {'type': 'sensor.camera.rgb', 'x': 1.2, 'y': -0.25, 'z': 1.3, 'roll': 0.0, 'pitch': -5.0, 'yaw': -75.0, 'width': 256, 'height': 128, 'fov': 90, 'id': 'left'},
                    {'type': 'sensor.camera.rgb', 'x': 1.3, 'y': 0.0, 'z': 1.3, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0, 'width': 256, 'height': 128, 'fov': 90, 'id': 'middle'},
                    {'type': 'sensor.camera.rgb', 'x': 1.2, 'y': 0.25, 'z': 1.3, 'roll': 0.0, 'pitch': -5.0, 'yaw': 75.0, 'width': 256, 'height': 128, 'fov': 90, 'id': 'right'},
                    {'type': 'sensor.speedometer', 'id': 'Speed'}]

        return sensors

    def run_step(self, input_data, timestamp, info_dict = None):
        """
        Execute one step of navigation.
        :return: control
        """
        images = []
        input_image = None
        if SIDE_IMAGES:
            images.append(cv2.cvtColor(input_data["left_rgb"][1][:,:,0:3], cv2.COLOR_BGR2RGB))
            images.append(cv2.cvtColor(input_data["middle_rgb"][1][:,:,0:3], cv2.COLOR_BGR2RGB))
            images.append(cv2.cvtColor(input_data["right_rgb"][1][:,:,0:3], cv2.COLOR_BGR2RGB))
        else:
            images.append(cv2.cvtColor(input_data["middle_rgb"][1][:,:,0:3], cv2.COLOR_BGR2RGB))

        for im_idx, im in enumerate(images):
            im = tf.cast(im, tf.float32)
            images[im_idx] = tf.keras.applications.mobilenet_v3.preprocess_input(im)
        if SIDE_IMAGES:
            input_image = tf.expand_dims(tf.concat([images[0], images[1], images[2]], 0), axis=0)
        else:
            input_image = tf.expand_dims(images[0], axis=0)
        
        input_command = tf.reshape(tf.one_hot(info_dict["instruction"], 3), [1,1,1,3])

        input_dict = {"image":input_image, "command": input_command}

        current_steering = float(self.model(input_dict).numpy()[0][0])
        print(current_steering)

        acceleration = self.long_pid.run_step(self.target_speed, info_dict["CEnvAgent"].get_speed())
        control = carla.VehicleControl()

        if acceleration >= 0.0:
            control.throttle = min(acceleration, self.max_throt)
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = min(abs(acceleration), self.max_brake)

        # Steering regulation: changes cannot happen abruptly, can't steer too much.

        if current_steering > self.past_steering + MAX_STEERING_DIFF:
            current_steering = self.past_steering + MAX_STEERING_DIFF
        elif current_steering < self.past_steering - MAX_STEERING_DIFF:
            current_steering = self.past_steering - MAX_STEERING_DIFF

        if current_steering >= 0:
            steering = min(self.max_steer, current_steering)
        else:
            steering = max(-self.max_steer, current_steering)

        control.steer = steering
        control.hand_brake = False
        control.manual_gear_shift = False
        self.past_steering = steering

        return control
        

    def destroy(self):
        """
        Destroy (clean-up) the agent
        :return:
        """
        pass



