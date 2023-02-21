from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track
from agents.navigation.controller import PIDLateralController, PIDLongitudinalController
import carla
from collections import deque
from config import MAX_STEERING_DIFF

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
    return 'RLAgent'

class AutoAgent(AutonomousAgent):
    def __init__(self, carla_host, carla_port, debug=False):
        super(AutoAgent, self).__init__(carla_host, carla_port, debug)
        self.long_pid = ModPIDLongitudinalController()
        self.lat_pid = ModPIDLateralController()
        self.past_steering = 0
        self.max_brake = 0.3
        self.max_throt = 0.75
        self.max_steer = 0.8

    def setup(self, path_to_conf_file):
        """
        Initialize everything needed by your agent and set the track attribute to the right type:
            Track.SENSORS : CAMERAS, LIDAR, RADAR, GPS and IMU sensors are allowed
            Track.MAP : OpenDRIVE map is also allowed
        """
        self.track = Track.SENSORS
        
    def sensors(self):  # pylint: disable=no-self-use
        sensors = [ {'type': 'sensor.camera.rgb', 'x': 1.2, 'y': -0.25, 'z': 1.3, 'roll': 0.0, 'pitch': 0.0, 'yaw': -45.0, 'width': 254, 'height': 254, 'fov': 90, 'id': 'left'},
                    {'type': 'sensor.camera.rgb', 'x': 1.3, 'y': 0.0, 'z': 1.3, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0, 'width': 254, 'height': 254, 'fov': 90, 'id': 'middle'},
                    {'type': 'sensor.camera.rgb', 'x': 1.2, 'y': 0.25, 'z': 1.3, 'roll': 0.0, 'pitch': 0.0, 'yaw': 45.0, 'width': 254, 'height': 254, 'fov': 90, 'id': 'right'},
                    {'type': 'sensor.speedometer', 'id': 'Speed'}]

        return sensors

    def run_step(self, input_data, timestamp, info_dict = None):
        """
        Execute one step of navigation.
        :return: control
        """
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
        if new_idx < len(info_dict["CEnvAgent"].get_route()) - 1:
            new_idx += 1

        waypoint = info_dict["CEnvAgent"].get_route()[new_idx][0]

        acceleration = self.long_pid.run_step(5, info_dict["CEnvAgent"].get_speed())
        current_steering = self.lat_pid.run_step(waypoint, info_dict["CEnvAgent"].get_actor().get_transform())
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



