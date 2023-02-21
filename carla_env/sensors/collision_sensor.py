from leaderboard.envs.sensor_interface import CallBack, GenericMeasurement, BaseReader
import carla
import math
import logging

class ModCallBack(CallBack):

    def __call__(self, data):
        if isinstance(data, carla.libcarla.Image):
            self._parse_image_cb(data, self._tag)
        elif isinstance(data, carla.libcarla.LidarMeasurement):
            self._parse_lidar_cb(data, self._tag)
        elif isinstance(data, carla.libcarla.RadarMeasurement):
            self._parse_radar_cb(data, self._tag)
        elif isinstance(data, carla.libcarla.GnssMeasurement):
            self._parse_gnss_cb(data, self._tag)
        elif isinstance(data, carla.libcarla.IMUMeasurement):
            self._parse_imu_cb(data, self._tag)
        elif isinstance(data, carla.libcarla.CollisionEvent):
            self._parse_collision_cb(data, self._tag)
        elif isinstance(data, GenericMeasurement):
            self._parse_pseudosensor(data, self._tag)
        else:
            logging.error('No callback method for this sensor.')

    # Parsing CARLA physical Sensors
    def _parse_collision_cb(self, collision, tag):
        self._data_provider.update_sensor(tag, collision, collision.frame)

class CollisionSensor(BaseReader):
    """ Class for collision sensors"""

    def __init__(self, vehicle, reading_frequency=1.0):
        """Constructor method"""
        self.history = []
        self.collided_actors = {}
        world = vehicle.get_world()
        blueprint = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(blueprint, carla.Transform(), attach_to=vehicle)
        # We need to pass the lambda a weak reference to
        # self to avoid circular references

        self.sensor.listen(lambda event: CollisionSensor._on_collision(event))
        super(CollisionSensor, self).__init__(vehicle, reading_frequency)

    def __call__(self):
        """ We convert the vehicle physics information into a convenient dictionary """

        # protect this access against timeout
        max_collision = 0
        for h in self.history:
            max_collision = max(max_collision, h[1])
        return max_collision

    @staticmethod
    def _on_collision(self, event):
        """On collision method"""
        self.collided_actors.add(event.other_actor)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 60:
            self.history.pop(0)

    def stop(self):
        self._run_ps = False
        self.sensor.stop()

    def destroy(self):
        self._run_ps = False
        self.sensor.stop()
        self.sensor.destroy()
        self.history = None
        self.collided_actors = None