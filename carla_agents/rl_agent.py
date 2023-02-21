import sys
import os
import glob
from leaderboard.autoagents.autonomous_agent import AutonomousAgent
import carla

'''
from autonomous_agent import AutonomousAgent
import carla
from data_loader import preprocess_image
from model_new import NewVariationalAutoEncoder
import numpy as np
from replay import CustomReplayBuffer
import tempfile
import tensorflow as tf
from tf_agents.agents.sac import sac_agent
from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.sac import tanh_normal_projection_network
from tf_agents.environments import suite_pybullet
from tf_agents.metrics import py_metrics
from tf_agents.networks import actor_distribution_network
from tf_agents.policies import greedy_policy, py_tf_eager_policy, random_py_policy
from tf_agents.replay_buffers import reverb_replay_buffer, reverb_utils 
from tf_agents.train import actor, learner, triggers
from tf_agents.train.utils import spec_utils, strategy_utils, train_utils
from tf_agents.trajectories import TimeStep, trajectory, StepType
from tf_agents.specs import ArraySpec, BoundedArraySpec
from tf_agents.typing.types import BoundedTensorSpec
from config import CRITIC_LEARNING_RATE, ACTOR_LEARNING_RATE, ALPHA_LEARNING_RATE, TARGET_UPDATE_TAU, \
    TARGET_UPDATE_PERIOD, GAMMA, REWARD_SCALE_FACTOR, REPLAY_BUFFER_CAPACITY, INITIAL_COLLECT_STEPS, \
    RL_BATCH_SIZE, GRADIENT_STEPS, ACTOR_FC_LAYER_PARAMS, CRITIC_JOINT_FC_LAYER_PARAMS, Z_SIZE, \
    VAE_GREEN_MULTIPLIER, VAE_BETA, VAE_KL_TOLERANCE, VAE_KERNEL_SIZE
'''
TRAIN_MODE = True
EVAL_MODE = False

def get_entry_point():
    return 'RLAgent'

class RLAgent(AutonomousAgent):
    def setup(self, path_to_conf_file):

        '''
        self.vae = NewVariationalAutoEncoder(latent_dim=Z_SIZE, green = VAE_GREEN_MULTIPLIER, beta= VAE_BETA, kl_tolerance= VAE_KL_TOLERANCE, kernel_size=VAE_KERNEL_SIZE)
        self.vae.load_weights()

        self.tempdir = tempfile.gettempdir()

        observation_spec = ArraySpec(shape=(128,), dtype=tf.float32, name="observation_spec")
        action_spec = BoundedArraySpec(shape=(2,), dtype=tf.float32, minimum=[], maximum=[], name="action_spec")
        time_step_spec = TimeStep(
        {'discount': BoundedArraySpec(shape=(), dtype=tf.float32, name='discount', minimum=np.array(0., dtype=tf.float32), maximum=np.array(1., dtype=tf.float32)),
        'observation': BoundedArraySpec(shape=(28,), dtype=tf.float32, name='observation', minimum=np.array([],
            dtype=tf.float32), maximum=np.array([], dtype=tf.float32)),
        'reward': tf.TensorSpec(shape=(), dtype=tf.float32, name='reward'),
        'step_type': tf.TensorSpec(shape=(), dtype=tf.int32, name='step_type')})


        use_gpu = True

        strategy = strategy_utils.get_strategy(tpu=False, use_gpu=use_gpu)

        with strategy.scope():
            critic_net = critic_network.CriticNetwork(
                    (observation_spec, action_spec),
                    observation_fc_layer_params=None,
                    action_fc_layer_params=None,
                    joint_fc_layer_params=CRITIC_JOINT_FC_LAYER_PARAMS,
                    kernel_initializer='glorot_uniform',
                    last_kernel_initializer='glorot_uniform')
            actor_net = actor_distribution_network.ActorDistributionNetwork(
                    observation_spec,
                    action_spec,
                    fc_layer_params=ACTOR_FC_LAYER_PARAMS,
                    continuous_projection_net=(
                        tanh_normal_projection_network.TanhNormalProjectionNetwork))

            train_step = train_utils.create_train_step()

            self.rl_agent = sac_agent.SacAgent(
                    time_step_spec,
                    action_spec,
                    actor_network=actor_net,
                    critic_network=critic_net,
                    actor_optimizer=tf.keras.optimizers.Adam(
                        learning_rate=ACTOR_LEARNING_RATE),
                    critic_optimizer=tf.keras.optimizers.Adam(
                        learning_rate=CRITIC_LEARNING_RATE),
                    alpha_optimizer=tf.keras.optimizers.Adam(
                        learning_rate=ALPHA_LEARNING_RATE),
                    target_update_tau=TARGET_UPDATE_TAU,
                    target_update_period=TARGET_UPDATE_PERIOD,
                    td_errors_loss_fn=tf.math.squared_difference,
                    gamma=GAMMA,
                    reward_scale_factor=REWARD_SCALE_FACTOR,
                    train_step_counter=train_step)

            self.rl_agent.initialize()
            tf_eval_policy = self.rl_agent.policy
            self.eval_policy = py_tf_eager_policy.PyTFEagerPolicy(
                tf_eval_policy, use_tf_function=True)

            tf_collect_policy = self.rl_agent.collect_policy
            self.collect_policy = py_tf_eager_policy.PyTFEagerPolicy(
                tf_collect_policy, use_tf_function=True)

            if TRAIN_MODE:

                self.replay_buffer = CustomReplayBuffer(REPLAY_BUFFER_CAPACITY,
                                        env_dict ={"obs": {"shape": Z_SIZE},
                                                    "act": {"shape": 2},
                                                    "rew": {},
                                                    "next_obs": {"shape": Z_SIZE},
                                                    "done": {}})

                saved_model_dir = os.path.join(self.tempdir, learner.POLICY_SAVED_MODEL_DIR)

                # Triggers to save the agent's policy checkpoints.
                learning_triggers = [
                    #triggers.PolicySavedModelTrigger(
                    #    saved_model_dir,
                    #    self.rl_agent,
                    #    train_step,
                    #    interval=policy_save_interval),
                    #triggers.StepPerSecondLogTrigger(train_step, interval=1000),
                    ]


                self.agent_learner = learner.Learner(
                    self.tempdir,
                    train_step,
                    self.rl_agent,
                    triggers=learning_triggers,
                    strategy=strategy)

                self.last_input_data = None
                self.last_action = None
                self.action_step = None
                self.policy_state = None


        '''
        """
        Initialize everything needed by your agent and set the track attribute to the right type:
            Track.SENSORS : CAMERAS, LIDAR, RADAR, GPS and IMU sensors are allowed
            Track.MAP : OpenDRIVE map is also allowed
        """
    def sensors(self):  # pylint: disable=no-self-use
        """
        Define the sensor suite required by the agent

        :return: a list containing the required sensors in the following format:

        [
            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                      'width': 300, 'height': 200, 'fov': 100, 'id': 'Left'},

            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': 0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                      'width': 300, 'height': 200, 'fov': 100, 'id': 'Right'},

            {'type': 'sensor.lidar.ray_cast', 'x': 0.7, 'y': 0.0, 'z': 1.60, 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0,
             'id': 'LIDAR'}
        ]

        names = ['left', 'middle', 'right']
        positions = [{'x': 1.2, 'y': -0.25, 'z': 1.3},{'x': 1.3, 'y': 0.0, 'z': 1.3,},{'x': 1.2, 'y': 0.25, 'z': 1.3}]
        rotations = [{'roll': 0.0, 'pitch': 0.0, 'yaw': -45.0},{'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0},{'roll': 0.0, 'pitch': 0.0, 'yaw': 45.0}]
        whf_list = [{'width': 256, 'height': 128, 'fov': 90},{'width': 256, 'height': 128, 'fov': 90},{'width': 256, 'height': 128, 'fov': 90}]

        """
        sensors = [ {'type': 'sensor.camera.rgb', 'x': 1.2, 'y': -0.25, 'z': 1.3, 'roll': 0.0, 'pitch': 0.0, 'yaw': -45.0, 'width': 256, 'height': 128, 'fov': 90, 'id': 'left'},
                    {'type': 'sensor.camera.rgb', 'x': 1.3, 'y': 0.0, 'z': 1.3, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0, 'width': 256, 'height': 128, 'fov': 90, 'id': 'middle'},
                    {'type': 'sensor.camera.rgb', 'x': 1.2, 'y': 0.25, 'z': 1.3, 'roll': 0.0, 'pitch': 0.0, 'yaw': 45.0, 'width': 256, 'height': 128, 'fov': 90, 'id': 'right'},
                    {'type': 'sensor.speedometer', 'id': 'Speed'}]

        return sensors

    def run_step(self, input_data, timestamp):
        """
        Execute one step of navigation.
        :return: control
        """
        
        

        #return action
        
        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 0.0
        control.hand_brake = False

        return control

    def destroy(self):
        """
        Destroy (clean-up) the agent
        :return:
        """
        pass

    def _train(self):
        #loss_info = self.agent_learner.run(iterations=GRADIENT_STEPS, iterator=iter(self.replay_buffer))
        #print(loss_info.loss.numpy())

        pass

    def add_to_buffer(self, input_data, action):

        input_rgb = input_data["rgb"]
        '''
        im = tf.expand_dims(preprocess_image(input_rgb), axis=0)
        latent_vector = self.vae.encode(im)
        latent_vector = [0]*128

        if not input_data["first"]:
            self.replay_buffer.add(obs=self.last_input_data["latent_vector"],
                act=self.last_action,
                rew=input_data["reward"],
                next_obs=latent_vector,
                done=input_data["game_over"],
                first=self.last_input_data["first"])

        input_data["latent_vector"] = latent_vector

        if input_data["game_over"]:
            self.last_input_data = None
            self.last_action = None
            self.replay_buffer.on_episode_end()
        else:
            self.last_input_data = input_data
            self.last_action = action
        '''

    def _process_state(self, input_data):
        '''
        # Make input_data into useful things

        input_rgb = input_data["rgb"]

        im = tf.expand_dims(preprocess_image(input_rgb), axis=0)
        latent_vector = self.vae.encode(im)

        if "game_over" in input_data and input_data["game_over"] == True:
            steptType = 2
        elif input_data["first"]:
            steptType = 0
        else:
            steptType = 1

        latent_vector = [0]*128

        time_step = TimeStep(StepType(np.asarray(steptType, dtype=np.int32)), np.asarray(input_data["reward"], dtype=np.float32), np.asarray(1.0, dtype=np.float32), np.asarray(latent_vector, dtype=np.float32))

        return time_step

        '''
    '''
    def _save_trayectory(self, next_time_step, next_action_step):
        if next_time_step.is_first():
            self.time_step = next_time_step
            self.action_step = next_action_step
        else:
            policy_state_to_save = self.action_step.state
            action_step_with_previous_state = self.action_step._replace(state=self.policy_state)
            traj = trajectory.from_transition(
                self.time_step, action_step_with_previous_state, next_time_step)

            self.rb_observer.observer(traj)

            self.time_step = next_time_step
            self.action_step = next_action_step
            self.policy_state = policy_state_to_save

    def _finish_episode(self):
        self.time_step = None
        self.action_step = None
        self.policy_state = None
    '''



