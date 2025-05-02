# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from gym import core, spaces
import numpy as np
from dm_env import specs
import collections

import gym
import mujoco

def _spec_to_box(spec):
    """
    Helper function sourced from: https://github.com/denisyarats/dmc2gym
    """
    def extract_min_max(s):
        assert s.dtype == np.float64 or s.dtype == np.float32
        dim = np.int(np.prod(s.shape))
        if type(s) == specs.Array:
            bound = np.inf * np.ones(dim, dtype=np.float32)
            return -bound, bound
        elif type(s) == specs.BoundedArray:
            zeros = np.zeros(dim, dtype=np.float32)
            return s.minimum + zeros, s.maximum + zeros

    mins, maxs = [], []
    for s in spec:
        mn, mx = extract_min_max(s)
        mins.append(mn)
        maxs.append(mx)
    low = np.concatenate(mins, axis=0)
    high = np.concatenate(maxs, axis=0)
    assert low.shape == high.shape
    return spaces.Box(low, high, dtype=np.float32)

class GymWrapper(core.Env):
    metadata = {"render.modes": ['rgb_array'], "video.frames_per_second": 25}

    def __init__(self, base_env):
        """
        Initializes 
        """
        self._base_env = base_env
        self._flat_dict = False

        # parses and stores action space
        self.action_space = _spec_to_box([base_env.action_spec()])
        # parses and stores (possibly nested) observation space
        if isinstance(base_env.observation_spec(), (dict, collections.OrderedDict)):
            obs_space = collections.OrderedDict()
            for k, v in base_env.observation_spec().items():
                obs_space[k] = _spec_to_box([v])
            self.observation_space = spaces.Dict(obs_space)
            if base_env.flat_obs:
                self.observation_space = self.observation_space['observations']
                self._flat_dict = True
        else:
            self.observation_space = _spec_to_box([base_env.observation_spec()])

    def reset(self):
        step = self._base_env.reset()
        obs = step.observation
        obs = obs['observations'] if self._flat_dict else obs
        return obs
    
    def step(self, action):
        step = self._base_env.step(action.astype(self.action_space.dtype))
        o = step.observation
        o = o['observations'] if self._flat_dict else o
        r = step.reward
        done = step.last()
        info = self._base_env.task.step_info
        return o, r, done, info

    def render(self, mode='rgb_array', height=240, width=320, camera_id=None):
        assert mode == 'rgb_array', "env only supports rgb_array rendering"
        if camera_id is None:
            camera_id = self._base_env.default_camera_id
        return self._base_env.physics.render(height=height, width=width, 
                                                camera_id=camera_id)

    @property
    def wrapped(self):
        return self._base_env
    
class PGDMObsWrapperObjCvelOnlyForce(gym.ObservationWrapper):
    def __init__(self, env, domain):
        super().__init__(env)

        # self.cfg = cfg

        self.num_pi_and_Q_observations = self.env.observation_space.shape[0]
        self.num_controlled_variables = 6

        self.observation_space = gym.spaces.Box(low=-10., high=10.,
                                                shape=(self.num_pi_and_Q_observations + self.num_controlled_variables,),
                                                dtype=self.env.observation_space.dtype)

        # indices of observations that will be passed as input to the policy and the Q-function networks
        self.pi_and_Q_observations = list(range(self.num_pi_and_Q_observations))

        # indices of the controlled variables
        self.controlled_variables = [self.num_pi_and_Q_observations + i for i in range(self.num_controlled_variables)]

        # domain, _ = cfg.env['name'].split('-')
        self.object_body_id = self.env.unwrapped._base_env.physics.model.name2id(domain + '/object', 'body')

        # self.object_body_id = self.env.unwrapped.sim.model.body(self.cfg.env.object[self.cfg.env.env_id]).id
        # self.finger_tip_bodies_ids = [self.env.unwrapped.sim.model.body(i).id for i in range(env.unwrapped.sim.model.nbody)
                                    #   if self.env.unwrapped.sim.model.body(i).name in ["distal_thumb", "distph2", "distph3", "distph4", "distph5"]]

    # def normal_force(self):

    #     total_normal_force = 0.
    #     for i_con, con in enumerate(self.env.unwrapped.sim.data.contact):
    #         if self.env.unwrapped.sim.model.geom(con.geom1).bodyid == self.object_body_id:
    #             if self.env.unwrapped.sim.model.geom(con.geom2).bodyid in self.finger_tip_bodies_ids:
    #                 total_normal_force += self.single_contact_normal_force(con, i_con)
    #         elif self.env.unwrapped.sim.model.geom(con.geom2).bodyid == self.object_body_id:
    #             if self.env.unwrapped.sim.model.geom(con.geom1).bodyid in self.finger_tip_bodies_ids:
    #                 total_normal_force += self.single_contact_normal_force(con, i_con)
    #     return np.array(total_normal_force)
        
    # def single_contact_normal_force(self, con, i_con):

    #     # https://roboti.us/book/programming.html
    #     contact_force = np.zeros((6,1))
    #     mujoco.mj_contactForce(self.env.unwrapped.sim.model._model, self.env.unwrapped.sim.data._data, i_con, contact_force)
    #     # force_vector = contact_force[:3,0]
    #     # contact_normal = con.frame[:3]
    #     # normal_force = abs(np.dot(force_vector, contact_normal))
    #     normal_force = abs(contact_force[0,0])

    #     return normal_force
        
    def observation(self, obs):

        # total_normal_force = self.normal_force()

        new_obs = np.concatenate((obs,
                                    self.env.unwrapped._base_env.physics.data.cvel[self.object_body_id]),
                                    # total_normal_force[None]), 
                                    axis=0).astype(np.float32)
        
        return new_obs