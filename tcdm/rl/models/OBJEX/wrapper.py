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
    
class PGDMObsWrapperObjQvelForce(gym.ObservationWrapper):
    def __init__(self, env, domain):
        super().__init__(env)

        self.num_pi_and_Q_observations = self.env.observation_space.shape[0]
        self.num_controlled_variables = 6 + 1

        self.observation_space = gym.spaces.Box(low=-10., high=10.,
                                                shape=(self.num_pi_and_Q_observations + self.num_controlled_variables,),
                                                dtype=self.env.observation_space.dtype)

        # indices of observations that will be passed as input to the policy and the Q-function networks
        self.pi_and_Q_observations = list(range(self.num_pi_and_Q_observations))

        # indices of the controlled variables
        self.controlled_variables = [self.num_pi_and_Q_observations + i for i in range(self.num_controlled_variables)]

        self.model = self.env.unwrapped._base_env.physics.model

        object_geom_name_to_id = {}
        adroit_geom_name_to_id = {}
        object_name = domain + "/"
        for i in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model.ptr, mujoco.mjtObj.mjOBJ_GEOM, i)
            if "adroit/C_th" in name or "adroit/C_ff" in name or "adroit/C_mf" in name or "adroit/C_rf" in name or "adroit/C_lf" in name:
                adroit_geom_name_to_id[name] = i
            elif object_name in name:
                object_geom_name_to_id[name] = i

        self.adroit_geom_ids = set(adroit_geom_name_to_id.values())
        self.object_geom_ids = set(object_geom_name_to_id.values())

    def normal_force(self):

        total_normal_force = 0.
        for i_con, con in enumerate(self.env.unwrapped._base_env.physics.data.contact):
            if con.geom1 in self.adroit_geom_ids and con.geom2 in self.object_geom_ids or \
                con.geom2 in self.adroit_geom_ids and con.geom1 in self.object_geom_ids:
                total_normal_force += self.single_contact_normal_force(i_con)
        return np.array(total_normal_force)
        
    def single_contact_normal_force(self, i_con):

        self.env.unwrapped._base_env.physics.forward()

        # https://roboti.us/book/programming.html
        contact_force = np.zeros((6,1))
        mujoco.mj_contactForce(self.model.ptr, self.env.unwrapped._base_env.physics.data.ptr, i_con, contact_force)
        normal_force = abs(contact_force[0,0])

        return normal_force
        
    def observation(self, obs):

        total_normal_force = self.normal_force()

        # controlled variable elements are scaled so that they have the same order of magnitude
        new_obs = np.concatenate((obs,
                                    self.env.unwrapped._base_env.physics.data.qvel[-6:-3]*10.,
                                    self.env.unwrapped._base_env.physics.data.qvel[-3:],
                                    total_normal_force[None]/10.), 
                                    axis=0).astype(np.float32)
        
        return new_obs

class PGDMObsWrapperObjQvelForceTable(gym.ObservationWrapper):
    def __init__(self, env, domain):
        super().__init__(env)

        self.num_pi_and_Q_observations = self.env.observation_space.shape[0]
        self.num_controlled_variables = 6 + 1
        self.num_added_observations = 1

        self.observation_space = gym.spaces.Box(low=-10., high=10.,
                                                shape=(self.num_pi_and_Q_observations
                                                       + self.num_controlled_variables
                                                       + self.num_added_observations,),
                                                dtype=self.env.observation_space.dtype)

        # indices of observations that will be passed as input to the policy and the Q-function networks
        self.pi_and_Q_observations = list(range(self.num_pi_and_Q_observations))

        # indices of the controlled variables
        self.controlled_variables = [self.num_pi_and_Q_observations + i for i in range(self.num_controlled_variables)]

        self.model = self.env.unwrapped._base_env.physics.model

        object_geom_name_to_id = {}
        adroit_geom_name_to_id = {}
        object_name = domain + "/"
        for i in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model.ptr, mujoco.mjtObj.mjOBJ_GEOM, i)
            if "adroit/C_th" in name or "adroit/C_ff" in name or "adroit/C_mf" in name or "adroit/C_rf" in name or "adroit/C_lf" in name:
                adroit_geom_name_to_id[name] = i
            elif object_name in name:
                object_geom_name_to_id[name] = i

        self.adroit_geom_ids = set(adroit_geom_name_to_id.values())
        self.object_geom_ids = set(object_geom_name_to_id.values())

        table_body_name_to_id = {}
        for i in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model.ptr, mujoco.mjtObj.mjOBJ_BODY, i)
            if "table" in name:
                table_body_name_to_id[name] = i

        self.table_body_ids = set(table_body_name_to_id.values())

    def object_touching_table(self):

        for i_con, con in enumerate(self.env.unwrapped._base_env.physics.data.contact):
            if self.model.ptr.geom(con.geom1).bodyid[0] in self.table_body_ids and con.geom2 in self.object_geom_ids or \
                self.model.ptr.geom(con.geom2).bodyid[0] in self.table_body_ids and con.geom1 in self.object_geom_ids:
                return np.ones(1)
        return np.zeros(1)

    def normal_force(self):

        total_normal_force = 0.
        for i_con, con in enumerate(self.env.unwrapped._base_env.physics.data.contact):
            if con.geom1 in self.adroit_geom_ids and con.geom2 in self.object_geom_ids or \
                con.geom2 in self.adroit_geom_ids and con.geom1 in self.object_geom_ids:
                total_normal_force += self.single_contact_normal_force(i_con)
        return np.array(total_normal_force)
        
    def single_contact_normal_force(self, i_con):

        self.env.unwrapped._base_env.physics.forward()

        # https://roboti.us/book/programming.html
        contact_force = np.zeros((6,1))
        mujoco.mj_contactForce(self.model.ptr, self.env.unwrapped._base_env.physics.data.ptr, i_con, contact_force)
        normal_force = abs(contact_force[0,0])

        return normal_force
        
    def observation(self, obs):

        total_normal_force = self.normal_force()

        object_touching_table = self.object_touching_table()

        # controlled variable elements are scaled so that they have the same order of magnitude
        new_obs = np.concatenate((obs,
                                    self.env.unwrapped._base_env.physics.data.qvel[-6:-3]*10.,
                                    self.env.unwrapped._base_env.physics.data.qvel[-3:],
                                    total_normal_force[None]/10.,
                                    object_touching_table), 
                                    axis=0).astype(np.float32)
        
        return new_obs
    
class PGDMObsWrapperTipEx(gym.ObservationWrapper):
    def __init__(self, env, domain):
        super().__init__(env)

        self.num_pi_and_Q_observations = self.env.observation_space.shape[0]
        self.num_controlled_variables = 21

        self.observation_space = gym.spaces.Box(low=-10., high=10.,
                                                shape=(self.num_pi_and_Q_observations + self.num_controlled_variables,),
                                                dtype=self.env.observation_space.dtype)

        # indices of observations that will be passed as input to the policy and the Q-function networks
        self.pi_and_Q_observations = list(range(self.num_pi_and_Q_observations))

        # indices of the controlled variables
        self.controlled_variables = [self.num_pi_and_Q_observations + i for i in range(self.num_controlled_variables)]

        self.model = self.env.unwrapped._base_env.physics.model

        self.tip_sites = []
        for i in range(self.model.nsite):
            name = mujoco.mj_id2name(self.model.ptr, mujoco.mjtObj.mjOBJ_SITE, i)
            if "adroit/S_grasp" in name:
                self.palm_sid = i
            elif "adroit/S_" in name and "tip" in name:
                self.tip_sites.append(i)

    def mat2euler(self, mat):
        _FLOAT_EPS = np.finfo(np.float64).eps
        _EPS4 = _FLOAT_EPS * 4.0
        """ Convert Rotation Matrix to Euler Angles """
        mat = np.asarray(mat, dtype=np.float64)
        assert mat.shape[-2:] == (3, 3), "Invalid shape matrix {}".format(mat)

        cy = np.sqrt(mat[..., 2, 2] * mat[..., 2, 2] + mat[..., 1, 2] * mat[..., 1, 2])
        condition = cy > _EPS4
        euler = np.empty(mat.shape[:-1], dtype=np.float64)
        euler[..., 2] = np.where(condition,
                                -np.arctan2(mat[..., 0, 1], mat[..., 0, 0]),
                                -np.arctan2(-mat[..., 1, 0], mat[..., 1, 1]))
        euler[..., 1] = np.where(condition,
                                -np.arctan2(-mat[..., 0, 2], cy),
                                -np.arctan2(-mat[..., 0, 2], cy))
        euler[..., 0] = np.where(condition,
                                -np.arctan2(mat[..., 1, 2], mat[..., 2, 2]),
                                0.0)
        return euler

    def observation(self, obs):

        palm_pos = self.env.unwrapped._base_env.physics.data.site_xpos[self.palm_sid]
        palm_ori = np.reshape(self.env.unwrapped._base_env.physics.data.site_xmat[self.palm_sid], (3, 3))
        palm_ori_euler = self.mat2euler(palm_ori)

        # palm_ori_inv = np.linalg.inv(palm_ori)
        palm_ori_inv = palm_ori.T

        # calculate digit tip positions relative to coordinate frame centered on palm
        tip0 = palm_ori_inv @ (self.env.unwrapped._base_env.physics.data.site_xpos[self.tip_sites[0]]-palm_pos)
        tip1 = palm_ori_inv @ (self.env.unwrapped._base_env.physics.data.site_xpos[self.tip_sites[1]]-palm_pos)
        tip2 = palm_ori_inv @ (self.env.unwrapped._base_env.physics.data.site_xpos[self.tip_sites[2]]-palm_pos)
        tip3 = palm_ori_inv @ (self.env.unwrapped._base_env.physics.data.site_xpos[self.tip_sites[3]]-palm_pos)
        tip4 = palm_ori_inv @ (self.env.unwrapped._base_env.physics.data.site_xpos[self.tip_sites[4]]-palm_pos)

        digit_tips = np.concatenate((tip0,
                                     tip1,
                                     tip2,
                                     tip3,
                                     tip4), axis=0)

        # controlled variable elements are scaled so that they have the same order of magnitude
        new_obs = np.concatenate((obs,
                                  palm_pos,
                                  palm_ori_euler,
                                  digit_tips,
                                  ), 
                                  axis=0).astype(np.float32)
        
        return new_obs