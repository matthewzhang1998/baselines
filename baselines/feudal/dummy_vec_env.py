import numpy as np
from gym import spaces
from collections import OrderedDict
from baselines.common.vec_env import VecEnv

class DummyVecEnv(VecEnv):
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        VecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space)
        self.hier = False
        shapes, dtypes = {}, {}
        self.keys = []
        obs_space = env.observation_space
        if isinstance(obs_space, spaces.Dict):
            assert isinstance(obs_space.spaces, OrderedDict)
            for key, box in obs_space.spaces.items():
                assert isinstance(box, spaces.Box)
                shapes[key] = box.shape
                dtypes[key] = box.dtype
                self.keys.append(key)
        elif isinstance(obs_space, spaces.Tuple):
            tokens, box = obs_space.spaces
            self.buf_tokens = {None: np.zeros((self.num_envs,) + (tokens.n,))}
            self.hier = True
            self.keys = [None]
            shapes, dtypes = {None: box.shape}, {None: box.dtype}
        else:
            box = obs_space
            assert isinstance(box, spaces.Box)
            self.keys = [None]
            shapes, dtypes = { None: box.shape }, { None: box.dtype }
        self.buf_obs = { k: np.zeros((self.num_envs,) + tuple(shapes[k]), dtype=dtypes[k]) for k in self.keys }
        self.buf_dones = np.zeros((self.num_envs,), dtype=np.bool)
        self.buf_rews  = np.zeros((self.num_envs,), dtype=np.float32)
        self.buf_infos = [{} for _ in range(self.num_envs)]
        self.actions = None

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        for e in range(self.num_envs):
            obs, self.buf_rews[e], self.buf_dones[e], self.buf_infos[e] = self.envs[e].step(self.actions[e])
            if self.buf_dones[e]:
                obs = self.envs[e].reset()
            self._save_obs(e, obs)
        return (self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones),
                self.buf_infos.copy())

    def reset(self):
        for e in range(self.num_envs):
            obs = self.envs[e].reset()
            self._save_obs(e, obs)
        return self._obs_from_buf()

    def close(self):
        return

    def _save_obs(self, e, obs):
        for k in self.keys:
            if self.hier==True:
                def one_hot(token):
                    assert token < self.buf_tokens[None].shape[-1]
                    one_hot = np.zeros((self.buf_tokens[None].shape[-1],))
                    one_hot[token] = 1
                    return one_hot
                    
                token, obs = obs
                self.buf_tokens[k][e] = one_hot(token)
                self.buf_obs[k][e] = obs
            elif k is None:
                self.buf_obs[k][e] = obs
            else:
                self.buf_obs[k][e] = obs[k]
                
    def goal(self, obs):
        # does not support keys
        goal = []
        for e in range(self.num_envs):
            goal.append(self.envs[e].env.get_goal_state(obs[e]))
        return np.array(goal, np.float32) 

    def _obs_from_buf(self):
        if self.hier==True:
            return (self.buf_tokens[None], self.buf_obs[None])
        elif self.keys==[None]:
            return self.buf_obs[None]
        else:
            return self.buf_obs
