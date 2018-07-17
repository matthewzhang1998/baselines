from baselines.common.vec_env import VecEnvWrapper
from baselines.common.running_mean_std import RunningMeanStd
import numpy as np
import tensorflow as tf

class VecNormalize(VecEnvWrapper):
    """
    Vectorized environment base class
    """
    def __init__(self, venv, ob=False, ret=True, clipob=10., cliprew=10., gamma=0.99, epsilon=1e-8):
        VecEnvWrapper.__init__(self, venv)
        self.hier = self.venv.hier
        if self.hier:
            obs_space = self.observation_space.spaces[1]
            self.ob_rms = RunningMeanStd(shape=obs_space.shape) if ob else None
            self.ret_rms = RunningMeanStd(shape=()) if ret else None
        else:    
            self.ob_rms = RunningMeanStd(shape=self.observation_space.shape) if ob else None
            self.ret_rms = RunningMeanStd(shape=()) if ret else None
        self.clipob = clipob
        self.cliprew = cliprew
        self.ret = np.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon

    def step_wait(self):
        """
        Apply sequence of actions to sequence of environments
        actions -> (observations, rewards, news)

        where 'news' is a boolean vector indicating whether each element is new.
        """
        obs, rews, news, infos = self.venv.step_wait()
        self.ret = self.ret * self.gamma + rews
        if self.hier:
            tokens, obs = obs
            obs = self._obfilt(obs)
            obs = (tokens, obs)
        else:
            obs = self._obfilt(obs)
        if self.ret_rms:
            self.ret_rms.update(self.ret)
            rews = np.clip(rews / np.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)
        return obs, rews, news, infos

    def goal(self, obs):
        return self.venv.goal(obs)

    def action(self, obs):
        return self.venv.action(obs)

    def _obfilt(self, obs):
        if self.ob_rms:
            self.ob_rms.update(obs)
            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
            return obs
        else:
            return obs
        
    def tf_filt(self, obs_tf):
        if self.ob_rms:
            obs_tf = tf.clip_by_value((obs_tf - self.ob_rms.mean) / \
                                      tf.cast(np.sqrt(self.ob_rms.var + self.epsilon), tf.float32),
                                      -self.clipob, self.clipob)
            return obs_tf
        else:
            return obs_tf

    def reset(self):
        """
        Reset all environments
        """
        obs = self.venv.reset()
        if self.hier:
            return obs[0], self._obfilt(obs[1])
        else:
            return self._obfilt(obs)
