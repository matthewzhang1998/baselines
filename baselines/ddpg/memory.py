import numpy as np
from gym_program.envs.program_env import token_unmap


class RingBuffer(object):
    def __init__(self, maxlen, shape, dtype='float32'):
        self.maxlen = maxlen
        self.start = 0
        self.length = 0
        self.data = np.zeros((maxlen,) + shape).astype(dtype)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise KeyError()
        return self.data[(self.start + idx) % self.maxlen]

    def get_batch(self, idxs):
        return self.data[(self.start + idxs) % self.maxlen]

    def append(self, v):
        if self.length < self.maxlen:
            # We have space, simply increase the length.
            self.length += 1
        elif self.length == self.maxlen:
            # No space, "remove" the first item.
            self.start = (self.start + 1) % self.maxlen
        else:
            # This should never happen.
            raise RuntimeError()
        self.data[(self.start + self.length - 1) % self.maxlen] = v


def array_min2d(x):
    x = np.array(x)
    if x.ndim >= 2:
        return x
    return x.reshape(-1, 1)


class Memory(object):
    def __init__(self, limit, action_shape, observation_shape):
        self.limit = limit

        self.observations0 = RingBuffer(limit, shape=observation_shape)
        self.actions = RingBuffer(limit, shape=action_shape)
        self.rewards = RingBuffer(limit, shape=(1,))
        self.terminals1 = RingBuffer(limit, shape=(1,))
        self.observations1 = RingBuffer(limit, shape=observation_shape)

    def sample(self, batch_size):
        # Draw such that we always have a proceeding element.
        batch_idxs = np.random.random_integers(self.nb_entries - 2, size=batch_size)

        obs0_batch = self.observations0.get_batch(batch_idxs)
        obs1_batch = self.observations1.get_batch(batch_idxs)
        action_batch = self.actions.get_batch(batch_idxs)
        reward_batch = self.rewards.get_batch(batch_idxs)
        terminal1_batch = self.terminals1.get_batch(batch_idxs)

        result = {
            'obs0': array_min2d(obs0_batch),
            'obs1': array_min2d(obs1_batch),
            'rewards': array_min2d(reward_batch),
            'actions': array_min2d(action_batch),
            'terminals1': array_min2d(terminal1_batch),
        }
        return result

    def append(self, obs0, action, reward, obs1, terminal1, training=True):
        if not training:
            return
        
        self.observations0.append(obs0)
        self.actions.append(action)
        self.rewards.append(reward)
        self.observations1.append(obs1)
        self.terminals1.append(terminal1)

    @property
    def nb_entries(self):
        return len(self.observations0)
    
class HierMemory(): #overrides memory class
    def __init__(self, tokens, *args):
        self.memory = {}
        for token in tokens:
            assert isinstance(token, str)
            self.memory[token] = [Memory(*args), 0]
        
    def append(self, obs0, action, reward, obs1, terminal1, training=True):
        token, obs0 = obs0
        token = token_unmap(token)
        _, obs1 = obs1
        assert token in self.memory
        self.memory[token][0].append(obs0, action, reward, obs1, terminal1, training=training)
        self.memory[token][1]+=1
        
    def sample(self, batch_size):
        # Draw such that we always have a proceeding element.
        sample_idxs = np.random.random_integers(self.nb_entries - 2, size=batch_size)
        
        samples = {}
        
        idx_right = 0
        for token in self.memory:
            idx_left = idx_right
            idx_right += self.memory[token][1]
            batch_idxs = np.where(np.logical_and(sample_idxs>=idx_left, sample_idxs<idx_right))
            
            obs0_batch = self.observations0.get_batch(batch_idxs)
            obs1_batch = self.observations1.get_batch(batch_idxs)
            action_batch = self.actions.get_batch(batch_idxs)
            reward_batch = self.rewards.get_batch(batch_idxs)
            terminal1_batch = self.terminals1.get_batch(batch_idxs)
    
            result = {
                'obs0': array_min2d(obs0_batch),
                'obs1': array_min2d(obs1_batch),
                'rewards': array_min2d(reward_batch),
                'actions': array_min2d(action_batch),
                'terminals1': array_min2d(terminal1_batch),
            }
            samples[token] = result
        return samples
        
    @property
    def nb_entries(self):
        nb_entries = 0
        for token in self.memory:
            nb_entries += self.memory[token][1]
        return nb_entries