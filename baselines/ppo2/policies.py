import numpy as np
import tensorflow as tf
from baselines import logger
from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch, lstm, lnlstm
from baselines.common.distributions import make_pdtype
from baselines.program.distributions import make_pdtype_hier
from gym_program.envs.program_env import obs_unmap, get_all_tokens
from gym.spaces import Tuple, Discrete


def nature_cnn(unscaled_images, **conv_kwargs):
    """
    CNN from Nature paper.
    """
    scaled_images = tf.cast(unscaled_images, tf.float32) / 255.
    activ = tf.nn.relu
    h = activ(conv(scaled_images, 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2),
                   **conv_kwargs))
    h2 = activ(conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2), **conv_kwargs))
    h3 = activ(conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2), **conv_kwargs))
    h3 = conv_to_fc(h3)
    return activ(fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2)))

class LnLstmPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, nlstm=256, reuse=False):
        nenv = nbatch // nsteps
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc)
        X = tf.placeholder(tf.uint8, ob_shape) #obs
        M = tf.placeholder(tf.float32, [nbatch]) #mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, nlstm*2]) #states
        self.pdtype = make_pdtype(ac_space)
        with tf.variable_scope("model", reuse=reuse):
            h = nature_cnn(X)
            xs = batch_to_seq(h, nenv, nsteps)
            ms = batch_to_seq(M, nenv, nsteps)
            h5, snew = lnlstm(xs, ms, S, 'lstm1', nh=nlstm)
            h5 = seq_to_batch(h5)
            vf = fc(h5, 'v', 1)
            self.pd, self.pi = self.pdtype.pdfromlatent(h5)

        v0 = vf[:, 0]
        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = np.zeros((nenv, nlstm*2), dtype=np.float32)

        def step(ob, state, mask):
            return sess.run([a0, v0, snew, neglogp0], {X:ob, S:state, M:mask})

        def value(ob, state, mask):
            return sess.run(v0, {X:ob, S:state, M:mask})

        self.X = X
        self.M = M
        self.S = S
        self.vf = vf
        self.step = step
        self.value = value

class LstmPolicy(object):

    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, nlstm=256, reuse=False):
        nenv = nbatch // nsteps

        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc)
        self.pdtype = make_pdtype(ac_space)
        X = tf.placeholder(tf.uint8, ob_shape) #obs
        M = tf.placeholder(tf.float32, [nbatch]) #mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, nlstm*2]) #states
        with tf.variable_scope("model", reuse=reuse):
            h = nature_cnn(X)
            xs = batch_to_seq(h, nenv, nsteps)
            ms = batch_to_seq(M, nenv, nsteps)
            h5, snew = lstm(xs, ms, S, 'lstm1', nh=nlstm)
            h5 = seq_to_batch(h5)
            vf = fc(h5, 'v', 1)
            self.pd, self.pi = self.pdtype.pdfromlatent(h5)

        v0 = vf[:, 0]
        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = np.zeros((nenv, nlstm*2), dtype=np.float32)

        def step(ob, state, mask):
            return sess.run([a0, v0, snew, neglogp0], {X:ob, S:state, M:mask})

        def value(ob, state, mask):
            return sess.run(v0, {X:ob, S:state, M:mask})

        self.X = X
        self.M = M
        self.S = S
        self.vf = vf
        self.step = step
        self.value = value

class CnnPolicy(object):

    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False, **conv_kwargs): #pylint: disable=W0613
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc)
        self.pdtype = make_pdtype(ac_space)
        X = tf.placeholder(tf.uint8, ob_shape) #obs
        with tf.variable_scope("model", reuse=reuse):
            h = nature_cnn(X, **conv_kwargs)
            vf = fc(h, 'v', 1)[:,0]
            self.pd, self.pi = self.pdtype.pdfromlatent(h, init_scale=0.01)

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None

        def step(ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X:ob})
            return a, v, self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X:ob})

        self.X = X
        self.vf = vf
        self.step = step
        self.value = value

class MlpPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False, noptions=1): #pylint: disable=W0613
        ob_shape = (None,) + ob_space.shape
        self.nbatch = nbatch
        self.pdtype = make_pdtype(ac_space)
        X = tf.placeholder(tf.float32, ob_shape, name='Ob') #obs
        with tf.variable_scope("model", reuse=reuse):
            activ = tf.tanh
            flatten = tf.layers.flatten
            pi_h1 = activ(fc(flatten(X), 'pi_fc1', nh=64, init_scale=np.sqrt(2)))
            pi_h2 = activ(fc(pi_h1, 'pi_fc2', nh=64, init_scale=np.sqrt(2)))
            vf_h1 = activ(fc(flatten(X), 'vf_fc1', nh=64, init_scale=np.sqrt(2)))
            vf_h2 = activ(fc(vf_h1, 'vf_fc2', nh=64, init_scale=np.sqrt(2)))
            vf = fc(vf_h2, 'vf', noptions)[:,0]

            self.pd, self.pi = self.pdtype.pdfromlatent(pi_h2, init_scale=0.01)
        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None

        def step(ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X:ob})
            return a, v, self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X:ob})

        self.X = X
        self.vf = vf
        self.step = step
        self.value = value

class HierPolicy(MlpPolicy):
    def __init__(self, sess, ob_space, *args, **kwargs):
        ob_space = ob_space.spaces[1]
        super(HierPolicy, self).__init__(sess, ob_space, *args, **kwargs)
    
    def hier_step(self, obs, *_args, **_kwargs):
        a = np.zeros((self.nbatch,))
        v = np.zeros((self.nbatch,)) 
        neglogp = np.zeros((self.nbatch,)) 
        obs_by_sketch = obs_unmap(obs)
        for i in range(len(obs_by_sketch)):
            kw, ob = obs_by_sketch[i]
            with tf.variable_scope(kw):
                 a[i], v[i], self.initial_state, neglogp[i] = self.step(ob, *_args, **_kwargs)
        return a, v, self.initial_state, neglogp
        
    def hier_value(self, obs, *_args, **_kwargs):
        v = np.zeros((self.nbatch,))
        obs_by_sketch = obs_unmap(obs)
        for i in range(len(obs_by_sketch)):
            kw, ob = obs_by_sketch[i]
            with tf.variable_scope(kw):
                v[i] = self.value(ob, *_args, **_kwargs)
        return v
    
class HierPolicy2(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False, noptions=1): #pylint: disable=W0613
        self.tokens = get_all_tokens()
        ob_shape = (None,) + ob_space.shape
        self.nbatch = nbatch
        self.pdtype = make_pdtype_hier(ac_space, len(self.tokens))
        X = tf.placeholder(tf.float32, ob_shape, name='Ob') #obs
        mask = tf.placeholder(tf.float32, (None, len(self.tokens)))
        with tf.variable_scope("model", reuse=reuse):
            activ = tf.tanh
            flatten = tf.layers.flatten
            pi_h1 = activ(fc(flatten(X), 'pi_fc1', nh=128, init_scale=np.sqrt(2)))
            pi_h2 = activ(fc(pi_h1, 'pi_fc2', nh=128, init_scale=np.sqrt(2)))
            vf_h1 = activ(fc(flatten(X), 'vf_fc1', nh=64, init_scale=np.sqrt(2)))
            vf_h2 = activ(fc(vf_h1, 'vf_fc2', nh=64, init_scale=np.sqrt(2)))
            vf = fc(vf_h2, 'vf', len(self.tokens))

            self.pd, self.pi = self.pdtype.pdfromlatent(pi_h2, init_scale=0.01)
        a = self.pd.sample() # vector of length noptions
        a_cast = tf.cast(a, tf.float32)
        neglogp = self.pd.neglogp(a) # vector of length noptions
        
        a0 = tf.cast(tf.reduce_sum(tf.multiply(a_cast, mask), axis=-1), tf.int64)
        neglogp0 = tf.reduce_sum(tf.multiply(neglogp, mask), axis=-1)
        vf0 = tf.reduce_sum(tf.multiply(vf, mask), axis=-1)
        self.initial_state = None

        def step(ob, *_args, **_kwargs):
            opmask, ob = ob
            a, v, neglogp = sess.run([a0, vf0, neglogp0], {X:ob, mask:opmask})
            return a, v, self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            opmask, ob = ob
            return sess.run(vf0, {X:ob, mask:opmask})

        self.X = X
        self.mask = mask
        self.vf = vf
        self.step = step
        self.value = value
        
BATCH = 2048
        
def RandomWalkPolicy(env):
    avg_rew=0
    avg_tstep=0
    rew=0
    tstep=0
    n_iterations=0     
    env.reset()
    assert isinstance(env.action_space, Discrete)
    nac = env.action_space.n
    for iteration in range(1000000):
        action=[np.random.randint(0, nac)]
        _, reward, done, infos = env.step(action)
        rew+=reward
        tstep+=1
        
        if done:
            n_iterations+=1
            avg_rew=avg_rew*(n_iterations - 1)/(n_iterations)
            avg_rew+=rew/n_iterations
            rew=0
            avg_tstep=avg_tstep*(n_iterations - 1)/n_iterations
            avg_tstep+=tstep/n_iterations
            tstep=0
            
        if iteration % BATCH == 0:
            logger.logkv('it', iteration)
            logger.logkv('eprewmean', avg_rew)
            logger.logkv('eplenmean', avg_tstep)
            logger.dumpkvs()
            
    env.close()
            
        
        