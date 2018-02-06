import baselines.common.tf_util as U
import tensorflow as tf
import gym
import numpy as np
from baselines.common.distributions import make_pdtype

class CnnPolicy(object):
    recurrent = False
    def __init__(self, name, ob_space, ac_space, kind='large'):
        with tf.variable_scope(name):
            self._init(ob_space, ac_space, kind)
            self.scope = tf.get_variable_scope().name

    def _init(self, ob_space, ac_space, kind):
        assert isinstance(ob_space, gym.spaces.Box)

        self.pdtype = pdtype = make_pdtype(ac_space)
        sequence_length = None

        ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape))
    
        
        if kind == 'small': # from A3C paper
            x1 = ob / 255.0
            with tf.name_scope("conv1"):
                x = tf.nn.relu(U.conv2d(x1, 2, "l1", [8, 8], [4, 4], pad="VALID"))
            #tf.summary.image("first_image", x1)
            with tf.name_scope("conv2"):
                x = tf.nn.relu(U.conv2d(x, 4, "l2", [4, 4], [2, 2], pad="VALID"))
            x = U.flattenallbut0(x)
            with tf.name_scope("dense1"):
                x = tf.nn.relu(U.dense(x, 256, 'lin', U.normc_initializer(1.0)))
            # x = tf.Print(x, [x])
            # if isinstance(ac_space, gym.spaces.Box):
            #     mean = U.dense(x, pdtype.param_shape()[0] // 2, "polfinal", U.normc_initializer(0.01))
            #     logstd = tf.get_variable(name="logstd", shape=[1, pdtype.param_shape()[0] // 2],
            #                              initializer=tf.zeros_initializer())
            #     logits = U.concatenate([mean, mean * 0.0 + logstd], axis=1)
            # else:
            #     logits = U.dense(x, pdtype.param_shape()[0], "logits", U.normc_initializer(0.01))

            with tf.name_scope("dense2"):
                logits = U.dense(x, pdtype.param_shape()[0], "logits", U.normc_initializer(0.01))

            with tf.name_scope("prob_distr"):
                self.pd = pdtype.pdfromflat(logits)

            # ac = self.pd.sample()#(tf.tanh(self.pd.sample()) + 1) / 2 * tf.constant(ac_space.high - ac_space.low, dtype=tf.float32) + tf.constant(ac_space.low, dtype=tf.float32)

            with tf.name_scope("value_predict"):
                self.vpred = U.dense(x, 1, "value", U.normc_initializer(1.0))[:, 0]

        elif kind == 'large': # Nature DQN
            x = ob / 255.0
            x = tf.nn.relu(U.conv2d(x, 32, "l1", [8, 8], [4, 4], pad="VALID"))
            x = tf.nn.relu(U.conv2d(x, 64, "l2", [4, 4], [2, 2], pad="VALID"))
            x = tf.nn.relu(U.conv2d(x, 64, "l3", [3, 3], [1, 1], pad="VALID"))
            x = U.flattenallbut0(x)
            x = tf.nn.relu(U.dense(x, 512, 'lin', U.normc_initializer(1.0)))
            logits = U.dense(x, pdtype.param_shape()[0], "logits", U.normc_initializer(0.01))
            self.pd = pdtype.pdfromflat(logits)
            self.vpred = U.dense(x, 1, "value", U.normc_initializer(1.0))[:, 0]
        elif kind == 'dense': 
            #x = ob / 20.0 + 0.5
            x = tf.nn.l2_normalize(ob, 0)
            l1 = tf.layers.dense(inputs=x, units=512*3, activation=tf.nn.tanh,name="l1")
            l2 = tf.layers.dense(inputs=l1, units=512*2, activation=tf.nn.tanh,name="l2")
            l3 = tf.layers.dense(l2,64*2, tf.nn.tanh,name="l3")
            # l4= tf.layers.dense(l3, 64*4, tf.nn.tanh, name="l4")
            logits = U.dense(l3, pdtype.param_shape()[0], "logits", U.normc_initializer(0.01))
            self.pd = pdtype.pdfromflat(logits)
            self.vpred = U.dense(l3, 1, "value", U.normc_initializer(1.0))[:, 0]
        else:
            raise NotImplementedError

        ac = self.pd.sample()

        self.state_in = []
        self.state_out = []

        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        # ac = self.pd.sample() # XXX

        # check = tf.add_check_numerics_ops()
        # # saver = tf.train.Saver()
        # # tf.summary.image("first_image", x1)
        # tf.summary.scalar("val_scal", tf.reshape(self.vpred, []))
        # tf.summary.scalar("ac_bool", tf.reshape(tf.slice(ac, [0,4], [1,1]), []))
        # tf.summary.scalar("top_or_front", tf.reshape(tf.slice(ac, [0, 5], [1, 1]), []))
        # tf.summary.scalar("pos_x", tf.reshape(tf.slice(ac, [0, 0], [1,1]), []))
        # tf.summary.scalar("pos_y", tf.reshape(tf.slice(ac, [0, 1], [1, 1]), []))
        # tf.summary.scalar("vel_x", tf.reshape(tf.slice(ac, [0, 2], [1,1]), []))
        # tf.summary.scalar("vel_y", tf.reshape(tf.slice(ac, [0, 3], [1, 1]), []))
        #
        # sess = tf.get_default_session()#tf.InteractiveSession()
        # from fs.osfs import OSFS
        # import os
        # dir = '/tmp/mylogdir/'  + tf.get_variable_scope().name + '/'
        # try:
        #     folder = OSFS(dir)
        # except:
        #     os.makedirs(dir)
        #     folder = OSFS(dir)
        # test_n = len(list(n for n in folder.listdir('./') if n.startswith('test')))
        # this_test = dir + "test" + str(test_n + 1)
        # # test_writer = tf.train.SummaryWriter(this_test)
        # file_writer = tf.summary.FileWriter(this_test, sess.graph) #ben
        # merged = tf.summary.merge_all() #ben
        # tf.global_variables_initializer().run() #ben

        merged = None
        if merged is not None:
            self._act = U.function([stochastic, ob], [ac, self.vpred], tensorboard_stuff={ "writer" : file_writer, "merged" : merged, "how_often" : 1}, check =check)
        else:
            self._act = U.function([stochastic, ob], [ac, self.vpred])


    def act(self, stochastic, ob):
        ac1, vpred1 =  self._act(stochastic, ob[None])
        return ac1[0], vpred1#[0] #ben
    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
    def get_initial_state(self):
        return []

