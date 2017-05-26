import numpy as np
import tensorflow as tf
np.random.seed(1)
tf.set_random_seed(1)

class PolicyGradient:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.95,
            output_graph=False,
    ):
        #动作空间的维数
        self.n_actions = n_actions
        #状态特征的维数
        self.n_features = n_features
        #学习速率
        self.lr = learning_rate
        #回报衰减率
        self.gamma = reward_decay
        #一条轨迹的观测值，动作值，和回报值
        self.ep_obs, self.ep_as, self.ep_rs = [],[],[]
        #创建策略网络
        self._build_net()
        #启动一个默认的会话
        self.sess = tf.Session()
        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)
        # 初始化会话中的变量
        self.sess.run(tf.global_variables_initializer())
    #创建策略网络的实现
    def _build_net(self):
        with tf.name_scope('input'):
            #创建占位符作为输入
            self.tf_obs = tf.placeholder(tf.float32, [None, self.n_features], name="observations")
            self.tf_acts = tf.placeholder(tf.int32, [None, ], name="actions_num")
            self.tf_vt = tf.placeholder(tf.float32, [None, ], name="actions_value")
        #第一层
        layer = tf.layers.dense(
            inputs=self.tf_obs,
            units=10,
            activation=tf.nn.tanh,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc1',
        )
        #第二层
        all_act = tf.layers.dense(
            inputs=layer,
            units=self.n_actions,
            activation=None,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc2'

        )
        #利用softmax函数得到每个动作的概率
        self.all_act_prob = tf.nn.softmax(all_act, name='act_prob')
        #定义损失函数
        with tf.name_scope('loss'):
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_act,labels=self.tf_acts)
            loss = tf.reduce_mean(neg_log_prob*self.tf_vt)
        #定义训练,更新参数
        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)
    #定义如何选择行为，即状态ｓ处的行为采样.根据当前的行为概率分布进行采样
    def choose_action(self, observation):
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs:observation[np.newaxis,:]})
        #按照给定的概率采样
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())
        return action
    def greedy(self, observation):
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs: observation[np.newaxis, :]})
        action = np.argmax(prob_weights.ravel())
        return action
    #定义存储，将一个回合的状态，动作和回报都保存在一起
    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)
    #学习，以便更新策略网络参数，一个episode之后学一回
    def learn(self):
        #计算一个episode的折扣回报
        discounted_ep_rs_norm = self._discount_and_norm_rewards()
        #调用训练函数更新参数
        self.sess.run(self.train_op, feed_dict={
            self.tf_obs: np.vstack(self.ep_obs),
            self.tf_acts: np.array(self.ep_as),
            self.tf_vt: discounted_ep_rs_norm,
        })
        #清空episode数据
        self.ep_obs, self.ep_as, self.ep_rs = [], [],[]
        return discounted_ep_rs_norm
    def _discount_and_norm_rewards(self):
        #折扣回报和
        discounted_ep_rs =np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add
        #归一化
        discounted_ep_rs-= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs





