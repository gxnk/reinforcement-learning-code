import gym
from policynet import PolicyGradient
import matplotlib.pyplot as plt
import time

DISPLAY_REWARD_THRESHOLD = 1000
RENDER = False

#创建一个环境
env = gym.make('CartPole-v0')
env.seed(1)
env = env.unwrapped

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

RL = PolicyGradient(
    n_actions=env.action_space.n,
    n_features=env.observation_space.shape[0],
    learning_rate=0.02,
    reward_decay=0.99,

)
#学习过程
for i_episode in range(85):
    observation = env.reset()
    while True:
        if RENDER: env.render()
        #采样动作，探索环境
        # action = RL.choose_action(observation)
        # observation_, reward, done, info = env.step(action)
        action = RL.choose_action(observation)

        observation_, reward, done, info = env.step(action)

        #将观测，动作和回报存储起来
        RL.store_transition(observation, action, reward)
        if done:
            ep_rs_sum = sum(RL.ep_rs)
            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.99+ep_rs_sum * 0.01
            if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True
            print("episode:", i_episode, "rewards:", int(running_reward))
            #每个episode学习一次
            vt = RL.learn()
            if i_episode == 0:
                plt.plot(vt)
                plt.xlabel('episode steps')
                plt.ylabel('normalized state-action value')
                plt.show()
            break

        #智能体探索一步
        observation = observation_
# #测试过程
for i in range(10):
    observation = env.reset()
    count = 0
    while True:
        # 采样动作，探索环境
        env.render()
        action = RL.greedy(observation)
        #action = RL.choose_action(observation)
        #action = RL.sample_action(observation)
        # print (action)
        # print(action1)
        observation_, reward, done, info = env.step(action)
        if done:
            print(count)
            break
        observation = observation_
        count+=1
        #time.sleep(0.001)
        print (count)





