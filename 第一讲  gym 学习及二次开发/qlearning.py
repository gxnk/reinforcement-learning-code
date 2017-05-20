import sys
import gym
import random
random.seed(0)
import time
import matplotlib.pyplot as plt

grid = gym.make('GridWorld-v0')
#grid=env.env                     #创建网格世界
states = grid.env.getStates()        #获得网格世界的状态空间
actions = grid.env.getAction()      #获得网格世界的动作空间
gamma = grid.env.getGamma()       #获得折扣因子
#计算当前策略和最优策略之间的差
best = dict() #储存最优行为值函数
def read_best():
    f = open("best_qfunc")
    for line in f:
        line = line.strip()
        if len(line) == 0: continue
        eles = line.split(":")
        best[eles[0]] = float(eles[1])
#计算值函数的误差
def compute_error(qfunc):
    sum1 = 0.0
    for key in qfunc:
        error = qfunc[key] -best[key]
        sum1 += error *error
    return sum1

#  贪婪策略
def greedy(qfunc, state):
    amax = 0
    key = "%d_%s" % (state, actions[0])
    qmax = qfunc[key]
    for i in range(len(actions)):  # 扫描动作空间得到最大动作值函数
        key = "%d_%s" % (state, actions[i])
        q = qfunc[key]
        if qmax < q:
            qmax = q
            amax = i
    return actions[amax]


#######epsilon贪婪策略
def epsilon_greedy(qfunc, state, epsilon):
    amax = 0
    key = "%d_%s"%(state, actions[0])
    qmax = qfunc[key]
    for i in range(len(actions)):    #扫描动作空间得到最大动作值函数
        key = "%d_%s"%(state, actions[i])
        q = qfunc[key]
        if qmax < q:
            qmax = q
            amax = i
    #概率部分
    pro = [0.0 for i in range(len(actions))]
    pro[amax] += 1-epsilon
    for i in range(len(actions)):
        pro[i] += epsilon/len(actions)

    ##选择动作
    r = random.random()
    s = 0.0
    for i in range(len(actions)):
        s += pro[i]
        if s>= r: return actions[i]
    return actions[len(actions)-1]

def qlearning(num_iter1, alpha, epsilon):
    x = []
    y = []
    qfunc = dict()   #行为值函数为字典
    #初始化行为值函数为0
    for s in states:
        for a in actions:
            key = "%d_%s"%(s,a)
            qfunc[key] = 0.0
    for iter1 in range(num_iter1):
        x.append(iter1)
        y.append(compute_error(qfunc))

        #初始化初始状态
        s = grid.reset()
        a = actions[int(random.random()*len(actions))]
        t = False
        count = 0
        while False == t and count <100:
            key = "%d_%s"%(s, a)
            #与环境进行一次交互，从环境中得到新的状态及回报
            s1, r, t1, i =grid.step(a)
            key1 = ""
            #s1处的最大动作
            a1 = greedy(qfunc, s1)
            key1 = "%d_%s"%(s1, a1)
            #利用qlearning方法更新值函数
            qfunc[key] = qfunc[key] + alpha*(r + gamma * qfunc[key1]-qfunc[key])
            #转到下一个状态
            s = s1;
            a = epsilon_greedy(qfunc, s1, epsilon)
            count += 1
    plt.plot(x,y,"-.,",label ="q alpha=%2.1f epsilon=%2.1f"%(alpha,epsilon))
    return qfunc


        

