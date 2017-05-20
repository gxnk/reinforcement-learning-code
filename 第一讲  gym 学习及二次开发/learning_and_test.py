import sys
import gym
from qlearning import *
import time
from gym import wrappers
#main函数
if __name__ == "__main__":
   # grid = grid_mdp.Grid_Mdp()  # 创建网格世界
    #states = grid.getStates()  # 获得网格世界的状态空间
    #actions = grid.getAction()  # 获得网格世界的动作空间
    sleeptime=0.5
    terminate_states= grid.env.getTerminate_states()
    #读入最优值函数
    read_best()
#    plt.figure(figsize=(12,6))
    #训练
    qfunc = dict()
    qfunc = qlearning(num_iter1=500, alpha=0.2, epsilon=0.2)
    #画图
    plt.xlabel("number of iterations")
    plt.ylabel("square errors")
    plt.legend()
   # 显示误差图像
    plt.show()
    time.sleep(sleeptime)
    #学到的值函数
    for s in states:
        for a in actions:
            key = "%d_%s"%(s,a)
            print("the qfunc of key (%s) is %f" %(key, qfunc[key]) )
            qfunc[key]
    #学到的策略为：
    print("the learned policy is:")
    for i in range(len(states)):
        if states[i] in terminate_states:
            print("the state %d is terminate_states"%(states[i]))
        else:
            print("the policy of state %d is (%s)" % (states[i], greedy(qfunc, states[i])))
    # 设置系统初始状态
    s0 = 1
    grid.env.setAction(s0)
    # 对训练好的策略进行测试
    grid = wrappers.Monitor(grid, './robotfindgold', force=True)  # 记录回放动画
   #随机初始化，寻找金币的路径
    for i in range(20):
        #随机初始化
        s0 = grid.reset()
        grid.render()
        time.sleep(sleeptime)
        t = False
        count = 0
        #判断随机状态是否在终止状态中
        if s0 in terminate_states:
            print("reach the terminate state %d" % (s0))
        else:
            while False == t and count < 100:
                a1 = greedy(qfunc, s0)
                print(s0, a1)
                grid.render()
                time.sleep(sleeptime)
                key = "%d_%s" % (s0, a)
                # 与环境进行一次交互，从环境中得到新的状态及回报
                s1, r, t, i = grid.step(a1)
                if True == t:
                    #打印终止状态
                    print(s1)
                    grid.render()
                    time.sleep(sleeptime)
                    print("reach the terminate state %d" % (s1))
                # s1处的最大动作
                s0 = s1
                count += 1




