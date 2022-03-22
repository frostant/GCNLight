import sys,os,re
import numpy as np
from matplotlib import pyplot as plt

path="origin/"
inputFile1 = path+"gowalla_1_0.001_0.5_0.0005_03220853_degreeNormL1.npy"
inputFile2 = path+"gowalla_1_0.001_0.5_0.0005_01110632_T.npy"
# {'precision':
precision=[]
recall=[]
ndcg=[]
loss=[]
# loss=[0.]
# stack=[precision,recall,ndcg,loss]
# np.load(inputFile1)

# TODO：示例图颜色和标签对应
def draw(name):
    stack=np.load(name)
    # stack=
    precision=stack[0,:]
    recall=stack[1,:]
    ndcg=stack[2,:]
    loss=stack[3,:]
    index=range(len(precision))
    plt.figure(1)
    plt.plot(index,precision)
    plt.savefig('./figure/precision.png')
    plt.figure(2)
    plt.plot(index,recall)
    plt.savefig('./figure/recall.png')
    plt.figure(3)
    plt.plot(index,ndcg)
    plt.savefig('./figure/ndcg.png')
    plt.figure(4)
    plt.plot(index,loss)
    plt.savefig('./figure/loss.png')

draw(inputFile1)
draw(inputFile2)

# # print(len(loss),len(ndcg))
# np.save(saveFile,stack)

# uu=np.load(saveFile)
# print(uu)
