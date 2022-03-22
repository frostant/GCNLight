import sys,os,re
import numpy as np

path="origin/"
# inputFile = "gowalla_1_0.001_0.5_0.0005_03220853_degreeNormL1.txt"
inputFile = path+"gowalla_1_0.001_0.5_0.0005_01110632_T.txt"
name = inputFile.rsplit('.',1)[0]
saveFile = name+'.npy'
print(saveFile)
# {'precision':
precision=[]
recall=[]
ndcg=[]
loss=[]
# loss=[0.]

with open(inputFile,"r") as fin:
    for lin in fin.readlines():
        if lin[:13]=="{'precision':":
            # print(lin)
            # lis=list(filter(str.isdigit,lin))
            lis=re.findall(r'-?\d+\.?\d*e?[-+]?\d*',lin)
            precision.append(float(lis[0]))
            recall.append(float(lis[1]))
            ndcg.append(float(lis[2]))
            # print(lis)
        if lin[:6]=="EPOCH[":
            # print(lin)
            lis=re.findall(r'-?\d+\.?\d*e?',lin)
            loss.append(float(lis[2]))

# EPOCH[11/1500] loss0.296-|Sample:10.77|

stack=[precision,recall,ndcg,loss]
if len(loss)!=len(ndcg):
    print(len(loss),"   ",len(ndcg))
    print("Error!!!!!!!!!!!!!!!#########")
    exit(0)
stack=np.array(stack)
# print(stack )
# precision=np.array(precision)
# recall=np.array(recall)
# ndcg=np.array(ndcg)
# loss=np.array(loss)
# print(precision)
# print(recall)
# print(ndcg)
# print(loss)

# print(len(loss),len(ndcg))
np.save(saveFile,stack)

uu=np.load(saveFile)
print(uu)