import sys,os
# import numpy as np
from collections import Counter
fname = "./lastfm/data1.txt"
tname = "test1.txt"
m1=m2=0
s1=set()
s2=set()
lis1=[]
lis2=[]

def countDegree(lis):
    cou=Counter(lis)
    cou=sorted(cou.items(),key=lambda x: x[0])
    print(cou)
    sum=0
    tim=0
    for i in cou:
        sum+=i[1]
        if i[1]<=12:
            tim+=1 
    print(tim)
    # value = np.array(cou.values())
    # sum=np.sum(value)
    print(sum)
    print(sum/cou[-1][0])
with open(fname,"r") as fin:
    content=fin.readlines()
    for lin in content:
        tmp=lin.split("\t")
        m1=max(m1,int(tmp[0]))
        m2=max(m2,int(tmp[1]))
        s1.add(int(tmp[0]))
        s2.add(int(tmp[1]))
        lis1.append(int(tmp[0]))
        lis2.append(int(tmp[1]))
        # print(tmp[0],end=",")
        # print(tmp[1])
countDegree(lis1) # 12 113
# countDegree(lis2) # 1 125
exit(0)

print(m1,m2)
print(len(s1))
print(len(s2))
arr1 = list(s1)
arr1=sorted(arr1)
d1=dict()
for i,x in enumerate(arr1):
    d1[x]=i
arr2 = list(s2)
arr2=sorted(arr2)
d2=dict()
for i,x in enumerate(arr2):
    d2[x]=i
dd=[]
for i in range(len(lis1)):
    dd.append((d1[lis1[i]],d2[lis2[i]]))
    # print(f"{d1[lis1[i]]},{d2[lis2[i]]}",file=ftrain)
dd=sorted(dd)
with open("lastfm.train","w") as ftrain:
    # for i in range(4476):
    #     print(f"0,{i}",file=ftrain)
    for i in range(len(dd)):
        # if dd[i][0]==0:
        #     continue 
        # print(f"{d1[lis1[i]]},{d2[lis2[i]]}",file=ftrain)
        print(f"{dd[i][0]},{dd[i][1]}",file=ftrain)

dd=[]
with open(tname,"r") as fin:
    content=fin.readlines()
    for lin in content:
        tmp=lin.split("\t")
        a0=int(tmp[0])
        a1=int(tmp[1])
        m1=max(m1,a0)
        m2=max(m2,a1)
        s1.add(a0)
        s2.add(a1)
        # lis1.append(a0)
        # lis2.append(a1)
        if a0 not in d1:
            continue
        if a1 not in d2:
            continue
        # for i in range(len(lis1)):
        dd.append((d1[a0],d2[a1]))
        # print(f"{d1[lis1[i]]},{d2[lis2[i]]}",file=ftrain)

dd=sorted(dd)
with open("lastfm.test","w") as ftest:
    for i in range(len(dd)):
        print(f"{dd[i][0]},{dd[i][1]}",file=ftest)
    