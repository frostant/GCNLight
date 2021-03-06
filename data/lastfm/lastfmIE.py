import sys,os
import numpy as np
import time
start =time.time()

print("SG")
# Inclusion-Exclusion
maxU=0
maxI=0
point=list()
trainPath = "lastfm1.train"
testPath = "lastfm1.test"
trainOut="outLastfm.train"
testOut="outLastfm.test"

with open(trainPath,"r") as fin:
    for lin in fin.readlines():
        a1,a2=lin.strip("\n").split(",")[:2]
        a1=int(a1)
        a2=int(a2)
        maxU=max(maxU,a1)
        maxI=max(maxI,a2)
        point.append((a1,a2))
maxU+=1
maxI+=1
n=maxU+maxI
du=np.zeros(n)
adj=np.zeros((n,n))
for (a1,a2) in point:
    adj[a1,a2+maxU]+=1
    adj[a2+maxU,a1]+=1
    du[a1]+=1
    du[a2+maxU]+=1
print(du)
print(adj)
adj2=np.matmul(adj,adj)
adj3=np.matmul(adj2,adj)
adj4=np.matmul(adj3,adj)
print(adj2)
du2=np.sum(adj2,axis=0)
rank4=adj4
rank=adj4
print(du2)
print(rank4)

for i in range(n):
    tt=rank[i,i]-adj2[i,i]*adj2[i,i]-du2[i]+adj2[i,i]
    tt/=2
    # print(i," self ",tt)

# with open(trainOut,"w") as fwi:
#     for (a1,a2) in point:
#         tt=adj3[a1,a2+maxU]-du[a1]-du[a2+maxU]+1
#         print(a1," ",a2," ",tt,file=fwi)
    
# for (a1,a2) in point:
#     rank4[a1,a2]-=du[a1]+du[a2]-1

#     print(a1,a2,"rank",rank4[a1,a2])
maxt=0
sumt=0
siz=0
zer=0
with open(trainOut,"w") as fwi:
    for (a1,a2) in point:
        tt=adj3[a1,a2+maxU]-du[a1]-du[a2+maxU]+1
        print(a1," ",a2," ",tt,file=fwi)
        maxt=max(maxt,tt)
        sumt=sumt+tt
        siz+=1
        if tt==0:
            zer+=1
meant=sumt/siz

print("maxt",maxt)
print("sumt",sumt)
print("siz",siz)
print("meamt",meant)
print("zer",zer)

maxt=0
sumt=0
siz=0
zer=0

#中间写上代码块
end = time.time()
print('Running time: %s Seconds'%(end-start))

with open(testPath,"r") as fout:
    with open(testOut,"w") as fwo:
        for lin in fout.readlines():
            a1,a2=lin.strip("\n").split(",")[:2]
            a1=int(a1)
            a2=int(a2)
            # maxU=max(maxU,a1)
            # maxI=max(maxI,a2)
            # point.append((a1,a2))
            if adj[a1,a2+maxU]:
                print("Chongbian",a1,a2)
                exit(0)
            tt=adj3[a1,a2+maxU]
            print(a1," ",a2," ",tt,file=fwo)
            maxt=max(maxt,tt)
            sumt=sumt+tt
            siz+=1
            if tt==0:
                zer+=1
meant=sumt/siz    

print("maxt",maxt)
print("sumt",sumt)
print("siz",siz)
print("meamt",meant)
print("zer",zer)

# maxt 302.0
# sumt 1240324.0
# siz 42135
# meamt 29.43690518571259
# zer 1730

# // 6:ans 1594
# // 8:ans 4
# // -1:132