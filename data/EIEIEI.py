import sys,os
import numpy as np
print("SG")
# Inclusion-Exclusion
maxU=0
maxI=0
point=list()
with open("gowalla.train","r") as fin:
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
# print(du)
# print(adj)
from scipy import sparse 
print("SS")
adj2=sparse.csr_matrix(adj)*sparse.csr_matrix(adj)
print("&&")
du2=np.sum(adj2,axis=0)
# adj2=np.matmul(adj,adj)
print("@@")
adj3=adj2*sparse.csr_matrix(adj)
# adj3=np.matmul(adj2,adj)
print("##")
# adj4=adj3.multiply(sparse.csr_matrix(adj))

# adj4=np.matmul(adj3,adj)
# print(adj2)

# rank4=adj4
# rank=adj4
# print(du2)
# print(rank4)

# for i in range(n):
#     tt=rank[i,i]-adj2[i,i]*adj2[i,i]-du2[i]+adj2[i,i]
#     tt/=2
#     print(i," self ",tt)

# for (a1,a2) in point:
#     tt=adj3[a1,a2+maxU]-du[a1]-du[a2+maxU]+1
#     print(a1,a2,"edge ",tt)
    
# for (a1,a2) in point:
#     rank4[a1,a2]-=du[a1]+du[a2]-1

#     print(a1,a2,"rank",rank4[a1,a2])
maxt=0
sumt=0
siz=0
zer=0
# index = adj3.indices().t()
# value = adj3.values()
# for i,(a1,a2) in enumerate(index):
#     tt=value[i]-du[a1]-du[a2]+1
#     print(a1," ",a2-maxU," ",tt)
#     maxt=max(maxt,tt)
#     sumt=sumt+tt
#     siz+=1
#     if tt==0:
#         zer+=1
with open("gowaV2.train","w") as ff:
    for (a1,a2) in point:
        tt=adj3[a1,a2+maxU]-du[a1]-du[a2+maxU]+1
        print(a1," ",a2," ",tt,file=ff)
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