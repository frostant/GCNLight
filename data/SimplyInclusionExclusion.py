import sys,os
import numpy as np
import torch
from scipy import sparse
print("SG")
# Inclusion-Exclusion
maxU=0
maxI=0
point=list()
# ll=list()
# aa=np.zeros((1000000,1000000))
# print(aa)
# # lim=10000000000
# # for i in range(lim):
# #     ll.append(lim)
# exit(0)
path="gowalla.train"
# path="lastfm.train"
# path="small.txt"
print(path)
with open(path,"r") as fin:
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
print("SSSS")
index1=list()
index2=list()
index = list()
data=list()
mat=torch.zeros((n,n))
for (a1,a2) in point:
    adj[a1,a2+maxU]+=1
    adj[a2+maxU,a1]+=1
    du[a1]+=1
    du[a2+maxU]+=1
    index1.append(a1)
    index1.append(a2+maxU)
    index2.append(a2+maxU)
    index2.append(a1)
    mat[a1,a2+maxU]=1
    mat[a2+maxU,a1]=1
    
    # index.append((a1,a2+maxU))
    # index.append((a2+maxU,a1))
    data.append(1)
    data.append(1)
print("@#@#%#@")

index1=torch.LongTensor(index1)
index2=torch.LongTensor(index2)
index=torch.stack([index1,index2],dim=0)
# index=torch.
data=torch.FloatTensor(data)
sprmat=torch.sparse.FloatTensor(index,data,torch.Size([n,n]))
# mat = sprmat.to_dense()
dim=1024
matList=list()
print(n," # ",int(n/dim))
for i in range(int(n/dim)):
    matList.append(mat[:,i*dim:(i+1)*dim])

mat2List=list()
for tmat in matList:
    mat2List.append(torch.sparse.mm(sprmat,tmat))
    
print("mats2")
mat3List=list()
for tmat in mat2List:
    mat3List.append(torch.sparse.mm(sprmat,tmat))
print("mats3")
exit(0)
mat2=torch.sparse.mm(sprmat, sprmat)
print(mat2)
print("mat2")
mat3=torch.sparse.mm(sprmat, mat2)
print(mat3)
mat4=torch.sparse.mm(sprmat, mat3)
print(mat4)
exit(0)    
spr = sparse.coo_matrix((data, (index1,index2)), shape=(n, n))
    
print(du)
print(adj)
# sparse_adj= 
# adj2=np.dot(adj,adj)
adj2=spr.dot(spr)
print(adj2)
print("adj2")
adj3=adj2.dot(spr)
# adj3=np.dot(adj2,adj)
# 改成sparse试试看吧
print(adj3.todense())
adj4=adj3.dot(spr)
print(adj4.todense())
# adj4=np.dot(adj3,adj)
du2=np.sum(adj2,axis=0)
# rank4=adj4
# rank=adj4
print(du2)
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
for i,(a1,a2) in enumerate(point):
    if i%50000==0: 
        print("doinig",i)
    tt=adj3[a1,a2+maxU]-du[a1]-du[a2+maxU]+1
    print(a1," ",a2," ",tt)
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