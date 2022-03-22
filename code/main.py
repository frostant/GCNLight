# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

import world
import utils
from world import cprint
import torch
import numpy as np
from tensorboardX import SummaryWriter
import time
import Procedure
from matplotlib import pyplot as plt
from os.path import join
# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import register
from register import dataset
from utils import onePrint
# torch.set_printoptions(profile="full")
# torch.set_printoptions(profile="default")
onePrint("Start")
Recmodel = register.MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)
bpr = utils.BPRLoss(Recmodel, world.config)
print("num user:",Recmodel.num_users)
print("num item:",Recmodel.num_items)
print("graph:",Recmodel.Graph[12][12])
weight_file = utils.getFileName()
print(f"load and save to {weight_file}")
if world.LOAD:
    try:
        Recmodel.load_state_dict(torch.load(weight_file,map_location=torch.device('cpu')))
        world.cprint(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")
# 加载参数

Neg_k = 1

# init tensorboard
if world.tensorboard:
    # tensorboard还不是很会用
    w : SummaryWriter = SummaryWriter(
                                    join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
                                    )
else:
    w = None
    world.cprint("not enable tensorflowboard")

from time import time
breakNum=30
tmp = [0.,0.,0]
recall_lis=[]
ndcg_lis=[]
precision_lis=[]
index_lis=[]
loss_lis=[]
try:
    for epoch in range(world.TRAIN_epochs):
        if epoch %10 == 0:
            start = time()
            cprint("[TEST]")
            tmp,result=Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'])
            during = time()-start
            print(f"{during:.2f}")
            print(tmp)
            recall_lis.append(result["recall"])
            ndcg_lis.append(result["ndcg"])
            precision_lis.append(result["precision"])
            if tmp[2]>breakNum:
                print("Beak because no update")
                break

        start = time()
        output_information,loss = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k,w=w)
        loss_lis.append(loss)
        during = time()-start
        # print(f"{during:.2f}")
        if epoch %10 == 0:
            print(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information}')
        torch.save(Recmodel.state_dict(), weight_file)
        # 存参数
        if (epoch+1)%200==0:
            print("Graph change")
            Recmodel.graph=Recmodel.GraphChange()
finally:
    if world.tensorboard:
        w.close()
print(tmp)

from utils import globalSet
print("Global Function",globalSet)
printParam=False
if printParam:
    with open("param.txt","w") as fout:
        torch.set_printoptions(profile="full")
        for parameters in Recmodel.parameters():#打印出参数矩阵及值
            print(parameters,file=fout)

index_lis=range(len(recall_lis))
plt.plot(index_lis,recall_lis)
plt.plot(index_lis,ndcg_lis)
plt.plot(index_lis,precision_lis)
plt.savefig('./导出的图片.png')
plt.figure(2)
loss_idx=range(len(loss_lis))
plt.plot(loss_idx,loss_lis)

plt.savefig('./导出的图片loss.png')