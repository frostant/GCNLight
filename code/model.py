"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Define models here
"""
import scipy.sparse as sp
import world
import torch
from dataloader import BasicDataset
from torch import nn
import numpy as np
import logging
from utils import onePrint


class BasicModel(nn.Module):    
    def __init__(self):
        super(BasicModel, self).__init__()
    
    def getUsersRating(self, users):
        raise NotImplementedError
    
class PairWiseModel(BasicModel):
    def __init__(self):
        super(PairWiseModel, self).__init__()
    def bpr_loss(self, users, pos, neg):
        """
        Parameters:
            users: users list 
            pos: positive items for corresponding users
            neg: negative items for corresponding users
        Return:
            (log-loss, l2-loss)
        """
        raise NotImplementedError
    
class PureMF(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(PureMF, self).__init__()
        self.num_users  = dataset.n_users
        self.num_items  = dataset.m_items
        self.latent_dim = config['latent_dim_rec']
        self.f = nn.Sigmoid()
        self.__init_weight()
        
    def __init_weight(self):
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        print("using Normal distribution N(0,1) initialization for PureMF")
        
    def getUsersRating(self, users):
        users = users.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item.weight
        scores = torch.matmul(users_emb, items_emb.t())
        return self.f(scores)
    
    def bpr_loss(self, users, pos, neg):
        users_emb = self.embedding_user(users.long())
        pos_emb   = self.embedding_item(pos.long())
        neg_emb   = self.embedding_item(neg.long())
        pos_scores= torch.sum(users_emb*pos_emb, dim=1)
        neg_scores= torch.sum(users_emb*neg_emb, dim=1)
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        reg_loss = (1/2)*(users_emb.norm(2).pow(2) + 
                          pos_emb.norm(2).pow(2) + 
                          neg_emb.norm(2).pow(2))/float(len(users))
        return loss, reg_loss
        
    def forward(self, users, items):
        users = users.long()
        items = items.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item(items)
        scores = torch.sum(users_emb*items_emb, dim=1)
        return self.f(scores)

class LightGCN(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset : dataloader.BasicDataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.num_users  = self.dataset.n_users
        self.num_items  = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        # self.W = nn.Linear(self.latent_dim,self.latent_dim, bias=True)
        # torch.nn.LayerNorm
        
        # self.W = nn.Linear((self.num_items+self.num_users),(self.num_items+self.num_users), bias=True)
        # self.weight1=torch.nn.Linear((self.num_items+self.num_users),(self.num_items+self.num_users))
    
#     def forward(self, input, adj):
#         support = self.W(input)
#         output = torch.spmm(adj, support)

# class GCN(nn.Module):
#     """
#     A Two-layer GCN.
#     """
#     def __init__(self, nfeat, nhid, nclass, dropout):
#         super(GCN, self).__init__()

#         self.gc1 = GraphConvolution(nfeat, nhid)
#         self.gc2 = GraphConvolution(nhid, nclass)
#         self.dropout = dropout

#     def forward(self, x, adj, use_relu=True):
#         x = self.gc1(x, adj)
#         if use_relu:
#             x = F.relu(x)
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = self.gc2(x, adj)
#         return x


        if self.config['pretrain'] == 0:
#             nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
#             nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
#             print('use xavier initilizer')
# random normal init seems to be a better choice when lightGCN actually don't use any non-linear activation function
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            world.cprint('use NORMAL distribution initilizer')
        else:
            self.embedding_user.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
            print('use pretarined data')
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()
        print(f"lgn is already to go(dropout:{self.config['dropout']})")

        # print("save_txt")
    def __dropout_x(self, x, keep_prob):
        # keep prob 保留神经元的概率, 类似1-dropout的rate
        # graph是sparse,他取出所有value,dropout部分节点,剩余节点扩大prob,以实现归一化
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob #!! 归一化
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g
    
    def __dropout_x__v1(self, x, m_user, m_item,keep_prob):
        # 方法1：点dropout
        size = x.size()
        index = x.indices().t()
        values = x.values()
        # print(index.shape)
        # degree = torch.sparse.sum(x,dim=1)
        # degree = degree.to_dense()
        # zero = torch.zeros(degree.shape).to(world.device)
        # degree = torch.where(degree>1e-9,1/degree,zero)
        
        random_index = torch.rand(m_user+m_item) + keep_prob
        random_index = random_index.int().bool().to(world.device)
        # uu = map(lambda x,y:min(x,y), index[:,0])
        # logging.warning(index.shape)
        # min_idx = torch.min(index, dim=1)
        # # logging.warning(min_idx.shape)
        # min_idx = random_index[min_idx]
        # # logging.warning(min_idx.shape)
        # indexT=index.t()
        # diff = torch.sub(indexT[0],indexT[1])
        # new_index = torch.where(diff!=0,min_idx,0).bool()

        # indexT=index.t()
        # diff = torch.sub(indexT[0],indexT[1])
        # adj_idx = torch.where(diff!=0,True,False)
        # self_idx = torch.bitwise_not(adj_idx)
        # index_a = index[adj_idx]
        # logging.warning(index_a.shape)
        # min_idx = torch.min(index_a, dim=1).values
        # logging.warning(min_idx.shape)
        # min_idx = random_index[min_idx]
        
        # # new_index = torch.bitwise_or(self_idx,min_idx)
        # indexa = index[min_idx]
        # indexs = index[self_idx]
        # index = torch.cat((indexa,indexs),0) 
        # # index = index[new_index] # !!!! error The shape of the mask [2474518] at index 0 does not match the shape of the indexed tensor [2544234, 2] at index 0
        
        # valuesa = values[min_idx]
        # valuess = values[self_idx]
        # values = torch.cat((valuesa,valuess))
        # values = values/keep_prob

        indexT=index.t()
        diff = torch.sub(indexT[0],indexT[1])
        adj_idx = torch.where(diff!=0,True,False)
        self_idx = torch.bitwise_not(adj_idx)
        # index_a = index[adj_idx]
        # logging.warning(index_a.shape)

        min_idx = torch.min(index, dim=1).values #G
        # print(min_idx.device)
        
        # logging.warning(min_idx.shape)
        min_idx = random_index[min_idx] # G
        # adj_idx = adj_idx.to("cpu")
        # print(adj_idx.device)
        min_idx = torch.bitwise_and(adj_idx, min_idx) #G 
        # print(min_idx.device)

        # new_index = torch.bitwise_or(self_idx,min_idx)
        indexa = index[min_idx]
        indexs = index[self_idx]
        index = torch.cat((indexa,indexs),0) 
        # index = index[new_index] # !!!! error The shape of the mask [2474518] at index 0 does not match the shape of the indexed tensor [2544234, 2] at index 0
        
        valuesa = values[min_idx]
        valuess = values[self_idx]
        values = torch.cat((valuesa,valuess))
        values = values/keep_prob

        # new_index = torch.where(diff!=0,min_idx,0).bool()

        # new_index = random_index[index[i][1]]

        # index = index[new_index] # !!!! error The shape of the mask [2474518] at index 0 does not match the shape of the indexed tensor [2544234, 2] at index 0
        # values = values[new_index]/keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g
        

    def __dropout_x__v2(self,x,keep_prob):
        # (1/du_1+1/du_2)/2*prob
        pass
    
    def __dropadd_v1(self,x, m_user, m_item,keep_prob):
        onePrint("dropaddV1")
        # LastFm Sparsity : 0.00620355984113386
        # # global 加边
        keep_prob=min(keep_prob,0.0062)
        # size = x.size()
        # index = x.indices().t()
        # values = x.values()
        # random_index = torch.rand(len(values)) + keep_prob
        # random_index = random_index.int().bool()
        # index = index[random_index]
        # values = values[random_index]/keep_prob #!! 归一化
        # g = torch.sparse.FloatTensor(index.t(), values, size)
        # return g
        x=x.to("cpu")
        size = x.size()
        index = x.indices().t()
        values = x.values()
        # indexT=index.t()
        # diff = torch.sub(indexT[0],indexT[1])
        # adj_idx = torch.where(diff!=0,True,False)
        # self_idx = torch.bitwise_not(adj_idx)
        # newValues=torch.ones(len(values)).float()

        random_index = torch.rand(m_user+m_item) + keep_prob / 2
        # random_index = random_index.int().bool().to(world.device) # to?!

        addidx=[]
        for i in range(m_user):
            for j in range(m_item):
                if(random_index[i]+random_index[j+m_user]>1.0):
                    addidx.append((i,j))
        npAdd=np.array(addidx)
        # oneAdd = np.ones(len(addidx))

        newIndex=np.concatenate((index,npAdd),axis=0)
        # np.concatenate()
        newIndex = torch.LongTensor(newIndex)
        print(newIndex.shape)
        newValues=torch.ones(len(values)+len(addidx)).int()
        print(newValues.shape)
        newG=torch.sparse.IntTensor(newIndex.t(),newValues,size)
        dense=newG.to_dense()
        D=torch.sum(dense,dim=1).float()
        D[D==0.]=1.
        D_sqrt = torch.sqrt(D).unsqueeze(dim=0)
        dense = dense/D_sqrt
        dense = dense/D_sqrt.t()
        index = dense.nonzero()
        data  = dense[dense >= 1e-9]
        assert len(index) == len(data)
        g = torch.sparse.FloatTensor(index.t(), data, size)
        g = g.coalesce().to(world.device)
        return g

    def __dropadd_v2(self,x, m_user, m_item,keep_prob):
        onePrint("dropaddV2")
        # LastFm Sparsity : 0.00620355984113386
        # # global 加边
        keep_prob=min(keep_prob,0.0062)
        onePrint(str(keep_prob))
        x=x.to("cpu")
        size = x.size()
        index = x.indices().t()
        values = x.values()
        
        
        random_index = torch.rand(m_user+m_item) + keep_prob / 2
        randUser = (torch.rand(int(keep_prob*m_user*m_item)) * m_user).long()
        randItem = (torch.rand(int(keep_prob*m_user*m_item)) * m_item).long()
        newIndex = torch.stack([randUser,randItem], dim=0)
        newIndex = torch.LongTensor(newIndex)
        oneLen=len(randItem)+len(values)
        # print(newIndex.shape)
        # print(index.t().shape)
        newIndex = torch.cat((index.t(),newIndex), dim=1)
        
        # random_index = random_index.int().bool().to(world.device) # to?!

        # addidx=[]
        # for i in range(m_user):
        #     for j in range(m_item):
        #         if(random_index[i]+random_index[j+m_user]>1.0):
        #             addidx.append((i,j))
        # npAdd=np.array(addidx)
        # # oneAdd = np.ones(len(addidx))

        # newIndex=np.concatenate((index,npAdd),axis=0)
        # # np.concatenate()
        # newIndex = torch.LongTensor(newIndex)
        # print(newIndex.shape)
        newValues=torch.ones(oneLen).int()
        # print(newValues.shape)
        newG=torch.sparse.IntTensor(newIndex,newValues,size)
        newG = newG.coalesce()
        dense=newG.to_dense()
        D=torch.sum(dense,dim=1).float()
        D[D==0.]=1.
        D_sqrt = torch.sqrt(D).unsqueeze(dim=0)
        dense = dense/D_sqrt
        dense = dense/D_sqrt.t()
        index = dense.nonzero()
        data  = dense[dense >= 1e-9]
        assert len(index) == len(data)
        g = torch.sparse.FloatTensor(index.t(), data, size)
        g = g.coalesce().to(world.device)
        return g


    def __dropadd_v3(self,x, m_user, m_item,trainSize, keep_prob):
        onePrint("dropaddV3"+str(keep_prob))
        # LastFm Sparsity : 0.00620355984113386
        # # global 加边
        # keep_prob=min(keep_prob,trainSize)
        addNum=int(trainSize*keep_prob)
        x=x.to("cpu")
        size = x.size()
        index = x.indices().t()
        values = x.values()
        
        random_index = torch.rand(m_user+m_item) + keep_prob / 2
        randUser = (torch.rand(addNum) * m_user).long()
        randItem = (torch.rand(addNum) * m_item).long()
        newIndex = torch.stack([randUser,randItem], dim=0)
        newIndex = torch.LongTensor(newIndex)
        oneLen=len(randItem)+len(values)
        # print(newIndex.shape)
        # print(index.t().shape)
        newIndex = torch.cat((index.t(),newIndex), dim=1)
        
        newValues=torch.ones(oneLen).int()
        # print(newValues.shape)
        newG=torch.sparse.IntTensor(newIndex,newValues,size)
        newG = newG.coalesce()
        dense=newG.to_dense()
        D=torch.sum(dense,dim=1).float()
        D[D==0.]=1.
        D_sqrt = torch.sqrt(D).unsqueeze(dim=0)
        dense = dense/D_sqrt
        dense = dense/D_sqrt.t()
        index = dense.nonzero()
        data  = dense[dense >= 1e-9]
        assert len(index) == len(data)
        g = torch.sparse.FloatTensor(index.t(), data, size)
        g = g.coalesce().to(world.device)
        return g

    def __dropadd_v4(self,x, lowUser,lowItem,trainSize, keep_prob):
        onePrint("dropaddV4")
        # LastFm Sparsity : 0.00620355984113386
        # # 孤立点 加边
        # keep_prob=min(keep_prob,trainSize)
        addNum=int(trainSize*keep_prob/4)
        # 除4是因为我只对平均值以下的节点进行添加
        x=x.to("cpu")
        size = x.size()
        index = x.indices().t()
        values = x.values()
        
        # random_index = torch.rand(m_user+m_item) + keep_prob / 2
        # print(lowUser.shape)
        randUserIdx = (torch.rand(addNum) * len(lowUser)).long()
        randItemIdx = (torch.rand(addNum) * len(lowItem)).long()
        # print(randUserIdx)
        randUser=lowUser[randUserIdx]
        randItem=lowItem[randItemIdx]
        newIndex = torch.stack([randUser,randItem], dim=0)
        newIndex = torch.LongTensor(newIndex)
        oneLen=len(randItem)+len(values)
        # print(newIndex.shape)
        # print(index.t().shape)
        newIndex = torch.cat((index.t(),newIndex), dim=1)
        
        newValues=torch.ones(oneLen).int()
        # print(newValues.shape)
        newG=torch.sparse.IntTensor(newIndex,newValues,size)
        newG = newG.coalesce()
        dense=newG.to_dense()
        D=torch.sum(dense,dim=1).float()
        D[D==0.]=1.
        D_sqrt = torch.sqrt(D).unsqueeze(dim=0)
        dense = dense/D_sqrt
        dense = dense/D_sqrt.t()
        index = dense.nonzero()
        data  = dense[dense >= 1e-9]
        assert len(index) == len(data)
        g = torch.sparse.FloatTensor(index.t(), data, size)
        g = g.coalesce().to(world.device)
        return g

    def __dropadd_v000(self,x,keep_prob):
        # 孤立点 加边
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob #!! 归一化
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g
    
    # @timer
    def create_adj_mat(self, is_subgraph=False, aug_type=0):
        n_nodes = self.n_users + self.n_items
        if is_subgraph and aug_type in [0, 1, 2] and self.ssl_ratio > 0:
            # data augmentation type --- 0: Node Dropout; 1: Edge Dropout; 2: Random Walk
            if aug_type == 0:
                drop_user_idx = randint_choice(self.n_users, size=self.n_users * self.ssl_ratio, replace=False)
                drop_item_idx = randint_choice(self.n_items, size=self.n_items * self.ssl_ratio, replace=False)
                indicator_user = np.ones(self.n_users, dtype=np.float32)
                indicator_item = np.ones(self.n_items, dtype=np.float32)
                indicator_user[drop_user_idx] = 0.
                indicator_item[drop_item_idx] = 0.
                diag_indicator_user = sp.diags(indicator_user)
                diag_indicator_item = sp.diags(indicator_item)
                R = sp.csr_matrix(
                    (np.ones_like(self.training_user, dtype=np.float32), (self.training_user, self.training_item)), 
                    shape=(self.n_users, self.n_items))
                R_prime = diag_indicator_user.dot(R).dot(diag_indicator_item)
                (user_np_keep, item_np_keep) = R_prime.nonzero()
                ratings_keep = R_prime.data
                tmp_adj = sp.csr_matrix((ratings_keep, (user_np_keep, item_np_keep+self.n_users)), shape=(n_nodes, n_nodes))
            if aug_type in [1, 2]:
                keep_idx = randint_choice(len(self.training_user), size=int(len(self.training_user) * (1 - self.ssl_ratio)), replace=False)
                user_np = np.array(self.training_user)[keep_idx]
                item_np = np.array(self.training_item)[keep_idx]
                ratings = np.ones_like(user_np, dtype=np.float32)
                tmp_adj = sp.csr_matrix((ratings, (user_np, item_np+self.n_users)), shape=(n_nodes, n_nodes))
        else:
            user_np = np.array(self.training_user)
            item_np = np.array(self.training_item)
            ratings = np.ones_like(user_np, dtype=np.float32)
            tmp_adj = sp.csr_matrix((ratings, (user_np, item_np+self.n_users)), shape=(n_nodes, n_nodes))
        adj_mat = tmp_adj + tmp_adj.T

        # pre adjcency matrix
        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)
        # print('use the pre adjcency matrix')

        return adj_matrix
    

    def __dropout(self, keep_prob):
        if self.A_split: # ?把邻接矩阵裁成n行
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph
    
    def _dropout__v1(self, keep_prob):
        if self.A_split:
            pass # TODO 
        else :
            graph = self.__dropout_x__v1(self.Graph, self.num_users, self.num_items, keep_prob)
        return graph
    
    def computer(self):
        # 手写正向传播
        # 
        """
        propagate methods for lightGCN
        """       
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]

        # print("allemb",all_emb.shape)
        if self.config['dropout']:
            onePrint("dropout")
            if self.training:
                # g_droped = self.__dropadd_v2(self.Graph, self.num_users, self.num_items, self.keep_prob)
                # g_droped = self.__dropadd_v3(self.Graph, self.num_users, self.num_items,self.dataset.trainDataSize, self.keep_prob)
                g_droped = self.__dropadd_v4(self.Graph, self.dataset.lowUser, self.dataset.lowItem,self.dataset.trainDataSize, self.keep_prob)
            else:
                g_droped = self.Graph        
        else:
            g_droped = self.Graph    
        # 得到dropout后的 图
        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                # hidden=all_emb.clone() # delete
                # out=self.W(hidden)
                # all_emb = torch.sparse.mm(g_droped, out)
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        #print(embs.size())
        light_out = torch.mean(embs, dim=1)
        # 把每层的embedding取平均作为其embedding
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items
    
    def getUsersRating(self, users):
        # 得到该user(是个ID)和所有item的得分
        all_users, all_items = self.computer()
        all_users = all_users[:self.dataset.old_n_user]
        users_emb = all_users[users.long()]
        items_emb = all_items[:self.dataset.old_m_item]
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating
    
    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        # 把本次的embedding和原来的embedding 都返回
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego
    
    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb, 
        userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
                         posEmb0.norm(2).pow(2)  +
                         negEmb0.norm(2).pow(2))/float(len(users))
        # 正则化是拿原来的embedding
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        
        return loss, reg_loss
       
    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        # print('forward')
        #all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma     = torch.sum(inner_pro, dim=1)
        return gamma
