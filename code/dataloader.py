"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Shuxian Bi (stanbi@mail.ustc.edu.cn),Jianbai Ye (gusye@mail.ustc.edu.cn)
Design Dataset here
Every dataset's index has to start at 0
"""
from io import SEEK_CUR
import os
from os.path import join
import sys

from numpy.core.defchararray import add
from numpy.core.fromnumeric import mean
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import world
from world import cprint
from time import time
from utils import onePrint
class BasicDataset(Dataset):
    def __init__(self):
        print("init dataset")
    
    @property
    def n_users(self):
        raise NotImplementedError
    
    @property
    def m_items(self):
        raise NotImplementedError
    
    @property
    def trainDataSize(self):
        raise NotImplementedError
    
    @property
    def testDict(self):
        raise NotImplementedError
    
    @property
    def allPos(self):
        raise NotImplementedError
    
    def getUserItemFeedback(self, users, items):
        raise NotImplementedError
    
    def getUserPosItems(self, users):
        raise NotImplementedError

    def getOldUserPosItems(self, users):
        
        raise NotImplementedError
    
    def getUserNegItems(self, users):
        """
        not necessary for large dataset
        it's stupid to return all neg items in super large dataset
        """
        raise NotImplementedError
    
    def getSparseGraph(self):
        """
        build a graph in torch.sparse.IntTensor.
        Details in NGCF's matrix form
        A = 
            |I,   R|
            |R^T, I|
        """
        raise NotImplementedError

# from collections import Counter
# import torch
# import numpy as np
# a = [1, 2, 3, 1, 1, 2]
# a=np.array(a)
# result = Counter(a)
# print (result)

    def add_virtual_V2(self,data, user,item,limitU,limitI):
        # 
        onePrint("addVirtualV2")
        from collections import Counter  
        print(data.shape)
        # uniUser = np.unique(user)
        # n=len(uniUser)
        add_node=[]
        coutUser=Counter(user)
        coutUser=sorted(coutUser.items(),key=lambda x:x[1])
        for (idx,sum) in coutUser:
            if sum<=limitU:
                add_node.append((idx,self.m_item)) # 开区间
                self.m_item+=1
            else :
                break
        print("add_Item",len(add_node))
        coutItem=Counter(item)
        coutItem=sorted(coutItem.items(),key=lambda x:x[1])
        for (idx,sum) in coutItem:
            if sum<=limitI:
                add_node.append((self.n_user,idx)) # 开区间
                self.n_user+=1
            else :
                break
        
        # self.n_user = self.old_n_user
        # self.m_item = self.old_m_item
        
        add_node=np.array(add_node)
        print("add_node=",len(add_node))
        # print(data.shape)
        # print(add_node.shape)
        return np.concatenate((data,add_node),axis=0)

def add_virtual_V1(data,limit):
    # 有bug，会把user和第1，2，3，..个item相连,而且应该对item也做类似操作
    onePrint("addVirtualV1")
    user=np.array(data[:,0])
    item=np.array(data[:,1])
    uniUser = np.unique(user)
    n=len(uniUser)
    add_node=[]
    for i in uniUser:
        sum=np.sum(user==i)
        if sum<limit:
            add_node.append((i,n))
            n+=1
    add_node=np.array(add_node)
    print("add_node=",len(add_node))
    # print(data.shape)
    # print(add_node.shape)
    return np.concatenate((data,add_node),axis=0)


class LastFM(BasicDataset):
    """
    Dataset type for pytorch \n
    Incldue graph information
    LastFM dataset
    """
    def __init__(self, path="../data/lastfm"):
        # train or test
        cprint("loading [last fm]")
        self.mode_dict = {'train':0, "test":1}
        self.mode    = self.mode_dict['train']
        # self.n_users = 1892
        # self.m_items = 4489
        _trainData = pd.read_table(join(path, 'data1.txt'), header=None).to_numpy()
        # print(trainData.head())
        testData  = pd.read_table(join(path, 'test1.txt'), header=None).to_numpy()
        # print(testData.head())
        trustNet  = pd.read_table(join(path, 'trustnetwork.txt'), header=None).to_numpy()
        # print(trustNet[:5])
        trustNet -= 1
        _trainData-= 1
        testData -= 1
        # print(_trainData.shape)
        trainData = _trainData[:,:2]
        trainData = trainData.astype("int")
        self.trainUser = np.array(trainData[:,0],dtype="int")
        self.trainItem = np.array(trainData[:,1],dtype="int")
        self.n_user = np.max(self.trainUser)
        self.m_item = np.max(self.trainItem)
        self.n_user+=1
        self.m_item+=1
        print(self.n_user)
        print(self.m_item)
        self.old_n_user=self.n_user
        self.old_m_item=self.m_item
        self.Old_UserItemNet=csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)), shape=(self.old_n_user, self.old_m_item))
        
        self.add_virtual_node = True 
        if self.add_virtual_node:
            # print(trainData.shape)
            trainData=self.add_virtual_V2(trainData,self.trainUser,self.trainItem,12,1)

        self.trustNet  = trustNet
        self.trainData = trainData
        self.testData  = testData
        self.trainUser = np.array(trainData[:,0])
        self.trainUniqueUsers = np.unique(self.trainUser)
        self.trainItem = np.array(trainData[:,1])
        # self.trainDataSize = len(self.trainUser)
        self.testUser  = np.array(testData[:,0])
        self.testUniqueUsers = np.unique(self.testUser)
        self.testItem  = np.array(testData[:,1])
        self.Graph = None
        print(f"LastFm Sparsity : {(len(self.trainUser) + len(self.testUser))/self.n_users/self.m_items}")
        

    #             self.__init_weight()

    # def __init_weight(self):
    
        # (users,users)
        self.socialNet    = csr_matrix((np.ones(len(trustNet)), (trustNet[:,0], trustNet[:,1]) ), shape=(self.n_users,self.n_users))
        # (users,items), bipartite graph
        self.UserItemNet  = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem) ), shape=(self.n_users,self.m_items)) 
        # pre-calculate
        self._allPos = self.getUserPosItems(list(range(self.n_users)))
        # matrix:有和这个user交互的item
        self.allNeg = []
        allItems    = set(range(self.m_items))
        for i in range(self.n_users):
            pos = set(self._allPos[i])
            neg = allItems - pos
            self.allNeg.append(np.array(list(neg)))
        self.__testDict = self.__build_test()
    # users = np.random.randint(0, dataset.n_users, user_num)

    @property
    def n_users(self):
        # return 1892
        return self.n_user
    
    @property
    def m_items(self):
        # return 4489
        return self.m_item
    
    @property
    def trainDataSize(self):
        return len(self.trainUser)
    
    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos

    def getSparseGraph(self):
        if self.Graph is None:
            user_dim = torch.LongTensor(self.trainUser)
            item_dim = torch.LongTensor(self.trainItem)
            
            first_sub = torch.stack([user_dim, item_dim + self.n_users])
            second_sub = torch.stack([item_dim+self.n_users, user_dim])
            # 1 user item 2 item user
            index = torch.cat([first_sub, second_sub], dim=1)
            data = torch.ones(index.size(-1)).int()
            self.Graph = torch.sparse.IntTensor(index, data, torch.Size([self.n_users+self.m_items, self.n_users+self.m_items]))
            dense = self.Graph.to_dense()
            # Graph:
            # # U I 
            # U 0 1
            # I 2 0 
            # 1 is first sub 2 is second sub
            D = torch.sum(dense, dim=1).float()
            D[D==0.] = 1.
            D_sqrt = torch.sqrt(D).unsqueeze(dim=0)
            dense = dense/D_sqrt
            dense = dense/D_sqrt.t()
            index = dense.nonzero()
            data  = dense[dense >= 1e-9]
            # D is 度矩阵
            assert len(index) == len(data)
            self.Graph = torch.sparse.FloatTensor(index.t(), data, torch.Size([self.n_users+self.m_items, self.n_users+self.m_items]))
            self.Graph = self.Graph.coalesce().to(world.device)

            self.lowUser,self.lowItem = self.getLower(D)
        return self.Graph
    
    def getLower(self,degree):
        # meanDegree=(self.trainUser/self.n_users,self.trainUser/self.m_items)
        # meanDegree
        meanUserD = torch.ones(self.n_users)*int(len(self.trainUser)/self.n_users)
        meanItemD = torch.ones(self.m_items)*int(len(self.trainUser)/self.m_items)
        meanDegree = torch.cat([meanUserD,meanItemD]).long()
        meanFlag = degree<=meanDegree
        # print(meanFlag[:15])
        # print(meanFlag[-15:])
        lowUser=[]
        for i in range(self.n_users):
            if meanFlag[i]:
                lowUser.append(i)
        lowItem=[]
        for i in range(self.m_items):
            if meanFlag[i]:
                lowItem.append(i)
        lowUser=torch.LongTensor(lowUser)
        lowItem=torch.LongTensor(lowItem)
        return lowUser,lowItem
    
    def __build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data
    
    def getUserItemFeedback(self, users, items):
        """
        users:
            shape [-1]
        items:
            shape [-1]
        return:
            feedback [-1]
        """
        # print(self.UserItemNet[users, items])
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1, ))
    
    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        # 使用scipy会得到(array([0, 0, 0], dtype=int32), array([0, 1, 2], dtype=int32))
        # 所以取第1维
        # return 二维， 表示第i个user和他交互的item的编号
        return posItems
    
    def getOldUserPosItems(self, users):
        posItems = []
        # print(self.Old_UserItemNet.shape)
        for user in users:
            posItems.append(self.Old_UserItemNet[user].nonzero()[1])
        # 使用scipy会得到(array([0, 0, 0], dtype=int32), array([0, 1, 2], dtype=int32))
        # 所以取第1维
        # return 二维， 表示第i个user和他交互的item的编号
        return posItems
    def getUserNegItems(self, users):
        negItems = []
        for user in users:
            negItems.append(self.allNeg[user])
        return negItems
            
    
    
    def __getitem__(self, index):
        user = self.trainUniqueUsers[index]
        # return user_id and the positive items of the user
        return user
    
    def switch2test(self):
        """
        change dataset mode to offer test data to dataloader
        """
        self.mode = self.mode_dict['test']
    
    def __len__(self):
        return len(self.trainUniqueUsers)

class Loader(BasicDataset):
    """
    Dataset type for pytorch \n
    Incldue graph information
    gowalla dataset
    """

    def __init__(self,config = world.config,path="../data/gowalla"):
        # train or test
        cprint(f'loading [{path}]')
        self.split = config['A_split']
        self.folds = config['A_n_fold']
        self.mode_dict = {'train': 0, "test": 1}
        self.mode = self.mode_dict['train']
        self.n_user = 0
        self.m_item = 0
        train_file = path + '/train.txt'
        test_file = path + '/test.txt'
        self.path = path
        trainUniqueUsers, trainItem, trainUser = [], [], []
        testUniqueUsers, testItem, testUser = [], [], []
        self.traindataSize = 0
        self.testDataSize = 0

        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    trainUniqueUsers.append(uid)
                    trainUser.extend([uid] * len(items))
                    trainItem.extend(items)
                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.traindataSize += len(items)
        # 输入数据可能是一行第一个是user,后面全是item, 或每行一个user-item..  
        # trainUser是所有user,trainItem是所有item 相互对应
        self.trainUniqueUsers = np.array(trainUniqueUsers)
        self.trainUser = np.array(trainUser)
        self.trainItem = np.array(trainItem)
        print("SHape",self.trainUser.shape)

        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    testUniqueUsers.append(uid)
                    testUser.extend([uid] * len(items))
                    testItem.extend(items)
                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.testDataSize += len(items)
        self.m_item += 1
        self.n_user += 1
        self.old_n_user=self.n_user
        self.old_m_item=self.m_item
        self.testUniqueUsers = np.array(testUniqueUsers)
        self.testUser = np.array(testUser)
        self.testItem = np.array(testItem)

        self.add_virtual_node = True 
        if self.add_virtual_node:
            trainData = np.stack((self.trainUser,self.trainItem),axis=1)
            # print(trainData.shape)
            trainData=self.add_virtual_V2(trainData,self.trainUser,self.trainItem,12,1)
            self.trainUser=trainData[:][0]
            self.trainItem=trainData[:][1]
            
        self.Graph = None
        print(f"{self.trainDataSize} interactions for training")
        print(f"{self.testDataSize} interactions for testing")
        print(f"{world.dataset} Sparsity : {(self.trainDataSize + self.testDataSize) / self.n_users / self.m_items}")
        self.trainSparsity=self.trainDataSize/self.n_users/self.m_items
        # (users,items), bipartite graph
        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)), shape=(self.n_user, self.m_item))
        self.Old_UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)), shape=(self.old_n_user, self.old_m_item))
        
        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        self.users_D[self.users_D == 0.] = 1
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.] = 1.
        # pre-calculate
        self._allPos = self.getUserPosItems(list(range(self.n_user)))
        self.__testDict = self.__build_test()
        print(f"{world.dataset} is ready to go")

    @property
    def n_users(self):
        return self.n_user
    
    @property
    def m_items(self):
        return self.m_item
    
    @property
    def trainDataSize(self):
        return self.traindataSize
    
    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos

    def _split_A_hat(self,A):
        A_fold = []
        fold_len = (self.n_users + self.m_items) // self.folds
        for i_fold in range(self.folds):
            start = i_fold*fold_len
            if i_fold == self.folds - 1:
                end = self.n_users + self.m_items
            else:
                end = (i_fold + 1) * fold_len
            A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce().to(world.device))
        return A_fold

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))
        
    def getSparseGraph(self):
        print("loading adjacency matrix")
        if self.Graph is None:
            try:
                pre_adj_mat = sp.load_npz(self.path + '/s_pre_adj_mat.npz')
                print("successfully loaded...")
                norm_adj = pre_adj_mat
            except :
                print("generating adjacency matrix")
                s = time()
                adj_mat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
                adj_mat = adj_mat.tolil()
                R = self.UserItemNet.tolil()
                adj_mat[:self.n_users, self.n_users:] = R
                adj_mat[self.n_users:, :self.n_users] = R.T
                adj_mat = adj_mat.todok()
                # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])
                
                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)
                
                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                end = time()
                print(f"costing {end-s}s, saved norm_mat...")
                sp.save_npz(self.path + '/s_pre_adj_mat.npz', norm_adj)

            if self.split == True:
                self.Graph = self._split_A_hat(norm_adj)
                print("done split matrix")
            else:
                self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
                self.Graph = self.Graph.coalesce().to(world.device)
                print("don't split the matrix")
        return self.Graph

    def __build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    def getUserItemFeedback(self, users, items):
        """
        users:
            shape [-1]
        items:
            shape [-1]
        return:
            feedback [-1]
        """
        # print(self.UserItemNet[users, items])
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1,))

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems

    def getOldUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.Old_UserItemNet[user].nonzero()[1])
        return posItems
    # def getUserNegItems(self, users):
    #     negItems = []
    #     for user in users:
    #         negItems.append(self.allNeg[user])
    #     return negItems
