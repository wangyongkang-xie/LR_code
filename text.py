# -*- coding: utf-8 -*-
"""
@Time ： 2020/12/17 19:10
@Auth ： Wang Yongkang
@File ：text.py
@IDE ：PyCharm
@Motto：ABC(Always Be Coding)

"""
# import os
#
# for i,j in zip((1,2),(3,4)):
#     print(i,j)
#
# a = os.path.join('c/tmp','init_model.md')
# print(a)
import argparse
import os

import torch
import numpy as np

# a = torch.Tensor([[[1, 2], [3, 4]]])
# print(torch.cuda.is_available())
# c = torch.Tensor([1,2,3,4])
# # a.size()
# b = a.view(-1)
# d = []
# d.append(b.numpy())
# d.append(c.numpy())
# # print(d)
# # d = torch.from_numpy(np.array(d))
# e = torch.Tensor(d)
# print(e)
# torch.Size([1, 3, 2])
# parser = argparse.ArgumentParser()
# parser.add_argument('--aggregation', type=str,
#                     choices=["normal_atten", "atten", "rule_out", "TrimmedMean", "Krum", "GeoMed"],
#                     default="Krum")
# parser.add_argument("--batch_size", type=int, default=100)
# parser.add_argument("--local_epoch", type=int, default=1)
# parser.add_argument("--select_ratio", type=float, default=1)  # 选取20个client
# parser.add_argument("--attack_mode", type=int, default=3)
# parser.add_argument("--attack_ratio", type=float, default=0.1)
# parser.add_argument("--beta", type=float, default=0.05)
import torch

# x = torch.FloatTensor([[1., 2.]])
# w1 = torch.FloatTensor([[2.], [1.]])
# w2 = torch.FloatTensor([3.])
# w1.requires_grad = True
# w2.requires_grad = True
#
# d = torch.matmul(x, w1)
# f = torch.matmul(d, w2)
# d.data = 1  # 因为这句, 代码报错了 RuntimeError: one of the variables needed for gradient computation has been modified by an
# # inplace operation
#
# f.backward()
# parser.add_argument("--del_num", type=int, default=2)
# args = parser.parse_args()
# aggregation = args.aggregation
# print(aggregation)
# a = [1,2,3,4,5,6]
# del a[[0,1,2,]]
# print(a)
# print(torch.cuda.get_device_name(0))
# print (os.path.abspath(os.path.join(os.getcwd(), "..")))
import hdmedians as hdm

# data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# data_1 = torch.tensor(data)
# # # data_2 = data.flatten()
# data_2 = data_1.cuda()
# min_data = min(data_2[:, 1])
# max_data = max(data_2[:, 1])
# dis = (max_data - min_data) / 5
# dis = dis.int()
# lis_data = [[] for _ in range(5)]
# # lis_data[1].append(2)
# lis_data = torch.tensor(lis_data)
# lis_data[1].append(5)
# data = data.float()
# data_1 = torch.median(data, dim=0)
# print(-2+data)
# dic = {1:2,3:3,4:2}
# print(sorted(dic, key=lambda d: d[1], reverse=False))
# nums = [1,2,3,4]
# import numpy as np
#
# # for i, j in enumerate(dic.items()):
# #     print(i)
# # # 均值
# # print(np.mean(list(dic.values())))
# # 中位数
# # np.median(nums)
dic = {0: 14494.568781676579,
       1: 15333.110340341089,
       2: 15331.580260897093,
       3: 15510.71502004608,
       4: 14241.074248047971,
       5: 15528.170693364036,
       6: 15115.938982167781,
       7: 15508.522489140243,
       8: 15504.184537980727,
       9: 14939.495702296657,
       10: 15406.202952779635,
       11: 15817.2142002082,
       12: 15738.150345417178,
       13: 15839.888520302091,
       14: 15566.837463802773,
       15: 15434.87730806485,
       16: 14550.110657364003,
       17: 15686.77631768636,
       18: 14497.707136053386,
       19: 15563.330107321894}
sort_data = sorted(dic.items(), key=lambda d:d[1],reverse=False)
for key,value in sort_data:
       print(key)
print(sort_data)
# for m, n in dic.items():
#     k = np.log(n)
#     print(k)
#     dic[m] = k
# mean_data = torch.tensor(list(dic.values())).cuda()
# # device = mean_data.is_cuda()
# # mean_data = dic.values()
# device = mean_data.device
# print(device)
import torch.nn.functional as F
# p1 = torch.tensor([1, 2, 3])
# p1 = p1.view(1,-1)
# p1 = p1.float()
# p2 = torch.tensor([4, 5, 6])
# p2 = p2.view(1,-1)
# p2 = p2.float()
# lis = F.pairwise_distance(p1, p2)
# dis = F.pairwise_distance(lis.view(-1,1), torch.zeros_like(lis).view(-1,1))
# lis = F.pairwise_distance([F.pairwise_distance(x1, x2) for x1, x2 in zip(p1, p2)], torch.zeros_like(p1))
# print(lis)
# a = np.array([1,2,3])
# np.delete(a,[1])
# print(a)
# print(27 ** (0.5))
# for i, j in dic.items():
#     if j < mean_data:
#         print(i)
# import numpy as np
# from sklearn.cluster import KMeans
# data = np.array(list(dic.values())).reshape(-1,1)
# # print(data)
# #假如我要构造一个聚类数为3的聚类器
# estimator = KMeans(n_clusters=2)#构造聚类器
# estimator.fit(data)#聚类
# label_pred = estimator.labels_ #获取聚类标签
# print(label_pred)

# a = np.eye(10)
# print(a)
import sklearn.metrics.pairwise as smp
# data = np.array([1,1,4,5,2,3,5,6])
# data = data.reshape(4,2)
# for i, j in enumerate(data):
#     print(i)
# print(data)
# # cs = smp.cosine_similarity(data.reshape(-1,1))
# grad_len = np.array(data.shape).prod()
# data = np.reshape(data,(grad_len))
# print(data)
# print(cs)
from scipy import optimize
# import numpy as np
# #
# # # 确定c,A,b,Aeq,beq
# #
# # # c = np.array([3, 1, 3])
# # # A = np.array([[-1, 2, 1], [0, 4, -3], [1, -3, 2], [1, 0, 0],[0, -1, 0]])
# # # b = np.array([4, 2, 3, 5, -3])
# # # # Aeq = np.array([[1, 1, 1]])
# # # # beq = np.array([7])
# # #
# # # # 求解
# # # res = optimize.linprog(-c, A, b)
# # # print(res)
# A = np.matrix([[1,0],[0,1]])
# # B = np.linalg.inv(A)
# # # C = np.array([344, 56])
# # D = np.sum(A)
# B = A.I
# print(B)
