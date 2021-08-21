# from copy import deepcopy

import numpy as np
import hdmedians as hdm
import torch
from scipy.stats import trim_mean
import sklearn.metrics.pairwise as smp
from sklearn.decomposition import PCA
from model import ServerModel
# import torch.nn.functional as F
class Baseline(object):
    # def __init__(self):
    #
    #     self.device = torch.device("cuda")
    def cal_Normal(self, input_weights, device):

        # input_weights = [input_weights[i] for i in range(0, len(input_weights), 1) if i not in need_del]
        grads = self.aggregate_grads(input_weights, device)
        return grads

    def cal_NoDetect(self, input_weights, device):
        # res = {}

        # input_weights = [input_weights[i] for i in range(0, len(input_weights), 1) if i not in need_del]
        grads = self.aggregate_grads(input_weights, device)
        return grads
        # return res

    def cal_TrimmedMean(self, input_weights, beta=0.05):
        # input_weights: list of client weights
        # input_weights[0]: list of a certain client weights
        # input_weights[i][j]: ndarray of a certain layer of a certain client weights
        res = {}

        for layer_idx in input_weights[0]['named_grads'].keys():
            # record the shape of the current layer
            shape_cur_layer = input_weights[0]['named_grads'][layer_idx].cpu().numpy().shape
            # one_layer_set = torch.stack([item['named_grads'][layer_idx].flatten() for item in input_weights])
            one_layer_set = [item['named_grads'][layer_idx].flatten().cpu().numpy() for item in input_weights]
            one_layer_set = np.array(one_layer_set).astype(float)
            # one_layer_set = one_layer_set.to(device)
            # one_layer_set = np.array( one_layer_set )
            # print(one_layer_set)
            one_layer_results = trim_mean(one_layer_set, beta)
            # print(one_layer_results)
            one_layer_results = np.reshape(np.array(one_layer_results), shape_cur_layer)
            one_layer_results = torch.FloatTensor(one_layer_results)
            # print(layer_idx,selected.mean())
            res[layer_idx] = one_layer_results
        return res

    def cal_GeoMed(self, input_weights):
        # input_weights: list of client weights
        # input_weights[0]: list of a certain client weights
        # input_weights[i][j]: ndarray of a certain layer of a certain client weights
        # input_weights = list(grads.values())
        res = {}

        for layer_idx in input_weights[0]['named_grads'].keys():
            # record the shape of the current layer
            shape_cur_layer = input_weights[0]['named_grads'][layer_idx].cpu().numpy().shape
            # one_layer_set = torch.stack([item['named_grads'][layer_idx].flatten() for item in input_weights])
            one_layer_set = [item['named_grads'][layer_idx].flatten().cpu().numpy() for item in input_weights]
            one_layer_set = np.array(one_layer_set).astype(float)
            # one_layer_set = one_layer_set.float()
            one_layer_set = hdm.geomedian(one_layer_set, axis=0)
            one_layer_set = np.reshape(np.array(one_layer_set), shape_cur_layer)
            one_layer_set = torch.FloatTensor(one_layer_set)
            # print(layer_idx,selected.mean())
            res[layer_idx] = one_layer_set

        return res

    def cal_Krum(self, input_weights, num_byz):
        # input_weights: list of client weights
        # input_weights[0]: list of a certain client weights
        # input_weights[i][j]: ndarray of a certain layer of a certain client weights
        # input_weights = list(grads['named_grads'].values())
        res = {}
        num_machines = len(input_weights)
        for layer_idx in input_weights[0]['named_grads'].keys():
            # record the shape of the current layer
            shape_cur_layer = input_weights[0]['named_grads'][layer_idx].cpu().numpy().shape
            one_layer_set = [item['named_grads'][layer_idx].flatten().cpu().numpy() for item in input_weights]
            # one_layer_set = torch.stack([item['named_grads'][layer_idx].view(1,-1) for item in input_weights])
            one_layer_set = np.array(one_layer_set).astype(float)
            # one_layer_set = torch.Tensor(one_layer_set)
            # one_layer_set = one_layer_set.to(self.device)
            score = []
            # for i in one_layer_set:
            #     print(len(i))
            # print(one_layer_set.shape)
            num_near = num_machines - num_byz - 2
            for i, w_i in enumerate(one_layer_set):
                dist = []
                for j, w_j in enumerate(one_layer_set):
                    if i != j:
                        dist.append(np.linalg.norm(w_i - w_j) ** 2)
                dist.sort(reverse=False)
                score.append(sum(dist[0:num_near]))
            i_star = score.index(min(score))

            selected = one_layer_set[i_star]
            # print("layer_idx:",selected.mean())

            selected = np.reshape(np.array(selected), shape_cur_layer)
            selected = torch.FloatTensor(selected)
            # print(layer_idx,selected.mean())
            res[layer_idx] = selected

        return res

    def cal_theroy(self, input_weights, num, del_num, device):
        """
        @param input_weights: 输入的各个client的权重
        @param num: 恶意攻击者个数
        """
        global need_del
        # res = {}
        # single_weight = np.empty(shape=(0,))
        weight_set = []
        points = [list(item['named_grads'].values()) for item in input_weights]
        for i in range(len(points)):
            # single_weight.clear()
            single_weight = np.empty(shape=(0,))
            for j in range(len(points[0])):
                single_weight = np.hstack((single_weight, points[i][j].flatten().cpu().numpy()))
            print(single_weight)
            # single_weight = np.array(single_weight)
            weight_set.append(single_weight)
            # del single_weight[:]
            # print("********")
            # print(type(points[i]),points[i].device)
            # points[k] = np.array(weight)
        weight_set = np.array(weight_set)
        # print(type(weight_set))
        pca = PCA(n_components=1)
        pca.fit(weight_set)
        one_layer_set = pca.transform(weight_set)
        print("***********************************")
        print(one_layer_set)
        one_layer_set = torch.tensor(one_layer_set)
        dic = {}
        for i in range(one_layer_set.shape[1]):
            lis = [[] for _ in range(num)]
            min_data = min(one_layer_set[:, i])
            max_data = max(one_layer_set[:, i])
            dis = (max_data - min_data) / num
            if dis < 0.0000001:
                continue
            for j, w_j in enumerate(one_layer_set[:, i]):
                k = (w_j - min_data) // dis
                k = k.int()
                if k == num:
                    lis[k - 1].append(j)
                else:
                    lis[k].append(j)
            for m in range(one_layer_set.shape[0]):
                if m not in dic.keys():
                    dic[m] = 0
                entropy = 0
                for n in range(len(lis)):
                    if m in lis[n]:
                        tem = lis[n][:]
                        tem.remove(m)
                        p = len(tem) / (one_layer_set.shape[0] - 1)
                        if p == 0:
                            continue
                        else:
                            entropy += -p * np.log(p)
                    else:
                        p = len(lis[n]) / (one_layer_set.shape[0] - 1)
                        if p == 0:
                            continue
                        else:
                            entropy += -p * np.log(p)
                dic[m] += entropy
        print(dic)
        if len(dic) == 0:
            pass
        else:
            need_del = []
            # error_data = {}
            max_entropy = max(dic.values())
            for key, value in dic.items():
                data = (1 + value) / (1 + max_entropy)
                dic[key] = data
            sort_d = sorted(dic.items(), key=lambda d: d[1], reverse=False)[:del_num]
            # mean_error = np.mean(list(dic.values()))
            for key, value in sort_d:
                # if value < mean_error:
                need_del.append(key)
            print(need_del)
        # print("***********************************")
        # print(one_layer_set)
        # for i, layer_idx in enumerate(input_weights[0]['named_grads'].keys()):
        #     if i == 0:
        #         dic = {}
        #         # record the shape of the current layer
        #         # shape_cur_layer = input_weights[0]['named_grads'][layer_idx].shape
        #         one_layer_set = [item['named_grads'][layer_idx].flatten().cpu().numpy() for item in input_weights]
        #         one_layer_set = torch.tensor(one_layer_set)
        #         # one_layer_set = torch.stack([item['named_grads'][layer_idx].flatten() for item in input_weights])
        #         # record the shape of the current layer
        #         # one_layer_set = one_layer_set.cpu()
        #         # print(one_layer_set)
        #         # one_layer_set = np.array(one_layer_set).astype(float)
        #         pca = PCA(n_components=10)
        #         pca.fit(one_layer_set)
        #         one_layer_set = pca.fit_transform(one_layer_set)
        #         one_layer_set = torch.tensor(one_layer_set)
        #         for i in range(one_layer_set.shape[1]):
        #             lis = [[] for _ in range(num)]
        #             min_data = min(one_layer_set[:, i])
        #             max_data = max(one_layer_set[:, i])
        #             dis = (max_data - min_data) / num
        #             if dis < 0.0000001:
        #                 continue
        #             for j, w_j in enumerate(one_layer_set[:, i]):
        #                 k = (w_j - min_data) // dis
        #                 k = k.int()
        #                 if k == num:
        #                     lis[k - 1].append(j)
        #                 else:
        #                     lis[k].append(j)
        #             for m in range(one_layer_set.shape[0]):
        #                 if m not in dic.keys():
        #                     dic[m] = 0
        #                 entropy = 0
        #                 for n in range(len(lis)):
        #                     if m in lis[n]:
        #                         tem = lis[n][:]
        #                         tem.remove(m)
        #                         p = len(tem) / (one_layer_set.shape[0] - 1)
        #                         if p == 0:
        #                             continue
        #                         else:
        #                             entropy += -p * np.log(p)
        #                     else:
        #                         p = len(lis[n]) / (one_layer_set.shape[0] - 1)
        #                         if p == 0:
        #                             continue
        #                         else:
        #                             entropy += -p * np.log(p)
        #                 dic[m] += entropy
        #         print(dic)
        #         if len(dic) == 0:
        #             pass
        #         else:
        #             need_del = []
        #             # error_data = {}
        #             max_entropy = max(dic.values())
        #             for key, value in dic.items():
        #                 data = (1 + value) / (1 + max_entropy)
        #                 dic[key] = data
        #             # sort_d = sorted(dic.items(), key=lambda d: d[1], reverse=False)[:del_num]
        #             mean_error = np.mean(list(dic.values()))
        #             for key, value in dic.items():
        #                 if value < mean_error:
        #                     need_del.append(key)
        #             print(need_del)
        # for layer_idx in input_weights[0]['named_grads'].keys():
        #     # shape_cur_layer = input_weights[0]['named_grads'][layer_idx].shape
        #     # one_layer_set = [item['named_grads'][layer_idx].flatten() for item in input_weights]
        #     # print(one_layer_set)
        #     # one_layer_set = np.array(one_layer_set).astype(float)
        #     one_layer_set = np.delete(input_weights, need_del, axis=0)
        #     selected = self.aggregate_grads(input_weights,backend,layer_idx)
        #     # selected = one_layer_set.mean(axis=0)
        #     # selected = np.reshape(np.array(selected), shape_cur_layer)
        #     # selected = torch.FloatTensor(selected)
        #     res[layer_idx] = selected
        input_weights = [input_weights[i] for i in range(0, len(input_weights), 1) if i not in need_del]
        grads = self.aggregate_grads(input_weights, device)
        return grads

    def cal_theroy_cluster(self, input_weights, num, del_num, device):
        """
        @param input_weights: 输入的各个client的权重
        @param num: 恶意攻击者个数
        """
        global need_del
        # res = {}
        for i, layer_idx in enumerate(input_weights[0]['named_grads'].keys()):
            if i == 2:
                dic = {}
                # record the shape of the current layer
                # shape_cur_layer = input_weights[0]['named_grads'][layer_idx].shape
                one_layer_set = [item['named_grads'][layer_idx].flatten() for item in input_weights]
                # one_layer_set = torch.stack([item['named_grads'][layer_idx].flatten() for item in input_weights])
                # record the shape of the current layer
                # one_layer_set = one_layer_set.cpu()
                # print(one_layer_set)
                one_layer_set = np.array(one_layer_set).astype(float)
                for i in range(one_layer_set.shape[1]):
                    lis = [[] for _ in range(num)]
                    min = one_layer_set[:, i].min()
                    max = one_layer_set[:, i].max()
                    dis = (max - min) / num
                    if dis < 0.0000001:
                        continue
                    for j, w_j in enumerate(one_layer_set[:, i]):
                        k = (w_j - min) // dis
                        k = int(k)
                        if k == num:
                            lis[k - 1].append(j)
                        else:
                            lis[k].append(j)
                    for m in range(one_layer_set.shape[0]):
                        if m not in dic.keys():
                            dic[m] = 0
                        entropy = 0
                        for n in range(len(lis)):
                            if m in lis[n]:
                                tem = lis[n][:]
                                tem.remove(m)
                                p = len(tem) / (one_layer_set.shape[0] - 1)
                                if p == 0:
                                    continue
                                else:
                                    entropy += -p * np.log(p)
                            else:
                                p = len(lis[n]) / (one_layer_set.shape[0] - 1)
                                if p == 0:
                                    continue
                                else:
                                    entropy += -p * np.log(p)
                        dic[m] += entropy
                # sorted(dic.items(), key=lambda item: item[1])
                print(dic)
                if len(dic) == 0:
                    pass
                # else:
                #     need_del = []
                #     mean = sum(item[1] for item in dic.items()) / len(dic)
                #     for item in dic.items():
                #         if item[1] <= mean:
                #             need_del.append(item[0])
                #     print(need_del)
                #     one_layer_set = np.delete(one_layer_set,need_del,axis=0)
                else:
                    need_del = []
                    sort_d = sorted(dic.items(), key=lambda d: d[1], reverse=False)[:del_num]
                    for key, value in sort_d:
                        need_del.append(key)
                    print(need_del)
        # for layer_idx in input_weights[0]['named_grads'].keys():
        #     # shape_cur_layer = input_weights[0]['named_grads'][layer_idx].shape
        #     # one_layer_set = [item['named_grads'][layer_idx].flatten() for item in input_weights]
        #     # print(one_layer_set)
        #     # one_layer_set = np.array(one_layer_set).astype(float)
        #     one_layer_set = np.delete(input_weights, need_del, axis=0)
        #     selected = self.aggregate_grads(input_weights,backend,layer_idx)
        #     # selected = one_layer_set.mean(axis=0)
        #     # selected = np.reshape(np.array(selected), shape_cur_layer)
        #     # selected = torch.FloatTensor(selected)
        #     res[layer_idx] = selected
        input_weights = [input_weights[i] for i in range(0, len(input_weights), 1) if i not in need_del]
        grads = self.aggregate_grads(input_weights, device)
        return grads

    def foolsgold(self, grads):
        n_clients = grads.shape[0]
        cs = smp.cosine_similarity(grads) - np.eye(n_clients)
        maxcs = np.max(cs, axis=1)
        # pardoning
        for i in range(n_clients):
            for j in range(n_clients):
                if i == j:
                    continue
                if maxcs[i] < maxcs[j]:
                    cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]
        wv = 1 - (np.max(cs, axis=1))
        wv[wv > 1] = 1
        wv[wv < 0] = 0

        # Rescale so that max value is wv
        wv = wv / np.max(wv)
        wv[(wv == 1)] = .99

        # Logit function
        wv = (np.log(wv / (1 - wv)) + 0.5)
        wv[(np.isinf(wv) + wv > 1)] = 1
        wv[(wv < 0)] = 0

        return wv

    def cal_foolsgold(self, input_weights):
        client_grads = [list(item['named_grads'].values()) for item in input_weights]
        num_clients = len(client_grads)
        grad_len = np.array(client_grads[0][-2].cpu().data.numpy().shape).prod()
        grads = np.zeros((num_clients, grad_len))
        for i in range(len(client_grads)):
            grads[i] = np.reshape(client_grads[i][-2].cpu().data.numpy(), (grad_len))
        wv = self.foolsgold(grads)  # Use FG
        print(wv)
        # self.wv_history.append(wv)
        agg_grads = []
        # Iterate through each layer
        for i in range(len(client_grads[0])):
            temp = wv[0] * client_grads[0][i].cpu().clone()
            # Aggregate gradients for a layer
            for c, client_grad in enumerate(client_grads):
                if c == 0:
                    continue
                temp += wv[c] * client_grad[i].cpu()
            temp = temp / len(client_grads)
            agg_grads.append(temp)

        return agg_grads

    # RFA
    def call_rfa(self, input_weights, device):
        """Computes geometric median of atoms with weights alphas using Weiszfeld's Algorithm
        """
        server_model = ServerModel()
        weighted_updates = server_model.update(updates=input_weights, device=device)
        return weighted_updates

    def aggregate_grads(self, grads, device):
        """Aggregate model gradients to models.

        Args:
            data: a list of grads' information
                item format:
                    {
                        'n_samples': xxx,
                        'named_grads': xxx,
                    }
        Return:
            named grads: {
                'layer_name1': grads1,
                'layer_name2': grads2,
                ...
            }
        """
        total_grads = {}
        n_total_samples = 0
        for gradinfo in grads:
            n_samples = gradinfo['n_samples']
            for k, v in gradinfo['named_grads'].items():
                if k not in total_grads:
                    total_grads[k] = []

                total_grads[k].append(v * n_samples)
            n_total_samples += n_samples

        gradients = {}
        for k, v in total_grads.items():
            # print('v', v[0].is_cuda)
            v = torch.tensor([item.cpu().detach().numpy() for item in v]).to(device)
            # v = torch.stack(v)
            gradients[k] = torch.sum(v, dim=0) / n_total_samples

        return gradients
